# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
import math
from typing import Optional

import torch
from torch import nn

# Special values for object tracking
_PADDING_NUM = -1  # Marks empty slots in buckets
_REMOVED_NUM = -1116  # Marks objects that have been removed


logger = logging.getLogger(__name__)


class MultiplexState:
    """
    MultiplexState records the state of multiplexing, for one or more buckets.

    At a high level, we deal with the conversion of tensors between the data space (batch_size, num_channels, ...)
    and the multiplex space (num_buckets, multiplex_count, num_channels, ...).

    The multiplex state stores the assignments of each batch element to a slot in a bucket.
    Each bucket has a fixed number of slots (multiplex_count), and not all slots need to be filled.
    The batch size should equate to total_valid_entries, which is the sum of the number of valid entries in each bucket.

    There are two main operations that this class supports:
        mux: convert tensors in the data space to the multiplex space.
        The mental model is that we start from a tensor of zeros that has the shape of the output,
        then we go through the valid entries and place them into the corresponding slots, indicated by the assignments.

        demux: convert tensors in the multiplex space to the data space.
        This is the reverse operation of mux. Note that zeros were used in mux for the padding slots,
        and that those slots are ignored in demux.

    There are also two utility functions for object mangement:
        add_objects: add new objects to the state by filling in empty slots
        remove_objects: remove objects from the state by marking them as removed (not the same as empty!)
    """

    def __init__(
        self,
        assignments: list[list[int]],
        device: torch.device,
        dtype: torch.dtype,
        allowed_bucket_capacity: int,
        *,
        object_ids: Optional[list[int]] = None,
    ):
        """
        assignments: a list of lists of object indices
            Each top-level list represents a bucket
            Each inner list represents the object indices that are in the bucket
            The object indices must ranges from 0 to num_valid_entries - 1, except for the following special values (all negatives):
                _PADDING_NUM, which denotes padding entries
                _REMOVED_NUM, which denotes an pre-existing object that got removed (currently not used during init)
            If you wish to save the "true" object IDs, i.e., during inference, you can bookkeep them here
        """
        self.device = device
        self.dtype = dtype

        # Initialize bucket assignments and precompute matrices
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(
        self, assignments: list[list[int]], *, object_ids: Optional[list[int]] = None
    ):
        self.assignments = assignments
        self.num_buckets = len(self.assignments)
        if self.num_buckets == 0:
            logger.error("No buckets found in the state")
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(self.assignments[0])
        assert all(
            len(self.assignments[i]) == self.multiplex_count
            for i in range(self.num_buckets)
        )

        # number of non-negative elements in the state
        self.total_valid_entries = sum(
            sum(1 for x in bucket if x >= 0) for bucket in self.assignments
        )
        self.total_non_padding_entries = sum(
            sum(1 for x in bucket if x != _PADDING_NUM) for bucket in self.assignments
        )

        # check the validity of the object IDs
        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries, (
                "object_ids should map 1:1 to the valid entries"
            )

        # check the validity of the assignments
        all_object_idxs = set()
        for bucket in self.assignments:
            valid_entries_in_bucket = sum(1 for x in bucket if x != _PADDING_NUM)
            assert valid_entries_in_bucket <= self.allowed_bucket_capacity, (
                f"{valid_entries_in_bucket=} > {self.allowed_bucket_capacity=}"
            )
            for obj_idx in bucket:
                if obj_idx >= 0:
                    assert obj_idx < self.total_non_padding_entries, (
                        f"object ID {obj_idx} >= {self.total_non_padding_entries}"
                    )
                    assert obj_idx not in all_object_idxs, "object IDs must be unique"
                    all_object_idxs.add(obj_idx)

        # Precompute and cache the actual selection matrices
        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        # returns the number of available slots in the state
        return (
            self.num_buckets * self.allowed_bucket_capacity
            - self.total_non_padding_entries
        )

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list[int]:
        # produce a list of consecutive indices that are available in the state
        # Note: prefer_new_buckets parameter is accepted for API compatibility but not used here
        # as the actual bucket allocation logic is in add_objects()
        assert num_objects > 0, f"{num_objects=} must be positive"
        if not allow_new_buckets:
            assert self.available_slots >= num_objects, (
                f"not enough available slots {self.available_slots} < {num_objects}"
            )

        return list(
            range(
                self.total_valid_entries,
                self.total_valid_entries + num_objects,
            )
        )

    def add_objects(
        self,
        object_indices: list[int],
        *,
        object_ids: Optional[list[int]] = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ):
        """
        Add new objects to the state by filling in empty slots and
        creating new buckets if necessary.

        object_indices must be sorted and follow existing object indices.
        If prefer_new_buckets is True, we skip filling existing slots and place
        the objects into freshly created buckets (requires allow_new_buckets=True).
        """
        if len(object_indices) == 0:
            return

        # we will modify this in-place
        object_indices = object_indices.copy()
        assert (object_ids is None) == (self.object_ids is None), (
            "object_ids must either be always given or always omitted"
        )

        if object_ids is not None:
            assert len(object_ids) == len(object_indices), (
                "object_ids must have the same length as object_indices"
            )
            object_ids = object_ids.copy()

        num_new_objects = len(object_indices)
        assert object_indices == sorted(object_indices), "object_indices must be sorted"
        object_indices.reverse()  # reverse so we can pop from the end
        if object_ids is not None:
            object_ids.reverse()

        if prefer_new_buckets:
            assert allow_new_buckets, "prefer_new_buckets requires allow_new_buckets"

        slots_filled = 0
        buckets_created = 0

        def _pop_next():
            idx = object_indices.pop()
            if object_ids is not None and self.object_ids is not None:
                self.object_ids.append(object_ids.pop())
            return idx

        if not prefer_new_buckets:
            # Fill empty slots in existing buckets first
            for bucket in self.assignments:
                for i in range(self.allowed_bucket_capacity):
                    if bucket[i] == _PADDING_NUM:
                        bucket[i] = _pop_next()
                        slots_filled += 1
                        if len(object_indices) == 0:
                            break
                if len(object_indices) == 0:
                    break

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(
                f"Cannot place objects {list(reversed(object_indices))} without creating new buckets"
            )

        # Create new buckets for remaining objects (or all objects if prefer_new_buckets)
        while len(object_indices) > 0:
            new_bucket = [_PADDING_NUM] * self.multiplex_count
            for i in range(self.allowed_bucket_capacity):
                if len(object_indices) == 0:
                    break
                new_bucket[i] = _pop_next()
            self.assignments.append(new_bucket)
            buckets_created += 1

        # reinitialize all the settings
        original_num_entries = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original_num_entries + num_new_objects, (
            f"{self.total_valid_entries=} != {original_num_entries=} + {num_new_objects=}"
        )

        logger.info(
            f"Filled {slots_filled} slots and created {buckets_created} new buckets"
        )
        logger.info(
            f"{self.num_buckets=}, {self.total_valid_entries=}, {self.total_non_padding_entries=}"
        )

    def remove_objects(self, object_indices: list[int], strict: bool = True):
        """
        Remove objects from the state by marking them as removed.
        Remove a bucket if all objects in the bucket are removed.

        Args:
            object_indices: List of object indices to remove
            strict: If True, will raise an error if any object indices are not found in the state

        Returns:
            List of bucket indices that we are going to keep
        """
        object_indices = object_indices.copy()

        # Mark objects as removed in assignments
        for bucket_idx, bucket in enumerate(self.assignments):
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in object_indices:
                    self.assignments[bucket_idx][slot_idx] = _REMOVED_NUM
                    object_indices.remove(obj_id)

        if strict:
            assert len(object_indices) == 0, (
                f"Failed to remove objects: {object_indices}"
            )

        # Check which buckets should be completely removed (all objects removed/paddings)
        # and which buckets we will keep
        buckets_to_remove = []
        buckets_to_keep = []
        for bucket_idx, bucket in enumerate(self.assignments):
            # Check if all objects in this bucket are removed or are paddings
            all_removed = all(
                obj_id in [_PADDING_NUM, _REMOVED_NUM] for obj_id in bucket
            )
            if all_removed:
                buckets_to_remove.append(bucket_idx)
                logger.info(
                    f"Bucket {bucket_idx} marked for removal - all objects removed/paddings"
                )
            else:
                buckets_to_keep.append(bucket_idx)

        # Remove buckets in reverse order to maintain correct indices
        for bucket_idx in reversed(buckets_to_remove):
            del self.assignments[bucket_idx]

        if len(buckets_to_keep) == 0:
            logger.info(f"Removing all buckets: {buckets_to_remove}; state invalidated")
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        # After removal, remap object IDs to be sequential
        # Collect all unique positive object IDs and create a mapping to sequential IDs
        all_positive_ids = set()
        for bucket in self.assignments:
            for obj_id in bucket:
                if obj_id >= 0:
                    all_positive_ids.add(obj_id)

        # Create mapping from old IDs to new sequential IDs
        sorted_ids = sorted(all_positive_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}

        # Apply the mapping to assignments to make IDs sequential
        for bucket in self.assignments:
            for i, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[i] = id_mapping[obj_id]

        # Update object_ids if they exist
        if self.object_ids is not None:
            # Create new object_ids array based on the remapped indices
            # We need to preserve the original object_ids for the objects that weren't removed
            new_object_ids = [None] * len(sorted_ids)

            # Map the original object_ids to their new positions
            for old_idx, new_idx in id_mapping.items():
                new_object_ids[new_idx] = self.object_ids[old_idx]

            assert not any(obj_id is None for obj_id in new_object_ids)
            self.object_ids = new_object_ids

        # Reinitialize the state to update all internal structures
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)

        logger.info(f"Removed these buckets: {buckets_to_remove}")
        logger.info(f"Kept these buckets: {buckets_to_keep}")
        logger.info(
            f"Remaining buckets: {self.num_buckets}, total valid entries: {self.total_valid_entries}"
        )

        return buckets_to_keep

    def _precompute_transition_matrices(self, device: torch.device, dtype: torch.dtype):
        """
        Precompute the transition matrices for maximum efficiency.
        Note that these should be partial permutation matrices.
        """
        # Create a transition matrix for muxing
        self.mux_matrix = torch.zeros(
            self.num_buckets * self.multiplex_count,
            self.total_valid_entries,
            device=device,
            dtype=dtype,
        )

        # Create a transition matrix for demuxing
        self.demux_matrix = torch.zeros(
            self.total_valid_entries,
            self.num_buckets * self.multiplex_count,
            device=device,
            dtype=dtype,
        )

        # Fill both matrices based on assignments
        for i in range(self.num_buckets):
            for j in range(self.multiplex_count):
                bucket_idx = i * self.multiplex_count + j
                object_idx = self.assignments[i][j]
                if object_idx >= 0:
                    self.mux_matrix[bucket_idx, object_idx] = 1.0
                    self.demux_matrix[object_idx, bucket_idx] = 1.0

    def mux(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiplexing operation
            x: self.total_valid_entries * ...

            return num_buckets * multiplex_count * ...
            with padding entries filled with 0
        """
        num_valid_entries = x.shape[0]
        assert num_valid_entries == self.total_valid_entries, (
            f"{num_valid_entries=} != {self.total_valid_entries=}"
        )
        output_shape = (
            self.num_buckets,
            self.multiplex_count,
        ) + x.shape[1:]

        x_flat = x.reshape(num_valid_entries, -1)

        # Apply mux matrix: (num_buckets * multiplex_count, batch_size) @ (batch_size, features)
        # Result: (num_buckets * multiplex_count, features)
        result_flat = self.mux_matrix @ x_flat

        result = result_flat.view(output_shape)
        return result

    def demux(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse operation of mux
            x: num_buckets, multiplex_count * ...
            Returns: total_valid_entries * ...
        """
        num_buckets, multiplex_count = x.shape[:2]
        assert num_buckets == self.num_buckets, f"{num_buckets=} != {self.num_buckets=}"
        assert multiplex_count == self.multiplex_count, (
            f"{multiplex_count=} != {self.multiplex_count=}"
        )
        output_shape = (self.total_valid_entries,) + x.shape[2:]

        x_flat = x.reshape(num_buckets * multiplex_count, -1)

        # Apply demux matrix: (total_valid_entries, num_buckets*multiplex_count) @ (num_buckets*multiplex_count, features)
        # Result: (total_valid_entries, features)
        result_flat = self.demux_matrix @ x_flat

        result = result_flat.view(output_shape)
        return result

    def get_valid_object_mask(self) -> torch.Tensor:
        """
        Returns a (num_buckets, multiplex_count) tensor with 1 for valid entries and 0 for padding entries
        """
        valid_mask = self.mux_matrix.sum(dim=1) > 0
        valid_mask = valid_mask.reshape(self.num_buckets, self.multiplex_count)

        return valid_mask

    def get_all_valid_object_idx(self) -> set[int]:
        """
        Returns a set of all valid object idx in the state
        Note that this returns the internal object idx representations,
        not the arbitrary object IDs that are passed in during initialization
        """
        all_valid_objects = {
            obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0
        }
        return all_valid_objects


class MultiplexController(nn.Module):
    def __init__(
        self,
        multiplex_count: int,
        full_shuffle: bool = False,
        eval_multiplex_count: int = -1,
    ):
        super().__init__()

        self.multiplex_count = multiplex_count
        self.full_shuffle = full_shuffle
        if eval_multiplex_count < 0:
            self.eval_multiplex_count = multiplex_count
        else:
            self.eval_multiplex_count = eval_multiplex_count
        assert self.multiplex_count >= 1

    @property
    def allowed_bucket_capacity(self) -> int:
        if self.training:
            return self.multiplex_count
        else:
            return self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        device: torch.device,
        dtype: torch.dtype,
        random: bool = True,
        *,
        object_ids: Optional[
            list[int]
        ] = None,  # object_ids is an auxiliary field that we pass to the state unmodified
    ) -> MultiplexState:
        # returns a state that maps elements in the batch to buckets of size self.multiplex_count

        allowed_bucket_capacity = self.allowed_bucket_capacity

        # the size of the bucket during training
        true_bucket_capacity = self.multiplex_count

        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)
        # each bucket contains at most self.multiplex_count elements
        # padding elements are marked with _PADDING_NUM (only the last bucket should contain _PADDING_NUM)

        if self.full_shuffle:
            # Shuffle all IDs, including the paddings
            ids = torch.cat(
                [
                    torch.arange(num_valid_entries, dtype=torch.long),
                    torch.tensor(
                        [_PADDING_NUM]
                        * (num_buckets * true_bucket_capacity - num_valid_entries),
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            if random:
                indices = torch.randperm(ids.shape[0], dtype=torch.long)
                ids = ids[indices]

            # convert to a list of list
            assignments = []
            for i in range(num_buckets):
                assignments.append(
                    ids[
                        i * true_bucket_capacity : (i + 1) * true_bucket_capacity
                    ].tolist()
                )
        else:
            # Only shuffle the the IDs within the first #batch_size slots, leave all paddings at the end
            if random:
                # randomly assign ids to buckets
                ids = torch.randperm(num_valid_entries, dtype=torch.int64)
            else:
                ids = torch.arange(num_valid_entries)
            # append with _PADDING_NUM to make a multiple of bucket_capacity
            total_elements = num_buckets * allowed_bucket_capacity
            if ids.shape[0] < total_elements:
                ids = torch.cat(
                    [
                        ids,
                        torch.tensor([_PADDING_NUM] * (total_elements - ids.shape[0])),
                    ]
                )

            # convert to a list of list
            assignments = []
            for i in range(num_buckets):
                assignments.append(
                    ids[
                        i * allowed_bucket_capacity : (i + 1) * allowed_bucket_capacity
                    ].tolist()
                    + [_PADDING_NUM] * (true_bucket_capacity - allowed_bucket_capacity)
                )

        return MultiplexState(
            assignments,
            device,
            dtype,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=object_ids,
        )
