# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
import random

from collections import defaultdict
from typing import List, Optional, Union

import torch

from sam3.train.data.sam3_image_dataset import Datapoint, FindQuery, Object


class FilterDataPointQueries:
    find_ids_to_filter: set = None
    get_ids_to_filter: set = None
    obj_ids_to_filter: set = None  # stored as pairs (img_id, obj_id)

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        """
        Compute set of query ids to keep, for both find and get queries
        """
        raise NotImplementedError

    def _do_filter_query(self, query: Union[FindQuery], query_id: int):
        assert self.find_ids_to_filter is not None

        return query_id in self.find_ids_to_filter


class FilterQueryWithText(FilterDataPointQueries):
    """
    Filter all datapoints which have query text in a specified list of exluded terms
    """

    def __init__(
        self, exclude_find_keys: List[str] = None, exclude_get_keys: List[str] = None
    ):
        self.find_filter_keys = exclude_find_keys if exclude_find_keys else []
        self.get_filter_keys = exclude_get_keys if exclude_get_keys else []

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        del_find_ids = []
        del_get_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if f_q.query_text in self.find_filter_keys:
                del_find_ids.append(i)

        self.find_ids_to_filter = set(del_find_ids)


class KeepMaxNumFindQueries(FilterDataPointQueries):
    def __init__(
        self, max_num_find_queries: int, retain_positive_queries: bool = False
    ):
        self.max_num_find_queries = max_num_find_queries
        self.retain_positive_queries = retain_positive_queries

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        num_find_queries = len(datapoint.find_queries)
        if num_find_queries <= self.max_num_find_queries:
            self.find_ids_to_filter = set()  # keep all find queries
            return

        if not self.retain_positive_queries:
            all_find_query_ids = list(range(num_find_queries))
            num_queries_to_filter = max(0, num_find_queries - self.max_num_find_queries)
            query_ids_to_filter = random.sample(
                all_find_query_ids, k=num_queries_to_filter
            )
        else:
            # keep up to max_num_find_queries postive find queries and fill
            # the remaining slots (if any) with negative find queries
            pos_find_ids, neg_find_ids = [], []
            for i, f_q in enumerate(datapoint.find_queries):
                # Negative finds return an empty list of object_ids_output
                if len(f_q.object_ids_output) == 0:
                    neg_find_ids.append(i)
                else:
                    pos_find_ids.append(i)

            if len(pos_find_ids) >= self.max_num_find_queries:
                # we have more positive find queries than `max_num_find_queries`,
                # so we subsample postive find queries and remove all negative find queries
                num_queries_to_filter = len(pos_find_ids) - self.max_num_find_queries
                query_ids_to_filter = random.sample(
                    pos_find_ids, k=num_queries_to_filter
                )
                query_ids_to_filter.extend(neg_find_ids)
            else:
                # we have fewer positive find queries than `max_num_find_queries`
                # so we need to fill the remaining with negative find queries
                num_queries_to_filter = num_find_queries - self.max_num_find_queries
                query_ids_to_filter = random.sample(
                    neg_find_ids, k=num_queries_to_filter
                )

        assert len(query_ids_to_filter) == num_find_queries - self.max_num_find_queries
        self.find_ids_to_filter = set(query_ids_to_filter)


class KeepMaxNumFindQueriesVideo(FilterDataPointQueries):
    def __init__(
        self,
        video_mosaic_max_num_find_queries_per_frame: int,
        retain_positive_queries: bool = False,
    ):
        self.video_mosaic_max_num_find_queries_per_frame = (
            video_mosaic_max_num_find_queries_per_frame
        )
        self.retain_positive_queries = retain_positive_queries

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        num_find_queries = len(datapoint.find_queries)

        findQueries_to_imageIds = defaultdict(list)
        max_queries_per_frame = True
        for i, f_q in enumerate(datapoint.find_queries):
            findQueries_to_imageIds[f_q.image_id].append(i)
            if (
                len(findQueries_to_imageIds[f_q.image_id])
                > self.video_mosaic_max_num_find_queries_per_frame
            ):
                max_queries_per_frame = False

        if max_queries_per_frame:
            self.find_ids_to_filter = set()
            return

        num_frames = len(findQueries_to_imageIds)
        findQueries_0 = findQueries_to_imageIds[0]
        num_find_queries_0 = len(findQueries_0)
        max_num_find_queries_per_frame = (
            self.video_mosaic_max_num_find_queries_per_frame
        )
        if not self.retain_positive_queries:
            find_query_ids_0 = list(range(num_find_queries_0))
            num_queries_to_filter = max(
                0, num_find_queries_0 - max_num_find_queries_per_frame
            )
            query_ids_to_filter_0 = random.sample(
                find_query_ids_0, k=num_queries_to_filter
            )
        else:
            # keep up to max_num_find_queries postive find queries and fill
            # the remaining slots (if any) with negative find queries
            pos_find_ids_0, neg_find_ids_0 = [], []
            for i, f_q_id in enumerate(findQueries_0):
                f_q = datapoint.find_queries[f_q_id]
                # Negative finds return an empty list of object_ids_output
                if len(f_q.object_ids_output) == 0:
                    neg_find_ids_0.append(i)
                else:
                    pos_find_ids_0.append(i)

            if len(pos_find_ids_0) >= max_num_find_queries_per_frame:
                # we have more positive find queries than `max_num_find_queries`,
                # so we subsample postive find queries and remove all negative find queries
                num_queries_to_filter = (
                    len(pos_find_ids_0) - max_num_find_queries_per_frame
                )
                query_ids_to_filter_0 = random.sample(
                    pos_find_ids_0, k=num_queries_to_filter
                )
                query_ids_to_filter_0.extend(neg_find_ids_0)
            else:
                # we have fewer positive find queries than `max_num_find_queries`
                # so we need to fill the remaining with negative find queries
                num_queries_to_filter = (
                    num_find_queries_0 - max_num_find_queries_per_frame
                )
                query_ids_to_filter_0 = random.sample(
                    neg_find_ids_0, k=num_queries_to_filter
                )

        # get based on frame 0 all find queries from all the frames with the same indices as in frame 0
        query_ids_to_filter = []
        for i in range(num_frames):
            findQueries_i = findQueries_to_imageIds[i]
            query_ids_to_filter.extend(
                [findQueries_i[j] for j in query_ids_to_filter_0]
            )

        assert (
            len(query_ids_to_filter)
            == num_find_queries
            - self.video_mosaic_max_num_find_queries_per_frame * num_frames
        )
        self.find_ids_to_filter = set(query_ids_to_filter)


class KeepSemanticFindQueriesOnly(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = {
            i for i, q in enumerate(datapoint.find_queries) if q.input_bbox is not None
        }  # filter (remove) geometric find queries (whose input_bbox is not None)

        # Keep all get queries which don't depend on filtered finds


class KeepUnaryFindQueriesOnly(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()

        # Keep all get queries which don't depend on filtered finds


class FilterZeroBoxQueries(FilterDataPointQueries):
    """
    Filters all find queries which predict a box with zero area
    """

    @staticmethod
    def _is_zero_area_object(obj: Object):
        # Check if height or width of bounding box is zero
        bbox = obj.bbox  # Assume in XYXY format
        height = bbox[..., 3].item() - bbox[..., 1].item()
        width = bbox[..., 2].item() - bbox[..., 0].item()

        return height == 0 or width == 0

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # Find objects with zero area
        # Assume only one image per datapoint
        image_objects = datapoint.images[0].objects
        exclude_objects = {
            obj_id
            for obj_id, obj in enumerate(image_objects)
            if self._is_zero_area_object(obj)
        }

        # If a query predicts an object with zero area, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            f_q_objects = set(f_q.object_ids_output)
            if len(exclude_objects.intersection(f_q_objects)) > 0:
                del_find_ids.append(i)

        self.find_ids_to_filter = set(del_find_ids)


class FilterFindQueriesWithTooManyOut(FilterDataPointQueries):
    """
    Filters all find queries which have more than a specified number of objects in the output
    """

    def __init__(self, max_num_objects: int):
        self.max_num_objects = max_num_objects

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # If a query predicts more than max_num_objects, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if len(f_q.object_ids_output) > self.max_num_objects:
                del_find_ids.append(i)

        self.find_ids_to_filter = set(del_find_ids)


class FilterEmptyTargets(FilterDataPointQueries):
    """
    Filters all targets which have zero area
    """

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        for img_id in range(len(datapoint.images)):
            for obj_id, obj in enumerate(datapoint.images[img_id].objects):
                if obj.area < 1e-6:
                    self.obj_ids_to_filter.add((img_id, obj_id))
        self.find_ids_to_filter = set()


class FilterNonExhaustiveFindQueries(FilterDataPointQueries):
    """
    Filters all find queries which are non-exhaustive
    """

    def __init__(self, exhaustivity_type: str):
        """
        Args:
            exhaustivity_type: Can be "pixel" or "instance":
                -pixel: filter queries where the union of all segments covers every pixel belonging to target class
                -instance: filter queries where there are non-separable or non annotated instances
        Note that instance exhaustivity implies pixel exhaustivity
        """
        assert exhaustivity_type in ["pixel", "instance"]
        self.exhaustivity_type = exhaustivity_type

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # If a query predicts more than max_num_objects, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if self.exhaustivity_type == "instance":
                if not f_q.is_exhaustive:
                    del_find_ids.append(i)
            elif self.exhaustivity_type == "pixel":
                if f_q.is_pixel_exhaustive is not None and not f_q.is_pixel_exhaustive:
                    del_find_ids.append(i)
            else:
                raise RuntimeError(
                    f"Unknown exhaustivity type {self.exhaustivity_type}"
                )

        self.find_ids_to_filter = set(del_find_ids)


class FilterInvalidGeometricQueries(FilterDataPointQueries):
    """
    Filters geometric queries whose output got deleted (eg due to cropping)
    """

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()

        # If a query predicts more than max_num_objects, drop the whole find query
        del_find_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if f_q.input_bbox is not None and f_q.query_text == "geometric":
                if len(f_q.object_ids_output) == 0:
                    del_find_ids.append(i)
        self.find_ids_to_filter = set(del_find_ids)


class FlexibleFilterFindGetQueries:
    def __init__(
        self, query_filter: FilterDataPointQueries, enabled: bool = True
    ) -> None:
        self.query_filter = query_filter
        self.enabled = enabled

    def __call__(self, datapoint, **kwargs):
        if not self.enabled:
            return datapoint

        # Identify all queries to filter
        self.query_filter.identify_queries_to_filter(datapoint=datapoint)

        del_find_ids = []
        del_get_ids = []
        for i, f_q in enumerate(datapoint.find_queries):
            if self.query_filter._do_filter_query(f_q, i):
                datapoint.find_queries[i] = None
                del_find_ids.append(i)

        new_find_queries = []
        new_get_queries = []

        find_old_to_new_map = {}
        get_old_to_new_map = {}

        find_counter = 0
        get_counter = 0

        for i, f_q in enumerate(datapoint.find_queries):
            if f_q is not None:
                find_old_to_new_map[i] = find_counter
                find_counter += 1
                new_find_queries.append(f_q)

        start_with_zero_check = False
        for n_f_q in new_find_queries:
            if n_f_q.query_processing_order == 0:
                start_with_zero_check = True
                break

        if len(new_find_queries) == 0:
            start_with_zero_check = True

        assert (
            start_with_zero_check
        ), "Invalid Find queries, they need to start at query_processing_order = 0"

        datapoint.find_queries = new_find_queries

        if len(datapoint.find_queries) == 0:
            print("Warning: No find queries left in datapoint, this is not allowed")
            print("Filtering function:", self.query_filter)
            print("Datapoint:", datapoint)
            raise ValueError

        # The deletion may have removed intermediate steps, so we need to remap to make them contiguous again
        all_stages = sorted(
            list(set(q.query_processing_order for q in datapoint.find_queries))
        )
        stage_map = {qpo: i for i, qpo in enumerate(all_stages)}
        for i in range(len(datapoint.find_queries)):
            qpo = datapoint.find_queries[i].query_processing_order
            datapoint.find_queries[i].query_processing_order = stage_map[qpo]

        # Final step, clear up objects that are not used anymore
        for img_id in range(len(datapoint.images)):
            all_objects_ids = set(
                i
                for find in datapoint.find_queries
                for i in find.object_ids_output
                if find.image_id == img_id
            )
            unused_ids = (
                set(range(len(datapoint.images[img_id].objects))) - all_objects_ids
            )
            for tgt_img_id, tgt_obj_id in self.query_filter.obj_ids_to_filter:
                if tgt_img_id == img_id:
                    unused_ids.add(tgt_obj_id)

            if len(unused_ids) > 0:
                old_objects = datapoint.images[img_id].objects
                object_old_to_new_map = {}
                new_objects = []
                for i, o in enumerate(old_objects):
                    if i not in unused_ids:
                        object_old_to_new_map[i] = len(new_objects)
                        new_objects.append(o)

                datapoint.images[img_id].objects = new_objects

                # Remap the outputs of the find queries
                affected_find_queries_ids = set()
                object_old_to_new_map_per_query = {}
                for fid, find in enumerate(datapoint.find_queries):
                    if find.image_id == img_id:
                        old_object_ids_output = find.object_ids_output
                        object_old_to_new_map_per_query[fid] = {}
                        find.object_ids_output = []
                        for oid, old_obj_id in enumerate(old_object_ids_output):
                            if old_obj_id not in unused_ids:
                                new_obj_id = object_old_to_new_map[old_obj_id]
                                find.object_ids_output.append(new_obj_id)
                                object_old_to_new_map_per_query[fid][oid] = (
                                    len(find.object_ids_output) - 1
                                )
                        affected_find_queries_ids.add(fid)

        # finally remove unused images
        all_imgs_to_keep = set()
        for f_q in datapoint.find_queries:
            all_imgs_to_keep.add(f_q.image_id)

        old_img_id_to_new_img_id = {}
        new_images = []
        for img_id, img in enumerate(datapoint.images):
            if img_id in all_imgs_to_keep:
                old_img_id_to_new_img_id[img_id] = len(new_images)
                new_images.append(img)
        datapoint.images = new_images

        for f_q in datapoint.find_queries:
            f_q.image_id = old_img_id_to_new_img_id[f_q.image_id]

        return datapoint


class AddPrefixSuffixToFindText:
    """
    Add prefix or suffix strings to find query text on the fly.

    If `condition_on_text` is True, the prefix or suffix strings are only added
    to those find query text in `condition_text_list` (case-insensitive).
    """

    def __init__(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        condition_on_text: bool = False,
        condition_text_list: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.condition_on_text = condition_on_text
        if self.condition_on_text:
            assert condition_text_list is not None
            self.condition_text_set = {s.lower().strip() for s in condition_text_list}
        self.enabled = enabled
        if self.enabled:
            logging.info(
                f"AddPrefixSuffixToFindText: prefix={prefix}, suffix={suffix}, "
                f"condition_on_text={condition_on_text}, condition_text_list={condition_text_list}"
            )

    def __call__(self, datapoint, **kwargs):
        if not self.enabled:
            return datapoint

        for find in datapoint.find_queries:
            if find.query_text == "geometric":
                # skip geometric find queries
                continue
            if (
                self.condition_on_text
                and find.query_text.lower().strip() not in self.condition_text_set
            ):
                # if condition_on_text is True, skip those queries not in condition_text_set
                continue

            # add prefix and/or suffix strings to the find query text
            if self.prefix is not None:
                find.query_text = self.prefix + find.query_text
            if self.suffix is not None:
                find.query_text = find.query_text + self.suffix

        return datapoint


class FilterCrowds(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        """
        Compute set of query ids to keep, for both find and get queries
        """
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()
        # self.get_ids_to_filter = set()
        for img_id, img in enumerate(datapoint.images):
            for obj_id, obj in enumerate(img.objects):
                if obj.is_crowd:
                    self.obj_ids_to_filter.add((img_id, obj_id))


class TextQueryToVisual:
    """
    Transform a test query to a visual query (with some proba), using any of the output targets as the prompt
    """

    def __init__(self, probability, keep_text_queries=False) -> None:
        self.probability = probability
        assert 0 <= probability <= 1
        self.keep_text_queries = keep_text_queries

    def __call__(self, datapoint: Datapoint, **kwargs):
        for find in datapoint.find_queries:
            if find.input_bbox is not None or find.input_points is not None:
                # skip geometric find queries
                continue

            if len(find.object_ids_output) == 0:
                # Can't create a visual query, skip
                continue

            if find.query_processing_order > 0:
                # Second stage query, can't use
                continue

            if random.random() > self.probability:
                continue

            selected_vq_id = random.choice(find.object_ids_output)
            img_id = find.image_id

            find.input_bbox = datapoint.images[img_id].objects[selected_vq_id].bbox
            find.input_bbox_label = torch.ones(1, dtype=torch.bool)
            if not self.keep_text_queries:
                find.query_text = "visual"

        return datapoint


class RemoveInputBoxes:
    """
    Remove input boxes from find queries
    """

    def __init__(self) -> None:
        pass

    def __call__(self, datapoint: Datapoint, **kwargs):
        for find in datapoint.find_queries:
            if find.input_bbox is None:
                continue

            if find.query_text == "geometric":
                print("Warning: removing input box from geometric find query")

            find.input_bbox = None
        return datapoint


class OverwriteTextQuery:
    """
    With some probability, overwrite the text query with a custom text
    """

    def __init__(self, target_text, probability=1.0) -> None:
        self.probability = probability
        self.target_text = target_text
        assert 0 <= probability <= 1

    def __call__(self, datapoint: Datapoint, **kwargs):
        for find in datapoint.find_queries:
            if random.random() > self.probability:
                continue

            find.query_text = self.target_text

        return datapoint
