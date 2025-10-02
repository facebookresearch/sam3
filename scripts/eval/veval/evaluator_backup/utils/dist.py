# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities related to distributed mode.

By default, the reduce of metrics and such are done on GPU, since it's more straightforward (we reuse the NCCL backend)
If you want to reduce on CPU instead (required for big datasets like GQA), use the env variable MDETR_CPU_REDUCE=1
"""

import datetime
import functools
import io
import os
import random
import time

import torch
import torch.distributed as dist


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """

    if dist.get_backend() == "nccl":
        # Increase timeout from 1800 sec to 43200 sec (12 hr) to avoid some processes
        # being much slower than others causing a timeout (which can happen in relation
        # or LVIS class mAP evaluation).
        timeout = 43200
        return dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=timeout),
        )

    return dist.group.WORLD


def all_gather(data, force_cpu=False, force_filesys=False, filesys_save_dir=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if os.getenv("MDETR_FILESYS_REDUCE_RANK_0_ONLY") == "1":
        return all_gather_via_filesys(
            data, filesys_save_dir, gather_to_rank_0_only=True
        )

    if os.getenv("MDETR_FILESYS_REDUCE") == "1" or force_filesys:
        return all_gather_via_filesys(data, filesys_save_dir)

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1" or force_cpu:
        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [
        torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)
    ]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy())
        obj = torch.load(buffer, weights_only=False)
        data_list.append(obj)

    return data_list


def all_gather_via_filesys(data, filesys_save_dir=None, gather_to_rank_0_only=False):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors), similar to
    `all_gather` above, but using filesystem instead of collective ops.

    If gather_to_rank_0_only is True, only rank 0 will load the gathered object list
    (and other ranks will have an empty list).
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    print("gathering via files")
    cpu_group = _get_global_gloo_group()

    # if unspecified, we will save to the current python file dir
    if filesys_save_dir is not None:
        save_dir = filesys_save_dir
    elif "EXP_DIR" in os.environ:
        # this is the experiment dir set in projects/onevision/dev/launch_job.py
        save_dir = os.environ["EXP_DIR"]
    else:
        # try the same directory where the code is stored
        save_dir = filesys_save_dir or os.path.dirname(__file__)
    save_dir = os.path.join(save_dir, "all_gather_via_filesys")
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    # use a timestamp and salt to distinguish different all_gather
    timestamp = int(time.time()) if is_main_process() else 0
    salt = random.randint(0, 2**31 - 1) if is_main_process() else 0
    # broadcast the timestamp and salt across ranks
    # (all-reduce will do the broadcasting since only rank 0 is non-zero)
    timestamp_and_salt = torch.tensor([timestamp, salt], dtype=torch.long)
    dist.all_reduce(timestamp_and_salt, group=cpu_group)
    timestamp, salt = timestamp_and_salt.tolist()

    # save the data to a file on the disk
    rank_save = get_rank()
    save_data_filename = f"data_to_gather_{timestamp}_{salt}_{rank_save}.pkl"
    save_data_path = os.path.join(save_dir, save_data_filename)
    assert not os.path.exists(save_data_path), f"{save_data_path} already exists"
    torch.save(data, save_data_path)
    dist.barrier(group=cpu_group)

    # read the data from the files
    data_list = []
    if rank_save == 0 or not gather_to_rank_0_only:
        for rank_load in range(world_size):
            load_data_filename = f"data_to_gather_{timestamp}_{salt}_{rank_load}.pkl"
            load_data_path = os.path.join(save_dir, load_data_filename)
            assert os.path.exists(load_data_path), f"cannot read {save_data_path}"
            data_list.append(torch.load(load_data_path, weights_only=False))
    dist.barrier(group=cpu_group)

    # delete the saved file
    os.remove(save_data_path)
    return data_list


def gather_to_rank_0_via_filesys(data, filesys_save_dir=None):
    """
    Gather any picklable data to rank 0 via filesystem, using all_gather_via_filesys.
    """
    return all_gather_via_filesys(data, filesys_save_dir, gather_to_rank_0_only=True)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0).cuda()
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    Returns:
        True if distributed training is enabled
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Returns:
        The number of processes in the process group
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Returns:
        The rank of the current process within the global process group.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Return true if the current process is the main one"""
    return get_rank() == 0


def broadcast_int(value, src_rank=0):
    """Broadcast integer or integer list `value` from `src_rank` to all ranks."""
    assert isinstance(value, int) or isinstance(value, list)
    world_size = get_world_size()
    if world_size <= 1:
        return value

    cpu_group = _get_global_gloo_group()
    t = torch.tensor(value, dtype=torch.int64)
    torch.distributed.broadcast(t, src=src_rank, group=cpu_group)

    if isinstance(value, int):
        return t.item()
    else:
        return t.tolist()
