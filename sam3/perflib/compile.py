# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from functools import wraps

import torch
from sam3.model.data_misc import BatchedDatapoint, NestedTensor
from torch.utils._pytree import tree_map_only


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, NestedTensor):
            tensors = fn(b.tensors)
            if b.mask is None:
                mask = None
            else:
                mask = fn(b.mask)
            return NestedTensor(tensors=tensors, mask=mask)
        if isinstance(b, torch.Tensor):
            return fn(b)
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def clone_output_wrapper(f):
    """
    Clone the CUDA output tensors of a function to avoid in-place operations.
    Uses tree_map_only (C-optimized pytree traversal) matching onevision's pattern.
    Requires NestedTensor to be registered as a pytree node (see data_misc.py).
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    """Compile with recursive_contiguous on inputs and recursive_clone on outputs.
    Used for SAM2 tracker components that need contiguous inputs for CUDA graphs."""
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            CUDAGRAPH_MODES = ["max-autotune", "reduce-overhead"]
            args = recursive_contiguous(args)
            kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*args, **kwargs)
            if mode in CUDAGRAPH_MODES:
                result = recursive_clone(result)
            return result

    return compiled_fn_wrapper


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    """
    Wraps a function and prints the shapes of all tensor inputs.
    Only prints when a new combination of shapes is seen.
    """
    seen_shapes = set()

    def get_shape(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 1:
                return tuple(get_shape(x) for x in obj)
            return get_shape(obj[0])
        elif isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        else:
            return type(obj).__name__

    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v))
            for k, v in kwargs.items()
            if isinstance(v, (torch.Tensor, list))
            and (len(keep_kwargs) > 0 and k in keep_kwargs)
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                print(f"[ShapeLogger] New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    wrapper.enable_logging = enable_logging

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enable_logging

    wrapper.set_logging = set_logging
    return wrapper
