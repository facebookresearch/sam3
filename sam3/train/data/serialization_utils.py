"""
Utility for serializing lists to numpy arrays to avoid copy-on-read overhead.

Based on: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
Reference: Detectron2's implementation
"""

import pickle
from typing import Any, List
import numpy as np


class NumpySerializedList:
    """
    Store a list of Python objects in a serialized numpy array.
    
    This class eliminates the "copy-on-read" problem in multiprocessing:
    - Python objects have refcounts that get updated even on read
    - Reading objects in forked workers causes copy-on-write to trigger
    - Each worker ends up copying the entire dataset
    
    Solution:
    - Serialize all objects into a single numpy array (no Python objects)
    - Numpy arrays have minimal refcounts
    - Workers share the numpy array without copying
    - Deserialize on-demand when accessing items
    
    Memory savings: ~6x reduction with 4 workers
    """
    
    def __init__(self, lst: List[Any]):
        """
        Serialize a list of Python objects.
        
        Args:
            lst: List of Python objects to serialize
        """
        # Serialize each item to bytes
        serialized = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in lst]
        
        # Store cumulative lengths for indexing
        self._addr = np.cumsum([len(x) for x in serialized], dtype=np.int64)
        
        # Concatenate all serialized data into single array
        self._lst = np.concatenate(serialized)
        
    def __len__(self):
        return len(self._addr)
    
    def __getitem__(self, idx: int):
        """
        Retrieve and deserialize item at index.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Deserialized Python object
        """
        start = 0 if idx == 0 else self._addr[idx - 1]
        end = self._addr[idx]
        
        # Use memoryview to avoid copy
        return pickle.loads(memoryview(self._lst[start:end]))


class TorchSerializedList:
    """
    Alternative implementation using torch.Tensor for spawn/forkserver mode.
    
    torch.Tensor can be pickled more efficiently than numpy in spawn mode.
    Use this if you're using multiprocessing_context='spawn' or 'forkserver'.
    """
    
    def __init__(self, lst: List[Any]):
        import torch
        
        # Serialize each item
        serialized = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in lst]
        
        # Store as torch tensors
        self._addr = torch.from_numpy(np.cumsum([len(x) for x in serialized], dtype=np.int64))
        self._lst = torch.from_numpy(np.concatenate(serialized))
        
    def __len__(self):
        return len(self._addr)
    
    def __getitem__(self, idx: int):
        start = 0 if idx == 0 else self._addr[idx - 1].item()
        end = self._addr[idx].item()
        return pickle.loads(bytes(self._lst[start:end].numpy()))
