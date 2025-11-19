# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Callable, Iterable, Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset


class TorchDataset:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        enable_distributed_sampler=True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert not isinstance(self.dataset, IterableDataset), "Not supported yet"
        if enable_distributed_sampler:
            self.sampler = DistributedSampler(self.dataset, shuffle=self.shuffle)
        else:
            self.sampler = None

    def get_loader(self, epoch) -> Iterable:
        if self.sampler:
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )
