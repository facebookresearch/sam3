# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import unittest
from functools import partial
from unittest.mock import patch

import torch
from sam3.train.data.torch_dataset import TorchDataset
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler


class TestTorchDatasetShuffle(unittest.TestCase):
    def _dataset(self, shuffle, enable_distributed_sampler=False):
        return TorchDataset(
            dataset=list(range(20)),
            batch_size=5,
            num_workers=0,
            shuffle=shuffle,
            pin_memory=False,
            drop_last=False,
            enable_distributed_sampler=enable_distributed_sampler,
        )

    def test_shuffle_without_distributed_sampler(self):
        loader = self._dataset(shuffle=True).get_loader(epoch=0)
        self.assertIsInstance(loader.sampler, RandomSampler)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            values = torch.cat(list(loader)).tolist()
        self.assertEqual(sorted(values), list(range(20)))
        self.assertNotEqual(values, list(range(20)))

    def test_no_shuffle_without_distributed_sampler(self):
        loader = self._dataset(shuffle=False).get_loader(epoch=0)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(torch.cat(list(loader)).tolist(), list(range(20)))

    def test_distributed_sampler_keeps_control_of_shuffle(self):
        sampler_factory = partial(DistributedSampler, num_replicas=1, rank=0)
        for shuffle in (True, False):
            with self.subTest(shuffle=shuffle):
                with patch(
                    "sam3.train.data.torch_dataset.DistributedSampler", sampler_factory
                ):
                    dataset = self._dataset(shuffle, enable_distributed_sampler=True)
                loader = dataset.get_loader(epoch=3)
                self.assertIs(loader.sampler, dataset.sampler)
                self.assertEqual(loader.sampler.epoch, 3)
                self.assertEqual(loader.sampler.shuffle, shuffle)
                self.assertEqual(
                    torch.cat(list(loader)).tolist(), list(dataset.sampler)
                )


if __name__ == "__main__":
    unittest.main()
