# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import builtins
import copy
import json
import os

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sam3.eval.coco_eval import CocoEvaluator
from sam3.eval.coco_eval_offline import (
    COCOevalCustom,
    CocoEvaluatorOfflineWithPredFileEvaluators,
)
from sam3.eval.ultrafast_coco import create_offline_evaluator

pytest.importorskip("ultrafast_pycocotools")


class Passthrough:
    def process_results(self, predictions):
        return copy.deepcopy(predictions)


def inputs(normalized=True):
    gt = COCO()
    masks = np.zeros((32, 32, 3), dtype=np.uint8, order="F")
    masks[2:14, 2:14, :2] = 1
    masks[18:30, 18:30, 2] = 1
    rles = mask_utils.encode(masks)
    for rle in rles:
        rle["counts"] = rle["counts"].decode()
    pose = torch.tensor(
        [[3 + i % 4, 4 + i // 4, 2] for i in range(17)], dtype=torch.float32
    )
    images = [
        dict(
            id=i, width=32, height=32, rarity=(i - 1) % 4, is_instance_exhaustive=i != 2
        )
        for i in range(1, 7)
    ]
    anns = []
    for i in range(1, 6):
        anns.append(
            dict(
                id=i,
                image_id=i,
                category_id=1 + i % 2,
                bbox=[2, 2, 12, 12],
                area=144 / 1024 if normalized else 144,
                segmentation=rles[0],
                keypoints=pose.flatten().tolist(),
                num_keypoints=17,
                iscrowd=int(i == 5),
            )
        )
    gt.dataset = dict(
        info={},
        images=images,
        categories=[dict(id=1, name="one"), dict(id=2, name="two")],
        annotations=anns,
    )
    gt.createIndex()
    predictions = {}
    for i in range(1, 6):
        predictions[i] = dict(
            boxes=torch.tensor(
                [
                    [18.0, 18.0, 30.0, 30.0],
                    [2.0, 2.0, 14.0, 14.0],
                    [2.0, 2.0, 14.0, 14.0],
                ]
            ),
            scores=torch.tensor([0.9, 0.8, 0.8]),
            labels=torch.tensor([1 + i % 2] * 3),
            masks=torch.from_numpy(
                masks[:, :, [2, 0, 1]].transpose(2, 0, 1).copy()
            ).unsqueeze(1),
            keypoints=torch.stack([pose + torch.tensor([16, 16, 0]), pose, pose]),
        )
    # Exercise the already-encoded RLE path as well as raw tensor masks.
    predictions[2]["masks_rle"] = [rles[2], rles[0], rles[1]]
    del predictions[2]["masks"]
    predictions[6] = {}
    return gt, predictions


def make(gt, backend, iou_types=("bbox", "segm"), **kwargs):
    return CocoEvaluator(
        copy.deepcopy(gt),
        list(iou_types),
        useCats=kwargs.pop("useCats", True),
        dump_dir=kwargs.pop("dump_dir", None),
        postprocessor=Passthrough(),
        backend=backend,
        **kwargs,
    )


def assert_arrays(actual, expected):
    for kind in expected.coco_evals[0]:
        a, e = actual.coco_evals[0][kind], expected.coco_evals[0][kind]
        for key in ("precision", "recall", "scores"):
            np.testing.assert_array_equal(a.eval[key], e.eval[key])
        assert a.stats[0] == e.stats[0]
        np.testing.assert_array_equal(a.stats[1], e.stats[1])


@pytest.mark.parametrize(
    "iou_types", [("bbox",), ("segm",), ("keypoints",), ("bbox", "segm")]
)
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("use_cats", [False, True])
def test_online_arrays_reset_and_custom_caps(iou_types, normalized, use_cats):
    if iou_types == ("keypoints",) and not use_cats:
        pytest.skip(
            "pycocotools computeOks does not support category-agnostic keypoints"
        )
    gt, predictions = inputs(normalized)
    candidates = [
        make(
            gt,
            backend,
            iou_types,
            use_normalized_areas=normalized,
            useCats=use_cats,
            maxdets=[1, 10, 500],
        )
        for backend in ("pycocotools", "ultrafast")
    ]
    for epoch in range(2):
        metrics = []
        for evaluator in candidates:
            evaluator._lazy_init()
            evaluator.reset()
            evaluator.update({})
            # Missing GT image 1 must still contribute to recall.
            evaluator.update({3: predictions[3], 2: predictions[2]})
            evaluator.update({4: predictions[4] if epoch == 0 else {}, 5: {}, 6: {}})
            # Conflicting duplicate: keep the first occurrence.
            evaluator.update({2: {}})
            metrics.append(evaluator.compute_synced())
        assert metrics[0] == metrics[1]
        assert_arrays(candidates[1], candidates[0])
        for evaluator in candidates:
            evaluator.accumulate([2, 4])
        for kind in iou_types:
            for key in ("precision", "recall", "scores"):
                np.testing.assert_array_equal(
                    candidates[1].coco_evals[0][kind].eval[key],
                    candidates[0].coco_evals[0][kind].eval[key],
                )
        # Return to the complete image set after a subset evaluation.
        assert candidates[0].summarize() == candidates[1].summarize()


@pytest.mark.parametrize("rarity", [False, True])
@pytest.mark.parametrize("exhaustive", [False, True])
def test_rarity_exhaustive_and_lazy_gt(tmp_path, rarity, exhaustive):
    gt, predictions = inputs()
    for ann in gt.dataset["annotations"]:
        ann["segmentation"]["counts"] = (
            ann["segmentation"]["counts"].decode()
            if isinstance(ann["segmentation"]["counts"], bytes)
            else ann["segmentation"]["counts"]
        )
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt.dataset))
    metrics = []
    for backend in ("pycocotools", "ultrafast"):
        evaluator = make(
            str(path), backend, average_by_rarity=rarity, exhaustive_only=exhaustive
        )
        evaluator.update(predictions)
        metrics.append(evaluator.compute_synced())
        assert evaluator.eval_img_ids == ([1, 3, 4, 5, 6] if exhaustive else None)
    assert metrics[0] == metrics[1]
    if rarity:
        assert "coco_eval_masks_rare_AP" in metrics[1]
        assert "coco_eval_masks_whole_image" not in metrics[1]
        assert "coco_eval_masks_AP_whole_image" in metrics[1]


@pytest.mark.parametrize("empty_update", [False, True])
def test_empty_dataset_predictions(empty_update):
    gt, _ = inputs()
    metrics = []
    for backend in ("pycocotools", "ultrafast"):
        evaluator = make(gt, backend, useCats=False)
        if empty_update:
            evaluator.update({i: {} for i in range(1, 7)})
        metrics.append(evaluator.compute_synced())
    assert metrics[0] == metrics[1]


def test_dumps(tmp_path):
    gt, predictions = inputs()
    all_dumps = []
    for backend in ("pycocotools", "ultrafast"):
        dump_dir, metrics_dir = (
            tmp_path / backend / "pred",
            tmp_path / backend / "metrics",
        )
        evaluator = make(
            gt,
            backend,
            ("segm",),
            dump_dir=str(dump_dir),
            metrics_dump_dir=str(metrics_dir),
        )
        evaluator.update(predictions)
        evaluator.compute_synced()
        all_dumps.append(
            (
                json.loads((dump_dir / "coco_predictions_0.json").read_text()),
                json.loads((metrics_dir / "coco_eval_img_metrics_0.json").read_text()),
            )
        )
    assert all_dumps[0] == all_dumps[1]


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
@pytest.mark.parametrize("positive", [False, True])
def test_offline_positive_split_and_public_file_api(tmp_path, iou_type, positive):
    gt, predictions = inputs(False)
    preparer = make(gt, "pycocotools", (iou_type,), use_normalized_areas=False)
    results = preparer.prepare(copy.deepcopy(predictions), iou_type)
    # False positives for a category absent from the GT on this image.
    absent = copy.deepcopy(results[0])
    absent.update(category_id=1, score=0.99)
    results.insert(0, absent)
    coco_dt = gt.loadRes(copy.deepcopy(results))
    reference = COCOevalCustom(
        copy.deepcopy(gt), copy.deepcopy(coco_dt), iou_type, dt_only_positive=positive
    )
    candidate = create_offline_evaluator(gt, coco_dt, iou_type, positive)
    for evaluator in (reference, candidate):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    for key in ("precision", "recall", "scores"):
        np.testing.assert_array_equal(candidate.eval[key], reference.eval[key])
    np.testing.assert_array_equal(candidate.stats, reference.stats)

    def encode(value):
        return value.decode() if isinstance(value, bytes) else value.tolist()

    gt_path, dt_path = tmp_path / "gt.json", tmp_path / "pred.json"
    gt_path.write_text(json.dumps(gt.dataset, default=encode))
    dt_path.write_text(json.dumps(results, default=encode))
    metrics = [
        CocoEvaluatorOfflineWithPredFileEvaluators(
            str(gt_path),
            tide=False,
            iou_type=iou_type,
            positive_split=positive,
            backend=backend,
        ).evaluate(dt_path)
        for backend in ("pycocotools", "ultrafast")
    ]
    assert metrics[0] == metrics[1]


def distributed_worker(rank, init_file, directory, empty_rank, filesys):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    os.environ["EXP_DIR"] = directory
    try:
        gt, predictions = inputs()
        if empty_rank:
            local = predictions if rank == 0 else {}
        else:
            local = (
                {2: predictions[2], 3: predictions[3]}
                if rank == 0
                else dict(predictions)
            )
            if rank == 1:
                local[2] = {}
                local[3] = {}
        metrics = []
        for backend in ("pycocotools", "ultrafast"):
            evaluator = make(
                gt,
                backend,
                useCats=False,
                gather_pred_via_filesys=filesys,
                average_by_rarity=True,
            )
            evaluator.update(local)
            metrics.append(evaluator.compute_synced())
        assert metrics[0] == metrics[1]
        if rank != 0:
            assert metrics[1] == {}
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize("filesys", [False, True])
def test_two_rank_gather(tmp_path, empty_rank, filesys):
    torch.multiprocessing.spawn(
        distributed_worker,
        args=(str(tmp_path / "gloo"), str(tmp_path), empty_rank, filesys),
        nprocs=2,
        join=True,
    )


def test_missing_dependency_and_invalid_backend(monkeypatch):
    gt, _ = inputs()
    with pytest.raises(ValueError, match="Unknown COCO backend"):
        make(gt, "invalid")
    with pytest.raises(ValueError, match="Unknown COCO backend"):
        CocoEvaluatorOfflineWithPredFileEvaluators("unused", backend="invalid")
    original_import = builtins.__import__

    def no_ultrafast(name, *args, **kwargs):
        if name.startswith("ultrafast_pycocotools"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultrafast)
    make(gt, "pycocotools")
    with pytest.raises(ModuleNotFoundError, match="ultrafast-pycocotools>=0.1.11"):
        make(gt, "ultrafast")
    with pytest.raises(ModuleNotFoundError, match="ultrafast-pycocotools>=0.1.11"):
        CocoEvaluatorOfflineWithPredFileEvaluators("unused", backend="ultrafast")


def test_subset_stride_matches_independent_reference():
    gt, predictions = inputs(False)
    candidate = make(gt, "pycocotools", use_normalized_areas=False)
    candidate.update(predictions)
    candidate.synchronize_between_processes()
    candidate.accumulate([2, 4])
    for kind in candidate.iou_types:
        reference = COCOeval(copy.deepcopy(gt), iouType=kind)
        reference.cocoDt = candidate._loadRes(
            gt, candidate.prepare(copy.deepcopy(predictions), kind)
        )
        reference.params.imgIds = [2, 4]
        reference.evaluate()
        reference.accumulate()
        for key in ("precision", "recall", "scores"):
            np.testing.assert_array_equal(
                candidate.coco_evals[0][kind].eval[key], reference.eval[key]
            )
    assert list(candidate.coco_evals[0]["bbox"].params.imgIds) == list(range(1, 7))


def test_all_normalized_area_buckets():
    gt = COCO()
    gt.dataset = dict(
        info={}, images=[], categories=[dict(id=1, name="object")], annotations=[]
    )
    predictions = {}
    for image_id, side in enumerate([3, 7, 20, 50, 90, 99], 1):
        mask = np.zeros((100, 100), dtype=np.uint8, order="F")
        mask[:side, :side] = 1
        rle = mask_utils.encode(mask)
        rle["counts"] = rle["counts"].decode()
        gt.dataset["images"].append(dict(id=image_id, width=100, height=100))
        gt.dataset["annotations"].append(
            dict(
                id=image_id,
                image_id=image_id,
                category_id=1,
                area=side * side / 10000,
                bbox=[0, 0, side, side],
                segmentation=rle,
                iscrowd=0,
            )
        )
        predictions[image_id] = dict(
            scores=torch.tensor([0.9]), labels=torch.tensor([1]), masks_rle=[rle]
        )
    gt.createIndex()
    candidates = [
        make(gt, backend, ("segm",)) for backend in ("pycocotools", "ultrafast")
    ]
    results = []
    for evaluator in candidates:
        evaluator.update(predictions)
        results.append(evaluator.compute_synced())
    assert results[0] == results[1]
    assert_arrays(candidates[1], candidates[0])
    for label in ("tiny", "small", "medium", "large", "huge", "whole_image"):
        assert results[1][f"coco_eval_masks_AP_{label}"] > 0.99
