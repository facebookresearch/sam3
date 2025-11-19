# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import json
import logging

import numpy as np
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from typing_extensions import override


class YTVIS(COCO):
    """
    Helper class for reading YT-VIS annotations
    """

    @override
    def __init__(self, annotation_file: str = None, ignore_gt_cats: bool = True):
        """
        Args:
            annotation_file: Path to the annotation file
            ignore_gt_cats: If True, we ignore the ground truth categories and replace them with a dummy "object" category. This is useful for Phrase AP evaluation.
        """
        self.ignore_gt_cats = ignore_gt_cats
        super().__init__(annotation_file=annotation_file)

    @override
    def createIndex(self):
        # We rename some keys to match the COCO format before creating the index.
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                if "video_id" in ann:
                    ann["image_id"] = int(ann.pop("video_id"))
                if self.ignore_gt_cats:
                    ann["category_id"] = -1
                else:
                    ann["category_id"] = int(ann["category_id"])
                if "bboxes" in ann:
                    # note that in some datasets we load under this YTVIS class,
                    # some "bboxes" could be None for when the GT object is invisible,
                    # so we replace them with [0, 0, 0, 0]
                    ann["bboxes"] = [
                        bbox if bbox is not None else [0, 0, 0, 0]
                        for bbox in ann["bboxes"]
                    ]
                if "areas" in ann:
                    # similar to "bboxes", some areas could be None for when the GT
                    # object is invisible, so we replace them with 0
                    areas = [a if a is not None else 0 for a in ann["areas"]]
                    # Compute average area of tracklet
                    ann["area"] = np.mean(areas)
        if "videos" in self.dataset:
            for vid in self.dataset["videos"]:
                vid["id"] = int(vid["id"])
            self.dataset["images"] = self.dataset.pop("videos")

        if self.ignore_gt_cats:
            self.dataset["categories"] = [
                {"supercategory": "object", "id": -1, "name": "object"}
            ]
        else:
            for cat in self.dataset["categories"]:
                cat["id"] = int(cat["id"])
        super().createIndex()

    @override
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        if len(areaRng) > 0:
            logging.warning(
                "Note that we filter out objects based on their *average* area across the video, not per frame area"
            )

        return super().getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=iscrowd)

    @override
    def showAnns(self, anns, draw_bbox=False):
        raise NotImplementedError("Showing annotations is not supported")

    @override
    def loadRes(self, resFile):
        # Adapted from COCO.loadRes to support tracklets/masklets
        res = YTVIS(ignore_gt_cats=self.ignore_gt_cats)
        res.dataset["images"] = [img for img in self.dataset["images"]]

        if type(resFile) == str:
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results is not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (
            set(annsImgIds) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "bboxes" in anns[0] and not anns[0]["bboxes"] == []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                bbs = [(bb if bb is not None else [0, 0, 0, 0]) for bb in ann["bboxes"]]
                xxyy = [[bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]] for bb in bbs]
                if not "segmentations" in ann:
                    ann["segmentations"] = [
                        [[x1, y1, x1, y2, x2, y2, x2, y1]] for (x1, x2, y1, y2) in xxyy
                    ]
                ann["areas"] = [bb[2] * bb[3] for bb in bbs]
                # NOTE: We also compute average area of a tracklet across video, allowing us to compute area based mAP.
                ann["area"] = np.mean(ann["areas"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentations" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                ann["bboxes"] = [
                    mask_util.toBbox(segm) for segm in ann["segmentations"]
                ]
                if "areas" not in ann:
                    ann["areas"] = [
                        mask_util.area(segm) for segm in ann["segmentations"]
                    ]
                # NOTE: We also compute average area of a tracklet across video, allowing us to compute area based mAP.
                ann["area"] = np.mean(ann["areas"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    @override
    def download(self, tarDir=None, imgIds=[]):
        raise NotImplementedError

    @override
    def loadNumpyAnnotations(self, data):
        raise NotImplementedError("We don't support numpy annotations for now")

    @override
    def annToRLE(self, ann):
        raise NotImplementedError("We expect masks to be already in RLE format")

    @override
    def annToMask(self, ann):
        raise NotImplementedError("We expect masks to be already in RLE format")
