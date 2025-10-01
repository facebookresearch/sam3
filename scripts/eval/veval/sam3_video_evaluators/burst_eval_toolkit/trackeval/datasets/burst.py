# flake8: noqa

import os

from ...trackeval import utils
from ...trackeval.datasets.burst_helpers.burst_base import BURSTBase
from ...trackeval.datasets.burst_helpers.format_converter import (
    GroundTruthBURSTFormatToTAOFormatConverter,
    PredictionBURSTFormatToTAOFormatConverter,
)


class BURST(BURSTBase):
    """Dataset class for TAO tracking"""

    @staticmethod
    def get_default_dataset_config():
        burst_config = BURSTBase.get_default_dataset_config()
        code_path = utils.get_code_path()

        # e.g. 'data/gt/tsunami/exemplar_guided/'
        burst_config["GT_FOLDER"] = os.path.join(
            code_path, "data/gt/burst/val/"
        )  # Location of GT data
        # e.g. 'data/trackers/tsunami/exemplar_guided/mask_guided/validation/'
        burst_config["TRACKERS_FOLDER"] = os.path.join(
            code_path, "data/trackers/burst/class-guided/"
        )  # Trackers location
        # set to True or False
        burst_config["EXEMPLAR_GUIDED"] = False
        return burst_config

    def _iou_type(self):
        return "mask"

    def _box_or_mask_from_det(self, det):
        return det["segmentation"]

    def _calculate_area_for_ann(self, ann):
        import pycocotools.mask as cocomask

        return cocomask.area(ann["segmentation"])

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_mask_ious(
            gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False
        )
        return similarity_scores

    def _is_exemplar_guided(self):
        exemplar_guided = self.config["EXEMPLAR_GUIDED"]
        return exemplar_guided

    def _postproc_ground_truth_data(self, data):
        return GroundTruthBURSTFormatToTAOFormatConverter(data).convert()

    def _postproc_prediction_data(self, data):
        return PredictionBURSTFormatToTAOFormatConverter(
            self.gt_data, data, exemplar_guided=self._is_exemplar_guided()
        ).convert()
