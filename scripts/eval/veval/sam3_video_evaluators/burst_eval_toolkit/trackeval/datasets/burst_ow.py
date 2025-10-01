# flake8: noqa

import json
import os

from .. import utils
from .burst_helpers.burst_ow_base import BURST_OW_Base
from .burst_helpers.format_converter import (
    GroundTruthBURSTFormatToTAOFormatConverter,
    PredictionBURSTFormatToTAOFormatConverter,
)


class BURST_OW(BURST_OW_Base):
    """Dataset class for TAO tracking"""

    @staticmethod
    def get_default_dataset_config():
        tao_config = BURST_OW_Base.get_default_dataset_config()
        code_path = utils.get_code_path()
        tao_config["GT_FOLDER"] = os.path.join(
            code_path, "data/gt/burst/all_classes/val/"
        )  # Location of GT data
        tao_config["TRACKERS_FOLDER"] = os.path.join(
            code_path, "data/trackers/burst/open-world/val/"
        )  # Trackers location
        return tao_config

    def _iou_type(self):
        return "mask"

    def _box_or_mask_from_det(self, det):
        if "segmentation" in det:
            return det["segmentation"]
        else:
            return det["mask"]

    def _calculate_area_for_ann(self, ann):
        import pycocotools.mask as cocomask

        seg = self._box_or_mask_from_det(ann)
        return cocomask.area(seg)

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_mask_ious(
            gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False
        )
        return similarity_scores

    def _postproc_ground_truth_data(self, data):
        return GroundTruthBURSTFormatToTAOFormatConverter(data).convert()

    def _postproc_prediction_data(self, data):
        # if it's a list, it's already in TAO format and not in Ali format
        # however the image ids do not match and need to be remapped
        if isinstance(data, list):
            _remap_image_ids(data, self.gt_data)
            return data

        return PredictionBURSTFormatToTAOFormatConverter(
            self.gt_data, data, exemplar_guided=False
        ).convert()


def _remap_image_ids(pred_data, ali_gt_data):
    code_path = utils.get_code_path()
    if "split" in ali_gt_data:
        split = ali_gt_data["split"]
    else:
        split = "val"

    if split in ("val", "validation"):
        tao_gt_path = os.path.join(code_path, "data/gt/tao/tao_validation/gt.json")
    else:
        tao_gt_path = os.path.join(
            code_path, "data/gt/tao/tao_test/test_without_annotations.json"
        )

    with open(tao_gt_path) as f:
        tao_gt = json.load(f)

    tao_img_by_id = {}
    for img in tao_gt["images"]:
        img_id = img["id"]
        tao_img_by_id[img_id] = img

    ali_img_id_by_filename = {}
    for ali_img in ali_gt_data["images"]:
        ali_img_id = ali_img["id"]
        file_name = ali_img["file_name"].replace("validation", "val")
        ali_img_id_by_filename[file_name] = ali_img_id

    ali_img_id_by_tao_img_id = {}
    for tao_img_id, tao_img in tao_img_by_id.items():
        file_name = tao_img["file_name"]
        ali_img_id = ali_img_id_by_filename[file_name]
        ali_img_id_by_tao_img_id[tao_img_id] = ali_img_id

    for det in pred_data:
        tao_img_id = det["image_id"]
        ali_img_id = ali_img_id_by_tao_img_id[tao_img_id]
        det["image_id"] = ali_img_id
