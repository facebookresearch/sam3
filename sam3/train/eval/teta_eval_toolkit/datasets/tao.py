# fmt: off
# flake8: noqa

"""TAO Dataset."""
import copy
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from .. import _timing
from ..config import get_default_dataset_config, init_config
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset


class TAO(_BaseDataset):
    """Dataset class for TAO tracking"""

    def __init__(self, config=None):
        """Initialize dataset, checking that all required files are present."""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = init_config(config, get_default_dataset_config(), self.get_name())
        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.should_classes_combine = True
        self.use_super_categories = False
        self.use_mask = self.config["USE_MASK"]


        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        if self.gt_fol.endswith(".json"):
            self.gt_data = json.load(open(self.gt_fol, "r"))
        else:
            gt_dir_files = [
                file for file in os.listdir(self.gt_fol) if file.endswith(".json")
            ]
            if len(gt_dir_files) != 1:
                raise TrackEvalException(
                    f"{self.gt_fol} does not contain exactly one json file."
                )

            with open(os.path.join(self.gt_fol, gt_dir_files[0])) as f:
                self.gt_data = json.load(f)

        # merge categories marked with a merged tag in TAO dataset
        self._merge_categories(self.gt_data["annotations"] + self.gt_data["tracks"])

        # get sequences to eval and sequence information
        self.seq_list = [
            vid["name"].replace("/", "-") for vid in self.gt_data["videos"]
        ]
        self.seq_name2seqid = {
            vid["name"].replace("/", "-"): vid["id"] for vid in self.gt_data["videos"]
        }
        # compute mappings from videos to annotation data
        self.video2gt_track, self.video2gt_image = self._compute_vid_mappings(
            self.gt_data["annotations"]
        )
        # compute sequence lengths
        self.seq_lengths = {vid["id"]: 0 for vid in self.gt_data["videos"]}
        for img in self.gt_data["images"]:
            self.seq_lengths[img["video_id"]] += 1
        self.seq2images2timestep = self._compute_image_to_timestep_mappings()
        self.seq2cls = {
            vid["id"]: {
                "pos_cat_ids": list(
                    {track["category_id"] for track in self.video2gt_track[vid["id"]]}
                ),
                "neg_cat_ids": vid["neg_category_ids"],
                "not_exh_labeled_cat_ids": vid["not_exhaustive_category_ids"],
            }
            for vid in self.gt_data["videos"]
        }

        # Get classes to eval
        considered_vid_ids = [self.seq_name2seqid[vid] for vid in self.seq_list]
        seen_cats = set(
            [
                cat_id
                for vid_id in considered_vid_ids
                for cat_id in self.seq2cls[vid_id]["pos_cat_ids"]
            ]
        )
        # only classes with ground truth are evaluated in TAO
        self.valid_classes = [
            cls["name"] for cls in self.gt_data["categories"] if cls["id"] in seen_cats
        ]
        cls_name2clsid_map = {
            cls["name"]: cls["id"] for cls in self.gt_data["categories"]
        }

        if self.config["CLASSES_TO_EVAL"]:
            self.class_list = [
                cls.lower() if cls.lower() in self.valid_classes else None
                for cls in self.config["CLASSES_TO_EVAL"]
            ]
            if not all(self.class_list):
                valid_cls = ", ".join(self.valid_classes)
                raise TrackEvalException(
                    "Attempted to evaluate an invalid class. Only classes "
                    f"{valid_cls} are valid (classes present in ground truth"
                    " data)."
                )
        else:
            self.class_list = [cls for cls in self.valid_classes]
        self.cls_name2clsid = {
            k: v for k, v in cls_name2clsid_map.items() if k in self.class_list
        }
        self.clsid2cls_name = {
            v: k for k, v in cls_name2clsid_map.items() if k in self.class_list
        }
        # get trackers to eval
        print(self.config["TRACKERS_TO_EVAL"] )
        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config["TRACKERS_TO_EVAL"] is not None) and (
            len(self.config["TK_DISPLAY_NAMES"]) == len(self.tracker_list)
        ):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.config["TK_DISPLAY_NAMES"])
            )
        else:
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )

        self.tracker_data = {tracker: dict() for tracker in self.tracker_list}

        for tracker in self.tracker_list:
            if self.tracker_sub_fol.endswith(".json"):
                with open(os.path.join(self.tracker_sub_fol)) as f:
                    curr_data = json.load(f)
            else:
                tr_dir = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol)
                tr_dir_files = [
                    file for file in os.listdir(tr_dir) if file.endswith(".json")
                ]
                if len(tr_dir_files) != 1:
                    raise TrackEvalException(
                        f"{tr_dir} does not contain exactly one json file."
                    )
                with open(os.path.join(tr_dir, tr_dir_files[0])) as f:
                    curr_data = json.load(f)

            # limit detections if MAX_DETECTIONS > 0
            if self.config["MAX_DETECTIONS"]:
                curr_data = self._limit_dets_per_image(curr_data)

            # fill missing video ids
            self._fill_video_ids_inplace(curr_data)

            # make track ids unique over whole evaluation set
            self._make_tk_ids_unique(curr_data)

            # merge categories marked with a merged tag in TAO dataset
            self._merge_categories(curr_data)

            # get tracker sequence information
            curr_vids2tracks, curr_vids2images = self._compute_vid_mappings(curr_data)
            self.tracker_data[tracker]["vids_to_tracks"] = curr_vids2tracks
            self.tracker_data[tracker]["vids_to_images"] = curr_vids2images

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the TAO format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes]:
            list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.

        if not is_gt, this returns a dict which contains the fields:
        [tk_ids, tk_classes, tk_confidences]:
            list (for each timestep) of 1D NDArrays (for each det).
        [tk_dets]: list (for each timestep) of lists of detections.
        """
        seq_id = self.seq_name2seqid[seq]
        # file location
        if is_gt:
            imgs = self.video2gt_image[seq_id]
        else:
            imgs = self.tracker_data[tracker]["vids_to_images"][seq_id]

        # convert data to required format
        num_timesteps = self.seq_lengths[seq_id]
        img_to_timestep = self.seq2images2timestep[seq_id]
        data_keys = ["ids", "classes", "dets"]
        if not is_gt:
            data_keys += ["tk_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for img in imgs:
            # some tracker data contains images without any ground truth info,
            # these are ignored
            if img["id"] not in img_to_timestep:
                continue
            t = img_to_timestep[img["id"]]
            anns = img["annotations"]
            if self.use_mask:
                # When using mask, extract segmentation data
                raw_data["dets"][t] = [ann.get("segmentation") for ann in anns]
            else:
                # When using bbox, extract bbox data
                raw_data["dets"][t] = np.atleast_2d([ann["bbox"] for ann in anns]).astype(
                    float
                )
            raw_data["ids"][t] = np.atleast_1d(
                [ann["track_id"] for ann in anns]
            ).astype(int)
            raw_data["classes"][t] = np.atleast_1d(
                [ann["category_id"] for ann in anns]
            ).astype(int)
            if not is_gt:
                raw_data["tk_confidences"][t] = np.atleast_1d(
                    [ann["score"] for ann in anns]
                ).astype(float)

        for t, d in enumerate(raw_data["dets"]):
            if d is None:
                raw_data["dets"][t] = np.empty((0, 4)).astype(float)
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if not is_gt:
                    raw_data["tk_confidences"][t] = np.empty(0)

        if is_gt:
            key_map = {"ids": "gt_ids", "classes": "gt_classes", "dets": "gt_dets"}
        else:
            key_map = {"ids": "tk_ids", "classes": "tk_classes", "dets": "tk_dets"}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data["num_timesteps"] = num_timesteps
        raw_data["neg_cat_ids"] = self.seq2cls[seq_id]["neg_cat_ids"]
        raw_data["not_exh_labeled_cls"] = self.seq2cls[seq_id][
            "not_exh_labeled_cat_ids"
        ]
        raw_data["seq"] = seq
        return raw_data

    def get_preprocessed_seq_data_thr(self, raw_data, cls, assignment=None):
        """Preprocess data for a single sequence for a single class.

        Inputs:
            raw_data: dict containing the data for the sequence already
                read in by get_raw_seq_data().
            cls: class to be evaluated.
        Outputs:
            gt_ids:
                list (for each timestep) of ids of GT tracks
            tk_ids:
                list (for each timestep) of ids of predicted tracks (all for TP
                matching (Det + AssocA))
            tk_overlap_ids:
                list (for each timestep) of ids of predicted tracks that overlap
                with GTs
            tk_neg_ids:
                list (for each timestep) of ids of predicted tracks that with
                the class id on the negative list for the current sequence.
            tk_exh_ids:
                list (for each timestep) of ids of predicted tracks that do not
                overlap with existing GTs but have the class id on the
                exhaustive annotated class list for the current sequence.
            tk_dets:
                list (for each timestep) of lists of detections that
                corresponding to the tk_ids
            tk_classes:
                list (for each timestep) of lists of classes that corresponding
                to the tk_ids
            tk_confidences:
                list (for each timestep) of lists of classes that corresponding
                to the tk_ids
            sim_scores:
                similarity score between gt_ids and tk_ids.
        """
        if cls != "all":
            cls_id = self.cls_name2clsid[cls]

        data_keys = [
            "gt_ids",
            "tk_ids",
            "gt_id_map",
            "tk_id_map",
            "gt_dets",
            "gt_classes",
            "gt_class_name",
            "tk_overlap_classes",
            "tk_overlap_ids",
            "tk_neg_ids",
            "tk_exh_ids",
            "tk_class_eval_tk_ids",
            "tk_dets",
            "tk_classes",
            "tk_confidences",
            "sim_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tk_ids = []
        num_gt_dets = 0
        num_tk_cls_dets = 0
        num_tk_overlap_dets = 0
        overlap_ious_thr = 0.5
        loc_and_asso_tk_ids = []

        for t in range(raw_data["num_timesteps"]):
            # only extract relevant dets for this class for preproc and eval
            if cls == "all":
                gt_class_mask = np.ones_like(raw_data["gt_classes"][t]).astype(bool)
            else:
                gt_class_mask = np.atleast_1d(
                    raw_data["gt_classes"][t] == cls_id
                ).astype(bool)

            # select GT that is not in the evaluating classes
            if assignment is not None and assignment:
                all_gt_ids = list(assignment[t].keys())
                gt_ids_in = raw_data["gt_ids"][t][gt_class_mask]
                gt_ids_out = set(all_gt_ids) - set(gt_ids_in)
                tk_ids_out = set([assignment[t][key] for key in list(gt_ids_out)])

            # compute overlapped tracks and add their ids to overlap_tk_ids
            sim_scores = raw_data["similarity_scores"]
            overlap_ids_masks = (sim_scores[t][gt_class_mask] >= overlap_ious_thr).any(
                axis=0
            )
            overlap_tk_ids_t = raw_data["tk_ids"][t][overlap_ids_masks]
            if assignment is not None and assignment:
                data["tk_overlap_ids"][t] = list(set(overlap_tk_ids_t) - tk_ids_out)
            else:
                data["tk_overlap_ids"][t] = list(set(overlap_tk_ids_t))

            loc_and_asso_tk_ids += data["tk_overlap_ids"][t]

            data["tk_exh_ids"][t] = []
            data["tk_neg_ids"][t] = []

            if cls == "all":
                continue

        # remove tk_ids that has been assigned to GT belongs to other classes.
        loc_and_asso_tk_ids = list(set(loc_and_asso_tk_ids))

        # remove all unwanted unmatched tracker detections
        for t in range(raw_data["num_timesteps"]):
            # add gt to the data
            if cls == "all":
                gt_class_mask = np.ones_like(raw_data["gt_classes"][t]).astype(bool)
            else:
                gt_class_mask = np.atleast_1d(
                    raw_data["gt_classes"][t] == cls_id
                ).astype(bool)
                data["gt_classes"][t] = cls_id
                data["gt_class_name"][t] = cls

            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            if self.use_mask:
                gt_dets = [raw_data['gt_dets'][t][ind] for ind in range(len(gt_class_mask)) if gt_class_mask[ind]]
            else:
                gt_dets = raw_data["gt_dets"][t][gt_class_mask]
            data["gt_ids"][t] = gt_ids
            data["gt_dets"][t] = gt_dets

            # filter pred and only keep those that highly overlap with GTs
            tk_mask = np.isin(
                raw_data["tk_ids"][t], np.array(loc_and_asso_tk_ids), assume_unique=True
            )
            tk_overlap_mask = np.isin(
                raw_data["tk_ids"][t],
                np.array(data["tk_overlap_ids"][t]),
                assume_unique=True,
            )

            tk_ids = raw_data["tk_ids"][t][tk_mask]
            if self.use_mask:
                tk_dets = [raw_data['tk_dets'][t][ind] for ind in range(len(tk_mask)) if
                            tk_mask[ind]]
            else:
                tk_dets = raw_data["tk_dets"][t][tk_mask]
            tracker_classes = raw_data["tk_classes"][t][tk_mask]

            # add overlap classes for computing the FP for Cls term
            tracker_overlap_classes = raw_data["tk_classes"][t][tk_overlap_mask]
            tracker_confidences = raw_data["tk_confidences"][t][tk_mask]
            sim_scores_masked = sim_scores[t][gt_class_mask, :][:, tk_mask]

            # add filtered prediction to the data
            data["tk_classes"][t] = tracker_classes
            data["tk_overlap_classes"][t] = tracker_overlap_classes
            data["tk_ids"][t] = tk_ids
            data["tk_dets"][t] = tk_dets
            data["tk_confidences"][t] = tracker_confidences
            data["sim_scores"][t] = sim_scores_masked
            data["tk_class_eval_tk_ids"][t] = set(
                list(data["tk_overlap_ids"][t])
                + list(data["tk_neg_ids"][t])
                + list(data["tk_exh_ids"][t])
            )

            # count total number of detections
            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            # the unique track ids are for association.
            unique_tk_ids += list(np.unique(data["tk_ids"][t]))

            num_tk_overlap_dets += len(data["tk_overlap_ids"][t])
            num_tk_cls_dets += len(data["tk_class_eval_tk_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        # re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            data["gt_id_map"] = {}
            for gt_id in unique_gt_ids:
                new_gt_id = gt_id_map[gt_id].astype(int)
                data["gt_id_map"][new_gt_id] = gt_id

            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(int)

        if len(unique_tk_ids) > 0:
            unique_tk_ids = np.unique(unique_tk_ids)
            tk_id_map = np.nan * np.ones((np.max(unique_tk_ids) + 1))
            tk_id_map[unique_tk_ids] = np.arange(len(unique_tk_ids))

            data["tk_id_map"] = {}
            for track_id in unique_tk_ids:
                new_track_id = tk_id_map[track_id].astype(int)
                data["tk_id_map"][new_track_id] = track_id

            for t in range(raw_data["num_timesteps"]):
                if len(data["tk_ids"][t]) > 0:
                    data["tk_ids"][t] = tk_id_map[data["tk_ids"][t]].astype(int)
                if len(data["tk_overlap_ids"][t]) > 0:
                    data["tk_overlap_ids"][t] = tk_id_map[
                        data["tk_overlap_ids"][t]
                    ].astype(int)

        # record overview statistics.
        data["num_tk_cls_dets"] = num_tk_cls_dets
        data["num_tk_overlap_dets"] = num_tk_overlap_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tk_ids"] = len(unique_tk_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        self._check_unique_ids(data)

        return data

    @_timing.time
    def get_preprocessed_seq_data(
        self, raw_data, cls, assignment=None, thresholds=[50, 75]
    ):
        """Preprocess data for a single sequence for a single class."""
        data = {}
        if thresholds is None:
            thresholds = [50]
        elif isinstance(thresholds, int):
            thresholds = [thresholds]

        for thr in thresholds:
            assignment_thr = None
            if assignment is not None:
                assignment_thr = assignment[thr]
            data[thr] = self.get_preprocessed_seq_data_thr(
                raw_data, cls, assignment_thr
            )

        return data

    def _calculate_similarities(self, gt_dets_t, tk_dets_t):
        """Compute similarity scores."""
        if self.use_mask:
            similarity_scores = self._calculate_mask_ious(gt_dets_t, tk_dets_t, is_encoded=True, do_ioa=False)
        else:
            similarity_scores = self._calculate_box_ious(gt_dets_t, tk_dets_t)
        return similarity_scores

    def _merge_categories(self, annotations):
        """Merges categories with a merged tag.

        Adapted from https://github.com/TAO-Dataset.
        """
        merge_map = {}
        for category in self.gt_data["categories"]:
            if "merged" in category:
                for to_merge in category["merged"]:
                    merge_map[to_merge["id"]] = category["id"]

        for ann in annotations:
            ann["category_id"] = merge_map.get(ann["category_id"], ann["category_id"])

    def _compute_vid_mappings(self, annotations):
        """Computes mappings from videos to corresponding tracks and images."""
        vids_to_tracks = {}
        vids_to_imgs = {}
        vid_ids = [vid["id"] for vid in self.gt_data["videos"]]

        # compute an mapping from image IDs to images
        images = {}
        for image in self.gt_data["images"]:
            images[image["id"]] = image

        for ann in annotations:
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]

            vid = ann["video_id"]
            if ann["video_id"] not in vids_to_tracks.keys():
                vids_to_tracks[ann["video_id"]] = list()
            if ann["video_id"] not in vids_to_imgs.keys():
                vids_to_imgs[ann["video_id"]] = list()

            # fill in vids_to_tracks
            tid = ann["track_id"]
            exist_tids = [track["id"] for track in vids_to_tracks[vid]]
            try:
                index1 = exist_tids.index(tid)
            except ValueError:
                index1 = -1
            if tid not in exist_tids:
                curr_track = {
                    "id": tid,
                    "category_id": ann["category_id"],
                    "video_id": vid,
                    "annotations": [ann],
                }
                vids_to_tracks[vid].append(curr_track)
            else:
                vids_to_tracks[vid][index1]["annotations"].append(ann)

            # fill in vids_to_imgs
            img_id = ann["image_id"]
            exist_img_ids = [img["id"] for img in vids_to_imgs[vid]]
            try:
                index2 = exist_img_ids.index(img_id)
            except ValueError:
                index2 = -1
            if index2 == -1:
                curr_img = {"id": img_id, "annotations": [ann]}
                vids_to_imgs[vid].append(curr_img)
            else:
                vids_to_imgs[vid][index2]["annotations"].append(ann)

        # sort annotations by frame index and compute track area
        for vid, tracks in vids_to_tracks.items():
            for track in tracks:
                track["annotations"] = sorted(
                    track["annotations"],
                    key=lambda x: images[x["image_id"]]["frame_index"],
                )
                # compute average area
                track["area"] = sum(x["area"] for x in track["annotations"]) / len(
                    track["annotations"]
                )

        # ensure all videos are present
        for vid_id in vid_ids:
            if vid_id not in vids_to_tracks.keys():
                vids_to_tracks[vid_id] = []
            if vid_id not in vids_to_imgs.keys():
                vids_to_imgs[vid_id] = []

        return vids_to_tracks, vids_to_imgs

    def _compute_image_to_timestep_mappings(self):
        """Computes a mapping from images to timestep in sequence."""
        images = {}
        for image in self.gt_data["images"]:
            images[image["id"]] = image

        seq_to_imgs_to_timestep = {vid["id"]: dict() for vid in self.gt_data["videos"]}
        for vid in seq_to_imgs_to_timestep:
            curr_imgs = [img["id"] for img in self.video2gt_image[vid]]
            curr_imgs = sorted(curr_imgs, key=lambda x: images[x]["frame_index"])
            seq_to_imgs_to_timestep[vid] = {
                curr_imgs[i]: i for i in range(len(curr_imgs))
            }

        return seq_to_imgs_to_timestep

    def _limit_dets_per_image(self, annotations):
        """Limits the number of detections for each image.

        Adapted from https://github.com/TAO-Dataset/.
        """
        max_dets = self.config["MAX_DETECTIONS"]
        img_ann = defaultdict(list)
        for ann in annotations:
            img_ann[ann["image_id"]].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda x: x["score"], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]

    def _fill_video_ids_inplace(self, annotations):
        """Fills in missing video IDs inplace.

        Adapted from https://github.com/TAO-Dataset/.
        """
        missing_video_id = [x for x in annotations if "video_id" not in x]
        if missing_video_id:
            image_id_to_video_id = {
                x["id"]: x["video_id"] for x in self.gt_data["images"]
            }
            for x in missing_video_id:
                x["video_id"] = image_id_to_video_id[x["image_id"]]

    @staticmethod
    def _make_tk_ids_unique(annotations):
        """Makes track IDs unqiue over the whole annotation set.

        Adapted from https://github.com/TAO-Dataset/.
        """
        track_id_videos = {}
        track_ids_to_update = set()
        max_track_id = 0
        for ann in annotations:
            t = ann["track_id"]
            if t not in track_id_videos:
                track_id_videos[t] = ann["video_id"]

            if ann["video_id"] != track_id_videos[t]:
                # track id is assigned to multiple videos
                track_ids_to_update.add(t)
            max_track_id = max(max_track_id, t)

        if track_ids_to_update:
            print("true")
            next_id = itertools.count(max_track_id + 1)
            new_tk_ids = defaultdict(lambda: next(next_id))
            for ann in annotations:
                t = ann["track_id"]
                v = ann["video_id"]
                if t in track_ids_to_update:
                    ann["track_id"] = new_tk_ids[t, v]
        return len(track_ids_to_update)
