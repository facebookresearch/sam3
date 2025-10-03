# flake8: noqa

# note: this file has been modified from its original version in TrackEval in
# https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/youtube_vis.py
# to support the following:
# 1) bbox evaluation (via `IOU_TYPE`)
# 2) passing GT and prediction data as Python objects (via `GT_JSON_OBJECT` and `TRACKER_JSON_OBJECT`)
# 3) specifying a custom dataset name (via `DATASET_NAME`)

import json
import os

import numpy as np

from .. import _timing, utils
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset


class YouTubeVIS(_BaseDataset):
    """Dataset class for YouTubeVIS tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/youtube_vis/"
            ),  # Location of GT data
            "TRACKERS_FOLDER": os.path.join(code_path, "data/trackers/youtube_vis/"),
            # Trackers location
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
            "CLASSES_TO_EVAL": None,  # Classes to eval (if None, all classes)
            "SPLIT_TO_EVAL": "train_sub_split",  # Valid: 'train', 'val', 'train_sub_split'
            "PRINT_CONFIG": True,  # Whether to print current config
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            # Added for video phrase AP evaluation -- allow directly specifying the GT JSON data and Tracker (result)
            # JSON data as Python objects, without reading from files.
            "GT_JSON_OBJECT": None,
            "TRACKER_JSON_OBJECT": None,
            "IOU_TYPE": "segm",
            "DATASET_NAME": "video",
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config())
        self.gt_fol = (
            self.config["GT_FOLDER"] + "youtube_vis_" + self.config["SPLIT_TO_EVAL"]
        )
        self.tracker_fol = (
            self.config["TRACKERS_FOLDER"]
            + "youtube_vis_"
            + self.config["SPLIT_TO_EVAL"]
        )
        self.use_super_categories = False
        self.should_classes_combine = True
        assert self.config["IOU_TYPE"] in ["segm", "bbox"]
        self.iou_type = self.config["IOU_TYPE"]
        print("=" * 100)
        print(f"Evaluate annotation type *{self.iou_type}*")
        self.dataset_name = self.config["DATASET_NAME"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]
        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]

        if self.config["GT_JSON_OBJECT"] is not None:
            # allow directly specifying the GT JSON data without reading from files
            gt_json = self.config["GT_JSON_OBJECT"]
            assert isinstance(gt_json, dict)
            assert "videos" in gt_json
            assert "categories" in gt_json
            assert "annotations" in gt_json
            self.gt_data = gt_json
        else:
            if not os.path.exists(self.gt_fol):
                print("GT folder not found: " + self.gt_fol)
                raise TrackEvalException(
                    "GT folder not found: " + os.path.basename(self.gt_fol)
                )
            gt_dir_files = [
                file for file in os.listdir(self.gt_fol) if file.endswith(".json")
            ]
            if len(gt_dir_files) != 1:
                raise TrackEvalException(
                    self.gt_fol + " does not contain exactly one json file."
                )

            with open(os.path.join(self.gt_fol, gt_dir_files[0])) as f:
                self.gt_data = json.load(f)

        # Get classes to eval
        self.valid_classes = [cls["name"] for cls in self.gt_data["categories"]]
        cls_name_to_cls_id_map = {
            cls["name"]: cls["id"] for cls in self.gt_data["categories"]
        }

        if self.config["CLASSES_TO_EVAL"]:
            self.class_list = [
                cls.lower() if cls.lower() in self.valid_classes else None
                for cls in self.config["CLASSES_TO_EVAL"]
            ]
            if not all(self.class_list):
                raise TrackEvalException(
                    "Attempted to evaluate an invalid class. Only classes "
                    + ", ".join(self.valid_classes)
                    + " are valid."
                )
        else:
            self.class_list = [cls["name"] for cls in self.gt_data["categories"]]
        self.class_name_to_class_id = {
            k: v for k, v in cls_name_to_cls_id_map.items() if k in self.class_list
        }

        # Get sequences to eval and check gt files exist
        self.seq_list = [
            vid["file_names"][0].split("/")[0] for vid in self.gt_data["videos"]
        ]
        self.seq_name_to_seq_id = {
            vid["file_names"][0].split("/")[0]: vid["id"]
            for vid in self.gt_data["videos"]
        }
        self.seq_lengths = {
            vid["id"]: len(vid["file_names"]) for vid in self.gt_data["videos"]
        }

        # encode masks and compute track areas
        self._prepare_gt_annotations()

        # Get trackers to eval
        if self.config["TRACKER_JSON_OBJECT"] is not None:
            # allow directly specifying the tracker JSON data without reading from files
            tracker_json = self.config["TRACKER_JSON_OBJECT"]
            assert isinstance(tracker_json, list)
            self.tracker_list = ["tracker"]
        elif self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config["TRACKERS_TO_EVAL"] is not None) and (
            len(self.config["TRACKER_DISPLAY_NAMES"]) == len(self.tracker_list)
        ):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.config["TRACKER_DISPLAY_NAMES"])
            )
        else:
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )

        # counter for globally unique track IDs
        self.global_tid_counter = 0

        self.tracker_data = dict()
        if self.config["TRACKER_JSON_OBJECT"] is not None:
            # allow directly specifying the tracker JSON data without reading from files
            tracker = self.tracker_list[0]
            self.tracker_data[tracker] = tracker_json
        else:
            for tracker in self.tracker_list:
                tracker_dir_path = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol
                )
                tr_dir_files = [
                    file
                    for file in os.listdir(tracker_dir_path)
                    if file.endswith(".json")
                ]
                if len(tr_dir_files) != 1:
                    raise TrackEvalException(
                        tracker_dir_path + " does not contain exactly one json file."
                    )

                with open(os.path.join(tracker_dir_path, tr_dir_files[0])) as f:
                    curr_data = json.load(f)

                self.tracker_data[tracker] = curr_data

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the YouTubeVIS format
        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [classes_to_gt_tracks]: dictionary with class values as keys and list of dictionaries (with frame indices as
                                keys and corresponding segmentations as values) for each track
        [classes_to_gt_track_ids, classes_to_gt_track_areas, classes_to_gt_track_iscrowd]: dictionary with class values
                                as keys and lists (for each track) as values

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        [classes_to_dt_tracks]: dictionary with class values as keys and list of dictionaries (with frame indices as
                                keys and corresponding segmentations as values) for each track
        [classes_to_dt_track_ids, classes_to_dt_track_areas]: dictionary with class values as keys and lists as values
        [classes_to_dt_track_scores]: dictionary with class values as keys and 1D numpy arrays as values
        """
        # select sequence tracks
        seq_id = self.seq_name_to_seq_id[seq]
        if is_gt:
            tracks = [
                ann for ann in self.gt_data["annotations"] if ann["video_id"] == seq_id
            ]
        else:
            tracks = self._get_tracker_seq_tracks(tracker, seq_id)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq_id]
        data_keys = ["ids", "classes", "dets"]
        if not is_gt:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        result_key = "segmentations" if self.iou_type == "segm" else "bboxes"
        for t in range(num_timesteps):
            raw_data["dets"][t] = [
                track[result_key][t] for track in tracks if track[result_key][t]
            ]
            raw_data["ids"][t] = np.atleast_1d(
                [track["id"] for track in tracks if track[result_key][t]]
            ).astype(int)
            raw_data["classes"][t] = np.atleast_1d(
                [track["category_id"] for track in tracks if track[result_key][t]]
            ).astype(int)
            if not is_gt:
                raw_data["tracker_confidences"][t] = np.atleast_1d(
                    [track["score"] for track in tracks if track[result_key][t]]
                ).astype(float)

        if is_gt:
            key_map = {"ids": "gt_ids", "classes": "gt_classes", "dets": "gt_dets"}
        else:
            key_map = {
                "ids": "tracker_ids",
                "classes": "tracker_classes",
                "dets": "tracker_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        all_cls_ids = {self.class_name_to_class_id[cls] for cls in self.class_list}
        classes_to_tracks = {
            cls: [track for track in tracks if track["category_id"] == cls]
            for cls in all_cls_ids
        }

        # mapping from classes to track representations and track information
        raw_data["classes_to_tracks"] = {
            cls: [
                {i: track[result_key][i] for i in range(len(track[result_key]))}
                for track in tracks
            ]
            for cls, tracks in classes_to_tracks.items()
        }
        raw_data["classes_to_track_ids"] = {
            cls: [track["id"] for track in tracks]
            for cls, tracks in classes_to_tracks.items()
        }
        raw_data["classes_to_track_areas"] = {
            cls: [track["area"] for track in tracks]
            for cls, tracks in classes_to_tracks.items()
        }

        if is_gt:
            raw_data["classes_to_gt_track_iscrowd"] = {
                cls: [track["iscrowd"] for track in tracks]
                for cls, tracks in classes_to_tracks.items()
            }
        else:
            raw_data["classes_to_dt_track_scores"] = {
                cls: np.array([track["score"] for track in tracks])
                for cls, tracks in classes_to_tracks.items()
            }

        if is_gt:
            key_map = {
                "classes_to_tracks": "classes_to_gt_tracks",
                "classes_to_track_ids": "classes_to_gt_track_ids",
                "classes_to_track_areas": "classes_to_gt_track_areas",
            }
        else:
            key_map = {
                "classes_to_tracks": "classes_to_dt_tracks",
                "classes_to_track_ids": "classes_to_dt_track_ids",
                "classes_to_track_areas": "classes_to_dt_track_areas",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.
        YouTubeVIS:
            In YouTubeVIS, the 4 preproc steps are as follow:
                1) There are 40 classes which are evaluated separately.
                2) No matched tracker dets are removed.
                3) No unmatched tracker dets are removed.
                4) No gt dets are removed.
            Further, for TrackMAP computation track representations for the given class are accessed from a dictionary
            and the tracks from the tracker data are sorted according to the tracker confidence.
        """
        cls_id = self.class_name_to_class_id[cls]

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):
            # Only extract relevant dets for this class for eval (cls)
            gt_class_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_dets = [
                raw_data["gt_dets"][t][ind]
                for ind in range(len(gt_class_mask))
                if gt_class_mask[ind]
            ]

            tracker_class_mask = np.atleast_1d(raw_data["tracker_classes"][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = [
                raw_data["tracker_dets"][t][ind]
                for ind in range(len(tracker_class_mask))
                if tracker_class_mask[ind]
            ]
            similarity_scores = raw_data["similarity_scores"][t][gt_class_mask, :][
                :, tracker_class_mask
            ]

            data["tracker_ids"][t] = tracker_ids
            data["tracker_dets"][t] = tracker_dets
            data["gt_ids"][t] = gt_ids
            data["gt_dets"][t] = gt_dets
            data["similarity_scores"][t] = similarity_scores

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[
                        data["tracker_ids"][t]
                    ].astype(int)

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        # Record overview statistics.
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        # get track representations
        data["gt_tracks"] = raw_data["classes_to_gt_tracks"][cls_id]
        data["gt_track_ids"] = raw_data["classes_to_gt_track_ids"][cls_id]
        data["gt_track_areas"] = raw_data["classes_to_gt_track_areas"][cls_id]
        data["gt_track_iscrowd"] = raw_data["classes_to_gt_track_iscrowd"][cls_id]
        data["dt_tracks"] = raw_data["classes_to_dt_tracks"][cls_id]
        data["dt_track_ids"] = raw_data["classes_to_dt_track_ids"][cls_id]
        data["dt_track_areas"] = raw_data["classes_to_dt_track_areas"][cls_id]
        data["dt_track_scores"] = raw_data["classes_to_dt_track_scores"][cls_id]
        data["iou_type"] = "mask"

        # sort tracker data tracks by tracker confidence scores
        if data["dt_tracks"]:
            idx = np.argsort(
                [-score for score in data["dt_track_scores"]], kind="mergesort"
            )
            data["dt_track_scores"] = [data["dt_track_scores"][i] for i in idx]
            data["dt_tracks"] = [data["dt_tracks"][i] for i in idx]
            data["dt_track_ids"] = [data["dt_track_ids"][i] for i in idx]
            data["dt_track_areas"] = [data["dt_track_areas"][i] for i in idx]

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if self.iou_type == "segm":
            similarity_scores = self._calculate_mask_ious(
                gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False
            )
        else:
            gt_dets_t = np.array(gt_dets_t, dtype=np.float32).reshape(-1, 4)
            tracker_dets_t = np.array(tracker_dets_t, dtype=np.float32).reshape(-1, 4)
            similarity_scores = self._calculate_box_ious(
                gt_dets_t, tracker_dets_t, box_format="xywh", do_ioa=False
            )
        return similarity_scores

    def _prepare_gt_annotations(self):
        """
        Prepares GT data by rle encoding segmentations and computing the average track area.
        :return: None
        """
        if self.iou_type == "segm":
            # only loaded when needed to reduce minimum requirements
            from pycocotools import mask as mask_utils

            for track in self.gt_data["annotations"]:
                h = track["height"]
                w = track["width"]
                for i, seg in enumerate(track["segmentations"]):
                    if seg is not None and isinstance(seg["counts"], list):
                        track["segmentations"][i] = mask_utils.frPyObjects(seg, h, w)
                areas = [a for a in track["areas"] if a]
                if len(areas) == 0:
                    track["area"] = 0
                else:
                    track["area"] = np.array(areas).mean()
        else:
            for track in self.gt_data["annotations"]:
                # For bbox eval, compute areas from bboxes if not already available
                areas = [a for a in track.get("areas", []) if a]
                if not areas:
                    areas = []
                    for bbox in track.get("bboxes", []):
                        if bbox is not None:
                            areas.append(bbox[2] * bbox[3])
                track["area"] = np.array(areas).mean() if areas else 0

    def _get_tracker_seq_tracks(self, tracker, seq_id):
        """
        Prepares tracker data for a given sequence. Extracts all annotations for given sequence ID, computes
        average track area and assigns a track ID.
        :param tracker: the given tracker
        :param seq_id: the sequence ID
        :return: the extracted tracks
        """
        # only loaded when needed to reduce minimum requirements
        from pycocotools import mask as mask_utils

        tracks = [
            ann for ann in self.tracker_data[tracker] if ann["video_id"] == seq_id
        ]
        for track in tracks:
            if "areas" not in track:
                if self.iou_type == "segm":
                    for seg in track["segmentations"]:
                        if seg:
                            track["areas"].append(mask_utils.area(seg))
                        else:
                            track["areas"].append(None)
                else:
                    for bbox in track["bboxes"]:
                        if bbox:
                            track["areas"].append(bbox[2] * bbox[3])
                        else:
                            track["areas"].append(None)
            areas = [a for a in track["areas"] if a]
            if len(areas) == 0:
                track["area"] = 0
            else:
                track["area"] = np.array(areas).mean()
            track["id"] = self.global_tid_counter
            self.global_tid_counter += 1
        return tracks

    def get_name(self):
        return self.dataset_name
