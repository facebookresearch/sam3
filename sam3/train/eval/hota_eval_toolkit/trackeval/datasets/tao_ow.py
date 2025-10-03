# flake8: noqa

import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import _timing, utils
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset


class TAO_OW(_BaseDataset):
    """Dataset class for TAO tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/tao/tao_training"
            ),  # Location of GT data
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/tao/tao_training"
            ),  # Trackers location
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
            "CLASSES_TO_EVAL": None,  # Classes to eval (if None, all classes)
            "SPLIT_TO_EVAL": "training",  # Valid: 'training', 'val'
            "PRINT_CONFIG": True,  # Whether to print current config
            "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            "MAX_DETECTIONS": 300,  # Number of maximal allowed detections per image (0 for unlimited)
            "SUBSET": "all",
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )
        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.should_classes_combine = True
        self.use_super_categories = False

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        gt_dir_files = [
            file for file in os.listdir(self.gt_fol) if file.endswith(".json")
        ]
        if len(gt_dir_files) != 1:
            raise TrackEvalException(
                self.gt_fol + " does not contain exactly one json file."
            )

        with open(os.path.join(self.gt_fol, gt_dir_files[0])) as f:
            self.gt_data = json.load(f)

        self.subset = self.config["SUBSET"]
        if self.subset != "all":
            # Split GT data into `known`, `unknown` or `distractor`
            self._split_known_unknown_distractor()
            self.gt_data = self._filter_gt_data(self.gt_data)

        # merge categories marked with a merged tag in TAO dataset
        self._merge_categories(self.gt_data["annotations"] + self.gt_data["tracks"])

        # Get sequences to eval and sequence information
        self.seq_list = [
            vid["name"].replace("/", "-") for vid in self.gt_data["videos"]
        ]
        self.seq_name_to_seq_id = {
            vid["name"].replace("/", "-"): vid["id"] for vid in self.gt_data["videos"]
        }
        # compute mappings from videos to annotation data
        self.videos_to_gt_tracks, self.videos_to_gt_images = self._compute_vid_mappings(
            self.gt_data["annotations"]
        )
        # compute sequence lengths
        self.seq_lengths = {vid["id"]: 0 for vid in self.gt_data["videos"]}
        for img in self.gt_data["images"]:
            self.seq_lengths[img["video_id"]] += 1
        self.seq_to_images_to_timestep = self._compute_image_to_timestep_mappings()
        self.seq_to_classes = {
            vid["id"]: {
                "pos_cat_ids": list(
                    {
                        track["category_id"]
                        for track in self.videos_to_gt_tracks[vid["id"]]
                    }
                ),
                "neg_cat_ids": vid["neg_category_ids"],
                "not_exhaustively_labeled_cat_ids": vid["not_exhaustive_category_ids"],
            }
            for vid in self.gt_data["videos"]
        }

        # Get classes to eval
        considered_vid_ids = [self.seq_name_to_seq_id[vid] for vid in self.seq_list]
        seen_cats = set(
            [
                cat_id
                for vid_id in considered_vid_ids
                for cat_id in self.seq_to_classes[vid_id]["pos_cat_ids"]
            ]
        )
        # only classes with ground truth are evaluated in TAO
        self.valid_classes = [
            cls["name"] for cls in self.gt_data["categories"] if cls["id"] in seen_cats
        ]
        # cls_name_to_cls_id_map = {cls['name']: cls['id'] for cls in self.gt_data['categories']}

        if self.config["CLASSES_TO_EVAL"]:
            # self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
            #                    for cls in self.config['CLASSES_TO_EVAL']]
            self.class_list = ["object"]  # class-agnostic
            if not all(self.class_list):
                raise TrackEvalException(
                    "Attempted to evaluate an invalid class. Only classes "
                    + ", ".join(self.valid_classes)
                    + " are valid (classes present in ground truth data)."
                )
        else:
            # self.class_list = [cls for cls in self.valid_classes]
            self.class_list = ["object"]  # class-agnostic
        # self.class_name_to_class_id = {k: v for k, v in cls_name_to_cls_id_map.items() if k in self.class_list}
        self.class_name_to_class_id = {"object": 1}  # class-agnostic

        # Get trackers to eval
        if self.config["TRACKERS_TO_EVAL"] is None:
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

        self.tracker_data = {tracker: dict() for tracker in self.tracker_list}

        for tracker in self.tracker_list:
            tr_dir_files = [
                file
                for file in os.listdir(
                    os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol)
                )
                if file.endswith(".json")
            ]
            if len(tr_dir_files) != 1:
                raise TrackEvalException(
                    os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol)
                    + " does not contain exactly one json file."
                )
            with open(
                os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, tr_dir_files[0]
                )
            ) as f:
                curr_data = json.load(f)

            # limit detections if MAX_DETECTIONS > 0
            if self.config["MAX_DETECTIONS"]:
                curr_data = self._limit_dets_per_image(curr_data)

            # fill missing video ids
            self._fill_video_ids_inplace(curr_data)

            # make track ids unique over whole evaluation set
            self._make_track_ids_unique(curr_data)

            # merge categories marked with a merged tag in TAO dataset
            self._merge_categories(curr_data)

            # get tracker sequence information
            curr_videos_to_tracker_tracks, curr_videos_to_tracker_images = (
                self._compute_vid_mappings(curr_data)
            )
            self.tracker_data[tracker]["vids_to_tracks"] = curr_videos_to_tracker_tracks
            self.tracker_data[tracker]["vids_to_images"] = curr_videos_to_tracker_images

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the TAO format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [classes_to_gt_tracks]: dictionary with class values as keys and list of dictionaries (with frame indices as
                                keys and corresponding segmentations as values) for each track
        [classes_to_gt_track_ids, classes_to_gt_track_areas, classes_to_gt_track_lengths]: dictionary with class values
                                as keys and lists (for each track) as values

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        [classes_to_dt_tracks]: dictionary with class values as keys and list of dictionaries (with frame indices as
                                keys and corresponding segmentations as values) for each track
        [classes_to_dt_track_ids, classes_to_dt_track_areas, classes_to_dt_track_lengths]: dictionary with class values
                                                                                           as keys and lists as values
        [classes_to_dt_track_scores]: dictionary with class values as keys and 1D numpy arrays as values
        """
        seq_id = self.seq_name_to_seq_id[seq]
        # File location
        if is_gt:
            imgs = self.videos_to_gt_images[seq_id]
        else:
            imgs = self.tracker_data[tracker]["vids_to_images"][seq_id]

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq_id]
        img_to_timestep = self.seq_to_images_to_timestep[seq_id]
        data_keys = ["ids", "classes", "dets"]
        if not is_gt:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for img in imgs:
            # some tracker data contains images without any ground truth information, these are ignored
            try:
                t = img_to_timestep[img["id"]]
            except KeyError:
                continue
            annotations = img["annotations"]
            raw_data["dets"][t] = np.atleast_2d(
                [ann["bbox"] for ann in annotations]
            ).astype(float)
            raw_data["ids"][t] = np.atleast_1d(
                [ann["track_id"] for ann in annotations]
            ).astype(int)
            raw_data["classes"][t] = np.atleast_1d([1 for _ in annotations]).astype(
                int
            )  # class-agnostic
            if not is_gt:
                raw_data["tracker_confidences"][t] = np.atleast_1d(
                    [ann["score"] for ann in annotations]
                ).astype(float)

        for t, d in enumerate(raw_data["dets"]):
            if d is None:
                raw_data["dets"][t] = np.empty((0, 4)).astype(float)
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if not is_gt:
                    raw_data["tracker_confidences"][t] = np.empty(0)

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

        # all_classes = [self.class_name_to_class_id[cls] for cls in self.class_list]
        all_classes = [1]  # class-agnostic

        if is_gt:
            classes_to_consider = all_classes
            all_tracks = self.videos_to_gt_tracks[seq_id]
        else:
            # classes_to_consider = self.seq_to_classes[seq_id]['pos_cat_ids'] \
            #                       + self.seq_to_classes[seq_id]['neg_cat_ids']
            classes_to_consider = all_classes  # class-agnostic
            all_tracks = self.tracker_data[tracker]["vids_to_tracks"][seq_id]

        # classes_to_tracks = {cls: [track for track in all_tracks if track['category_id'] == cls]
        #                      if cls in classes_to_consider else [] for cls in all_classes}
        classes_to_tracks = {
            cls: [track for track in all_tracks] if cls in classes_to_consider else []
            for cls in all_classes
        }  # class-agnostic

        # mapping from classes to track information
        raw_data["classes_to_tracks"] = {
            cls: [
                {
                    det["image_id"]: np.atleast_1d(det["bbox"])
                    for det in track["annotations"]
                }
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
        raw_data["classes_to_track_lengths"] = {
            cls: [len(track["annotations"]) for track in tracks]
            for cls, tracks in classes_to_tracks.items()
        }

        if not is_gt:
            raw_data["classes_to_dt_track_scores"] = {
                cls: np.array(
                    [
                        np.mean([float(x["score"]) for x in track["annotations"]])
                        for track in tracks
                    ]
                )
                for cls, tracks in classes_to_tracks.items()
            }

        if is_gt:
            key_map = {
                "classes_to_tracks": "classes_to_gt_tracks",
                "classes_to_track_ids": "classes_to_gt_track_ids",
                "classes_to_track_lengths": "classes_to_gt_track_lengths",
                "classes_to_track_areas": "classes_to_gt_track_areas",
            }
        else:
            key_map = {
                "classes_to_tracks": "classes_to_dt_tracks",
                "classes_to_track_ids": "classes_to_dt_track_ids",
                "classes_to_track_lengths": "classes_to_dt_track_lengths",
                "classes_to_track_areas": "classes_to_dt_track_areas",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data["num_timesteps"] = num_timesteps
        raw_data["neg_cat_ids"] = self.seq_to_classes[seq_id]["neg_cat_ids"]
        raw_data["not_exhaustively_labeled_cls"] = self.seq_to_classes[seq_id][
            "not_exhaustively_labeled_cat_ids"
        ]
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
        TAO:
            In TAO, the 4 preproc steps are as follow:
                1) All classes present in the ground truth data are evaluated separately.
                2) No matched tracker detections are removed.
                3) Unmatched tracker detections are removed if there is not ground truth data and the class does not
                    belong to the categories marked as negative for this sequence. Additionally, unmatched tracker
                    detections for classes which are marked as not exhaustively labeled are removed.
                4) No gt detections are removed.
            Further, for TrackMAP computation track representations for the given class are accessed from a dictionary
            and the tracks from the tracker data are sorted according to the tracker confidence.
        """
        cls_id = self.class_name_to_class_id[cls]
        is_not_exhaustively_labeled = cls_id in raw_data["not_exhaustively_labeled_cls"]
        is_neg_category = cls_id in raw_data["neg_cat_ids"]

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data["num_timesteps"]):
            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_dets = raw_data["gt_dets"][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data["tracker_classes"][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = raw_data["tracker_dets"][t][tracker_class_mask]
            tracker_confidences = raw_data["tracker_confidences"][t][tracker_class_mask]
            similarity_scores = raw_data["similarity_scores"][t][gt_class_mask, :][
                :, tracker_class_mask
            ]

            # Match tracker and gt dets (with hungarian algorithm).
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = (
                    matching_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
                )
                match_cols = match_cols[actually_matched_mask]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            if gt_ids.shape[0] == 0 and not is_neg_category:
                to_remove_tracker = unmatched_indices
            elif is_not_exhaustively_labeled:
                to_remove_tracker = unmatched_indices
            else:
                to_remove_tracker = np.array([], dtype=int)

            # remove all unwanted unmatched tracker detections
            data["tracker_ids"][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data["tracker_confidences"][t] = np.delete(
                tracker_confidences, to_remove_tracker, axis=0
            )
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

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
        data["gt_track_lengths"] = raw_data["classes_to_gt_track_lengths"][cls_id]
        data["gt_track_areas"] = raw_data["classes_to_gt_track_areas"][cls_id]
        data["dt_tracks"] = raw_data["classes_to_dt_tracks"][cls_id]
        data["dt_track_ids"] = raw_data["classes_to_dt_track_ids"][cls_id]
        data["dt_track_lengths"] = raw_data["classes_to_dt_track_lengths"][cls_id]
        data["dt_track_areas"] = raw_data["classes_to_dt_track_areas"][cls_id]
        data["dt_track_scores"] = raw_data["classes_to_dt_track_scores"][cls_id]
        data["not_exhaustively_labeled"] = is_not_exhaustively_labeled
        data["iou_type"] = "bbox"

        # sort tracker data tracks by tracker confidence scores
        if data["dt_tracks"]:
            idx = np.argsort(
                [-score for score in data["dt_track_scores"]], kind="mergesort"
            )
            data["dt_track_scores"] = [data["dt_track_scores"][i] for i in idx]
            data["dt_tracks"] = [data["dt_tracks"][i] for i in idx]
            data["dt_track_ids"] = [data["dt_track_ids"][i] for i in idx]
            data["dt_track_lengths"] = [data["dt_track_lengths"][i] for i in idx]
            data["dt_track_areas"] = [data["dt_track_areas"][i] for i in idx]
        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t)
        return similarity_scores

    def _merge_categories(self, annotations):
        """
        Merges categories with a merged tag. Adapted from https://github.com/TAO-Dataset
        :param annotations: the annotations in which the classes should be merged
        :return: None
        """
        merge_map = {}
        for category in self.gt_data["categories"]:
            if "merged" in category:
                for to_merge in category["merged"]:
                    merge_map[to_merge["id"]] = category["id"]

        for ann in annotations:
            ann["category_id"] = merge_map.get(ann["category_id"], ann["category_id"])

    def _compute_vid_mappings(self, annotations):
        """
        Computes mappings from Videos to corresponding tracks and images.
        :param annotations: the annotations for which the mapping should be generated
        :return: the video-to-track-mapping, the video-to-image-mapping
        """
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

            # Fill in vids_to_tracks
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

            # Fill in vids_to_imgs
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
                # Computer average area
                track["area"] = sum(x["area"] for x in track["annotations"]) / len(
                    track["annotations"]
                )

        # Ensure all videos are present
        for vid_id in vid_ids:
            if vid_id not in vids_to_tracks.keys():
                vids_to_tracks[vid_id] = []
            if vid_id not in vids_to_imgs.keys():
                vids_to_imgs[vid_id] = []

        return vids_to_tracks, vids_to_imgs

    def _compute_image_to_timestep_mappings(self):
        """
        Computes a mapping from images to the corresponding timestep in the sequence.
        :return: the image-to-timestep-mapping
        """
        images = {}
        for image in self.gt_data["images"]:
            images[image["id"]] = image

        seq_to_imgs_to_timestep = {vid["id"]: dict() for vid in self.gt_data["videos"]}
        for vid in seq_to_imgs_to_timestep:
            curr_imgs = [img["id"] for img in self.videos_to_gt_images[vid]]
            curr_imgs = sorted(curr_imgs, key=lambda x: images[x]["frame_index"])
            seq_to_imgs_to_timestep[vid] = {
                curr_imgs[i]: i for i in range(len(curr_imgs))
            }

        return seq_to_imgs_to_timestep

    def _limit_dets_per_image(self, annotations):
        """
        Limits the number of detections for each image to config['MAX_DETECTIONS']. Adapted from
        https://github.com/TAO-Dataset/
        :param annotations: the annotations in which the detections should be limited
        :return: the annotations with limited detections
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
        """
        Fills in missing video IDs inplace. Adapted from https://github.com/TAO-Dataset/
        :param annotations: the annotations for which the videos IDs should be filled inplace
        :return: None
        """
        missing_video_id = [x for x in annotations if "video_id" not in x]
        if missing_video_id:
            image_id_to_video_id = {
                x["id"]: x["video_id"] for x in self.gt_data["images"]
            }
            for x in missing_video_id:
                x["video_id"] = image_id_to_video_id[x["image_id"]]

    @staticmethod
    def _make_track_ids_unique(annotations):
        """
        Makes the track IDs unqiue over the whole annotation set. Adapted from https://github.com/TAO-Dataset/
        :param annotations: the annotation set
        :return: the number of updated IDs
        """
        track_id_videos = {}
        track_ids_to_update = set()
        max_track_id = 0
        for ann in annotations:
            t = ann["track_id"]
            if t not in track_id_videos:
                track_id_videos[t] = ann["video_id"]

            if ann["video_id"] != track_id_videos[t]:
                # Track id is assigned to multiple videos
                track_ids_to_update.add(t)
            max_track_id = max(max_track_id, t)

        if track_ids_to_update:
            print("true")
            next_id = itertools.count(max_track_id + 1)
            new_track_ids = defaultdict(lambda: next(next_id))
            for ann in annotations:
                t = ann["track_id"]
                v = ann["video_id"]
                if t in track_ids_to_update:
                    ann["track_id"] = new_track_ids[t, v]
        return len(track_ids_to_update)

    def _split_known_unknown_distractor(self):
        all_ids = set(
            [i for i in range(1, 2000)]
        )  # 2000 is larger than the max category id in TAO-OW.
        # `knowns` includes 78 TAO_category_ids that corresponds to 78 COCO classes.
        # (The other 2 COCO classes do not have corresponding classes in TAO).
        self.knowns = {
            4,
            13,
            1038,
            544,
            1057,
            34,
            35,
            36,
            41,
            45,
            58,
            60,
            579,
            1091,
            1097,
            1099,
            78,
            79,
            81,
            91,
            1115,
            1117,
            95,
            1122,
            99,
            1132,
            621,
            1135,
            625,
            118,
            1144,
            126,
            642,
            1155,
            133,
            1162,
            139,
            154,
            174,
            185,
            699,
            1215,
            714,
            717,
            1229,
            211,
            729,
            221,
            229,
            747,
            235,
            237,
            779,
            276,
            805,
            299,
            829,
            852,
            347,
            371,
            382,
            896,
            392,
            926,
            937,
            428,
            429,
            961,
            452,
            979,
            980,
            982,
            475,
            480,
            993,
            1001,
            502,
            1018,
        }
        # `distractors` is defined as in the paper "Opening up Open-World Tracking"
        self.distractors = {
            20,
            63,
            108,
            180,
            188,
            204,
            212,
            247,
            303,
            403,
            407,
            415,
            490,
            504,
            507,
            513,
            529,
            567,
            569,
            588,
            672,
            691,
            702,
            708,
            711,
            720,
            736,
            737,
            798,
            813,
            815,
            827,
            831,
            851,
            877,
            883,
            912,
            971,
            976,
            1130,
            1133,
            1134,
            1169,
            1184,
            1220,
        }
        self.unknowns = all_ids.difference(self.knowns.union(self.distractors))

    def _filter_gt_data(self, raw_gt_data):
        """
        Filter out irrelevant data in the raw_gt_data
        Args:
            raw_gt_data: directly loaded from json.

        Returns:
            filtered gt_data
        """
        valid_cat_ids = list()
        if self.subset == "known":
            valid_cat_ids = self.knowns
        elif self.subset == "distractor":
            valid_cat_ids = self.distractors
        elif self.subset == "unknown":
            valid_cat_ids = self.unknowns
        # elif self.subset == "test_only_unknowns":
        #     valid_cat_ids = test_only_unknowns
        else:
            raise Exception("The parameter `SUBSET` is incorrect")

        filtered = dict()
        filtered["videos"] = raw_gt_data["videos"]
        # filtered["videos"] = list()
        unwanted_vid = set()
        # for video in raw_gt_data["videos"]:
        #     datasrc = video["name"].split('/')[1]
        #     if datasrc in data_srcs:
        #         filtered["videos"].append(video)
        #     else:
        #         unwanted_vid.add(video["id"])

        filtered["annotations"] = list()
        for ann in raw_gt_data["annotations"]:
            if (ann["video_id"] not in unwanted_vid) and (
                ann["category_id"] in valid_cat_ids
            ):
                filtered["annotations"].append(ann)

        filtered["tracks"] = list()
        for track in raw_gt_data["tracks"]:
            if (track["video_id"] not in unwanted_vid) and (
                track["category_id"] in valid_cat_ids
            ):
                filtered["tracks"].append(track)

        filtered["images"] = list()
        for image in raw_gt_data["images"]:
            if image["video_id"] not in unwanted_vid:
                filtered["images"].append(image)

        filtered["categories"] = list()
        for cat in raw_gt_data["categories"]:
            if cat["id"] in valid_cat_ids:
                filtered["categories"].append(cat)

        filtered["info"] = raw_gt_data["info"]
        filtered["licenses"] = raw_gt_data["licenses"]

        return filtered
