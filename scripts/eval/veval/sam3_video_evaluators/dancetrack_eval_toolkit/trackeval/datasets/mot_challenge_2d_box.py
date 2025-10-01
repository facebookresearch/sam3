# flake8: noqa

import configparser
import csv
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import _timing, utils
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset


class MotChallenge2DBox(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "data/gt/mot_challenge/"
            ),  # Location of GT data
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/mot_challenge/"
            ),  # Trackers location
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
            "CLASSES_TO_EVAL": ["pedestrian"],  # Valid: ['pedestrian']
            "BENCHMARK": "MOT17",  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            "SPLIT_TO_EVAL": "train",  # Valid: 'train', 'test', 'all'
            "INPUT_AS_ZIP": False,  # Whether tracker input files are zipped
            "PRINT_CONFIG": True,  # Whether to print current config
            "DO_PREPROC": True,  # Whether to perform preprocessing (never done for MOT15)
            "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            "SEQMAP_FOLDER": None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            "SEQMAP_FILE": "None",  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            "SEQ_INFO": None,  # If not None, directly specify sequences to eval and their number of timesteps
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",  # '{gt_folder}/{seq}/gt/gt.txt'
            "SKIP_SPLIT_FOL": False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )

        self.benchmark = self.config["BENCHMARK"]
        gt_set = self.config["BENCHMARK"] + "-" + self.config["SPLIT_TO_EVAL"]
        self.gt_set = gt_set
        if not self.config["SKIP_SPLIT_FOL"]:
            split_fol = gt_set
        else:
            split_fol = ""
        self.gt_fol = os.path.join(self.config["GT_FOLDER"], split_fol)
        self.tracker_fol = os.path.join(self.config["TRACKERS_FOLDER"], split_fol)
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config["INPUT_AS_ZIP"]
        self.do_preproc = self.config["DO_PREPROC"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        # Get classes to eval
        self.valid_classes = ["pedestrian"]
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only pedestrian class is valid."
            )
        self.class_name_to_class_id = {
            "pedestrian": 1,
            "person_on_vehicle": 2,
            "car": 3,
            "bicycle": 4,
            "motorbike": 5,
            "non_mot_vehicle": 6,
            "static_person": 7,
            "distractor": 8,
            "occluder": 9,
            "occluder_on_ground": 10,
            "occluder_full": 11,
            "reflection": 12,
            "crowd": 13,
        }
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException("No sequences are selected to be evaluated.")

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(
                    gt_folder=self.gt_fol, seq=seq
                )
                if not os.path.isfile(curr_file):
                    print("GT file not found " + curr_file)
                    raise TrackEvalException("GT file not found for sequence: " + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, "data.zip")
            if not os.path.isfile(curr_file):
                print("GT file not found " + curr_file)
                raise TrackEvalException(
                    "GT file not found: " + os.path.basename(curr_file)
                )

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

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol + ".zip"
                )
                if not os.path.isfile(curr_file):
                    print("Tracker file not found: " + curr_file)
                    raise TrackEvalException(
                        "Tracker file not found: "
                        + tracker
                        + "/"
                        + os.path.basename(curr_file)
                    )
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(
                        self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt"
                    )
                    if not os.path.isfile(curr_file):
                        print("Tracker file not found: " + curr_file)
                        raise TrackEvalException(
                            "Tracker file not found: "
                            + tracker
                            + "/"
                            + self.tracker_sub_fol
                            + "/"
                            + os.path.basename(curr_file)
                        )

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]

            # If sequence length is 'None' tries to read sequence length from .ini files.
            for seq, seq_length in seq_lengths.items():
                if seq_length is None:
                    ini_file = os.path.join(self.gt_fol, seq, "seqinfo.ini")
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException(
                            "ini file does not exist: "
                            + seq
                            + "/"
                            + os.path.basename(ini_file)
                        )
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data["Sequence"]["seqLength"])
        else:
            if self.config["SEQMAP_FILE"]:
                seqmap_file = self.config["SEQMAP_FILE"]
            else:
                if self.config["SEQMAP_FOLDER"] in [None, "None"]:
                    seqmap_file = os.path.join(
                        self.config["GT_FOLDER"], "seqmaps", self.gt_set + ".txt"
                    )
                else:
                    seqmap_file = os.path.join(
                        self.config["SEQMAP_FOLDER"], self.gt_set + ".txt"
                    )
            if not os.path.isfile(seqmap_file):
                print("no seqmap found: " + seqmap_file)
                raise TrackEvalException(
                    "no seqmap found: " + os.path.basename(seqmap_file)
                )
            with open(seqmap_file) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == "":
                        continue
                    seq = row[0]
                    seq_list.append(seq)
                    ini_file = os.path.join(self.gt_fol, seq, "seqinfo.ini")
                    if not os.path.isfile(ini_file):
                        raise TrackEvalException(
                            "ini file does not exist: "
                            + seq
                            + "/"
                            + os.path.basename(ini_file)
                        )
                    ini_data = configparser.ConfigParser()
                    ini_data.read(ini_file)
                    seq_lengths[seq] = int(ini_data["Sequence"]["seqLength"])
        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, "data.zip")
            else:
                zip_file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol + ".zip"
                )
            file = seq + ".txt"
        else:
            zip_file = None
            if is_gt:
                file = self.config["GT_LOC_FORMAT"].format(
                    gt_folder=self.gt_fol, seq=seq
                )
            else:
                file = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt"
                )

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(
            file, is_zipped=self.data_is_zipped, zip_file=zip_file
        )

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_crowd_ignore_regions", "gt_extras"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = "Ground-truth"
            else:
                text = "Tracking"
            raise TrackEvalException(
                text
                + " data contains the following invalid timesteps in seq %s: " % seq
                + ", ".join([str(x) + ", " for x in extra_time_keys])
            )

        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=float)
                except ValueError:
                    if is_gt:
                        raise TrackEvalException(
                            "Cannot convert gt data for sequence %s to float. Is data corrupted?"
                            % seq
                        )
                    else:
                        raise TrackEvalException(
                            "Cannot convert tracking data from tracker %s, sequence %s to float. Is data corrupted?"
                            % (tracker, seq)
                        )
                try:
                    raw_data["dets"][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data["ids"][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    if is_gt:
                        err = (
                            "Cannot load gt data from sequence %s, because there is not enough "
                            "columns in the data." % seq
                        )
                        raise TrackEvalException(err)
                    else:
                        err = (
                            "Cannot load tracker data from tracker %s, sequence %s, because there is not enough "
                            "columns in the data." % (tracker, seq)
                        )
                        raise TrackEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data["classes"][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data["classes"][t] = np.ones_like(raw_data["ids"][t])
                    else:
                        raise TrackEvalException(
                            "GT data is not in a valid format, there is not enough rows in seq %s, timestep %i."
                            % (seq, t)
                        )
                if is_gt:
                    gt_extras_dict = {
                        "zero_marked": np.atleast_1d(time_data[:, 6].astype(int))
                    }
                    raw_data["gt_extras"][t] = gt_extras_dict
                else:
                    raw_data["tracker_confidences"][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data["dets"][t] = np.empty((0, 4))
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {"zero_marked": np.empty(0)}
                    raw_data["gt_extras"][t] = gt_extras_dict
                else:
                    raw_data["tracker_confidences"][t] = np.empty(0)
            if is_gt:
                raw_data["gt_crowd_ignore_regions"][t] = np.empty((0, 4))

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

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        distractor_class_names = [
            "person_on_vehicle",
            "static_person",
            "distractor",
            "reflection",
        ]
        if self.benchmark == "MOT20":
            distractor_class_names.append("non_mot_vehicle")
        distractor_classes = [
            self.class_name_to_class_id[x] for x in distractor_class_names
        ]
        cls_id = self.class_name_to_class_id[cls]

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
            # Get all data
            gt_ids = raw_data["gt_ids"][t]
            gt_dets = raw_data["gt_dets"][t]
            gt_classes = raw_data["gt_classes"][t]
            gt_zero_marked = raw_data["gt_extras"][t]["zero_marked"]

            tracker_ids = raw_data["tracker_ids"][t]
            tracker_dets = raw_data["tracker_dets"][t]
            tracker_classes = raw_data["tracker_classes"][t]
            tracker_confidences = raw_data["tracker_confidences"][t]
            similarity_scores = raw_data["similarity_scores"][t]

            # Evaluation is ONLY valid for pedestrian class
            if len(tracker_classes) > 0 and np.max(tracker_classes) > 1:
                raise TrackEvalException(
                    "Evaluation is only valid for pedestrian class. Non pedestrian class (%i) found in sequence %s at "
                    "timestep %i." % (np.max(tracker_classes), raw_data["seq"], t)
                )

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            to_remove_tracker = np.array([], int)
            if (
                self.do_preproc
                and self.benchmark != "MOT15"
                and gt_ids.shape[0] > 0
                and tracker_ids.shape[0] > 0
            ):
                # Check all classes are valid:
                invalid_classes = np.setdiff1d(
                    np.unique(gt_classes), self.valid_class_numbers
                )
                if len(invalid_classes) > 0:
                    print(" ".join([str(x) for x in invalid_classes]))
                    raise (
                        TrackEvalException(
                            "Attempting to evaluate using invalid gt classes. "
                            "This warning only triggers if preprocessing is performed, "
                            "e.g. not for MOT15 or where prepropressing is explicitly disabled. "
                            "Please either check your gt data, or disable preprocessing. "
                            "The following invalid classes were found in timestep "
                            + str(t)
                            + ": "
                            + " ".join([str(x) for x in invalid_classes])
                        )
                    )

                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = (
                    matching_scores[match_rows, match_cols] > 0 + np.finfo("float").eps
                )
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(
                    gt_classes[match_rows], distractor_classes
                )
                to_remove_tracker = match_cols[is_distractor_class]

            # Apply preprocessing to remove all unwanted tracker dets.
            data["tracker_ids"][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data["tracker_confidences"][t] = np.delete(
                tracker_confidences, to_remove_tracker, axis=0
            )
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)
            if self.do_preproc and self.benchmark != "MOT15":
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & (
                    np.equal(gt_classes, cls_id)
                )
            else:
                # There are no classes for MOT15
                gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            data["gt_ids"][t] = gt_ids[gt_to_keep_mask]
            data["gt_dets"][t] = gt_dets[gt_to_keep_mask, :]
            data["similarity_scores"][t] = similarity_scores[gt_to_keep_mask]

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

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(
            gt_dets_t, tracker_dets_t, box_format="xywh"
        )
        return similarity_scores
