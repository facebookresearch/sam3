# flake8: noqa

import os
import time
import traceback
from functools import partial
from multiprocessing.pool import Pool

import numpy as np

from . import _timing, utils
from .metrics import Count
from .utils import TrackEvalException

try:
    import tqdm

    TQDM_IMPORTED = True
except ImportError as _:
    TQDM_IMPORTED = False


class Evaluator:
    """Evaluator class for evaluating different metrics for different datasets"""

    @staticmethod
    def get_default_eval_config():
        """Returns the default config values for evaluation"""
        code_path = utils.get_code_path()
        default_config = {
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 8,
            "BREAK_ON_ERROR": True,  # Raises exception and exits with error
            "RETURN_ON_ERROR": False,  # if not BREAK_ON_ERROR, then returns from function on error
            "LOG_ON_ERROR": os.path.join(
                code_path, "error_log.txt"
            ),  # if not None, save any errors into a log file.
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": True,
            "TIME_PROGRESS": True,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_EMPTY_CLASSES": True,  # If False, summary files are not output for classes with no detections
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": True,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise the evaluator with a config file"""
        self.config = utils.init_config(config, self.get_default_eval_config(), "Eval")
        # Only run timing analysis if not run in parallel.
        if self.config["TIME_PROGRESS"] and not self.config["USE_PARALLEL"]:
            _timing.DO_TIMING = True
            if self.config["DISPLAY_LESS_PROGRESS"]:
                _timing.DISPLAY_LESS_PROGRESS = True

    def _combine_results(
        self,
        res,
        metrics_list,
        metric_names,
        dataset,
        res_field="COMBINED_SEQ",
        target_tag=None,
    ):
        assert res_field.startswith("COMBINED_SEQ")
        # collecting combined cls keys (cls averaged, det averaged, super classes)
        tracker_list, seq_list, class_list = dataset.get_eval_info()
        combined_cls_keys = []
        res[res_field] = {}

        # narrow the target for evaluation
        if target_tag is not None:
            target_video_ids = [
                annot["video_id"]
                for annot in dataset.gt_data["annotations"]
                if target_tag in annot["tags"]
            ]
            vid2name = {
                video["id"]: video["file_names"][0].split("/")[0]
                for video in dataset.gt_data["videos"]
            }
            target_video_ids = set(target_video_ids)
            target_video = [vid2name[video_id] for video_id in target_video_ids]

            if len(target_video) == 0:
                raise TrackEvalException(
                    "No sequences found with the tag %s" % target_tag
                )

            target_annotations = [
                annot
                for annot in dataset.gt_data["annotations"]
                if annot["video_id"] in target_video_ids
            ]
            assert all(target_tag in annot["tags"] for annot in target_annotations), (
                f"Not all annotations in the target sequences have the target tag {target_tag}. "
                "We currently only support a target tag at the sequence level, not at the annotation level."
            )
        else:
            target_video = seq_list

        # combine sequences for each class
        for c_cls in class_list:
            res[res_field][c_cls] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                curr_res = {
                    seq_key: seq_value[c_cls][metric_name]
                    for seq_key, seq_value in res.items()
                    if not seq_key.startswith("COMBINED_SEQ")
                    and seq_key in target_video
                }
                res[res_field][c_cls][metric_name] = metric.combine_sequences(curr_res)
        # combine classes
        if dataset.should_classes_combine:
            combined_cls_keys += [
                "cls_comb_cls_av",
                "cls_comb_det_av",
                "all",
            ]
            res[res_field]["cls_comb_cls_av"] = {}
            res[res_field]["cls_comb_det_av"] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                cls_res = {
                    cls_key: cls_value[metric_name]
                    for cls_key, cls_value in res[res_field].items()
                    if cls_key not in combined_cls_keys
                }
                res[res_field]["cls_comb_cls_av"][metric_name] = (
                    metric.combine_classes_class_averaged(cls_res)
                )
                res[res_field]["cls_comb_det_av"][metric_name] = (
                    metric.combine_classes_det_averaged(cls_res)
                )
        # combine classes to super classes
        if dataset.use_super_categories:
            for cat, sub_cats in dataset.super_categories.items():
                combined_cls_keys.append(cat)
                res[res_field][cat] = {}
                for metric, metric_name in zip(metrics_list, metric_names):
                    cat_res = {
                        cls_key: cls_value[metric_name]
                        for cls_key, cls_value in res[res_field].items()
                        if cls_key in sub_cats
                    }
                    res[res_field][cat][metric_name] = (
                        metric.combine_classes_det_averaged(cat_res)
                    )
        return res, combined_cls_keys

    def _summarize_results(
        self,
        res,
        tracker,
        metrics_list,
        metric_names,
        dataset,
        res_field,
        combined_cls_keys,
    ):
        config = self.config
        output_fol = dataset.get_output_fol(tracker)
        tracker_display_name = dataset.get_display_name(tracker)
        for c_cls in res[
            res_field
        ].keys():  # class_list + combined classes if calculated
            summaries = []
            details = []
            num_dets = res[res_field][c_cls]["Count"]["Dets"]
            if config["OUTPUT_EMPTY_CLASSES"] or num_dets > 0:
                for metric, metric_name in zip(metrics_list, metric_names):
                    # for combined classes there is no per sequence evaluation
                    if c_cls in combined_cls_keys:
                        table_res = {res_field: res[res_field][c_cls][metric_name]}
                    else:
                        table_res = {
                            seq_key: seq_value[c_cls][metric_name]
                            for seq_key, seq_value in res.items()
                        }

                    if config["PRINT_RESULTS"] and config["PRINT_ONLY_COMBINED"]:
                        dont_print = (
                            dataset.should_classes_combine
                            and c_cls not in combined_cls_keys
                        )
                        if not dont_print:
                            metric.print_table(
                                {res_field: table_res[res_field]},
                                tracker_display_name,
                                c_cls,
                                res_field,
                                res_field,
                            )
                    elif config["PRINT_RESULTS"]:
                        metric.print_table(
                            table_res, tracker_display_name, c_cls, res_field, res_field
                        )
                    if config["OUTPUT_SUMMARY"]:
                        summaries.append(metric.summary_results(table_res))
                    if config["OUTPUT_DETAILED"]:
                        details.append(metric.detailed_results(table_res))
                    if config["PLOT_CURVES"]:
                        metric.plot_single_tracker_results(
                            table_res,
                            tracker_display_name,
                            c_cls,
                            output_fol,
                        )
                if config["OUTPUT_SUMMARY"]:
                    utils.write_summary_results(summaries, c_cls, output_fol)
                if config["OUTPUT_DETAILED"]:
                    utils.write_detailed_results(details, c_cls, output_fol)

    @_timing.time
    def evaluate(self, dataset_list, metrics_list, show_progressbar=False):
        """Evaluate a set of metrics on a set of datasets"""
        config = self.config
        metrics_list = metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dataset_name in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()
            print(
                "\nEvaluating %i tracker(s) on %i sequence(s) for %i class(es) on %s dataset using the following "
                "metrics: %s\n"
                % (
                    len(tracker_list),
                    len(seq_list),
                    len(class_list),
                    dataset_name,
                    ", ".join(metric_names),
                )
            )

            # Evaluate each tracker
            for tracker in tracker_list:
                # if not config['BREAK_ON_ERROR'] then go to next tracker without breaking
                try:
                    # Evaluate each sequence in parallel or in series.
                    # returns a nested dict (res), indexed like: res[seq][class][metric_name][sub_metric field]
                    # e.g. res[seq_0001][pedestrian][hota][DetA]
                    print("\nEvaluating %s\n" % tracker)
                    time_start = time.time()
                    if config["USE_PARALLEL"]:
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)

                            with Pool(config["NUM_PARALLEL_CORES"]) as pool, tqdm.tqdm(
                                total=len(seq_list)
                            ) as pbar:
                                _eval_sequence = partial(
                                    eval_sequence,
                                    dataset=dataset,
                                    tracker=tracker,
                                    class_list=class_list,
                                    metrics_list=metrics_list,
                                    metric_names=metric_names,
                                )
                                results = []
                                for r in pool.imap(
                                    _eval_sequence, seq_list_sorted, chunksize=20
                                ):
                                    results.append(r)
                                    pbar.update()
                                res = dict(zip(seq_list_sorted, results))

                        else:
                            with Pool(config["NUM_PARALLEL_CORES"]) as pool:
                                _eval_sequence = partial(
                                    eval_sequence,
                                    dataset=dataset,
                                    tracker=tracker,
                                    class_list=class_list,
                                    metrics_list=metrics_list,
                                    metric_names=metric_names,
                                )
                                results = pool.map(_eval_sequence, seq_list)
                                res = dict(zip(seq_list, results))
                    else:
                        res = {}
                        if show_progressbar and TQDM_IMPORTED:
                            seq_list_sorted = sorted(seq_list)
                            for curr_seq in tqdm.tqdm(seq_list_sorted):
                                res[curr_seq] = eval_sequence(
                                    curr_seq,
                                    dataset,
                                    tracker,
                                    class_list,
                                    metrics_list,
                                    metric_names,
                                )
                        else:
                            for curr_seq in sorted(seq_list):
                                res[curr_seq] = eval_sequence(
                                    curr_seq,
                                    dataset,
                                    tracker,
                                    class_list,
                                    metrics_list,
                                    metric_names,
                                )

                    # Combine results over all sequences and then over all classes
                    res, combined_cls_keys = self._combine_results(
                        res, metrics_list, metric_names, dataset, "COMBINED_SEQ"
                    )

                    if np.all(
                        ["tags" in annot for annot in dataset.gt_data["annotations"]]
                    ):
                        # Combine results over the challenging sequences and then over all classes
                        # currently only support "tracking_challenging_pair"
                        res, _ = self._combine_results(
                            res,
                            metrics_list,
                            metric_names,
                            dataset,
                            "COMBINED_SEQ_CHALLENGING",
                            "tracking_challenging_pair",
                        )

                    # Print and output results in various formats
                    if config["TIME_PROGRESS"]:
                        print(
                            "\nAll sequences for %s finished in %.2f seconds"
                            % (tracker, time.time() - time_start)
                        )

                    self._summarize_results(
                        res,
                        tracker,
                        metrics_list,
                        metric_names,
                        dataset,
                        "COMBINED_SEQ",
                        combined_cls_keys,
                    )
                    if "COMBINED_SEQ_CHALLENGING" in res:
                        self._summarize_results(
                            res,
                            tracker,
                            metrics_list,
                            metric_names,
                            dataset,
                            "COMBINED_SEQ_CHALLENGING",
                            combined_cls_keys,
                        )

                    # Output for returning from function
                    output_res[dataset_name][tracker] = res
                    output_msg[dataset_name][tracker] = "Success"

                except Exception as err:
                    output_res[dataset_name][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dataset_name][tracker] = str(err)
                    else:
                        output_msg[dataset_name][tracker] = "Unknown error occurred."
                    print("Tracker %s was unable to be evaluated." % tracker)
                    print(err)
                    traceback.print_exc()
                    if config["LOG_ON_ERROR"] is not None:
                        with open(config["LOG_ON_ERROR"], "a") as f:
                            print(dataset_name, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print("\n\n\n", file=f)
                    if config["BREAK_ON_ERROR"]:
                        raise err
                    elif config["RETURN_ON_ERROR"]:
                        return output_res, output_msg

        return output_res, output_msg


@_timing.time
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""

    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res
