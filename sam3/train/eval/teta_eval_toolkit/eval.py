# fmt: off
# flake8: noqa

import copy
import os
import pickle
import time
import traceback
from functools import partial
from multiprocessing.pool import Pool

import numpy as np

from . import _timing, utils
from .config import get_default_eval_config, init_config
from .utils import TrackEvalException


class Evaluator:
    """Evaluator class for evaluating different metrics for each datasets."""

    def __init__(self, config=None):
        """Initialize the evaluator with a config file."""
        self.config = init_config(config, get_default_eval_config(), "Eval")
        # Only run timing analysis if not run in parallel.
        if self.config["TIME_PROGRESS"] and not self.config["USE_PARALLEL"]:
            _timing.DO_TIMING = True
            if self.config["DISPLAY_LESS_PROGRESS"]:
                _timing.DISPLAY_LESS_PROGRESS = True

    @_timing.time
    def evaluate(self, dataset_list, metrics_list):
        """Evaluate a set of metrics on a set of datasets."""
        config = self.config
        metrics_list = metrics_list
        metric_names = utils.validate_metrics_list(metrics_list)
        dataset_names = [dataset.get_name() for dataset in dataset_list]
        output_res = {}
        output_msg = {}

        for dataset, dname in zip(dataset_list, dataset_names):
            # Get dataset info about what to evaluate
            output_res[dname] = {}
            output_msg[dname] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()
            print(
                f"\nEvaluating {len(tracker_list)} tracker(s) on "
                f"{len(seq_list)} sequence(s) for {len(class_list)} class(es)"
                f" on {dname} dataset using the following "
                f'metrics: {", ".join(metric_names)}\n'
            )

            # Evaluate each tracker
            for tracker in tracker_list:
                try:
                    output_res, output_msg = self.evaluate_tracker(
                        tracker,
                        dataset,
                        dname,
                        class_list,
                        metrics_list,
                        metric_names,
                        seq_list,
                        output_res,
                        output_msg,
                    )
                except Exception as err:
                    output_res[dname][tracker] = None
                    if type(err) == TrackEvalException:
                        output_msg[dname][tracker] = str(err)
                    else:
                        output_msg[dname][tracker] = "Unknown error occurred."
                    print("Tracker %s was unable to be evaluated." % tracker)
                    print(err)
                    traceback.print_exc()
                    if config["LOG_ON_ERROR"] is not None:
                        with open(config["LOG_ON_ERROR"], "a") as f:
                            print(dname, file=f)
                            print(tracker, file=f)
                            print(traceback.format_exc(), file=f)
                            print("\n\n\n", file=f)
                    if config["BREAK_ON_ERROR"]:
                        raise err
                    elif config["RETURN_ON_ERROR"]:
                        return output_res, output_msg

        return output_res, output_msg

    def evaluate_tracker(
        self,
        tracker,
        dataset,
        dname,
        class_list,
        metrics_list,
        metric_names,
        seq_list,
        output_res,
        output_msg,
    ):
        """Evaluate each sequence in parallel or in series."""
        print("\nEvaluating %s\n" % tracker)
        time_start = time.time()
        config = self.config
        if config["USE_PARALLEL"]:
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
            for curr_seq in sorted(seq_list):
                res[curr_seq] = eval_sequence(
                    curr_seq, dataset, tracker, class_list, metrics_list, metric_names
                )


        # collecting combined cls keys (cls averaged, det averaged, super classes)
        cls_keys = []
        res["COMBINED_SEQ"] = {}
        # combine sequences for each class
        for c_cls in class_list:
            res["COMBINED_SEQ"][c_cls] = {}
            for metric, mname in zip(metrics_list, metric_names):
                curr_res = {
                    seq_key: seq_value[c_cls][mname]
                    for seq_key, seq_value in res.items()
                    if seq_key != "COMBINED_SEQ"
                }
                # combine results over all sequences and then over all classes
                res["COMBINED_SEQ"][c_cls][mname] = metric.combine_sequences(curr_res)

        # combine classes
        if dataset.should_classes_combine:
            if config["OUTPUT_PER_SEQ_RES"]:
                video_keys = res.keys()
            else:
                video_keys = ["COMBINED_SEQ"]
            for v_key in video_keys:
                cls_keys += ["average"]
                res[v_key]["average"] = {}
                for metric, mname in zip(metrics_list, metric_names):
                    cls_res = {
                        cls_key: cls_value[mname]
                        for cls_key, cls_value in res[v_key].items()
                        if cls_key not in cls_keys
                    }
                    res[v_key]["average"][
                        mname
                    ] = metric.combine_classes_class_averaged(
                        cls_res, ignore_empty=True
                    )

        # combine classes to super classes
        if dataset.use_super_categories:
            for cat, sub_cats in dataset.super_categories.items():
                cls_keys.append(cat)
                res["COMBINED_SEQ"][cat] = {}
                for metric, mname in zip(metrics_list, metric_names):
                    cat_res = {
                        cls_key: cls_value[mname]
                        for cls_key, cls_value in res["COMBINED_SEQ"].items()
                        if cls_key in sub_cats
                    }
                    res["COMBINED_SEQ"][cat][
                        mname
                    ] = metric.combine_classes_det_averaged(cat_res)
        # Print and output results in various formats
        if config["TIME_PROGRESS"]:
            print(
                f"\nAll sequences for {tracker} finished in"
                f" {time.time() - time_start} seconds"
            )
        output_fol = dataset.get_output_fol(tracker)
        os.makedirs(output_fol, exist_ok=True)

        # take a mean of each field of each thr
        if config["OUTPUT_PER_SEQ_RES"]:
            all_res = copy.deepcopy(res)
            summary_keys = res.keys()
        else:
            all_res = copy.deepcopy(res["COMBINED_SEQ"])
            summary_keys = ["COMBINED_SEQ"]
        thr_key_list = [50]
        for s_key in summary_keys:
            for metric, mname in zip(metrics_list, metric_names):
                if mname != "TETA":
                    if s_key == "COMBINED_SEQ":
                        metric.print_table(
                            {"COMBINED_SEQ": res["COMBINED_SEQ"][cls_keys[0]][mname]},
                            tracker,
                            cls_keys[0],
                        )
                    continue

                for c_cls in res[s_key].keys():
                    for thr in thr_key_list:
                        all_res[s_key][c_cls][mname][thr] = metric._summary_row(
                            res[s_key][c_cls][mname][thr]
                        )
                    x = (
                        np.array(list(all_res[s_key][c_cls]["TETA"].values()))
                        .astype("float")
                        .mean(axis=0)
                    )
                    all_res_summary = list(x.round(decimals=2).astype("str"))
                    all_res[s_key][c_cls][mname]["ALL"] = all_res_summary
                if config["OUTPUT_SUMMARY"] and s_key == "COMBINED_SEQ":
                    for t in thr_key_list:
                        metric.print_summary_table(
                            all_res[s_key][cls_keys[0]][mname][t],
                            t,
                            tracker,
                            cls_keys[0],
                        )

        if config["OUTPUT_TEM_RAW_DATA"]:
            out_file = os.path.join(output_fol, "teta_summary_results.pth")
            pickle.dump(all_res, open(out_file, "wb"))
            print("Saved the TETA summary results.")

        # output
        output_res[dname][mname] = all_res[s_key][cls_keys[0]][mname][t]
        output_msg[dname][tracker] = "Success"

        return output_res, output_msg


@_timing.time
def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence."""
    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}

    if "TETA" in metric_names:
        thresholds = [50]
        data_all_class = dataset.get_preprocessed_seq_data(
            raw_data, "all", thresholds=thresholds
        )
        teta = metrics_list[metric_names.index("TETA")]
        assignment = teta.compute_global_assignment(data_all_class)

        # create a dict to save Cls_FP for each class in different thr.
        cls_fp = {
            key: {
                cls: np.zeros((len(np.arange(0.5, 0.99, 0.05)))) for cls in class_list
            }
            for key in thresholds
        }

    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls, assignment, thresholds)

        for metric, mname in zip(metrics_list, metric_names):
            if mname == "TETA":
                seq_res[cls][mname], cls_fp, _ = metric.eval_sequence(
                    data, cls, dataset.clsid2cls_name, cls_fp
                )
            else:
                seq_res[cls][mname] = metric.eval_sequence(data)

    if "TETA" in metric_names:
        for thr in thresholds:
            for cls in class_list:
                seq_res[cls]["TETA"][thr]["Cls_FP"] += cls_fp[thr][cls]

    return seq_res
