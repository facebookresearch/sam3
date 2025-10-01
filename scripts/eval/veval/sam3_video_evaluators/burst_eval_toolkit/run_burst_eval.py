# flake8: noqa

import json
import os.path as osp
import time
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from tabulate import tabulate
from tqdm import tqdm

from .trackeval.datasets.burst import BURST
from .trackeval.datasets.burst_ow import BURST_OW
from .trackeval.metrics import Count, HOTA, TrackMAP


def eval_sequence(seq, dataset, class_list, metrics_list, metric_names):
    """Function for evaluating a single sequence"""
    try:
        raw_data = dataset.get_raw_seq_data(None, seq)
        seq_res = {}
        for cls_name in class_list:
            seq_res[cls_name] = {}
            data = dataset.get_preprocessed_seq_data(raw_data, cls_name)
            for metric, met_name in zip(metrics_list, metric_names):
                seq_res[cls_name][met_name] = metric.eval_sequence(data)

    except Exception as exc:
        print(f"ERROR occurred while processing the following sequence: '{seq}'")
        raise exc

    return seq_res


def print_results(dataset, res):
    class_name_to_id = {x["name"]: x["id"] for x in dataset.gt_data["categories"]}
    known_list = [
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
    ]

    row_labels = ("HOTA", "DetA", "AssA", "AP")
    res = res["COMBINED_SEQ"]

    def average_metric(m):
        return 100 * sum(m) / len(m)

    all_names = [
        x for x in res.keys() if (x != "cls_comb_cls_av") and (x != "cls_comb_det_av")
    ]

    class_split_names = {
        "All": [
            x
            for x in res.keys()
            if (x != "cls_comb_cls_av") and (x != "cls_comb_det_av")
        ],
        "Common": [x for x in all_names if class_name_to_id[x] in known_list],
        "Uncommon": [x for x in all_names if class_name_to_id[x] not in known_list],
    }

    # table columns: 'all', 'common', 'uncommon'
    # table rows: HOTA, AssA, DetA, mAP
    table_data = []
    output_dict = defaultdict(dict)

    for row_label in row_labels:
        row = [row_label]
        for split_name in ["All", "Common", "Uncommon"]:
            split_classes = class_split_names[split_name]

            if row_label == "AP":
                m = average_metric(
                    [res[c]["TrackMAP"]["AP_all"].mean() for c in split_classes]
                )
                row.append(m)
                output_dict[row_label][split_name.lower()] = m
            else:
                m = average_metric(
                    [res[c]["HOTA"][row_label].mean() for c in split_classes]
                )
                row.append(m)
                output_dict[row_label][split_name.lower()] = m

        table_data.append(row)

    print(tabulate(table_data, ["Metric", "All", "Common", "Uncommon"], floatfmt=".2f"))
    return output_dict


def main(args):
    dataset_config = BURST.get_default_dataset_config()

    print("Reading files...")
    gt_file_path = osp.join(args.gt, "all_classes.json")
    assert osp.exists(
        gt_file_path
    ), f"GT file not found at expected location: {gt_file_path}"
    dataset = BURST(pred_path=args.pred, gt_path=gt_file_path, config=dataset_config)
    metrics_list = [HOTA(), TrackMAP(), Count()]
    metric_names = ["HOTA", "TrackMAP", "Count"]

    dataset.config["EXEMPLAR_GUIDED"] = args.task == "exemplar_guided"

    time_start = time.time()
    _, seq_list, class_list = dataset.get_eval_info()
    seq_list_sorted = sorted(seq_list)

    print("Running evaluation...")
    if args.nprocs > 1:
        with Pool(args.nprocs) as pool, tqdm(total=len(seq_list)) as pbar:
            _eval_sequence = partial(
                eval_sequence,
                dataset=dataset,
                class_list=class_list,
                metrics_list=metrics_list,
                metric_names=metric_names,
            )

            results = []
            for r in pool.imap(_eval_sequence, seq_list_sorted, chunksize=20):
                results.append(r)
                pbar.update()

            res = dict(zip(seq_list_sorted, results))

    else:
        res = {}
        seq_list_sorted = sorted(seq_list)
        for curr_seq in tqdm(seq_list_sorted):
            res[curr_seq] = eval_sequence(
                curr_seq, dataset, class_list, metrics_list, metric_names
            )

    # Combine results over all sequences and then over all classes

    # collecting combined cls keys (cls averaged, det averaged, super classes)
    combined_cls_keys = []
    res["COMBINED_SEQ"] = {}

    # combine sequences for each class
    for c_cls in class_list:
        res["COMBINED_SEQ"][c_cls] = {}
        for metric, metric_name in zip(metrics_list, metric_names):
            curr_res = {
                seq_key: seq_value[c_cls][metric_name]
                for seq_key, seq_value in res.items()
                if seq_key != "COMBINED_SEQ"
            }
            res["COMBINED_SEQ"][c_cls][metric_name] = metric.combine_sequences(curr_res)

    # combine classes
    if dataset.should_classes_combine:
        combined_cls_keys += ["cls_comb_cls_av", "cls_comb_det_av", "all"]
        res["COMBINED_SEQ"]["cls_comb_cls_av"] = {}
        res["COMBINED_SEQ"]["cls_comb_det_av"] = {}
        for metric, metric_name in zip(metrics_list, metric_names):
            cls_res = {
                cls_key: cls_value[metric_name]
                for cls_key, cls_value in res["COMBINED_SEQ"].items()
                if cls_key not in combined_cls_keys
            }
            res["COMBINED_SEQ"]["cls_comb_cls_av"][metric_name] = (
                metric.combine_classes_class_averaged(cls_res)
            )
            res["COMBINED_SEQ"]["cls_comb_det_av"][metric_name] = (
                metric.combine_classes_det_averaged(cls_res)
            )

    # combine classes to super classes
    if dataset.use_super_categories:
        for cat, sub_cats in dataset.super_categories.items():
            combined_cls_keys.append(cat)
            res["COMBINED_SEQ"][cat] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                cat_res = {
                    cls_key: cls_value[metric_name]
                    for cls_key, cls_value in res["COMBINED_SEQ"].items()
                    if cls_key in sub_cats
                }
                res["COMBINED_SEQ"][cat][metric_name] = (
                    metric.combine_classes_det_averaged(cat_res)
                )

    # Print and output results in various formats
    print(f"Eval took {(time.time() - time_start):.3f} seconds\n")

    out_dict = print_results(dataset, res)
    return out_dict


def print_results_ow(res_all, res_common, res_uncommon):
    row_labels = ("OWTA", "DetRe", "AssA")
    res_all = res_all["COMBINED_SEQ"]["cls_comb_cls_av"]
    res_common = res_common["COMBINED_SEQ"]["cls_comb_cls_av"]
    res_uncommon = res_uncommon["COMBINED_SEQ"]["cls_comb_cls_av"]

    # table columns: 'all', 'common', 'uncommon'
    # table rows: HOTA, AssA, DetRe
    table_data = []

    def average_metric(m):
        return m.mean().item() * 100.0

    output_dict = defaultdict(dict)

    for row_label in row_labels:
        row = [row_label]

        m_all = average_metric(res_all["HOTA"][row_label])
        row.append(m_all)
        output_dict[row_label]["all"] = m_all

        m_common = average_metric(res_common["HOTA"][row_label])
        row.append(m_common)
        output_dict[row_label]["common"] = m_common

        m_unc = average_metric(res_uncommon["HOTA"][row_label])
        row.append(m_unc)
        output_dict[row_label]["uncommon"] = m_unc

        table_data.append(row)

    print(tabulate(table_data, ["Metric", "All", "Common", "Uncommon"], floatfmt=".2f"))
    return output_dict


def main_ow(args):
    dataset_config = BURST_OW.get_default_dataset_config()
    time_start = time.time()

    def process_split(split_file_name):
        gt_file_path = osp.join(args.gt, split_file_name)
        assert osp.exists(
            gt_file_path
        ), f"GT file not found at expected location: {gt_file_path}"
        dataset = BURST_OW(
            pred_path=args.pred, gt_path=gt_file_path, config=dataset_config
        )
        metrics_list = [HOTA(), TrackMAP(), Count()]
        metric_names = ["HOTA", "Count"]

        _, seq_list, class_list = dataset.get_eval_info()
        seq_list_sorted = sorted(seq_list)

        if args.nprocs > 1:
            with Pool(args.nprocs) as pool, tqdm(total=len(seq_list)) as pbar:
                _eval_sequence = partial(
                    eval_sequence,
                    dataset=dataset,
                    class_list=class_list,
                    metrics_list=metrics_list,
                    metric_names=metric_names,
                )

                results = []
                for r in pool.imap(_eval_sequence, seq_list_sorted, chunksize=20):
                    results.append(r)
                    pbar.update()

                res = dict(zip(seq_list_sorted, results))

        else:
            res = {}
            seq_list_sorted = sorted(seq_list)
            for curr_seq in tqdm(seq_list_sorted):
                res[curr_seq] = eval_sequence(
                    curr_seq, dataset, class_list, metrics_list, metric_names
                )

        # Combine results over all sequences and then over all classes

        # collecting combined cls keys (cls averaged, det averaged, super classes)
        combined_cls_keys = []
        res["COMBINED_SEQ"] = {}

        # combine sequences for each class
        for c_cls in class_list:
            res["COMBINED_SEQ"][c_cls] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                curr_res = {
                    seq_key: seq_value[c_cls][metric_name]
                    for seq_key, seq_value in res.items()
                    if seq_key != "COMBINED_SEQ"
                }
                res["COMBINED_SEQ"][c_cls][metric_name] = metric.combine_sequences(
                    curr_res
                )

        # combine classes
        if dataset.should_classes_combine:
            combined_cls_keys += ["cls_comb_cls_av", "cls_comb_det_av", "all"]
            res["COMBINED_SEQ"]["cls_comb_cls_av"] = {}
            res["COMBINED_SEQ"]["cls_comb_det_av"] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                cls_res = {
                    cls_key: cls_value[metric_name]
                    for cls_key, cls_value in res["COMBINED_SEQ"].items()
                    if cls_key not in combined_cls_keys
                }
                res["COMBINED_SEQ"]["cls_comb_cls_av"][metric_name] = (
                    metric.combine_classes_class_averaged(cls_res)
                )
                res["COMBINED_SEQ"]["cls_comb_det_av"][metric_name] = (
                    metric.combine_classes_det_averaged(cls_res)
                )

        # combine classes to super classes
        if dataset.use_super_categories:
            for cat, sub_cats in dataset.super_categories.items():
                combined_cls_keys.append(cat)
                res["COMBINED_SEQ"][cat] = {}
                for metric, metric_name in zip(metrics_list, metric_names):
                    cat_res = {
                        cls_key: cls_value[metric_name]
                        for cls_key, cls_value in res["COMBINED_SEQ"].items()
                        if cls_key in sub_cats
                    }
                    res["COMBINED_SEQ"][cat][metric_name] = (
                        metric.combine_classes_det_averaged(cat_res)
                    )

        return res

    print(
        "[WARN] Do not overwrite the prediction file until this eval script has finished executing"
    )
    print("(1/3) Running eval for all classes...")
    res_all = process_split("all_classes.json")

    print("(2/3) Running eval for common classes...")
    res_common = process_split("common_classes.json")

    print("(3/3) Running eval for uncommon classes...")
    res_uncommon = process_split("uncommon_classes.json")

    # Print and output results in various formats
    print(f"Eval took {(time.time() - time_start):.3f} seconds\n")

    out_dict = print_results_ow(res_all, res_common, res_uncommon)
    return out_dict


def run_burst_eval(args):
    parser = ArgumentParser(add_help=False)

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--pred", required=True, help="Path to predicted results JSON file"
    )
    required.add_argument(
        "--gt",
        required=True,
        help="Path to directory containing ground-truth JSON files e.g. /path/to/burst/annotations/val",
    )
    required.add_argument(
        "--task",
        required=True,
        choices=("class_guided", "exemplar_guided", "open_world"),
        help="The task to evaluate",
    )

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--nprocs",
        type=int,
        default=8,
        help="Number of parallel processes to use. Setting this to <=1 disables parallelization",
    )

    optional.add_argument(
        "--output",
        "-o",
        required=False,
        help="Optional file path. The metrics will be dumped to this location as a JSON dict",
    )

    optional.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )

    parsed_args = parser.parse_args(args)
    if parsed_args.task == "open_world":
        return main_ow(parsed_args)
    else:
        return main(parsed_args)


if __name__ == "__main__":
    import sys

    run_burst_eval(sys.argv[1:])
