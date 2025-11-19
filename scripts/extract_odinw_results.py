# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""This script summarizes odinw results"""

"""
python3 scripts/extract_odinw_results.py --res_dir /path/to/results/directory
Expected directory structure:
results_directory/
├── AerialMaritimeDrone_large/val_stats.json
├── Aquarium/val_stats.json
├── CottontailRabbits/val_stats.json
└── ...
"""
import argparse
import json
import os

VAL13_SET = [
    "AerialMaritimeDrone_large",
    "Aquarium",
    "CottontailRabbits",
    "EgoHands_generic",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "Raccoon",
    "ShellfishOpenImages",
    "VehiclesOpenImages",
    "pistols",
    "pothole",
    "thermalDogsAndPeople",
]

METRIC_NAME = "coco_eval_bbox_AP"


def parse_args():
    parser = argparse.ArgumentParser("ODinW results aggregation script")

    parser.add_argument(
        "--res_dir",
        required=True,
        type=str,
        help="Parent directory containing subdirectories for each dataset with val_stats.json files",
    )

    return parser.parse_args()


def main(args):
    # Dictionary to store results for each metric type
    metric_results = {METRIC_NAME: []}
    subset_results = {subset: {} for subset in VAL13_SET}

    # Process each subset directory
    for subset in VAL13_SET:
        subset_dir = os.path.join(args.res_dir, subset)
        val_stats_path = os.path.join(subset_dir, "val_stats.json")

        if not os.path.exists(val_stats_path):
            print(f"Warning: {val_stats_path} not found, skipping {subset}")
            continue

        try:
            res = json.load(open(val_stats_path))
            subset_results[subset] = res

            # Extract metrics for this subset and group by metric type
            for key, value in res.items():
                if key.endswith(METRIC_NAME):
                    metric_results[METRIC_NAME].append(value)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {val_stats_path}: {e}")
            continue

    # Print results
    values = metric_results[METRIC_NAME]
    if values:
        avg = sum(values) / len(values)
        print(f"Average {METRIC_NAME}: {avg:.4f} ({len(values)} datasets)")

        # Show individual dataset results
        for subset in VAL13_SET:
            if subset in subset_results and subset_results[subset]:
                for res_key, res_value in subset_results[subset].items():
                    if res_key.endswith(METRIC_NAME):
                        print(f"  {subset}: {res_value:.4f}")
                        break
    else:
        print(f"No results found for {METRIC_NAME}")


if __name__ == "__main__":
    main(parse_args())
