# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Script to extract and analyze training results from Roboflow VL100 experiments.

This script processes training logs and configuration files to extract model performance
metrics and training parameters for analysis and comparison.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


# Constants
CONFIG_FILENAME = "config_resolved.yaml"
RESULTS_FILENAME = "val_stats.json"
BBOX_AP_METRIC = "Meters_train/val_roboflow100/detection/coco_eval_bbox_AP"

# Roboflow dataset categories organized by domain
ROBOFLOW_CATEGORIES = {
    "sports": [
        "actions",
        "aerial-pool",
        "ball",
        "bibdetection",
        "football-player-detection",
        "lacrosse-object-detection",
    ],
    "other": [
        "buoy-onboarding",
        "car-logo-detection",
        "clashroyalechardetector",
        "cod-mw-warzone",
        "countingpills",
        "everdaynew",
        "flir-camera-objects",
        "halo-infinite-angel-videogame",
        "mahjong",
        "new-defects-in-wood",
        "orionproducts",
        "pill",
        "soda-bottles",
        "taco-trash-annotations-in-context",
        "the-dreidel-project",
    ],
    "aerial": [
        "aerial-airport",
        "aerial-cows",
        "aerial-sheep",
        "apoce-aerial-photographs-for-object-detection-of-construction-equipment",
        "electric-pylon-detection-in-rsi",
        "floating-waste",
        "human-detection-in-floods",
        "sssod",
        "uavdet-small",
        "wildfire-smoke",
        "zebrasatasturias",
    ],
    "medical": [
        "canalstenosis",
        "crystal-clean-brain-tumors-mri-dataset",
        "dentalai",
        "inbreast",
        "liver-disease",
        "nih-xray",
        "spinefrxnormalvindr",
        "stomata-cells",
        "train",
        "ufba-425",
        "urine-analysis1",
        "x-ray-id",
        "xray",
    ],
    "document": [
        "activity-diagrams",
        "all-elements",
        "circuit-voltages",
        "invoice-processing",
        "label-printing-defect-version-2",
        "macro-segmentation",
        "paper-parts",
        "signatures",
        "speech-bubbles-detection",
        "wine-labels",
    ],
    "industrial": [
        "-grccs",
        "13-lkc01",
        "2024-frc",
        "aircraft-turnaround-dataset",
        "asphaltdistressdetection",
        "cable-damage",
        "conveyor-t-shirts",
        "dataconvert",
        "deeppcb",
        "defect-detection",
        "fruitjes",
        "infraredimageofpowerequipment",
        "ism-band-packet-detection",
        "l10ul502",
        "needle-base-tip-min-max",
        "recode-waste",
        "screwdetectclassification",
        "smd-components",
        "truck-movement",
        "tube",
        "water-meter",
        "wheel-defect-detection",
    ],
    "flora_fauna": [
        "aquarium-combined",
        "bees",
        "deepfruits",
        "exploratorium-daphnia",
        "grapes-5",
        "grass-weeds",
        "gwhd2021",
        "into-the-vale",
        "jellyfish",
        "marine-sharks",
        "orgharvest",
        "peixos-fish",
        "penguin-finder-seg",
        "pig-detection",
        "roboflow-trained-dataset",
        "sea-cucumbers-new-tiles",
        "thermal-cheetah",
        "tomatoes-2",
        "trail-camera",
        "underwater-objects",
        "varroa-mites-detection--test-set",
        "wb-prova",
        "weeds4",
    ],
}


def load_jsonl_last_row(file_path: str, keys: List[str]) -> Optional[Dict[str, Any]]:
    """
    Load the last row from a JSONL file and extract specific keys.

    Args:
        file_path: Path to the JSONL file
        keys: List of keys to extract from the last row

    Returns:
        Dictionary with extracted key-value pairs, or None if file not found/empty
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None

    last_row = None
    try:
        with open(file_path, "r") as file:
            for line in file:
                last_row = json.loads(line.strip())

        if last_row is None:
            print(f"Warning: Empty JSONL file: {file_path}")
            return None

        return {key: last_row.get(key) for key in keys}

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to read {file_path}: {e}")
        return None


def find_config_files(directory: str, filename: str = CONFIG_FILENAME) -> List[str]:
    """
    Recursively find configuration files with a specific filename.

    Args:
        directory: Root directory to search
        filename: Target filename to search for

    Returns:
        List of full paths to matching files
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        # Skip code directories
        if "/code/" in root:
            continue
        if filename in files:
            matching_files.append(os.path.join(root, filename))
    return matching_files


def extract_config_parameters(config_path: str, keys: List[str]) -> Dict[str, Any]:
    """
    Extract specific parameters from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file
        keys: List of keys to extract from the 'scratch' section

    Returns:
        Dictionary containing extracted parameters
    """
    try:
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract parameters from scratch section
        scratch_params = {key: data["scratch"].get(key) for key in keys}

        # Add computed parameters
        launcher = data.get("launcher", {})
        scratch_params["batch_size"] = int(launcher.get("gpus_per_node", 1)) * int(
            launcher.get("num_nodes", 1)
        )
        scratch_params["lr_scale"] = data["scratch"].get("lr_scale")

        roboflow_train = data.get("roboflow_train", {})
        scratch_params["roboflow_num_images"] = roboflow_train.get("num_images")

        return scratch_params

    except Exception as e:
        print(f"Error: Failed to parse config file {config_path}: {e}")
        return {}


def calculate_average(values_dict: Dict[str, float]) -> float:
    """
    Calculate the average of values in a dictionary.

    Args:
        values_dict: Dictionary with numeric values

    Returns:
        Average of all values, or 0 if empty
    """
    if not values_dict:
        return 0.0
    return sum(values_dict.values()) / len(values_dict)


def extract_category_results(log_dir: str, categories: List[str]) -> Dict[str, float]:
    """
    Extract bbox AP results for specific categories from log files.

    Args:
        log_dir: Directory containing category log subdirectories
        categories: List of category names to extract results for

    Returns:
        Dictionary mapping category names to bbox AP scores
    """
    results = {}
    metric_keys = [BBOX_AP_METRIC]

    for category in categories:
        result_file = os.path.join(log_dir, f"logs/{category}/{RESULTS_FILENAME}")
        category_result = load_jsonl_last_row(result_file, metric_keys)

        if category_result is not None and category_result[BBOX_AP_METRIC] is not None:
            results[category] = category_result[BBOX_AP_METRIC]

    return results


def analyze_experiment_results(config_path: str) -> None:
    """
    Analyze results from a single experiment configuration.

    Args:
        config_path: Path to the experiment configuration file
    """
    print("=" * 80)
    print(f"Analyzing experiment: {config_path}")
    print("=" * 80)

    # Extract configuration parameters
    config_keys = [
        "lr_transformer",
        "lr_vision_backbone",
        "lr_language_backbone",
        "max_data_epochs",
    ]

    config_params = extract_config_parameters(config_path, config_keys)
    print("Configuration Parameters:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    print()

    # Extract results for each category
    experiment_dir = os.path.dirname(config_path)
    category_results = {}
    category_averages = {}
    all_scores = []

    for super_category, categories in ROBOFLOW_CATEGORIES.items():
        category_results[super_category] = extract_category_results(
            experiment_dir, categories
        )

        if category_results[super_category]:
            category_averages[super_category] = calculate_average(
                category_results[super_category]
            )
            all_scores.extend(category_results[super_category].values())

    # Print results summary
    print("Results by Category:")
    for super_category, avg_score in category_averages.items():
        num_categories = len(category_results[super_category])
        print(f"  {super_category}: {avg_score:.4f} (n={num_categories})")

    print(f"\nOverall Results:")
    print(f"  Weighted average: {calculate_average(category_averages):.4f}")
    print(f"  Total categories: {len(all_scores)}")
    print(f"  True average: {sum(all_scores) / len(all_scores):.4f}")
    print()


def print_results_table(results_data: List[Dict[str, Any]]) -> None:
    """
    Print results in a formatted table.

    Args:
        results_data: List of dictionaries containing results data
    """
    if not results_data:
        print("No results data to display.")
        return

    df = pd.DataFrame(results_data)
    print("\nResults Summary Table:")
    print("=" * 60)
    print(df.to_string(index=False))


def main() -> None:
    """Main function to orchestrate the results extraction and analysis."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze Roboflow VL100 training results"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Root directory path containing experiment results",
    )

    args = parser.parse_args()

    # Find all configuration files
    config_files = find_config_files(args.path, CONFIG_FILENAME)

    if not config_files:
        print(f"No configuration files found in {args.path}")
        return

    print(f"Found {len(config_files)} experiment configurations")
    print()

    # Analyze each experiment
    for config_file in config_files:
        try:
            analyze_experiment_results(config_file)
        except Exception as e:
            print(f"Error analyzing {config_file}: {e}")
            continue


if __name__ == "__main__":
    main()
