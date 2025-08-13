import argparse
import json
import os

import pandas as pd
import yaml

all_keys = {
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


def load_json_and_get_keys_from_last_row(file_path, keys):
    """
    Load JSON data from a file and return specific keys from the last row.
    :param file_path: Path to the JSON file.
    :param keys: List of keys to extract from the last row.
    :return: Dictionary with the specified keys and their values from the last row.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError("JSON data is not a non-empty list.")
    last_row = data[-1]
    return {key: last_row.get(key) for key in keys}


def load_jsonl_and_get_last_row_keys(file_path, keys):
    last_row = None
    try:
        with open(file_path, "r") as file:
            for line in file:
                last_row = json.loads(line.strip())

        if last_row is None:
            print("The JSONL file is empty.")
            return None

        result = {key: last_row.get(key) for key in keys}
        return result
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Failed to parse JSON in {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def find_yaml_files_recursively(directory, filename):
    """Recursively find YAML files with a specific filename."""
    matching_files = []
    for root, _, files in os.walk(directory):
        if "/code/" in root:
            continue
        for file in files:
            if file == filename:
                matching_files.append(os.path.join(root, file))
    return matching_files


def sample_keys_from_yaml(file_path, keys):
    """Extract specific keys from a YAML file."""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        out_dict = {key: data["scratch"].get(key) for key in keys}
        out_dict["batch_size"] = int(data["launcher"]["gpus_per_node"]) * int(
            data["launcher"]["num_nodes"]
        )
        out_dict["lr_scale"] = data["scratch"].get("lr_scale", None)
        out_dict["roboflow_num_images"] = data["roboflow_train"].get("num_images", None)
        return out_dict


def average_values_of_dict(dict):
    total = 0
    count = 0
    for value in dict.values():
        total += value
        count += 1
    return total / count


def main():

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    # Add arguments
    parser.add_argument("-p", "--path", type=str, help="directory path", required=True)
    args = parser.parse_args()

    directory = args.path
    filename = "config_resolved.yaml"  # Replace with your specific filename

    keys_to_sample = [
        "lr_transformer",
        "lr_vision_backbone",
        "lr_language_backbone",
        "max_data_epochs",
    ]  # Replace with your specific keys

    yaml_files = find_yaml_files_recursively(directory, filename)

    for yaml_file in yaml_files:
        sampled_data = sample_keys_from_yaml(yaml_file, keys_to_sample)

        print("####################")
        print(f"File: {yaml_file}")
        print("Sampled Data:", sampled_data)

        # get results
        res_file = os.path.dirname(yaml_file)
        res_file = os.path.join(res_file, "logs/val_stats.json")

        extract_keys = {}
        for super_cat, cats in all_keys.items():
            extract_keys[super_cat] = [
                f"Meters_train/val_roboflow100_*/detection/roboflow100_{cat}/coco_eval_bbox_AP"
                for cat in cats
            ]

        results = {}
        results_average = {}
        all_results = []
        for super_cat, keys in extract_keys.items():
            results[super_cat] = load_jsonl_and_get_last_row_keys(res_file, keys)
            results_average[super_cat] = average_values_of_dict(results[super_cat])
            all_results.extend(results[super_cat].values())

        print("Average Results:", results_average)
        print("All category average:", average_values_of_dict(results_average))
        print("Total categories", len(all_results))
        print("True average:", sum(all_results) / len(all_results))


def print_table(data):
    """
    Prints a list of dictionaries as a table with keys as columns.
    :param data: List of dictionaries
    """
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    # Print the DataFrame as a table
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
