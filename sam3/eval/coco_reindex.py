# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Self-contained COCO JSON re-indexing function that creates temporary files.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def reindex_coco_to_temp(input_json_path: str) -> Optional[str]:
    """
    Convert 0-indexed COCO JSON file to 1-indexed and save to temporary location.

    Args:
        input_json_path: Path to the input COCO JSON file

    Returns:
        Path to the new 1-indexed JSON file in temporary directory, or None if no conversion needed

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input file is not valid JSON
        ValueError: If input file is not a valid COCO format
    """

    def is_coco_json(data: Dict[str, Any]) -> bool:
        """Check if data appears to be a COCO format file."""
        if not isinstance(data, dict):
            return False
        # A COCO file should have at least one of these keys
        coco_keys = {"images", "annotations", "categories"}
        return any(key in data for key in coco_keys)

    def check_zero_indexed(data: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """
        Check if annotations, images, or categories start from index 0.

        Returns:
            Tuple of (annotations_zero_indexed, images_zero_indexed, categories_zero_indexed)
        """
        annotations_zero = False
        images_zero = False
        categories_zero = False

        # Check annotations
        annotations = data.get("annotations", [])
        if annotations and any(ann.get("id", -1) == 0 for ann in annotations):
            annotations_zero = True

        # Check images
        images = data.get("images", [])
        if images and any(img.get("id", -1) == 0 for img in images):
            images_zero = True

        # Check categories
        categories = data.get("categories", [])
        if categories and any(cat.get("id", -1) == 0 for cat in categories):
            categories_zero = True

        return annotations_zero, images_zero, categories_zero

    def reindex_coco_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert 0-indexed COCO data to 1-indexed."""
        modified_data = data.copy()

        annotations_zero, images_zero, categories_zero = check_zero_indexed(data)

        # Create ID mapping for consistency
        image_id_mapping = {}
        category_id_mapping = {}

        # Process images first (since annotations reference image IDs)
        if images_zero and "images" in modified_data:
            for img in modified_data["images"]:
                old_id = img["id"]
                new_id = old_id + 1
                image_id_mapping[old_id] = new_id
                img["id"] = new_id

        # Process categories (since annotations reference category IDs)
        if categories_zero and "categories" in modified_data:
            for cat in modified_data["categories"]:
                old_id = cat["id"]
                new_id = old_id + 1
                category_id_mapping[old_id] = new_id
                cat["id"] = new_id

        # Process annotations
        if "annotations" in modified_data:
            for ann in modified_data["annotations"]:
                # Update annotation ID if needed
                if annotations_zero:
                    ann["id"] = ann["id"] + 1

                # Update image_id reference if images were reindexed
                if images_zero and ann.get("image_id") is not None:
                    old_image_id = ann["image_id"]
                    if old_image_id in image_id_mapping:
                        ann["image_id"] = image_id_mapping[old_image_id]

                # Update category_id reference if categories were reindexed
                if categories_zero and ann.get("category_id") is not None:
                    old_category_id = ann["category_id"]
                    if old_category_id in category_id_mapping:
                        ann["category_id"] = category_id_mapping[old_category_id]

        return modified_data

    # Validate input path
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file not found: {input_json_path}")

    # Load and validate JSON data
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {input_json_path}: {e}")

    # Validate COCO format
    if not is_coco_json(data):
        raise ValueError(
            f"File does not appear to be in COCO format: {input_json_path}"
        )

    # Check if reindexing is needed
    annotations_zero, images_zero, categories_zero = check_zero_indexed(data)

    if not (annotations_zero or images_zero or categories_zero):
        # No conversion needed - just copy to temp location
        input_path = Path(input_json_path)
        temp_dir = tempfile.mkdtemp()
        temp_filename = f"{input_path.stem}_1_indexed{input_path.suffix}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return temp_path

    # Perform reindexing
    modified_data = reindex_coco_data(data)

    # Create temporary file
    input_path = Path(input_json_path)
    temp_dir = tempfile.mkdtemp()
    temp_filename = f"{input_path.stem}_1_indexed{input_path.suffix}"
    temp_path = os.path.join(temp_dir, temp_filename)

    # Write modified data to temporary file
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)

    return temp_path


# Example usage and test function
def test_reindex_function():
    """Test the reindex function with a sample COCO file."""

    # Create a test COCO file
    test_data = {
        "info": {"description": "Test COCO dataset", "version": "1.0", "year": 2023},
        "images": [
            {"id": 0, "width": 640, "height": 480, "file_name": "test1.jpg"},
            {"id": 1, "width": 640, "height": 480, "file_name": "test2.jpg"},
        ],
        "categories": [
            {"id": 0, "name": "person", "supercategory": "person"},
            {"id": 1, "name": "car", "supercategory": "vehicle"},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "bbox": [100, 100, 50, 75],
                "area": 3750,
                "iscrowd": 0,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [200, 150, 120, 80],
                "area": 9600,
                "iscrowd": 0,
            },
        ],
    }

    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f, indent=2)
        test_file_path = f.name

    try:
        # Test the function
        result_path = reindex_coco_to_temp(test_file_path)
        print(f"Original file: {test_file_path}")
        print(f"Converted file: {result_path}")

        # Load and display the result
        with open(result_path, "r") as f:
            result_data = json.load(f)

        print("\nConverted data sample:")
        print(f"First image ID: {result_data['images'][0]['id']}")
        print(f"First category ID: {result_data['categories'][0]['id']}")
        print(f"First annotation ID: {result_data['annotations'][0]['id']}")
        print(f"First annotation image_id: {result_data['annotations'][0]['image_id']}")
        print(
            f"First annotation category_id: {result_data['annotations'][0]['category_id']}"
        )

        # Clean up
        os.unlink(result_path)
        os.rmdir(os.path.dirname(result_path))

    finally:
        # Clean up test file
        os.unlink(test_file_path)


if __name__ == "__main__":
    test_reindex_function()
