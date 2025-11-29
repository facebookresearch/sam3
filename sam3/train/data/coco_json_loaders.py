# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from pycocotools import mask as mask_util


# ============================================================================
# Utility Functions
# ============================================================================


def convert_boxlist_to_normalized_tensor(box_list, image_width, image_height):
    """
    Converts a list of bounding boxes to a normalized PyTorch tensor.

    Args:
        box_list (list of list or tuples): Each box is [x_min, y_min, x_max, y_max].
        image_width (int or float): Width of the image.
        image_height (int or float): Height of the image.

    Returns:
        torch.Tensor: Normalized tensor of shape (N, 4), values in [0, 1].
    """
    boxes = torch.tensor(box_list, dtype=torch.float32)
    boxes[:, [0, 2]] /= image_width  # x_min, x_max
    boxes[:, [1, 3]] /= image_height  # y_min, y_max
    boxes = boxes.clamp(0, 1)
    return boxes


def load_coco_and_group_by_image(json_path: str) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Load COCO JSON file and group annotations by image.

    Args:
        json_path (str): Path to COCO JSON file.

    Returns:
        Tuple containing:
            - List of dicts with 'image' and 'annotations' keys
            - Dict mapping category IDs to category names
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    sorted_image_ids = sorted(images.keys())

    grouped = []
    for image_id in sorted_image_ids:
        image_info = images[image_id]
        grouped.append(
            {"image": image_info, "annotations": anns_by_image.get(image_id, [])}
        )

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    return grouped, cat_id_to_name


def ann_to_rle(segm, im_info: Dict) -> Dict:
    """
    Convert annotation which can be polygons or uncompressed RLE to RLE.

    Args:
        segm: Segmentation data (polygon list or RLE dict)
        im_info (dict): Image info containing 'height' and 'width'

    Returns:
        RLE encoded segmentation
    """
    h, w = im_info["height"], im_info["width"]

    if isinstance(segm, list):
        # Polygon - merge all parts into one mask RLE code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm["counts"], list):
        # Uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # Already RLE
        rle = segm

    return rle


# ============================================================================
# COCO Training API
# ============================================================================


class COCO_FROM_JSON:
    """
    COCO training API for loading box-only annotations from JSON.
    Groups all annotations per image and creates queries per category.
    """

    def __init__(
        self,
        annotation_file,
        prompts=None,
        include_negatives=True,
        category_chunk_size=None,
    ):
        """
        Initialize the COCO training API.

        Args:
            annotation_file (str): Path to COCO JSON annotation file
            prompts: Optional custom prompts for categories
            include_negatives (bool): Whether to include negative examples (categories with no instances)
        """
        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(
            annotation_file
        )
        self._sorted_cat_ids = sorted(list(self._cat_idx_to_text.keys()))
        self.prompts = None
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size
            if category_chunk_size is not None
            else len(self._sorted_cat_ids)
        )
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        if prompts is not None:
            prompts = eval(prompts)
            self.prompts = {}
            for loc_dict in prompts:
                self.prompts[int(loc_dict["id"])] = loc_dict["name"]
            assert len(self.prompts) == len(
                self._sorted_cat_ids
            ), "Number of prompts must match number of categories"

    def getDatapointIds(self):
        """Return all datapoint indices for training."""
        return list(range(len(self._raw_data) * len(self.category_chunks)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]

        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,  # Single image per datapoint
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        annot_template = {
            "image_id": 0,
            "bbox": None,  # Normalized bbox in xywh
            "area": None,  # Unnormalized area
            "segmentation": None,  # RLE encoded
            "object_id": None,
            "is_crowd": None,
            "id": None,
        }

        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]

        # Group annotations by category
        cat_id_to_anns = defaultdict(list)
        for ann in raw_annotations:
            cat_id_to_anns[ann["category_id"]].append(ann)

        annotations_by_cat_sorted = [
            (cat_id, cat_id_to_anns[cat_id]) for cat_id in cat_chunk
        ]

        for cat_id, anns in annotations_by_cat_sorted:
            if len(anns) == 0 and not self.include_negatives:
                continue

            cur_ann_ids = []

            # Create annotations for this category
            for ann in anns:
                annotation = annot_template.copy()
                annotation["id"] = len(annotations)
                annotation["object_id"] = annotation["id"]
                annotation["is_crowd"] = ann["iscrowd"]

                normalized_boxes = convert_boxlist_to_normalized_tensor(
                    [ann["bbox"]], width, height
                )
                bbox = normalized_boxes[0]

                annotation["area"] = (bbox[2] * bbox[3]).item()
                annotation["bbox"] = bbox

                if (
                    "segmentation" in ann
                    and ann["segmentation"] is not None
                    and ann["segmentation"] != []
                ):
                    annotation["segmentation"] = ann_to_rle(
                        ann["segmentation"], im_info=image_info
                    )

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])

            # Create query for this category
            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            query["query_text"] = (
                self._cat_idx_to_text[cat_id]
                if self.prompts is None
                else self.prompts[cat_id]
            )
            query["object_ids_output"] = cur_ann_ids
            queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dict
        """
        img_idx = idx // len(self.category_chunks)
        img_data = self._raw_data[img_idx]["image"]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


# ============================================================================
# SAM3 Evaluation APIs
# ============================================================================


class SAM3_EVAL_API_FROM_JSON_NP:
    """
    SAM3 evaluation API for loading noun phrase queries from JSON.
    """

    def __init__(self, annotation_file):
        """
        Initialize the SAM3 evaluation API.

        Args:
            annotation_file (str): Path to SAM3 JSON annotation file
        """
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self._image_data = data["images"]

    def getDatapointIds(self):
        """Return all datapoint indices."""
        return list(range(len(self._image_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        cur_img_data = self._image_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        # Create query
        query = query_template.copy()
        query["id"] = len(queries)
        query["original_cat_id"] = int(cur_img_data["queried_category"])
        query["query_text"] = cur_img_data["text_input"]
        query["object_ids_output"] = []
        queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dict
        """
        img_data = self._image_data[idx]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class SAM3_VEVAL_API_FROM_JSON_NP:
    """
    SAM3 video evaluation API for loading noun phrase queries from JSON.
    """

    def __init__(self, annotation_file):
        """
        Initialize the SAM3 video evaluation API.

        Args:
            annotation_file (str): Path to SAM3 video JSON annotation file
        """
        with open(annotation_file, "r") as f:
            data = json.load(f)

        assert "video_np_pairs" in data, "Incorrect data format"

        self._video_data = data["videos"]
        self._video_id_to_np_ids = defaultdict(list)
        self._cat_id_to_np = {}

        for cat_dict in data["categories"]:
            self._cat_id_to_np[cat_dict["id"]] = cat_dict["name"]

        for video_np_dict in data["video_np_pairs"]:
            self._video_id_to_np_ids[video_np_dict["video_id"]].append(
                video_np_dict["category_id"]
            )
            assert (
                self._cat_id_to_np[video_np_dict["category_id"]]
                == video_np_dict["noun_phrase"]
            ), "Category name does not match text input"

    def getDatapointIds(self):
        """Return all datapoint indices."""
        return list(range(len(self._video_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific video datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        cur_vid_data = self._video_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        all_np_ids = self._video_id_to_np_ids[cur_vid_data["id"]]

        for np_id in all_np_ids:
            text_input = self._cat_id_to_np[np_id]

            for i, image_path in enumerate(cur_vid_data["file_names"]):
                query = query_template.copy()
                query["id"] = len(queries)
                query["original_cat_id"] = np_id
                query["query_text"] = text_input
                query["image_id"] = i
                query["query_processing_order"] = i
                query["object_ids_output"] = []
                queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific video datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dicts for all frames
        """
        video_data = self._video_data[idx]
        images = [
            {
                "id": i,
                "file_name": file_name,
                "original_img_id": video_data["id"],
                "coco_img_id": video_data["id"],
            }
            for i, file_name in enumerate(video_data["file_names"])
        ]
        return images


class SAM3_DEFECT_DIR_LOADER:
    def __init__(self, annotation_file, text_prompt=None, include_negatives=True):
        self.root = annotation_file
        self.include_negatives = include_negatives
        self.default_prompt = "defect" if text_prompt is None else text_prompt
        self._items = []
        self._build_index()

    def getDatapointIds(self):
        return list(range(len(self._items)))

    def loadImagesFromDatapoint(self, idx):
        item = self._items[idx]
        return [
            {
                "id": 0,
                "file_name": item["file_name"],
                "original_img_id": item["orig_id"],
                "coco_img_id": item["orig_id"],
            }
        ]

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        item = self._items[idx]
        w = item["width"]
        h = item["height"]
        annotations = []
        for m in item.get("masks", []):
            x0, y0, x1, y1 = m["bbox_xyxy"]
            x = x0 / w
            y = y0 / h
            bw = max(0.0, (x1 - x0) / w)
            bh = max(0.0, (y1 - y0) / h)
            ann = {
                "image_id": 0,
                "bbox": torch.tensor([x, y, bw, bh], dtype=torch.float32).tolist(),
                "area": float(bw * bh),
                "segmentation": m.get("rle", None),
                "object_id": len(annotations),
                "is_crowd": 0,
                "id": len(annotations),
            }
            annotations.append(ann)
        queries = []
        q = {
            "id": 0,
            "original_cat_id": 1,
            "object_ids_output": list(range(len(annotations))),
            "query_text": item.get("prompt", self.default_prompt),
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        if len(annotations) == 0 and not self.include_negatives:
            pass
        else:
            queries.append(q)
        return queries, annotations

    def _build_index(self):
        items = []
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for dirpath, dirnames, filenames in os.walk(self.root):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(full_path, self.root)
                    try:
                        from PIL import Image as PILImage
                        with open(full_path, "rb") as f:
                            img = PILImage.open(f).convert("RGB")
                        w, h = img.size
                    except Exception:
                        continue
                    prompt = self._infer_prompt(rel_path)
                    masks = self._find_masks_for(rel_path, w, h)
                    items.append(
                        {
                            "file_name": rel_path.replace("\\", "/"),
                            "orig_id": len(items),
                            "width": w,
                            "height": h,
                            "prompt": prompt,
                            "masks": masks,
                        }
                    )
        self._items = items

    def _infer_prompt(self, rel_path):
        low = rel_path.lower()
        if "isic" in low or "brain" in low or "mri" in low:
            return "lesion"
        if "kvasir" in low or "colon" in low or "cvc" in low:
            return "polyp"
        if "visa" in low or "mvtec" in low or "btad" in low or "dagm" in low or "dtd" in low:
            return "defect"
        return self.default_prompt

    def _find_masks_for(self, rel_path, w, h):
        masks = []
        base = os.path.splitext(os.path.basename(rel_path))[0]
        full = os.path.join(self.root, rel_path)
        candidate_dirs = []
        parts = rel_path.split(os.sep)
        if "images" in parts:
            i = parts.index("images")
            masks_dir = os.path.join(self.root, os.path.join(*parts[:i]), "masks")
            candidate_dirs.append(masks_dir)
        if "test" in parts:
            i = parts.index("test")
            gt_dir = os.path.join(self.root, os.path.join(*parts[:i]), "ground_truth")
            candidate_dirs.append(gt_dir)
        if "data" in [p.lower() for p in parts]:
            try:
                j = [p.lower() for p in parts].index("data")
                masks_dir = os.path.join(self.root, os.path.join(*parts[:j+1]), "Masks")
                candidate_dirs.append(masks_dir)
            except ValueError:
                pass
        if "train" in parts:
            i = parts.index("train")
            label_dir = os.path.join(self.root, os.path.join(*parts[:i+1]), "Label")
            candidate_dirs.append(label_dir)
        if "test" in parts:
            i = parts.index("test")
            label_dir = os.path.join(self.root, os.path.join(*parts[:i+1]), "Label")
            candidate_dirs.append(label_dir)
        if "ground_truth" in parts:
            i = parts.index("ground_truth")
            candidate_dirs.append(os.path.join(self.root, os.path.join(*parts[:i+1])))
        mask_paths = []
        for d in candidate_dirs:
            if not os.path.isdir(d):
                continue
            for dirpath, dirnames, filenames in os.walk(d):
                for fname in filenames:
                    fbase = os.path.splitext(fname)[0]
                    if base == fbase or base in fbase:
                        p = os.path.join(dirpath, fname)
                        mask_paths.append(p)
        rles = []
        for mp in mask_paths:
            try:
                from PIL import Image as PILImage
                import numpy as np
                with open(mp, "rb") as f:
                    m = PILImage.open(f)
                if m.mode != "1" and m.mode != "L":
                    m = m.convert("L")
                mask = np.array(m)
                if mask.max() > 1:
                    mask = (mask > 0).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("ascii")
                ys, xs = np.where(mask > 0)
                if xs.size == 0 or ys.size == 0:
                    continue
                x0 = int(xs.min())
                y0 = int(ys.min())
                x1 = int(xs.max())
                y1 = int(ys.max())
                bbox_xyxy = [x0, y0, x1, y1]
                rles.append({"rle": rle, "bbox_xyxy": bbox_xyxy})
            except Exception:
                continue
        return rles
