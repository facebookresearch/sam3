from pathlib import Path
import json
import shutil
import math
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TAKE_ID = "00a6dd13-d5b0-4743-b252-ed61e61f1d49"

DATA_ROOT = Path("/home/users/ntu/gwang016/scratch/datasets/Ego-Exo4D-Relation-Test")

TAKE_ROOT = Path(
    "/home/users/ntu/gwang016/scratch/datasets/Ego-Exo4D-Relation-Test/"
    "extracted/work/yuqian_fu/Ego/data_segswap_test/"
    "00a6dd13-d5b0-4743-b252-ed61e61f1d49"
)

SOURCE_VIEW_NAME = "aria06_214-1"
TARGET_VIEW_NAME = "cam01"

SOURCE_VIEW_DIR = TAKE_ROOT / SOURCE_VIEW_NAME
TARGET_VIEW_DIR = TAKE_ROOT / TARGET_VIEW_NAME

JSON_CANDIDATES = [
    DATA_ROOT / "ego2exo_test.json",
    DATA_ROOT / "exo2ego_test.json",
]

OUT_DIR = Path("/home/users/ntu/gwang016/scratch/cross_view_prompt_assets_00a6dd13")

CONTACT_SHEET_COLS = 4
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_SIZE = CONTACT_SHEET_COLS * CONTACT_SHEET_ROWS

# 如果 cam01 图片太多，可以改成 2/3/5。1 表示所有图片都用。
FRAME_STRIDE = 1

THUMB_WIDTH = 320
THUMB_HEIGHT = 240

OVERLAY_ALPHA = 120


def log(msg: str):
    print(msg, flush=True)


def numeric_stem(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return None


def list_image_files(folder: Path):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(folder.glob(ext))

    valid = []
    for p in files:
        sid = numeric_stem(p)
        if sid is not None:
            valid.append(p)

    valid = sorted(valid, key=lambda x: numeric_stem(x))
    return valid


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def path_contains_source_view_and_frame(s: str, frame_id: int):
    s = str(s)
    return TAKE_ID in s and SOURCE_VIEW_NAME in s and str(frame_id) in Path(s).stem


def path_contains_source_view(s: str):
    s = str(s)
    return TAKE_ID in s and SOURCE_VIEW_NAME in s


def item_paths_as_strings(item):
    paths = []

    def rec(x):
        if isinstance(x, str):
            paths.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(item)
    return paths


def find_matching_json_item(source_frame_id: int):
    """
    优先找：
    1. TAKE_ID + SOURCE_VIEW_NAME + source_frame_id 精确匹配
    2. TAKE_ID + SOURCE_VIEW_NAME 匹配
    3. TAKE_ID 匹配
    """
    exact_matches = []
    view_matches = []
    take_matches = []

    for jp in JSON_CANDIDATES:
        if not jp.exists():
            warnings.warn(f"JSON not found: {jp}")
            continue

        data = load_json(jp)

        for key, item in data.items():
            paths = item_paths_as_strings(item)
            joined = "\n".join(paths + [str(key)])

            if TAKE_ID not in joined:
                continue

            if any(path_contains_source_view_and_frame(p, source_frame_id) for p in paths):
                exact_matches.append((jp, key, item))
            elif any(path_contains_source_view(p) for p in paths):
                view_matches.append((jp, key, item))
            else:
                take_matches.append((jp, key, item))

    if exact_matches:
        return exact_matches[0], "exact_take_view_frame_match"

    if view_matches:
        return view_matches[0], "fallback_take_view_match"

    if take_matches:
        return take_matches[0], "fallback_take_only_match"

    return None, "no_match"


def get_target_prompt(item):
    candidates = []

    prompt = item.get("prompt", {})
    if isinstance(prompt, dict):
        candidates.append(prompt.get("text"))
        candidates.append(prompt.get("target_prompt"))
        candidates.append(prompt.get("object_text"))
        candidates.append(prompt.get("query"))

        anns = prompt.get("first_frame_anns", {})
        if isinstance(anns, dict):
            for ann in anns.values():
                if isinstance(ann, dict):
                    candidates.append(ann.get("text"))
                    candidates.append(ann.get("label"))
                    candidates.append(ann.get("category"))
                    candidates.append(ann.get("name"))

    objects = item.get("objects", {})
    if isinstance(objects, dict):
        for obj in objects.values():
            if isinstance(obj, dict):
                candidates.append(obj.get("text"))
                candidates.append(obj.get("label"))
                candidates.append(obj.get("category"))
                candidates.append(obj.get("name"))
    elif isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                candidates.append(obj.get("text"))
                candidates.append(obj.get("label"))
                candidates.append(obj.get("category"))
                candidates.append(obj.get("name"))

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()

    return "unknown target object"


def decode_rle(segmentation):
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        warnings.warn("pycocotools not installed. Cannot decode COCO RLE segmentation.")
        return None

    try:
        mask = mask_utils.decode(segmentation)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        return mask.astype(bool)
    except Exception as e:
        warnings.warn(f"Failed to decode RLE: {e}")
        return None


def polygon_to_mask(polygons, width, height):
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if isinstance(polygons, list):
        # COCO polygon may be [[x1,y1,x2,y2,...], ...] or [x1,y1,...]
        if len(polygons) > 0 and all(isinstance(x, (int, float)) for x in polygons):
            pts = [(polygons[i], polygons[i + 1]) for i in range(0, len(polygons) - 1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        else:
            for poly in polygons:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=255)

    return np.array(mask_img) > 0


def bbox_to_mask(bbox, width, height):
    mask = np.zeros((height, width), dtype=bool)

    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    x, y, w, h = bbox[:4]

    # 默认按照 COCO [x, y, width, height]
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
        return mask

    return None


def extract_candidate_annotations(item):
    anns = []

    prompt = item.get("prompt", {})
    if isinstance(prompt, dict):
        first_frame_anns = prompt.get("first_frame_anns")
        if isinstance(first_frame_anns, dict):
            for ann_id, ann in first_frame_anns.items():
                if isinstance(ann, dict):
                    ann = dict(ann)
                    ann["_source"] = f"prompt.first_frame_anns.{ann_id}"
                    anns.append(ann)

    objects = item.get("objects")
    if isinstance(objects, dict):
        for obj_id, obj in objects.items():
            if isinstance(obj, dict):
                obj = dict(obj)
                obj["_source"] = f"objects.{obj_id}"
                anns.append(obj)
    elif isinstance(objects, list):
        for i, obj in enumerate(objects):
            if isinstance(obj, dict):
                obj = dict(obj)
                obj["_source"] = f"objects[{i}]"
                anns.append(obj)

    return anns


def mask_from_annotation(ann, width, height):
    # 1. direct binary mask
    for key in ["mask", "masks"]:
        if key in ann:
            raw = ann[key]
            try:
                arr = np.array(raw)
                if arr.ndim == 2:
                    return arr.astype(bool), key
            except Exception:
                pass

    # 2. segmentation
    seg = ann.get("segmentation")
    if seg is not None:
        if isinstance(seg, dict):
            # COCO RLE
            mask = decode_rle(seg)
            if mask is not None:
                return resize_mask_if_needed(mask, width, height), "segmentation_rle"
        elif isinstance(seg, list):
            mask = polygon_to_mask(seg, width, height)
            if mask is not None and mask.sum() > 0:
                return mask, "segmentation_polygon"

    # 3. bbox fallback
    for key in ["bbox", "box", "bounding_box"]:
        if key in ann:
            mask = bbox_to_mask(ann[key], width, height)
            if mask is not None:
                return mask, f"{key}_fallback"

    return None, None


def resize_mask_if_needed(mask, width, height):
    if mask.shape == (height, width):
        return mask.astype(bool)

    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((width, height), Image.NEAREST)
    return np.array(img) > 0


def choose_mask_from_item(item, width, height):
    anns = extract_candidate_annotations(item)

    tried = []
    for ann in anns:
        mask, source = mask_from_annotation(ann, width, height)
        tried.append({
            "ann_source": ann.get("_source"),
            "keys": list(ann.keys()),
            "mask_source": source,
            "mask_found": mask is not None,
            "mask_area": int(mask.sum()) if mask is not None else 0,
            "text": ann.get("text") or ann.get("label") or ann.get("category") or ann.get("name"),
        })

        if mask is not None and mask.sum() > 0:
            return mask, source, ann, tried

    return None, None, None, tried


def get_font(size=20):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_text_with_bg(draw, xy, text, font, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except Exception:
        w, h = draw.textsize(text, font=font)
        bbox = (x, y, x + w, y + h)

    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg,
    )
    draw.text((x, y), text, font=font, fill=fill)


def save_mask_and_overlay(source_img_path, mask, target_prompt, source_frame_id):
    img = Image.open(source_img_path).convert("RGB")
    width, height = img.size

    if mask is not None:
        mask = resize_mask_if_needed(mask, width, height)
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_path = OUT_DIR / "source_mask.png"
        mask_img.save(mask_path)

        overlay = img.convert("RGBA")
        red = Image.new("RGBA", (width, height), (255, 0, 0, 0))
        red_arr = np.array(red)
        red_arr[mask] = [255, 0, 0, OVERLAY_ALPHA]
        red = Image.fromarray(red_arr)
        overlay = Image.alpha_composite(overlay, red).convert("RGB")
    else:
        mask_path = None
        overlay = img.copy()

    draw = ImageDraw.Draw(overlay)
    font = get_font(18)

    lines = [
        f"TAKE_ID: {TAKE_ID}",
        f"source view: {SOURCE_VIEW_NAME}",
        f"frame id: {source_frame_id}",
        f"target: {target_prompt}",
        "red mask = target object from JSON" if mask is not None else "NO JSON MASK FOUND",
    ]

    y = 10
    for line in lines:
        draw_text_with_bg(draw, (10, y), line, font)
        y += 28

    overlay_path = OUT_DIR / "source_mask_overlay.png"
    overlay.save(overlay_path)

    return mask_path, overlay_path


def make_contact_sheet(image_paths, out_path):
    sheet_w = CONTACT_SHEET_COLS * THUMB_WIDTH
    sheet_h = CONTACT_SHEET_ROWS * THUMB_HEIGHT
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    font = get_font(18)

    frame_ids = []

    for idx, img_path in enumerate(image_paths):
        row = idx // CONTACT_SHEET_COLS
        col = idx % CONTACT_SHEET_COLS

        x = col * THUMB_WIDTH
        y = row * THUMB_HEIGHT

        img = Image.open(img_path).convert("RGB")
        img.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

        paste_x = x + (THUMB_WIDTH - img.width) // 2
        paste_y = y + (THUMB_HEIGHT - img.height) // 2
        sheet.paste(img, (paste_x, paste_y))

        fid = numeric_stem(img_path)
        frame_ids.append(fid)

        draw.rectangle([x, y, x + THUMB_WIDTH - 1, y + THUMB_HEIGHT - 1], outline=(255, 255, 255))
        draw_text_with_bg(draw, (x + 8, y + 8), f"frame {fid}", font)

    sheet.save(out_path, quality=95)
    return frame_ids


def generate_contact_sheets():
    target_files_all = list_image_files(TARGET_VIEW_DIR)

    if FRAME_STRIDE > 1:
        target_files = target_files_all[::FRAME_STRIDE]
    else:
        target_files = target_files_all

    contact_dir = OUT_DIR / "target_contact_sheets"
    ensure_dir(contact_dir)

    index = []

    num_sheets = math.ceil(len(target_files) / CONTACT_SHEET_SIZE)
    for sheet_idx in range(num_sheets):
        start = sheet_idx * CONTACT_SHEET_SIZE
        end = min((sheet_idx + 1) * CONTACT_SHEET_SIZE, len(target_files))
        chunk = target_files[start:end]

        if not chunk:
            continue

        first_frame = numeric_stem(chunk[0])
        last_frame = numeric_stem(chunk[-1])

        out_path = contact_dir / f"cam01_sheet_{sheet_idx:04d}_frames_{first_frame}_{last_frame}.jpg"
        frame_ids = make_contact_sheet(chunk, out_path)

        index.append({
            "sheet_index": sheet_idx,
            "sheet_path": str(out_path),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frame_ids": frame_ids,
            "num_frames": len(frame_ids),
        })

        if sheet_idx % 20 == 0:
            log(f"Generated sheet {sheet_idx + 1}/{num_sheets}: {out_path}")

    index_path = OUT_DIR / "contact_sheet_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return target_files_all, index, index_path, contact_dir


def write_prompt_file(target_prompt):
    prompt_path = OUT_DIR / "prompt_for_chatgpt.md"

    text = f"""# Cross-view object localization prompt

I will upload these files:

1. `source_first_frame.jpg`
2. `source_mask_overlay.png`
3. One or more `cam01_sheet_*.jpg` contact sheets from the target third-person view.

The red mask in `source_mask_overlay.png` marks the source-view target object from the official JSON annotation.

Target object text: **{target_prompt}**

Task:

Please inspect the source image and the red source mask to understand the target object's appearance. Then inspect the third-person target-view contact sheets. Treat the cam01 frames as a continuous video represented by image frames.

Find the time segment / contact sheet where the same target object appears most clearly in the third-person view.

Please output:

```json
{{
  "target_identity_summary": "Describe the object marked by the source mask.",
  "best_contact_sheet": "filename or sheet index",
  "best_frame_ids": ["frame id 1", "frame id 2"],
  "reason": "Why this segment is best.",
  "frames": [
    {{
      "frame_id": "...",
      "target_visible": true,
      "confidence": 0.0,
      "approx_location": "left/center/right + upper/middle/lower + nearby objects",
      "visual_description": "Detailed appearance of the target object in this frame",
      "recommended_sam_prompt": "A prompt suitable for SAM3, not too generic",
      "bbox_if_visible": [x1, y1, x2, y2],
      "notes": "Mention uncertainty or similar distractor objects"
    }}
  ]
}}