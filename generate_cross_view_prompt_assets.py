from pathlib import Path
import json
import shutil
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TAKE_ID = "00a6dd13-d5b0-4743-b252-ed61e61f1d49"

DATA_ROOT = Path("/home/users/ntu/gwang016/scratch/datasets/Ego-Exo4D-Relation-Test")

TAKE_ROOT = DATA_ROOT / "extracted/work/yuqian_fu/Ego/data_segswap_test" / TAKE_ID

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

THUMB_W = 320
THUMB_H = 240


def log(x):
    print(x, flush=True)


def numeric_stem(p: Path):
    try:
        return int(p.stem)
    except Exception:
        return None


def list_images(folder: Path):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(folder.glob(ext))

    files = [p for p in files if numeric_stem(p) is not None]
    files.sort(key=lambda p: numeric_stem(p))
    return files


def get_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_text_bg(draw, xy, text, font):
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except Exception:
        w, h = draw.textsize(text, font=font)
        bbox = (x, y, x + w, y + h)

    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def collect_strings(x):
    out = []

    def rec(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for vv in v.values():
                rec(vv)
        elif isinstance(v, list):
            for vv in v:
                rec(vv)

    rec(x)
    return out


def find_json_item(source_frame_id: int):
    """
    Try to find one JSON entry matching this take and source view.
    Exact first-frame match may fail, because this old test take may not align perfectly.
    """
    exact = []
    view = []
    take = []

    for jp in JSON_CANDIDATES:
        if not jp.exists():
            log(f"[WARN] JSON not found: {jp}")
            continue

        data = json.load(open(jp))

        for key, item in data.items():
            strings = collect_strings(item) + [str(key)]
            joined = "\n".join(strings)

            if TAKE_ID not in joined:
                continue

            has_view = SOURCE_VIEW_NAME in joined
            has_frame = str(source_frame_id) in joined

            if has_view and has_frame:
                exact.append((jp, key, item))
            elif has_view:
                view.append((jp, key, item))
            else:
                take.append((jp, key, item))

    if exact:
        return exact[0], "exact_take_view_frame"
    if view:
        return view[0], "fallback_take_view"
    if take:
        return take[0], "fallback_take_only"
    return None, "no_match"


def get_target_prompt(item):
    candidates = []

    prompt = item.get("prompt", {})
    if isinstance(prompt, dict):
        for k in ["text", "target_prompt", "object_text", "query"]:
            candidates.append(prompt.get(k))

        anns = prompt.get("first_frame_anns", {})
        if isinstance(anns, dict):
            for ann in anns.values():
                if isinstance(ann, dict):
                    for k in ["text", "label", "category", "name"]:
                        candidates.append(ann.get(k))

    objects = item.get("objects", {})
    if isinstance(objects, dict):
        iterable = objects.values()
    elif isinstance(objects, list):
        iterable = objects
    else:
        iterable = []

    for obj in iterable:
        if isinstance(obj, dict):
            for k in ["text", "label", "category", "name"]:
                candidates.append(obj.get(k))

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()

    return "unknown target object"


def decode_rle(seg):
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        log("[WARN] pycocotools not installed; cannot decode RLE mask.")
        return None

    try:
        m = mask_utils.decode(seg)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return m.astype(bool)
    except Exception as e:
        log(f"[WARN] RLE decode failed: {e}")
        return None


def polygon_to_mask(poly, width, height):
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if isinstance(poly, list):
        if len(poly) > 0 and all(isinstance(x, (int, float)) for x in poly):
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        else:
            for one in poly:
                if isinstance(one, list) and len(one) >= 6:
                    pts = [(one[i], one[i + 1]) for i in range(0, len(one) - 1, 2)]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=255)

    arr = np.array(mask_img) > 0
    if arr.sum() == 0:
        return None
    return arr


def bbox_to_mask(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    x, y, w, h = bbox[:4]
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def extract_annotations(item):
    anns = []

    prompt = item.get("prompt", {})
    if isinstance(prompt, dict):
        first_frame_anns = prompt.get("first_frame_anns", {})
        if isinstance(first_frame_anns, dict):
            for ann_id, ann in first_frame_anns.items():
                if isinstance(ann, dict):
                    ann = dict(ann)
                    ann["_source"] = f"prompt.first_frame_anns.{ann_id}"
                    anns.append(ann)

    objects = item.get("objects", {})
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


def find_mask_from_json_item(item, width, height):
    anns = extract_annotations(item)
    tried = []

    for ann in anns:
        mask = None
        mask_source = None

        # direct mask
        for k in ["mask", "masks"]:
            if k in ann:
                try:
                    arr = np.array(ann[k])
                    if arr.ndim == 2:
                        mask = arr.astype(bool)
                        mask_source = k
                except Exception:
                    pass

        # segmentation
        if mask is None and "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask = decode_rle(seg)
                mask_source = "segmentation_rle" if mask is not None else None
            elif isinstance(seg, list):
                mask = polygon_to_mask(seg, width, height)
                mask_source = "segmentation_polygon" if mask is not None else None

        # bbox fallback
        if mask is None:
            for k in ["bbox", "box", "bounding_box"]:
                if k in ann:
                    mask = bbox_to_mask(ann[k], width, height)
                    if mask is not None:
                        mask_source = f"{k}_fallback"
                        break

        if mask is not None and mask.shape != (height, width):
            img = Image.fromarray(mask.astype(np.uint8) * 255)
            img = img.resize((width, height), Image.NEAREST)
            mask = np.array(img) > 0

        tried.append({
            "ann_source": ann.get("_source"),
            "keys": list(ann.keys()),
            "text": ann.get("text") or ann.get("label") or ann.get("category") or ann.get("name"),
            "mask_found": mask is not None,
            "mask_source": mask_source,
            "mask_area": int(mask.sum()) if mask is not None else 0,
        })

        if mask is not None and mask.sum() > 0:
            return mask, mask_source, ann, tried

    return None, None, None, tried


def save_source_overlay(source_img_path, mask, target_prompt, frame_id):
    img = Image.open(source_img_path).convert("RGB")
    width, height = img.size

    mask_path = None

    if mask is not None:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_path = OUT_DIR / "source_mask.png"
        mask_img.save(mask_path)

        base = img.convert("RGBA")
        red = np.zeros((height, width, 4), dtype=np.uint8)
        red[mask] = [255, 0, 0, 120]
        red_img = Image.fromarray(red, mode="RGBA")
        overlay = Image.alpha_composite(base, red_img).convert("RGB")
    else:
        overlay = img.copy()

    draw = ImageDraw.Draw(overlay)
    font = get_font(18)

    lines = [
        f"TAKE_ID: {TAKE_ID}",
        f"source view: {SOURCE_VIEW_NAME}",
        f"frame id: {frame_id}",
        f"target: {target_prompt}",
        "red mask = JSON target" if mask is not None else "NO JSON MASK FOUND",
    ]

    y = 10
    for line in lines:
        draw_text_bg(draw, (10, y), line, font)
        y += 28

    overlay_path = OUT_DIR / "source_mask_overlay.png"
    overlay.save(overlay_path, quality=95)

    return mask_path, overlay_path


def make_contact_sheet(chunk, out_path):
    sheet = Image.new(
        "RGB",
        (CONTACT_SHEET_COLS * THUMB_W, CONTACT_SHEET_ROWS * THUMB_H),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(sheet)
    font = get_font(18)

    frame_ids = []

    for i, p in enumerate(chunk):
        row = i // CONTACT_SHEET_COLS
        col = i % CONTACT_SHEET_COLS
        x0 = col * THUMB_W
        y0 = row * THUMB_H

        img = Image.open(p).convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H))

        px = x0 + (THUMB_W - img.width) // 2
        py = y0 + (THUMB_H - img.height) // 2
        sheet.paste(img, (px, py))

        fid = numeric_stem(p)
        frame_ids.append(fid)

        draw.rectangle(
            [x0, y0, x0 + THUMB_W - 1, y0 + THUMB_H - 1],
            outline=(255, 255, 255),
            width=1,
        )
        draw_text_bg(draw, (x0 + 8, y0 + 8), f"frame {fid}", font)

    sheet.save(out_path, quality=95)
    return frame_ids


def generate_contact_sheets():
    target_files = list_images(TARGET_VIEW_DIR)
    contact_dir = OUT_DIR / "target_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    index = []
    n_sheets = math.ceil(len(target_files) / CONTACT_SHEET_SIZE)

    for si in range(n_sheets):
        start = si * CONTACT_SHEET_SIZE
        end = min((si + 1) * CONTACT_SHEET_SIZE, len(target_files))
        chunk = target_files[start:end]
        if not chunk:
            continue

        first_frame = numeric_stem(chunk[0])
        last_frame = numeric_stem(chunk[-1])
        out_path = contact_dir / f"cam01_sheet_{si:04d}_frames_{first_frame}_{last_frame}.jpg"

        frame_ids = make_contact_sheet(chunk, out_path)

        index.append({
            "sheet_index": si,
            "sheet_path": str(out_path),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frame_ids": frame_ids,
            "num_frames": len(frame_ids),
        })

        if si % 20 == 0:
            log(f"Generated contact sheet {si + 1}/{n_sheets}: {out_path}")

    index_path = OUT_DIR / "contact_sheet_index.json"
    json.dump(index, open(index_path, "w"), indent=2)

    return target_files, contact_dir, index_path, index


def write_prompt_file(target_prompt):
    prompt_path = OUT_DIR / "prompt_for_chatgpt.md"

    text = "\n".join([
        "# Cross-view object localization prompt",
        "",
        "I will upload:",
        "1. source_first_frame.jpg",
        "2. source_mask_overlay.png",
        "3. one or more cam01 contact sheets",
        "",
        "The red mask in source_mask_overlay.png marks the target object in the source first-person view.",
        "",
        f"Target object text: {target_prompt}",
        "",
        "Task:",
        "Inspect the source image and red mask to understand the target object's appearance.",
        "Then inspect the third-person cam01 contact sheets.",
        "Treat the cam01 frames as a continuous video represented by image frames.",
        "",
        "Find the segment where the same target object appears most clearly, largest, and least occluded.",
        "",
        "Please output JSON:",
        "{",
        '  "target_identity_summary": "...",',
        '  "best_contact_sheet": "...",',
        '  "best_frame_ids": ["..."],',
        '  "reason": "...",',
        '  "frames": [',
        "    {",
        '      "frame_id": "...",',
        '      "target_visible": true,',
        '      "confidence": 0.0,',
        '      "approx_location": "...",',
        '      "visual_description": "...",',
        '      "recommended_sam_prompt": "...",',
        '      "bbox_if_visible": [x1, y1, x2, y2],',
        '      "notes": "..."',
        "    }",
        "  ]",
        "}",
        "",
        "Important:",
        "- Do not simply choose any similar object.",
        "- The source red mask defines the target identity.",
        "- Use appearance, surrounding context, and temporal continuity.",
        "- If uncertain, say uncertain.",
        "",
    ])

    prompt_path.write_text(text)
    return prompt_path


def main():
    log("=" * 80)
    log("Generate cross-view prompt assets")
    log("=" * 80)

    for p in [DATA_ROOT, TAKE_ROOT, SOURCE_VIEW_DIR, TARGET_VIEW_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Missing path: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_images = list_images(SOURCE_VIEW_DIR)
    if not source_images:
        raise RuntimeError(f"No source images found: {SOURCE_VIEW_DIR}")

    source_first = source_images[0]
    source_frame_id = numeric_stem(source_first)

    log(f"Source first frame: {source_first}")
    log(f"Source frame id: {source_frame_id}")

    source_copy = OUT_DIR / "source_first_frame.jpg"
    shutil.copy(source_first, source_copy)

    match, match_mode = find_json_item(source_frame_id)

    json_file = None
    json_key = None
    item = None
    target_prompt = "unknown target object"
    mask = None
    mask_source = None
    matched_ann = None
    tried = []

    if match is None:
        log("[WARN] No JSON item matched this take/source view.")
    else:
        json_file, json_key, item = match
        target_prompt = get_target_prompt(item)

        log(f"JSON match mode: {match_mode}")
        log(f"JSON file: {json_file}")
        log(f"JSON key: {json_key}")
        log(f"Target prompt: {target_prompt}")

        img = Image.open(source_first).convert("RGB")
        width, height = img.size

        mask, mask_source, matched_ann, tried = find_mask_from_json_item(item, width, height)

        if mask is not None:
            log(f"Mask found: {mask_source}, area={int(mask.sum())}")
        else:
            log("[WARN] No mask found in JSON item. Overlay will have no red mask.")

    mask_path, overlay_path = save_source_overlay(
        source_copy,
        mask,
        target_prompt,
        source_frame_id,
    )

    log(f"Source first frame copied to: {source_copy}")
    log(f"Source mask overlay saved to: {overlay_path}")

    log("Generating cam01 contact sheets...")
    target_files, contact_dir, contact_index_path, contact_index = generate_contact_sheets()

    log(f"Target cam01 frame count: {len(target_files)}")
    log(f"Contact sheet count: {len(contact_index)}")
    log(f"Contact sheet dir: {contact_dir}")

    prompt_path = write_prompt_file(target_prompt)

    metadata = {
        "take_id": TAKE_ID,
        "source_view_name": SOURCE_VIEW_NAME,
        "target_view_name": TARGET_VIEW_NAME,
        "source_view_dir": str(SOURCE_VIEW_DIR),
        "target_view_dir": str(TARGET_VIEW_DIR),
        "source_first_frame_original": str(source_first),
        "source_first_frame_path": str(source_copy),
        "source_frame_id": source_frame_id,
        "json_match_mode": match_mode,
        "json_file": str(json_file) if json_file else None,
        "json_key": json_key,
        "target_prompt": target_prompt,
        "mask_found": mask is not None,
        "mask_source": mask_source,
        "source_mask_path": str(mask_path) if mask_path else None,
        "source_mask_overlay_path": str(overlay_path),
        "number_of_target_frames": len(target_files),
        "contact_sheet_dir": str(contact_dir),
        "number_of_contact_sheets": len(contact_index),
        "contact_sheet_index_path": str(contact_index_path),
        "prompt_for_chatgpt_path": str(prompt_path),
        "matched_annotation_source": matched_ann.get("_source") if matched_ann else None,
        "matched_annotation_keys": list(matched_ann.keys()) if matched_ann else None,
        "tried_annotations": tried,
    }

    metadata_path = OUT_DIR / "metadata.json"
    json.dump(metadata, open(metadata_path, "w"), indent=2)

    log(f"Metadata saved to: {metadata_path}")
    log(f"Prompt saved to: {prompt_path}")

    log("\nDone. Upload these to ChatGPT:")
    log(f"1. {source_copy}")
    log(f"2. {overlay_path}")
    log(f"3. Several images from {contact_dir}")
    log(f"4. Use prompt text from {prompt_path}")


if __name__ == "__main__":
    main()
