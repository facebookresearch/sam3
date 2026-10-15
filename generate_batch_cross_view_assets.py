from pathlib import Path
import json
import shutil
import math
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# 修改这里
# =========================

DATASET_ROOT = Path(
    "/home/users/ntu/gwang016/scratch/datasets/Ego-Exo4D-Relation-Test/"
    "extracted/work/yuqian_fu/Ego/data_segswap_test"
)

OUT_ROOT = Path(
    "/home/users/ntu/gwang016/scratch/batch_cross_view_prompt_assets"
)

# 如果只想处理指定 5 个 take，把完整 take_id 填到这里。
# 如果留空 []，脚本会自动处理 DATASET_ROOT 下所有包含 annotation.json 的 take 文件夹。
TAKE_IDS = [
    "0a22a1c1-844c-4f62-8eeb-f16eee62357f",
    "0a3868ef-fdba-4aba-bc02-5028d1ed26f4",
    "0aebd388-55a4-4425-a9ee-f09b64d94a00",
    "0b82763e-b9ee-40e5-8dd5-b8da7e862662",
    "0bacb5bb-591d-4756-a2cf-ed90793e65bb",
]

# 只处理前 N 个 case。设置为 None 表示不限制。
MAX_CASES = 5

# overview 用于粗筛，拼得多一点。
OVERVIEW_COLS = 6
OVERVIEW_ROWS = 6
OVERVIEW_THUMB_W = 220
OVERVIEW_THUMB_H = 165

# detail 用于精筛，拼得清楚一点。
DETAIL_COLS = 4
DETAIL_ROWS = 4
DETAIL_THUMB_W = 320
DETAIL_THUMB_H = 240

# detail contact sheet 每隔多少帧取一张。1 表示所有帧都做。
DETAIL_STRIDE = 1

# overview 总共均匀抽多少帧。36 对应 6x6。
OVERVIEW_NUM_FRAMES = OVERVIEW_COLS * OVERVIEW_ROWS


# =========================
# 工具函数
# =========================

def log(msg):
    print(msg, flush=True)


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


def draw_text_bg(draw, xy, text, font, fill=(255, 255, 255), bg=(0, 0, 0)):
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


def safe_case_name(take_id: str):
    return take_id.replace("/", "_")


def discover_cases():
    if TAKE_IDS:
        case_dirs = []
        for tid in TAKE_IDS:
            p = DATASET_ROOT / tid
            if p.exists() and (p / "annotation.json").exists():
                case_dirs.append(p)
            else:
                log(f"[WARN] Case missing or no annotation.json: {p}")
        return case_dirs

    case_dirs = []
    for p in sorted(DATASET_ROOT.iterdir()):
        if p.is_dir() and (p / "annotation.json").exists():
            case_dirs.append(p)

    if MAX_CASES is not None:
        case_dirs = case_dirs[:MAX_CASES]

    return case_dirs


def find_source_and_target_views(case_dir: Path):
    aria_dirs = sorted([p for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("aria")])
    cam_dirs = sorted([p for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("cam")])

    if not aria_dirs:
        raise RuntimeError(f"No aria* source view dirs found in {case_dir}")

    if not cam_dirs:
        raise RuntimeError(f"No cam* target view dirs found in {case_dir}")

    # 默认取第一个 aria 作为 source。
    source_dir = aria_dirs[0]

    return source_dir, cam_dirs


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def decode_rle_mask(rle):
    """
    Decode COCO RLE mask with keys: size, counts.
    """
    try:
        from pycocotools import mask as mask_utils
    except Exception as e:
        raise RuntimeError(
            "pycocotools is required to decode this annotation.json RLE mask. "
            "Install it with: python -m pip install pycocotools"
        ) from e

    if not isinstance(rle, dict):
        return None

    if "size" not in rle or "counts" not in rle:
        return None

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)

    return mask.astype(bool)


def choose_annotation_object_and_source(annotation, source_dir_name):
    """
    Handle annotation format:
    annotation["masks"][object_name][view_name][frame_id] = {"size": ..., "counts": ...}

    Returns:
    object_name, view_name, frame_id, rle
    """
    masks_root = annotation.get("masks")

    if not isinstance(masks_root, dict) or not masks_root:
        return None, None, None, None

    object_names = list(masks_root.keys())

    # 当前 case 通常只有一个目标物体，比如 "CPR dummy"
    object_name = object_names[0]
    obj_data = masks_root[object_name]

    if not isinstance(obj_data, dict):
        return object_name, None, None, None

    view_names = list(obj_data.keys())

    # 优先用当前 source view，比如 aria02_214-1
    if source_dir_name in obj_data:
        view_name = source_dir_name
    else:
        # fallback：优先找 aria 开头的 view
        aria_views = [v for v in view_names if str(v).startswith("aria")]
        view_name = aria_views[0] if aria_views else view_names[0]

    frame_dict = obj_data.get(view_name, {})
    if not isinstance(frame_dict, dict) or not frame_dict:
        return object_name, view_name, None, None

    # frame id 是字符串，比如 "0", "30", ...
    frame_ids = []
    for k in frame_dict.keys():
        try:
            frame_ids.append(int(k))
        except Exception:
            pass

    if not frame_ids:
        return object_name, view_name, None, None

    # 选最早一帧作为 source frame
    frame_id = min(frame_ids)
    rle = frame_dict[str(frame_id)]

    return object_name, view_name, frame_id, rle


def get_annotation_driven_source(annotation, case_dir, default_source_dir):
    """
    根据 annotation.json 决定真正的 source view 和 source frame。
    如果 annotation 中存在 aria view，就用 annotation 里的 aria view 和最早 frame。
    """
    masks_root = annotation.get("masks", {})
    if not isinstance(masks_root, dict) or not masks_root:
        return default_source_dir, None, "unknown target object", None, None

    object_name = list(masks_root.keys())[0]
    obj_data = masks_root[object_name]

    if not isinstance(obj_data, dict):
        return default_source_dir, None, object_name, None, None

    view_names = list(obj_data.keys())
    aria_views = [v for v in view_names if str(v).startswith("aria")]

    if aria_views:
        source_view_name = aria_views[0]
    else:
        source_view_name = default_source_dir.name

    source_dir = case_dir / source_view_name
    if not source_dir.exists():
        source_dir = default_source_dir

    object_name, view_name, frame_id, rle = choose_annotation_object_and_source(
        annotation,
        source_dir.name,
    )

    return source_dir, frame_id, object_name, view_name, rle

# =========================
# annotation 解析：尽量兼容
# =========================

def collect_candidate_dicts(obj, path="root"):
    """
    递归收集可能包含 mask / segmentation / bbox / text 的 dict。
    """
    out = []

    if isinstance(obj, dict):
        keys = set(obj.keys())
        interesting = {
            "mask", "masks", "segmentation", "segments",
            "bbox", "box", "bounding_box",
            "text", "label", "category", "name", "object", "object_name"
        }
        if keys & interesting:
            d = dict(obj)
            d["_json_path"] = path
            out.append(d)

        for k, v in obj.items():
            out.extend(collect_candidate_dicts(v, f"{path}.{k}"))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(collect_candidate_dicts(v, f"{path}[{i}]"))

    return out


def extract_target_prompt(annotation):
    candidates = []

    def rec(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.lower() in ["text", "label", "category", "name", "object", "object_name", "target", "target_prompt"]:
                    if isinstance(v, str):
                        candidates.append(v)
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(annotation)

    for c in candidates:
        c = c.strip()
        if c and len(c) < 120:
            return c

    return "unknown target object"


def decode_rle(seg):
    try:
        from pycocotools import mask as mask_utils
    except Exception:
        log("[WARN] pycocotools not installed; cannot decode RLE.")
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

    try:
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
    except Exception as e:
        log(f"[WARN] polygon_to_mask failed: {e}")
        return None

    arr = np.array(mask_img) > 0
    if arr.sum() == 0:
        return None
    return arr


def bbox_to_mask(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        x, y, w, h = bbox[:4]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
    except Exception:
        return None

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def resize_mask_if_needed(mask, width, height):
    if mask.shape == (height, width):
        return mask.astype(bool)

    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((width, height), Image.NEAREST)
    return np.array(img) > 0


def mask_from_candidate(d, width, height):
    # direct mask
    for k in ["mask", "masks"]:
        if k in d:
            try:
                arr = np.array(d[k])
                if arr.ndim == 2:
                    return resize_mask_if_needed(arr.astype(bool), width, height), k
            except Exception:
                pass

    # segmentation
    for k in ["segmentation", "segments"]:
        if k in d:
            seg = d[k]
            if isinstance(seg, dict):
                m = decode_rle(seg)
                if m is not None:
                    return resize_mask_if_needed(m, width, height), f"{k}_rle"
            elif isinstance(seg, list):
                m = polygon_to_mask(seg, width, height)
                if m is not None:
                    return resize_mask_if_needed(m, width, height), f"{k}_polygon"

    # bbox fallback
    for k in ["bbox", "box", "bounding_box"]:
        if k in d:
            m = bbox_to_mask(d[k], width, height)
            if m is not None:
                return m, f"{k}_fallback"

    return None, None


def find_first_mask(annotation, width, height):
    candidates = collect_candidate_dicts(annotation)

    tried = []
    for d in candidates:
        m, source = mask_from_candidate(d, width, height)
        text = (
            d.get("text")
            or d.get("label")
            or d.get("category")
            or d.get("name")
            or d.get("object_name")
            or d.get("object")
        )

        info = {
            "json_path": d.get("_json_path"),
            "keys": list(d.keys()),
            "text": text,
            "mask_found": m is not None,
            "mask_source": source,
            "mask_area": int(m.sum()) if m is not None else 0,
        }
        tried.append(info)

        if m is not None and m.sum() > 0:
            return m, source, d, tried

    return None, None, None, tried


# =========================
# 图像输出
# =========================

def save_source_overlay(case_out, take_id, source_view_name, source_img_path, mask, target_prompt):
    img = Image.open(source_img_path).convert("RGB")
    width, height = img.size
    frame_id = numeric_stem(source_img_path)

    source_copy = case_out / "source_first_frame.jpg"
    shutil.copy(source_img_path, source_copy)

    mask_path = None
    if mask is not None:
        mask = resize_mask_if_needed(mask, width, height)
        mask_path = case_out / "source_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

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
        f"take: {take_id}",
        f"source view: {source_view_name}",
        f"source frame: {frame_id}",
        f"target: {target_prompt}",
        "red mask = annotation target" if mask is not None else "NO MASK FOUND",
    ]

    y = 10
    for line in lines:
        draw_text_bg(draw, (10, y), line, font)
        y += 28

    overlay_path = case_out / "source_mask_overlay.png"
    overlay.save(overlay_path, quality=95)

    return source_copy, mask_path, overlay_path


def make_sheet(image_paths, out_path, cols, rows, thumb_w, thumb_h, title=None):
    sheet_w = cols * thumb_w
    sheet_h = rows * thumb_h
    if title:
        title_h = 40
    else:
        title_h = 0

    sheet = Image.new("RGB", (sheet_w, sheet_h + title_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    font = get_font(16)
    title_font = get_font(20)

    if title:
        draw_text_bg(draw, (10, 8), title, title_font)

    frame_ids = []

    for i, p in enumerate(image_paths[: cols * rows]):
        row = i // cols
        col = i % cols

        x0 = col * thumb_w
        y0 = row * thumb_h + title_h

        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))

        px = x0 + (thumb_w - img.width) // 2
        py = y0 + (thumb_h - img.height) // 2
        sheet.paste(img, (px, py))

        fid = numeric_stem(p)
        frame_ids.append(fid)

        draw.rectangle(
            [x0, y0, x0 + thumb_w - 1, y0 + thumb_h - 1],
            outline=(255, 255, 255),
            width=1,
        )
        draw_text_bg(draw, (x0 + 6, y0 + 6), f"frame {fid}", font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)
    return frame_ids


def evenly_sample(files, n):
    if len(files) <= n:
        return files
    idxs = np.linspace(0, len(files) - 1, n).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    return [files[i] for i in idxs]


def generate_overview_sheets(case_out, take_id, cam_name, target_files):
    out_dir = case_out / "target_contact_sheets_overview" / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = evenly_sample(target_files, OVERVIEW_NUM_FRAMES)

    first_frame = numeric_stem(sampled[0]) if sampled else None
    last_frame = numeric_stem(sampled[-1]) if sampled else None

    out_path = out_dir / f"{cam_name}_overview_{first_frame}_{last_frame}.jpg"

    frame_ids = make_sheet(
        sampled,
        out_path,
        OVERVIEW_COLS,
        OVERVIEW_ROWS,
        OVERVIEW_THUMB_W,
        OVERVIEW_THUMB_H,
        title=f"{take_id} | {cam_name} overview | {first_frame}-{last_frame}",
    )

    return [{
        "type": "overview",
        "cam_name": cam_name,
        "sheet_path": str(out_path),
        "frame_ids": frame_ids,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "num_frames": len(frame_ids),
    }]


def generate_detail_sheets(case_out, take_id, cam_name, target_files):
    out_dir = case_out / "target_contact_sheets_detail" / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if DETAIL_STRIDE > 1:
        files = target_files[::DETAIL_STRIDE]
    else:
        files = target_files

    sheet_size = DETAIL_COLS * DETAIL_ROWS
    n_sheets = math.ceil(len(files) / sheet_size)

    index = []

    for si in range(n_sheets):
        start = si * sheet_size
        end = min((si + 1) * sheet_size, len(files))
        chunk = files[start:end]
        if not chunk:
            continue

        first_frame = numeric_stem(chunk[0])
        last_frame = numeric_stem(chunk[-1])

        out_path = out_dir / f"{cam_name}_detail_{si:04d}_frames_{first_frame}_{last_frame}.jpg"

        frame_ids = make_sheet(
            chunk,
            out_path,
            DETAIL_COLS,
            DETAIL_ROWS,
            DETAIL_THUMB_W,
            DETAIL_THUMB_H,
            title=f"{take_id} | {cam_name} detail {si:04d} | {first_frame}-{last_frame}",
        )

        index.append({
            "type": "detail",
            "cam_name": cam_name,
            "sheet_index": si,
            "sheet_path": str(out_path),
            "frame_ids": frame_ids,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "num_frames": len(frame_ids),
        })

    return index


def write_case_prompt(case_out, take_id, target_prompt):
    prompt_path = case_out / "prompt_for_chatgpt.md"

    text = "\n".join([
        "# Cross-view target localization prompt",
        "",
        f"case_id: {take_id}",
        f"target_prompt from annotation: {target_prompt}",
        "",
        "I will upload:",
        "1. source_first_frame.jpg",
        "2. source_mask_overlay.png",
        "3. overview contact sheet(s) from cam views",
        "4. later, detail contact sheet(s) if needed",
        "",
        "The red mask in source_mask_overlay.png defines the target object identity.",
        "",
        "Task:",
        "Find the same object in the target cam contact sheets.",
        "First use overview sheets to identify the best time range.",
        "Then use detail sheets to select the clearest frames.",
        "",
        "For this case, output:",
        "{",
        '  "case_id": "...",',
        '  "source_target_description": "...",',
        '  "best_cam": "...",',
        '  "best_contact_sheet": "...",',
        '  "best_frame_ids": ["..."],',
        '  "best_segment_reason": "...",',
        '  "recommended_sam_prompt": "...",',
        '  "frames": [',
        "    {",
        '      "frame_id": "...",',
        '      "target_visible": true,',
        '      "confidence": 0.0,',
        '      "approx_location": "...",',
        '      "visual_description": "...",',
        '      "bbox_if_visible": [x1, y1, x2, y2],',
        '      "reason_selected_or_rejected": "...",',
        '      "possible_distractors": "..."',
        "    }",
        "  ],",
        '  "rejected_candidates": [',
        "    {",
        '      "frame_id": "...",',
        '      "reason_rejected": "..."',
        "    }",
        "  ],",
        '  "uncertainty": "..."',
        "}",
        "",
        "Important:",
        "- Do not simply choose any visually similar object.",
        "- The source red mask defines the target identity.",
        "- Explain why selected frames are better than rejected frames.",
        "- If uncertain, say uncertain.",
        "",
    ])

    prompt_path.write_text(text)
    return prompt_path


# =========================
# 单 case / batch 主流程
# =========================

def process_case(case_dir: Path):
    take_id = case_dir.name
    case_out = OUT_ROOT / safe_case_name(take_id)
    case_out.mkdir(parents=True, exist_ok=True)

    log("\n" + "=" * 100)
    log(f"Processing case: {take_id}")
    log("=" * 100)

    annotation_path = case_dir / "annotation.json"
    annotation = load_json(annotation_path)

    default_source_dir, cam_dirs = find_source_and_target_views(case_dir)

    # annotation-driven source view / frame / object / mask
    source_dir, ann_frame_id, target_prompt, ann_view_name, ann_rle = get_annotation_driven_source(
        annotation,
        case_dir,
        default_source_dir,
    )

    source_images = list_images(source_dir)
    if not source_images:
        raise RuntimeError(f"No images in source view: {source_dir}")

    # 如果 annotation 指定 frame，比如 0，就优先用对应图片
    if ann_frame_id is not None:
        candidate_source = source_dir / f"{ann_frame_id}.jpg"
        if not candidate_source.exists():
            candidate_source = source_dir / f"{ann_frame_id}.png"

        if candidate_source.exists():
            source_first = candidate_source
        else:
            print(f"[WARN] Annotation frame {ann_frame_id} not found in {source_dir}; fallback to first image.")
            source_first = source_images[0]
    else:
        source_first = source_images[0]

    source_frame_id = numeric_stem(source_first)

    img = Image.open(source_first).convert("RGB")
    width, height = img.size

    mask = None
    mask_source = None
    matched_ann = None
    tried = []

    if ann_rle is not None:
        try:
            mask = decode_rle_mask(ann_rle)
            if mask is not None:
                mask = resize_mask_if_needed(mask, width, height)
                mask_source = f"annotation.masks.{target_prompt}.{source_dir.name}.{source_frame_id}"
                matched_ann = {
                    "_json_path": mask_source,
                    "object_name": target_prompt,
                    "view_name": source_dir.name,
                    "frame_id": source_frame_id,
                }
                tried.append({
                    "json_path": mask_source,
                    "text": target_prompt,
                    "mask_found": True,
                    "mask_source": mask_source,
                    "mask_area": int(mask.sum()),
                })
        except Exception as e:
            print(f"[WARN] Failed to decode annotation RLE mask: {e}")
            tried.append({
                "json_path": f"annotation.masks.{target_prompt}.{source_dir.name}.{source_frame_id}",
                "text": target_prompt,
                "mask_found": False,
                "mask_source": "rle_decode_failed",
                "error": str(e),
            })

    if mask is None:
        print("[WARN] Annotation-driven RLE mask failed; fallback to generic parser.")
        mask, mask_source, matched_ann, tried_generic = find_first_mask(annotation, width, height)
        tried.extend(tried_generic)

    log(f"Source view: {source_dir.name}")
    log(f"Source first frame: {source_first.name}")
    log(f"Target prompt: {target_prompt}")
    log(f"Mask found: {mask is not None}, source={mask_source}")

    source_copy, mask_path, overlay_path = save_source_overlay(
        case_out,
        take_id,
        source_dir.name,
        source_first,
        mask,
        target_prompt,
    )

    all_sheet_index = []

    for cam_dir in cam_dirs:
        target_files = list_images(cam_dir)
        if not target_files:
            log(f"[WARN] No images in cam view: {cam_dir}")
            continue

        log(f"Generating sheets for {cam_dir.name}, frames={len(target_files)}")

        overview_index = generate_overview_sheets(case_out, take_id, cam_dir.name, target_files)
        detail_index = generate_detail_sheets(case_out, take_id, cam_dir.name, target_files)

        all_sheet_index.extend(overview_index)
        all_sheet_index.extend(detail_index)

    contact_index_path = case_out / "contact_sheet_index.json"
    json.dump(all_sheet_index, open(contact_index_path, "w"), indent=2)

    prompt_path = write_case_prompt(case_out, take_id, target_prompt)

    metadata = {
        "case_id": take_id,
        "case_dir": str(case_dir),
        "annotation_path": str(annotation_path),
        "source_view": source_dir.name,
        "source_view_dir": str(source_dir),
        "source_first_frame_original": str(source_first),
        "source_first_frame_path": str(source_copy),
        "source_frame_id": source_frame_id,
        "target_cam_views": [p.name for p in cam_dirs],
        "target_prompt": target_prompt,
        "mask_found": mask is not None,
        "mask_source": mask_source,
        "matched_annotation_json_path": matched_ann.get("_json_path") if matched_ann else None,
        "source_mask_path": str(mask_path) if mask_path else None,
        "source_mask_overlay_path": str(overlay_path),
        "contact_sheet_index_path": str(contact_index_path),
        "prompt_for_chatgpt_path": str(prompt_path),
        "num_contact_sheets": len(all_sheet_index),
        "tried_annotations": tried[:50],
    }

    metadata_path = case_out / "metadata.json"
    json.dump(metadata, open(metadata_path, "w"), indent=2)

    log(f"Output case dir: {case_out}")
    log(f"Overlay: {overlay_path}")
    log(f"Metadata: {metadata_path}")
    log(f"Contact index: {contact_index_path}")

    return metadata


def main():
    log("=" * 100)
    log("Batch cross-view prompt asset generation")
    log("=" * 100)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    case_dirs = discover_cases()
    log(f"Discovered cases: {len(case_dirs)}")

    if not case_dirs:
        raise RuntimeError("No cases found. Check DATASET_ROOT or TAKE_IDS.")

    batch_metadata = []

    for case_dir in case_dirs:
        try:
            md = process_case(case_dir)
            batch_metadata.append(md)
        except Exception as e:
            log(f"[ERROR] Failed case {case_dir.name}: {e}")
            batch_metadata.append({
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "error": str(e),
            })

    batch_index_path = OUT_ROOT / "batch_metadata.json"
    json.dump(batch_metadata, open(batch_index_path, "w"), indent=2)

    log("\n" + "=" * 100)
    log("DONE")
    log("=" * 100)
    log(f"Batch output root: {OUT_ROOT}")
    log(f"Batch metadata: {batch_index_path}")
    log("")
    log("For ChatGPT upload, start with each case's:")
    log("1. source_mask_overlay.png")
    log("2. target_contact_sheets_overview/<cam_name>/*.jpg")
    log("3. metadata.json")
    log("")
    log("Then upload detail sheets only for the selected time ranges.")


if __name__ == "__main__":
    main()
