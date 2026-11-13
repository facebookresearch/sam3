import json
import shutil
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("/home/users/ntu/gwang016/scratch/datasets/Ego-Exo4D-Relation-Test")
IMAGE_ROOT = DATA_ROOT / "extracted" / "work" / "yuqian_fu" / "Ego" / "data_segswap_test"

EGO2EXO_JSON = DATA_ROOT / "ego2exo_test.json"
EXO2EGO_JSON = DATA_ROOT / "exo2ego_test.json"

OUT_ROOT = Path("/home/users/ntu/gwang016/scratch/sam3_test_same_object")
N_FRAMES = 20


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def split_key(key):
    take_id, frame = key.rsplit("_", 1)
    return take_id, int(frame)


def get_target_text(item):
    anns = item.get("prompt", {}).get("first_frame_anns", {})
    if anns:
        first_ann = next(iter(anns.values()))
        if "text" in first_ann:
            return first_ann["text"]

    objects = item.get("objects", {})
    if objects:
        first_obj = next(iter(objects.values()))
        return (
            first_obj.get("final_caption")
            or first_obj.get("formated")
            or first_obj.get("crop_category")
            or first_obj.get("image_caption")
        )

    return "target_object"


def clean_target_text(text):
    text = str(text)
    if text.endswith("_0"):
        text = text[:-2]
    return text.replace("_", " ")


def group_by_take_object(data):
    grouped = defaultdict(dict)

    for key, item in data.items():
        take_id, frame = split_key(key)
        target_text = get_target_text(item)
        target_prompt = clean_target_text(target_text)

        group_key = (take_id, target_prompt)
        grouped[group_key][frame] = {
            "key": key,
            "item": item,
            "target_text": target_text,
            "target_prompt": target_prompt,
        }

    return grouped


def evenly_sample(frames, n):
    frames = sorted(frames)
    if len(frames) <= n:
        return frames

    idxs = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in idxs]


def copy_direction(direction_name, grouped_data, group_key, selected_frames, out_dir):
    target_out = out_dir / f"{direction_name}_target"
    target_out.mkdir(parents=True, exist_ok=True)

    copied = []
    take_id, target_prompt = group_key

    for frame in selected_frames:
        record = grouped_data[group_key][frame]
        key = record["key"]
        item = record["item"]
        target_text = record["target_text"]
        target_prompt = record["target_prompt"]

        rel_path = item["video_path"][0]
        src = IMAGE_ROOT / rel_path

        if not src.exists():
            print(f"[WARN] Missing image: {src}")
            continue

        dst = target_out / f"{key}.jpg"
        shutil.copy2(src, dst)

        copied.append({
            "key": key,
            "frame": frame,
            "source_image": str(src),
            "copied_image": str(dst),
            "target_text": target_text,
            "target_prompt": target_prompt,
            "direction": direction_name,
            "prompt_image": item["prompt"]["first_frame_image"],
        })

    return {
        "direction": direction_name,
        "take_id": take_id,
        "target_prompt": target_prompt,
        "num_copied": len(copied),
        "copied": copied,
    }


def main():
    ego2exo = load_json(EGO2EXO_JSON)
    exo2ego = load_json(EXO2EGO_JSON)

    ego_grouped = group_by_take_object(ego2exo)
    exo_grouped = group_by_take_object(exo2ego)

    common_group_keys = sorted(set(ego_grouped.keys()) & set(exo_grouped.keys()))
    print("Common take_id + object groups:", len(common_group_keys))

    candidates = []

    for group_key in common_group_keys:
        ego_frames = set(ego_grouped[group_key].keys())
        exo_frames = set(exo_grouped[group_key].keys())
        common_frames = sorted(ego_frames & exo_frames)

        if len(common_frames) >= N_FRAMES:
            take_id, target_prompt = group_key
            candidates.append({
                "take_id": take_id,
                "target_prompt": target_prompt,
                "num_common_frames": len(common_frames),
                "group_key": group_key,
            })

    if not candidates:
        raise RuntimeError("No common take_id + target object group with enough paired frames found.")

    candidates = sorted(candidates, key=lambda x: x["num_common_frames"], reverse=True)
    chosen = candidates[0]
    group_key = chosen["group_key"]

    print("Chosen group:")
    print(json.dumps({
        "take_id": chosen["take_id"],
        "target_prompt": chosen["target_prompt"],
        "num_common_frames": chosen["num_common_frames"],
    }, indent=2, ensure_ascii=False))

    common_frames = sorted(
        set(ego_grouped[group_key].keys()) & set(exo_grouped[group_key].keys())
    )

    trim = int(len(common_frames) * 0.05)
    if len(common_frames) > 40:
        common_frames = common_frames[trim:-trim]

    selected_frames = evenly_sample(common_frames, N_FRAMES)

    print("Selected frames:")
    print(selected_frames)

    if OUT_ROOT.exists():
        for p in OUT_ROOT.rglob("*.jpg"):
            p.unlink()
        meta_path = OUT_ROOT / "meta.json"
        if meta_path.exists():
            meta_path.unlink()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta = {
        "chosen_group": {
            "take_id": chosen["take_id"],
            "target_prompt": chosen["target_prompt"],
            "num_common_frames": chosen["num_common_frames"],
        },
        "selected_target_prompt": chosen["target_prompt"],
        "selected_frames": selected_frames,
        "ego2exo": copy_direction("ego2exo", ego_grouped, group_key, selected_frames, OUT_ROOT),
        "exo2ego": copy_direction("exo2ego", exo_grouped, group_key, selected_frames, OUT_ROOT),
    }

    with open(OUT_ROOT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done.")
    print("Output root:", OUT_ROOT)
    print("Meta:", OUT_ROOT / "meta.json")
    print(json.dumps({
        "take_id": meta["chosen_group"]["take_id"],
        "target_prompt": meta["selected_target_prompt"],
        "selected_frames": selected_frames,
        "ego2exo_num": meta["ego2exo"]["num_copied"],
        "exo2ego_num": meta["exo2ego"]["num_copied"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
