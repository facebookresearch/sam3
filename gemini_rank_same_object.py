import os
import json
import re
import time
from pathlib import Path
from PIL import Image
from google import genai

ROOT = Path("/home/users/ntu/gwang016/scratch/sam3_test_same_object")
META_PATH = ROOT / "meta.json"
OUT_PATH = ROOT / "gemini_ranked_candidates.json"

TOP_K = 5

VIEWS = {
    "ego2exo_target": ROOT / "ego2exo_target",
    "exo2ego_target": ROOT / "exo2ego_target",
}


def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def load_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)


def get_frame_from_filename(filename: str) -> int:
    # f4a17391-..._2490.jpg -> 2490
    stem = Path(filename).stem
    return int(stem.rsplit("_", 1)[1])


def rank_one_view(client, view_name: str, image_dir: Path, target_prompt: str):
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda p: get_frame_from_filename(p.name))

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"\nRanking view: {view_name}")
    print(f"Image dir: {image_dir}")
    print(f"Num images: {len(image_paths)}")
    print(f"Target object: {target_prompt}")

    contents = [
        f"""
You are selecting candidate images for object segmentation.

Target object: {target_prompt}
View name: {view_name}

You will see multiple images from the same video sequence.
For each image, judge whether the target object is visible and suitable for segmentation.

Return JSON only. Do not use markdown fences.

Required JSON schema:
{{
  "view_name": "{view_name}",
  "target_object": "{target_prompt}",
  "results": [
    {{
      "image_id": "filename.jpg",
      "target_present": true,
      "relative_size_score": 0.0,
      "visibility_score": 0.0,
      "segmentation_suitability_score": 0.0,
      "occlusion_level": "none",
      "reason": "short explanation"
    }}
  ]
}}

Scoring rules:
- target_present: true only if the target object is visible.
- relative_size_score: 0.0 to 1.0. Higher means the visible target object occupies more image area.
- visibility_score: 0.0 to 1.0. Higher means the object is clearer and easier to see.
- segmentation_suitability_score: 0.0 to 1.0. Higher means easier for SAM-style segmentation.
- occlusion_level must be one of: none, low, medium, high.
- If target is absent, set target_present=false and all three scores to 0.
- Prefer images where the target object is large, clear, complete, and not heavily occluded.
- Make sure every input image appears exactly once in results.
"""
    ]

    for p in image_paths:
        contents.append(f"Image filename: {p.name}")
        contents.append(Image.open(p).convert("RGB"))

    last_error = None

    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            break
        except Exception as e:
            last_error = e
            wait = 10 * (attempt + 1)
            print(f"[WARN] Gemini failed on attempt {attempt + 1}/6: {e}")
            print(f"[WARN] Sleeping {wait} seconds before retry...")
            time.sleep(wait)
    else:
        raise last_error

    raw_text = response.text
    print(f"\n===== Raw Gemini response for {view_name} =====")
    print(raw_text)

    data = json.loads(clean_json_text(raw_text))

    results = data["results"]

    for r in results:
        r["view_name"] = view_name
        r["image_path"] = str(image_dir / r["image_id"])
        r["frame"] = get_frame_from_filename(r["image_id"])

    valid = [r for r in results if r.get("target_present") is True]

    ranked = sorted(
        valid,
        key=lambda r: (
            float(r.get("relative_size_score", 0)),
            float(r.get("visibility_score", 0)),
            float(r.get("segmentation_suitability_score", 0)),
        ),
        reverse=True,
    )

    top_candidates = ranked[:TOP_K]

    print(f"\nTop {TOP_K} for {view_name}:")
    for i, r in enumerate(top_candidates, 1):
        print(
            i,
            r["image_id"],
            "frame=", r["frame"],
            "size=", r["relative_size_score"],
            "visibility=", r["visibility_score"],
            "sam=", r["segmentation_suitability_score"],
            "occlusion=", r["occlusion_level"],
        )

    return {
        "view_name": view_name,
        "image_dir": str(image_dir),
        "all_results": results,
        "top_candidates": top_candidates,
    }


def main():
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("Please set GEMINI_API_KEY first.")

    meta = load_meta()
    target_prompt = meta["selected_target_prompt"]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    if OUT_PATH.exists():
        print(f"[RESUME] Loading existing output: {OUT_PATH}")
        with open(OUT_PATH, "r") as f:
            output = json.load(f)
    else:
        output = {
            "root": str(ROOT),
            "target_prompt": target_prompt,
            "top_k": TOP_K,
            "views": {},
        }

    for view_name, image_dir in VIEWS.items():
        if view_name in output["views"]:
            print(f"[SKIP] {view_name} already exists in output.")
            continue

        output["views"][view_name] = rank_one_view(
            client=client,
            view_name=view_name,
            image_dir=image_dir,
            target_prompt=target_prompt,
        )

        # 每跑完一个 view 就立刻保存，避免第二个 view 崩了导致前面结果丢失
        with open(OUT_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[SAVED] Partial result saved after {view_name}: {OUT_PATH}")
        time.sleep(10)

    print("\nDone.")
    print("Saved Gemini ranking to:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()

