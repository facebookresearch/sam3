from pathlib import Path
import json

TAKE_ID = "0a22a1c1-844c-4f62-8eeb-f16eee62357f"

ANN_PATH = Path(
    "/home/users/ntu/gwang016/scratch/datasets/Ego-Exo4D-Relation-Test/"
    "extracted/work/yuqian_fu/Ego/data_segswap_test"
) / TAKE_ID / "annotation.json"

data = json.load(open(ANN_PATH))

print("=" * 80)
print("annotation path:", ANN_PATH)
print("top-level type:", type(data))
print("=" * 80)

if isinstance(data, dict):
    print("top-level keys:")
    for k in list(data.keys())[:100]:
        print(" ", repr(k), "->", type(data[k]))

elif isinstance(data, list):
    print("list length:", len(data))
    print("first item type:", type(data[0]) if data else None)
    if data:
        print("first item keys:", data[0].keys() if isinstance(data[0], dict) else None)

print("\n" + "=" * 80)
print("Recursive key summary")
print("=" * 80)

def walk(x, prefix="root", depth=0, max_depth=4):
    if depth > max_depth:
        return

    if isinstance(x, dict):
        for k, v in list(x.items())[:30]:
            p = f"{prefix}.{k}"
            print("  " * depth + f"{p}: {type(v).__name__}")
            walk(v, p, depth + 1, max_depth)

    elif isinstance(x, list):
        print("  " * depth + f"{prefix}: list len={len(x)}")
        if x:
            walk(x[0], prefix + "[0]", depth + 1, max_depth)

walk(data, max_depth=5)

print("\n" + "=" * 80)
print("Search important keywords")
print("=" * 80)

keywords = [
    "mask", "segmentation", "segment", "rle", "bbox", "box",
    "label", "text", "prompt", "object", "category",
    "aria", "cam", "frame", "annotation"
]

def search_keys(x, prefix="root"):
    if isinstance(x, dict):
        for k, v in x.items():
            kl = str(k).lower()
            if any(q in kl for q in keywords):
                print(f"{prefix}.{k}: type={type(v).__name__}")
                if isinstance(v, (str, int, float)):
                    print("   value:", repr(v)[:200])
                elif isinstance(v, list):
                    print("   list len:", len(v))
                    if v and isinstance(v[0], (str, int, float)):
                        print("   first:", repr(v[0])[:200])
                elif isinstance(v, dict):
                    print("   dict keys sample:", list(v.keys())[:10])
            search_keys(v, f"{prefix}.{k}")

    elif isinstance(x, list):
        for i, v in enumerate(x[:5]):
            search_keys(v, f"{prefix}[{i}]")

search_keys(data)
