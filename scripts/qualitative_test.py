"""
SAM3 Qualitative Test — supports both SAM3 and SAM3.1.

Tests text prompt detection + propagation on a synthetic video.
Checkpoints are auto-downloaded from HuggingFace.

Usage:
  python scripts/qualitative_test.py                    # SAM 3.1 default
  python scripts/qualitative_test.py --version sam3     # SAM 3
  python scripts/qualitative_test.py --video /path/to/video.mp4
"""

import argparse
import getpass
import os
import shutil

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image as PIL_Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image, ImageDraw


OUTPUT_DIR = "/tmp/sam3_qualitative_test"

MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 64, 128),
    (128, 255, 0),
    (64, 128, 255),
    (255, 200, 0),
    (0, 200, 128),
    (200, 0, 128),
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 0),
    (0, 128, 128),
]


def extract_frames(video_path, output_dir):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        n = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        print(f"Using existing {n} frames in {output_dir}")
        return n
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames to {output_dir}")
    return idx


def synthesize_video(out_dir, num_objects=5, n_frames=30, width=1024, height=1024):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    colors = [
        tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(num_objects)
    ]
    positions = [
        [
            float(np.random.randint(80, width - 80)),
            float(np.random.randint(80, height - 80)),
        ]
        for _ in range(num_objects)
    ]
    velocities = [
        [np.random.choice([-1, 1]) * 15, np.random.choice([-1, 1]) * 15]
        for _ in range(num_objects)
    ]
    for i in range(n_frames):
        img = PIL_Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for j in range(num_objects):
            x, y = positions[j]
            draw.ellipse([(x - 50, y - 50), (x + 50, y + 50)], fill=colors[j])
            vx, vy = velocities[j]
            positions[j] = [
                np.clip(x + vx, 50, width - 50),
                np.clip(y + vy, 50, height - 50),
            ]
            if x < 50 or x > width - 50:
                velocities[j][0] *= -1
            if y < 50 or y > height - 50:
                velocities[j][1] *= -1
        img.save(os.path.join(out_dir, f"{i:05d}.jpg"))
    print(f"Generated {n_frames} synthetic frames with {num_objects} circles")
    return n_frames


def load_frame(frame_dir, frame_idx):
    return cv2.cvtColor(
        cv2.imread(os.path.join(frame_dir, f"{frame_idx:05d}.jpg")),
        cv2.COLOR_BGR2RGB,
    )


def render_overlay(frame_rgb, masks_by_obj, alpha=0.4):
    overlay = frame_rgb.copy().astype(np.float32)
    for obj_id, mask in sorted(masks_by_obj.items()):
        color = MASK_COLORS[obj_id % len(MASK_COLORS)]
        mask_bool = mask.astype(bool)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )
    return overlay.astype(np.uint8)


def save_overlay(frame_rgb, masks_by_obj, output_path, title=None):
    overlay = render_overlay(frame_rgb, masks_by_obj)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=100)
    ax.imshow(overlay)
    for obj_id, mask in sorted(masks_by_obj.items()):
        mask_bool = mask.astype(bool)
        if mask_bool.any():
            ys, xs = np.where(mask_bool)
            cx, cy = int(xs.mean()), int(ys.mean())
            color_rgb = MASK_COLORS[obj_id % len(MASK_COLORS)]
            facecolor = (color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255)
            ax.text(
                cx,
                cy,
                str(obj_id),
                color="white",
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=facecolor, alpha=0.8),
            )
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def collect_propagation(model, session_id):
    mask_dict = {}
    for response in model.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        frame_idx = response.get("frame_index")
        if frame_idx is None:
            continue
        outputs = response.get("outputs", {})
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")
        if binary_masks is None:
            mask_dict[frame_idx] = {}
            continue
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.cpu().numpy()
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.cpu().numpy()
        masks = {}
        for i, oid in enumerate(obj_ids):
            m = binary_masks[i]
            if m.ndim == 3:
                m = m[0]
            masks[int(oid)] = m
        mask_dict[frame_idx] = masks
    torch.cuda.synchronize()
    return mask_dict


def main():
    parser = argparse.ArgumentParser(description="SAM3 Qualitative Test")
    parser.add_argument(
        "--version", type=str, default="sam3.1", choices=["sam3", "sam3.1"]
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file. If not provided, generates synthetic video.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (auto-downloads from HuggingFace if not provided)",
    )
    parser.add_argument(
        "--text_prompt", type=str, default="circle", help="Text prompt for detection"
    )
    parser.add_argument(
        "--n_frames", type=int, default=30, help="Number of frames for synthetic video"
    )
    args = parser.parse_args()

    username = getpass.getuser()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{username}"
    os.environ["USE_PERFLIB"] = "1"
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Prepare video frames
    frame_dir = "/tmp/sam3_qualitative_frames"
    if args.video:
        n_frames = extract_frames(args.video, frame_dir)
    else:
        n_frames = synthesize_video(frame_dir, n_frames=args.n_frames)

    img = load_frame(frame_dir, 0)
    img_h, img_w = img.shape[:2]
    print(f"Video: {img_w}x{img_h}, {n_frames} frames")

    # Build model
    from sam3 import build_sam3_predictor

    print(f"\nBuilding {args.version} model...")
    build_kwargs = dict(version=args.version, compile=False, async_loading_frames=False)
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = args.checkpoint
    model = build_sam3_predictor(**build_kwargs)

    # Start session
    response = model.handle_request(
        {"type": "start_session", "resource_path": frame_dir}
    )
    session_id = response["session_id"]
    print(f"Session: {session_id}")

    # Test: text prompt -> propagate
    out_dir = os.path.join(OUTPUT_DIR, f"{args.version}_text_{args.text_prompt}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    print(f"\nTest: text prompt '{args.text_prompt}' -> propagate")
    model.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": args.text_prompt,
        }
    )

    mask_dict = collect_propagation(model, session_id)
    print(f"Propagated through {len(mask_dict)} frames")

    # Save overlays
    saved = 0
    for frame_idx in sorted(mask_dict.keys()):
        if frame_idx % 5 != 0:
            continue
        masks = mask_dict[frame_idx]
        if not masks:
            continue
        frame_rgb = load_frame(frame_dir, frame_idx)
        save_overlay(
            frame_rgb,
            masks,
            os.path.join(out_dir, f"frame_{frame_idx:05d}.png"),
            title=f"{args.version} | frame {frame_idx} | {len(masks)} objects",
        )
        saved += 1

    # Print results
    frame0 = mask_dict.get(0, {})
    print(f"\nDetected {len(frame0)} objects on frame 0:")
    for obj_id, mask in sorted(frame0.items()):
        mask_bool = mask.astype(bool)
        n_pixels = int(mask_bool.sum())
        if mask_bool.any():
            ys, xs = np.where(mask_bool)
            print(
                f"  obj {obj_id}: centroid ({int(xs.mean())}, {int(ys.mean())}), {n_pixels} pixels"
            )

    print(f"\nSaved {saved} overlay images to {out_dir}")
    print(
        "QUALITATIVE TEST PASSED"
        if len(frame0) > 0
        else "WARNING: No objects detected!"
    )

    # Cleanup
    if not args.video:
        shutil.rmtree(frame_dir)


if __name__ == "__main__":
    main()
