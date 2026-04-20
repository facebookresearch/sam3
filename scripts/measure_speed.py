"""
SAM3 Speed Test — supports both SAM3 and SAM3.1 (multiplex).

Generates synthetic video with moving circles, runs text-prompt detection
+ propagation, and measures FPS. Checkpoints are auto-downloaded from
HuggingFace if not provided.

Usage:
  # SAM 3.1 (default, auto-downloads from HuggingFace):
  python scripts/measure_speed.py

  # SAM 3 (non-multiplex):
  python scripts/measure_speed.py --version sam3

  # Custom settings:
  python scripts/measure_speed.py --num_objects 32 --n_frames 100 --no-compile
  python scripts/measure_speed.py --version sam3.1 --compile --num_objects 5
"""

import argparse
import getpass
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image, ImageDraw


def max_memory_allocated():
    max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    max_memory_allocated_percentage = int(
        100 * (max_memory_allocated_bytes / total_memory)
    )
    max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
    print(
        f"max_memory_allocated_bytes: {max_memory_allocated_bytes}MiB or {max_memory_allocated_percentage}%"
    )


def synthesize_video_data(
    num_objects: int,
    out_dir: str,
    radius: int,
    speed: int,
    width: int,
    height: int,
    n_frames: int,
):
    circle_colors = [
        tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(num_objects)
    ]

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    positions = []
    velocities = []
    for _ in range(num_objects):
        px = float(np.random.randint(radius, width - radius))
        py = float(np.random.randint(radius, height - radius))
        vx = np.random.choice([-1, 1]) * speed
        vy = np.random.choice([-1, 1]) * speed
        positions.append([px, py])
        velocities.append([vx, vy])

    print(f"Generate {n_frames} frames with {num_objects} objects")
    for i in range(n_frames):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for obj_idx in range(num_objects):
            x, y = positions[obj_idx]
            rx, ry = round(x), round(y)
            draw.ellipse(
                [(rx - radius, ry - radius), (rx + radius, ry + radius)],
                fill=circle_colors[obj_idx],
            )
            vx, vy = velocities[obj_idx]
            x += vx
            y += vy
            positions[obj_idx] = [
                np.clip(x, radius, width - radius),
                np.clip(y, radius, height - radius),
            ]
            if x - radius < 0 or x + radius > width:
                vx *= -1
            if y - radius < 0 or y + radius > height:
                vy *= -1
            velocities[obj_idx] = [vx, vy]

        img.save(os.path.join(out_dir, f"{i:03d}.jpg"))


def profiler_runner(fn, profile_save_dir=None, profile_end_frame=-1, *args, **kwargs):
    if profile_save_dir is None:
        profile_save_dir = os.path.expanduser("~/traces")

    os.environ["ENABLE_PROFILING"] = "1"
    os.environ["PROFILE_SAVE_DIR"] = profile_save_dir
    if profile_end_frame >= 0:
        os.environ["PROFILE_END_FRAME"] = str(profile_end_frame)

    print(f"Profiling enabled. Traces will be saved to: {profile_save_dir}")
    if profile_end_frame >= 0:
        print(f"Profiling will stop at frame: {profile_end_frame}")

    try:
        result = fn(*args, **kwargs)
    finally:
        os.environ.pop("ENABLE_PROFILING", None)
        os.environ.pop("PROFILE_SAVE_DIR", None)
        os.environ.pop("PROFILE_END_FRAME", None)

    return result


def main_loop(model_wrapper, session_id, text_prompt):
    model_wrapper.handle_request({"type": "reset_session", "session_id": session_id})
    model_wrapper.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt,
        }
    )

    t0 = time.perf_counter()
    frame_count = 0
    for _response in model_wrapper.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        frame_count += 1
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if frame_count > 0:
        return frame_count / (t1 - t0)
    return -1


def run_test(
    version: str,
    profile: bool,
    video_dir: str,
    num_objects: int,
    radius: int,
    speed: int,
    width: int,
    height: int,
    n_frames: int,
    synthesize_data: bool = True,
    profile_save_dir: str = None,
    profile_end_frame: int = -1,
    do_compile: bool = True,
    checkpoint_path: str = None,
) -> float:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if synthesize_data:
        synthesize_video_data(
            num_objects=num_objects,
            out_dir=video_dir,
            radius=radius,
            speed=speed,
            width=width,
            height=height,
            n_frames=n_frames,
        )

    from sam3 import build_sam3_predictor

    print(f"Building {version} model...")
    build_kwargs = dict(
        version=version,
        compile=do_compile,
        async_loading_frames=False,
    )
    if checkpoint_path:
        build_kwargs["checkpoint_path"] = checkpoint_path
    if version == "sam3.1":
        build_kwargs["warm_up"] = do_compile
        build_kwargs["max_num_objects"] = num_objects

    model_wrapper = build_sam3_predictor(**build_kwargs)

    # Initialize session
    response = model_wrapper.handle_request(
        {"type": "start_session", "resource_path": video_dir}
    )
    session_id = response["session_id"]

    print("\nWarm-up round.")
    NUM_WARMUP_TRIES = 3
    fps = 0
    for _ in range(NUM_WARMUP_TRIES):
        fps = max(
            main_loop(
                model_wrapper=model_wrapper, session_id=session_id, text_prompt="circle"
            ),
            fps,
        )

    print("\nProfile round.")
    if profile:
        profiler_runner(
            main_loop,
            profile_save_dir=profile_save_dir or os.path.expanduser("~/traces"),
            profile_end_frame=profile_end_frame,
            model_wrapper=model_wrapper,
            session_id=session_id,
            text_prompt="circle",
        )
    else:
        fps = max(
            main_loop(
                model_wrapper=model_wrapper, session_id=session_id, text_prompt="circle"
            ),
            fps,
        )

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\nTiming round {i + 1} ")
        fps = max(
            main_loop(
                model_wrapper=model_wrapper, session_id=session_id, text_prompt="circle"
            ),
            fps,
        )
        print(f"Frames per second (FPS): {fps:.2f}")
        max_memory_allocated()

    if synthesize_data:
        print("\nDeleting temporary video directory.")
        shutil.rmtree(video_dir)

    return fps


if __name__ == "__main__":
    username = getpass.getuser()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{username}"
    os.environ["USE_PERFLIB"] = "1"

    parser = argparse.ArgumentParser(description="SAM3 Speed Test")
    parser.add_argument(
        "--version",
        type=str,
        default="sam3.1",
        choices=["sam3", "sam3.1"],
        help="Model version (default: sam3.1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (auto-downloads from HuggingFace if not provided)",
    )
    parser.add_argument(
        "--video_dir", type=str, default="/tmp/segment-anything-3/synth_video"
    )
    parser.add_argument("--num_objects", type=int, default=5)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile",
    )
    parser.add_argument("--no-torch-profiling", action="store_false", dest="profile")
    parser.add_argument(
        "--no-data-synthesis", action="store_false", dest="synthesize_data"
    )
    parser.add_argument("--profile-save-dir", type=str, default=None)
    parser.add_argument("--profile-end-frame", type=int, default=-1)

    args = parser.parse_args()

    run_test(
        version=args.version,
        profile=args.profile,
        num_objects=args.num_objects,
        video_dir=args.video_dir,
        radius=args.radius,
        speed=args.speed,
        width=args.width,
        height=args.height,
        n_frames=args.n_frames,
        synthesize_data=args.synthesize_data,
        profile_save_dir=args.profile_save_dir,
        profile_end_frame=args.profile_end_frame,
        do_compile=args.compile,
        checkpoint_path=args.checkpoint,
    )
