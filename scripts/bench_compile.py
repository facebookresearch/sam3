# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 encoder torch.compile A/B benchmark — production-faithful.

Builds ``Sam3VideoPredictorMultiGPU`` with the EXACT constructor arguments that
scene_service's ``Sam3DenseTrackingModel.load_model`` uses (see
``fbcode/genai/media_editing/project/sam_stateful/inferencer/dense_tracking.py``),
plus the ``sam3_v4.pt`` production checkpoint, with ``compile`` toggled on/off.
It then measures steady-state propagation throughput (FPS), per-frame latency,
and peak GPU memory on a synthetic moving-circles video.

This isolates the win from ``torch.compile`` (dominated by the ViT image encoder,
which is the ~1.3s hotspot) on the same model + GPU class the IPNext tenant serves.

Usage:
  buck run @fbcode//mode/opt \
    fbcode//deeplearning/projects/sam3_release:bench_compile -- \
      --checkpoint /tmp/sam3_v4.pt --num_objects 5 --n_frames 50 --compile
  # eager baseline:
  buck run @fbcode//mode/opt \
    fbcode//deeplearning/projects/sam3_release:bench_compile -- \
      --checkpoint /tmp/sam3_v4.pt --num_objects 5 --n_frames 50 --no-compile
"""

import argparse
import getpass
import os
import time

import torch

# Reuse the author's synthetic-video + timing helpers (same package, via :sam3).
from scripts.measure_speed import main_loop, max_memory_allocated, synthesize_video_data


def build_prod_predictor(checkpoint_path: str, do_compile: bool):
    """Construct the predictor identically to Sam3DenseTrackingModel.load_model."""
    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

    return Sam3VideoPredictorMultiGPU(
        checkpoint_path=checkpoint_path,
        bpe_path=None,
        has_presence_token=True,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=False,
        apply_temporal_disambiguation=True,
        async_loading_frames=False,
        video_loader_type="cv2",
        compile=do_compile,
        gpus_to_use=[0],
    )


def time_scene_frame(
    model_wrapper,
    image_path: str,
    labels: list[str],
    n_iters: int = 12,
    warmup_iters: int = 5,
) -> tuple[float, float]:
    """Replicate scene_service's per-frame SAM3 sequence (Sam3Client.getMasksStream).

    For each ingested frame, subscribe_scene_objects drives exactly:
        start_session(single image) -> add_prompt(label) x K -> close/reset
    i.e. a FRESH session per frame (image re-encoded every frame, no cross-frame
    reuse) with the K subscription labels sharing that one image encode. This is
    the real production unit -- NOT video propagate_in_video. Returns (median, min)
    ms per frame.

    Runs warmup_iters discarded iterations first so the single-image detection path
    is compiled/settled (the propagate warm-up does NOT exercise this path), then
    times n_iters."""

    def _one_frame() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        resp = model_wrapper.handle_request(
            {"type": "start_session", "resource_path": image_path}
        )
        sid = resp["session_id"]
        for label in labels:
            model_wrapper.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": label,
                }
            )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        model_wrapper.handle_request({"type": "reset_session", "session_id": sid})
        return dt

    for _ in range(warmup_iters):
        _one_frame()
    times_ms = [_one_frame() for _ in range(n_iters)]
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0]  # median, min


def run(
    checkpoint_path: str,
    num_objects: int,
    n_frames: int,
    radius: int,
    speed: int,
    width: int,
    height: int,
    video_dir: str,
    do_compile: bool,
    full_warmup: bool = False,
    native_warmup: bool = False,
) -> float:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    synthesize_video_data(
        num_objects=num_objects,
        out_dir=video_dir,
        radius=radius,
        speed=speed,
        width=width,
        height=height,
        n_frames=n_frames,
    )

    mode = "COMPILED" if do_compile else "EAGER"
    print(f"\n=== Building {mode} predictor from {checkpoint_path} ===")
    model_wrapper = build_prod_predictor(checkpoint_path, do_compile)

    # --native-warmup mirrors the production Sam3DenseTrackingModel.warmup path:
    # install the compile wrappers directly via _compile_model() (NOT via a video
    # propagate) and skip the propagate rounds entirely, so the only thing that
    # compiles the served graphs is the single-image add_prompt warm-up inside
    # time_scene_frame. Run under TORCH_LOGS=recompiles to prove no recompile
    # fires on the TIMED single-image iters (all recompiles absorbed in warmup).
    best_fps = 0.0
    if do_compile and native_warmup:
        print("Native warmup: _compile_model() direct (no propagate)...")
        model_wrapper.model._compile_model()
    else:
        response = model_wrapper.handle_request(
            {"type": "start_session", "resource_path": video_dir}
        )
        session_id = response["session_id"]

        if do_compile and full_warmup:
            try:
                print("Warming up torch.compile (varying object counts)...")
                model_wrapper.model.warm_up_compilation()
            except Exception as e:
                print(f"warm_up_compilation() failed ({e!r}); relying on lazy compile.")

        print("Warm-up rounds...")
        fps = 0.0
        for _ in range(3):
            fps = max(main_loop(model_wrapper, session_id, "circle"), fps)

        print("Timing rounds...")
        for i in range(10):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            f = main_loop(model_wrapper, session_id, "circle")
            best_fps = max(best_fps, f)
            print(f"  round {i + 1}: {f:.2f} FPS")
        max_memory_allocated()

    #
    # scene_service per-frame SAM3 cost: start_session(single image) + K add_prompt.
    image_path = os.path.join(video_dir, "000.jpg")
    scene_ms = {}
    for k in (1, 3):
        labels = ["circle", "square", "triangle"][:k]
        med, mn = time_scene_frame(model_wrapper, image_path, labels)
        scene_ms[k] = med
        print(f"  scene per-frame K={k} labels: {med:.2f} ms median ({mn:.2f} ms min)")

    per_frame_ms = 1000.0 / best_fps if best_fps > 0 else float("nan")
    print(
        f"\n=== RESULT {mode}: "
        f"scene/frame K=1 {scene_ms[1]:.2f} ms | K=3 {scene_ms[3]:.2f} ms | "
        f"propagate {best_fps:.2f} FPS ({per_frame_ms:.2f} ms/frame) | "
        f"num_objects={num_objects} {width}x{height} ==="
    )
    return best_fps


def main() -> None:
    username = getpass.getuser()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{username}"
    os.environ["USE_PERFLIB"] = "1"

    parser = argparse.ArgumentParser(
        description="SAM3 production-model torch.compile A/B benchmark"
    )
    parser.add_argument("--checkpoint", type=str, default="/tmp/sam3_v4.pt")
    parser.add_argument(
        "--video_dir", type=str, default="/tmp/sam3_bench_compile/synth_video"
    )
    parser.add_argument("--num_objects", type=int, default=5)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the model; use --no-compile for the eager baseline",
    )
    parser.add_argument(
        "--full-warmup",
        action="store_true",
        help="run prod warm_up_compilation (object-count sweep) instead of lazy compile",
    )
    parser.add_argument(
        "--native-warmup",
        action="store_true",
        help="mirror prod Sam3DenseTrackingModel.warmup: _compile_model() direct + "
        "single-image warm only (no propagate); validates 0 serve-time recompiles",
    )

    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        num_objects=args.num_objects,
        n_frames=args.n_frames,
        radius=args.radius,
        speed=args.speed,
        width=args.width,
        height=args.height,
        video_dir=args.video_dir,
        do_compile=args.compile,
        full_warmup=args.full_warmup,
        native_warmup=args.native_warmup,
    )


if __name__ == "__main__":
    main()
