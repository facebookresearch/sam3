#!/usr/bin/env python3
"""Merge a head-only SAM3 training checkpoint into a sam3.pt the demo can load.

The trainer saves module names such as ``transformer.*``. ``build_sam3_image_model``
loads keys that start with ``detector.``. Backbone weights stay on the original
checkpoint because that run kept them frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

TRAINABLE_PREFIXES = (
    "transformer.",
    "segmentation_head.",
    "dot_prod_scoring.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/home/ygz/Projects/sam3/sam3.pt"),
        help="Original sam3.pt with detector.* keys",
    )
    parser.add_argument(
        "--train-ckpt",
        type=Path,
        required=True,
        help="outputs/tank_head_ft/checkpoints/checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ygz/Projects/sam3/outputs/tank_head_ft/sam3_tank_head.pt"),
    )
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    trained_blob = torch.load(args.train_ckpt, map_location="cpu", weights_only=True)
    trained = trained_blob["model"] if isinstance(trained_blob, dict) and "model" in trained_blob else trained_blob

    replaced = 0
    missing = []
    for name, value in trained.items():
        if not name.startswith(TRAINABLE_PREFIXES):
            continue
        key = "detector." + name
        if key not in base:
            missing.append(key)
            continue
        base[key] = value.detach().to(dtype=base[key].dtype, device="cpu").contiguous()
        replaced += 1

    if replaced == 0:
        raise SystemExit("No detector head weights were copied. Check --train-ckpt.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(f"Wrote {args.output} ({replaced} tensors updated)")
    if missing:
        print(f"{len(missing)} trainable tensors were not in the base checkpoint")
        for key in missing[:20]:
            print(f"  {key}")


if __name__ == "__main__":
    main()
