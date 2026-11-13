# Camera-relative 20%–30% temporal window pipeline

Use `generate_temporal_cross_view_assets_ratio.py` when the target segment must cover approximately 20%–30% of the complete sampled third-person camera sequence.

The original fixed-size generator is retained for backward compatibility. The ratio generator imports its stable source-mask and contact-sheet utilities, but replaces target window construction with camera-relative windows.

## Pull the branch

```bash
git switch feature/temporal-cross-view-assets
git pull origin feature/temporal-cross-view-assets
```

## Test one case first

```bash
python generate_temporal_cross_view_assets_ratio.py \
  --data-root /scratch/datasets/Ego-Exo4D-Relation-Test \
  --output-root /scratch/users/ntu/gwang016/temporal_cross_view_assets_ratio \
  --window-ratios 0.20 0.25 0.30 \
  --window-stride-ratio 0.05 \
  --target-sample-every 1 \
  --sheet-columns 6 \
  --max-windows-per-cam 0 \
  --max-cases 1 \
  --overwrite
```

After generation, inspect:

- `batch_summary.json`
- `<case_id>/metadata.json`
- `<case_id>/temporal_window_index.json`
- `<case_id>/prompt_for_temporal_analysis.md`
- `<case_id>/target_temporal_windows/<cam>/`

Each candidate window records:

- `requested_video_ratio`
- `actual_sampled_frame_ratio`
- `actual_frame_span_ratio`
- `window_size_sampled_frames`
- exact `frame_ids`
- continuity and frame-gap information

The primary duration definition is `actual_sampled_frame_ratio`: the number of frames in the window divided by the complete sampled frame count of that camera. `actual_frame_span_ratio` is retained as a secondary diagnostic because missing frame IDs can make timestamp-span ratios less reliable.

## Run six cases

After the one-case output and Codex analysis are correct:

```bash
python generate_temporal_cross_view_assets_ratio.py \
  --data-root /scratch/datasets/Ego-Exo4D-Relation-Test \
  --output-root /scratch/users/ntu/gwang016/temporal_cross_view_assets_ratio \
  --window-ratios 0.20 0.25 0.30 \
  --window-stride-ratio 0.05 \
  --target-sample-every 1 \
  --sheet-columns 6 \
  --max-windows-per-cam 0 \
  --max-cases 6 \
  --overwrite
```

## Interpretation

The generated Codex prompt asks for the relatively best complete 20%–30% window. A single clear frame must not make an otherwise unstable window win. The third-person boxes are AI-estimated localization regions only; they are not masks.

Large missing-frame gaps still split the sequence into contiguous runs. If no run is long enough to contain a whole-camera-relative 20%–30% window, the unavailable specification is recorded in `unavailable_window_specs` rather than generating a window across the gap.
