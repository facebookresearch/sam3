# Temporal Ego-Exo cross-view pipeline

This branch separates the workflow into a deterministic server stage and a local Codex analysis stage.

## 1. Server: generate one test case

Install the required packages in the active environment:

```bash
pip install pycocotools pillow numpy
```

Run one case first:

```bash
python generate_temporal_cross_view_assets.py \
  --data-root /scratch/datasets/Ego-Exo4D-Relation-Test \
  --output-root /scratch/users/ntu/gwang016/temporal_cross_view_assets \
  --window-sizes 16 \
  --window-stride 8 \
  --target-sample-every 1 \
  --max-cases 1 \
  --overwrite
```

Check:

- `batch_summary.json`
- `<case_id>/metadata.json`
- `<case_id>/source_best_mask_overlay.png`
- `<case_id>/source_mask_area_ranking.csv`
- `<case_id>/temporal_window_index.json`
- several generated target contact sheets

The source frame is selected from decoded `annotation.json` masks by maximum mask area ratio. Each `aria*` view also keeps its own best frame under `source_view_bests/`.

## 2. Server: generate at least six cases

```bash
python generate_temporal_cross_view_assets.py \
  --data-root /scratch/datasets/Ego-Exo4D-Relation-Test \
  --output-root /scratch/users/ntu/gwang016/temporal_cross_view_assets \
  --window-sizes 16 \
  --window-stride 8 \
  --target-sample-every 1 \
  --max-cases 6 \
  --overwrite
```

For multi-scale windows:

```bash
--window-sizes 10 16 20 --window-stride 5
```

This creates many more sheets. If the output is too large, first increase `--target-sample-every` rather than using a very small arbitrary `--max-windows-per-cam`. A positive `--max-windows-per-cam` uniformly keeps windows and may miss short events.

The script calculates the median frame-id gap for each camera and splits the sequence when a gap exceeds `median_gap * --max-gap-factor`. Therefore no contact sheet crosses a large missing-frame boundary.

## 3. Download assets to the local machine

Download the complete output root. Paths in `temporal_window_index.json` are relative to each case directory, so the folders remain portable.

Example:

```bash
rsync -avz <server>:/scratch/users/ntu/gwang016/temporal_cross_view_assets/ \
  ./temporal_cross_view_assets/
```

## 4. Local Codex analysis

For each case, ask Codex to follow:

```text
Read prompt_for_temporal_analysis.md, metadata.json, temporal_window_index.json,
source_best_mask_overlay.png, and all referenced temporal contact sheets.
Evaluate complete windows rather than isolated frames. Write strict JSON to
temporal_analysis_result.json. Compare the target's normalized spatial region,
position drift, scale change, visibility, occlusion, identity consistency, and
segmentation suitability. Mark uncertain/failed rather than forcing a result.
```

The prompt requires:

- best camera and continuous frame range;
- top alternatives and rejected ranges;
- temporal scores;
- 3–5 representative frames;
- normalized target region estimates;
- regional comparison against rejected windows;
- uncertainty and an identity-specific SAM3 prompt.

These region boxes are AI-estimated localization evidence, not masks.

## 5. Render one case after Codex writes JSON

Copy `render_temporal_analysis_results.py` into the local working directory or run it from the repository:

```bash
python render_temporal_analysis_results.py \
  --case-dir ./temporal_cross_view_assets/<case_id> \
  --analysis-json ./temporal_cross_view_assets/<case_id>/temporal_analysis_result.json \
  --overwrite
```

Outputs:

```text
analysis_outputs/
  selected_best_segment_contact_sheet.jpg
  alternative_segment_01.jpg
  rejected_segment_01.jpg
  selected_vs_rejected_region_comparison.jpg
  method_feasibility_report.md
  render_summary.json
```

## 6. Render all analyzed cases

After Codex has written `temporal_analysis_result.json` in each of the six case folders:

```bash
python render_temporal_analysis_results.py \
  --assets-root ./temporal_cross_view_assets \
  --overwrite
```

Cases without an analysis JSON are skipped and recorded in `render_batch_summary.json`.

## Notes

- The server script never invents a third-view mask.
- A single clear target frame cannot make an unstable window win.
- `window_id` should always be included in the Codex JSON because it resolves overlapping windows unambiguously.
- If a target is transparent, too small, heavily occluded, or confused with similar objects, use `status: uncertain` or `status: failed`.
