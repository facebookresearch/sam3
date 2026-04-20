# Release Notes

## SAM 3.1 — March 27, 2026

SAM 3.1 introduces **Object Multiplex**, a shared-memory approach for joint multi-object tracking that is significantly faster without sacrificing accuracy. This release also includes new model checkpoints and optimized inference.

### Object Multiplex

SAM 3's video pipeline processes each tracked object independently, which scales linearly with the number of objects. Object Multiplex groups objects into fixed-capacity buckets and processes them jointly, drastically reducing redundant computation. For technical details, see Appendix H (Object Multiplex) in the [SAM 3 paper](https://arxiv.org/abs/2511.16719).

<p align="center">
  <img src="assets/sam3.1_diagram.png" width="720" />
</p>

#### Key Improvements
- **~7x speedup** at 128 objects on a single H100 GPU compared to the SAM 3 November 2025 release
- Inference optimizations that significantly improve multi-object tracking efficiency:
  - Reduced CPU-GPU synchronization in detection-tracker association and other heuristics
  - Enhanced `torch.compile` support with improved operation fusion
  - Batched postprocessing and vision encoder to increase GPU utilization
- Mixed results on SA-Co/VEval video benchmarks, with notable improvement on YT-Temporal-1B (+2.1 cgF1)
- Improved VOS performance on 6 out of 7 benchmarks, including +2.0 on the challenging MOSEv2

#### Inference Efficiency

<p align="center">
  <img src="assets/sam3.1_efficiency.png" width="720" />
</p>

#### Video PCS with Text Prompt

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="3" style="border-right: 2px solid #ddd; padding: 12px 16px">Model</th>
      <th colspan="6" style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">SA-Co/VEval benchmark test split</th>
      <th colspan="4" style="text-align: center; padding: 10px 16px">Public benchmarks</th>
    </tr>
    <tr>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">SA-V</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">YT-Temporal-1B</th>
      <th colspan="2" style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">SmartGlasses</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">LVVIS</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">BURST</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">YTVIS21</th>
      <th style="text-align: center; padding: 10px 16px">OVIS</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 10px 16px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">pHOTA</th>
      <th style="text-align: center; padding: 10px 16px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">pHOTA</th>
      <th style="text-align: center; padding: 10px 16px">cgF1</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">pHOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">test mAP</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">test HOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">val mAP</th>
      <th style="text-align: center; padding: 10px 16px">val mAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3</td>
      <td style="text-align: center; padding: 10px 16px">30.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">58.0</td>
      <td style="text-align: center; padding: 10px 16px">50.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">69.9</td>
      <td style="text-align: center; padding: 10px 16px">36.4</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">63.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">36.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">44.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">57.4</td>
      <td style="text-align: center; padding: 10px 16px">60.5</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3.1</td>
      <td style="text-align: center; padding: 10px 16px">30.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">58.7</td>
      <td style="text-align: center; padding: 10px 16px">52.9</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">70.7</td>
      <td style="text-align: center; padding: 10px 16px">36.3</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 16px">64.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">34.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">43.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">56.6</td>
      <td style="text-align: center; padding: 10px 16px">61.5</td>
    </tr>
  </tbody>
</table>

</div>

#### Video Object Segmentation (VOS)

<div align="center">
<table style="min-width: 60%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 2px solid #ddd; padding: 12px 16px">Model</th>
      <th colspan="5" style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">J&amp;F</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">G</th>
      <th style="text-align: center; padding: 10px 16px">J&amp;Ḟ</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 10px 16px">MOSEv1 val</th>
      <th style="text-align: center; padding: 10px 16px">DAVIS17 val</th>
      <th style="text-align: center; padding: 10px 16px">LVOSv2 val</th>
      <th style="text-align: center; padding: 10px 16px">SA-V val</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">SA-V test</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">YTVOS19 val</th>
      <th style="text-align: center; padding: 10px 16px">MOSEv2 val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3</td>
      <td style="text-align: center; padding: 10px 16px">78.4</td>
      <td style="text-align: center; padding: 10px 16px">92.2</td>
      <td style="text-align: center; padding: 10px 16px">88.5</td>
      <td style="text-align: center; padding: 10px 16px">83.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">84.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">89.7</td>
      <td style="text-align: center; padding: 10px 16px">60.3</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 16px">SAM 3.1</td>
      <td style="text-align: center; padding: 10px 16px">79.6</td>
      <td style="text-align: center; padding: 10px 16px">92.7</td>
      <td style="text-align: center; padding: 10px 16px">89.2</td>
      <td style="text-align: center; padding: 10px 16px">83.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">85.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 16px">89.3</td>
      <td style="text-align: center; padding: 10px 16px">62.3</td>
    </tr>
  </tbody>
</table>
</div>

### New Checkpoints

The SAM 3.1 checkpoints are available on the [Hugging Face repo](https://huggingface.co/facebook/sam3.1). See [Getting Started](README.md#getting-started) for download and authentication instructions.

### Notebooks

- [`sam3.1_video_predictor_example.ipynb`](examples/sam3.1_video_predictor_example.ipynb): Demonstrates how to use SAM 3.1 with Object Multiplex for video segmentation and dense tracking with text and point prompts.

### Contributors

[Arpit Kalla](https://github.com/arpitkalla), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Christian Puhrsch](https://github.com/cpuhrsch), [Ho Kei Cheng](https://hkchengrex.com/), [Joseph Greer](https://scholar.google.com/citations?user=guL96CkAAAAJ&hl=en), [Meng Wang](https://github.com/mengwa41), [Miran Heo](https://sites.google.com/view/miranheo), [Pengchuan Zhang](https://pzzhang.github.io/pzzhang/), [Roman Rädle](https://scholar.google.com/citations?user=Tpt57v0AAAAJ&hl=en), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en)
