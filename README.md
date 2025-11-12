# SAM 3: Segment Anything with Concepts

Meta Superintelligence Labs

Nicolas Carion $^{\*}$, Laura Gustafson $^{\*}$, Yuan-Ting Hu $^{\*}$, Shoubhik Debnath $^{\*}$, Ronghang Hu $^{\*}$, Didac Suris $^{\*}$,
Chaitanya Ryali $^{\*}$, Kalyan Vasudev Alwala $^{\*}$, Haitham Khedr $^{\*}$, Andrew Huang, Jie Lei, Tengyu Ma, Baishan
Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos
Afouras, Effrosyni Mavroudi, Katherine Xu $^{◦}$, Tsung-Han Wu $^{◦}$, Yu Zhou $^{◦}$, Liliane Momeni $^{◦}$, Rishi Hazra $^{◦}$,
Shuangrui Ding $^{◦}$, Sagar Vaze $^{◦}$, Francois Porcher $^{◦}$, Feng Li $^{◦}$, Siyuan Li $^{◦}$, Aishwarya Kamath $^{◦}$, Ho Kei
Cheng $^{◦}$, Piotr Dollar $^{\dagger}$, Nikhila Ravi $^{\dagger}$, Kate Saenko $^{\dagger}$, Pengchuan Zhang $^{\dagger}$, Christoph Feichtenhofer $^{\dagger}$

$^{\*}$ core contributor, $^{◦}$ intern, $^{\dagger}$ project lead, order is random within groups

[[`Paper`](LINK_TO_PAPER)] [[`Project`](LINK_TO_PROJECT)] [[`Demo`](LINK_TO_DEMO)] [[`Dataset`](LINK_TO_DATASET)] [[`Blog`](LINK_TO_BLOG)] [[`BibTeX`](HOW_TO_CITE)]

![SAM 3 architecture](assets/model_diagram.png?raw=true)
SAM 3 is a unified foundation model for visual grounding in images and videos. SAM 3 detects, segments, and tracks objects using text and geometric prompts such as points, boxes, and masks. We build a scalable a data engine that leverages SAM 3, human annotators, and AI models in the loop, which allows dramatic speed-ups in annotation. This allowed us to create SA-Co training dataset set with over **4 million unique concepts**, the largest high-quality open-vocab segmentation dataset to date. (TODO: We might need to set the tone here because we don't release the training set)

![SA-Co dataset](assets/sa_co_dataset.jpg?raw=true)

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher


1. **Create a new Conda environment:**

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

2. **Install PyTorch with CUDA support:**

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. **Clone the repository and install the package:**

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

4. **Install additional dependencies for example notebooks or development:**

```bash
# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

## Getting Started

### Basic Usage

```python
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

## Examples

The `examples` directory contains notebooks demonstrating how to use SAM3 with various types of prompts:

- [`sam3_image_predictor_example.ipynb`](examples/sam3_image_predictor_example.ipynb) : Demonstrates how to prompt SAM 3 with text and visual box prompts on images.
- [`sam3_video_predictor_example.ipynb`](examples/sam3_video_predictor_example.ipynb) : Demonstrates how to prompt SAM 3 with text prompts on videos, and doing further interactive refinements with points.
- [`sam3_image_batched_inference.ipynb`](examples/sam3_image_batched_inference.ipynb) : Demonstrates how to run batched inference with SAM 3 on images.
- [`saco_gold_silver_vis_example.ipynb`](examples/saco_gold_silver_vis_example.ipynb) : Shows a few examples from SA-Co image evaluation set.
- [`saco_veval_vis_example.ipynb`](examples/saco_veval_vis_example.ipynb) : Shows a few examples from SA-Co video evaluation set.

There are additional notebooks in the examples directory that demonstrate how to use SAM 3 for interactive instance segmentation in images and videos (SAM 1/2 tasks), or as a tool for an MLLM, and how to run evaluations on the SA-Co dataset.

To run the Jupyter notebook examples:

```bash
# Make sure you have the notebooks dependencies installed
pip install -e ".[notebooks]"

# Start Jupyter notebook
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## Model
SAM 3 consists of a detector and a tracker that share a vision encoder. The detector is a DETR-based model conditioned on text, geometry, and image exemplars. The tracker inherits the SAM 2 transformer encoder-decoder architecture, supporting video segmentation and interactive refinement.

## Image Results (TODO: Select few baselines/metrics, or screenshot table from paper)

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="3" style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th colspan="3" style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">Instance Segmentation</th>
      <th colspan="5" style="text-align: center; padding: 12px 20px">Box Detection</th>
    </tr>
    <tr>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">SA-Co</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">COCO</th>
      <th style="text-align: center; padding: 12px 20px">SA-Co</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP<sub>o</sub>
</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Human</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">OWLv2*</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">DINO-X</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Gemini 2.5</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
  </tbody>
</table>
</div>

## Video Results

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SA-V test</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">YT-Temporal-1B test</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SmartGlasses test</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVVIS test</th>
      <th style="text-align: center; padding: 12px 20px">BURST test</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">mAP</th>
      <th style="text-align: center; padding: 12px 20px">HOTA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Human</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">0.0</td>
      <td style="text-align: center; padding: 10px 20px">0.0</td>
    </tr>
  </tbody>
</table>
</div>

## SA-Co Dataset

We release 2 image benchmarks, [SA-Co gold](scripts/eval/gold/README.md) and [SA-Co silver](scripts/eval/silver/README.md), and a video benchmark [SA-Co/VEval](scripts/eval/veval/README.md). See the linked READMEs for more details.

## Development

To set up the development environment:

```bash
# Make sure you have the development dependencies installed
pip install -e ".[dev,train]"
```

To format the code:
```bash
ufmt format .
```

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the TODO License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

TODO
