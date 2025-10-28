# SAM3

SAM3 is a foundational visual grounding model that detects, segments and tracks objects in images and videos. SAM3 supports object category detection and instance segmentation using text and visual example prompts, as well as vocabulary-free promptable segmentation using point, mask or box prompts as in SAM2. SAM3 can detect objects based on open-vocabulary noun phrases1, and allows the user to interactively refine the output with additional points or boxes.

**Note**: The current model definition only supports images, video is coming soon.

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

### Option 1: Using Conda (Recommended)

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

4. **Install additional dependencies for examples and interactive usage:**

```bash
# For running examples
pip install -e ".[examples]"

# For development
pip install -e ".[dev]"
```



## Usage

### Basic Usage

```python
import torch
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import numpy as np

# Load the model
model = build_sam3_image_model(bpe_path="path/to/bpe/vocabulary.txt", checkpoint_path="path/to/checkpoint.pth")
processor = Sam3Processor()

# Load an image
image = Image.open("your_image_path.jpg")
inference_state = processor(image, instance_prompt=False)

# Pass your text prompt to the model
processor.add_prompt(inference_state, text_str="YOUR CONCEPT",  instance_prompt=False)
outputs = processor.postprocess_output(inference_state ,output_prob_thresh=0.5)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = outputs["out_binary_masks"], outputs["out_boxes_xywh"], outputs["out_probs"]

# For point-to-mask and box-to-mask inference, please checkout the examples directory
```

### Interactive Segmentation

SAM3 supports various types of prompts:
- Point prompts (foreground and background)
- Box prompts
- Mask prompts
- Text prompts

Check out the examples directory for more detailed usage examples.

## Examples

The `examples` directory contains scripts and notebooks demonstrating how to use SAM3:

- `sam3_image_multiway_prompting.ipynb`: Jupyter notebook demonstrating various prompt types

To run the Jupyter notebook examples:

```bash
# Make sure you have the interactive dependencies installed
pip install -e ".[examples]"

# Start Jupyter notebook
jupyter notebook examples/sam3_image_multiway_prompting.ipynb
```

## Features

- Multi-modal prompting (points, boxes, masks, text)
- High-quality segmentation masks
- Fast inference
- Support for interactive segmentation workflows

## Development

To set up the development environment:

```bash
# Make sure you have the development dependencies installed
pip install -e ".[dev]"
```

To formwat the code:
```bash
ufmt format .
```

To run tests:

```bash
pytest
```

## License

This project is licensed under the TODO License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the research and work done by Meta AI Research.
