# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3 (Segment Anything Model 3) is Meta's unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts (points, boxes, masks). The model introduces open-vocabulary segmentation capabilities, handling 270K+ unique concepts.

Key capabilities:
- **Image segmentation**: Text and visual prompts on static images
- **Video tracking**: Dense object tracking with temporal disambiguation
- **Interactive refinement**: Point-based refinement for both images and videos
- **Agent mode**: Complex multi-step segmentation using LLM reasoning

## Installation and Setup

### Basic Installation
```bash
# Create conda environment
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch with CUDA support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install SAM3
pip install -e .

# For notebooks
pip install -e ".[notebooks]"

# For development and training
pip install -e ".[dev,train]"
```

### Authentication
SAM 3 requires Hugging Face authentication:
```bash
# Generate access token at https://huggingface.co/settings/tokens
# Request access to https://huggingface.co/facebook/sam3
hf auth login
```

## Development Commands

### Code Formatting
```bash
# Format all code (uses ufmt with ruff-api backend)
ufmt format .
```

### Running Tests
Tests are located in `sam3/perflib/tests/`. Run with pytest:
```bash
pytest sam3/perflib/tests/
```

### Running Examples
```bash
# Launch Jupyter notebooks
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## Training

### Local Training
```bash
# Single GPU
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Multi-GPU on single node
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4
```

### Cluster Training
```bash
# Basic cluster training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1

# With SLURM settings
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 1 \
    --partition gpu_partition \
    --account my_account \
    --num-gpus 8 \
    --num-nodes 2
```

### Evaluation
```bash
# Roboflow 100-VL zero-shot evaluation
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml

# ODinW13 zero-shot evaluation
python sam3/train/train.py -c configs/odinw13/odinw_text_only.yaml

# ODinW13 10-shot training (seed 300)
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```

Training outputs are saved to the `experiment_log_dir` specified in config:
- `config.yaml` / `config_resolved.yaml` - Configuration files
- `checkpoints/` - Model checkpoints
- `tensorboard/` - TensorBoard logs
- `logs/` - Text logs
- `submitit_logs/` - Cluster job logs

Monitor training:
```bash
tensorboard --logdir /path/to/experiment_log_dir/tensorboard
```

## Architecture

### High-Level Model Structure

SAM 3 uses a **decoupled detector-tracker** design with shared vision encoder (848M parameters total):

1. **Detector** (`Sam3Image`/`Sam3ImageOnVideoMultiGPU`): DETR-based model for open-vocabulary detection and segmentation
2. **Tracker** (`Sam3TrackerPredictor`): SAM 2-style transformer for temporal propagation and interactive refinement

### Key Components

#### Vision-Language Backbone (`SAM3VLBackbone`)
- **Visual encoder**: ViT-based (`ViT` in `model/vitdet.py`) with 1024 embed dim, 32 layers
- **Visual neck**: `Sam3DualViTDetNeck` creates multi-scale feature pyramid with 4 levels
- **Text encoder**: `VETextEncoder` with custom tokenizer (`SimpleTokenizer`)
- Vision and text features are fused for multimodal understanding

#### Detector Components
- **Transformer encoder** (`TransformerEncoderFusion`): 6 layers, fuses visual and text features
- **Transformer decoder** (`TransformerDecoder`): 6 layers with 200 queries, includes:
  - **Presence token**: Improves discrimination between similar prompts ("player in white" vs "player in red")
  - **Box refinement**: Iterative bounding box refinement across decoder layers
  - **DAC (Deformable Attention Cross-attention)**: For efficient cross-attention to image features
- **Segmentation head** (`UniversalSegmentationHead`): Pixel-level mask prediction with upsampling
- **Geometry encoder** (`SequenceGeometryEncoder`): Encodes visual prompts (points, boxes, masks)
- **Dot product scoring**: Matches text prompts to detected objects

#### Tracker Components (for video)
- **Memory encoder** (`SimpleMaskEncoder`): Encodes past frame masks into memory features
- **Memory attention**: RoPE-based attention over memory frames (max 7 frames, configurable)
- **Temporal disambiguation**: Association logic to maintain object identity across frames
  - IoU-based matching between detections and tracks
  - Hotstart delay/thresholds for new track initialization
  - Keep-alive counters for temporarily occluded objects

### Model Building

Models are built via `model_builder.py`:

```python
# Image model
model = build_sam3_image_model(
    device='cuda',
    eval_mode=True,
    checkpoint_path=None,  # Auto-downloads from HuggingFace
    enable_inst_interactivity=False,  # Set True for SAM 1-style point prompts
)

# Video model
video_model = build_sam3_video_model(
    apply_temporal_disambiguation=True,  # Enable tracking heuristics
    compile=False,  # Set True to compile with torch.compile
)

# Video predictor (high-level API with session management)
predictor = build_sam3_video_predictor(gpus_to_use=[0, 1])  # Multi-GPU support
```

### Data Flow

**Image Inference:**
1. Image → ViT encoder → multi-scale features
2. Text prompt → Text encoder → text embeddings
3. Visual + text features → Transformer encoder (fusion)
4. Transformer decoder → object queries → boxes + embeddings
5. Segmentation head → pixel-level masks

**Video Inference:**
1. **Detection phase**: Detector runs on each frame to find objects matching text prompt
2. **Tracking phase**: Tracker propagates masks across frames using memory attention
3. **Association**: Match detections to existing tracks via IoU and appearance
4. **Reconditioning**: Periodically re-run detector to refresh tracks (every 16 frames by default)

### Important File Locations

- **Model definitions**: `sam3/model/`
  - `sam3_image.py` - Detector model
  - `sam3_tracking_predictor.py` - Tracker model
  - `sam3_video_inference.py` - Video inference pipeline with association logic
  - `sam3_video_predictor.py` - High-level video predictor API
- **Training**: `sam3/train/`
  - `train.py` - Entry point with Hydra config
  - `trainer.py` - Training loop
  - `matcher.py` - Hungarian matcher for detection training
  - `loss/` - Loss functions
  - `data/` - Data loaders and augmentations
- **Evaluation**: `sam3/eval/`
  - `coco_eval.py` - COCO-style evaluation
  - `cgf1_eval.py` - Concept-grounded F1 evaluation
  - `hota_eval_toolkit/` - HOTA metrics for video
  - `teta_eval_toolkit/` - TETA metrics for video
- **Agent**: `sam3/agent/`
  - `agent_core.py` - Multi-step reasoning agent
  - `client_sam3.py` - SAM3 inference client
  - `client_llm.py` - LLM client for reasoning
- **Performance**: `sam3/perflib/`
  - Triton kernels and optimized operations
  - `compile.py` - torch.compile utilities

## Training Details

### Configuration System

Training uses Hydra for configuration management. Configs are in `sam3/train/configs/`:
- `eval_base.yaml` - Base evaluation config
- `roboflow_v100/` - Roboflow 100-VL dataset configs
- `odinw13/` - ODinW13 dataset configs
- `gold_image_evals/` - SA-Co Gold evaluation
- `saco_video_evals/` - SA-Co VEval evaluation

Key config sections:
- `paths`: Dataset paths, BPE tokenizer, experiment output directory
- `launcher`: GPU/node configuration
- `submitit`: Cluster execution settings (SLURM)
- `trainer.mode`: Set to `"train"` or `"val"` for training/evaluation

### Job Arrays

For dataset sweeps (e.g., training on all 100 Roboflow datasets):
```yaml
submitit:
  job_array:
    num_tasks: 100
    task_index: 0  # Auto-set by SLURM array index
```

### Data Format

Training expects:
- Roboflow 100-VL: Organized by supercategory with `train/`, `valid/`, `test/` splits
- ODinW: Organized by dataset name with `train/`, `valid/`, `test/` splits

## Agent System

The SAM 3 Agent (`sam3/agent/agent_core.py`) enables complex text prompts via multi-step reasoning:

1. LLM breaks down complex query into sub-queries
2. Each sub-query calls SAM 3 via `segment_phrase` tool
3. Results are accumulated and reasoned over
4. Agent can refine queries based on intermediate results

Example: "segment all animals that are not dogs" → LLM first segments "animals", then filters out "dogs"

## Common Gotchas

1. **Checkpoint loading**: Models auto-download from HuggingFace by default. Set `load_from_HF=False` and provide `checkpoint_path` for local checkpoints.

2. **Memory usage**: Video models maintain memory of past frames. For long videos, consider:
   - Reducing `num_maskmem` (default: 7)
   - Enabling `offload_output_to_cpu_for_eval=True`
   - Processing in chunks

3. **Text prompt format**: Text prompts should be noun phrases describing objects, not full sentences. Good: "a red car", Bad: "find me all the red cars in this image"

4. **Presence token**: The presence token improves discrimination but requires careful prompt design. For negative prompts (no matching objects), the model outputs empty predictions.

5. **Compilation**: `torch.compile` can speed up inference but increases startup time. Best for batch processing.

6. **Multi-GPU video**: Video predictor supports multi-GPU via `gpus_to_use` parameter for parallel frame processing.

## Code Style

- **Formatting**: Use `ufmt format .` (Black-compatible with ruff backend)
- **Type hints**: Pyre type checker is used (`# pyre-unsafe` comments disable checking)
- **Imports**: Standard library → third-party → local
- **Line length**: 88 characters (Black default)
