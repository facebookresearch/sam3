# SAM 3 Installation Guide for NVIDIA Jetson Platforms

This guide provides detailed instructions for installing and running SAM 3 on NVIDIA Jetson devices, including AGX Orin, Orin Nano, and Orin NX running JetPack 6.x.

## Prerequisites

### Hardware Requirements
- **NVIDIA Jetson Device**: AGX Orin, Orin Nano, or Orin NX
- **JetPack**: 6.0 or later (tested on JetPack 6.2 / L4T R36.4)
- **Storage**: At least 10GB free space for model checkpoints
- **Memory**: 8GB+ RAM recommended for optimal performance

### Software Requirements
- **Python**: 3.10 (compatible with NVIDIA PyTorch builds for JetPack 6.x)
- **CUDA**: 12.6 or higher (included in JetPack 6.x)
- **PyTorch**: 2.8.0 or higher (from NVIDIA Jetson AI Lab)

## Installation Steps

### 1. Verify JetPack Version

```bash
cat /etc/nv_tegra_release
# Should show: R36.x.x (JetPack 6.x)
```

### 2. Create Virtual Environment

```bash
# Install venv if not available
sudo apt install python3.10-venv

# Create and activate virtual environment
python3 -m venv sam3_env
source sam3_env/bin/activate
pip install --upgrade pip
```

### 3. Install PyTorch for Jetson

Install PyTorch 2.8.0 from NVIDIA's Jetson AI Lab repository:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Install SAM 3

Clone and install SAM 3:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"  # Include notebook dependencies
```

### 5. Request Model Access

SAM 3 requires accessing gated model weights:

1. Visit https://huggingface.co/facebook/sam3
2. Click "Request access" and accept terms
3. Wait for Meta approval (usually within 1-2 days)

### 6. Download Model Checkpoints

After approval:

```bash
pip install huggingface-hub
huggingface-cli login  # Enter your Hugging Face token

# Download checkpoints
huggingface-cli download facebook/sam3 --local-dir ./checkpoints
```

## Performance Optimization

### Enable Maximum Performance Mode

```bash
# Enable all CPU cores at maximum frequency
sudo jetson_clocks

# Set to maximum performance mode (MODE 0)
sudo nvpmodel -m 0
```

### Verify Performance Settings

```bash
# Check current power mode
sudo nvpmodel -q

# Monitor system performance
jtop  # Install with: sudo pip install jetson-stats
```

## Usage Examples

### Basic Import Test

```python
import torch
from sam3 import build_sam3_image_model

# Verify CUDA
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Image Segmentation

See the example notebooks in `examples/` directory:
- `sam3_image_predictor_example.ipynb` - Image segmentation
- `sam3_video_predictor_example.ipynb` - Video segmentation

### Using FP16 for Faster Inference

```python
# Enable half precision (FP16) for ~2x speedup
model = model.half()
```

## Performance Expectations

Tested on **NVIDIA Jetson AGX Orin Developer Kit** (JetPack 6.2):

- **Image Segmentation**: ~100-300ms per frame (640x640, FP32)
- **With FP16**: ~50-150ms per frame
- **Memory Usage**: ~2-4GB VRAM
- **Model Size**: 848M parameters

### Optimization Tips

1. **Use FP16**: Reduces memory and increases speed
2. **Lower Resolution**: Process at 480x480 instead of 640x640
3. **Batch Processing**: Process multiple frames together when possible
4. **Frame Skipping**: For real-time video, process every 2nd or 3rd frame

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Enable FP16 or reduce batch size
```python
model = model.half()  # Use FP16
```

### Issue: Slow Performance

**Solution**: Ensure jetson_clocks is enabled and nvpmodel is set to max performance
```bash
sudo jetson_clocks
sudo nvpmodel -m 0
```

### Issue: Import Errors

**Solution**: Ensure all dependencies are installed
```bash
pip install -e ".[notebooks]"
```

## Known Limitations

- **Python 3.10 Only**: NVIDIA PyTorch 2.8.0 for Jetson only supports Python 3.10
- **No TensorRT Optimization**: TensorRT acceleration not yet implemented (future enhancement)
- **Video Processing**: Real-time processing requires frame skipping or lower resolution

## Supported Platforms

- ✅ Jetson AGX Orin (tested)
- ✅ Jetson Orin Nano (should work, not extensively tested)
- ✅ Jetson Orin NX (should work, not extensively tested)
- ❌ Jetson Nano / TX2 / Xavier (JetPack 6.x not available)

## Getting Help

- **GitHub Issues**: https://github.com/facebookresearch/sam3/issues
- **Jetson Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems

## References

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-orin)
- [PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)
- [SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
