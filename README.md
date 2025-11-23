# SAM 3: Segment Anything 3

SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, 
segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor 
[SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified 
by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. 

## Prerequisites

- ✅ Python 3.12 or higher
- ✅ PyTorch 2.7 or higher
- ✅ CUDA 12.2 or greater (not necessarily required, you can also use CPU)

## Installation

1. **Create a new virtual environment:**

    ```bash
    python3 -m venv "sam3test"
    source sam3test/bin/active
    ```

2. **Install PyTorch with CUDA support (CUDA>=12.2):**

    ```bash
    pip install torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126
    ```

3. **Clone the repository and install the package:**

    ```bash
    git clone https://github.com/RizwanMunawar/sam3-inference
    cd sam3-inference
    pip install -e .
    ```

⚠️ **Note:** Access to the `sam3.pt` checkpoint must be requested via the SAM 3 Hugging Face [repository](https://huggingface.co/facebook/sam3).
Once your request is approved, you’ll be able to download and use the `sam3.pt` model for inference with the example shown below.

### Inference on Image

![image-inference-readme-demo.jpg](/assets/image-inference-demo.jpg)

```python
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

label_to_predict = "white dog"  # this will be used as prompt for inference.

url = "assets/images/dogs.jpg"
image = Image.open(url)  # Image load

# SAM3 model load
processor = Sam3Processor(build_sam3_image_model(checkpoint_path="sam3.pt"))

# Run inference with text prompt
results = processor.set_text_prompt(state=processor.set_image(image), 
                                    prompt=label_to_predict)

# Visualization
result_image = draw_box_and_masks(cv2.imread(url, cv2.COLOR_RGB2BGR),  # PIL -> OpenCV
                                  results=results,
                                  show_boxes=True,
                                  show_masks=True,
                                  line_width=4,
                                  label=label_to_predict)

cv2.imwrite("sam3_results.png", result_image)  # Save (optional)
```

### Inference on video

Coming soon....

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.

## References

- [SAM3 offical implementation](https://github.com/facebookresearch/sam3)
- [OpenCV repository](https://github.com/opencv/opencv)
