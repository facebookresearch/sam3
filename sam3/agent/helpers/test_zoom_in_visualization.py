import os
from PIL import Image
import json
from improving_check_each_mask import save_single_mask_para_visualization_zoomin

# Paths
json_path = "/fsx-onevision/yuzhou1/code/out/sam_out/-fsx-onevision-yuzhou1-datasets-reasonseg-LISA-dataset-val-609761865_a6078603bb_o.jpg/skateboarder.json"
output_dir = "/fsx-onevision/yuzhou1/code/agent/helpers/test"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load JSON metadata
with open(json_path, 'r') as f:
    data = json.load(f)

# Load the original image
image_path = data["original_image_path"]
image = Image.open(image_path)

# Process each mask
for i, mask_rle in enumerate(data["pred_masks"]):
    # Construct object_data dict as expected by the function
    object_data = {
        "labels": [{"noun_phrase": f"mask_{i}"}],
        "segmentation": {
            "counts": mask_rle,
            "size": [data["orig_img_h"], data["orig_img_w"]]
        }
    }

    # Call the visualization function
    pil_img, prompt, index2ans = save_single_mask_para_visualization_zoomin(object_data, image)

    # Save the output image
    output_path = os.path.join(output_dir, f"mask_visualization_{i}.png")
    pil_img.save(output_path)
    print(f"Saved visualization for mask {i} to {output_path}")

print("All mask visualizations saved.")