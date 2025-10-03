
import pycocotools.mask as mask_utils
import numpy as np
import json
import os
from PIL import Image
from .viz import visualize, zoom_in_and_visualize


if __name__ == "__main__":

    # 1. Define the path to your JSON file
    json_file_path = '/fsx-onevision/yuzhou1/code/out/sam_out/-fsx-onevision-yuzhou1-datasets-reasonseg-LISA-dataset-val-609761865_a6078603bb_o.jpg/person.json'
    image_file_path = '/fsx-onevision/yuzhou1/datasets/reasonseg/LISA/dataset/val/609761865_a6078603bb_o.jpg'
    image_output_path = '/fsx-onevision/yuzhou1/code/sam3/sam3/agent/viz_out/person.jpg'
    # label_mode = "a"
    label_mode = "1"


    # 2. load json file and extract the necessary data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    boxes_array = np.array(data['pred_boxes'])
    coco_rle_masks = [{'size': (data["orig_img_h"], data["orig_img_w"]), 'counts': rle} for rle in data['pred_masks']]
    img = np.array(Image.open(image_file_path).convert('RGB'))
    
    # use binary masks instead of coco_rle_masks
    binary_masks = [mask_utils.decode(i) for i in coco_rle_masks]


    # 3. Call the visualize function and save the output image
    viz_image = visualize(img, boxes_array, coco_rle_masks, binary_masks, label_mode="1")
    viz_image.save(image_output_path)
    print("saved image to ", image_output_path)