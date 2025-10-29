
import pycocotools.mask as mask_utils
import numpy as np
import json
import os
from PIL import Image
from .viz import visualize


if __name__ == "__main__":

    # 1. Define the path to your JSON file
    json_file_path = '/fsx-onevision/yuzhou1/code/out/sam_out/-fsx-onevision-yuzhou1-datasets-reasonseg-LISA-dataset-val-609761865_a6078603bb_o.jpg/person.json'
    image_file_path = '/fsx-onevision/yuzhou1/datasets/reasonseg/LISA/dataset/val/609761865_a6078603bb_o.jpg'
    image_output_path = '/fsx-onevision/yuzhou1/code/sam3/sam3/agent/viz_out/person_from_json.jpg'
    # label_mode = "a"
    label_mode = "1"


    # 2. load json file and extract the necessary data
    with open(json_file_path, 'r') as f:
        data = json.load(f)


    # 3. Call the visualize function and save the output image
    viz_image = visualize(data)
    viz_image.save(image_output_path)
    print("saved image to ", image_output_path)