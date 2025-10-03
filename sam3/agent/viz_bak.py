import numpy as np
from PIL import Image
import json
import cv2
from .helpers.visualizer import Visualizer
import pycocotools.mask as mask_utils    
from .helpers.improving_check_each_mask import save_single_mask_para_visualization_zoomin



# main visualization function
def visualize(img, boxes, masks, binary_masks, alpha=0.15, label_mode="1", font_size_multiplier=1.2, boarder_width_multiplier=0, color=None):

    # print("len(masks):", len(masks), "len(boxes):", len(boxes), "len(binary_masks):", len(binary_masks))
    assert (len(masks) == len(boxes))  # masks and bboxes should have the same length

    viz = Visualizer(img, font_size_multiplier=font_size_multiplier, boarder_width_multiplier=boarder_width_multiplier)
    viz.overlay_instances(
        boxes=boxes,
        masks=masks,
        binary_masks=binary_masks,
        # assigned_colors = ["#00FF00"]*len(masks),
        assigned_colors=[color] * len(masks) if color else None,
        alpha=alpha,
        label_mode=label_mode,
    )
    viz_image = Image.fromarray(viz.output.get_image())
    
    return viz_image


# function to be called for visualizing check_each_mask
def zoom_in_and_visualize(input_json, index, mask_alpha=0.15):
    """
    """
    object_data = {
        "labels": [{"noun_phrase": f"mask_{index}"}],
        "segmentation": {
            "counts": input_json["pred_masks"][index],
            "size": [input_json["orig_img_h"], input_json["orig_img_w"]]
        }
    }
    image = Image.open(input_json["original_image_path"])

    pil_img, _, _, color_hex = save_single_mask_para_visualization_zoomin(object_data, image, mask_alpha=mask_alpha)
    return pil_img, color_hex

    



def visualize_masks_from_result_json(result_json: dict, color=None):
    """
    Given a output json dictionary "result_json" containing the following keys:
    
    'original_image_path': str -- the path to the original image,
    'orig_img_h': int -- the height of the original image,
    'orig_img_w': int -- the width of the original image,
    'pred_boxes': list -- the predicted bounding boxes,
    'pred_scores': list -- the predicted scores,
    'pred_masks': list -- the predicted masks,
    
    this function will load the original image, render the masks on the image, and return the rendered image.
    
    """
    # get the boxes, coco_rle_masks, and binary masks
    boxes_array = np.array(result_json['pred_boxes'])
    coco_rle_masks = [{'size': (result_json["orig_img_h"], result_json["orig_img_w"]), 'counts': rle} for rle in result_json['pred_masks']]
    binary_masks = [mask_utils.decode(i) for i in coco_rle_masks]
    
    # load the original image
    img = cv2.imread(result_json['original_image_path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # visualize the masks on the image
    image_w_rendered_masks = visualize(img, boxes_array, coco_rle_masks, binary_masks, color=color)
    return image_w_rendered_masks