import numpy as np
from PIL import Image
import json
import cv2
from helpers.visualizer import Visualizer
import pycocotools.mask as mask_utils    
from helpers.improving_check_each_mask import save_single_mask_para_visualization_zoomin



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




if __name__ == "__main__":

    # 1. Define the path to your JSON file
    json_file_path = '/fsx-onevision/yuzhou1/code/out/sam_out/-fsx-onevision-yuzhou1-datasets-reasonseg-LISA-dataset-val-609761865_a6078603bb_o.jpg/skateboarder.json'
    image_file_path = '/fsx-onevision/yuzhou1/datasets/reasonseg/LISA/dataset/val/609761865_a6078603bb_o.jpg'
    image_output_path = '/fsx-onevision/yuzhou1/code/agent/bak/skateboarder.jpg'
    # label_mode = "a"
    label_mode = "1"
    # alpha=0.25
    # font_size_multiplier=2 
    # boarder_width_multiplier=1


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




























    # img: a numpy array of shape (H, W, C), where H and W correspond to
    #                 the height and width of the image respectively. C is the number of
    #                 color channels. The image is required to be in RGB format since that
    #                 is a requirement of the Matplotlib library. The image is also expected
    #                 to be in the range [0, 255].



    # encoded masks (masks-like object): Supported types are:

    #                 * :class:`detectron2.structures.PolygonMasks`,
    #                   :class:`detectron2.structures.BitMasks`.
    #                 * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
    #                   The first level of the list corresponds to individual instances. The second
    #                   level to all the polygon that compose the instance, and the third level
    #                   to the polygon coordinates. The third level should have the format of
    #                   [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
    #                 * list[ndarray]: each ndarray is a binary mask of shape (H, W).
    #                 * list[dict]: each dict is a COCO-style RLE.


    # masks are binary masks

