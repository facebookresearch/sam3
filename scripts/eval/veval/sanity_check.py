# Just some plot utils
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util
import video_prep

def bitget(val, idx):
    return (val >> idx) & 1


def pascal_color_map():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bitget(ind, channel) << shift
        ind >>= 3

    return colormap.astype(np.uint8)


def draw_masks_to_frame(
    frame: np.ndarray, masks: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    masked_frame = frame
    for mask, color in zip(masks, colors):
        curr_masked_frame = np.where(mask[..., None], color, masked_frame)
        masked_frame = cv2.addWeighted(masked_frame, 0.75, curr_masked_frame, 0.25, 0)

        if int(cv2.__version__[0]) > 3:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
        else:
            _, contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )

            # _, contours, _ = cv2.findContours(
            #     np.array(mask, dtype=np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            # )
        cv2.drawContours(masked_frame, contours, -1, (255, 255, 255), 7)  # White outer contour
        cv2.drawContours(masked_frame, contours, -1, (0, 0, 0), 5)  # Black middle contour
        cv2.drawContours(masked_frame, contours, -1, color.tolist(), 3)  # Original color inner contour
    return masked_frame


COLORS = pascal_color_map()[1:]

def get_frame_from_video(cap: cv2.VideoCapture, frame_number: int, num_frames: int = None):
    '''
    frame_number: start of frame index to read
    num_frames: total number of frames to read after frame_number
        if None return only the current frame
    '''
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    if num_frames:
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames
    else:
        ret, frame = cap.read()
        return frame



def get_masked_frame(masklet, frame_num, jpg_path):
    print(jpg_path)
    im = cv2.imread(jpg_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    masks = np.array([mask_util.decode(rle) for rle in [masklet[frame_num]]])
    masked_frame = draw_masks_to_frame(
        frame=im, masks=masks, colors=COLORS[: len(masks)]
    )
    return masked_frame


def sanity_check(saco_yt1b_id, masklet_loc, frame_num):
    id_map_df = pd.read_json("/home/tym/code/git_clone/sam3/scripts/saco_veval/saco_veval_data/saco_id_map/saco_yt1b_to_yt_video_id_map.json")

    with open("saco_veval_data/09242025/saco_veval_yt1b_test.json", "r") as f:
        data = json.load(f)

    video_df = pd.DataFrame(data["videos"])
    annot_df = pd.DataFrame(data["annotations"])
    pair_df = pd.DataFrame(data["video_np_pairs"])

    
    vis_df = annot_df[annot_df.video_id == saco_yt1b_id]
    masklet = vis_df.iloc[masklet_loc].segmentations
    
    # Find the corresponding yt_video_id_w_timestamps using the mapping
    mapping_row = id_map_df[id_map_df.saco_yt1b_id == saco_yt1b_id]
    
    yt_video_id_w_timestamps = mapping_row.iloc[0].yt_video_id_w_timestamps
    yt_video_id = mapping_row.iloc[0].yt_video_id

    video_prep.main(yt_video_id)
    
    # Define the paths
    org_jpg_path = f"/fsx-onevision/shared/data/YouTube1B/unified/JPEGImages/{yt_video_id_w_timestamps}_fps6/00000.jpg"
    new_jpg_path = f"/home/tym/code/git_clone/sam3/scripts/saco_veval/JPEGImages/{yt_video_id}/00000.jpg"
    

    org_masked_frame = get_masked_frame(masklet, frame_num, org_jpg_path)
    new_masked_frame = get_masked_frame(masklet, frame_num, new_jpg_path)
    
    return org_masked_frame, new_masked_frame


def sanity_check_jpg_overlap(saco_yt1b_id, masklet_loc, jpg_folder="JPEGImages"):
    id_map_df = pd.read_json("/home/tym/code/git_clone/sam3/scripts/saco_veval/saco_veval_data/saco_id_map/saco_yt1b_to_yt_video_id_map.json")

    with open("saco_veval_data/09242025/saco_veval_yt1b_test.json", "r") as f:
        data = json.load(f)

    video_df = pd.DataFrame(data["videos"])
    annot_df = pd.DataFrame(data["annotations"])
    pair_df = pd.DataFrame(data["video_np_pairs"])

    
    vis_df = annot_df[annot_df.video_id == saco_yt1b_id]
    masklet = vis_df.iloc[masklet_loc].segmentations
    
    # Find the corresponding yt_video_id_w_timestamps using the mapping
    mapping_row = id_map_df[id_map_df.saco_yt1b_id == saco_yt1b_id]
    
    yt_video_id_w_timestamps = mapping_row.iloc[0].yt_video_id_w_timestamps
    yt_video_id = mapping_row.iloc[0].yt_video_id

    option = None
    if jpg_folder == "JPEGImages":
        option = 2
    elif jpg_folder == "JPEGImages_by_matching_json":
        option = 3
    else:
        raise ValueError(f"Invalid jpg_folder: {jpg_folder}")

    video_prep.main(yt_video_id, option=option)
    
    # Define the paths
    org_jpg_path = f"/fsx-onevision/shared/data/YouTube1B/unified/JPEGImages/{yt_video_id_w_timestamps}_fps6/00000.jpg"
    new_jpg_path = f"/home/tym/code/git_clone/sam3/scripts/saco_veval/{jpg_folder}/{yt_video_id}/00000.jpg"

    # Load the images as numpy arrays
    org_jpg = cv2.imread(org_jpg_path)
    org_jpg = cv2.cvtColor(org_jpg, cv2.COLOR_BGR2RGB)
    new_jpg = cv2.imread(new_jpg_path)
    new_jpg = cv2.cvtColor(new_jpg, cv2.COLOR_BGR2RGB)
    # Calculate the difference between the images
    diff = cv2.absdiff(org_jpg, new_jpg)
    
    # Get image information
    org_height, org_width = org_jpg.shape[:2]
    new_height, new_width = new_jpg.shape[:2]
    
    # Get file sizes
    org_file_size = os.path.getsize(org_jpg_path)
    new_file_size = os.path.getsize(new_jpg_path)
    
    # Create info dictionary
    info = {
        "org_jpg": [org_height, org_width, org_file_size],
        "new_jpg": [new_height, new_width, new_file_size]
    }
    
    return org_jpg, new_jpg, diff, info
    

if __name__ == "__main__":
    sanity_check("saco_yt1b_000558", 0, 0)