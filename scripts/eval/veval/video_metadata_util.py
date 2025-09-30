import os
import cv2
from glob import glob
from tqdm import tqdm


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "height": height,
        "width": width,
        "num_frames": frame_count,
        "fps": fps,
        "duration": duration,
    }

def check_downloaded_video_metadata():
    for video_path in tqdm(glob("./downloads/*.mp4")):
        metadata = get_video_metadata(video_path)
        print(
            f"{os.path.basename(video_path)}: height={metadata['height']}, width={metadata['width']}, num_frames={metadata['num_frames']}, fps={metadata['fps']:.2f}, duration={metadata['duration']:.2f}s"
        )


def check_video_frame_metadata():
    jpeg_paths = glob(
        "/fsx-onevision-auto-sync/tym/sam3_video/release/media/release_09242025/*yt1b*/*/00000.jpg"
    )
    for jpeg_path in tqdm(jpeg_paths):
        # Load image to get height and width only
        img = cv2.imread(jpeg_path)
        if img is None:
            raise ValueError(f"Could not load image file: {jpeg_path}")
        height, width = img.shape[:2]
        metadata = {"height": height, "width": width}
        print(
            f"{os.path.basename(jpeg_path)}: height={metadata['height']}, width={metadata['width']}"
        )