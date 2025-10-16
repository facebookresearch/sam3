import argparse
import hashlib
import json
import subprocess
from multiprocessing import cpu_count, Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_video_info(video_path):
    """Extract video information using ffprobe."""
    try:
        # Run ffprobe to get video information in JSON format
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Extract stream information
        stream = data.get("streams", [{}])[0]

        # Get frame rate as string (format like "30/1" or "24000/1001")
        fps = stream.get("r_frame_rate")

        # Get duration directly from stream
        duration = stream.get("duration")
        duration = float(duration) if duration else None

        # Get number of frames directly from stream
        nb_frames = stream.get("nb_frames")
        nb_frames = int(nb_frames) if nb_frames else None

        # Get width and height
        width = stream.get("width")
        height = stream.get("height")

        return {
            "width": int(width) if width else None,
            "height": int(height) if height else None,
            "fps": fps,
            "nb_frames": nb_frames,
            "duration": duration,
        }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return {
            "width": None,
            "height": None,
            "fps": None,
            "nb_frames": None,
            "duration": None,
        }


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating SHA256 for {file_path}: {e}")
        return None


def process_single_video(video_file):
    """Process a single video file and return its information."""
    # Get file size
    file_size = video_file.stat().st_size

    # Get video information
    video_info = get_video_info(str(video_file))

    # Calculate SHA256
    sha256 = calculate_sha256(str(video_file))

    # Combine all information
    result = {
        "video_name": video_file.name,
        "file_size": file_size,
        "width": video_info["width"],
        "height": video_info["height"],
        "fps": video_info["fps"],
        "nb_frames": video_info["nb_frames"],
        "duration": video_info["duration"],
        "sha256": sha256,
    }
    return result


def process_videos(videos_dir, num_workers=None):
    """Process all video files in the directory and return a DataFrame."""
    videos_path = Path(videos_dir)
    if not videos_path.exists():
        print(f"Directory not found: {videos_dir}")
        return pd.DataFrame()

    # Get all .mp4 files using glob
    video_files = list(videos_path.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {videos_dir}")
        return pd.DataFrame()

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    print(
        f"Found {len(video_files)} video files. Processing with {num_workers} workers..."
    )

    # Process videos using multiprocessing with tqdm progress bar
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_video, video_files),
                total=len(video_files),
                desc="Processing videos",
            )
        )

    # Create DataFrame
    df = pd.DataFrame(results)
    return df


def main():
    """Main function to process videos and generate specs."""
    parser = argparse.ArgumentParser(
        description="Extract video specifications using ffprobe and calculate SHA256 checksums."
    )
    parser.add_argument(
        "--videos_path", type=str, help="Path to directory containing .mp4 videos"
    )
    parser.add_argument("--res_file", type=str, help="Path to output CSV file")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPU cores)",
    )

    args = parser.parse_args()

    # Process all videos and create DataFrame
    df = process_videos(args.videos_path, num_workers=args.num_workers)

    # Display the DataFrame
    print("\n" + "=" * 80)
    print("Video Information Summary:")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(args.res_file, index=False)
    print(f"\nResults saved to: {args.res_file}")


if __name__ == "__main__":
    main()
