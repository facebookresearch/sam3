# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Script to download images from public URLs needed for SAC-Gold test set

Usage: python download_metaclip_urls.py --url-list-file gold_metaclip_filename_urls_mapping_release.json --out-dir <out_folder> --max_workers 50
"""

import json
import os
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def is_valid_image(content):
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        # img.verify()  # Verify that it is, in fact, an image
        return True
    except Exception:
        return False


def get_filename_from_fid(fid, args):
    path_components = fid.split("_")
    filename = f"{args.out_dir}/images/{path_components[-3]}/{path_components[-2]}/{Path(fid).name}"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    return filename


def download_image(fid, url, args):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Referer": "https://www.aliexpress.com/",
    }
    for attempt in range(args.retry_count):
        try:
            response = requests.get(url, headers=headers, timeout=args.request_timeout)
            if is_valid_image(response.content):
                filename = get_filename_from_fid(fid, args)
                with open(filename, "wb") as f:
                    f.write(response.content)
                return filename
            else:
                print(f"Failed ({response.status_code}): {url}")
        except requests.exceptions.Timeout as e:
            print(f"Timeout error {url}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Invalid {url}: {e}")
        except:
            print(f"Other exception {url}")
        time.sleep(args.throttle_delay)
    print(f"Giving up: {url}")
    return None


def download_images(url_list, args):
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(download_image, fid, url, args) for fid, url in url_list
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Exception: {exc}")


def main():
    """
    Script that downloads images from public URLs

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url-list-file",
        type=str,
        default="gold_metaclip_filename_urls_mapping_release.json",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="/fsx-onevision/shoubhikdn/urls_stats/metaclip/gold_test_release_0930/",
    )
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--request_timeout", type=int, default=5)
    parser.add_argument("--retry_count", type=int, default=3)
    parser.add_argument("--throttle_delay", type=float, default=0.1)
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Split json file
    with open(args.url_list_file, "r") as json_file:
        json_list = json.load(json_file)
        urls = [(k, v) for k, v in zip(json_list.keys(), json_list.values())]

    start = time.time()
    download_images(urls, args)
    print("Time taken (sec): ", time.time() - start)


if __name__ == "__main__":
    main()
