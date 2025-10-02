# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

""" Script to download images from public URLs needed for SAC-Gold test set 

    Usage: python download_metaclip_urls_submitit.py --url-list-file gold_metaclip_filename_urls_mapping.json --out-dir <out_folder> --n-jobs 100
"""

import io
import json
import os
from io import BytesIO

from pathlib import Path

import numpy as np

import requests
import submitit
from PIL import Image
from tqdm import tqdm


def get_job_dir(root, job_id=None):
    job_folder = "%j" if job_id is None else str(job_id)
    return os.path.join(root, "jobs", job_folder)


def is_valid_image(content):
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        # img.verify()  # Verify that it is, in fact, an image
        return True
    except Exception:
        return False


class Launcher:
    def __init__(self, args, job_id, url_list):
        self.args = args
        self.job_id = job_id
        self.url_list = url_list

    def check_valid_urls(self):
        print("Processing json list of len ", len(self.url_list))
        valid_urls = {}
        timeout_urls, invalid_urls = [], []

        for inp in tqdm(self.url_list):
            fid, url = inp
            result = {"image_id": fid, "url": url}
            try:
                # headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                    "Referer": "https://www.aliexpress.com/",
                }

                response = requests.get(url, headers=headers, timeout=5)
                if is_valid_image(response.content):
                    if response.status_code in valid_urls:
                        valid_urls[response.status_code].append(result)
                    else:
                        valid_urls[response.status_code] = [result]

                    path_components = fid.split("_")
                    filename = f"{self.args.out_dir}/images/{path_components[-3]}/{path_components[-2]}/{Path(fid).name}"
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    # print(f"Image saved successfully as {filename}")
            except requests.exceptions.Timeout as e:
                print(f"Timeout error {url}: {e}")
                timeout_urls.append(result)
            except requests.exceptions.RequestException as e:
                print(f"Invalid {url}: {e}")
                invalid_urls.append(result)
            except:
                print(f"Other exception {url}")
                invalid_urls.append(result)

        valid_out = f"{self.args.out_dir}/valid_{self.job_id}.json"
        others_out = f"{self.args.out_dir}/others_{self.job_id}.json"
        with open(valid_out, "w") as valid_json_fp:
            json.dump(valid_urls, valid_json_fp)

        other_urls = {"invalid": invalid_urls, "timeout": timeout_urls}
        with open(others_out, "w") as other_json_fp:
            json.dump(other_urls, other_json_fp)

    def __call__(self):
        return self.check_valid_urls()


def main():
    """
    Slurm script that downloads images from public URLs

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
        default="/fsx-onevision/shoubhikdn/urls_stats/metaclip/gold_test_release",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        required=True,
    )
    parser.add_argument("--timeout", type=int, default=1440)
    parser.add_argument("--account", type=str, default="onevision")
    parser.add_argument("--qos", type=str, default="onevision_high")
    parser.add_argument(
        "--slurm-output-dir",
        type=str,
        default="/fsx-onevision/shoubhikdn/urls_stats/slurm",
    )
    args = parser.parse_args()

    # Split json file
    with open(args.url_list_file, "r") as json_file:
        json_list = json.load(json_file)
        urls = [(k, v) for k, v in zip(json_list.keys(), json_list.values())]
        chunked_url_list = [x for x in np.array_split(urls, args.n_jobs)]

    # Build SLURM executor
    jobs_dir = get_job_dir(args.slurm_output_dir)
    cpus_per_task = 12
    executor = submitit.AutoExecutor(folder=jobs_dir, cluster=None)
    executor.update_parameters(
        timeout_min=args.timeout,
        gpus_per_node=0,
        tasks_per_node=1,
        cpus_per_task=cpus_per_task,
        slurm_account=args.account,
        slurm_qos=args.qos,
    )
    executor.update_parameters(slurm_srun_args=["-vv", "--cpu-bind", "none"])

    # Create output folder
    os.makedirs(args.out_dir, exist_ok=True)

    # Launch jobs
    jobs = []
    with executor.batch():
        for job_id in range(args.n_jobs):
            launcher = Launcher(args, job_id, chunked_url_list[job_id])
            job = executor.submit(launcher)
            jobs.append(job)

    for j in jobs:
        print(f"Slurm JobID: {j.job_id}")
    print(f"Saving slurm outputs to {jobs_dir}")


if __name__ == "__main__":
    main()
