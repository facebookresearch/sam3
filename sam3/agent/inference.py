import pycocotools.mask as mask_utils
from tqdm import tqdm
from typing import List
import numpy as np
import cv2
import argparse
import json
import os
import copy
from PIL import Image
from .client_llm import send_generate_request as send_llama_request
from .client_sam3 import call_sam_service as call_sam_service_orig
from .viz import visualize, visualize_masks_from_result_json
import tempfile
import matplotlib.pyplot as plt
import shutil
import getpass
import random
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
USER = getpass.getuser()

# Model name mapping from short names to full model strings
MODEL_NAME_MAPPING = {
    "qwen2.5_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5_72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "llama4_maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
}

def get_full_model_name(short_name: str) -> str:
    """Convert short model name to full model string"""
    if short_name in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[short_name]
    else:
        # If it's already a full model name, return as is
        raise ValueError(f"Unknown model name: {short_name}")

def get_cluster_type():
    host_name = os.uname()[1]
    release_name = os.uname().release
    if "rsc" in host_name:
        return "rsc"
    elif "aws" in release_name:
        if os.path.isdir("/fsx-onevision/"):
            return "aws"
        elif os.path.isfile("/checkpoint/sam3/shared/.cluster_name_aws_h100"):
            return "aws_h100"
        elif os.path.isfile("/checkpoint/sam3/shared/.cluster_name_aws_h100_1"):
            return "aws_h100_1"
        else:
            raise RuntimeError("Cannot identify the job cluster")
    elif "amzn" in release_name:
        return "fair_sc"
    elif os.path.isdir("/private/home/"):
        return "fair"
    elif os.path.isdir("/data/nllb/"):
        return "azure"
    else:
        return "oss"
_cluster = get_cluster_type()
if _cluster == "fair_sc":
    DEFAULT_USER_FOLDER = f"/checkpoint/sam3/{USER}/"
elif _cluster == "aws":
    DEFAULT_USER_FOLDER = f"/fsx-onevision/{USER}/"

# Use a single output directory for everything
output_dir = "agent_output"

# Global variables for server URLs - will be set by main()
sam_server_url = None
llm_server_url = None

# ----------------- Define the version of SAM3 Agent to use -----------------
from .agent_core import agent_inference



def run_single_image_inference(args):
    """Run inference on a single image with provided prompt"""
    image_path = args.image
    text_prompt = args.prompt
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file names
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    prompt_for_filename = text_prompt.replace("/", "_").replace(" ", "_")
    
    base_filename = f"{image_basename}_{prompt_for_filename}_Agent_{args.MLLM}"
    output_json_path = os.path.join(output_dir, f"{base_filename}_Pred.json")
    output_image_path = os.path.join(output_dir, f"{base_filename}_Pred.png")
    agent_history_path = os.path.join(output_dir, f"{base_filename}_History.json")
    
    # Check if output already exists and skip
    if os.path.exists(output_json_path):
        print(f"Output JSON {output_json_path} already exists. Skipping.")
        return
    
    print(f"\n\n\n--------------Processing single image with prompt: {text_prompt}--------------\n")
    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    #try:
    agent_history, final_output_dict, rendered_final_output = agent_inference(
        image_path, text_prompt, 
        send_generate_request=send_generate_request, 
        call_sam_service=call_sam_service
    )
    
    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_path"] = image_path
    
    # Save outputs
    json.dump(final_output_dict, open(output_json_path, 'w'), indent=4)
    json.dump(agent_history, open(agent_history_path, 'w'), indent=4)
    rendered_final_output.save(output_image_path)
    
    print(f"\n✅ Successfully processed single image!")
    print(f"Output JSON: {output_json_path}")
    print(f"Output Image: {output_image_path}")
    print(f"Agent History: {agent_history_path}")


def main():
    parser = argparse.ArgumentParser(description="Run agent inference on single images.")
    
    # Single image mode arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a single image file for inference"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for single image inference"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="qwen2.5_7b",
        help="Version of the LLM model to use (e.g., qwen2.5_72b, llama4scout)"
    )
    parser.add_argument(
        "--sam3-host",
        default="localhost",
        help="Host where the SAM3 server is running (default: localhost)"
    )
    parser.add_argument(
        "--sam3-port", 
        default="9313",
        help="Port where the SAM3 server is running (default: 9313)"
    )
    parser.add_argument(
        "--llm-host",
        default="localhost",
        help="Host where the LLM server is running (default: localhost)"
    )
    parser.add_argument(
        "--llm-port",
        default="8001",
        help="Port where the LLM server is running (default: 8001)"
    )
    
    args = parser.parse_args()
    
    # Set up global server URLs
    global sam_server_url, llm_server_url, send_generate_request, call_sam_service
    sam_server_url = f"http://{getattr(args, 'sam3_host')}:{getattr(args, 'sam3_port')}/segment"
    llm_server_url = f"http://{args.llm_host}:{args.llm_port}"
    send_generate_request = partial(send_llama_request, server_url=llm_server_url, model=get_full_model_name(args.model))
    call_sam_service = partial(call_sam_service_orig, server_url=sam_server_url)
    
    # Convert model name and set MLLM identifier for backward compatibility
    model_str = args.model.split("/")[-1].replace("-", "_")
    args.MLLM = model_str
    
    print(f"Using SAM3 server: {sam_server_url}")
    print(f"Using LLM server: {llm_server_url}")
    print(f"Using model: {args.model}")
    
    # Run single image inference
    run_single_image_inference(args)


if __name__ == "__main__":
    main()
