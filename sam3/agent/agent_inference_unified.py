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
from client_llama4 import send_generate_request as send_llama_request
from client_sam3 import call_sam_service as call_sam_service_orig
from viz import visualize, visualize_masks_from_result_json
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
    "qwen3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen3_235b": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "llama4_maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
}




def get_full_model_name(short_name: str) -> str:
    """Convert short model name to full model string"""
    if short_name in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[short_name]
    else:
        # If it's already a full model name, return as is
        return short_name

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

sam_output_dir = f"{DEFAULT_USER_FOLDER}/code/out/sam_out"
qwen_output_dir = f"{DEFAULT_USER_FOLDER}/code/out/qwen_out"
parser_output_dir = f"{DEFAULT_USER_FOLDER}/code/out/parser_out"

# Global variables for server URLs - will be set by main()
sam_server_url = None
llm_server_url = None

# ----------------- Define the version of SAM3 Agent to use -----------------
from agent_v5 import agent_inference

# -------------- Dataset-specific imports --------------
try:
    from omnilabeltools import OmniLabel, OmniLabelEval, visualize_image_sample
except ImportError:
    print("Warning: omnilabeltools not available. OmniLabel dataset won't work.")
    OmniLabel = None

try:
    from refer import REFER
except ImportError:
    print("Warning: refer not available. RefCOCO datasets won't work.")
    REFER = None


def process_omnilabel_item(task_data):
    """Worker function for processing a single OmniLabel item"""
    (image_id, description_id, image_path, text_prompt, 
     raw_image_save_path, output_json_path, output_image_path, 
     agent_history_path, overwrite_existing_preds) = task_data
    
    # Check if output JSON exists and skip if overwrite is False
    if os.path.exists(output_json_path) and not overwrite_existing_preds:
        return f"Output JSON {output_json_path} already exists and overwrite is False. Skipping image_{image_id}_desc_{description_id}."

    print(f"\n\n\n--------------Processing image {image_id} with prompt: {text_prompt}--------------\n")
    #try:
    agent_history, final_output_dict, rendered_final_output = agent_inference(image_path, text_prompt, send_generate_request=send_generate_request, call_sam_service=call_sam_service)
    final_output_dict["description_id"] = description_id
    final_output_dict["image_id"] = image_id
    
    if not os.path.isfile(raw_image_save_path):
        shutil.copy(image_path, raw_image_save_path)
    json.dump(final_output_dict, open(output_json_path, 'w'), indent=4)
    json.dump(agent_history, open(agent_history_path, 'w'), indent=4)
    rendered_final_output.save(output_image_path)
    
    return f"Successfully processed image {image_id} with description {description_id}\nOutput JSON: {output_json_path}✅\nOutput Image: ✅{output_image_path}"

    #except Exception as e:
    #    return f"Error processing image {image_id}: {str(e)}"


def run_omnilabel_inference(args):
    """Run inference on OmniLabel dataset"""
    if OmniLabel is None:
        raise ImportError("omnilabeltools not available. Cannot run OmniLabel inference.")
    
    annotations_json_path = f'{DEFAULT_USER_FOLDER}/datasets/omnilabel/annotation/omnilabel_val_v0.1.3/dataset_all_val_v0.1.3.json'
    dataset_folder_path = f'{DEFAULT_USER_FOLDER}/datasets/omnilabel/dataset'
    output_folder_path = f'{DEFAULT_USER_FOLDER}/datasets/omnilabel/out'
    os.makedirs(output_folder_path, exist_ok=True)
    
    omni_label = OmniLabel(path_json=annotations_json_path)
    
    with open(annotations_json_path, 'r') as f:
        annotation_data = json.load(f)

    # Collect all tasks
    tasks = []
    for description in annotation_data['descriptions']:
        description_id = description['id']
        # we only want to process free-form text descriptions, not object categories
        if description["anno_info"]["type"] == "object_description":
            for image_id in description['image_ids']:
                image_path = os.path.join(dataset_folder_path, omni_label.get_image_sample(image_id)['file_name'])
                raw_image_save_path = os.path.join(output_folder_path, f"{image_id}_Raw.{omni_label.get_image_sample(image_id)['file_name'].split('.')[-1]}")
                
                text_prompt = description['text']
                text_prompt_for_save_path = text_prompt.replace("/", "_") if "/" in text_prompt else text_prompt

                output_json_path = os.path.join(output_folder_path, f"{image_id}_{text_prompt_for_save_path}_Agent_v{args.ver}_{args.MLLM}_Pred.json")
                output_image_path = os.path.join(output_folder_path, f"{image_id}_{text_prompt_for_save_path}_Agent_v{args.ver}_{args.MLLM}_Pred.png")
                agent_history_path = os.path.join(output_folder_path, f"{image_id}_{text_prompt_for_save_path}_Agent_v{args.ver}_{args.MLLM}_History.json")

                tasks.append((image_id, description_id, image_path, text_prompt, 
                            raw_image_save_path, output_json_path, output_image_path, 
                            agent_history_path, args.overwrite_existing_preds))
        else:
            assert description["anno_info"]["type"] == "object_category"

    # Process tasks with multiprocessing
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_omnilabel_item, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing OmniLabel"):
                result = future.result()
                print(result)
    else:
        for task in tqdm(tasks, desc="Processing OmniLabel"):
            result = process_omnilabel_item(task)
            print(result)


def process_reasonseg_item(task_data):
    """Worker function for processing a single ReasonSeg item"""
    (file_name, folder_path, input_json_path, output_image_path, 
     output_json_path, agent_history_path, overwrite_existing_preds) = task_data
    
    # Check if output JSON exists and skip if overwrite is False
    if os.path.exists(output_json_path) and not overwrite_existing_preds:
        return f"Output JSON {output_json_path} already exists and overwrite is False. Skipping {file_name}."

    if not os.path.exists(input_json_path):
        return f"⚠️ JSON file not found for {file_name}, skipping."

    #try:
    with open(input_json_path, "r") as jf:
        annotation_data = json.load(jf)
        prompt_list = annotation_data.get("text", [])
        if not prompt_list:
            return f"⚠️ No 'text' field found in {input_json_path}, skipping."
        prompt = prompt_list[0]

    print(f"\n\n\n--------------Processing filename: {file_name} with prompt: {prompt}--------------\n")
    agent_history, final_output_dict, rendered_final_output = agent_inference(os.path.join(folder_path, file_name + ".jpg"), prompt, send_generate_request=send_generate_request, call_sam_service=call_sam_service)
    
    json.dump(final_output_dict, open(output_json_path, 'w'), indent=4)  
    json.dump(agent_history, open(agent_history_path, 'w'), indent=4)
    rendered_final_output.save(output_image_path)
    
    return f"Successfully processed {file_name}\nOutput JSON: {output_json_path}✅\nOutput Image: ✅{output_image_path}"

    #except Exception as e:
    #    return f"Error processing {file_name}: {str(e)}"


def run_reasonseg_inference(args):
    """Run inference on ReasonSeg dataset"""
    dataset_dir = f"{DEFAULT_USER_FOLDER}/datasets/reasonseg/LISA/dataset"
    
    for split in args.split:
        folder_path = os.path.join(dataset_dir, split)
        # Collect all filenames ending with .jpg (without extension), but not gt.jpg or Pred.jpg
        file_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(folder_path)
            if f.endswith(".jpg") and not f.endswith("gt.jpg") and not f.endswith("Pred.jpg")
        ]
        if not args.no_shuffle:
            random.shuffle(file_names)

        # Collect all tasks
        tasks = []
        for file_name in file_names:
            input_json_path = os.path.join(folder_path, f"{file_name}.json")
            output_image_path = os.path.join(folder_path, f"{file_name}_Agent_v{args.ver}_{args.MLLM}_Pred.png")
            output_json_path = os.path.join(folder_path, f"{file_name}_Agent_v{args.ver}_{args.MLLM}_Pred.json")
            agent_history_path = os.path.join(folder_path, f"{file_name}_Agent_v{args.ver}_{args.MLLM}_History.json")
            
            tasks.append((file_name, folder_path, input_json_path, output_image_path, 
                         output_json_path, agent_history_path, args.overwrite_existing_preds))

        # Process tasks with multiprocessing
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_reasonseg_item, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Processing ReasonSeg {split}"):
                    result = future.result()
                    print(result)
        else:
            for task in tqdm(tasks, desc=f"Processing ReasonSeg {split}"):
                result = process_reasonseg_item(task)
                print(result)


def process_refcoco_item(task_data):
    """Worker function for processing a single RefCOCO item"""
    (image_id, ref_id, image_path, text_prompt, text_prompt_folder_path,
     output_json_path, output_image_path, agent_history_path, overwrite_existing_preds) = task_data
    
    # Check if output JSON exists and skip if overwrite is False
    if os.path.exists(output_json_path) and not overwrite_existing_preds:
        return f"Output JSON {output_json_path} already exists and overwrite is False. Skipping image_{image_id}_prompt_{text_prompt}."

    #try:
    os.makedirs(text_prompt_folder_path, exist_ok=True)

    print(f"\n\n\n--------------Processing image {image_id} with prompt: {text_prompt}--------------\n")
    agent_history, final_output_dict, rendered_final_output = agent_inference(image_path, text_prompt, send_generate_request=send_generate_request, call_sam_service=call_sam_service)
    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_id"] = image_id
    
    json.dump(final_output_dict, open(output_json_path, 'w'), indent=4)
    json.dump(agent_history, open(agent_history_path, 'w'), indent=4)
    rendered_final_output.save(output_image_path)
    
    return f"Successfully processed image {image_id} with ref {ref_id}\nOutput JSON: {output_json_path}✅\nOutput Image: ✅{output_image_path}"
    #except Exception as e:
    #    return f"Error processing image {image_id}: {str(e)}"


def process_refcocog_item(task_data):
    """Worker function for processing a single RefCOCOg item with splitBy and split info"""
    (image_id, ref_id, image_path, text_prompt, text_prompt_folder_path,
     output_json_path, output_image_path, agent_history_path, overwrite_existing_preds, splitBy, split) = task_data
    
    # Check if output JSON exists and skip if overwrite is False
    if os.path.exists(output_json_path) and not overwrite_existing_preds:
        return f"Output JSON {output_json_path} already exists and overwrite is False. Skipping image_{image_id}_prompt_{text_prompt}."

    #try:
    os.makedirs(text_prompt_folder_path, exist_ok=True)

    print(f"\n\n\n--------------Processing {splitBy} {split} image {image_id} with prompt: {text_prompt}--------------\n")
    agent_history, final_output_dict, rendered_final_output = agent_inference(image_path, text_prompt, send_generate_request=send_generate_request, call_sam_service=call_sam_service)
    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_id"] = image_id
    final_output_dict["splitBy"] = splitBy
    final_output_dict["split"] = split
    
    json.dump(final_output_dict, open(output_json_path, 'w'), indent=4)
    json.dump(agent_history, open(agent_history_path, 'w'), indent=4)
    rendered_final_output.save(output_image_path)
    
    return f"Successfully processed {splitBy} {split} image {image_id} with ref {ref_id}\nOutput JSON: {output_json_path}✅\nOutput Image: ✅{output_image_path}"
    #except Exception as e:
    #    return f"Error processing {splitBy} {split} image {image_id}: {str(e)}"


def run_refcoco_base_inference(args, dataset_name):
    """Run inference on RefCOCO/RefCOCO+ datasets - completely standalone"""
    if REFER is None:
        raise ImportError("refer not available. Cannot run RefCOCO inference.")
    
    data_root = f'{DEFAULT_USER_FOLDER}/datasets/refcoco/refer/data'
    output_root = f'{DEFAULT_USER_FOLDER}/datasets/refcoco/out'
    
    # Handle splitBy - default to "unc" for refcoco/refcoco+
    if args.splitBy is None:
        splitBy = "unc"
    else:
        splitBy = args.splitBy[0] if isinstance(args.splitBy, list) else args.splitBy
    
    # Validate splits for refcoco/refcoco+
    allowed_splits = {"val", "testA", "testB"}
    valid_splits = []
    for s in args.split:
        if s in allowed_splits:
            valid_splits.append(s)
        else:
            print(f"Warning: split '{s}' not in {sorted(allowed_splits)}; skipping.")
    
    image_folder_path = os.path.join(data_root, 'images/mscoco/images/train2014')
    output_base = os.path.join(output_root, dataset_name)
    
    refer = REFER(data_root, dataset_name, splitBy)
    
    for split in valid_splits:
        output_folder_path = os.path.join(output_base, split)
        ref_ids = sorted(refer.getRefIds(split=split))
        if args.num is not None:
            ref_ids = ref_ids[:args.num]
        if not args.no_shuffle:
            random.shuffle(ref_ids)

        # Collect all tasks for refcoco/refcoco+
        tasks = []
        for ref_id in ref_ids:
            ref = refer.Refs[ref_id]
            image_id, ref_id_str = str(ref['image_id']), str(ref_id)
            image_path = os.path.join(image_folder_path, refer.Imgs[ref['image_id']]['file_name'])
            
            for sent in ref['sentences']:
                text_prompt = sent['sent']
                text_prompt_for_save_path = text_prompt.replace("/", "_") if "/" in text_prompt else text_prompt

                text_prompt_folder_path = os.path.join(output_folder_path, image_id, ref_id_str, text_prompt_for_save_path)
                output_json_path = os.path.join(text_prompt_folder_path, f"Agent_v{args.ver}_{args.MLLM}_Pred.json")
                output_image_path = os.path.join(text_prompt_folder_path, f"Agent_v{args.ver}_{args.MLLM}_Pred.png")
                agent_history_path = os.path.join(text_prompt_folder_path, f"Agent_v{args.ver}_{args.MLLM}_History.json")

                tasks.append((image_id, ref_id_str, image_path, text_prompt, text_prompt_folder_path,
                            output_json_path, output_image_path, agent_history_path, args.overwrite_existing_preds))

        # Process tasks for refcoco/refcoco+
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_refcoco_item, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Processing {dataset_name} {split}"):
                    result = future.result()
                    print(result)
        else:
            for task in tqdm(tasks, desc=f"Processing {dataset_name} {split}"):
                result = process_refcoco_item(task)
                print(result)


def run_refcocog_inference(args):
    """Run inference on RefCOCOg dataset - completely standalone"""
    if REFER is None:
        raise ImportError("refer not available. Cannot run RefCOCOg inference.")
    
    refcocog_data_root = f'{DEFAULT_USER_FOLDER}/datasets/refcoco/refer/data'
    refcocog_output_root = f'{DEFAULT_USER_FOLDER}/datasets/refcoco/out'
    refcocog_dataset_name = "refcocog"
    
    # RefCOCOg specific splitBy handling
    refcocog_splitBys = []
    if args.splitBy is None:
        refcocog_splitBys = ["umd", "google"]
    else:
        refcocog_splitBys = args.splitBy if isinstance(args.splitBy, list) else [args.splitBy]
    
    # RefCOCOg specific split handling
    refcocog_splits_arg = args.split if isinstance(args.split, list) else args.split
    
    refcocog_image_folder_path = os.path.join(refcocog_data_root, 'images/mscoco/images/train2014')
    
    for refcocog_splitBy in refcocog_splitBys:
        refcocog_refer = REFER(refcocog_data_root, refcocog_dataset_name, refcocog_splitBy)
        
        if refcocog_splits_arg is None:
            # RefCOCOg default behavior: umd -> val,test; google -> val
            refcocog_pairs = [("val", refcocog_splitBy)] + ([("test", refcocog_splitBy)] if refcocog_splitBy == "umd" else [])
        else:
            # Run provided splits for each splitBy in RefCOCOg
            refcocog_pairs = [(s, refcocog_splitBy) for s in (refcocog_splits_arg if isinstance(refcocog_splits_arg, list) else [refcocog_splits_arg])]
        
        for refcocog_split, refcocog_sb in refcocog_pairs:
            print(f"\n=== Processing RefCOCOg {refcocog_sb} {refcocog_split} set ===")
            refcocog_output_folder_path = os.path.join(refcocog_output_root, refcocog_dataset_name, f"{refcocog_sb}_{refcocog_split}")
            
            refcocog_ref_ids = sorted(refcocog_refer.getRefIds(split=refcocog_split))
            if args.num is not None:
                refcocog_ref_ids = refcocog_ref_ids[:args.num]
            if not args.no_shuffle:
                random.shuffle(refcocog_ref_ids)

            # Collect all RefCOCOg tasks
            refcocog_tasks = []
            for refcocog_ref_id in refcocog_ref_ids:
                refcocog_ref = refcocog_refer.Refs[refcocog_ref_id]
                refcocog_image_id, refcocog_ref_id_str = str(refcocog_ref['image_id']), str(refcocog_ref_id)
                refcocog_image_path = os.path.join(refcocog_image_folder_path, refcocog_refer.Imgs[refcocog_ref['image_id']]['file_name'])
                
                # Process all text prompts for RefCOCOg
                for refcocog_sent in refcocog_ref['sentences']:
                    refcocog_text_prompt = refcocog_sent['sent']
                    refcocog_text_prompt_for_save_path = refcocog_text_prompt.replace("/", "_") if "/" in refcocog_text_prompt else refcocog_text_prompt

                    refcocog_text_prompt_folder_path = os.path.join(refcocog_output_folder_path, refcocog_image_id, refcocog_ref_id_str, refcocog_text_prompt_for_save_path)
                    refcocog_output_json_path = os.path.join(refcocog_text_prompt_folder_path, f"Agent_v{args.ver}_{args.MLLM}_Pred.json")
                    refcocog_output_image_path = os.path.join(refcocog_text_prompt_folder_path, f"Agent_v{args.ver}_{args.MLLM}_Pred.png")
                    refcocog_agent_history_path = os.path.join(refcocog_text_prompt_folder_path, f"Agent_v{args.ver}_{args.MLLM}_History.json")

                    # RefCOCOg specific task data with splitBy and split info
                    refcocog_task_data = (refcocog_image_id, refcocog_ref_id_str, refcocog_image_path, refcocog_text_prompt, refcocog_text_prompt_folder_path,
                                       refcocog_output_json_path, refcocog_output_image_path, refcocog_agent_history_path, args.overwrite_existing_preds, refcocog_sb, refcocog_split)
                    refcocog_tasks.append(refcocog_task_data)

            # Process RefCOCOg tasks with multiprocessing
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    refcocog_futures = [executor.submit(process_refcocog_item, task) for task in refcocog_tasks]
                    for refcocog_future in tqdm(as_completed(refcocog_futures), total=len(refcocog_tasks), desc=f"Processing RefCOCOg {refcocog_sb}_{refcocog_split}"):
                        refcocog_result = refcocog_future.result()
                        print(refcocog_result)
            else:
                for refcocog_task in tqdm(refcocog_tasks, desc=f"Processing RefCOCOg {refcocog_sb}_{refcocog_split}"):
                    refcocog_result = process_refcocog_item(refcocog_task)
                    print(refcocog_result)


def run_single_image_inference(args):
    """Run inference on a single image with provided prompt"""
    image_path = args.image
    text_prompt = args.prompt
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create output directory
    output_dir = f"{DEFAULT_USER_FOLDER}/code/out/single_image_inference"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file names
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    prompt_for_filename = text_prompt.replace("/", "_").replace(" ", "_")
    
    base_filename = f"{image_basename}_{prompt_for_filename}_Agent_v{args.ver}_{args.MLLM}"
    output_json_path = os.path.join(output_dir, f"{base_filename}_Pred.json")
    output_image_path = os.path.join(output_dir, f"{base_filename}_Pred.png")
    agent_history_path = os.path.join(output_dir, f"{base_filename}_History.json")
    
    # Check if output already exists and skip if overwrite is False
    if os.path.exists(output_json_path) and not args.overwrite_existing_preds:
        print(f"Output JSON {output_json_path} already exists and overwrite is False. Skipping.")
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
        
    #except Exception as e:
    #    print(f"❌ Error processing single image: {str(e)}")
    #    raise


def main():
    parser = argparse.ArgumentParser(description="Run agent inference on datasets or single images.")
    
    # Create mutually exclusive group for dataset vs single image mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dataset",
        type=str,
        choices=["omnilabel", "reasonseg", "refcoco", "refcoco+", "refcocog"],
        help="Dataset to run inference on"
    )
    mode_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file for inference"
    )
    
    # Single image mode arguments
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for single image inference (required when using --image)"
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+", 
        default=None,
        help="Which splits to use (depends on dataset)"
    )
    parser.add_argument(
        "--splitBy",
        type=str,
        nargs="+",
        default=None,
        help="Split type for RefCOCO datasets (unc for refcoco/refcoco+, umd/google for refcocog). Can specify multiple values."
    )
    parser.add_argument(
        "--ver",
        type=str,
        default="5.23",
        help="Version of the agent being used (e.g., 2.3, 2.5)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="qwen72b",
        help="Version of the LLM model to use (e.g., qwen72b, llama4scout)"
    )
    parser.add_argument(
        "--server-host",
        default="localhost",
        help="Host where the SAM3 server is running (default: localhost)"
    )
    parser.add_argument(
        "--server-port", 
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
    parser.add_argument(
        "--overwrite_existing_preds",
        action="store_true",
        help="Whether to overwrite existing prediction JSON files. Default is False."
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Whether to shuffle the input images before processing. Default is False."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of samples to run inference on (for RefCOCO datasets)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel processing. Default is 1 (no multiprocessing)."
    )
    
    args = parser.parse_args()
    
    # Validate single image mode requirements
    if args.image and not args.prompt:
        parser.error("--prompt is required when using --image")
    
    # Set up global server URLs
    global sam_server_url, llm_server_url, send_generate_request, call_sam_service
    sam_server_url = f"http://{args.server_host}:{args.server_port}/segment"
    llm_server_url = f"http://{args.llm_host}:{args.llm_port}"
    send_generate_request = partial(send_llama_request, server_url=llm_server_url, model=get_full_model_name(args.model))
    call_sam_service = partial(call_sam_service_orig, server_url=sam_server_url)
    
    # Convert model name and set MLLM identifier for backward compatibility
    model_str = args.model.split("/")[-1].replace("-", "_")
    args.MLLM = model_str
    
    print(f"Using SAM3 server: {sam_server_url}")
    print(f"Using LLM server: {llm_server_url}")
    print(f"Using model: {args.model}")
    print(f"Using {args.workers} worker(s)")
    
    # Handle single image mode
    if args.image:
        run_single_image_inference(args)
        return
    
    # Handle dataset mode
    # Set default splits if not provided
    if args.split is None:
        if args.dataset == "omnilabel":
            args.split = ["val"]
        elif args.dataset == "reasonseg":
            args.split = ["val", "test"]
        elif args.dataset in ["refcoco", "refcoco+"]:
            args.split = ["val", "testA", "testB"]
        elif args.dataset == "refcocog":
            args.split = ["val", "test"]
    
    # Validate splits based on dataset
    valid_splits = {
        "omnilabel": ["val"],
        "reasonseg": ["val", "test"],
        "refcoco": ["val", "testA", "testB"],
        "refcoco+": ["val", "testA", "testB"],
        "refcocog": ["val", "test"]
    }
    
    for split in args.split:
        if split not in valid_splits[args.dataset]:
            raise ValueError(f"Split '{split}' is not valid for dataset '{args.dataset}'. Valid splits: {valid_splits[args.dataset]}")
    
    # Run the appropriate dataset inference function
    if args.dataset == "omnilabel":
        run_omnilabel_inference(args)
    elif args.dataset == "reasonseg":
        run_reasonseg_inference(args)
    elif args.dataset in ["refcoco", "refcoco+"]:
        run_refcoco_base_inference(args, args.dataset)
    elif args.dataset == "refcocog":
        run_refcocog_inference(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
