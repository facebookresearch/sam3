# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import copy
import json
import os

import cv2
from PIL import Image

from .client_llm import send_generate_request
from .client_sam3 import call_sam_service
from .viz import visualize


def save_debug_messages(messages_list, debug, debug_folder_path, debug_jsonl_path):
    """Save messages to debug jsonl file if debug is enabled"""
    if debug and debug_jsonl_path:
        # Ensure the debug directory exists before writing
        os.makedirs(debug_folder_path, exist_ok=True)
        with open(debug_jsonl_path, "w") as f:
            for msg in messages_list:
                f.write(json.dumps(msg, indent=4) + "\n")


def cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path):
    """Clean up debug files when function successfully returns"""
    if debug and debug_folder_path:
        try:
            if os.path.exists(debug_jsonl_path):
                os.remove(debug_jsonl_path)
            if os.path.exists(debug_folder_path):
                os.rmdir(debug_folder_path)
        except Exception as e:
            print(f"Warning: Could not clean up debug files: {e}")


def count_images(messages):
    """Count the total number of images present in the messages history."""
    total = 0
    for message in messages:
        # Check if message has content (should be a list)
        if "content" in message and isinstance(message["content"], list):
            # Iterate through each content item
            for content_item in message["content"]:
                # Check if content item is a dict with type "image"
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                ):
                    total += 1
    return total


def _prune_messages_for_next_round(
    messages_list,
    used_text_prompts,
    latest_sam3_text_prompt,
    img_path,
    initial_text_prompt,
):
    """Return a new messages list that contains only:
    1) messages[:2] (with optional warning text added to the second message's content)
    2) the latest assistant message (and everything after it) that contains a segment_phrase tool call
    """
    # There should not be more than 10 messages in the conversation history
    assert len(messages_list) < 10

    # Part 1: always keep the first two message JSONs
    part1 = copy.deepcopy(messages_list[:2])

    # Part 2: search backwards for the latest assistant message containing a segment_phrase tool call
    part2_start_idx = None
    for idx in range(len(messages_list) - 1, 1, -1):
        msg = messages_list[idx]
        # We only consider assistant messages with a "content" list
        if msg.get("role") != "assistant" or "content" not in msg:
            continue
        # Look for any content element that is a text containing the segment_phrase tool call
        for content in msg["content"]:
            if (
                isinstance(content, dict)
                and content.get("type") == "text"
                and "<tool>" in content.get("text", "")
                and "segment_phrase" in content.get("text", "")
            ):
                part2_start_idx = idx
                break
        if part2_start_idx is not None:
            break

    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []

    # Part 3: decide whether to add warning text to the second message in part1
    previously_used = (
        [p for p in used_text_prompts if p != latest_sam3_text_prompt]
        if latest_sam3_text_prompt
        else list(used_text_prompts)
    )
    if part2 and len(previously_used) > 0:
        warning_text = f'Note that we have previously called the segment_phrase tool with each "text_prompt" in this list: {list(previously_used)}, but none of the generated results were satisfactory. So make sure that you do not use any of these phrases as the "text_prompt" to call the segment_phrase tool again.'
        # Replace the second message entirely to keep exactly 2 content items
        part1[1] = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'."
                    + " "
                    + warning_text,
                },
            ],
        }
        assert len(part1[1]["content"]) == 2

    # Build the new messages list: part1 (with optional warning), then part2
    new_messages = list(part1)
    new_messages.extend(part2)
    return new_messages


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    max_generations: int = 100,
    output_dir="../../sam3_agent_out",
):
    """
    Given a text prompt and an image, this tool will perform all aspects of agentic problem solving,
    while saving sam3 and MLLM outputs to their respective directories.

    Args:
        img_path: Path to the input image
        initial_text_prompt: Initial text prompt from the user
        debug: Whether to enable debug mode
        max_generations: Maximum number of send_generate_request calls allowed (default: 100)
    """
    # setup dir
    sam_output_dir = os.path.join(output_dir, "sam_out")
    error_save_dir = os.path.join(output_dir, "none_out")
    debug_save_dir = os.path.join(output_dir, "agent_debug_out")
    os.makedirs(sam_output_dir, exist_ok=True)
    os.makedirs(error_save_dir, exist_ok=True)
    os.makedirs(debug_save_dir, exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MLLM_SYSTEM_PROMPT_PATH = os.path.join(
        current_dir, "system_prompts/system_prompt.txt"
    )
    ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH = os.path.join(
        current_dir, "system_prompts/system_prompt_iterative_checking.txt"
    )
    # init variables
    PATH_TO_LATEST_OUTPUT_JSON = ""
    LATEST_SAM3_TEXT_PROMPT = ""
    USED_TEXT_PROMPTS = (
        set()
    )  # Track all previously used text prompts for segment_phrase
    generation_count = 0  # Counter for number of send_generate_request calls

    # debug setup
    debug_folder_path = None
    debug_jsonl_path = None
    if debug:
        debug_folder_path = os.path.join(
            debug_save_dir, f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
        )
        debug_jsonl_path = os.path.join(debug_folder_path, "debug_history.json")
        os.makedirs(debug_folder_path, exist_ok=True)

    # The helper functions are now defined outside the agent_inference function
    with open(MLLM_SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()
    with open(ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH, "r") as f:
        iterative_checking_system_prompt = f.read().strip()

    # Construct the initial message list
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'.",
                },
            ],
        },
    ]
    print(f"> Text prompt: {initial_text_prompt}")
    print(f"> Image path: {img_path}")

    print("\n\n")
    print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
    print("\n\n")
    generated_text = send_generate_request(messages)
    print(f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n")
    while generated_text is not None:
        save_debug_messages(messages, debug, debug_folder_path, debug_jsonl_path)
        assert (
            "<tool>" in generated_text,
            f"Generated text does not contain <tool> tag: {generated_text}",
        )
        generated_text = generated_text.split("</tool>", 1)[0] + "</tool>"
        tool_call_json_str = (
            generated_text.split("<tool>")[-1]
            .split("</tool>")[0]
            .strip()
            .replace(r"}}}", r"}}")  # remove extra } if any
        )
        try:
            tool_call = json.loads(tool_call_json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in tool call: {tool_call_json_str}")

        if PATH_TO_LATEST_OUTPUT_JSON == "":
            # The first tool call must be segment_phrase or report_no_mask
            assert (
                tool_call["name"] == "segment_phrase"
                or tool_call["name"] == "report_no_mask"
            )

        if tool_call["name"] == "segment_phrase":
            print("🔍 Calling segment_phrase tool...")
            assert list(tool_call["parameters"].keys()) == ["text_prompt"]

            # Check if this text_prompt has been used before
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            if current_text_prompt in USED_TEXT_PROMPTS:
                print(
                    f"❌ Text prompt '{current_text_prompt}' has been used before. Requesting a different prompt."
                )
                duplicate_prompt_message = f"You have previously used '{current_text_prompt}' as your text_prompt to call the segment_phrase tool. You may not use it again. Please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase prompt, while adhering to all the rules stated in the system prompt. You must also never use any of the following text_prompt(s): {str(list(USED_TEXT_PROMPTS))}."
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": duplicate_prompt_message}],
                    }
                )
            else:
                # Add the text_prompt to the set of used prompts
                USED_TEXT_PROMPTS.add(current_text_prompt)
                LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                PATH_TO_LATEST_OUTPUT_JSON = call_sam_service(
                    image_path=img_path,
                    text_prompt=current_text_prompt,
                    output_folder_path=sam_output_dir,
                )
                sam3_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                sam3_output_image_path = sam3_outputs["output_image_path"]
                num_masks = len(sam3_outputs["pred_boxes"])

                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                if num_masks == 0:
                    print("❌ No masks generated by SAM3, reporting no mask to Qwen.")
                    sam3_output_text_message = f"The segment_phrase tool did not generate any masks for the text_prompt '{current_text_prompt}'. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message}
                            ],
                        }
                    )
                else:
                    sam3_output_text_message = rf"The segment_phrase tool generated {num_masks} available masks. All {num_masks} available masks are rendered in this image below, now you must analyze the {num_masks} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message},
                                {"type": "image", "image": sam3_output_image_path},
                            ],
                        }
                    )
                print("\n\n>>> sam3_output_text_message:\n", sam3_output_text_message)

        elif tool_call["name"] == "examine_each_mask":
            print("🔍 Calling examine_each_mask tool...")
            assert LATEST_SAM3_TEXT_PROMPT != ""

            # Make sure that the last message is a image
            assert (
                messages[-1]["content"][1]["type"] == "image"
            ), "Second content element should be an image"
            messages.pop()  # Remove the last user message
            # Add simplified replacement message
            simplified_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The segment_phrase tool generated several masks. Now you must analyze the mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action.",
                    }
                ],
            }
            messages.append(simplified_message)

            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            num_masks = len(current_outputs["pred_masks"])
            masks_to_keep = []

            # MLLM check the mask one by one
            for i in range(num_masks):
                print(f"🔍 Checking mask {i+1}/{num_masks}...")
                image_w_mask_i, image_w_zoomed_in_mask_i = visualize(current_outputs, i)

                image_w_zoomed_in_mask_i_path = os.path.join(
                    sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                ).replace(".png", f"_zoom_in_mask_{i + 1}.png")
                image_w_mask_i_path = os.path.join(
                    sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                ).replace(".png", f"_selected_mask_{i + 1}.png")
                image_w_zoomed_in_mask_i.save(image_w_zoomed_in_mask_i_path)
                image_w_mask_i.save(image_w_mask_i_path)

                iterative_checking_messages = [
                    {"role": "system", "content": iterative_checking_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"The raw input image: "},
                            {"type": "image", "image": img_path},
                            {
                                "type": "text",
                                "text": f"The initial user input query is: '{initial_text_prompt}'",
                            },
                            {
                                "type": "text",
                                "text": f"Image with the predicted segmentation mask rendered on it: ",
                            },
                            {"type": "image", "image": image_w_mask_i_path},
                            {
                                "type": "text",
                                "text": f"Image with the zoomed-in mask: ",
                            },
                            {"type": "image", "image": image_w_zoomed_in_mask_i_path},
                        ],
                    },
                ]
                checking_generated_text = send_generate_request(
                    iterative_checking_messages
                )

                # Process the generated text to determine if the mask should be kept or rejected
                if checking_generated_text is None:
                    raise ValueError(
                        "Generated text is None, which is unexpected. Please check the Qwen server and the input parameters."
                    )
                print(f"Generated text for mask {i+1}: {checking_generated_text}")
                verdict = (
                    checking_generated_text.split("<verdict>")[-1]
                    .split("</verdict>")[0]
                    .strip()
                )
                if "Accept" in verdict:
                    assert not "Reject" in verdict
                    print(f"Mask {i+1} accepted, keeping it in the outputs.")
                    masks_to_keep.append(i)
                elif "Reject" in verdict:
                    assert not "Accept" in verdict
                    print(f"Mask {i+1} rejected, removing it from the outputs.")
                else:
                    raise ValueError(
                        f"Unexpected verdict in generated text: {checking_generated_text}. Expected 'Accept' or 'Reject'."
                    )

            updated_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
                "pred_scores": [
                    current_outputs["pred_scores"][i] for i in masks_to_keep
                ],
                "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
            }

            image_w_check_masks = visualize(updated_outputs)
            image_w_check_masks_path = os.path.join(
                sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png"
            ).replace(
                ".png",
                f"_selected_masks_{'-'.join(map(str, [i+1 for i in masks_to_keep]))}.png".replace(
                    "/", "_"
                ),
            )
            image_w_check_masks.save(image_w_check_masks_path)
            # save the updated json outputs and append to message history
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )
            if len(masks_to_keep) == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The original user query was: '{initial_text_prompt}'. The examine_each_mask tool examined and rejected all of the masks generated by the segment_phrase tool. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt.",
                            }
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The original user query was: '{initial_text_prompt}'. After calling the examine_each_mask tool on the available masks, the number of available masks is now {len(masks_to_keep)}. All {len(masks_to_keep)} available masks are rendered in this image below, now you must analyze the {len(masks_to_keep)} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action.",
                            },
                            {"type": "image", "image": image_w_check_masks_path},
                        ],
                    }
                )

            # Create a new filename based on the original path to avoid filename length issues
            base_path = PATH_TO_LATEST_OUTPUT_JSON
            # Remove any existing "masks_" suffix to avoid duplication
            if "masks_" in base_path:
                base_path = base_path.split("masks_")[0] + ".json"
            # Create new filename with current masks; use a clearer suffix when empty
            if len(masks_to_keep) == 0:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", "masks_none.json"
                )
            else:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", f"masks_{'_'.join(map(str, masks_to_keep))}.json"
                )
            json.dump(updated_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)

        elif tool_call["name"] == "select_masks_and_return":
            print("🔍 Calling select_masks_and_return tool...")
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))

            assert list(tool_call["parameters"].keys()) == ["final_answer_masks"]
            masks_to_keep = tool_call["parameters"]["final_answer_masks"]

            # Keep only valid mask indices, remove duplicates, and preserve deterministic ascending order
            available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
            masks_to_keep = sorted({i for i in masks_to_keep if i in available_masks})
            # Change this to a update message telling the model to try again along with information about errors made.

            final_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [
                    current_outputs["pred_boxes"][i - 1] for i in masks_to_keep
                ],
                "pred_scores": [
                    current_outputs["pred_scores"][i - 1] for i in masks_to_keep
                ],
                "pred_masks": [
                    current_outputs["pred_masks"][i - 1] for i in masks_to_keep
                ],
            }

            rendered_final_output = visualize(final_outputs)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )

            # Clean up debug files before successful return
            cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path)
            return messages, final_outputs, rendered_final_output

        elif tool_call["name"] == "report_no_mask":
            print("🔍 Calling report_no_mask tool...")
            height, width = cv2.imread(img_path).shape[:2]
            final_outputs = {
                "original_image_path": img_path,
                "orig_img_h": height,
                "orig_img_w": width,
                "pred_boxes": [],
                "pred_scores": [],
                "pred_masks": [],
            }
            rendered_final_output = Image.open(img_path)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )
            return messages, final_outputs, rendered_final_output

        else:
            raise ValueError(f"Unknown tool call: {tool_call['name']}")

        # sometimes the MLLM don't know when to stop, and generates multiple tool calls in one round, so we need to split the generated text by </tool> and only keep the first one

        for message in messages:
            if message["role"] == "assistant" and "content" in message:
                for content in message["content"]:
                    if (
                        isinstance(content, dict)
                        and content.get("type") == "text"
                        and "text" in content
                    ):
                        content["text"] = (
                            content["text"].split("</tool>", 1)[0] + "</tool>\n\n"
                        )
        # Prune the messages history before the next MLLM generation round according to the 3-part rules.
        # This keeps history compact and ensures the model sees only the allowed parts.
        messages = _prune_messages_for_next_round(
            messages,
            USED_TEXT_PROMPTS,
            LATEST_SAM3_TEXT_PROMPT,
            img_path,
            initial_text_prompt,
        )
        # make sure there can never be more than 2 images in the context
        assert count_images(messages) <= 2
        generation_count += 1
        if generation_count > max_generations:
            raise ValueError(
                f"Exceeded maximum number of allowed generation requests ({max_generations})"
            )

        print("\n\n")
        print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
        print("\n\n")
        generated_text = send_generate_request(messages)
        print(
            f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n"
        )

    print("\n\n>>> SAM 3 Agent execution ended.\n\n")

    error_save_path = os.path.join(
        error_save_dir,
        f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_error_history.json",
    )
    with open(error_save_path, "w") as f:
        json.dump(messages, f, indent=4)
    print("Saved messages history that caused error to:", error_save_path)
    raise ValueError(
        rf"Generated text is None, which is unexpected. Please check the Qwen server and the input parameters for image path: {img_path} and initial text prompt: {initial_text_prompt}."
    )
