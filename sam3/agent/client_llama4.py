import requests
import json
import os
import base64
from PIL import Image
from urllib.parse import quote


def send_generate_request(messages, server_url=None, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", max_tokens=4096):
    """
    Sends a POST request to the OpenAI-compatible API endpoint with the given messages list.
    
    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        max_tokens (int): Maximum number of tokens to generate (default: 4096)
        
    Returns:
        str: The generated response text from the server.
    """
    # OpenAI-compatible API endpoint
    endpoint = f"{server_url}/v1/chat/completions"
    
    for message in messages:
        if message["role"] == "user":
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Convert image path to base64 format
                    image_path = c["image"]
                    
                    print("image_path", image_path)
                    new_image_path = image_path.replace("?", "%3F")  # Escape ? in the path
                    
                    # Read the image file and convert to base64
                    try:
                        # Create the proper image_url structure
                        c["image_url"] = {
                            #"url": f"data:{mime_type};base64,{base64_image}",
                            'url': f"file://{new_image_path}",  # Use file URL for resized images
                            "detail": "high"
                        }
                        c["type"] = "image_url"
                        c.pop("image", None)
                        
                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {new_image_path}: {e}")
                        continue
                    
    # Construct the payload in OpenAI format
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        #"temperature": 0.01,
    }

    print("payload:", payload)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('LLAMA4_API_KEY', 'placeholder-api-key')}"
    }
    
    #print(f"Sending request to {endpoint} with payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        print(f"Received response: {json.dumps(data, indent=2)}")
        
        # Extract the response content from the OpenAI-compatible format
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            print(f"Unexpected response format: {data}")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def extract_components(response_text):
    """
    Extracts thought and tool call components from the response text.
    
    Args:
        response_text (str): The full response text
        
    Returns:
        tuple: (thought, tool_call) extracted from the response
    """
    thought = ""
    tool_call = {}
    
    # Extract thought if present
    if "<think>" in response_text and "</think>" in response_text:
        thought = response_text.split("<think>")[-1].split("</think>")[0].strip()
    
    # Extract tool call if present
    if "<tool>" in response_text and "</tool>" in response_text:
        tool_call_str = response_text.split("<tool>")[-1].split("</tool>")[0].strip()
        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            print(f"Failed to parse tool call: {tool_call_str}")
    
    return thought, tool_call
