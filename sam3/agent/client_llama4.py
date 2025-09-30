import requests
import json
import os
import base64
from PIL import Image
from urllib.parse import quote


def resize_image(image_path, width=1024, output_dir="/fsx-onevision/jialez/datasets/reasonseg/LISA/dataset/val_resized/"):
    """
    Resize an image to have a width of width pixels while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the original image
        output_dir (str): Directory to save the resized image
        
    Returns:
        str: Path to the resized image
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the image
        with Image.open(image_path) as img:
            # Get original dimensions
            original_width, original_height = img.size
            
            # If width is already width or smaller, no need to resize
            if original_width <= width:
                return image_path
            
            # Calculate new height to maintain aspect ratio
            new_width = width
            new_height = int((new_width * original_height) / original_width)
            
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Generate output filename
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_resized{ext}")
            
            # Save the resized image
            resized_img.save(output_path, quality=95, optimize=True)
            
            print(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            print(f"Saved resized image to: {output_path}")
            
            return output_path
            
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return image_path  # Return original path if resize fails

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
                    
                    # Resize image to width 1024 before processing
                    #resized_image_path = resize_image(image_path)
                    print("image_path", image_path)
                    resized_image_path = image_path.replace("?", "%3F")  # Escape ? in the path
                    
                    # Read the image file and convert to base64
                    try:
                        '''
                        with open(resized_image_path, "rb") as image_file:
                            image_data = image_file.read()
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        # Determine the image format from file extension
                        if resized_image_path.lower().endswith(('.png',)):
                            mime_type = "image/png"
                        elif resized_image_path.lower().endswith(('.jpg', '.jpeg')):
                            mime_type = "image/jpeg"
                        elif resized_image_path.lower().endswith(('.gif',)):
                            mime_type = "image/gif"
                        elif resized_image_path.lower().endswith(('.webp',)):
                            mime_type = "image/webp"
                        else:
                            mime_type = "image/jpeg"  # default fallback
                        '''
                        # Create the proper image_url structure
                        c["image_url"] = {
                            #"url": f"data:{mime_type};base64,{base64_image}",
                            'url': f"file://{resized_image_path}",  # Use file URL for resized images
                            "detail": "high"
                        }
                        c["type"] = "image_url"
                        c.pop("image", None)
                        
                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {resized_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {resized_image_path}: {e}")
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

if __name__ == "__main__":
    # Example usage
    server_url = "http://127.0.0.1:8000"  # Replace with your actual server URL
    
    # Path to system prompt file - update this to your actual path
    system_prompt_path = "/storage/home/jialez/code/onevision/agent/system_prompt.txt"
    
    # Read system prompt if file exists, otherwise use a default
    system_prompt = "You are a helpful AI assistant that can understand images and text."
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read().strip()
    
    # Construct the messages list as expected by the OpenAI API
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/...",  # Replace with actual base64 image or file path
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "What can you see in this image?"
                }
            ]
        },
    ]
    
    # Send the request
    generated_text = send_generate_request(server_url, messages)
    
    if generated_text:
        print("Generated response:")
        print(generated_text)
        
        # Extract thought and tool call components
        thought, tool_call = extract_components(generated_text)
        
        if thought:
            print("\nThought:")
            print(thought)
        
        if tool_call:
            print("\nTool call:")
            print(json.dumps(tool_call, indent=2))
    else:
        print("Failed to get a response from the server.")
