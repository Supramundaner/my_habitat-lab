"""
Step 3: LLM room selection using Google Gemini API with proxy.
"""

import os
import json
from typing import Dict, Any

try:
    from google import genai
    from google.genai import types
    import PIL.Image
except ImportError:
    print("Warning: google packages not found. Please install them with: pip install google-generativeai")
    genai = None

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def select_room_with_llm(topdown_path: str, room_annotation_path: str, 
                        config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Use LLM to select target room from room annotation.
    
    Args:
        topdown_path: Path to topdown view image
        room_annotation_path: Path to room annotation image
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with LLM response and results
    """
    print(f"📁 Loading images for LLM analysis:")
    print(f"  - Topdown: {topdown_path}")
    print(f"  - Room annotation: {room_annotation_path}")
    
    # Check if images exist
    if not os.path.exists(topdown_path):
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    if not os.path.exists(room_annotation_path):
        raise FileNotFoundError(f"Room annotation image not found: {room_annotation_path}")
    
    # Get LLM configuration
    llm_config = config['llm_config']
    api_key = llm_config['api_key']
    base_url = llm_config.get('base_url', 'https://api.openai-proxy.org/google/v1beta/models')
    model = llm_config.get('model', 'gemini-2.0-flash')
    max_tokens = llm_config.get('max_tokens', 1000)
    
    # Load prompt template
    prompt_path = config['prompts']['choose_room_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    print(f"🤖 LLM Configuration:")
    print(f"  - Base URL: {base_url}")
    print(f"  - Model: {model}")
    print(f"  - Max tokens: {max_tokens}")
    print(f"  - Prompt loaded from: {prompt_path}")
    
    try:
        if genai is None:
            raise RuntimeError("google packages not installed")
        
        # Set up client with proxy using environment variable and HttpOptions
        os.environ['API_KEY'] = api_key
        http_options = types.HttpOptions(base_url=base_url)
        client = genai.Client(api_key=api_key, http_options=http_options)
        
        # Load images as binary data
        print("🔄 Loading images...")
        with open(topdown_path, 'rb') as f:
            topdown_bytes = f.read()
        with open(room_annotation_path, 'rb') as f:
            room_annotation_bytes = f.read()
        
        print("🚀 Sending request to LLM...")
        
        # Create image parts using types.Part
        topdown_part = types.Part.from_bytes(
            data=topdown_bytes,
            mime_type='image/png'
        )
        room_annotation_part = types.Part.from_bytes(
            data=room_annotation_bytes,
            mime_type='image/png'
        )
        
        # Prepare prompt parts
        prompt_parts = [
            prompt_template,
            "\n\nImage 1 - Topdown view:",
            topdown_part,
            "\n\nImage 2 - Room annotation with numbers:",
            room_annotation_part
        ]
        
        # Generate content
        response = client.models.generate_content(
            model=model,
            contents=prompt_parts
        )
        
        if not response or not response.text:
            raise RuntimeError("Empty response from LLM")
        
        raw_response = response.text.strip()
        print(f"📝 Raw LLM response: '{raw_response}'")
        
        # Parse the room number from response
        selected_room = None
        try:
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'\d+', raw_response)
            if numbers:
                selected_room = int(numbers[0])
                print(f"✓ Extracted room number: {selected_room}")
            else:
                print("⚠️ No number found in LLM response")
        except Exception as e:
            print(f"⚠️ Error parsing room number: {e}")
        
        # Save LLM interaction log
        llm_log = {
            "timestamp": str(__import__('datetime').datetime.now()),
            "base_url": base_url,
            "model": model,
            "prompt_template": prompt_template,
            "raw_response": raw_response,
            "parsed_room_number": selected_room,
            "images_used": {
                "topdown_view": topdown_path,
                "room_annotation": room_annotation_path
            }
        }
        
        log_path = os.path.join(output_dir, "llm_room_selection_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(llm_log, f, indent=2, ensure_ascii=False)
        print(f"✓ LLM interaction log saved to: {log_path}")
        
        if selected_room is None:
            raise ValueError(f"Could not parse room number from LLM response: '{raw_response}'")
        
        return {
            "generated_files": {
                "llm_log": log_path
            },
            "llm_response": {
                "raw_response": raw_response,
                "selected_room": selected_room,
                "model_used": model
            },
            "results": {
                "selected_room_number": selected_room
            }
        }
        
    except Exception as e:
        error_msg = f"LLM room selection failed: {str(e)}"
        print(f"✗ {error_msg}")
        
        # Save error log
        error_log = {
            "error": error_msg,
            "model": model,
            "base_url": base_url,
            "prompt_path": prompt_path,
            "images": {
                "topdown_view": topdown_path,
                "room_annotation": room_annotation_path
            }
        }
        
        error_log_path = os.path.join(output_dir, "llm_room_selection_error.json")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)
        
        raise RuntimeError(error_msg)

# Fallback function for testing without LLM
def select_room_manually(room_annotation_path: str, output_dir: str, room_number: int = 1) -> Dict[str, Any]:
    """
    Fallback function to manually select a room for testing.
    
    Args:
        room_annotation_path: Path to room annotation image
        output_dir: Output directory path  
        room_number: Room number to select (default: 1)
        
    Returns:
        Dictionary with manual selection results
    """
    print(f"🔧 Manual room selection: Room {room_number}")
    
    manual_log = {
        "method": "manual_selection",
        "selected_room": room_number,
        "room_annotation_image": room_annotation_path
    }
    
    log_path = os.path.join(output_dir, "manual_room_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(manual_log, f, indent=2, ensure_ascii=False)
    
    return {
        "generated_files": {
            "selection_log": log_path
        },
        "llm_response": {
            "raw_response": f"Manual selection: Room {room_number}",
            "selected_room": room_number,
            "model_used": "manual"
        },
        "results": {
            "selected_room_number": room_number
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python step_3_llm_room_selection.py <topdown_path> <room_annotation_path> <config_path> [manual_room_number]")
        print("If manual_room_number is provided, manual selection will be used instead of LLM")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    room_annotation_path = sys.argv[2] 
    config_path = sys.argv[3]
    manual_room = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if manual_room is not None:
        result = select_room_manually(room_annotation_path, output_dir, manual_room)
        print("Step 3 completed (manual):", result)
    else:
        result = select_room_with_llm(topdown_path, room_annotation_path, config, output_dir)
        print("Step 3 completed (LLM):", result)
