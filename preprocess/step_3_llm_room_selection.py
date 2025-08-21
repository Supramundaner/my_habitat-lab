"""
Step 3: LLM room selection using Google Gemini API with proxy.
"""

import os
import json
from typing import Dict, Any, List

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

def get_available_rooms(output_dir: str) -> List[int]:
    """Get available room numbers from room segmentation results."""
    try:
        output_json_path = os.path.join(output_dir, "output.json")
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            room_bboxes = output_data.get("final_results", {}).get("room_bounding_boxes", {})
            return [int(room_id) for room_id in room_bboxes.keys()]
    except Exception as e:
        print(f"⚠️ Warning: Could not load available rooms from output.json: {e}")
    return []

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
    
    # Get available rooms for validation
    available_rooms = get_available_rooms(output_dir)
    if not available_rooms:
        print("⚠️ Warning: Could not determine available rooms, proceeding without validation")
    else:
        print(f"✓ Available rooms: {sorted(available_rooms)}")
    
    # Get LLM configuration
    llm_config = config['llm_config']
    api_key = llm_config['api_key']
    base_url = llm_config.get('base_url', 'https://api.openai-proxy.org/google/v1beta/models')
    model = llm_config.get('model', 'gemini-2.0-flash')
    max_tokens = llm_config.get('max_tokens', 1000)
    max_retries = llm_config.get('max_retries', 3)  # 添加重试次数配置
    
    # Load prompt template
    prompt_path = config['prompts']['choose_room_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    # Get goal object and replace placeholder
    goal_object = config['scene_config']['goal_object']
    prompt_template = prompt_template.format(goal_object=goal_object)
    
    print(f"🤖 LLM Configuration:")
    print(f"  - Base URL: {base_url}")
    print(f"  - Model: {model}")
    print(f"  - Max tokens: {max_tokens}")
    print(f"  - Max retries: {max_retries}")
    print(f"  - Goal object: {goal_object}")
    print(f"  - Prompt loaded from: {prompt_path}")
    
    # 重试逻辑
    selected_room = None
    retry_count = 0
    all_responses = []  # 记录所有响应
    
    while retry_count < max_retries and selected_room is None:
        try:
            if genai is None:
                raise RuntimeError("google packages not installed")
            
            # Set up client with proxy using environment variable and HttpOptions
            os.environ['API_KEY'] = api_key
            http_options = types.HttpOptions(base_url=base_url)
            client = genai.Client(api_key=api_key, http_options=http_options)
            
            # Load images as binary data
            print(f"🔄 Loading images... (Attempt {retry_count + 1}/{max_retries})")
            with open(topdown_path, 'rb') as f:
                topdown_bytes = f.read()
            with open(room_annotation_path, 'rb') as f:
                room_annotation_bytes = f.read()
            
            print(f"🚀 Sending request to LLM... (Attempt {retry_count + 1}/{max_retries})")
            
            # 构建prompt，包含可用房间信息
            enhanced_prompt = prompt_template
            if available_rooms:
                enhanced_prompt += f"\n\nAvailable rooms in this scene: {sorted(available_rooms)}"
                enhanced_prompt += f"\nPlease select ONLY from these available room numbers: {sorted(available_rooms)}"
            
            if retry_count > 0:
                enhanced_prompt += f"\n\nPREVIOUS ATTEMPTS FAILED: Your previous responses were not valid room numbers."
                enhanced_prompt += f"\nPlease respond with ONLY a valid room number from the available options."
            
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
                enhanced_prompt,
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
            print(f"📝 Raw LLM response (Attempt {retry_count + 1}): '{raw_response}'")
            
            # 记录响应
            all_responses.append({
                "attempt": retry_count + 1,
                "raw_response": raw_response,
                "timestamp": str(__import__('datetime').datetime.now())
            })
            
            # Parse the room number from response
            try:
                # Try to extract a number from the response
                import re
                numbers = re.findall(r'\d+', raw_response)
                if numbers:
                    candidate_room = int(numbers[0])
                    
                    # 验证房间是否在可用范围内
                    if available_rooms and candidate_room not in available_rooms:
                        print(f"⚠️ LLM selected room {candidate_room} not in available rooms {available_rooms}")
                        print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        continue
                    else:
                        selected_room = candidate_room
                        print(f"✓ Valid room number selected: {selected_room}")
                        break
                else:
                    print("⚠️ No number found in LLM response")
                    print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    continue
                    
            except Exception as e:
                print(f"⚠️ Error parsing room number: {e}")
                print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                continue
                
        except Exception as e:
            print(f"⚠️ LLM request failed (Attempt {retry_count + 1}): {e}")
            all_responses.append({
                "attempt": retry_count + 1,
                "error": str(e),
                "timestamp": str(__import__('datetime').datetime.now())
            })
            retry_count += 1
            if retry_count < max_retries:
                print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
    
    # 如果所有重试都失败了
    if selected_room is None:
        if available_rooms:
            print(f"⚠️ All LLM attempts failed. Using fallback: first available room {available_rooms[0]}")
            selected_room = available_rooms[0]
            final_response = f"Fallback selection: Room {selected_room} (LLM attempts failed)"
        else:
            raise ValueError(f"Could not parse valid room number from LLM responses after {max_retries} attempts")
    else:
        final_response = all_responses[-1]["raw_response"]
    
    # Save LLM interaction log
    llm_log = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "base_url": base_url,
        "model": model,
        "max_retries": max_retries,
        "attempts_made": retry_count,
        "available_rooms": available_rooms,
        "prompt_template": prompt_template,
        "all_responses": all_responses,
        "final_selected_room": selected_room,
        "images_used": {
            "topdown_view": topdown_path,
            "room_annotation": room_annotation_path
        }
    }
    
    log_path = os.path.join(output_dir, "llm_room_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(llm_log, f, indent=2, ensure_ascii=False)
    print(f"✓ LLM interaction log saved to: {log_path}")
    
    return {
        "generated_files": {
            "llm_log": log_path
        },
        "llm_response": {
            "raw_response": final_response,
            "selected_room": selected_room,
            "model_used": model,
            "attempts_made": retry_count,
            "all_responses": all_responses
        },
        "results": {
            "selected_room_number": selected_room
        }
    }

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
