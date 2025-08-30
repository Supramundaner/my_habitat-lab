"""
Step 4: LLM room selection using visual analysis.
Enhanced version that uses both room annotation and goal image for better selection.
"""

import os
import json
import base64
from typing import Dict, Any, List

try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    print("Warning: volcenginesdkarkruntime not found. Please install with: pip install volcenginesdkarkruntime")
    Ark = None

def encode_image(image_path):
    """Encode image as base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_with_images(client, room_annotation_path, goal_image_path, prompt_text, model):
    """Call LLM API with multiple images"""
    # Encode images
    room_annotation_base64 = encode_image(room_annotation_path)
    goal_image_base64 = encode_image(goal_image_path)
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{goal_image_base64}"
                        },
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{room_annotation_base64}"
                        },
                    },
                ],
            },
        ],
    )
    return response

def parse_llm_response(response):
    """Parse LLM response to extract room number"""
    full_response = response.choices[0].message.content
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    
    # Try to extract room number from the response
    lines = full_response.split('\\n')
    content = ""
    
    # Look for "Final Answer:" or just a number at the end
    for line in reversed(lines):
        line = line.strip()
        if line.lower().startswith('final answer:'):
            # Extract number after "Final Answer:"
            answer_part = line.split(':', 1)[1].strip()
            if answer_part.isdigit():
                content = answer_part
                break
        elif line.isdigit():
            content = line
            break
    
    return {
        "reasoning_content": reasoning_content,
        "content": content
    }

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_available_rooms(output_dir: str) -> List[int]:
    """Get available room numbers from room segmentation results."""
    try:
        results_path = os.path.join(output_dir, "room_segmentation_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [room['room_id'] for room in data['room_annotations']]
    except Exception as e:
        print(f"Warning: Failed to load room data: {e}")
    
    return []

def select_room_with_llm(topdown_path: str, room_annotation_path: str, goal_image_path: str,
                        config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Use LLM to select target room from room annotation and goal image.
    
    Args:
        topdown_path: Path to topdown view image
        room_annotation_path: Path to room annotation image
        goal_image_path: Path to goal image
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with LLM response and results
    """
    print(f"📁 Loading images for LLM analysis:")
    print(f"  - Topdown: {topdown_path}")
    print(f"  - Room annotation: {room_annotation_path}")
    print(f"  - Goal image: {goal_image_path}")
    
    # Check if images exist
    for path, name in [(topdown_path, "topdown"), (room_annotation_path, "room annotation"), (goal_image_path, "goal image")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name.capitalize()} image not found: {path}")
    
    # Get available rooms for validation
    available_rooms = get_available_rooms(output_dir)
    if not available_rooms:
        raise ValueError("No rooms found in segmentation results")
    else:
        print(f"📋 Available rooms: {available_rooms}")
    
    if Ark is None:
        # Fallback to manual selection (first room)
        print("⚠️ LLM not available, using fallback room selection")
        selected_room = available_rooms[0] if available_rooms else 1
        
        return {
            "raw_response": None,
            "selected_room": selected_room,
            "model_used": "fallback",
            "available_rooms": available_rooms,
            "attempts_made": 1,
            "all_responses": []
        }
    
    # Get LLM configuration
    llm_config = config['llm_config']
    api_key = llm_config['api_key']
    base_url = llm_config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3')
    model = llm_config.get('model', 'doubao-seed-1-6-250615')
    max_retries = llm_config.get('max_retries', 3)
    
    # Load prompt template
    prompt_path = config['prompts']['choose_room_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    # Get goal object and replace placeholder
    goal_object = config['scene_config']['goal_object']
    prompt_template = prompt_template.format(goal_object=goal_object)
    
    # Enhanced prompt for image-instance navigation
    enhanced_prompt = f"""I need you to analyze these images to select the best room for finding a specific object instance.

The first image shows the target object that I need to find - this is the exact instance I'm looking for.

The second image shows a room layout with different rooms numbered and colored. Each room is labeled with a number.

Your task is to identify which room would most likely contain the target object shown in the first image.

Consider:
1. The type and characteristics of the target object
2. Typical room types where such objects are found
3. The layout and apparent function of each numbered room
4. Size and accessibility of rooms

Available rooms: {available_rooms}

Please provide your reasoning and then give your final answer as just the room number.

Format: Final Answer: [room_number]"""
    
    print(f"🤖 LLM Configuration:")
    print(f"  - Model: {model}")
    print(f"  - Max retries: {max_retries}")
    print(f"  - Goal object: {goal_object}")
    print(f"  - Prompt from: {prompt_path}")
    
    # Initialize Ark client
    client = Ark(
        api_key=api_key,
        base_url=base_url
    )
    
    # Retry logic
    selected_room = None
    retry_count = 0
    all_responses = []
    
    while retry_count < max_retries and selected_room is None:
        retry_count += 1
        print(f"\\n🔄 LLM attempt {retry_count}/{max_retries}")
        
        try:
            # Call LLM
            response = call_llm_with_images(
                client, room_annotation_path, goal_image_path, enhanced_prompt, model
            )
            
            # Parse response
            parsed_response = parse_llm_response(response)
            all_responses.append(parsed_response)
            
            print(f"📝 LLM response: {parsed_response['content']}")
            
            # Validate room number
            try:
                room_num = int(parsed_response['content'])
                if room_num in available_rooms:
                    selected_room = room_num
                    print(f"✅ Valid room selected: {room_num}")
                else:
                    print(f"❌ Invalid room number: {room_num} (not in available rooms: {available_rooms})")
            except (ValueError, TypeError):
                print(f"❌ Could not parse room number from response: {parsed_response['content']}")
                
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            all_responses.append({"error": str(e)})
    
    # If all retries failed, use fallback
    if selected_room is None:
        print(f"⚠️ All LLM attempts failed, using fallback room: {available_rooms[0]}")
        selected_room = available_rooms[0] if available_rooms else 1
        final_response = {"error": "All LLM attempts failed, using fallback"}
    else:
        final_response = all_responses[-1] if all_responses else {}
        print(f"✅ Successfully selected room: {selected_room}")
    
    # Save LLM interaction log
    llm_log = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "model": model,
        "max_retries": max_retries,
        "attempts_made": retry_count,
        "available_rooms": available_rooms,
        "prompt_template": enhanced_prompt,
        "all_responses": all_responses,
        "final_selected_room": selected_room,
        "images_used": {
            "topdown_view": topdown_path,
            "room_annotation": room_annotation_path,
            "goal_image": goal_image_path
        }
    }
    
    log_path = os.path.join(output_dir, "llm_room_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(llm_log, f, indent=2, ensure_ascii=False)
    print(f"✓ LLM interaction log saved to: {log_path}")
    
    # Save LLM content for preprocessing
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    if selected_room is not None and all_responses:
        final_content = {
            "reasoning": final_response.get("reasoning_content", ""),
            "content": final_response.get("content", str(selected_room)),
            "selected_room": selected_room,
            "success": True
        }
    else:
        final_content = {
            "reasoning": "LLM selection failed, used fallback",
            "content": str(selected_room),
            "selected_room": selected_room,
            "success": False
        }
    
    content_path = os.path.join(preprocess_dir, "llm_room_selection_content.json")
    with open(content_path, 'w', encoding='utf-8') as f:
        json.dump(final_content, f, indent=2, ensure_ascii=False)
    print(f"✓ LLM content saved to preprocessing directory: {content_path}")
    
    return {
        "generated_files": {
            "llm_log": log_path,
            "llm_content": content_path
        },
        "results": {
            "selected_room": selected_room,
            "attempts_made": retry_count,
            "llm_success": selected_room is not None and len(all_responses) > 0,
            "available_rooms": available_rooms
        },
        "llm_responses": {
            "room_selection": {
                "selected_room": selected_room,
                "response": final_response
            }
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python step_4_llm_room_selection.py <topdown_path> <room_annotation_path> <goal_image_path> <config_json>")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    room_annotation_path = sys.argv[2]
    goal_image_path = sys.argv[3]
    config_path = sys.argv[4]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    result = select_room_with_llm(topdown_path, room_annotation_path, goal_image_path, config, output_dir)
    print("Step 4 completed:", result)
