"""
Step 6: Navigation node selection within the target room.
Crops the graph to the selected room and uses LLM to choose optimal navigation node.
Enhanced for image-instance navigation with goal image guidance.
"""

import os
import cv2
import json
import base64
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    print("Warning: volcenginesdkarkruntime not found. Please install with: pip install volcenginesdkarkruntime")
    Ark = None

def encode_image(image_path):
    """Encode image as base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_with_images(client, original_image_path, nodes_image_path, goal_image_path, prompt_text, model):
    """Call LLM API with three images"""
    # Encode images
    original_image_base64 = encode_image(original_image_path)
    nodes_image_base64 = encode_image(nodes_image_path)
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
                            "url": f"data:image/png;base64,{original_image_base64}"
                        },
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{nodes_image_base64}"
                        },
                    },
                ],
            },
        ],
    )
    return response

def parse_llm_response(response):
    """Parse LLM response to extract node number"""
    full_response = response.choices[0].message.content
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    
    # Try to extract node number
    lines = full_response.split('\\n')
    content = ""
    
    # Look for various answer formats with more flexible logic
    import re
    for line in reversed(lines):
        line = line.strip()
        if line.lower().startswith('final answer:'):
            answer_part = line.split(':', 1)[1].strip()
            if answer_part.isdigit():
                content = answer_part
                break
        elif line.isdigit():
            content = line
            break
        # Try to match common answer formats with keywords
        if any(keyword in line.lower() for keyword in ["答案", "answer", "node", "节点", "选择", "select"]):
            # Extract numbers from the line
            numbers = re.findall(r'\d+', line)
            if numbers:
                content = numbers[-1]  # Take the last number found
                break
    
    return {
        "reasoning_content": reasoning_content,
        "content": content
    }

def crop_image_to_bbox(image: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
    """Crop image to bounding box."""
    x_min = max(0, bbox['x_min'])
    y_min = max(0, bbox['y_min']) 
    x_max = min(image.shape[1], bbox['x_max'])
    y_max = min(image.shape[0], bbox['y_max'])
    
    return image[y_min:y_max, x_min:x_max]

def get_nodes_in_bbox(nodes_data: List[Dict], bbox: Dict[str, int]) -> List[Dict]:
    """Get nodes that fall within the bounding box."""
    nodes_in_room = []
    
    for node in nodes_data:
        # Handle both list format [x, y] and dict format {"x": x, "y": y}
        pixel_coords = node["pixel_coordinates"]
        if isinstance(pixel_coords, list):
            x, y = pixel_coords[0], pixel_coords[1]
        else:
            x = pixel_coords["x"]
            y = pixel_coords["y"]
        
        if (bbox['x_min'] <= x <= bbox['x_max'] and 
            bbox['y_min'] <= y <= bbox['y_max']):
            nodes_in_room.append(node)
    
    return nodes_in_room

def draw_nodes_on_cropped_image(cropped_image: np.ndarray, nodes_in_room: List[Dict], 
                               bbox: Dict[str, int], node_radius: int = 8) -> np.ndarray:
    """Draw nodes on cropped image with adjusted coordinates."""
    result_image = cropped_image.copy()
    
    for node in nodes_in_room:
        # Handle both list format [x, y] and dict format {"x": x, "y": y}
        pixel_coords = node["pixel_coordinates"]
        if isinstance(pixel_coords, list):
            x, y = pixel_coords[0], pixel_coords[1]
        else:
            x = pixel_coords["x"]
            y = pixel_coords["y"]
            
        # Adjust coordinates relative to crop
        x = int(x - bbox['x_min'])
        y = int(y - bbox['y_min'])
        node_id = node["node_id"]
        
        # Ensure coordinates are within cropped image bounds
        if 0 <= x < result_image.shape[1] and 0 <= y < result_image.shape[0]:
            # Draw node circle
            cv2.circle(result_image, (x, y), node_radius, (0, 0, 255), -1)  # Red filled circle
            
            # Draw node ID
            cv2.putText(result_image, str(node_id), (x + 10, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def select_node_with_llm(original_room_image: np.ndarray, 
                         room_image_with_nodes: np.ndarray,
                         goal_image_path: str,
                         nodes_in_room: List[Dict], 
                         config: Dict[str, Any], 
                         output_dir: str) -> Dict[str, Any]:
    """Use LLM to select optimal navigation node using three images."""
    
    if Ark is None:
        # Fallback to first available node
        if nodes_in_room:
            selected_node = nodes_in_room[0]
            return {
                "raw_response": None,
                "selected_node_id": selected_node['node_id'],
                "selected_node_data": selected_node,
                "model_used": "fallback",
                "attempts_made": 1,
                "all_responses": []
            }
        else:
            raise ValueError("No nodes available in room and no LLM fallback possible")
    
    # Get available node IDs
    available_node_ids = [node['node_id'] for node in nodes_in_room]
    
    # LLM configuration
    llm_config = config['llm_config']
    api_key = llm_config['api_key']
    base_url = llm_config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3')
    model = llm_config.get('model', 'doubao-seed-1-6-250615')
    max_retries = llm_config.get('max_retries', 3)
    
    goal_object = config['scene_config']['goal_object']
    
    # Enhanced prompt for image-instance navigation
    enhanced_prompt = f"""I need you to analyze these three images to select the best navigation node for finding a specific object instance.

The first image shows the target object instance that I need to find - this is the exact object I'm looking for.

The second image shows the room layout without navigation nodes.

The third image shows the same room with numbered navigation nodes (red circles with numbers).

Your task is to select the navigation node that would provide the best position to find and reach the target object shown in the first image.

Consider:
1. The likely location of the target object based on its type and typical placement
2. Proximity to areas where such objects are commonly found
3. Clear line of sight and accessibility to potential object locations
4. Room layout and obstacles that might block access
5. Strategic positioning for searching the room effectively

Available nodes: {available_node_ids}

Please provide your detailed analysis and reasoning, then give your final answer as just the node number.

Format: Final Answer: [node_number]"""
    
    print(f"🤖 Using LLM for node selection:")
    print(f"  - Model: {model}")
    print(f"  - Goal object: {goal_object}")
    print(f"  - Available nodes: {available_node_ids}")
    print(f"  - Max retries: {max_retries}")
    
    # Save images temporarily for LLM
    temp_original_path = os.path.join(output_dir, "temp_room_original.png")
    temp_nodes_path = os.path.join(output_dir, "temp_room_with_nodes.png")
    
    cv2.imwrite(temp_original_path, original_room_image)
    cv2.imwrite(temp_nodes_path, room_image_with_nodes)
    
    # Initialize client
    client = Ark(
        api_key=api_key,
        base_url=base_url
    )
    
    # Retry logic
    selected_node_id = None
    selected_node_data = None
    retry_count = 0
    all_responses = []
    
    while retry_count < max_retries and selected_node_id is None:
        retry_count += 1
        print(f"\\n🔄 LLM attempt {retry_count}/{max_retries}")
        
        try:
            # Call LLM
            response = call_llm_with_images(
                client, temp_original_path, temp_nodes_path, goal_image_path,
                enhanced_prompt, model
            )
            
            # Parse response
            parsed_response = parse_llm_response(response)
            all_responses.append(parsed_response)
            
            print(f"📝 LLM response: {parsed_response['content']}")
            
            # Validate node ID
            try:
                node_id = int(parsed_response['content'])
                if node_id in available_node_ids:
                    selected_node_id = node_id
                    selected_node_data = next(node for node in nodes_in_room if node['node_id'] == node_id)
                    print(f"✅ Valid node selected: {node_id}")
                else:
                    print(f"❌ Invalid node ID: {node_id} (not in available nodes: {available_node_ids})")
            except (ValueError, TypeError):
                print(f"❌ Could not parse node ID from response: {parsed_response['content']}")
                
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            all_responses.append({"error": str(e)})
    
    # Clean up temporary files
    for temp_path in [temp_original_path, temp_nodes_path]:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # If all retries failed, use fallback
    if selected_node_id is None:
        print(f"⚠️ All LLM attempts failed, using first available node: {available_node_ids[0]}")
        selected_node_id = available_node_ids[0]
        selected_node_data = nodes_in_room[0]
        final_response = {"error": "All LLM attempts failed, using fallback"}
    else:
        final_response = all_responses[-1] if all_responses else {}
        print(f"✅ Successfully selected node: {selected_node_id}")
    
    return {
        "raw_response": final_response,
        "selected_node_id": selected_node_id,
        "selected_node_data": selected_node_data,
        "model_used": model,
        "available_nodes": nodes_in_room,
        "attempts_made": retry_count,
        "all_responses": all_responses
    }

def select_navigation_node(graph_path: str, topdown_path: str, goal_image_path: str, 
                          room_bbox: Dict[str, int], selected_room: int, 
                          config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Select navigation node within the target room using LLM guidance.
    
    Args:
        graph_path: Path to graph with topdown view image
        topdown_path: Path to original topdown view image
        goal_image_path: Path to goal image
        room_bbox: Bounding box of the selected room
        selected_room: Selected room number
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading files for node selection:")
    print(f"  - Graph: {graph_path}")
    print(f"  - Topdown: {topdown_path}")
    print(f"  - Goal image: {goal_image_path}")
    print(f"  - Selected room: {selected_room}")
    print(f"  - Room bbox: {room_bbox}")
    
    # Load images
    graph_image = cv2.imread(graph_path)
    topdown_image = cv2.imread(topdown_path)
    
    if graph_image is None:
        raise FileNotFoundError(f"Graph image not found: {graph_path}")
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    if not os.path.exists(goal_image_path):
        raise FileNotFoundError(f"Goal image not found: {goal_image_path}")
    
    # Load navigation nodes data
    nodes_path = os.path.join(output_dir, "navigation_nodes.json")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Navigation nodes data not found: {nodes_path}")
    
    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)
    
    # Get nodes in the selected room
    nodes_in_room = get_nodes_in_bbox(nodes_data['nodes'], room_bbox)
    
    if len(nodes_in_room) == 0:
        raise ValueError(f"No navigation nodes found in room {selected_room}")
    
    print(f"📊 Found {len(nodes_in_room)} nodes in room {selected_room}")
    
    # Crop images to room area
    cropped_original = crop_image_to_bbox(topdown_image, room_bbox)
    cropped_with_nodes = crop_image_to_bbox(graph_image, room_bbox)
    
    # Enhance the cropped image with nodes
    enhanced_nodes_image = draw_nodes_on_cropped_image(cropped_original, nodes_in_room, room_bbox)
    
    # Save cropped images
    cropped_original_path = os.path.join(output_dir, f"room_{selected_room}_original.png")
    cropped_nodes_path = os.path.join(output_dir, f"room_{selected_room}_with_nodes.png")
    
    cv2.imwrite(cropped_original_path, cropped_original)
    cv2.imwrite(cropped_nodes_path, enhanced_nodes_image)
    
    print(f"✓ Room images saved:")
    print(f"  - Original: {cropped_original_path}")
    print(f"  - With nodes: {cropped_nodes_path}")
    
    # Use LLM to select the best node
    llm_result = select_node_with_llm(
        cropped_original, enhanced_nodes_image, goal_image_path,
        nodes_in_room, config, output_dir
    )
    
    selected_node = llm_result['selected_node_data']
    
    # Save node selection results
    selection_log = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "selected_room": selected_room,
        "room_bbox": room_bbox,
        "available_nodes": [node['node_id'] for node in nodes_in_room],
        "selected_node": selected_node,
        "llm_result": llm_result,
        "images_used": {
            "graph_image": graph_path,
            "topdown_image": topdown_path,
            "goal_image": goal_image_path,
            "cropped_original": cropped_original_path,
            "cropped_with_nodes": cropped_nodes_path
        }
    }
    
    log_path = os.path.join(output_dir, "node_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(selection_log, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Node selection log saved to: {log_path}")
    print(f"🎯 Selected node {selected_node['node_id']} at world coordinates {selected_node['world_coordinates']}")
    
    return {
        "generated_files": {
            "node_selection_log": log_path,
            "cropped_original": cropped_original_path,
            "cropped_with_nodes": cropped_nodes_path
        },
        "results": {
            "selected_node_id": selected_node['node_id'],
            "selected_node_world_coords": selected_node['world_coordinates'],
            "selected_room": selected_room,
            "total_nodes_in_room": len(nodes_in_room),
            "llm_attempts": llm_result['attempts_made'],
            "selection_success": True
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 7:
        print("Usage: python step_6_node_selection.py <graph_path> <topdown_path> <goal_image_path> <room_bbox_json> <selected_room> <config_json>")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    topdown_path = sys.argv[2]
    goal_image_path = sys.argv[3]
    room_bbox_json = sys.argv[4]
    selected_room = int(sys.argv[5])
    config_path = sys.argv[6]
    
    with open(room_bbox_json, 'r') as f:
        room_bbox = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    result = select_navigation_node(graph_path, topdown_path, goal_image_path, room_bbox, selected_room, config, output_dir)
    print("Step 6 completed:", result)
