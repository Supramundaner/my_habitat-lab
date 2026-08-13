"""
Step 5: Navigation node selection within the target room.
Crops the graph to the selected room and uses LLM to choose optimal navigation node.
"""

import base64
import cv2
import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    from byteplussdkarkruntime import Ark
except ImportError:
    print(
        "Warning: byteplussdkarkruntime is unavailable; install it with "
        "`python3 -m pip install byteplussdkarkruntime` before LLM selection"
    )
    Ark = None

if __package__:
    from .config import load_preprocessing_config, resolve_llm_config
else:
    from config import load_preprocessing_config, resolve_llm_config

def encode_image(image_path):
    """编码图片为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_with_images(
    client,
    original_image_path,
    nodes_image_path,
    prompt_text,
    model,
    max_tokens=1000,
):
    """调用LLM API进行推理，使用两张图片"""
    # 编码图片
    original_image_base64 = encode_image(original_image_path)
    nodes_image_base64 = encode_image(nodes_image_path)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
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
    """解析LLM响应，分离reasoning_content和content"""
    message = response.choices[0].message
    full_response = getattr(message, "content", "") or ""
    reasoning_content = getattr(message, "reasoning_content", "") or ""
    
    # 尝试提取最终答案（假设是最后一个数字或在特定格式中）
    lines = full_response.split('\n')
    content = ""
    
    # 查找最后一行中的数字作为最终答案
    for line in reversed(lines):
        line = line.strip()
        if line.isdigit():
            content = line
            break
        # 也尝试匹配常见的答案格式
        if "答案" in line or "Answer" in line or "node" in line or "point" in line:
            # 提取数字
            import re
            numbers = re.findall(r'\d+', line)
            if numbers:
                content = numbers[-1]
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
        x, y = node['pixel_coordinates']
        if (bbox['x_min'] <= x <= bbox['x_max'] and 
            bbox['y_min'] <= y <= bbox['y_max']):
            
            # Adjust coordinates relative to cropped image
            relative_x = x - bbox['x_min']
            relative_y = y - bbox['y_min']
            
            node_copy = node.copy()
            node_copy['relative_pixel_coordinates'] = [relative_x, relative_y]
            nodes_in_room.append(node_copy)
    
    return nodes_in_room

def draw_nodes_on_cropped_image(cropped_image: np.ndarray, nodes_in_room: List[Dict], 
                               node_radius: int = 8) -> np.ndarray:
    """Draw nodes on cropped image with adjusted coordinates."""
    result_image = cropped_image.copy()
    
    for node in nodes_in_room:
        x, y = node['relative_pixel_coordinates']
        node_id = node['node_id']
        
        # Draw node circle
        cv2.circle(result_image, (int(x), int(y)), node_radius, (0, 255, 0), -1)  # Green filled
        cv2.circle(result_image, (int(x), int(y)), node_radius, (0, 0, 0), 2)     # Black border
        
        # Draw node number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text = str(node_id)
        
        # Calculate text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int(x - text_width // 2)
        text_y = int(y + text_height // 2)
        
        # Draw text with white color and black outline
        cv2.putText(result_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)  # Black outline
        cv2.putText(result_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)  # White text
    
    return result_image

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def select_node_with_llm(original_room_image: np.ndarray, 
                         room_image_with_nodes: np.ndarray, 
                         nodes_in_room: List[Dict], 
                         config: Dict[str, Any], 
                         output_dir: str) -> Dict[str, Any]:
    """Use LLM to select optimal navigation node using two images with retry logic."""
    
    if Ark is None:
        raise RuntimeError(
            "byteplussdkarkruntime is required for BytePlus Ark; install it "
            "with `python3 -m pip install byteplussdkarkruntime`"
        )
    
    # Get available node IDs for validation
    available_node_ids = [node['node_id'] for node in nodes_in_room]
    if not available_node_ids:
        raise ValueError("No navigation nodes are available in the selected room")

    # Both LLM stages share the same validated, runtime-only configuration.
    llm_settings = resolve_llm_config(config)
    api_key = llm_settings.api_key
    base_url = llm_settings.base_url
    model = llm_settings.model
    max_tokens = llm_settings.max_tokens
    max_retries = llm_settings.max_retries
    fallback_on_failure = llm_settings.fallback_on_failure
    
    prompt_path = config['prompts']['choose_node_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    goal_object = config['scene_config']['goal_object']
    prompt_template = prompt_template.format(goal_object=goal_object)
    
    print(f"🤖 Using LLM for node selection:")
    print(f"  - Model: {model}")
    print(f"  - Goal object: {goal_object}")
    print(f"  - Available nodes: {available_node_ids}")
    print(f"  - Max retries: {max_retries}")
    print(f"  - Prompt from: {prompt_path}")
    
    # 重试逻辑
    selected_node_id = None
    selected_node_data = None
    retry_count = 0
    all_responses = []
    
    while retry_count < max_retries and selected_node_id is None:
        try:
            # Set up API client
            client = Ark(base_url=base_url, api_key=api_key)
            
            print(f"🚀 Sending node selection request to LLM... (Attempt {retry_count + 1}/{max_retries})")

            enhanced_prompt = f"{prompt_template}"
            
            # 保存临时图片文件
            temp_original_image_path = os.path.join(output_dir, "temp_original_room_image.png")
            temp_nodes_image_path = os.path.join(output_dir, "temp_room_with_nodes.png")

            try:
                if not cv2.imwrite(
                    temp_original_image_path, original_room_image
                ):
                    raise OSError(
                        "Failed to write the original-room image for node selection"
                    )
                if not cv2.imwrite(
                    temp_nodes_image_path, room_image_with_nodes
                ):
                    raise OSError(
                        "Failed to write the node-overlay image for node selection"
                    )
                response = call_llm_with_images(
                    client,
                    temp_original_image_path,
                    temp_nodes_image_path,
                    enhanced_prompt,
                    model,
                    max_tokens,
                )
            finally:
                for temporary_path in (
                    temp_original_image_path,
                    temp_nodes_image_path,
                ):
                    try:
                        os.remove(temporary_path)
                    except FileNotFoundError:
                        pass
            
            if not response or not response.choices[0].message.content:
                raise RuntimeError("Empty response from LLM")
            
            # 解析响应
            parsed_result = parse_llm_response(response)
            raw_response = parsed_result["reasoning_content"]
            content = parsed_result["content"]
            
            print(f"📝 Raw LLM response (Attempt {retry_count + 1}): '{raw_response}'")
            print(f"📝 Extracted content: '{content}'")
            
            # 记录响应
            all_responses.append({
                "attempt": retry_count + 1,
                "raw_response": raw_response,
                "extracted_content": content,
                "timestamp": str(__import__('datetime').datetime.now())
            })
            
            # 解析并验证节点ID
            try:
                if content and content.isdigit():
                    candidate_id = int(content)
                    if candidate_id in available_node_ids:
                        selected_node_id = candidate_id
                        # 找到对应的节点数据
                        for node in nodes_in_room:
                            if node['node_id'] == selected_node_id:
                                selected_node_data = node
                                break
                        print(f"✓ Valid node ID selected: {selected_node_id}")
                        break
                    else:
                        print(f"⚠️ LLM selected node {candidate_id} not in available nodes {available_node_ids}")
                        print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        continue
                else:
                    # Fallback: try to extract from full response
                    import re
                    numbers = re.findall(r'\d+', raw_response)
                    if numbers:
                        candidate_id = int(numbers[0])
                        if candidate_id in available_node_ids:
                            selected_node_id = candidate_id
                            # 找到对应的节点数据
                            for node in nodes_in_room:
                                if node['node_id'] == selected_node_id:
                                    selected_node_data = node
                                    break
                            print(f"✓ Valid node ID selected: {selected_node_id}")
                            break
                        else:
                            print(f"⚠️ LLM selected node {candidate_id} not in available nodes {available_node_ids}")
                            print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
                            retry_count += 1
                            continue
                    else:
                        print("⚠️ No number found in LLM response")
                        print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        continue
                    
            except Exception as e:
                print(f"⚠️ Error parsing node ID: {e}")
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
    if selected_node_id is None and fallback_on_failure:
        print(f"⚠️ All LLM attempts failed. Using fallback: first available node {available_node_ids[0]}")
        selected_node_id = available_node_ids[0]
        for node in nodes_in_room:
            if node['node_id'] == selected_node_id:
                selected_node_data = node
                break
        final_response = f"Fallback selection: Node {selected_node_id} (LLM attempts failed)"
    elif selected_node_id is not None:
        final_response = all_responses[-1]["raw_response"]
    else:
        raise RuntimeError(
            "LLM node selection failed after "
            f"{max_retries} attempts; set llm_config.fallback_on_failure "
            "to true only if arbitrary fallback selection is acceptable"
        )
    
    return {
        "raw_response": final_response,
        "selected_node_id": selected_node_id,
        "selected_node_data": selected_node_data,
        "model_used": model,
        "available_nodes": nodes_in_room,
        "attempts_made": len(all_responses),
        "all_responses": all_responses
    }

def select_navigation_node(graph_path: str, topdown_path: str, room_bbox: Dict[str, int], 
                          selected_room: int, config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Select navigation node within the target room.
    
    Args:
        graph_path: Path to graph with topdown view image
        topdown_path: Path to original topdown view image
        room_bbox: Bounding box of the selected room
        selected_room: Selected room number
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading files for node selection:")
    print(f"  - Graph image: {graph_path}")
    print(f"  - Topdown image: {topdown_path}")
    print(f"  - Selected room: {selected_room}")
    print(f"  - Room bbox: {room_bbox}")
    
    # Load images
    graph_image = cv2.imread(graph_path)
    topdown_image = cv2.imread(topdown_path)
    
    if graph_image is None:
        raise FileNotFoundError(f"Graph image not found: {graph_path}")
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    
    # Load navigation nodes data
    nodes_json_path = os.path.join(output_dir, "navigation_nodes.json")
    with open(nodes_json_path, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)
    
    all_nodes = nodes_data['nodes']
    print(f"✓ Loaded {len(all_nodes)} navigation nodes")
    
    # Get nodes within the room bounding box
    nodes_in_room = get_nodes_in_bbox(all_nodes, room_bbox)
    print(f"✓ Found {len(nodes_in_room)} nodes in room {selected_room}")
    
    if not nodes_in_room:
        raise RuntimeError(f"No navigation nodes found in room {selected_room}")
    
    # Crop graph image to room
    cropped_graph = crop_image_to_bbox(graph_image, room_bbox)
    
    # Create room-specific graph visualization
    room_with_graph = draw_nodes_on_cropped_image(
        crop_image_to_bbox(topdown_image, room_bbox), 
        nodes_in_room,
        config['graph_generation'].get('node_radius_pixels', 8)
    )
    original_graph = crop_image_to_bbox(topdown_image, room_bbox)
    
    # 这张是画了nodes的图
    room_with_graph = draw_nodes_on_cropped_image(
        original_graph, # 使用已经裁剪好的原图作为底图
        nodes_in_room,
        config['graph_generation'].get('node_radius_pixels', 8)
    )
    
    # 保存 room_with_graph (这部分逻辑不变)
    room_with_graph_path = os.path.join(output_dir, "room_with_graph.png")
    cv2.imwrite(room_with_graph_path, room_with_graph)
    print(f"✓ Room with graph saved to: {room_with_graph_path}")
    
    llm_response = select_node_with_llm(
        original_room_image=original_graph,
        room_image_with_nodes=room_with_graph,
        nodes_in_room=nodes_in_room,
        config=config,
        output_dir=output_dir,
    )
    selected_node_data = llm_response["selected_node_data"]
    if selected_node_data is None:
        raise RuntimeError("LLM returned no selected-node data")
    
    print(f"🎯 Selected node {llm_response['selected_node_id']} for navigation")
    
    # Create final visualization with selected node highlighted
    final_topdown = topdown_image.copy()
    
    # Draw all nodes in light green
    node_radius = config['graph_generation'].get('node_radius_pixels', 8)
    for node in all_nodes:
        x, y = node['pixel_coordinates']
        cv2.circle(final_topdown, (int(x), int(y)), node_radius, (144, 238, 144), -1)  # Light green
        cv2.circle(final_topdown, (int(x), int(y)), node_radius, (0, 0, 0), 1)         # Black border
    
    # Highlight selected node in bright red
    selected_x, selected_y = selected_node_data['pixel_coordinates']
    cv2.circle(final_topdown, (int(selected_x), int(selected_y)), node_radius + 3, (0, 0, 255), -1)  # Red highlight
    cv2.circle(final_topdown, (int(selected_x), int(selected_y)), node_radius + 3, (255, 255, 255), 2)  # White border
    
    # Draw node number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 3
    text = str(selected_node_data['node_id'])
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = int(selected_x - text_width // 2)
    text_y = int(selected_y + text_height // 2)
    
    cv2.putText(final_topdown, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)  # Black outline
    cv2.putText(final_topdown, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)  # White text
    
    # Save final result
    node_with_topdown_path = os.path.join(output_dir, "node_with_topdown.png")
    cv2.imwrite(node_with_topdown_path, final_topdown)
    print(f"✓ Final result saved to: {node_with_topdown_path}")
    
    # Save node selection log
    selection_log = {
        "selected_room": selected_room,
        "room_bounding_box": room_bbox,
        "nodes_in_room": len(nodes_in_room),
        "selected_node": selected_node_data,
        "llm_response": llm_response
    }
    
    log_path = os.path.join(output_dir, "node_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(
            selection_log, f, indent=2, ensure_ascii=False, allow_nan=False
        )
    print(f"✓ Node selection log saved to: {log_path}")
    
    return {
        "generated_files": {
            "room_with_graph": room_with_graph_path,
            "node_with_topdown": node_with_topdown_path,
            "node_selection_log": log_path
        },
        "llm_response": llm_response,
        "results": {
            "selected_node_id": selected_node_data['node_id'],
            "selected_node_pixel_coordinates": selected_node_data['pixel_coordinates'],
            "selected_node_world_coordinates": selected_node_data.get('world_coordinates'),
            "total_nodes_in_room": len(nodes_in_room),
            "room_bounding_box": room_bbox
        }
    }

if __name__ == "__main__":
    import sys
    import numpy as np
    
    if len(sys.argv) != 6:
        print(
            "Usage: python3 -m "
            "reason_navi.preprocessing.select_navigation_node "
            "<graph_path> <topdown_path> <room_bbox_json> "
            "<selected_room> <config_path>"
        )
        sys.exit(1)
    
    graph_path = sys.argv[1]
    topdown_path = sys.argv[2]
    room_bbox_json = sys.argv[3]
    selected_room = int(sys.argv[4])
    config_path = sys.argv[5]
    
    # Parse room bbox
    room_bbox = json.loads(room_bbox_json)
    
    config = load_preprocessing_config(config_path)
    
    output_dir = config['output']['output_dir']
    
    result = select_navigation_node(graph_path, topdown_path, room_bbox, selected_room, config, output_dir)
    print("Step 5 completed:", result)
