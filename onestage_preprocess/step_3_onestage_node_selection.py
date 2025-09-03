"""
Step 3 (One-stage): Direct navigation node selection without room segmentation.
Uses LLM to directly select optimal navigation node from the entire topdown view.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    from byteplussdkarkruntime import Ark
    import base64
except ImportError:
    print("Warning: volcenginesdkarkruntime not found. Please install them with: pip install volcenginesdkarkruntime")
    Ark = None

def encode_image(image_path):
    """编码图片为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_with_images(client, original_image_path, nodes_image_path, prompt_text, model):
    """调用LLM API进行推理，使用两张图片"""
    # 编码图片
    original_image_base64 = encode_image(original_image_path)
    nodes_image_base64 = encode_image(nodes_image_path)
    
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
    full_response = response.choices[0].message.content
    reasoning_content = response.choices[0].message.reasoning_content
    
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

def draw_nodes_on_image(image: np.ndarray, nodes_data: List[Dict], 
                       node_radius: int = 8) -> np.ndarray:
    """Draw nodes on image with node IDs."""
    result_image = image.copy()
    
    for node in nodes_data:
        x, y = node['pixel_coordinates']
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

def select_node_with_llm(original_image: np.ndarray, 
                         image_with_nodes: np.ndarray, 
                         all_nodes: List[Dict], 
                         config: Dict[str, Any], 
                         output_dir: str) -> Dict[str, Any]:
    """Use LLM to select optimal navigation node using two images with retry logic."""
    
    if Ark is None:
        raise RuntimeError("volcenginesdkarkruntime package not installed")
    
    # Get available node IDs for validation
    available_node_ids = [node['node_id'] for node in all_nodes]
    
    # LLM configuration
    llm_config = config['llm_config']
    api_key = llm_config['api_key']
    base_url = llm_config.get('base_url', None)
    model = llm_config.get('model', 'seed-1-6-250615')
    max_tokens = llm_config.get('max_tokens', 1000)
    max_retries = llm_config.get('max_retries', 3)
    
    prompt_path = config['prompts']['choose_node_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    goal_object = config['scene_config']['goal_object']
    prompt_template = prompt_template.format(goal_object=goal_object)
    
    print(f"🤖 Using LLM for one-stage node selection:")
    print(f"  - Model: {model}")
    print(f"  - Goal object: {goal_object}")
    print(f"  - Total nodes: {len(available_node_ids)}")
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
            
            print(f"🚀 Sending one-stage node selection request to LLM... (Attempt {retry_count + 1}/{max_retries})")

            enhanced_prompt = f"{prompt_template}\n\nAvailable nodes in this scene: {sorted(available_node_ids)}"
            
            # 保存临时图片文件
            temp_original_image_path = os.path.join(output_dir, "temp_original_image.png")
            cv2.imwrite(temp_original_image_path, original_image)
            
            temp_nodes_image_path = os.path.join(output_dir, "temp_image_with_nodes.png")
            cv2.imwrite(temp_nodes_image_path, image_with_nodes)

            # 调用LLM API
            response = call_llm_with_images(client, temp_original_image_path, temp_nodes_image_path, enhanced_prompt, model)
            
            # 清理临时文件
            os.remove(temp_original_image_path)
            os.remove(temp_nodes_image_path)
            
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
                        for node in all_nodes:
                            if node['node_id'] == selected_node_id:
                                selected_node_data = node
                                break
                        print(f"✓ Valid node ID selected: {selected_node_id}")
                        break
                    else:
                        print(f"⚠️ LLM selected node {candidate_id} not in available nodes")
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
                            for node in all_nodes:
                                if node['node_id'] == selected_node_id:
                                    selected_node_data = node
                                    break
                            print(f"✓ Valid node ID selected: {selected_node_id}")
                            break
                        else:
                            print(f"⚠️ LLM selected node {candidate_id} not in available nodes")
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
    if selected_node_id is None:
        print(f"⚠️ All LLM attempts failed. Using fallback: first available node {available_node_ids[0]}")
        selected_node_id = available_node_ids[0]
        for node in all_nodes:
            if node['node_id'] == selected_node_id:
                selected_node_data = node
                break
        final_response = f"Fallback selection: Node {selected_node_id} (LLM attempts failed)"
    else:
        final_response = all_responses[-1]["raw_response"]
    
    return {
        "raw_response": final_response,
        "selected_node_id": selected_node_id,
        "selected_node_data": selected_node_data,
        "model_used": model,
        "available_nodes": all_nodes,
        "attempts_made": retry_count,
        "all_responses": all_responses
    }

def select_navigation_node_onestage(topdown_path: str, graph_path: str, 
                                  config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    One-stage navigation node selection directly from the topdown view.
    
    Args:
        topdown_path: Path to topdown view image
        graph_path: Path to graph with topdown view image
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading files for one-stage node selection:")
    print(f"  - Topdown image: {topdown_path}")
    print(f"  - Graph image: {graph_path}")
    
    # Load images
    topdown_image = cv2.imread(topdown_path)
    graph_image = cv2.imread(graph_path)
    
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    if graph_image is None:
        raise FileNotFoundError(f"Graph image not found: {graph_path}")
    
    # Load navigation nodes data
    nodes_json_path = os.path.join(output_dir, "navigation_nodes.json")
    with open(nodes_json_path, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)
    
    all_nodes = nodes_data['nodes']
    print(f"✓ Loaded {len(all_nodes)} navigation nodes")
    
    if not all_nodes:
        raise RuntimeError("No navigation nodes found")
    
    # Create visualization with all nodes
    image_with_nodes = draw_nodes_on_image(
        topdown_image, 
        all_nodes,
        config['graph_generation'].get('node_radius_pixels', 8)
    )
    
    # Save visualization
    nodes_visualization_path = os.path.join(output_dir, "topdown_with_nodes.png")
    cv2.imwrite(nodes_visualization_path, image_with_nodes)
    print(f"✓ Nodes visualization saved to: {nodes_visualization_path}")
    
    # Use LLM to select the optimal node
    try:
        llm_result = select_node_with_llm(
            original_image=topdown_image,
            image_with_nodes=image_with_nodes,
            all_nodes=all_nodes,
            config=config,
            output_dir=output_dir
        )
        
        selected_node_data = llm_result['selected_node_data']
        llm_response = llm_result
        
        if selected_node_data is None:
            print("⚠️ LLM selection failed, using first available node as fallback")
            selected_node_data = all_nodes[0]
            llm_response['selected_node_id'] = selected_node_data['node_id']
            llm_response['selected_node_data'] = selected_node_data
            
    except Exception as e:
        print(f"⚠️ LLM node selection failed: {e}")
        print("Using first available node as fallback")
        selected_node_data = all_nodes[0]
        llm_response = {
            "raw_response": f"Fallback selection: node {selected_node_data['node_id']}",
            "selected_node_id": selected_node_data['node_id'],
            "selected_node_data": selected_node_data,
            "model_used": "fallback",
            "error": str(e)
        }
    
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
    node_with_topdown_path = os.path.join(output_dir, "selected_node_with_topdown.png")
    cv2.imwrite(node_with_topdown_path, final_topdown)
    print(f"✓ Final result saved to: {node_with_topdown_path}")
    
    # Save node selection log
    selection_log = {
        "method": "one_stage_selection",
        "total_nodes": len(all_nodes),
        "selected_node": selected_node_data,
        "llm_response": llm_response,
        "goal_object": config['scene_config']['goal_object']
    }
    
    log_path = os.path.join(output_dir, "onestage_node_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(selection_log, f, indent=2, ensure_ascii=False)
    print(f"✓ Node selection log saved to: {log_path}")
    
    # 保存LLM的完整content和reasoning content到预处理位置
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    # 保存最终成功的LLM响应的完整内容
    if 'all_responses' in llm_response and llm_response['all_responses'] and selected_node_data is not None:
        # 找到最终成功的响应（最后一个有效响应）
        successful_response = None
        for response in reversed(llm_response['all_responses']):
            if "error" not in response and response.get("extracted_content"):
                successful_response = response
                break
        
        if successful_response:
            llm_content_data = {
                "timestamp": successful_response["timestamp"],
                "model": llm_response['model_used'],
                "selected_node_id": llm_response['selected_node_id'],
                "full_reasoning_content": successful_response["raw_response"],
                "extracted_content": successful_response["extracted_content"],
                "attempt_number": successful_response["attempt"],
                "total_available_nodes": len(all_nodes),
                "goal_object": config['scene_config']['goal_object'],
                "method": "one_stage_selection",
                "prompt_used": "One-stage node selection prompt"
            }
        else:
            # 如果没有成功的响应，使用fallback信息
            llm_content_data = {
                "timestamp": str(__import__('datetime').datetime.now()),
                "model": llm_response['model_used'],
                "selected_node_id": llm_response['selected_node_id'],
                "full_reasoning_content": llm_response['raw_response'],
                "extracted_content": str(llm_response['selected_node_id']),
                "attempt_number": "fallback",
                "total_available_nodes": len(all_nodes),
                "goal_object": config['scene_config']['goal_object'],
                "method": "one_stage_selection",
                "prompt_used": "One-stage node selection prompt",
                "note": "Used fallback node selection due to LLM failures"
            }
    else:
        # 如果没有任何响应，创建默认记录
        llm_content_data = {
            "timestamp": str(__import__('datetime').datetime.now()),
            "model": llm_response.get('model_used', 'unknown'),
            "selected_node_id": llm_response['selected_node_id'],
            "full_reasoning_content": "No valid LLM response received",
            "extracted_content": str(llm_response['selected_node_id']) if llm_response['selected_node_id'] else "None",
            "attempt_number": 0,
            "total_available_nodes": len(all_nodes),
            "goal_object": config['scene_config']['goal_object'],
            "method": "one_stage_selection",
            "prompt_used": "One-stage node selection prompt",
            "note": "No valid LLM response - using fallback"
        }
    
    # 保存到preprocess目录
    llm_content_path = os.path.join(preprocess_dir, "llm_node_selection_content.json")
    with open(llm_content_path, 'w', encoding='utf-8') as f:
        json.dump(llm_content_data, f, indent=2, ensure_ascii=False)
    print(f"✓ LLM content saved to preprocessing directory: {llm_content_path}")
    
    # 同时保存一个简化的reasoning文本文件，便于阅读
    reasoning_text_path = os.path.join(preprocess_dir, "llm_reasoning.txt")
    with open(reasoning_text_path, 'w', encoding='utf-8') as f:
        f.write(f"LLM One-Stage Node Selection Reasoning\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {llm_content_data['timestamp']}\n")
        f.write(f"Model: {llm_content_data['model']}\n")
        f.write(f"Method: {llm_content_data['method']}\n")
        f.write(f"Goal Object: {llm_content_data['goal_object']}\n")
        f.write(f"Total Available Nodes: {llm_content_data['total_available_nodes']}\n")
        f.write(f"Selected Node ID: {llm_content_data['selected_node_id']}\n")
        f.write(f"Attempt Number: {llm_content_data['attempt_number']}\n\n")
        f.write(f"Full LLM Response:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"{llm_content_data['full_reasoning_content']}\n\n")
        f.write(f"Extracted Answer: {llm_content_data['extracted_content']}\n")
        if 'note' in llm_content_data:
            f.write(f"\nNote: {llm_content_data['note']}\n")
    print(f"✓ LLM reasoning text saved to: {reasoning_text_path}")
    
    return {
        "generated_files": {
            "topdown_with_nodes": nodes_visualization_path,
            "selected_node_with_topdown": node_with_topdown_path,
            "node_selection_log": log_path,
            "llm_content": llm_content_path,
            "reasoning_text": reasoning_text_path
        },
        "llm_response": llm_response,
        "results": {
            "selected_node_id": selected_node_data['node_id'],
            "selected_node_pixel_coordinates": selected_node_data['pixel_coordinates'],
            "selected_node_world_coordinates": selected_node_data.get('world_coordinates'),
            "total_nodes": len(all_nodes),
            "method": "one_stage_selection"
        }
    }

# Fallback function for testing without LLM
def select_node_manually_onestage(topdown_path: str, output_dir: str, node_id: int = 1) -> Dict[str, Any]:
    """
    Fallback function to manually select a node for testing.
    
    Args:
        topdown_path: Path to topdown view image
        output_dir: Output directory path  
        node_id: Node ID to select (default: 1)
        
    Returns:
        Dictionary with manual selection results
    """
    print(f"🔧 Manual one-stage node selection: Node {node_id}")
    
    # Load navigation nodes data
    nodes_json_path = os.path.join(output_dir, "navigation_nodes.json")
    with open(nodes_json_path, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)
    
    all_nodes = nodes_data['nodes']
    selected_node_data = None
    
    # Find the selected node
    for node in all_nodes:
        if node['node_id'] == node_id:
            selected_node_data = node
            break
    
    if selected_node_data is None:
        # Use first available node as fallback
        selected_node_data = all_nodes[0]
        node_id = selected_node_data['node_id']
        print(f"⚠️ Node {node_id} not found, using first available node: {selected_node_data['node_id']}")
    
    manual_log = {
        "method": "manual_onestage_selection",
        "selected_node_id": node_id,
        "selected_node_data": selected_node_data,
        "topdown_image": topdown_path
    }
    
    log_path = os.path.join(output_dir, "manual_onestage_node_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(manual_log, f, indent=2, ensure_ascii=False)
    
    # 保存手动选择的内容到预处理位置，保持与LLM选择的一致性
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    manual_content_data = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "model": "manual_selection",
        "selected_node_id": node_id,
        "full_reasoning_content": f"Manual one-stage node selection: Node {node_id} was manually specified",
        "extracted_content": str(node_id),
        "attempt_number": 1,
        "total_available_nodes": len(all_nodes),
        "goal_object": "unknown (manual selection)",
        "method": "one_stage_selection_manual",
        "prompt_used": "N/A (manual selection)",
        "note": "Node was manually selected, not through LLM reasoning"
    }
    
    # 保存到preprocess目录
    manual_content_path = os.path.join(preprocess_dir, "llm_node_selection_content.json")
    with open(manual_content_path, 'w', encoding='utf-8') as f:
        json.dump(manual_content_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Manual selection content saved to preprocessing directory: {manual_content_path}")
    
    # 同时保存一个简化的reasoning文本文件
    reasoning_text_path = os.path.join(preprocess_dir, "llm_reasoning.txt")
    with open(reasoning_text_path, 'w', encoding='utf-8') as f:
        f.write(f"Node Selection Reasoning (Manual One-Stage)\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {manual_content_data['timestamp']}\n")
        f.write(f"Method: Manual One-Stage Selection\n")
        f.write(f"Selected Node ID: {manual_content_data['selected_node_id']}\n\n")
        f.write(f"Reasoning:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"{manual_content_data['full_reasoning_content']}\n\n")
        f.write(f"Final Answer: {manual_content_data['extracted_content']}\n")
        f.write(f"\nNote: {manual_content_data['note']}\n")
    print(f"✓ Manual selection reasoning text saved to: {reasoning_text_path}")
    
    return {
        "generated_files": {
            "selection_log": log_path,
            "llm_content": manual_content_path,
            "reasoning_text": reasoning_text_path
        },
        "llm_response": {
            "raw_response": f"Manual selection: Node {node_id}",
            "selected_node_id": node_id,
            "selected_node_data": selected_node_data,
            "model_used": "manual"
        },
        "results": {
            "selected_node_id": node_id,
            "selected_node_pixel_coordinates": selected_node_data['pixel_coordinates'],
            "selected_node_world_coordinates": selected_node_data.get('world_coordinates'),
            "total_nodes": len(all_nodes),
            "method": "one_stage_selection_manual"
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python step_3_onestage_node_selection.py <topdown_path> <graph_path> <config_path> [manual_node_id]")
        print("If manual_node_id is provided, manual selection will be used instead of LLM")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    graph_path = sys.argv[2]
    config_path = sys.argv[3]
    manual_node = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if manual_node is not None:
        result = select_node_manually_onestage(topdown_path, output_dir, manual_node)
        print("Step 3 (one-stage) completed (manual):", result)
    else:
        result = select_navigation_node_onestage(topdown_path, graph_path, config, output_dir)
        print("Step 3 (one-stage) completed (LLM):", result)
