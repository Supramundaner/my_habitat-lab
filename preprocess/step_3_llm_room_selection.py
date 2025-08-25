"""
Step 3: LLM room selection using Doubao API.
"""

import os
import json
from typing import Dict, Any, List

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

def call_llm_with_image(client, room_annotation_path, prompt_text,model):
    """调用LLM API进行推理"""
    # 编码room annotation图片
    room_annotation_base64 = encode_image(room_annotation_path)
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{room_annotation_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ],
    )
    return response

def parse_llm_response(response):
    """解析LLM响应，分离reasoning_content和content"""
    full_response = response.choices[0].message.content
    
    # 尝试提取最终答案（假设是最后一个数字或在特定格式中）
    lines = full_response.split('\n')
    content = ""
    reasoning_content = full_response
    
    # 查找最后一行中的数字作为最终答案
    for line in reversed(lines):
        line = line.strip()
        if line.isdigit():
            content = line
            break
        # 也尝试匹配常见的答案格式
        if "答案" in line or "Answer" in line or "Room" in line or "房间" in line:
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

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_available_rooms(output_dir: str) -> List[int]:
    """Get available room numbers from room segmentation results."""
    # Try to find room information from different possible sources during workflow
    room_ids = []
    
    try:
        # First try: Check if there's a step2 intermediate result file
        # (This would be saved by workflow orchestrator if it exists)
        step2_result_path = os.path.join(output_dir, "step2_results.json")
        if os.path.exists(step2_result_path):
            with open(step2_result_path, 'r', encoding='utf-8') as f:
                step2_data = json.load(f)
            room_bboxes = step2_data.get("results", {}).get("room_bounding_boxes", {})
            room_ids = [int(room_id) for room_id in room_bboxes.keys()]
            if room_ids:
                print(f"✓ Found available rooms from step2_results.json: {sorted(room_ids)}")
                return room_ids
    except Exception as e:
        print(f"⚠️ Could not load from step2_results.json: {e}")
    
    try:
        # Second try: Check final output.json (for completed workflows)
        output_json_path = os.path.join(output_dir, "output.json")
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            room_bboxes = output_data.get("final_results", {}).get("room_bounding_boxes", {})
            room_ids = [int(room_id) for room_id in room_bboxes.keys()]
            if room_ids:
                print(f"✓ Found available rooms from output.json: {sorted(room_ids)}")
                return room_ids
    except Exception as e:
        print(f"⚠️ Could not load from output.json: {e}")
    
    try:
        # Third try: Try to parse room annotation image to get room numbers
        # This is a fallback method by analyzing the room annotation image
        room_annotation_path = os.path.join(output_dir, "room_annotation.png")
        if os.path.exists(room_annotation_path):
            print("⚠️ Attempting to extract room numbers from room_annotation.png...")
            # This is a basic OCR-like approach - in practice, you might want to use OCR
            # For now, we'll make a reasonable assumption based on typical room counts
            # You could implement actual OCR here if needed
            print("⚠️ Image-based room detection not implemented, using fallback")
    except Exception as e:
        print(f"⚠️ Could not analyze room annotation image: {e}")
    
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
    api_key =  llm_config['api_key'] 
    base_url= llm_config['base_url']
    model = llm_config.get('model', 'seed-1-6-250615')
    max_tokens = llm_config.get('max_tokens', 1000)
    max_retries = llm_config.get('max_retries', 3)
    
    # Load prompt template
    prompt_path = config['prompts']['choose_room_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    # Get goal object and replace placeholder
    goal_object = config['scene_config']['goal_object']
    prompt_template = prompt_template.format(goal_object=goal_object)
    
    print(f"🤖 LLM Configuration:")
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
            if Ark is None:
                raise RuntimeError("volcenginesdkarkruntime package not installed")
            
            # Set up client
            client = Ark(base_url=base_url,api_key=api_key)
            
            print(f" Sending request to LLM... (Attempt {retry_count + 1}/{max_retries})")
            
            # 构建prompt，包含可用房间信息
            enhanced_prompt = prompt_template
            if available_rooms:
                enhanced_prompt += f"\n\nAvailable rooms in this scene: {sorted(available_rooms)}"
            
            # 调用LLM API
            response = call_llm_with_image(client, room_annotation_path, enhanced_prompt, model)
            
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
            
            # Parse the room number from response
            try:
                if content and content.isdigit():
                    candidate_room = int(content)
                    
                    # 验证房间是否在可用范围内
                    if available_rooms and candidate_room not in available_rooms:
                        print(f"⚠️ LLM selected room {candidate_room} not in available rooms {available_rooms}")
                        print(f"� Retrying... ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        continue
                    else:
                        selected_room = candidate_room
                        print(f"✓ Valid room number selected: {selected_room}")
                        break
                else:
                    # Fallback: try to extract from full response
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
    
    # 保存LLM的完整content和reasoning content到预处理位置
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    # 保存最终成功的LLM响应的完整内容
    if all_responses and selected_room is not None:
        # 找到最终成功的响应（最后一个有效响应）
        successful_response = None
        for response in reversed(all_responses):
            if "error" not in response and response.get("extracted_content"):
                successful_response = response
                break
        
        if successful_response:
            llm_content_data = {
                "timestamp": successful_response["timestamp"],
                "model": model,
                "selected_room": selected_room,
                "full_reasoning_content": successful_response["raw_response"],
                "extracted_content": successful_response["extracted_content"],
                "attempt_number": successful_response["attempt"],
                "available_rooms": available_rooms,
                "goal_object": config['scene_config']['goal_object'],
                "prompt_used": enhanced_prompt if 'enhanced_prompt' in locals() else prompt_template
            }
        else:
            # 如果没有成功的响应，使用fallback信息
            llm_content_data = {
                "timestamp": str(__import__('datetime').datetime.now()),
                "model": model,
                "selected_room": selected_room,
                "full_reasoning_content": final_response,
                "extracted_content": str(selected_room),
                "attempt_number": "fallback",
                "available_rooms": available_rooms,
                "goal_object": config['scene_config']['goal_object'],
                "prompt_used": enhanced_prompt if 'enhanced_prompt' in locals() else prompt_template,
                "note": "Used fallback room selection due to LLM failures"
            }
    else:
        # 如果没有任何响应，创建默认记录
        llm_content_data = {
            "timestamp": str(__import__('datetime').datetime.now()),
            "model": model,
            "selected_room": selected_room,
            "full_reasoning_content": "No valid LLM response received",
            "extracted_content": str(selected_room) if selected_room else "None",
            "attempt_number": 0,
            "available_rooms": available_rooms,
            "goal_object": config['scene_config']['goal_object'],
            "prompt_used": prompt_template,
            "note": "No valid LLM response - using fallback"
        }
    
    # 保存到preprocess目录
    llm_content_path = os.path.join(preprocess_dir, "llm_room_selection_content.json")
    with open(llm_content_path, 'w', encoding='utf-8') as f:
        json.dump(llm_content_data, f, indent=2, ensure_ascii=False)
    print(f"✓ LLM content saved to preprocessing directory: {llm_content_path}")
    
    # 同时保存一个简化的reasoning文本文件，便于阅读
    reasoning_text_path = os.path.join(preprocess_dir, "llm_reasoning.txt")
    with open(reasoning_text_path, 'w', encoding='utf-8') as f:
        f.write(f"LLM Room Selection Reasoning\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {llm_content_data['timestamp']}\n")
        f.write(f"Model: {llm_content_data['model']}\n")
        f.write(f"Goal Object: {llm_content_data['goal_object']}\n")
        f.write(f"Available Rooms: {llm_content_data['available_rooms']}\n")
        f.write(f"Selected Room: {llm_content_data['selected_room']}\n")
        f.write(f"Attempt Number: {llm_content_data['attempt_number']}\n\n")
        f.write(f"Prompt Used:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"{llm_content_data['prompt_used']}\n\n")
        f.write(f"Full LLM Response:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"{llm_content_data['full_reasoning_content']}\n\n")
        f.write(f"Extracted Answer: {llm_content_data['extracted_content']}\n")
        if 'note' in llm_content_data:
            f.write(f"\nNote: {llm_content_data['note']}\n")
    print(f"✓ LLM reasoning text saved to: {reasoning_text_path}")
    
    return {
        "generated_files": {
            "llm_log": log_path,
            "llm_content": llm_content_path,
            "reasoning_text": reasoning_text_path
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
    
    # 保存手动选择的内容到预处理位置，保持与LLM选择的一致性
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    manual_content_data = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "model": "manual_selection",
        "selected_room": room_number,
        "full_reasoning_content": f"Manual room selection: Room {room_number} was manually specified",
        "extracted_content": str(room_number),
        "attempt_number": 1,
        "available_rooms": "unknown (manual selection)",
        "goal_object": "unknown (manual selection)",
        "prompt_used": "N/A (manual selection)",
        "note": "Room was manually selected, not through LLM reasoning"
    }
    
    # 保存到preprocess目录
    manual_content_path = os.path.join(preprocess_dir, "llm_room_selection_content.json")
    with open(manual_content_path, 'w', encoding='utf-8') as f:
        json.dump(manual_content_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Manual selection content saved to preprocessing directory: {manual_content_path}")
    
    # 同时保存一个简化的reasoning文本文件
    reasoning_text_path = os.path.join(preprocess_dir, "llm_reasoning.txt")
    with open(reasoning_text_path, 'w', encoding='utf-8') as f:
        f.write(f"Room Selection Reasoning (Manual)\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {manual_content_data['timestamp']}\n")
        f.write(f"Method: Manual Selection\n")
        f.write(f"Selected Room: {manual_content_data['selected_room']}\n\n")
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
