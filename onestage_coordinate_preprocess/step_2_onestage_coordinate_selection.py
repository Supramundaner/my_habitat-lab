"""
Step 3 (One-stage Coordinate): Direct coordinate selection from topdown image.
Uses LLM to directly select optimal navigation coordinate from the normalized topdown view.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re

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

def call_llm_with_image(client, image_path, prompt_text, model):
    """调用LLM API进行推理，使用单张图片"""
    # 编码图片
    image_base64 = encode_image(image_path)
    
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
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
    )
    return response

def parse_llm_response_for_coordinate(response):
    """解析LLM响应，提取坐标信息"""
    full_response = response.choices[0].message.content
    reasoning_content = response.choices[0].message.reasoning_content
    
    # 尝试提取坐标
    coordinate_patterns = [
        r'coordinate\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',  # coordinate (x, y)
        r'\(\s*(\d+)\s*,\s*(\d+)\s*\)',                # (x, y)
        r'(\d+)\s*,\s*(\d+)',                          # x, y
    ]
    
    extracted_coordinate = None
    
    # 首先在content中查找
    for pattern in coordinate_patterns:
        matches = re.findall(pattern, full_response, re.IGNORECASE)
        if matches:
            # 取最后一个匹配（通常是最终答案）
            x, y = matches[-1]
            extracted_coordinate = (int(x), int(y))
            break
    
    # 如果content中没找到，在reasoning_content中查找
    if extracted_coordinate is None and reasoning_content:
        for pattern in coordinate_patterns:
            matches = re.findall(pattern, reasoning_content, re.IGNORECASE)
            if matches:
                x, y = matches[-1]
                extracted_coordinate = (int(x), int(y))
                break
    
    return {
        "reasoning_content": reasoning_content,
        "content": full_response,
        "extracted_coordinate": extracted_coordinate
    }

def normalize_image_to_size(image: np.ndarray, target_size: int = 1000) -> Tuple[np.ndarray, float]:
    """
    将图像标准化到指定尺寸，保持纵横比
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (默认1000x1000)
    
    Returns:
        标准化后的图像和缩放比例
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建目标尺寸的白色背景
    normalized = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    
    # 计算居中位置
    start_y = (target_size - new_h) // 2
    start_x = (target_size - new_w) // 2
    
    # 将缩放后的图像放置在中心
    normalized[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    return normalized, scale

def validate_coordinate(coordinate: Tuple[int, int], image: np.ndarray, 
                       threshold: int = 50) -> bool:
    """
    验证坐标是否有效（不在墙上，在可导航区域）
    
    Args:
        coordinate: (x, y) 坐标
        image: 标准化后的图像
        threshold: 判断是否为可导航区域的阈值
    
    Returns:
        坐标是否有效
    """
    x, y = coordinate
    h, w = image.shape[:2]
    
    # 检查坐标是否在图像范围内
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    
    # 检查是否在可导航区域（假设白色区域为可导航）
    if len(image.shape) == 3:
        pixel_value = image[y, x]
        # 对于彩色图像，检查是否接近白色
        if np.mean(pixel_value) > threshold:
            return True
    else:
        # 对于灰度图像
        pixel_value = image[y, x]
        if pixel_value > threshold:
            return True
    
    return False

def convert_normalized_to_original_coordinate(normalized_coord: Tuple[int, int], 
                                            original_image_shape: Tuple[int, int],
                                            normalized_size: int = 1000) -> Tuple[float, float]:
    """
    将标准化坐标转换回原始图像坐标
    
    Args:
        normalized_coord: 标准化坐标 (x, y)
        original_image_shape: 原始图像形状 (height, width)
        normalized_size: 标准化尺寸
    
    Returns:
        原始图像坐标 (x, y)
    """
    norm_x, norm_y = normalized_coord
    orig_h, orig_w = original_image_shape[:2]
    
    # 计算缩放比例
    scale = normalized_size / max(orig_h, orig_w)
    
    # 计算在标准化图像中的偏移
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    start_x = (normalized_size - new_w) // 2
    start_y = (normalized_size - new_h) // 2
    
    # 转换坐标
    orig_x = (norm_x - start_x) / scale
    orig_y = (norm_y - start_y) / scale
    
    return (orig_x, orig_y)

def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def select_coordinate_with_llm(image: np.ndarray, 
                              config: Dict[str, Any], 
                              output_dir: str) -> Dict[str, Any]:
    """Use LLM to select optimal navigation coordinate with retry logic."""
    
    if Ark is None:
        raise RuntimeError("volcenginesdkarkruntime package not installed")
    
    # LLM configuration
    llm_config = config['llm_config']
    api_key = llm_config['api_key']
    base_url = llm_config.get('base_url', None)
    model = llm_config.get('model', 'seed-1-6-250615')
    max_tokens = llm_config.get('max_tokens', 1000)
    max_retries = llm_config.get('max_retries', 3)
    
    # Coordinate selection configuration
    coord_config = config.get('coordinate_selection', {})
    normalize_size = coord_config.get('image_normalize_size', 1000)
    validate_coords = coord_config.get('coordinate_validation', True)
    
    prompt_path = config['prompts']['choose_coordinate_prompt']
    prompt_template = load_prompt_template(prompt_path)
    
    goal_object = config['scene_config']['goal_object']
    prompt_template = prompt_template.format(goal_object=goal_object)
    
    print(f"🤖 Using LLM for coordinate selection:")
    print(f"  - Model: {model}")
    print(f"  - Goal object: {goal_object}")
    print(f"  - Original image size: {image.shape[:2]}")
    print(f"  - Coordinate output expected in: {normalize_size}x{normalize_size} space")
    print(f"  - Max retries: {max_retries}")
    print(f"  - Prompt from: {prompt_path}")
    
    # Save original image for LLM (no normalization)
    original_image_path = os.path.join(output_dir, "llm_input_original.png")
    cv2.imwrite(original_image_path, image)
    print(f"✓ Original image saved for LLM input: {original_image_path}")
    
    # Also create normalized image for debugging and validation
    normalized_image, scale_factor = normalize_image_to_size(image, normalize_size)
    normalized_image_path = os.path.join(output_dir, "normalized_debug.png")
    cv2.imwrite(normalized_image_path, normalized_image)
    print(f"✓ Normalized debug image saved: {normalized_image_path}")
    
    # 重试逻辑
    selected_coordinate = None
    retry_count = 0
    all_responses = []
    
    while retry_count < max_retries and selected_coordinate is None:
        try:
            # Set up API client
            client = Ark(base_url=base_url, api_key=api_key)
            
            print(f"🚀 Sending coordinate selection request to LLM... (Attempt {retry_count + 1}/{max_retries})")
            print(f"📸 Sending original resolution image: {image.shape[:2]}")

            # 调用LLM API with ORIGINAL image (not normalized)
            response = call_llm_with_image(client, original_image_path, prompt_template, model)
            
            if not response or not response.choices[0].message.content:
                raise RuntimeError("Empty response from LLM")
            
            # 解析响应
            parsed_result = parse_llm_response_for_coordinate(response)
            raw_response = parsed_result["reasoning_content"]
            content = parsed_result["content"]
            extracted_coord = parsed_result["extracted_coordinate"]
            
            print(f"📝 Raw LLM response (Attempt {retry_count + 1}): '{content[:200]}...'")
            print(f"📝 Extracted coordinate (normalized space): {extracted_coord}")
            
            # 记录响应
            all_responses.append({
                "attempt": retry_count + 1,
                "raw_response": raw_response,
                "content": content,
                "extracted_coordinate": extracted_coord,
                "timestamp": str(__import__('datetime').datetime.now())
            })
            
            # 验证坐标 (LLM输出应该在标准化空间中)
            if extracted_coord is not None:
                x, y = extracted_coord
                
                # 检查坐标范围 (应该在0-1000范围内)
                if 0 <= x <= normalize_size and 0 <= y <= normalize_size:
                    # 如果启用了验证，检查是否在可导航区域
                    if validate_coords:
                        # 将标准化坐标转换为原始图像坐标进行验证
                        orig_coord = convert_normalized_to_original_coordinate((x, y), image.shape, normalize_size)
                        orig_x, orig_y = int(orig_coord[0]), int(orig_coord[1])
                        
                        # 检查原始图像中的像素值
                        if (0 <= orig_x < image.shape[1] and 0 <= orig_y < image.shape[0] and
                            validate_coordinate((orig_x, orig_y), image)):
                            selected_coordinate = (x, y)
                            print(f"✓ Valid coordinate selected (normalized): {selected_coordinate}")
                            print(f"✓ Corresponding original coordinate: ({orig_x}, {orig_y})")
                            break
                        else:
                            print(f"⚠️ Coordinate {extracted_coord} (normalized) is not in navigable area")
                            print(f"⚠️ Original coordinate would be: ({orig_x}, {orig_y})")
                    else:
                        selected_coordinate = (x, y)
                        print(f"✓ Coordinate selected (validation disabled): {selected_coordinate}")
                        break
                else:
                    print(f"⚠️ Coordinate {extracted_coord} is out of bounds (0-{normalize_size})")
            else:
                print("⚠️ No coordinate found in LLM response")
            
            print(f"🔄 Retrying... ({retry_count + 1}/{max_retries})")
            retry_count += 1
                
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
    
    # 如果所有重试都失败了，使用fallback坐标
    if selected_coordinate is None:
        # 使用图像中心作为fallback (在标准化空间中)
        fallback_x = normalize_size // 2
        fallback_y = normalize_size // 2
        selected_coordinate = (fallback_x, fallback_y)
        print(f"⚠️ All LLM attempts failed. Using fallback coordinate (normalized): {selected_coordinate}")
        final_response = f"Fallback coordinate selection: ({fallback_x}, {fallback_y}) (LLM attempts failed)"
    else:
        final_response = all_responses[-1]["content"] if all_responses else "No response recorded"
    
    # 转换标准化坐标到原始图像坐标
    original_coordinate = convert_normalized_to_original_coordinate(
        selected_coordinate, image.shape, normalize_size
    )
    
    print(f"🎯 Final selection:")
    print(f"  - Normalized coordinate (1000x1000): {selected_coordinate}")
    print(f"  - Original coordinate ({image.shape[1]}x{image.shape[0]}): {original_coordinate}")
    
    return {
        "raw_response": final_response,
        "selected_coordinate_normalized": selected_coordinate,
        "selected_coordinate_original": original_coordinate,
        "model_used": model,
        "attempts_made": retry_count,
        "all_responses": all_responses,
        "normalization_info": {
            "scale_factor": scale_factor,
            "normalized_size": normalize_size,
            "original_shape": image.shape,
            "llm_input_was_original": True  # 标记LLM接收的是原始图像
        }
    }

def select_navigation_coordinate_onestage(topdown_path: str, 
                                        config: Dict[str, Any], 
                                        output_dir: str) -> Dict[str, Any]:
    """
    One-stage navigation coordinate selection directly from the topdown view.
    
    Args:
        topdown_path: Path to topdown view image
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading topdown image for coordinate selection:")
    print(f"  - Topdown image: {topdown_path}")
    
    # Load image
    topdown_image = cv2.imread(topdown_path)
    
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    
    print(f"✓ Loaded topdown image with shape: {topdown_image.shape}")
    
    # Use LLM to select the optimal coordinate
    try:
        llm_result = select_coordinate_with_llm(
            image=topdown_image,
            config=config,
            output_dir=output_dir
        )
        
        selected_coord_norm = llm_result['selected_coordinate_normalized']
        selected_coord_orig = llm_result['selected_coordinate_original']
        
    except Exception as e:
        print(f"⚠️ LLM coordinate selection failed: {e}")
        # Fallback to center of image
        h, w = topdown_image.shape[:2]
        fallback_coord = (w // 2, h // 2)
        selected_coord_orig = fallback_coord
        
        # Also calculate normalized coordinate
        normalize_size = config.get('coordinate_selection', {}).get('image_normalize_size', 1000)
        _, scale_factor = normalize_image_to_size(topdown_image, normalize_size)
        selected_coord_norm = (normalize_size // 2, normalize_size // 2)
        
        llm_result = {
            "raw_response": f"Fallback coordinate selection: {fallback_coord}",
            "selected_coordinate_normalized": selected_coord_norm,
            "selected_coordinate_original": selected_coord_orig,
            "model_used": "fallback",
            "error": str(e)
        }
    
    print(f"🎯 Selected coordinate: {selected_coord_orig} (original), {selected_coord_norm} (normalized)")
    
    # Create visualization with selected coordinate
    visualization_image = topdown_image.copy()
    
    # Draw coordinate marker
    x, y = int(selected_coord_orig[0]), int(selected_coord_orig[1])
    
    # Draw a red circle at the selected coordinate
    cv2.circle(visualization_image, (x, y), 10, (0, 0, 255), -1)  # Red filled circle
    cv2.circle(visualization_image, (x, y), 15, (255, 255, 255), 3)  # White border
    
    # Draw coordinate text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text = f"({x}, {y})"
    
    # Position text above the circle
    text_x = x - 50
    text_y = y - 25
    
    # Draw text with background
    cv2.putText(visualization_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)  # Black outline
    cv2.putText(visualization_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)  # White text
    
    # Save visualization
    coordinate_visualization_path = os.path.join(output_dir, "selected_coordinate_visualization.png")
    cv2.imwrite(coordinate_visualization_path, visualization_image)
    print(f"✓ Coordinate visualization saved to: {coordinate_visualization_path}")
    
    # Create normalized visualization
    normalize_size = config.get('coordinate_selection', {}).get('image_normalize_size', 1000)
    normalized_image, _ = normalize_image_to_size(topdown_image, normalize_size)
    
    norm_x, norm_y = selected_coord_norm
    cv2.circle(normalized_image, (norm_x, norm_y), 8, (0, 0, 255), -1)  # Red filled circle
    cv2.circle(normalized_image, (norm_x, norm_y), 12, (255, 255, 255), 2)  # White border
    
    # Draw coordinate text on normalized image
    norm_text = f"({norm_x}, {norm_y})"
    norm_text_x = norm_x - 40
    norm_text_y = norm_y - 20
    
    cv2.putText(normalized_image, norm_text, (norm_text_x, norm_text_y), font, 0.7, (0, 0, 0), 3)  # Black outline
    cv2.putText(normalized_image, norm_text, (norm_text_x, norm_text_y), font, 0.7, (255, 255, 255), 2)  # White text
    
    normalized_visualization_path = os.path.join(output_dir, "selected_coordinate_normalized_visualization.png")
    cv2.imwrite(normalized_visualization_path, normalized_image)
    print(f"✓ Normalized coordinate visualization saved to: {normalized_visualization_path}")
    
    # Save coordinate selection log
    selection_log = {
        "method": "one_stage_coordinate_selection",
        "selected_coordinate_original": selected_coord_orig,
        "selected_coordinate_normalized": selected_coord_norm,
        "llm_response": llm_result,
        "goal_object": config['scene_config']['goal_object'],
        "topdown_image_shape": topdown_image.shape,
        "normalization_info": llm_result.get('normalization_info', {})
    }
    
    log_path = os.path.join(output_dir, "coordinate_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(selection_log, f, indent=2, ensure_ascii=False)
    print(f"✓ Coordinate selection log saved to: {log_path}")
    
    # 保存LLM的完整内容到预处理位置
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    # 保存最终成功的LLM响应的完整内容
    if 'all_responses' in llm_result and llm_result['all_responses']:
        # 找到最终成功的响应
        successful_response = None
        for response in reversed(llm_result['all_responses']):
            if "error" not in response and response.get("extracted_coordinate"):
                successful_response = response
                break
        
        if successful_response:
            llm_content_data = {
                "timestamp": successful_response["timestamp"],
                "model": llm_result['model_used'],
                "selected_coordinate_normalized": selected_coord_norm,
                "selected_coordinate_original": selected_coord_orig,
                "full_reasoning_content": successful_response["raw_response"],
                "full_content": successful_response["content"],
                "extracted_coordinate": successful_response["extracted_coordinate"],
                "attempt_number": successful_response["attempt"],
                "goal_object": config['scene_config']['goal_object'],
                "method": "one_stage_coordinate_selection",
                "prompt_used": "Coordinate selection prompt"
            }
        else:
            # 如果没有成功的响应，使用fallback信息
            llm_content_data = {
                "timestamp": str(__import__('datetime').datetime.now()),
                "model": llm_result['model_used'],
                "selected_coordinate_normalized": selected_coord_norm,
                "selected_coordinate_original": selected_coord_orig,
                "full_reasoning_content": llm_result['raw_response'],
                "full_content": llm_result['raw_response'],
                "extracted_coordinate": selected_coord_norm,
                "attempt_number": "fallback",
                "goal_object": config['scene_config']['goal_object'],
                "method": "one_stage_coordinate_selection",
                "prompt_used": "Coordinate selection prompt",
                "note": "Used fallback coordinate selection due to LLM failures"
            }
    else:
        # 如果没有任何响应，创建默认记录
        llm_content_data = {
            "timestamp": str(__import__('datetime').datetime.now()),
            "model": llm_result.get('model_used', 'unknown'),
            "selected_coordinate_normalized": selected_coord_norm,
            "selected_coordinate_original": selected_coord_orig,
            "full_reasoning_content": "No valid LLM response received",
            "full_content": "No valid LLM response received",
            "extracted_coordinate": selected_coord_norm,
            "attempt_number": 0,
            "goal_object": config['scene_config']['goal_object'],
            "method": "one_stage_coordinate_selection",
            "prompt_used": "Coordinate selection prompt",
            "note": "No valid LLM response - using fallback"
        }
    
    # 保存到preprocess目录
    llm_content_path = os.path.join(preprocess_dir, "llm_coordinate_selection_content.json")
    with open(llm_content_path, 'w', encoding='utf-8') as f:
        json.dump(llm_content_data, f, indent=2, ensure_ascii=False)
    print(f"✓ LLM content saved to preprocessing directory: {llm_content_path}")
    
    # 同时保存一个简化的reasoning文本文件，便于阅读
    reasoning_text_path = os.path.join(preprocess_dir, "llm_reasoning.txt")
    with open(reasoning_text_path, 'w', encoding='utf-8') as f:
        f.write(f"LLM Coordinate Selection Reasoning\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {llm_content_data['timestamp']}\n")
        f.write(f"Model: {llm_content_data['model']}\n")
        f.write(f"Method: {llm_content_data['method']}\n")
        f.write(f"Goal Object: {llm_content_data['goal_object']}\n")
        f.write(f"Selected Coordinate (Normalized): {llm_content_data['selected_coordinate_normalized']}\n")
        f.write(f"Selected Coordinate (Original): {llm_content_data['selected_coordinate_original']}\n")
        f.write(f"Attempt Number: {llm_content_data['attempt_number']}\n\n")
        f.write(f"Full LLM Response:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"{llm_content_data['full_reasoning_content']}\n\n")
        f.write(f"Extracted Coordinate: {llm_content_data['extracted_coordinate']}\n")
        if 'note' in llm_content_data:
            f.write(f"\nNote: {llm_content_data['note']}\n")
    print(f"✓ LLM reasoning text saved to: {reasoning_text_path}")
    
    return {
        "generated_files": {
            "coordinate_visualization": coordinate_visualization_path,
            "normalized_visualization": normalized_visualization_path,
            "coordinate_selection_log": log_path,
            "llm_content": llm_content_path,
            "reasoning_text": reasoning_text_path
        },
        "llm_response": llm_result,
        "results": {
            "selected_coordinate_original": selected_coord_orig,
            "selected_coordinate_normalized": selected_coord_norm,
            "method": "one_stage_coordinate_selection",
            "goal_object": config['scene_config']['goal_object']
        }
    }

# Fallback function for testing without LLM
def select_coordinate_manually_onestage(topdown_path: str, output_dir: str, 
                                      coordinate: Tuple[int, int] = (500, 500)) -> Dict[str, Any]:
    """
    Fallback function to manually select a coordinate for testing.
    
    Args:
        topdown_path: Path to topdown view image
        output_dir: Output directory path  
        coordinate: Coordinate to select (default: (500, 500))
        
    Returns:
        Dictionary with manual selection results
    """
    print(f"🔧 Manual coordinate selection: {coordinate}")
    
    # Load image
    topdown_image = cv2.imread(topdown_path)
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    
    # Normalize coordinate to original image
    h, w = topdown_image.shape[:2]
    
    # If coordinate is in normalized space (0-1000), convert to original space
    if coordinate[0] <= 1000 and coordinate[1] <= 1000:
        # Assume it's normalized coordinate
        normalize_size = 1000
        original_coord = convert_normalized_to_original_coordinate(coordinate, topdown_image.shape, normalize_size)
        original_coord = (int(original_coord[0]), int(original_coord[1]))
        normalized_coord = coordinate
    else:
        # Assume it's already in original space
        original_coord = coordinate
        # Convert to normalized
        _, scale_factor = normalize_image_to_size(topdown_image, 1000)
        normalized_coord = (int(coordinate[0] * scale_factor), int(coordinate[1] * scale_factor))
    
    # Clamp coordinates to image bounds
    original_coord = (
        max(0, min(w-1, original_coord[0])),
        max(0, min(h-1, original_coord[1]))
    )
    normalized_coord = (
        max(0, min(1000, normalized_coord[0])),
        max(0, min(1000, normalized_coord[1]))
    )
    
    manual_log = {
        "method": "manual_coordinate_selection",
        "selected_coordinate_original": original_coord,
        "selected_coordinate_normalized": normalized_coord,
        "topdown_image": topdown_path
    }
    
    log_path = os.path.join(output_dir, "manual_coordinate_selection_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(manual_log, f, indent=2, ensure_ascii=False)
    
    # 保存手动选择的内容到预处理位置，保持与LLM选择的一致性
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    manual_content_data = {
        "timestamp": str(__import__('datetime').datetime.now()),
        "model": "manual_selection",
        "selected_coordinate_original": original_coord,
        "selected_coordinate_normalized": normalized_coord,
        "full_reasoning_content": f"Manual coordinate selection: ({original_coord[0]}, {original_coord[1]}) was manually specified",
        "full_content": f"Manual selection: coordinate {original_coord}",
        "extracted_coordinate": normalized_coord,
        "attempt_number": 1,
        "goal_object": "unknown (manual selection)",
        "method": "one_stage_coordinate_selection_manual",
        "prompt_used": "N/A (manual selection)",
        "note": "Coordinate was manually selected, not through LLM reasoning"
    }
    
    # 保存到preprocess目录
    manual_content_path = os.path.join(preprocess_dir, "llm_coordinate_selection_content.json")
    with open(manual_content_path, 'w', encoding='utf-8') as f:
        json.dump(manual_content_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Manual selection content saved to preprocessing directory: {manual_content_path}")
    
    # 同时保存一个简化的reasoning文本文件
    reasoning_text_path = os.path.join(preprocess_dir, "llm_reasoning.txt")
    with open(reasoning_text_path, 'w', encoding='utf-8') as f:
        f.write(f"Coordinate Selection Reasoning (Manual)\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {manual_content_data['timestamp']}\n")
        f.write(f"Method: Manual Coordinate Selection\n")
        f.write(f"Selected Coordinate (Original): {manual_content_data['selected_coordinate_original']}\n")
        f.write(f"Selected Coordinate (Normalized): {manual_content_data['selected_coordinate_normalized']}\n\n")
        f.write(f"Reasoning:\n")
        f.write(f"{'-' * 20}\n")
        f.write(f"{manual_content_data['full_reasoning_content']}\n\n")
        f.write(f"Final Answer: {manual_content_data['extracted_coordinate']}\n")
        f.write(f"\nNote: {manual_content_data['note']}\n")
    print(f"✓ Manual selection reasoning text saved to: {reasoning_text_path}")
    
    return {
        "generated_files": {
            "selection_log": log_path,
            "llm_content": manual_content_path,
            "reasoning_text": reasoning_text_path
        },
        "llm_response": {
            "raw_response": f"Manual selection: coordinate {original_coord}",
            "selected_coordinate_original": original_coord,
            "selected_coordinate_normalized": normalized_coord,
            "model_used": "manual"
        },
        "results": {
            "selected_coordinate_original": original_coord,
            "selected_coordinate_normalized": normalized_coord,
            "method": "one_stage_coordinate_selection_manual"
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python step_3_onestage_coordinate_selection.py <topdown_path> <config_path> [manual_coordinate_x,y]")
        print("If manual_coordinate is provided, manual selection will be used instead of LLM")
        print("Example: python step_3_onestage_coordinate_selection.py topdown.png config.json 500,600")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    config_path = sys.argv[2]
    manual_coord = None
    
    if len(sys.argv) > 3:
        try:
            x, y = map(int, sys.argv[3].split(','))
            manual_coord = (x, y)
        except ValueError:
            print("Error: Manual coordinate should be in format 'x,y' (e.g., '500,600')")
            sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if manual_coord is not None:
        result = select_coordinate_manually_onestage(topdown_path, output_dir, manual_coord)
        print("Step 3 (coordinate-based) completed (manual):", result)
    else:
        result = select_navigation_coordinate_onestage(topdown_path, config, output_dir)
        print("Step 3 (coordinate-based) completed (LLM):", result)
