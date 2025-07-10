import json
import os
import time
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def classify_image_with_local_model(image_path: str, model, processor):

    """
    Calls the local Qwen-VL model to classify an image.
    
    Args:
        image_path: Path to the contextual crop image.
        
    Returns:
        A dictionary with the classification result, or None on error.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None


    # 沿用原始脚本中精心设计的Prompt
    system_prompt_text = """
    You are an expert robot perception assistant. Your task is to determine if an object in a top-down view of a room is an obstacle for a wheeled cleaning robot.
    An 'obstacle' is something the robot CANNOT or SHOULD NOT drive over. This includes furniture, walls, large objects, fragile items, or things that could entangle the robot.
    A 'non-obstacle' is something the robot CAN safely drive over. This primarily includes flat floor coverings like rugs, carpets, or doormats.

    **The image given is a flat room, no raised platform or steps/stairs exist. Then you can safely classify the connection or different rooms as not an obstacle.**
    
    The user will provide an image where the object in question is highlighted in semi-transparent red.

    The selection criteria are:
    1. The obstacles appears in the indoor scene are the usual furniture (chair, table, sofa, etc.), walls, large objects, fragile items, or things that could entangle the robot. **Only if you consider the given object is a furniture, you should classify it as an obstacle.**
    2. If you are not confident about the object, you should classify it as not an obstacle.
    3. For the connection parts between different rooms. even the colors are not the same, you *should not* classify it as an obstacle. If the area seems to be next to a door, you *should not* classify it as an obstacle.

    
    You MUST respond in a valid JSON format with two keys:
    1. "is_obstacle": a boolean value (true or false).
    2. "reason": a brief, one-sentence explanation for your decision.
    
    Example response for a chair:
    {
      "is_obstacle": true, 
      "reason": "This is a chair, which is a piece of furniture that a robot cannot drive over."
    }
    
    Example response for a rug:
    {
      "is_obstacle": false,
      "reason": "This is a flat rug on the floor, which a robot can safely traverse."
    }
    """
    
    user_prompt_text = "Analyze the highlighted object in this image and provide your assessment in the required JSON format."

    try:
        # 准备模型输入
        image = Image.open(image_path)
        
        # 构建符合Qwen-VL格式的messages
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt_text},
                # 注意：apply_chat_template需要一个假的image占位符来生成正确的文本prompt
                # 真正的image对象会在下一步的processor调用中传入
                {"image": "placeholder"}, 
            ]},
        ]
        
        # 生成文本prompt并准备输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        # 模型推理
        # 将max_new_tokens设置得合理一些，因为我们只需要一个简短的JSON
        output_ids = model.generate(**inputs, max_new_tokens=256)
        
        # 解码并提取纯文本输出
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # 清理并解析JSON响应 (与原始脚本逻辑相同)
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_text)
        
        if "is_obstacle" in result and "reason" in result:
            return result
        else:
            print(f"Error: Local model response for {image_path} is missing required keys. Response: {cleaned_text}")
            return None

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from model response for {image_path}.")
        print(f"Raw response: '{response_text}'")
        return None
    except Exception as e:
        print(f"An error occurred while calling local model for {image_path}: {e}")
        return None


# --- 4. 主逻辑 (与原始脚本完全不变) ---
def chat(base_dir, file_name, model, processor):
    
    INPUT_METADATA_PATH = os.path.join(base_dir, "data", "processed", file_name, "metadata.json")
    BASE_INPUT_DIR = os.path.join(base_dir, "data", "processed", file_name)
    OUTPUT_METADATA_PATH = os.path.join(base_dir, "data", "processed", file_name, "metadata_with_obstacles.json")

    print(f"--- Step 1: Loading Metadata ---")
    if not os.path.exists(INPUT_METADATA_PATH):
        print(f"Error: Metadata file not found at {INPUT_METADATA_PATH}")
        return
        
    with open(INPUT_METADATA_PATH, 'r') as f:
        metadata_list = json.load(f)
    print(f"Loaded {len(metadata_list)} objects to classify.")

    # 断点续传逻辑
    if os.path.exists(OUTPUT_METADATA_PATH):
        print("Found existing output file. Resuming from last classified object.")
        with open(OUTPUT_METADATA_PATH, 'r') as f:
            classified_data = json.load(f)
        classified_ids = {item['id'] for item in classified_data if 'classification' in item}
        unclassified_objects = [item for item in metadata_list if item['id'] not in classified_ids]
        print(f"{len(classified_ids)} objects already classified. {len(unclassified_objects)} remaining.")
        
        processed_items_dict = {item['id']: item for item in classified_data}
        unprocessed_items = []
        for item in metadata_list:
            if item['id'] not in processed_items_dict:
                unprocessed_items.append(item)
            else:
                processed_items_dict[item['id']].update(item)
        
        metadata_list = list(processed_items_dict.values())
        unclassified_objects = unprocessed_items

    else:
        unclassified_objects = metadata_list
        print("No existing output file. Starting from scratch.")

    print(f"\n--- Step 2: Classifying Objects using Local Qwen-VL Model ---")
    
    newly_classified_items = []
    for item in tqdm(unclassified_objects, desc="Classifying Objects"):
        crop_path = os.path.join(BASE_INPUT_DIR, item['crop_image_path'])
        
        classification_result = classify_image_with_local_model(crop_path, model, processor)
        
        if classification_result:
            item['classification'] = classification_result
        else:
            # 失败时的回退逻辑
            item['classification'] = {"is_obstacle": True, "reason": "Local model call failed or returned invalid format."}

        newly_classified_items.append(item)

    print(f"\n--- Step 3: Saving Final Results ---")
    
    final_data = metadata_list if os.path.exists(OUTPUT_METADATA_PATH) else []
    
    final_data_dict = {item['id']: item for item in final_data}
    for item in newly_classified_items:
        final_data_dict[item['id']] = item

    sorted_final_data = sorted(list(final_data_dict.values()), key=lambda x: x['id'])
    
    with open(OUTPUT_METADATA_PATH, 'w') as f:
        json.dump(sorted_final_data, f, indent=4)
        
    print(f"Classification complete. Final results saved to {OUTPUT_METADATA_PATH}")

if __name__ == "__main__":
    chat()