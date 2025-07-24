from google import genai
from google.genai import types
import PIL.Image
import os
import json
import time
from tqdm import tqdm

# --- 配置 ---




# --- Gemini API 调用函数 (重构后) ---
def classify_image_with_gemini(image_path: str, base_image_path: str):
    """
    Calls the Gemini 2.5 flash API using the client/types.Part style.
    
    Args:
        image_path: Path to the contextual crop image.
        
    Returns:
        A dictionary with the classification result, or None on error.
    """
    try:
        # 读取图片为二进制数据
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        with open(base_image_path, 'rb') as f:
            base_image_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    # 实例化一个客户端
    api_key = os.getenv('API_KEY')
    http_options = types.HttpOptions(base_url="https://api.openai-proxy.org/google")
    client = genai.Client(api_key=api_key, http_options=http_options)

    # 精心设计的Prompt
    system_prompt_text = """
    You are an expert robot perception assistant. Your task is to determine if an object in a top-down view of a room is an obstacle for a wheeled cleaning robot. 
    
    You're given a global view of the room and a local view of the object for you to decide. The object was highlighted in semi-transparent red.

    An 'obstacle' is something the robot CANNOT or SHOULD NOT drive over. This includes furniture, walls, large objects, fragile items, or things that could entangle the robot.

    A 'non-obstacle' is something the robot CAN safely drive over. This primarily includes flat floor coverings like rugs, carpets, or doormats.

    **The image given is a flat room, no raised platform or steps/stairs exist. Then you can safely classify the connection or different rooms as not an obstacle.**

    The selection criteria are:
    1. The obstacles appears in the indoor scene are the usual furniture, walls, large objects, fragile items, or things that could entangle the robot. **Only if you consider the given object is a furniture, you should classify it as an obstacle.**
    2. If you are not confident about the object, you should classify it as not an obstacle.
    3. For the connection parts between different rooms. even the colors are not the same, you *should not* classify it as an obstacle. If the area seems to be next to a door, you *should not* classify it as an obstacle.
    4. If the object seems to be a part of the room, you *should not* classify it as an obstacle.

    
    You MUST respond in a valid JSON format with two keys:
    1. "is_obstacle": a boolean value (true or false).
    2. "reason": a brief, one-sentence explanation for your decision.
    
    Example response for a chair:
    {
      "is_obstacle": true, 
      "reason": "This is a chair, which is a piece of furniture that a robot cannot drive over."
    }
    
    Example response for a room:
    {
      "is_obstacle": false,
      "reason": "This is a flat room. Then you can safely classify the connection or different rooms as not an obstacle."
    }
    """
    
    user_prompt_text = "Analyze the highlighted object in this image and provide your assessment in the required JSON format."
    # --- 【核心修改】 ---
    # 使用 types.Part 来构建请求内容
    # 文件上传主要用于多轮对话中复用，对于单次请求，直接传入bytes更高效
    image_part_local = types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/png' # 明确指定MIME类型
    )
    image_part_global = types.Part.from_bytes(
        data=base_image_bytes,
        mime_type='image/png' # 明确指定MIME类型
    )
    
    prompt_parts = [
        system_prompt_text,
        user_prompt_text,
        image_part_local,
        image_part_global
    ]
    # --- 【修改结束】 ---

    # 发起API请求
    try:
        # generate_content 的参数现在是 contents=[...parts...]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts,
            config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            )
            # (可选) 你可以在这里添加 generation_config, safety_settings 等
        )
        
        # 清理并解析JSON响应
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(response_text)
        
        if "is_obstacle" in result and "reason" in result:
            return result
        else:
            print(f"Error: API response for {image_path} is missing required keys. Response: {response_text}")
            return None

    except Exception as e:
        print(f"An error occurred while calling Gemini API for {image_path}: {e}")
        # 访问失败响应的文本可能需要通过 response.prompt_feedback 检查
        if 'response' in locals() and hasattr(response, 'text'):
             print(f"Failed response text: {response.text}")
        else:
             print("No valid response object.")
        return None

# --- 主逻辑 (完全不变) ---
def chat(base_dir, file_name):
    INPUT_METADATA_PATH = os.path.join(base_dir, "data", "processed", file_name, "metadata.json")
    BASE_INPUT_DIR = os.path.join(base_dir, "data", "processed", file_name)
    ORIGIN_IMAGE_PATH = os.path.join(base_dir, "data", "top_down", file_name+".png")
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
        
        # 使用更新后的列表，保留已分类数据，并只迭代未分类的部分
        # 注意：这里需要确保合并后的列表没有重复项，并且能正确迭代
        processed_items_dict = {item['id']: item for item in classified_data}
        unprocessed_items = []
        for item in metadata_list:
            if item['id'] not in processed_items_dict:
                unprocessed_items.append(item)
            else: # 更新元数据，以防原始json有变动但id相同
                processed_items_dict[item['id']].update(item)
        
        metadata_list = list(processed_items_dict.values())
        unclassified_objects = unprocessed_items # 只迭代未处理的

    else:
        unclassified_objects = metadata_list
        print("No existing output file. Starting from scratch.")

    print(f"\n--- Step 2: Classifying Objects using Gemini ---")
    
    newly_classified_items = []
    # 使用tqdm创建进度条
    for item in tqdm(unclassified_objects, desc="Classifying Objects"):
        crop_path = os.path.join(BASE_INPUT_DIR, item['crop_image_path'])
        classification_result = classify_image_with_gemini(crop_path, ORIGIN_IMAGE_PATH)
        
        if classification_result:
            item['classification'] = classification_result
        else:
            item['classification'] = {"is_obstacle": True, "reason": "API call failed or returned invalid format."}

        newly_classified_items.append(item)
        time.sleep(1)

    print(f"\n--- Step 3: Saving Final Results ---")
    
    # 合并旧的和新分类的结果
    final_data = metadata_list if os.path.exists(OUTPUT_METADATA_PATH) else []
    
    # 更新已有条目并添加新条目
    final_data_dict = {item['id']: item for item in final_data}
    for item in newly_classified_items:
        final_data_dict[item['id']] = item

    # 排序并保存
    sorted_final_data = sorted(list(final_data_dict.values()), key=lambda x: x['id'])
    
    with open(OUTPUT_METADATA_PATH, 'w') as f:
        json.dump(sorted_final_data, f, indent=4)
        
    print(f"Classification complete. Final results saved to {OUTPUT_METADATA_PATH}")

