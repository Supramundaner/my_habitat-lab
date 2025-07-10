import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm



def create_unwalkable_map(base_dir, file_name):
    """
    Combines a wall map with AI-classified obstacle masks to create a final unwalkable area map.
    In the final map, black (0) is unwalkable, white (255) is walkable.
    """
    WALL_MAP_PATH = os.path.join(base_dir, "data", "processed", file_name, "threshold.png")
    MASKS_DIR = os.path.join(base_dir, "data", "processed", file_name, "masks")
    METADATA_PATH = os.path.join(base_dir, "data", "processed", file_name, "metadata_with_obstacles.json")
    OUTPUT_UNWALKABLE_MAP_PATH = os.path.join(base_dir, "data", "processed", file_name, "unwalkable_map_final.png")

    print("--- Step 1: Loading Input Data ---")

    # 1. 加载墙体图
    try:
        wall_map_img = Image.open(WALL_MAP_PATH).convert('L') # 'L'模式为灰度图
        wall_map_np = np.array(wall_map_img)
        print(f"Loaded wall map from '{WALL_MAP_PATH}' with shape {wall_map_np.shape}")
    except FileNotFoundError:
        print(f"Error: Wall map not found at '{WALL_MAP_PATH}'")
        return

    # 2. 加载AI分类的元数据
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f)
        print(f"Loaded metadata for {len(metadata_list)} objects from '{METADATA_PATH}'")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{METADATA_PATH}'")
        return

    # 3. 初始化最终的不可行走地图
    # 创建一个和墙体图一样大的全白图片 (255代表可行走)
    unwalkable_map_np = np.full(wall_map_np.shape, 255, dtype=np.uint8)
    
    # 将墙体（在wall_map_np中值为0的像素）绘制到新地图上，设为黑色 (0)
    # np.where(condition, value_if_true, value_if_false)
    unwalkable_map_np = np.where(wall_map_np == 0, 0, unwalkable_map_np)
    print("Initialized final map and added walls.")

    print("\n--- Step 2: Adding Obstacles to the Map ---")
    
    obstacle_count = 0
    # 4. 遍历所有物体
    for item in tqdm(metadata_list, desc="Processing Obstacles"):
        # 5. 检查是否为障碍物
        # 确保 'classification' 键存在且 'is_obstacle' 为 True
        if 'classification' in item and item['classification'].get('is_obstacle') is True:
            obstacle_count += 1
            mask_path = os.path.join(MASKS_DIR, os.path.basename(item['mask_path']))
            
            try:
                # 读取该障碍物的全尺寸二值掩码图
                obstacle_mask_img = Image.open(mask_path).convert('L')
                obstacle_mask_np = np.array(obstacle_mask_img)

                # 确保mask尺寸一致
                if obstacle_mask_np.shape != unwalkable_map_np.shape:
                    print(f"Warning: Mask {mask_path} has a different shape. Skipping.")
                    continue

                # 将这个障碍物的区域（在mask中值为255的像素）也设置为不可行走 (0)
                # 这就是取并集 (Union) 的操作
                unwalkable_map_np[obstacle_mask_np == 255] = 0

            except FileNotFoundError:
                print(f"Warning: Mask file not found at '{mask_path}'. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing mask {mask_path}: {e}")

    print(f"\nProcessed a total of {obstacle_count} obstacles and added them to the map.")

    print("\n--- Step 3: Saving the Final Unwalkable Map ---")
    
    # 6. 保存最终的地图
    final_map_img = Image.fromarray(unwalkable_map_np)
    final_map_img.save(OUTPUT_UNWALKABLE_MAP_PATH)
    
    print(f"Successfully created and saved the final unwalkable map to '{OUTPUT_UNWALKABLE_MAP_PATH}'")
    print("In the output image: Black = Unwalkable (walls + obstacles), White = Walkable.")
