import cv2
import numpy as np
import os

SPACING_IN_METERS_PER_PIXEL = 0.0108  # (m/pixel) 
MASK_IMAGE_PATH = "/home/yaoaa/habitat-lab/habitat_video_project/eval/output/y9hTuugGdiq/7151/preprocess/wall_mask.png"
OUTPUT_IMAGE_PATH = "/home/yaoaa/habitat-lab/room_segment.png"

# 定义一个物理距离（米），小于这个宽度的缝隙或孔洞将被闭运算填充。
# 例如，0.1米（10厘米）可以有效过滤掉门缝或小噪声。
MORPH_CLOSING_WIDTH_METERS = 0.01

# 房间种子距离墙壁的最小距离设置：
# - 设置为 None: 使用 Otsu 自动阈值化动态确定最佳距离（推荐）
# - 设置为具体数值（如 0.9）: 使用固定阈值
# 定义一个物理距离（米）。任何像素点，如果它离最近墙壁的距离大于这个值，
# 就被认为是房间的"核心"或"种子"。
# 设置为 None 表示使用 Otsu 自动阈值化动态确定最佳距离
# 设置为具体数值（如 0.9）表示使用固定阈值
SEED_MIN_DISTANCE_FROM_WALL_METERS = None  # 使用 None 启用动态阈值

# 4. 显示与调试 (Display and Debugging)
SHOW_VISUALIZATION_WINDOW = True
DEBUG_MODE = True
DEBUG_OUTPUT_DIR = "debug_steps_physical"


# ==============================================================================
# 核心代码逻辑 (Core Code Logic) - 通常无需修改以下部分
# ==============================================================================

def segment_rooms_physical(mask_image_path: str, spacing: float, closing_width_m: float, seed_dist_m: float = None) -> np.ndarray:
    """
    Segments rooms using physics-based parameters.
    
    Args:
        mask_image_path: Path to the input mask image
        spacing: Meters per pixel resolution
        closing_width_m: Physical width for morphological closing in meters
        seed_dist_m: Minimum distance from wall for room seeds in meters. 
                    If None, uses Otsu's automatic thresholding.
    
    Returns:
        Segmented room image as numpy array
    """
    if DEBUG_MODE and not os.path.exists(DEBUG_OUTPUT_DIR):
        os.makedirs(DEBUG_OUTPUT_DIR)
        
    # --- 参数从物理单位（米）转换为像素单位 ---
    # 计算形态学核的像素大小。确保结果是奇数。
    kernel_size_px = int(closing_width_m / spacing)
    if kernel_size_px % 2 == 0: kernel_size_px += 1
    
    print("--- 物理参数到像素参数的转换 ---")
    print(f"形态学核宽度: {closing_width_m} m  ->  {kernel_size_px} pixels")
    
    if seed_dist_m is not None:
        # 使用固定的种子距离阈值
        seed_dist_px = seed_dist_m / spacing
        print(f"种子距离阈值: {seed_dist_m} m  ->  {seed_dist_px:.2f} pixels (固定)")
    else:
        print("将使用 Otsu 自动阈值化动态确定种子距离阈值")
    print("------------------------------------")

    # 1. Load the image and create a binary mask
    img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"错误: 在路径 '{mask_image_path}' 未找到图像文件。")
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "0_original_gray.png"), img)
    
    # 2. Binarize the mask
    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "1_binary_mask.png"), binary_mask)

    # 3. Pre-processing with morphological closing using the calculated kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_px, kernel_size_px))
    closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # 迭代次数也可以调整
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "2_closed_mask.png"), closing)

    # 4. Compute Euclidean Distance Field (EDF)
    # 距离变换的结果直接就是像素距离，这正是我们需要的！
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    
    if DEBUG_MODE:
        # 保存原始距离变换（用于调试）
        dist_vis_raw = dist_transform.copy()
        cv2.normalize(dist_vis_raw, dist_vis_raw, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "3_distance_transform_raw.png"), dist_vis_raw.astype(np.uint8))

    # 5. Derive seed regions (sure foreground)
    if seed_dist_m is not None:
        # 使用固定阈值
        seed_dist_px = seed_dist_m / spacing
        _, sure_fg = cv2.threshold(dist_transform, seed_dist_px, 255, cv2.THRESH_BINARY)
        print(f"使用固定阈值: {seed_dist_px:.2f} pixels ({seed_dist_m} m)")
    else:
        # 使用 Otsu 自动阈值化 (参考 distance_transform 函数)
        print(f"距离变换范围: {np.min(dist_transform):.2f} 到 {np.max(dist_transform):.2f} pixels")
        
        # 归一化距离变换用于 Otsu 阈值化
        dist_normalized = dist_transform.copy()
        cv2.normalize(dist_normalized, dist_normalized, 0, 255, cv2.NORM_MINMAX)
        dist_normalized = np.uint8(dist_normalized)
        
        # 应用高斯模糊后进行 Otsu 阈值化 (参考原函数)
        blur = cv2.GaussianBlur(dist_normalized, (11, 11), 0)  # 使用对称核
        if DEBUG_MODE:
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "3b_distance_blur.png"), blur)
        
        # Otsu 阈值化
        otsu_threshold, sure_fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 将阈值转换回原始距离变换的尺度
        actual_threshold_px = (otsu_threshold / 255.0) * np.max(dist_transform)
        actual_threshold_m = actual_threshold_px * spacing
        
        print(f"Otsu 自动阈值: {otsu_threshold} (归一化)")
        print(f"实际距离阈值: {actual_threshold_px:.2f} pixels ({actual_threshold_m:.3f} m)")
        
        if DEBUG_MODE:
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "3c_distance_otsu.png"), sure_fg)
    
    sure_fg = np.uint8(sure_fg)
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "4_sure_foreground_seeds.png"), sure_fg)
    
    # 后续步骤与之前版本逻辑相同...
    # 6. Define sure background and unknown regions
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "5_unknown_region.png"), unknown)

    # 7. Create markers for the watershed algorithm
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 8. Apply the watershed algorithm
    img_color_for_watershed = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color_for_watershed, markers)

    # 9. Visualize the result
    num_rooms = np.max(markers)
    colors = [np.random.randint(50, 256, 3).tolist() for _ in range(num_rooms + 1)]
    segmented_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

    for i in range(1, num_rooms + 1):
        segmented_image[markers == i] = colors[i]
    
    segmented_image[markers == -1] = [0, 0, 255] # Red boundaries

    # 10. Restore original walls
    segmented_image[binary_mask == 0] = [0, 0, 0]

    return segmented_image

def main():
    """
    主执行函数，读取全局配置并运行分割流程。
    """
    try:
        segmented_result = segment_rooms_physical(
            MASK_IMAGE_PATH,
            SPACING_IN_METERS_PER_PIXEL,
            MORPH_CLOSING_WIDTH_METERS,
            SEED_MIN_DISTANCE_FROM_WALL_METERS
        )
        
        # ... 后续的保存和显示代码与之前版本完全相同 ...
        if OUTPUT_IMAGE_PATH is None:
            base_name = os.path.basename(MASK_IMAGE_PATH)
            name, ext = os.path.splitext(base_name)
            output_filename = f"segmented_{name}{ext}"
        else:
            output_filename = OUTPUT_IMAGE_PATH
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        cv2.imwrite(output_filename, segmented_result)
        print(f"分割成功！结果已保存至: {output_filename}")
        if DEBUG_MODE:
            print(f"调试步骤已保存至目录: '{DEBUG_OUTPUT_DIR}'")

        if SHOW_VISUALIZATION_WINDOW:
            original_mask_color = cv2.imread(MASK_IMAGE_PATH)
            if original_mask_color.shape[:2] != segmented_result.shape[:2]:
                segmented_result_resized = cv2.resize(segmented_result, (original_mask_color.shape[1], original_mask_color.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                segmented_result_resized = segmented_result
            combined_display = np.hstack([original_mask_color, segmented_result_resized])
            cv2.imshow("Original Mask vs. Physical Segmented Rooms", combined_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n程序运行中发生未知错误: {e}")

if __name__ == '__main__':
    main()