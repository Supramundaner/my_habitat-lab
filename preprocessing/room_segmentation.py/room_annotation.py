# room_annotation.py (version 5 - The Bug Squasher)

import cv2
import numpy as np
import os
# 导入修改后的函数
from room_segment import segment_rooms_physical

# ... (全局参数和 find_robust_center 函数保持不变) ...
# ==============================================================================
# 全局参数配置区域 (Global Parameters Configuration)
# ==============================================================================
ORIGINAL_IMAGE_PATH = "/home/yaoaa/habitat-lab/preprocessing/data/top_down/sample.png"
MASK_IMAGE_PATH = "/home/yaoaa/habitat-lab/preprocessing/data/processed/sample/threshold.png"
ANNOTATED_IMAGE_PATH = "annotated_room_view.png"
SPACING_IN_METERS_PER_PIXEL = 0.004128490445715825
MORPH_CLOSING_WIDTH_METERS = 0.01
SEED_MIN_DISTANCE_FROM_WALL_METERS = 0.9
HIGH_CONTRAST_COLORS = [(60,20,220),(255,128,0),(40,180,0),(0,210,255),(230,0,180),(255,128,128),(0,128,255),(128,0,128)]
ANNOTATION_ALPHA = 0.4
BOUNDARY_COLOR = (0, 0, 255) # Red
BOUNDARY_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_COLOR = (255, 255, 255) # White
FONT_THICKNESS = 2
FONT_BACKGROUND_COLOR = (0, 0, 0) # Black
MIN_ROOM_AREA_PIXELS = 1000

# ==============================================================================
# 核心代码逻辑
# ==============================================================================
def find_robust_center(mask, contour):
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    dist = cv2.pointPolygonTest(contour, (center_x, center_y), False)
    if dist > 0:
        return (center_x, center_y)
    else:
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        return max_loc

def annotate_rooms_on_image(original_img_path, mask_img_path):
    original_image = cv2.imread(original_img_path)
    if original_image is None:
        raise FileNotFoundError(f"错误: 原始图像 '{original_img_path}' 未找到。")
    
    annotated_image = original_image.copy()
    overlay = original_image.copy()
    
    print("步骤 1/4: 正在使用掩码图进行房间分割...")
    # --- **核心修改点**: 接收两个返回值 ---
    _, markers = segment_rooms_physical(
        mask_img_path,
        SPACING_IN_METERS_PER_PIXEL,
        MORPH_CLOSING_WIDTH_METERS,
        SEED_MIN_DISTANCE_FROM_WALL_METERS
    )
    
    print("步骤 2/4: 过滤无效的小区域...")
    valid_rooms_info = []
    # --- **核心修改点**: 直接遍历 markers 矩阵中的唯一ID ---
    # np.unique会返回所有整数ID，包括-1(边界), 0(未定义), 1(背景)
    room_ids = np.unique(markers)
    
    for room_id in room_ids:
        # 跳过非房间的特殊标签
        if room_id == -1 or room_id == 0 or room_id == 1:
            continue
            
        # 直接根据整数ID创建掩码，这是最准确的方式
        room_mask = np.zeros(markers.shape, dtype=np.uint8)
        room_mask[markers == room_id] = 255
        
        area = cv2.countNonZero(room_mask)
        
        if area > MIN_ROOM_AREA_PIXELS:
            contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                valid_rooms_info.append({'mask': room_mask, 'contour': main_contour})

    print(f"步骤 3/4: 找到 {len(valid_rooms_info)} 个有效房间。正在进行标注...")
    
    # ... (后续的绘图和标注逻辑完全不变) ...
    for i, room_info in enumerate(valid_rooms_info):
        color = HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]
        overlay[room_info['mask'] == 255] = color
        cv2.drawContours(annotated_image, [room_info['contour']], -1, BOUNDARY_COLOR, BOUNDARY_THICKNESS)
    
    final_image = cv2.addWeighted(overlay, ANNOTATION_ALPHA, annotated_image, 1.0 - ANNOTATION_ALPHA, 0)

    for i, room_info in enumerate(valid_rooms_info):
        room_number = i + 1
        center_point = find_robust_center(room_info['mask'], room_info['contour'])
        cX, cY = center_point
        text = str(room_number)
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        box_padding = 5
        box_x1 = cX - text_width // 2 - box_padding
        box_y1 = cY - text_height // 2 - box_padding - baseline
        box_x2 = cX + text_width // 2 + box_padding
        box_y2 = cY + text_height // 2 + box_padding
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2
        box_x1, box_y1 = max(0, box_x1), max(0, box_y1)
        box_x2, box_y2 = min(final_image.shape[1], box_x2), min(final_image.shape[0], box_y2)
        cv2.rectangle(final_image, (box_x1, box_y1), (box_x2, box_y2), FONT_BACKGROUND_COLOR, -1)
        cv2.putText(final_image, text, (text_x, text_y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    print("步骤 4/4: 标注完成！")
    return final_image

def main():
    # ... (main 函数完全不变) ...
    print("--- 开始房间标注流程 (V5 - Bug Squasher) ---")
    try:
        final_image = annotate_rooms_on_image(ORIGINAL_IMAGE_PATH, MASK_IMAGE_PATH)
        cv2.imwrite(ANNOTATED_IMAGE_PATH, final_image)
        print(f"成功！已将标注结果保存至: '{ANNOTATED_IMAGE_PATH}'")
        cv2.imshow("Annotated Room View (V5)", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()