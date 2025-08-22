"""
Step 2: Room segmentation and annotation.
Uses the wall mask to segment rooms and create room annotations with bounding boxes.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.cluster import DBSCAN

def segment_rooms_physical(mask_image_path: str, spacing: float, closing_width_m: float, seed_dist_m: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments rooms using physics-based parameters.
    Modified version of the original function to return both segmented image and markers.
    
    Args:
        mask_image_path: Path to the input mask image
        spacing: Meters per pixel resolution
        closing_width_m: Physical width for morphological closing in meters
        seed_dist_m: Minimum distance from wall for room seeds in meters. 
                    If None, uses Otsu's automatic thresholding.
    
    Returns:
        Tuple of (segmented_image, markers)
    """
    # Convert physical parameters to pixel units
    kernel_size_px = int(closing_width_m / spacing)
    if kernel_size_px % 2 == 0: 
        kernel_size_px += 1
    
    print(f"🔧 Parameter conversion:")
    print(f"  - Morphological kernel: {closing_width_m}m -> {kernel_size_px} pixels")
    
    if seed_dist_m is not None:
        # 使用固定的种子距离阈值
        seed_dist_px = seed_dist_m / spacing
        print(f"  - Seed distance threshold: {seed_dist_m}m -> {seed_dist_px:.2f} pixels (固定)")
    else:
        print("  - 将使用 Otsu 自动阈值化动态确定种子距离阈值")
    
    # Load and process the mask
    img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask image not found: {mask_image_path}")
    
    # Binarize the mask (255 for walkable, 0 for walls)
    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_px, kernel_size_px))
    closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Compute distance transform
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    
    # Create seed regions (sure foreground) with dynamic or fixed thresholding
    if seed_dist_m is not None:
        # 使用固定阈值
        seed_dist_px = seed_dist_m / spacing
        _, sure_fg = cv2.threshold(dist_transform, seed_dist_px, 255, cv2.THRESH_BINARY)
        print(f"  ✓ 使用固定阈值: {seed_dist_px:.2f} pixels ({seed_dist_m} m)")
    else:
        # 使用 Otsu 自动阈值化 (参考 distance_transform 函数)
        print(f"  📊 距离变换范围: {np.min(dist_transform):.2f} 到 {np.max(dist_transform):.2f} pixels")
        
        # 归一化距离变换用于 Otsu 阈值化
        dist_normalized = dist_transform.copy()
        cv2.normalize(dist_normalized, dist_normalized, 0, 255, cv2.NORM_MINMAX)
        dist_normalized = np.uint8(dist_normalized)
        
        # 应用高斯模糊后进行 Otsu 阈值化 (参考原函数)
        blur = cv2.GaussianBlur(dist_normalized, (21, 21), 0)  # 使用对称核
        
        # Otsu 阈值化
        otsu_threshold, sure_fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 将阈值转换回原始距离变换的尺度
        actual_threshold_px = (otsu_threshold / 255.0) * np.max(dist_transform)
        actual_threshold_m = actual_threshold_px * spacing
        
        print(f"  ✓ Otsu 自动阈值: {otsu_threshold} (归一化)")
        print(f"  ✓ 实际距离阈值: {actual_threshold_px:.2f} pixels ({actual_threshold_m:.3f} m)")
    
    sure_fg = np.uint8(sure_fg)
    
    # Define sure background and unknown regions
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Create markers for watershed algorithm
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm
    img_color_for_watershed = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color_for_watershed, markers)
    
    # Create segmentation result (color filling disabled)
    num_rooms = np.max(markers)
    # colors = [np.random.randint(50, 256, 3).tolist() for _ in range(num_rooms + 1)]
    segmented_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    
    # Color filling disabled - keep segmented_image as black background
    # for i in range(1, num_rooms + 1):
    #     if i < len(colors):
    #         segmented_image[markers == i] = colors[i]
    
    # Mark boundaries in red (keeping this for visualization)
    segmented_image[markers == -1] = [0, 0, 255]
    
    # Show walkable areas in white instead of original walls
    segmented_image[binary_mask == 255] = [255, 255, 255]  # White for walkable areas
    # Walls remain black (0, 0, 0)
    
    return segmented_image, markers

def find_robust_center(mask: np.ndarray, contour: np.ndarray) -> Tuple[int, int]:
    """Find a robust center point for a room."""
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Check if geometric center is inside the contour
    dist = cv2.pointPolygonTest(contour, (center_x, center_y), False)
    if dist > 0:
        return (center_x, center_y)
    else:
        # Use distance transform to find the most central point
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        return max_loc

def perform_room_segmentation(topdown_path: str, wall_mask_path: str, metadata_path: str, 
                            config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Perform room segmentation and create annotated image with bounding boxes.
    
    Args:
        topdown_path: Path to topdown view image
        wall_mask_path: Path to wall mask image
        metadata_path: Path to metadata JSON file
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading images:")
    print(f"  - Topdown: {topdown_path}")
    print(f"  - Wall mask: {wall_mask_path}")
    print(f"  - Metadata: {metadata_path}")
    
    # Load metadata to get spacing
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    spacing_in_meters_per_pixel = metadata['topdown_metadata']['spacing_in_meters_per_pixel']
    print(f"📏 Pixel spacing: {spacing_in_meters_per_pixel:.6f} m/pixel")
    
    # Load original topdown image
    original_image = cv2.imread(topdown_path)
    if original_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    
    # Get room segmentation parameters
    room_config = config['room_segmentation']
    morph_closing_width_meters = room_config.get('morph_closing_width_meters', 0.01)
    # 不再从配置读取 seed_min_distance_from_wall_meters，使用动态阈值
    seed_min_distance_from_wall_meters = room_config.get('seed_min_distance_from_wall_meters', None)
    min_room_area_pixels = room_config.get('min_room_area_pixels', 1000)
    
    print(f"🔧 Room segmentation parameters:")
    print(f"  - Morphological closing: {morph_closing_width_meters}m")
    print(f"  - Seed distance from walls: 动态阈值 (Otsu 自动)")
    print(f"  - Minimum room area: {min_room_area_pixels} pixels")
    
    # Perform room segmentation
    segmented_image, markers = segment_rooms_physical(
        wall_mask_path,
        spacing_in_meters_per_pixel,
        morph_closing_width_meters,
        seed_min_distance_from_wall_meters  
    )
    
    # Filter and annotate valid rooms
    annotated_image = original_image.copy()
    overlay = original_image.copy()
    
    # High contrast colors for room annotation
    colors = [
        (60, 20, 220), (255, 128, 0), (40, 180, 0), (0, 210, 255),
        (230, 0, 180), (255, 128, 128), (0, 128, 255), (128, 0, 128),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 0)
    ]
    
    valid_rooms_info = []
    room_bounding_boxes = {}
    
    # Get unique room IDs (skip special markers)
    room_ids = np.unique(markers)
    room_counter = 0
    
    print(f"🔍 Processing {len(room_ids)} potential rooms...")
    
    for room_id in room_ids:
        # Skip non-room markers
        if room_id == -1 or room_id == 0 or room_id == 1:
            continue
        
        # Create mask for this room
        room_mask = np.zeros(markers.shape, dtype=np.uint8)
        room_mask[markers == room_id] = 255
        
        area = cv2.countNonZero(room_mask)
        
        if area > min_room_area_pixels:
            contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(main_contour)
                bounding_box = {
                    "x_min": int(x),
                    "y_min": int(y), 
                    "x_max": int(x + w),
                    "y_max": int(y + h),
                    "width": int(w),
                    "height": int(h),
                    "area_pixels": int(area)
                }
                
                room_counter += 1
                room_number = room_counter
                
                valid_rooms_info.append({
                    'room_id': room_id,
                    'room_number': room_number,
                    'mask': room_mask,
                    'contour': main_contour,
                    'bounding_box': bounding_box
                })
                
                room_bounding_boxes[str(room_number)] = bounding_box
                
                print(f"  ✓ Room {room_number}: {area:,} pixels, bbox=({x},{y},{x+w},{y+h})")
    
    print(f"✓ Found {len(valid_rooms_info)} valid rooms")
    
    # Create room annotation (color filling disabled)
    # for i, room_info in enumerate(valid_rooms_info):
    #     color = colors[i % len(colors)]
    #     room_number = room_info['room_number']
    #     
    #     # Color the room area
    #     overlay[room_info['mask'] == 255] = color
    
    # Blend overlay with original image (disabled - using original image)
    # alpha = 0.4
    # final_image = cv2.addWeighted(overlay, alpha, annotated_image, 1.0 - alpha, 0)
    final_image = annotated_image.copy()  # Use original image without color overlay
    
    # Add room numbers
    for room_info in valid_rooms_info:
        room_number = room_info['room_number']
        center_point = find_robust_center(room_info['mask'], room_info['contour'])
        cX, cY = center_point
        
        # Draw text background
        text = str(room_number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Background rectangle
        padding = 5
        bg_x1 = cX - text_width // 2 - padding
        bg_y1 = cY - text_height // 2 - padding - baseline
        bg_x2 = cX + text_width // 2 + padding
        bg_y2 = cY + text_height // 2 + padding
        
        cv2.rectangle(final_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Text
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2
        cv2.putText(final_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    
    # Save room annotation image
    room_annotation_path = os.path.join(output_dir, "room_annotation.png")
    cv2.imwrite(room_annotation_path, final_image)
    print(f"✓ Room annotation saved to: {room_annotation_path}")
    
    # Save segmentation result
    segmentation_path = os.path.join(output_dir, "room_segmentation.png")
    cv2.imwrite(segmentation_path, segmented_image)
    print(f"✓ Room segmentation saved to: {segmentation_path}")
    
    return {
        "generated_files": {
            "room_annotation": room_annotation_path,
            "room_segmentation": segmentation_path
        },
        "results": {
            "num_rooms": len(valid_rooms_info),
            "room_bounding_boxes": room_bounding_boxes,
            "segmentation_parameters": {
                "spacing_in_meters_per_pixel": spacing_in_meters_per_pixel,
                "morph_closing_width_meters": morph_closing_width_meters,
                "seed_distance_threshold": "dynamic_otsu",  # 表示使用动态阈值
                "min_room_area_pixels": min_room_area_pixels
            }
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python step_2_room_segmentation.py <topdown_path> <wall_mask_path> <metadata_path> <config_path>")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    wall_mask_path = sys.argv[2] 
    metadata_path = sys.argv[3]
    config_path = sys.argv[4]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    result = perform_room_segmentation(topdown_path, wall_mask_path, metadata_path, config, output_dir)
    print("Step 2 completed:", result)
