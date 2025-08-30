"""
Step 3: Room segmentation and annotation.
Uses the wall mask to segment rooms and create room annotations with bounding boxes.
Adapted from the object-goal navigation version.
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
    
    Args:
        mask_image_path: Path to the input mask image
        spacing: Meters per pixel resolution
        closing_width_m: Physical width for morphological closing in meters
        seed_dist_m: Minimum distance from wall for room seeds in meters
    
    Returns:
        Tuple of (segmented_image, markers)
    """
    # Convert physical parameters to pixel units
    kernel_size_px = int(closing_width_m / spacing)
    if kernel_size_px % 2 == 0: 
        kernel_size_px += 1
    
    print(f"🔧 Parameter conversion:")
    print(f"  - Morphological kernel: {closing_width_m}m -> {kernel_size_px} pixels")
    
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
    
    # Create seed regions using Otsu thresholding
    dist_normalized = dist_transform.copy()
    cv2.normalize(dist_normalized, dist_normalized, 0, 255, cv2.NORM_MINMAX)
    dist_normalized = np.uint8(dist_normalized)
    
    # Apply Gaussian blur and Otsu thresholding
    blur = cv2.GaussianBlur(dist_normalized, (85, 85), 0)
    _, sure_fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    
    # Create colored segmentation result
    num_rooms = np.max(markers)
    colors = [np.random.randint(50, 256, 3).tolist() for _ in range(num_rooms + 1)]
    segmented_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    
    for i in range(1, num_rooms + 1):
        segmented_image[markers == i] = colors[i]
    
    # Mark boundaries in red
    segmented_image[markers == -1] = [0, 0, 255]
    
    # Restore original walls (black)
    segmented_image[binary_mask == 0] = [0, 0, 0]
    
    return segmented_image, markers

def create_room_annotation(original_image: np.ndarray, markers: np.ndarray, 
                         min_room_area_pixels: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create room annotation with bounding boxes and labels.
    
    Args:
        original_image: Original topdown image
        markers: Watershed markers
        min_room_area_pixels: Minimum room area in pixels
        
    Returns:
        Tuple of (annotated_image, room_data)
    """
    annotated_image = original_image.copy()
    room_annotations = []
    
    # Get unique room IDs (excluding background and boundaries)
    unique_rooms = np.unique(markers)
    unique_rooms = unique_rooms[(unique_rooms > 0) & (unique_rooms != -1)]
    
    valid_room_id = 1
    for room_id in unique_rooms:
        # Get room mask
        room_mask = (markers == room_id).astype(np.uint8)
        room_area = np.sum(room_mask)
        
        if room_area < min_room_area_pixels:
            continue
        
        # Find contours and bounding box
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add room number label
            label_pos = (x + 10, y + 30)
            cv2.putText(annotated_image, str(valid_room_id), label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            room_info = {
                "room_id": valid_room_id,
                "original_marker_id": int(room_id),
                "bounding_box": {"x_min": int(x), "y_min": int(y), "x_max": int(x + w), "y_max": int(y + h)},
                "center": {"x": int(center_x), "y": int(center_y)},
                "area_pixels": int(room_area)
            }
            room_annotations.append(room_info)
            valid_room_id += 1
    
    room_data = {
        "total_rooms": len(room_annotations),
        "room_annotations": room_annotations,
        "min_room_area_pixels": min_room_area_pixels
    }
    
    return annotated_image, room_data

def perform_room_segmentation(topdown_path: str, wall_mask_path: str, metadata_path: str, 
                            config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Perform room segmentation and create annotations.
    
    Args:
        topdown_path: Path to topdown view image
        wall_mask_path: Path to wall mask image
        metadata_path: Path to metadata JSON file
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Loading files for room segmentation:")
    print(f"  - Topdown: {topdown_path}")
    print(f"  - Wall mask: {wall_mask_path}")
    print(f"  - Metadata: {metadata_path}")
    
    # Load images and metadata
    topdown_image = cv2.imread(topdown_path)
    wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    if wall_mask is None:
        raise FileNotFoundError(f"Wall mask not found: {wall_mask_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Get parameters
    spacing = metadata['topdown_metadata']['spacing_in_meters_per_pixel']
    seg_config = config['room_segmentation']
    
    closing_width_m = seg_config['closing_width_m']
    min_room_area_m2 = seg_config['min_room_area_m2']
    
    print(f"🔧 Segmentation parameters:")
    print(f"  - Spacing: {spacing:.6f} m/pixel")
    print(f"  - Closing width: {closing_width_m} m")
    print(f"  - Min room area: {min_room_area_m2} m²")
    
    # Convert minimum area to pixels
    min_room_area_pixels = int(min_room_area_m2 / (spacing * spacing))
    print(f"  - Min room area in pixels: {min_room_area_pixels}")
    
    # Perform room segmentation
    segmented_image, markers = segment_rooms_physical(
        wall_mask_path, spacing, closing_width_m
    )
    
    # Save segmented image
    segmented_path = os.path.join(output_dir, "room_segmentation.png")
    cv2.imwrite(segmented_path, segmented_image)
    print(f"✓ Room segmentation saved to: {segmented_path}")
    
    # Create room annotation
    annotated_image, room_data = create_room_annotation(
        topdown_image, markers, min_room_area_pixels
    )
    
    # Save room annotation
    annotation_path = os.path.join(output_dir, "room_annotation.png")
    cv2.imwrite(annotation_path, annotated_image)
    print(f"✓ Room annotation saved to: {annotation_path}")
    
    # Save room segmentation results
    results_path = os.path.join(output_dir, "room_segmentation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(room_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Room segmentation results saved to: {results_path}")
    
    print(f"📊 Room segmentation completed:")
    print(f"  - Total rooms found: {room_data['total_rooms']}")
    
    return {
        "generated_files": {
            "room_segmentation": segmented_path,
            "room_annotation": annotation_path,
            "room_results": results_path
        },
        "results": {
            "total_rooms": room_data['total_rooms'],
            "min_room_area_pixels": min_room_area_pixels,
            "segmentation_success": True
        }
    }

if __name__ == "__main__":
    # Test function
    import sys
    if len(sys.argv) != 5:
        print("Usage: python step_3_room_segmentation.py <topdown_path> <wall_mask_path> <metadata_path> <config_json>")
        sys.exit(1)
    
    topdown_path = sys.argv[1]
    wall_mask_path = sys.argv[2]
    metadata_path = sys.argv[3]
    config_path = sys.argv[4]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    result = perform_room_segmentation(topdown_path, wall_mask_path, metadata_path, config, output_dir)
    print("Step 3 completed:", result)
