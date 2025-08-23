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
        blur = cv2.GaussianBlur(dist_normalized, (85, 85), 0)  # 使用对称核
        
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
    
    # Create colored segmentation result
    num_rooms = np.max(markers)
    colors = [np.random.randint(50, 256, 3).tolist() for _ in range(num_rooms + 1)]
    segmented_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    
    for i in range(1, num_rooms + 1):
        if i < len(colors):
            segmented_image[markers == i] = colors[i]
    
    # Mark boundaries in red
    segmented_image[markers == -1] = [0, 0, 255]
    
    # Restore original walls (black)
    segmented_image[binary_mask == 0] = [0, 0, 0]
    
    return segmented_image, markers


def calculate_boundary_length(room1_id: int, room2_id: int, markers: np.ndarray) -> int:
    """
    Calculate the length of shared boundary between two rooms using watershed boundaries.
    Optimized vectorized version without loops.
    
    Args:
        room1_id: ID of the first room
        room2_id: ID of the second room  
        markers: Watershed markers array
        
    Returns:
        Length of shared boundary in pixels
    """
    # Get boundary pixels (marked as -1 in watershed)
    boundary_mask = (markers == -1)
    if not np.any(boundary_mask):
        return 0
    
    # Create padded version of markers for neighbor checking
    padded_markers = np.pad(markers, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    # For each boundary pixel, check if it's between room1 and room2
    # We'll use 8-connectivity to check neighbors
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    shared_boundary_count = 0
    boundary_coords = np.where(boundary_mask)
    
    if len(boundary_coords[0]) == 0:
        return 0
    
    # Vectorized neighbor checking
    for dy, dx in neighbor_offsets:
        # Get neighbor positions (accounting for padding offset)
        neighbor_y = boundary_coords[0] + 1 + dy  # +1 for padding offset
        neighbor_x = boundary_coords[1] + 1 + dx
        
        # Get neighbor values
        neighbor_values = padded_markers[neighbor_y, neighbor_x]
        
        # Check if neighbors are room1_id
        room1_neighbors = (neighbor_values == room1_id)
        # Check if neighbors are room2_id  
        room2_neighbors = (neighbor_values == room2_id)
        
        # For boundary pixels that have both room1 and room2 as neighbors
        # We need to check all 8 neighbors for each boundary pixel
        if dy == -1 and dx == -1:  # First iteration, initialize arrays
            has_room1 = room1_neighbors.copy()
            has_room2 = room2_neighbors.copy()
        else:
            has_room1 |= room1_neighbors  # Accumulate room1 neighbors
            has_room2 |= room2_neighbors  # Accumulate room2 neighbors
    
    # Count boundary pixels that have both room1 and room2 as neighbors
    shared_boundary_count = np.sum(has_room1 & has_room2)
    return int(shared_boundary_count)

def calculate_room_centroid(room_id: int, markers: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid of a room.
    
    Args:
        room_id: The ID of the room
        markers: Watershed markers array
        
    Returns:
        Tuple of (centroid_x, centroid_y)
    """
    room_mask = (markers == room_id)
    y_coords, x_coords = np.where(room_mask)
    
    if len(x_coords) == 0:
        return (0, 0)
    
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    return (centroid_x, centroid_y)

def calculate_merge_score(small_room_id: int, candidate_room_id: int, markers: np.ndarray, 
                         adjacent_candidates: List[int], min_room_area_pixels: int) -> float:
    """
    Calculate merge score based on area (33%), centroid distance (33%), and boundary length (33%).
    Normalization only considers adjacent mergeable rooms.
    Higher score means better merge candidate.
    
    Args:
        small_room_id: ID of the small room to be merged
        candidate_room_id: ID of the candidate room for merging
        markers: Watershed markers array
        adjacent_candidates: List of all adjacent rooms that can be merged with
        min_room_area_pixels: Minimum area threshold for rooms
        
    Returns:
        Merge score (higher is better)
    """
    # Calculate areas
    small_area = np.sum(markers == small_room_id)
    candidate_area = np.sum(markers == candidate_room_id)
    
    # Calculate centroids
    small_centroid = calculate_room_centroid(small_room_id, markers)
    candidate_centroid = calculate_room_centroid(candidate_room_id, markers)
    
    # Calculate distance between centroids
    distance = np.sqrt((small_centroid[0] - candidate_centroid[0])**2 + 
                      (small_centroid[1] - candidate_centroid[1])**2)
    
    # Calculate boundary length
    boundary_length = calculate_boundary_length(small_room_id, candidate_room_id, markers)
    
    # Normalize scores using only adjacent mergeable rooms
    mergeable_candidates = [rid for rid in adjacent_candidates 
                           if np.sum(markers == rid) >= min_room_area_pixels]
    
    if len(mergeable_candidates) == 0:
        return 0  # No valid candidates
    
    # Area score: larger areas get higher scores (normalized by max adjacent area)
    adjacent_areas = [np.sum(markers == rid) for rid in mergeable_candidates]
    max_adjacent_area = max(adjacent_areas)
    area_score = candidate_area / max_adjacent_area if max_adjacent_area > 0 else 0
    
    # Distance score: closer rooms get higher scores (normalized by max adjacent distance)
    adjacent_distances = []
    for rid in mergeable_candidates:
        adj_centroid = calculate_room_centroid(rid, markers)
        adj_distance = np.sqrt((small_centroid[0] - adj_centroid[0])**2 + 
                              (small_centroid[1] - adj_centroid[1])**2)
        adjacent_distances.append(adj_distance)
    
    max_adjacent_distance = max(adjacent_distances) if adjacent_distances else 1
    distance_score = 1.0 - (distance / max_adjacent_distance) if max_adjacent_distance > 0 else 0
    
    # Boundary score: longer boundaries get higher scores (normalized by max adjacent boundary)
    adjacent_boundaries = [calculate_boundary_length(small_room_id, rid, markers) 
                          for rid in mergeable_candidates]
    max_adjacent_boundary = max(adjacent_boundaries) if adjacent_boundaries else 1
    boundary_score = boundary_length / max_adjacent_boundary if max_adjacent_boundary > 0 else 0
    
    # Weighted combination (33% each: area, distance, boundary)
    merge_score = 0.0 * area_score + 0.3 * distance_score + 0.7 * boundary_score
    
    return merge_score

def merge_small_rooms(markers: np.ndarray, min_room_area_pixels: int, spacing: float) -> np.ndarray:
    """
    Iteratively merge small rooms into adjacent larger rooms.
    After each merge, recalculate boundaries to ensure correct adjacency detection.
    
    Args:
        markers: Initial watershed markers
        min_room_area_pixels: Minimum area threshold for rooms
        spacing: Meters per pixel resolution for boundary length filtering
        
    Returns:
        Updated markers with merged rooms
    """
    merged_markers = markers.copy()
    merge_log = []
    
    while True:
        # Get all current room IDs and their areas
        room_ids = np.unique(merged_markers)
        room_ids = room_ids[room_ids > 1]  # Filter out boundaries and background
        
        if len(room_ids) == 0:
            break
            
        # Find small rooms
        small_rooms = []
        for room_id in room_ids:
            area = np.sum(merged_markers == room_id)
            if area < min_room_area_pixels:
                small_rooms.append((room_id, area))
        
        if len(small_rooms) == 0:
            break  # No more small rooms to merge
        
        # Sort small rooms by area (smallest first)
        small_rooms.sort(key=lambda x: x[1])
        
        merged_in_this_iteration = False
        
        for small_room_id, small_area in small_rooms:
            # Check if this room still exists (might have been merged already)
            if np.sum(merged_markers == small_room_id) == 0:
                continue
                
            # Find adjacent rooms using improved method that handles boundaries correctly
            # Pass spacing for boundary length filtering (0.3m minimum)
            adjacent_rooms = find_adjacent_rooms(small_room_id, merged_markers, spacing, 0.3)
            
            if not adjacent_rooms:
                print(f"  ⚠️ Small room {small_room_id} ({small_area} pixels) has no adjacent rooms, keeping as is")
                continue
            
            # Calculate merge scores for all adjacent rooms
            best_candidate = None
            best_score = -1
            
            # Filter adjacent rooms to find those that can be merged with (meet size requirement)
            mergeable_adjacent = [adj_id for adj_id in adjacent_rooms 
                                if np.sum(merged_markers == adj_id) >= min_room_area_pixels]
            
            for adj_room_id in mergeable_adjacent:
                adj_area = np.sum(merged_markers == adj_room_id)
                # Pass adjacent candidates and min area for proper normalization
                score = calculate_merge_score(small_room_id, adj_room_id, merged_markers, 
                                            adjacent_rooms, min_room_area_pixels)
                if score > best_score:
                    best_score = score
                    best_candidate = adj_room_id
            
            if best_candidate is not None:
                # Perform the merge
                candidate_area = np.sum(merged_markers == best_candidate)
                
                # CRITICAL FIX: Update boundaries after merging
                merged_markers = perform_room_merge_with_boundary_update(
                    merged_markers, small_room_id, best_candidate
                )
                
                merge_log.append({
                    'small_room': small_room_id,
                    'small_area': small_area,
                    'target_room': best_candidate,
                    'target_area': candidate_area,
                    'merge_score': best_score
                })
                
                print(f"  ✓ Merged room {small_room_id} ({small_area} pixels) into room {best_candidate} ({candidate_area} pixels), score: {best_score:.3f}")
                merged_in_this_iteration = True
                
                # Break after each merge to recalculate everything fresh
                break
            else:
                print(f"  ⚠️ Small room {small_room_id} ({small_area} pixels) has no suitable merge candidates")
        
        if not merged_in_this_iteration:
            break  # No merges performed in this iteration
    
    print(f"✓ Completed room merging: {len(merge_log)} merges performed")
    return merged_markers

def find_adjacent_rooms(room_id: int, markers: np.ndarray, spacing: float = None, min_boundary_length_m: float = 0.3) -> List[int]:
    """
    Find all rooms that are adjacent to the given room, handling boundaries correctly.
    Only considers rooms adjacent if their shared boundary length >= min_boundary_length_m.
    Optimized vectorized version.
    
    Args:
        room_id: The ID of the room to find neighbors for
        markers: Watershed markers array
        spacing: Meters per pixel resolution (optional, for boundary filtering)
        min_boundary_length_m: Minimum boundary length in meters to consider rooms adjacent
        
    Returns:
        List of adjacent room IDs with sufficient boundary length
    """
    # Create mask for the current room
    room_mask = (markers == room_id).astype(np.uint8)
    
    # Use morphological dilation to find adjacent regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  
    dilated = cv2.dilate(room_mask, kernel, iterations=2)
    
    # Find what rooms the dilated area touches
    adjacent_region_mask = (dilated - room_mask).astype(bool)
    
    # Get unique room IDs in the adjacent region
    adjacent_room_ids = np.unique(markers[adjacent_region_mask])
    
    # Filter out boundaries (-1), background (0, 1), and self
    candidate_rooms = [int(adj_id) for adj_id in adjacent_room_ids 
                      if adj_id > 1 and adj_id != room_id]
    
    # If no spacing provided, return all candidates (backward compatibility)
    if spacing is None:
        return candidate_rooms
    
    # Filter by boundary length threshold
    valid_adjacent_rooms = []
    
    for candidate_id in candidate_rooms:
        boundary_length_pixels = calculate_boundary_length(room_id, candidate_id, markers)
        boundary_length_m = boundary_length_pixels * spacing
        
        if boundary_length_m >= min_boundary_length_m:
            valid_adjacent_rooms.append(candidate_id)
        else:
            print(f"    - Filtering out room {candidate_id}: boundary length {boundary_length_m:.3f}m < {min_boundary_length_m}m threshold")
    
    return valid_adjacent_rooms

def perform_room_merge_with_boundary_update(markers: np.ndarray, source_room_id: int, target_room_id: int) -> np.ndarray:
    """
    Merge source room into target room and update boundaries appropriately.
    Optimized vectorized version.
    
    Args:
        markers: Current markers array
        source_room_id: Room to be merged (will disappear)
        target_room_id: Target room to merge into
        
    Returns:
        Updated markers with merged rooms and corrected boundaries
    """
    updated_markers = markers.copy()
    
    # Step 1: Merge the rooms (change all source_room_id pixels to target_room_id)
    updated_markers[markers == source_room_id] = target_room_id
    
    # Step 2: Update boundaries between the merged rooms using vectorized operations
    boundary_mask = (markers == -1)
    
    if not np.any(boundary_mask):
        return updated_markers
    
    # Create padded markers for efficient neighbor checking
    padded_markers = np.pad(updated_markers, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    # Get boundary pixel coordinates
    boundary_coords = np.where(boundary_mask)
    
    if len(boundary_coords[0]) == 0:
        return updated_markers
    
    # For each boundary pixel, check if all neighbors are the same room
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Collect all neighbor values for each boundary pixel
    all_neighbors = []
    for dy, dx in neighbor_offsets:
        # Account for padding offset (+1)
        neighbor_y = boundary_coords[0] + 1 + dy
        neighbor_x = boundary_coords[1] + 1 + dx
        neighbor_vals = padded_markers[neighbor_y, neighbor_x]
        # Only consider valid rooms (> 1)
        neighbor_vals = np.where(neighbor_vals > 1, neighbor_vals, 0)
        all_neighbors.append(neighbor_vals)
    
    # Stack neighbors for each boundary pixel (shape: [8, num_boundary_pixels])
    all_neighbors = np.stack(all_neighbors, axis=0)
    
    # For each boundary pixel, find unique non-zero neighbors
    for i, (y, x) in enumerate(zip(boundary_coords[0], boundary_coords[1])):
        neighbors = all_neighbors[:, i]
        unique_neighbors = np.unique(neighbors[neighbors > 0])
        
        # If all neighboring rooms are the same, convert boundary to that room
        if len(unique_neighbors) == 1 and np.sum(neighbors > 0) >= 2:
            updated_markers[y, x] = unique_neighbors[0]
    
    return updated_markers

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
    Perform room segmentation with small room merging and create annotated image.
    
    Flow:
    1. Apply EDF + Watershed algorithm to get initial room segmentation
    2. Merge small rooms iteratively into adjacent rooms
    3. Generate room_segmentation.png with merged results
    4. Generate room_annotation.png with boundaries and numbering
    
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
    seed_min_distance_from_wall_meters = room_config.get('seed_min_distance_from_wall_meters', None)
    min_room_area_pixels = room_config.get('min_room_area_pixels', 1000)
    
    print(f"🔧 Room segmentation parameters:")
    print(f"  - Morphological closing: {morph_closing_width_meters}m")
    print(f"  - Seed distance from walls: {'Dynamic (Otsu)' if seed_min_distance_from_wall_meters is None else f'{seed_min_distance_from_wall_meters}m'}")
    print(f"  - Minimum room area: {min_room_area_pixels} pixels")
    
    print(f"\n🔄 Step 1: Initial room segmentation (EDF + Watershed)")
    # Perform initial room segmentation
    segmented_image, initial_markers = segment_rooms_physical(
        wall_mask_path,
        spacing_in_meters_per_pixel,
        morph_closing_width_meters,
        seed_min_distance_from_wall_meters  
    )
    
    # Count initial rooms
    initial_room_ids = np.unique(initial_markers)
    initial_room_ids = initial_room_ids[initial_room_ids > 1]
    print(f"  ✓ Initial segmentation: {len(initial_room_ids)} rooms detected")
    
    print(f"\n🔄 Step 2: Merging small rooms")
    # Merge small rooms with automatic boundary updates
    final_markers = merge_small_rooms(initial_markers, min_room_area_pixels, spacing_in_meters_per_pixel)
    
    # Count final rooms
    final_room_ids = np.unique(final_markers)
    final_room_ids = final_room_ids[final_room_ids > 1]
    print(f"  ✓ After merging: {len(final_room_ids)} rooms remaining")
    
    print(f"\n🔄 Step 3: Generating room_segmentation.png")
    # Create final segmentation image with merged results
    final_segmented_image = create_segmentation_visualization(final_markers, original_image)
    
    # Save segmentation result
    segmentation_path = os.path.join(output_dir, "room_segmentation.png")
    cv2.imwrite(segmentation_path, final_segmented_image)
    print(f"  ✓ Room segmentation saved to: {segmentation_path}")
    
    print(f"\n🔄 Step 4: Generating room_annotation.png with boundaries and numbering")
    # Create room annotation with boundaries and numbers
    annotated_image, room_bounding_boxes = create_room_annotation(
        original_image, final_markers, min_room_area_pixels
    )
    
    # Save room annotation image
    annotation_path = os.path.join(output_dir, "room_annotation.png")
    cv2.imwrite(annotation_path, annotated_image)
    print(f"  ✓ Room annotation saved to: {annotation_path}")
    
    return {
        "generated_files": {
            "room_annotation": annotation_path,
            "room_segmentation": segmentation_path
        },
        "results": {
            "num_initial_rooms": len(initial_room_ids),
            "num_final_rooms": len(final_room_ids),
            "room_bounding_boxes": room_bounding_boxes,
            "segmentation_parameters": {
                "spacing_in_meters_per_pixel": spacing_in_meters_per_pixel,
                "morph_closing_width_meters": morph_closing_width_meters,
                "seed_distance_threshold": "dynamic_otsu" if seed_min_distance_from_wall_meters is None else f"{seed_min_distance_from_wall_meters}m",
                "min_room_area_pixels": min_room_area_pixels
            }
        }
    }

def clean_merged_boundaries(markers: np.ndarray) -> np.ndarray:
    """
    Clean up boundaries after room merging by removing internal boundaries.
    Optimized vectorized version.
    
    Args:
        markers: Watershed markers after merging
        
    Returns:
        Cleaned markers with updated boundaries
    """
    cleaned_markers = markers.copy()
    boundary_mask = (markers == -1)
    
    if not np.any(boundary_mask):
        return cleaned_markers
    
    # Create padded markers for efficient neighbor checking  
    padded_markers = np.pad(markers, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    # Get boundary coordinates
    boundary_coords = np.where(boundary_mask)
    
    # For each boundary pixel, collect all valid room neighbors
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Vectorized neighbor collection
    all_neighbors = []
    for dy, dx in neighbor_offsets:
        neighbor_y = boundary_coords[0] + 1 + dy  # +1 for padding
        neighbor_x = boundary_coords[1] + 1 + dx
        neighbor_vals = padded_markers[neighbor_y, neighbor_x]
        # Only keep valid room IDs (> 1)
        neighbor_vals = np.where(neighbor_vals > 1, neighbor_vals, 0)
        all_neighbors.append(neighbor_vals)
    
    # Stack neighbors (shape: [8, num_boundary_pixels])
    all_neighbors = np.stack(all_neighbors, axis=0)
    
    # Process each boundary pixel
    for i, (y, x) in enumerate(zip(boundary_coords[0], boundary_coords[1])):
        neighbors = all_neighbors[:, i]
        valid_neighbors = neighbors[neighbors > 0]
        
        if len(valid_neighbors) > 0:
            unique_neighbors = np.unique(valid_neighbors)
            # If all neighbors belong to the same room, convert boundary to that room
            if len(unique_neighbors) == 1:
                cleaned_markers[y, x] = unique_neighbors[0]
    
    return cleaned_markers
    
def create_segmentation_visualization(markers: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Create colored segmentation visualization from markers.
    
    Args:
        markers: Watershed markers after merging
        original_image: Original topdown image for reference
        
    Returns:
        Colored segmentation image
    """
    # Get unique room IDs
    room_ids = np.unique(markers)
    room_ids = room_ids[room_ids > 1]  # Filter out boundaries and background
    
    # Generate random colors for each room
    colors = {}
    for room_id in room_ids:
        colors[room_id] = np.random.randint(50, 256, 3).tolist()
    
    # Create segmented image
    segmented_image = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    
    for room_id in room_ids:
        segmented_image[markers == room_id] = colors[room_id]
    
    # Mark remaining boundaries in red (only valid boundaries should remain after cleaning)
    segmented_image[markers == -1] = [0, 0, 255]
    
    # Restore original walls (black)
    wall_mask = (markers == 0) | (markers == 1)
    segmented_image[wall_mask] = [0, 0, 0]
    
    return segmented_image

def create_room_annotation(original_image: np.ndarray, markers: np.ndarray, 
                         min_room_area_pixels: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create room annotation with boundaries and numbering.
    
    Args:
        original_image: Original topdown image
        markers: Final watershed markers
        min_room_area_pixels: Minimum room area threshold
        
    Returns:
        Tuple of (annotated_image, room_bounding_boxes)
    """
    # Start with original image
    final_image = original_image.copy()
    
    # Get valid room information
    valid_rooms_info = []
    room_bounding_boxes = {}
    
    room_ids = np.unique(markers)
    room_ids = room_ids[room_ids > 1]  # Filter out boundaries and background
    room_counter = 0
    
    for room_id in room_ids:
        # Create mask for this room
        room_mask = (markers == room_id).astype(np.uint8) * 255
        area = cv2.countNonZero(room_mask)
        
        if area >= min_room_area_pixels:
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
    
    # Draw thick boundaries between rooms - use cleaned markers
    for room_info in valid_rooms_info:
        room_mask = room_info['mask']
        
        # Find contours for room boundaries
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Draw thick white boundaries
            #pass
            cv2.drawContours(final_image, contours, -1, (255, 255, 255), thickness=2)
    
    # Draw boundaries from cleaned watershed markers
    #boundary_mask = (markers == -1).astype(np.uint8) * 255
    # Dilate the boundaries to make them thicker
    #kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #thick_boundary = cv2.dilate(boundary_mask, kernel_boundary, iterations=2)
    #final_image[thick_boundary == 255] = [255, 255, 255]  # White boundaries
    
    # Add room numbers - smaller and less obtrusive
    for room_info in valid_rooms_info:
        room_number = room_info['room_number']
        center_point = find_robust_center(room_info['mask'], room_info['contour'])
        cX, cY = center_point
        
        # Draw compact text styling
        text = str(room_number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Smaller font
        font_thickness = 2  # Thinner text
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Compact background rectangle
        padding = 4  # Reduced padding
        bg_x1 = cX - text_width // 2 - padding
        bg_y1 = cY - text_height // 2 - padding - baseline
        bg_x2 = cX + text_width // 2 + padding
        bg_y2 = cY + text_height // 2 + padding
        
        # Draw compact black background with thin white border
        cv2.rectangle(final_image, (bg_x1-1, bg_y1-1), (bg_x2+1, bg_y2+1), (255, 255, 255), -1)  # Thin white border
        cv2.rectangle(final_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # Black background
        
        # White text
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2
        cv2.putText(final_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    
    return final_image, room_bounding_boxes

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
