"""
Multi-point selection utilities for the navigation target selection workflow.
Provides functions for marking selected nodes on images and managing iterative selections.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

def get_marker_color(iteration: int, config: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Get marker color for a specific iteration.
    
    Args:
        iteration: Current iteration number (0-based)
        config: Configuration dictionary
        
    Returns:
        BGR color tuple
    """
    multi_point_config = config.get('multi_point_config', {})
    colors = multi_point_config.get('marker_colors', [
        [0, 0, 255],    # Red
        [255, 0, 0],    # Blue  
        [0, 255, 0],    # Green
        [255, 255, 0],  # Cyan
        [255, 0, 255]   # Magenta
    ])
    
    # Cycle through colors if we have more iterations than colors
    color_bgr = colors[iteration % len(colors)]
    return tuple(color_bgr)

def mark_selected_node(image: np.ndarray, node_result: Dict[str, Any], 
                      iteration: int, config: Dict[str, Any]) -> np.ndarray:
    """
    Mark a selected node on the image with colored markers.
    
    Args:
        image: Input image to mark
        node_result: Node selection result containing pixel coordinates
        iteration: Current iteration number (0-based)  
        config: Configuration dictionary
        
    Returns:
        Image with marked node
    """
    marked_image = image.copy()
    
    # Get configuration
    multi_point_config = config.get('multi_point_config', {})
    marker_size = multi_point_config.get('marker_size', 30)
    marker_thickness = multi_point_config.get('marker_thickness', 4)
    show_numbers = multi_point_config.get('show_iteration_numbers', True)
    
    # Get node pixel coordinates
    selected_node = node_result.get('selected_node', {})
    if 'pixel_coordinates' not in selected_node:
        print(f"⚠️  Warning: No pixel coordinates found in node result for iteration {iteration}")
        return marked_image
    
    node_coords = selected_node['pixel_coordinates']
    x, y = int(node_coords[0]), int(node_coords[1])
    
    # Get color for this iteration
    color = get_marker_color(iteration, config)
    
    print(f"🎨 Marking node at ({x}, {y}) with color {color} for iteration {iteration + 1}")
    
    # Draw cross marker
    cv2.drawMarker(marked_image, (x, y), color, cv2.MARKER_TILTED_CROSS, marker_size, marker_thickness)
    
    # Draw circle around the marker
    cv2.circle(marked_image, (x, y), marker_size - 5, color, marker_thickness - 1)
    
    # Add iteration number if enabled
    if show_numbers:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 3
        text = str(iteration + 1)
        
        # Calculate text position (above the marker)
        text_x = x - 15
        text_y = y - marker_size - 10
        
        # Draw text with black outline for better visibility
        cv2.putText(marked_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(marked_image, text, (text_x, text_y), font, font_scale, color, font_thickness)
    
    return marked_image

def mark_multiple_nodes(image: np.ndarray, selected_results: List[Dict[str, Any]], 
                       config: Dict[str, Any]) -> np.ndarray:
    """
    Mark multiple selected nodes on the image.
    
    Args:
        image: Input image to mark
        selected_results: List of selection results from previous iterations
        config: Configuration dictionary
        
    Returns:
        Image with all marked nodes
    """
    marked_image = image.copy()
    
    for i, result in enumerate(selected_results):
        node_result = result.get('node_result', {})
        marked_image = mark_selected_node(marked_image, node_result, i, config)
    
    return marked_image

def save_marked_images(original_topdown: np.ndarray, original_room_annotation: np.ndarray,
                      selected_results: List[Dict[str, Any]], iteration: int, 
                      config: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """
    Save marked versions of topdown and room annotation images for current iteration.
    
    Args:
        original_topdown: Original topdown view image
        original_room_annotation: Original room annotation image  
        selected_results: List of selection results from previous iterations
        iteration: Current iteration number (0-based)
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Tuple of (marked_topdown_path, marked_room_annotation_path)
    """
    # Mark images with all previous selections
    marked_topdown = mark_multiple_nodes(original_topdown, selected_results, config)
    marked_room_annotation = mark_multiple_nodes(original_room_annotation, selected_results, config)
    
    # Save marked images
    marked_topdown_path = os.path.join(output_dir, f"topdown_marked_iter_{iteration}.png")
    marked_room_annotation_path = os.path.join(output_dir, f"room_annotation_marked_iter_{iteration}.png")
    
    cv2.imwrite(marked_topdown_path, marked_topdown)
    cv2.imwrite(marked_room_annotation_path, marked_room_annotation)
    
    print(f"✓ Saved marked images for iteration {iteration + 1}:")
    print(f"  - Topdown: {marked_topdown_path}")
    print(f"  - Room annotation: {marked_room_annotation_path}")
    
    return marked_topdown_path, marked_room_annotation_path

def create_multi_point_summary(selected_results: List[Dict[str, Any]], 
                              config: Dict[str, Any], output_dir: str) -> str:
    """
    Create a summary of all multi-point selections.
    
    Args:
        selected_results: List of all selection results
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Path to the summary JSON file
    """
    summary_data = {
        "workflow_status": "completed_successfully",
        "total_iterations": len(selected_results),
        "k_points": config.get('k_points', len(selected_results)),
        "goal_object": config.get('scene_config', {}).get('goal_object', 'unknown'),
        "iterations": []
    }
    
    for i, result in enumerate(selected_results):
        room_result = result.get('room_result', {})
        node_result = result.get('node_result', {})
        action_result = result.get('action_result', {})
        
        # Extract room selection info
        room_selection = room_result.get('llm_response', {})
        selected_room = room_selection.get('selected_room')
        room_attempts = room_selection.get('attempts_made', 1)
        
        # Extract node selection info  
        node_selection = node_result.get('llm_response', {})
        selected_node_id = node_selection.get('selected_node_id')
        node_attempts = node_selection.get('attempts_made', 1)
        
        # Extract world coordinates
        selected_node = node_result.get('selected_node', {})
        world_coordinates = selected_node.get('world_coordinates')
        
        # Extract action file path
        action_files = action_result.get('generated_files', {})
        action_file = action_files.get('action_json')
        
        iteration_summary = {
            "iteration": i + 1,
            "room_selection": {
                "selected_room": selected_room,
                "attempts_made": room_attempts
            },
            "node_selection": {
                "selected_node_id": selected_node_id,
                "world_coordinates": world_coordinates,
                "attempts_made": node_attempts
            },
            "action_file": os.path.basename(action_file) if action_file else None
        }
        
        summary_data["iterations"].append(iteration_summary)
    
    # Save summary
    summary_path = os.path.join(output_dir, "multi_point_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Multi-point summary saved to: {summary_path}")
    return summary_path

def validate_k_points_config(config: Dict[str, Any]) -> int:
    """
    Validate and return the k_points configuration value.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated k_points value
        
    Raises:
        ValueError: If k_points is invalid
    """
    k_points = config.get('k_points', 1)
    
    if not isinstance(k_points, int) or k_points < 1:
        raise ValueError(f"k_points must be a positive integer, got: {k_points}")
    
    if k_points > 10:
        print(f"⚠️  Warning: k_points={k_points} is quite large, this may take a long time")
    
    return k_points

def generate_iteration_prompt_addition(iteration: int, selected_results: List[Dict[str, Any]], 
                                     goal_object: str, prompt_type: str = "room") -> str:
    """
    Generate additional prompt text for LLM based on previous iterations.
    
    Args:
        iteration: Current iteration number (0-based)
        selected_results: List of previous selection results
        goal_object: Target object name
        prompt_type: Type of prompt ("room" or "node")
        
    Returns:
        Additional prompt text
    """
    if iteration == 0:
        return ""
    
    if prompt_type == "room":
        # Room selection prompt addition
        prev_rooms = []
        for result in selected_results:
            room_result = result.get('room_result', {})
            room_selection = room_result.get('llm_response', {})
            selected_room = room_selection.get('selected_room')
            if selected_room is not None:
                prev_rooms.append(selected_room)
        
        return f"""
        
IMPORTANT CONTEXT: You can see COLORED CROSS markers on the map. These indicate {iteration} previously selected navigation points.
- Previous selections were in rooms: {prev_rooms}
- Please choose a room that provides a DIFFERENT strategic advantage for finding the {goal_object}.
- Consider alternative approaches, backup locations, or different vantage points.
- Diversity in room selection increases the chances of successful navigation.
        """
    
    elif prompt_type == "node":
        # Node selection prompt addition
        prev_nodes = []
        for result in selected_results:
            node_result = result.get('node_result', {})
            node_selection = node_result.get('llm_response', {})
            selected_node_id = node_selection.get('selected_node_id')
            if selected_node_id is not None:
                prev_nodes.append(selected_node_id)
        
        return f"""
        
IMPORTANT CONTEXT: You can see COLORED CROSS markers on the original map. These indicate {iteration} previously selected navigation points.
- Previous node selections: {prev_nodes}
- Please choose a node that provides a DIFFERENT approach to finding the {goal_object}.
- Consider different angles, distances, or strategic positions within this room.
- Avoid selecting nodes too close to the marked previous selections.
        """
    
    return ""

if __name__ == "__main__":
    # Test the utility functions
    print("Multi-point utilities loaded successfully!")
