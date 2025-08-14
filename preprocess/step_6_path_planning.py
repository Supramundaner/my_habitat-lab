"""
Step 6: Path planning from target_coordinate to selected node.
When target_coordinate is not null, perform A* pathfinding to generate waypoints and action.json.
"""

import os
import cv2
import json
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import heapq

def pixel_to_world(pixel_coords: List[float], metadata: Dict[str, Any]) -> List[float]:
    """Convert pixel coordinates to world coordinates."""
    topdown_meta = metadata['topdown_metadata']
    origin_pixels = topdown_meta['origin_in_pixels']
    spacing = topdown_meta['spacing_in_meters_per_pixel']
    
    # Convert to world coordinates
    x_world = (pixel_coords[0] - origin_pixels[0]) * spacing
    z_world = (pixel_coords[1] - origin_pixels[1]) * spacing
    
    return [x_world, z_world]

def world_to_pixel(world_coords: List[float], metadata: Dict[str, Any]) -> List[int]:
    """Convert world coordinates to pixel coordinates."""
    topdown_meta = metadata['topdown_metadata']
    origin_pixels = topdown_meta['origin_in_pixels']
    spacing = topdown_meta['spacing_in_meters_per_pixel']
    
    # Convert to pixel coordinates
    x_pixel = int(world_coords[0] / spacing + origin_pixels[0])
    z_pixel = int(world_coords[1] / spacing + origin_pixels[1])
    
    return [x_pixel, z_pixel]

def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_neighbors(pos: Tuple[int, int], mask: np.ndarray) -> List[Tuple[int, int]]:
    """Get valid neighbors for A* search."""
    height, width = mask.shape
    x, y = pos
    neighbors = []
    
    # 8-connected neighbors
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < width and 0 <= ny < height:
                # Check if walkable (255 in wall_mask means walkable)
                if mask[ny, nx] == 255:
                    neighbors.append((nx, ny))
    
    return neighbors

def a_star_pathfinding(start: Tuple[int, int], goal: Tuple[int, int], wall_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    A* pathfinding algorithm.
    
    Args:
        start: Start position (x, y) in pixel coordinates
        goal: Goal position (x, y) in pixel coordinates
        wall_mask: Binary mask where 255 = walkable, 0 = wall
        
    Returns:
        List of waypoints from start to goal, or empty list if no path found
    """
    if wall_mask[start[1], start[0]] == 0 or wall_mask[goal[1], goal[0]] == 0:
        print(f"⚠️ Start {start} or goal {goal} is not walkable")
        return []
    
    # Priority queue: (f_score, g_score, position)
    open_set = [(0.0, 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: euclidean_distance(start, goal)}
    
    visited = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current, wall_mask):
            if neighbor in visited:
                continue
            
            # Calculate movement cost (diagonal movement costs more)
            dx = abs(neighbor[0] - current[0])
            dy = abs(neighbor[1] - current[1])
            movement_cost = 1.414 if (dx == 1 and dy == 1) else 1.0
            
            tentative_g_score = g_score[current] + movement_cost
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
    
    # No path found
    return []

def sample_waypoints(path: List[Tuple[int, int]], spacing_meters: float, metadata: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Sample waypoints from path every spacing_meters.
    
    Args:
        path: List of pixel coordinates
        spacing_meters: Distance between waypoints in meters
        metadata: Metadata for coordinate conversion
        
    Returns:
        List of sampled waypoints
    """
    if len(path) < 2:
        return path
    
    # Convert spacing to pixels
    spacing_pixels = spacing_meters / metadata['topdown_metadata']['spacing_in_meters_per_pixel']
    
    waypoints = [path[0]]  # Always include start
    
    last_waypoint = path[0]
    total_distance = 0.0
    
    for i in range(1, len(path)):
        current = path[i]
        segment_distance = euclidean_distance(last_waypoint, current)
        total_distance += euclidean_distance(path[i-1], current)
        
        # Check if we've traveled enough distance for next waypoint
        if total_distance >= spacing_pixels:
            waypoints.append(current)
            last_waypoint = current
            total_distance = 0.0
    
    # Always include goal if it's not already added
    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])
    
    return waypoints

def create_action_json(start_world: List[float], waypoints_world: List[List[float]], 
                      config: Dict[str, Any]) -> Dict[str, Any]:
    """Create action.json with agent state and move sequence."""
    
    # Get agent state from config
    agent_position = config['scene_config']['target_coordinate']
    agent_rotation = config['scene_config']['rotation']
    goal_object = config['scene_config']['goal_object']
    # If rotation is null, use default [0, 0, 0, 0]
    if agent_rotation is None:
        agent_rotation = [0, 0, 0, 0]
    
    # Create action sequence
    sequence = []
    for waypoint in waypoints_world:
        sequence.append({
            "type": "move_to",
            "params": {
                "x": waypoint[0],
                "z": waypoint[1]
            }
        })
    
    action_data = {
        "agent_state": {
            "position": agent_position,
            "rotation": agent_rotation
        },
        "action": [
            {
                "sequence": sequence,
                "target": goal_object
            }
        ]
    }
    
    return action_data

def visualize_path_on_topdown(topdown_image: np.ndarray, start_pixel: List[int], 
                             goal_pixel: List[int], path: List[Tuple[int, int]], 
                             waypoints: List[Tuple[int, int]]) -> np.ndarray:
    """
    Visualize the path planning result on topdown view.
    
    Args:
        topdown_image: Original topdown view image
        start_pixel: Start position in pixels
        goal_pixel: Goal position in pixels  
        path: Full A* path
        waypoints: Sampled waypoints
        
    Returns:
        Annotated image
    """
    result = topdown_image.copy()
    
    # Draw full path in light blue
    if len(path) > 1:
        path_points = np.array(path, dtype=np.int32)
        for i in range(len(path_points) - 1):
            cv2.line(result, tuple(path_points[i]), tuple(path_points[i+1]), 
                    (255, 255, 0), 3)  # Light blue line
    
    # Draw waypoints in green circles
    for i, waypoint in enumerate(waypoints):
        x, y = waypoint
        cv2.circle(result, (x, y), 12, (0, 255, 0), -1)  # Green filled circle
        cv2.circle(result, (x, y), 12, (0, 0, 0), 2)     # Black border
        
        # Add waypoint number
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i)
        cv2.putText(result, text, (x-6, y+6), font, 0.7, (255, 255, 255), 2)
    
    # Draw start point in blue
    cv2.circle(result, tuple(start_pixel), 20, (255, 0, 0), -1)  # Blue
    cv2.circle(result, tuple(start_pixel), 20, (255, 255, 255), 3)  # White border
    cv2.putText(result, "START", (start_pixel[0]-25, start_pixel[1]-25), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    # Draw goal point in red
    cv2.circle(result, tuple(goal_pixel), 20, (0, 0, 255), -1)  # Red
    cv2.circle(result, tuple(goal_pixel), 20, (255, 255, 255), 3)  # White border
    cv2.putText(result, "GOAL", (goal_pixel[0]-25, goal_pixel[1]+40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    return result

def path_planning_step(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Perform path planning from target_coordinate to selected node.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print("\\n" + "="*60)
    print("STEP 6: Path planning and action generation")
    print("="*60)
    
    # Check if target_coordinate is provided
    target_coord = config['scene_config']['target_coordinate']
    if target_coord is None:
        print("⚠️ target_coordinate is null, skipping path planning")
        return {
            "generated_files": {},
            "results": {"message": "target_coordinate is null, path planning skipped"}
        }
    
    print(f"📍 Target coordinate: {target_coord}")
    
    # Load metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load wall mask
    wall_mask_path = os.path.join(output_dir, "wall_mask.png")
    if not os.path.exists(wall_mask_path):
        raise FileNotFoundError(f"Wall mask not found: {wall_mask_path}")
    
    wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"✓ Wall mask loaded: {wall_mask.shape}")
    
    # Load topdown view  
    topdown_path = os.path.join(output_dir, "topdown_view.png")
    if not os.path.exists(topdown_path):
        raise FileNotFoundError(f"Topdown view not found: {topdown_path}")
    
    topdown_image = cv2.imread(topdown_path)
    print(f"✓ Topdown view loaded: {topdown_image.shape}")
    
    # Load node selection results
    node_log_path = os.path.join(output_dir, "node_selection_log.json")
    if not os.path.exists(node_log_path):
        raise FileNotFoundError(f"Node selection log not found: {node_log_path}")
    
    with open(node_log_path, 'r', encoding='utf-8') as f:
        node_log = json.load(f)
    
    selected_node = node_log['selected_node']
    goal_pixel = [int(selected_node['pixel_coordinates'][0]), 
                  int(selected_node['pixel_coordinates'][1])]
    goal_world = selected_node['world_coordinates']
    
    print(f"🎯 Selected node: {selected_node['node_id']}")
    print(f"📍 Goal position (world): {goal_world}")
    print(f"📍 Goal position (pixel): {goal_pixel}")
    
    # Convert start coordinate to pixels (only use x, z from 3D coordinate)
    start_world = [target_coord[0], target_coord[2]]  # Use x, z from (x, y, z)
    start_pixel = world_to_pixel(start_world, metadata)
    
    print(f"🚀 Start position (world): {start_world}")
    print(f"🚀 Start position (pixel): {start_pixel}")
    
    # Perform A* pathfinding
    print("🔍 Performing A* pathfinding...")
    path = a_star_pathfinding(tuple(start_pixel), tuple(goal_pixel), wall_mask)
    
    if not path:
        raise RuntimeError("No path found from start to goal")
    
    print(f"✓ Path found with {len(path)} points")
    
    # Sample waypoints every 5 meters
    waypoint_spacing = 5.0  # meters
    waypoints = sample_waypoints(path, waypoint_spacing, metadata)
    print(f"✓ Sampled {len(waypoints)} waypoints (every {waypoint_spacing}m)")
    
    # Convert waypoints to world coordinates
    waypoints_world = []
    for waypoint in waypoints:
        world_coord = pixel_to_world(list(waypoint), metadata)
        waypoints_world.append(world_coord)
    
    # Create action.json
    action_data = create_action_json(start_world, waypoints_world, config)
    
    action_path = os.path.join(output_dir, "action.json")
    with open(action_path, 'w', encoding='utf-8') as f:
        json.dump(action_data, f, indent=4)
    
    print(f"✓ Action file saved: {action_path}")
    
    # Create visualization
    path_image = visualize_path_on_topdown(topdown_image, start_pixel, goal_pixel, 
                                          path, waypoints)
    
    path_visualization_path = os.path.join(output_dir, "path.png")
    cv2.imwrite(path_visualization_path, path_image)
    print(f"✓ Path visualization saved: {path_visualization_path}")
    
    # Create path log
    path_log = {
        "start_coordinate_3d": target_coord,
        "start_coordinate_2d": start_world,
        "start_pixel": start_pixel,
        "goal_coordinate_2d": goal_world,
        "goal_pixel": goal_pixel,
        "selected_node_id": selected_node['node_id'],
        "path_length_pixels": len(path),
        "waypoints_count": len(waypoints),
        "waypoint_spacing_meters": waypoint_spacing,
        "waypoints_world": waypoints_world,
        "full_path_pixels": path
    }
    
    path_log_path = os.path.join(output_dir, "path_planning_log.json")
    with open(path_log_path, 'w', encoding='utf-8') as f:
        json.dump(path_log, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Path planning log saved: {path_log_path}")
    
    return {
        "generated_files": {
            "action_json": action_path,
            "path_visualization": path_visualization_path,
            "path_planning_log": path_log_path
        },
        "results": {
            "start_coordinate_3d": target_coord,
            "start_coordinate_2d": start_world,
            "goal_coordinate_2d": goal_world,
            "selected_node_id": selected_node['node_id'],
            "path_found": True,
            "path_length": len(path),
            "waypoints_count": len(waypoints),
            "waypoint_spacing_meters": waypoint_spacing
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python step_6_path_planning.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    output_dir = config['output']['output_dir']
    
    try:
        result = path_planning_step(config, output_dir)
        print("Step 6 completed successfully:", result)
    except Exception as e:
        print(f"Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
