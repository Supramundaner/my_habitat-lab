"""
Step 7: Generate action.json for Image-Instance Navigation.
This step consolidates the agent's starting state, the selected target's information,
and the path to the wall mask into a single action.json file.
Also generates path.png showing starting position and destination.
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, Any

def world_to_pixel(world_coords, nodes_data):
    """Convert world coordinates to pixel coordinates using navigation nodes as reference."""
    # Find the closest nodes to establish coordinate transformation
    min_dist = float('inf')
    closest_node = None
    
    for node in nodes_data['nodes']:
        node_world = node['world_coordinates']
        dist = ((world_coords[0] - node_world[0])**2 + (world_coords[1] - node_world[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    
    if closest_node is None:
        # Fallback if no nodes available
        return (500, 500)  # Default center position
    
    # If the target is exactly a node, return its pixel coordinates
    if min_dist < 0.01:  # Very close match
        pixel_coords = closest_node['pixel_coordinates']
        return (int(pixel_coords[0]), int(pixel_coords[1]))
    
    # For points not exactly on nodes, use linear interpolation
    # Find two closest nodes for better interpolation
    nodes_by_distance = sorted(nodes_data['nodes'], 
                              key=lambda n: ((world_coords[0] - n['world_coordinates'][0])**2 + 
                                           (world_coords[1] - n['world_coordinates'][1])**2)**0.5)
    
    if len(nodes_by_distance) >= 2:
        node1, node2 = nodes_by_distance[0], nodes_by_distance[1]
        
        # Simple linear interpolation based on world distance
        world1 = node1['world_coordinates']
        world2 = node2['world_coordinates']
        pixel1 = (node1['pixel_coordinates'][0], node1['pixel_coordinates'][1])
        pixel2 = (node2['pixel_coordinates'][0], node2['pixel_coordinates'][1])
        
        # Calculate interpolation weights
        dist1 = ((world_coords[0] - world1[0])**2 + (world_coords[1] - world1[1])**2)**0.5
        dist2 = ((world_coords[0] - world2[0])**2 + (world_coords[1] - world2[1])**2)**0.5
        
        if dist1 + dist2 > 0:
            weight1 = dist2 / (dist1 + dist2)  # Inverse distance weighting
            weight2 = dist1 / (dist1 + dist2)
        else:
            weight1, weight2 = 0.5, 0.5
        
        # Interpolate pixel coordinates
        pixel_x = int(weight1 * pixel1[0] + weight2 * pixel2[0])
        pixel_y = int(weight1 * pixel1[1] + weight2 * pixel2[1])
        
        return (pixel_x, pixel_y)
    
    # Fallback to closest node
    pixel_coords = closest_node['pixel_coordinates']
    return (int(pixel_coords[0]), int(pixel_coords[1]))

def generate_path_visualization(topdown_path: str, agent_position: list, target_coords: list, 
                              output_dir: str, nodes_data: Dict = None) -> str:
    """Generate path.png with starting position and destination marked."""
    
    # Load topdown image
    topdown_image = cv2.imread(topdown_path)
    if topdown_image is None:
        raise FileNotFoundError(f"Topdown image not found: {topdown_path}")
    
    # Convert world coordinates to pixel coordinates
    # For agent position, use x and z (ignore y which is height)
    agent_world_coords = [agent_position[0], agent_position[2]]  # [x, z]
    
    if nodes_data:
        agent_pixel = world_to_pixel(agent_world_coords, nodes_data)
        target_pixel = world_to_pixel(target_coords, nodes_data)
        print(f"  - Using navigation nodes for coordinate transformation")
    else:
        # This should not happen now, but keep as fallback
        print("⚠️ No navigation nodes data available")
        height, width = topdown_image.shape[:2]
        agent_pixel = (width // 2, height // 2)
        target_pixel = (width // 2 + 50, height // 2 + 50)
    
    print(f"  - Agent world coords: {agent_world_coords}")
    print(f"  - Target world coords: {target_coords}")
    print(f"  - Agent pixel position: {agent_pixel}")
    print(f"  - Target pixel position: {target_pixel}")
    
    # Create visualization
    result_image = topdown_image.copy()
    
    # Draw path line (optional - simple straight line)
    cv2.line(result_image, agent_pixel, target_pixel, (0, 255, 255), 3)  # Cyan line
    
    # Draw starting position (green circle)
    cv2.circle(result_image, agent_pixel, 15, (0, 255, 0), -1)  # Green filled circle
    cv2.circle(result_image, agent_pixel, 18, (0, 0, 0), 2)     # Black outline
    cv2.putText(result_image, "START", (agent_pixel[0] + 25, agent_pixel[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw destination (red circle)
    cv2.circle(result_image, target_pixel, 15, (0, 0, 255), -1)  # Red filled circle
    cv2.circle(result_image, target_pixel, 18, (0, 0, 0), 2)     # Black outline
    cv2.putText(result_image, "DEST", (target_pixel[0] + 25, target_pixel[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add legend with better formatting
    legend_y = 30
    # Add background for legend
    cv2.rectangle(result_image, (5, 5), (400, 110), (0, 0, 0), -1)  # Black background
    cv2.rectangle(result_image, (5, 5), (400, 110), (255, 255, 255), 2)  # White border
    
    cv2.putText(result_image, "Navigation Path", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_image, f"Start: [{agent_world_coords[0]:.2f}, {agent_world_coords[1]:.2f}]", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result_image, f"Target: [{target_coords[0]:.2f}, {target_coords[1]:.2f}]", (10, legend_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Calculate and display distance
    distance = ((target_coords[0] - agent_world_coords[0])**2 + (target_coords[1] - agent_world_coords[1])**2)**0.5
    cv2.putText(result_image, f"Distance: {distance:.2f} units", (10, legend_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Save path visualization
    path_image_path = os.path.join(output_dir, "path.png")
    cv2.imwrite(path_image_path, result_image)
    
    return path_image_path

def path_planning_step(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generate action.json file containing the agent's initial state,
    target information, and wall mask path.

    Args:
        config: Configuration dictionary
        output_dir: Output directory path

    Returns:
        Dictionary with generated files and results
    """
    print("="*60)
    print("STEP 7: Generating action.json for Image-Instance Navigation")
    print("="*60)

    # 1. Get agent's initial state from config
    agent_position = config['scene_config'].get('agent_position')
    agent_rotation = config['scene_config'].get('agent_rotation')
    
    if agent_position is None:
        raise ValueError("'agent_position' (agent's starting position) not found in scene_config")
        
    # Use a default rotation if it's null
    if agent_rotation is None:
        agent_rotation = [0, 0, 0, 1]  # Default quaternion
        print("⚠️ Agent rotation is null, using default [0, 0, 0, 1]")

    agent_state = {
        "position": agent_position,
        "rotation": agent_rotation
    }
    print(f"✓ Agent initial state loaded: position={agent_position}, rotation={agent_rotation}")

    # 2. Get target information from node selection results
    node_log_path = os.path.join(output_dir, "node_selection_log.json")
    if not os.path.exists(node_log_path):
        raise FileNotFoundError(f"Node selection log not found: {node_log_path}")
    
    with open(node_log_path, 'r', encoding='utf-8') as f:
        node_log = json.load(f)
    
    selected_node = node_log['selected_node']
    
    # Get target coordinates (world coordinates)
    target_world_coords = selected_node.get('world_coordinates')
    if target_world_coords is None:
        raise ValueError("Selected node is missing 'world_coordinates'")
        
    # Get goal object information
    goal_object_category = config['scene_config'].get('goal_object')
    if goal_object_category is None:
        raise ValueError("'goal_object' category not found in scene_config")

    # Load goal metadata for additional information
    goal_metadata_path = os.path.join(output_dir, "goal_metadata.json")
    goal_object_name = goal_object_category  # Default to category
    
    if os.path.exists(goal_metadata_path):
        with open(goal_metadata_path, 'r', encoding='utf-8') as f:
            goal_metadata = json.load(f)
        goal_object_name = goal_metadata.get('goal_object_name', goal_object_category)

    target_info = {
        "coordinate": target_world_coords,  # These are already [x, z]
        "name": goal_object_category,
        "id": goal_object_name
    }
    print(f"✓ Target info loaded: node_id={selected_node['node_id']}, coordinate={target_world_coords}, name={goal_object_name}")

    # 3. Get the absolute path to the wall mask
    wall_mask_relative_path = os.path.join(output_dir, "wall_mask.png")
    if not os.path.exists(wall_mask_relative_path):
        raise FileNotFoundError(f"Wall mask not found at: {wall_mask_relative_path}")
    
    wall_mask_absolute_path = os.path.abspath(wall_mask_relative_path)
    print(f"✓ Wall mask path resolved: {wall_mask_absolute_path}")

    # 4. Get goal image or description path (support both modes)
    use_text_nav = config.get('scene_config', {}).get('use_text_nav', False)
    
    if use_text_nav:
        # TextNav mode - use text description
        goal_path = os.path.join(output_dir, "goal_description.txt")
        if not os.path.exists(goal_path):
            raise FileNotFoundError(f"Goal description not found at: {goal_path}")
        goal_absolute_path = os.path.abspath(goal_path)
        print(f"✓ Goal description path resolved: {goal_absolute_path}")
    else:
        # Original mode - use goal image
        goal_path = os.path.join(output_dir, "goal_image.png")
        if not os.path.exists(goal_path):
            raise FileNotFoundError(f"Goal image not found at: {goal_path}")
        goal_absolute_path = os.path.abspath(goal_path)
        print(f"✓ Goal image path resolved: {goal_absolute_path}")

    # 5. Generate path visualization
    print("📍 Generating path visualization...")
    
    # Find topdown image
    topdown_path = os.path.join(output_dir, "topdown_view.png")
    if not os.path.exists(topdown_path):
        raise FileNotFoundError(f"Topdown view not found at: {topdown_path}")
    
    # Load navigation nodes data for coordinate transformation
    nodes_path = os.path.join(output_dir, "navigation_nodes.json")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Navigation nodes data not found: {nodes_path}")
    
    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)
    
    print("✓ Navigation nodes loaded for coordinate transformation")
    
    # Generate path visualization
    path_image_path = generate_path_visualization(
        topdown_path, agent_position, target_world_coords, output_dir, nodes_data
    )
    print(f"✓ Path visualization saved: {path_image_path}")

    # 6. Assemble the final action.json data for Image-Instance Navigation
    action_data = {
        "agent_state": agent_state,
        "target_info": target_info,
        "wall_mask": wall_mask_absolute_path,
        "goal_path": goal_absolute_path,
        "navigation_type": "text_navigation" if use_text_nav else "image_instance_navigation",
        "metadata": {
            "episode_info": {
                "episodes_file": config['scene_config']['episodes_file'],
                "episode_id": config['scene_config']['episode_id']
            },
            "selected_room": node_log['selected_room'],
            "selected_node_id": selected_node['node_id'],
            "workflow_completed": True
        }
    }

    # 7. Save the action.json file
    action_path = os.path.join(output_dir, "action.json")
    with open(action_path, 'w', encoding='utf-8') as f:
        json.dump(action_data, f, indent=4, ensure_ascii=False)
    print(f"✓ Action file saved successfully: {action_path}")
    
    # 8. Return the results
    return {
        "generated_files": {
            "action_json": action_path,
            "path_visualization": path_image_path
        },
        "results": {
            "action_file_generated": True,
            "path_visualization_generated": True,
            "agent_start_position": agent_position,
            "target_node_id": selected_node['node_id'],
            "target_coordinate": target_world_coords,
            "target_name": goal_object_name,
            "target_category": goal_object_category,
            "navigation_type": "image_instance_navigation"
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python step_7_path_planning.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Get output directory from config
    output_dir = config.get('output', {}).get('output_dir', 'output_insnav')
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' not found. This script assumes previous steps have run.")

    try:
        result = path_planning_step(config, output_dir)
        print("\\nStep 7 completed successfully:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\\nStep 7 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
