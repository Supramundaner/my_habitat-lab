"""
Step 5: Generate action.json.
This step consolidates the agent's starting state, the selected coordinate information,
and generates an action.json file for the next stage of the pipeline.
"""

import os
import json
from typing import Dict, Any

def path_planning_step(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generates an action.json file containing the agent's initial state
    and the target coordinate information.

    Args:
        config: Configuration dictionary.
        output_dir: Output directory path.

    Returns:
        Dictionary with generated files and results.
    """
    print("="*60)
    print("STEP 5: Generating action.json")
    print("="*60)

    # 1. Get agent's initial state from config
    agent_position = config['scene_config'].get('target_coordinate')
    agent_rotation = config['scene_config'].get('rotation')
    
    if agent_position is None:
        raise ValueError("'target_coordinate' (agent's starting position) not found in scene_config.")
        
    # Use a default rotation if it's null, as per the example
    if agent_rotation is None:
        agent_rotation = [0, 0, 0, 0]
        print("⚠️ Agent rotation is null, using default [0, 0, 0, 0].")

    agent_state = {
        "position": agent_position,
        "rotation": agent_rotation
    }
    print(f"✓ Agent initial state loaded: position={agent_position}, rotation={agent_rotation}")

    # 2. Get target coordinate information
    # Load the results from the coordinate selection step
    coordinate_log_path = os.path.join(output_dir, "coordinate_selection_log.json")
    
    if not os.path.exists(coordinate_log_path):
        raise FileNotFoundError(f"Coordinate selection log not found at: {coordinate_log_path}")
    
    with open(coordinate_log_path, 'r', encoding='utf-8') as f:
        coordinate_log = json.load(f)
    
    # Get the selected coordinate (in original image space)
    selected_coord_original = coordinate_log['selected_coordinate_original']
    selected_coord_normalized = coordinate_log['selected_coordinate_normalized']
    
    # Convert pixel coordinates to world coordinates
    # This will need to be implemented based on the metadata from topdown generation
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Convert pixel coordinates to world coordinates using metadata
    world_coordinate = pixel_to_world_coordinate(
        selected_coord_original, metadata
    )
    
    goal_object_name = config['scene_config'].get('goal_object')
    if goal_object_name is None:
        raise ValueError("'goal_object' name not found in scene_config.")

    target_info = {
        "coordinate": world_coordinate,  # World coordinates [x, z]
        "pixel_coordinate": selected_coord_original,  # Original pixel coordinates
        "normalized_coordinate": selected_coord_normalized,  # Normalized coordinates
        "name": goal_object_name
    }
    print(f"✓ Target info loaded: pixel_coord={selected_coord_original}, world_coord={world_coordinate}, name={goal_object_name}")

    # 3. Assemble the final action.json data
    action_data = {
        "agent_state": agent_state,
        "target_info": target_info,
        "method": "coordinate_selection",
        "workflow_type": "one_stage_coordinate"
    }

    # 4. Save the action.json file
    action_path = os.path.join(output_dir, "action.json")
    with open(action_path, 'w', encoding='utf-8') as f:
        json.dump(action_data, f, indent=4)
    print(f"✓ Action file saved successfully: {action_path}")
    
    # 5. Return the results
    return {
        "generated_files": {
            "action_json": action_path
        },
        "results": {
            "action_file_generated": True,
            "agent_start_position": agent_position,
            "target_pixel_coordinate": selected_coord_original,
            "target_world_coordinate": world_coordinate,
            "target_name": goal_object_name
        }
    }

def pixel_to_world_coordinate(pixel_coord, metadata):
    """
    Convert pixel coordinate to world coordinate using metadata.
    
    Args:
        pixel_coord: (x, y) pixel coordinate in the topdown image
        metadata: Metadata dictionary from topdown generation
    
    Returns:
        [x, z] world coordinate
    """
    try:
        # Extract transformation parameters from metadata
        top_down_map = metadata.get('top_down_map', {})
        map_resolution = top_down_map.get('map_resolution')
        lower_bound = top_down_map.get('lower_bound', [0, 0])  # [x_min, z_min]
        upper_bound = top_down_map.get('upper_bound', [1, 1])  # [x_max, z_max]
        
        if map_resolution is None:
            raise ValueError("map_resolution not found in metadata")
        
        pixel_x, pixel_y = pixel_coord
        
        # Convert pixel coordinates to world coordinates
        # Note: pixel y corresponds to world z, and we need to flip y axis
        world_x = lower_bound[0] + pixel_x * map_resolution
        world_z = upper_bound[1] - pixel_y * map_resolution  # Flip y axis
        
        return [world_x, world_z]
        
    except Exception as e:
        print(f"⚠️ Error converting pixel to world coordinate: {e}")
        print(f"Using fallback conversion...")
        
        # Fallback: simple scaling based on assumed bounds
        pixel_x, pixel_y = pixel_coord
        
        # Assume reasonable defaults if metadata is incomplete
        world_x = (pixel_x / 1000.0) * 10.0 - 5.0  # Map to [-5, 5] range
        world_z = (pixel_y / 1000.0) * 10.0 - 5.0  # Map to [-5, 5] range
        
        print(f"Fallback conversion: pixel {pixel_coord} -> world [{world_x:.3f}, {world_z:.3f}]")
        return [world_x, world_z]

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
    
    # Assume output_dir is defined in config
    output_dir = config.get('output', {}).get('output_dir', 'output')
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' not found. This script assumes previous steps have run.")

    try:
        result = path_planning_step(config, output_dir)
        print("\nStep 6 completed successfully:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nStep 6 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)