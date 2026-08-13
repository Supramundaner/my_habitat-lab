"""
Step 6: Generate action.json.
This step consolidates the agent's starting state, the selected target's information,
and the path to the wall mask into a single action.json file for the next stage of the pipeline.
"""

import os
import json
from typing import Dict, Any

if __package__:
    from .config import load_preprocessing_config
else:
    from config import load_preprocessing_config

def build_navigation_request(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generates an action.json file containing the agent's initial state,
    the target information, and the path to the wall mask.

    Args:
        config: Configuration dictionary.
        output_dir: Output directory path.

    Returns:
        Dictionary with generated files and results.
    """
    print("="*60)
    print("STEP 6: Generating action.json")
    print("="*60)

    # 1. Get agent's initial state from config
    agent_position = config['scene_config'].get('target_coordinate')
    agent_rotation = config['scene_config'].get('rotation')
    
    if agent_position is None:
        raise ValueError("'target_coordinate' (agent's starting position) not found in scene_config.")
        
    # Use a default rotation if it's null, as per the example
    if agent_rotation is None:
        agent_rotation = [0, 0, 0, 1]
        print("⚠️ Agent rotation is null, using identity [0, 0, 0, 1].")

    agent_state = {
        "position": agent_position,
        "rotation": agent_rotation
    }
    print(f"✓ Agent initial state loaded: position={agent_position}, rotation={agent_rotation}")

    # 2. Get target information
    # Load the results from the node selection step
    node_log_path = os.path.join(output_dir, "node_selection_log.json")
    if not os.path.exists(node_log_path):
        raise FileNotFoundError(f"Node selection log not found: {node_log_path}")
    
    with open(node_log_path, 'r', encoding='utf-8') as f:
        node_log = json.load(f)
    
    selected_node = node_log['selected_node']
    
    # Get target coordinates (world coordinates) and name
    target_world_coords = selected_node.get('world_coordinates')
    if target_world_coords is None:
        raise ValueError("Selected node is missing 'world_coordinates'.")
        
    goal_object_name = config['scene_config'].get('goal_object')
    if goal_object_name is None:
        raise ValueError("'goal_object' name not found in scene_config.")

    target_info = {
        "coordinate": target_world_coords, # These are already [x, z]
        "name": goal_object_name
    }
    print(f"✓ Target info loaded: node_id={selected_node['node_id']}, coordinate={target_world_coords}, name={goal_object_name}")

    # 3. Keep the wall-mask reference portable relative to action.json.
    wall_mask_path = os.path.join(output_dir, "wall_mask.png")
    if not os.path.exists(wall_mask_path):
        raise FileNotFoundError(f"Wall mask not found at: {wall_mask_path}")

    action_path = os.path.join(output_dir, "action.json")
    wall_mask_reference = os.path.relpath(
        wall_mask_path, start=os.path.dirname(action_path)
    )
    print(f"✓ Wall mask path stored relative to action.json: {wall_mask_reference}")

    # Preserve the world-space projection contract used to create the wall
    # mask.  The online runtime validates this before resizing the image.
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Top-down metadata not found at: {metadata_path}")
    metadata_reference = os.path.relpath(
        metadata_path, start=os.path.dirname(action_path)
    )

    # 4. Assemble the final action.json data
    action_data = {
        "agent_state": agent_state,
        "target_info": target_info,
        "wall_mask": wall_mask_reference,
        "map_metadata": metadata_reference,
    }

    # 5. Save the action.json file
    temporary_action_path = f"{action_path}.tmp"
    try:
        with open(temporary_action_path, 'w', encoding='utf-8') as f:
            json.dump(action_data, f, indent=4, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_action_path, action_path)
    except Exception:
        try:
            os.unlink(temporary_action_path)
        except FileNotFoundError:
            pass
        raise
    print(f"✓ Action file saved successfully: {action_path}")
    
    # 6. Return the results
    return {
        "generated_files": {
            "action_json": action_path
        },
        "results": {
            "action_file_generated": True,
            "agent_start_position": agent_position,
            "target_node_id": selected_node['node_id'],
            "target_coordinate": target_world_coords,
            "target_name": goal_object_name
        }
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m "
            "reason_navi.preprocessing.build_navigation_request <config_path>"
        )
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_preprocessing_config(config_path)
    
    # Assume output_dir is defined in config
    output_dir = config.get('output', {}).get('output_dir', 'output')
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' not found. This script assumes previous steps have run.")

    try:
        result = build_navigation_request(config, output_dir)
        print("\nStep 6 completed successfully:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nStep 6 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
