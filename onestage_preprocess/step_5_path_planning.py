"""
Step 6: Generate action.json.
This step consolidates the agent's starting state, the selected target's information,
and the path to the wall mask into a single action.json file for the next stage of the pipeline.
"""

import os
import json
from typing import Dict, Any

def path_planning_step(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
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
        agent_rotation = [0, 0, 0, 0]
        print("⚠️ Agent rotation is null, using default [0, 0, 0, 0].")

    agent_state = {
        "position": agent_position,
        "rotation": agent_rotation
    }
    print(f"✓ Agent initial state loaded: position={agent_position}, rotation={agent_rotation}")

    # 2. Get target information
    # Load the results from the one-stage node selection step
    # First try the one-stage log file, then fallback to traditional log file for compatibility
    onestage_node_log_path = os.path.join(output_dir, "onestage_node_selection_log.json")
    node_log_path = os.path.join(output_dir, "node_selection_log.json")
    
    if os.path.exists(onestage_node_log_path):
        log_file_path = onestage_node_log_path
        print(f"✓ Using one-stage node selection log: {log_file_path}")
    elif os.path.exists(node_log_path):
        log_file_path = node_log_path
        print(f"✓ Using traditional node selection log: {log_file_path}")
    else:
        raise FileNotFoundError(f"Node selection log not found. Tried:\n  - {onestage_node_log_path}\n  - {node_log_path}")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
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

    # 3. Get the absolute path to the wall mask
    wall_mask_relative_path = os.path.join(output_dir, "wall_mask.png")
    if not os.path.exists(wall_mask_relative_path):
        raise FileNotFoundError(f"Wall mask not found at: {wall_mask_relative_path}")
    
    wall_mask_absolute_path = os.path.abspath(wall_mask_relative_path)
    print(f"✓ Wall mask path resolved: {wall_mask_absolute_path}")

    # 4. Assemble the final action.json data
    action_data = {
        "agent_state": agent_state,
        "target_info": target_info,
        "wall_mask": wall_mask_absolute_path
    }

    # 5. Save the action.json file
    action_path = os.path.join(output_dir, "action.json")
    with open(action_path, 'w', encoding='utf-8') as f:
        json.dump(action_data, f, indent=4)
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