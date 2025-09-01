"""
Step 0: Render goal image from episode data.
This step renders the target image that the agent needs to find.
"""

import os
import cv2
import json
import numpy as np
import habitat_sim
import magnum as mn
from typing import Dict, Any, Tuple

def validate_and_snap_position(sim: habitat_sim.Simulator, position: np.ndarray) -> np.ndarray:
    """
    Validate and snap position to navmesh if possible.
    
    Args:
        sim: Habitat-Sim simulator instance
        position: Target position [x, y, z]
        
    Returns:
        Validated position (snapped to navmesh if possible)
    """
    target_position = np.array(position, dtype=np.float32)
    return target_position
        
  

def make_habitat_configuration(scene_path: str, resolution: Tuple[int, int] = (512, 512), hfov: float = 90.0):
    """Create Habitat-Sim configuration for image rendering."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.load_semantic_mesh = False
    backend_cfg.enable_physics = False
    backend_cfg.random_seed = 1

    # RGB sensor configuration
    rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
    rgb_sensor_cfg.uuid = "color_sensor"
    rgb_sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_sensor_cfg.resolution = resolution
    rgb_sensor_cfg.position = [0.0, 0.0, 0.0]  # No offset from agent position
    rgb_sensor_cfg.hfov = hfov
    rgb_sensor_cfg.clear_color = [0.0, 0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])

def render_goal_image(scene_path: str, goal_image_data: Dict[str, Any], 
                     goal_name: str, goal_category: str, output_dir: str) -> Dict[str, Any]:
    """
    Render goal image from episode data.
    
    Args:
        scene_path: Path to the scene file
        goal_image_data: Image goal data containing position, rotation, hfov, etc.
        goal_name: Name of the goal object
        goal_category: Category of the goal object
        output_dir: Output directory path
        
    Returns:
        Dictionary with generated files and results
    """
    print(f"📁 Scene path: {scene_path}")
    print(f"🎯 Goal object: {goal_name} ({goal_category})")
    
    # Extract rendering parameters from goal_image_data
    position = goal_image_data['position']
    rotation = goal_image_data['rotation']  # quaternion [x, y, z, w]
    hfov = goal_image_data['hfov']
    image_dimensions = goal_image_data['image_dimensions']
    
    print(f"📷 Rendering parameters:")
    print(f"  - Position: {position}")
    print(f"  - Rotation: {rotation}")
    print(f"  - HFOV: {hfov}°")
    print(f"  - Dimensions: {image_dimensions}")
    
    # Check if scene file exists
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    
    # Create Habitat-Sim configuration
    sim_cfg = make_habitat_configuration(scene_path, tuple(image_dimensions), hfov)
    
    try:
        # Initialize simulator
        sim = habitat_sim.Simulator(sim_cfg)
        
        # Ensure navmesh is loaded for position validation
        if not sim.pathfinder.is_loaded:
            print("Warning: Navmesh not loaded, attempting to recompute...")
            try:
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
                print("✓ Navmesh recomputed successfully")
            except Exception as e:
                print(f"⚠️  Failed to recompute navmesh: {e}")
        
        # Validate and snap position to navmesh
        target_position = validate_and_snap_position(sim, position)
        
        # Set agent state using proper quaternion conversion
        agent = sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = target_position
        
        # Convert quaternion from [x, y, z, w] to habitat_sim quaternion (w, x, y, z)
        # Ensure all quaternion components are finite float values
        quat_x, quat_y, quat_z, quat_w = rotation
        
        # Validate quaternion components
        if not all(np.isfinite([quat_x, quat_y, quat_z, quat_w])):
            print(f"⚠️  Invalid quaternion values: {rotation}, using identity quaternion")
            quat_x, quat_y, quat_z, quat_w = 0.0, 0.0, 0.0, 1.0
        
        # Create numpy quaternion (w, x, y, z order for numpy.quaternion)
        try:
            agent_state.rotation = np.quaternion(float(quat_w), float(quat_x), float(quat_y), float(quat_z))
        except Exception as e:
            print(f"⚠️  Error creating quaternion: {e}, using identity quaternion")
            agent_state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        
        # Set the agent state
        agent.set_state(agent_state)
        
        # Verify agent state was set correctly
        current_state = agent.get_state()
        print(f"🤖 Agent positioned at: {current_state.position}")
        print(f"🧭 Agent rotation: {current_state.rotation}")
        # Render the image
        observations = sim.get_sensor_observations()
        
        # Get RGB image from the correct sensor
        if "color_sensor" in observations:
            rgb_image = observations["color_sensor"]
        else:
            # Fallback to any available RGB sensor
            rgb_keys = [k for k in observations.keys() if 'rgb' in k.lower() or 'color' in k.lower() or 'rgba' in k.lower()]
            if rgb_keys:
                rgb_image = observations[rgb_keys[0]]
                print(f"📷 Using sensor: {rgb_keys[0]}")
            else:
                # List all available sensors for debugging
                print(f"❌ Available sensors: {list(observations.keys())}")
                raise RuntimeError("No RGB sensor found in observations")
        
        print(f"🖼️  Rendered image shape: {rgb_image.shape}")
        print(f"🎨 Image data type: {rgb_image.dtype}, range: [{rgb_image.min()}, {rgb_image.max()}]")
        
        # Convert from RGBA to RGB if necessary
        if rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]
            print("✂️  Converted RGBA to RGB")
        
        # Ensure image is in correct format (0-255 uint8)
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            # Convert from [0,1] float to [0,255] uint8
            rgb_image = (rgb_image * 255).astype(np.uint8)
            print("🔄 Converted float image to uint8")
        elif rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
            print(f"🔄 Converted {rgb_image.dtype} to uint8")
        
        # Save goal image
        goal_image_path = os.path.join(output_dir, "goal_image.png")
        
        # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(goal_image_path, bgr_image)
        
        if success:
            print(f"✓ Goal image saved to: {goal_image_path}")
        else:
            raise RuntimeError(f"Failed to save image to: {goal_image_path}")
        
        # Save goal metadata with more comprehensive information
        goal_metadata = {
            "goal_object_name": goal_name,
            "goal_object_category": goal_category,
            "rendering_parameters": {
                "original_position": position,
                "used_position": target_position.tolist(),
                "rotation_quaternion": rotation,
                "hfov": hfov,
                "image_dimensions": image_dimensions
            },
            "agent_state": {
                "final_position": current_state.position.tolist(),
                "final_rotation": [current_state.rotation.x, current_state.rotation.y, 
                                current_state.rotation.z, current_state.rotation.w]
            },
            "image_path": goal_image_path,
            "image_size": rgb_image.shape[:2],
            "sensor_used": "color_sensor" if "color_sensor" in observations else rgb_keys[0] if 'rgb_keys' in locals() else "unknown"
        }
        
        metadata_path = os.path.join(output_dir, "goal_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(goal_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Goal metadata saved to: {metadata_path}")
        
        return {
            "generated_files": {
                "goal_image": goal_image_path,
                "goal_metadata": metadata_path
            },
            "results": {
                "goal_object_name": goal_name,
                "goal_object_category": goal_category,
                "image_size": rgb_image.shape[:2],
                "rendering_success": True
            }
        }
        
    except Exception as e:
        # Clean up on error
        if 'sim' in locals():
            sim.close()
        raise RuntimeError(f"Failed to render goal image: {e}")
    finally:
        # Clean up simulator
        if 'sim' in locals():
            sim.close()


if __name__ == "__main__":
    # Test function
    import sys
    if len(sys.argv) != 5:
        print("Usage: python step_0_render_goal_image.py <scene_path> <goal_image_data_json> <goal_name> <goal_category>")
        sys.exit(1)
    
    scene_path = sys.argv[1]
    goal_image_data_json = sys.argv[2]
    goal_name = sys.argv[3]
    goal_category = sys.argv[4]
    
    with open(goal_image_data_json, 'r') as f:
        goal_image_data = json.load(f)
    
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    result = render_goal_image(scene_path, goal_image_data, goal_name, goal_category, output_dir)
    print("Step 0 completed:", result)
