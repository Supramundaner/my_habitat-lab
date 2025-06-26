import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image
import os

def generate_topdown_view(scene_filepath, output_filepath, resolution=(1024, 1024)):
    """
    Generates a top-down orthographic projection of a Habitat scene.

    :param scene_filepath: Path to the .glb scene file.
    :param output_filepath: Path to save the output PNG image.
    :param resolution: A tuple (width, height) for the output image.
    """
    if not os.path.exists(scene_filepath):
        print(f"Error: Scene file not found at {scene_filepath}")
        return

    # --- 1. Configure the simulator ---
    # Create simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_filepath
    backend_cfg.enable_physics = False
    backend_cfg.random_seed = 1

    # --- 2. Configure the Orthographic Sensor ---
    ortho_sensor_spec = habitat_sim.CameraSensorSpec()
    ortho_sensor_spec.uuid = "ortho_rgba_sensor"
    ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    ortho_sensor_spec.resolution = [resolution[1], resolution[0]] # height, width
    ortho_sensor_spec.position = mn.Vector3(0, 0, 0) # Will be overridden by agent state
    # The key part: set sensor to Orthographic
    ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC

    # Create agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [ortho_sensor_spec]
    
    # Create the full configuration
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    
    # --- 3. Instantiate the simulator ---
    sim = habitat_sim.Simulator(cfg)

    # --- 4. Determine camera position and scale based on scene bounds ---
    # Get the bounding box of the navigable area of the scene
    scene_bounds = sim.pathfinder.get_bounds()
    scene_center = (scene_bounds[0] + scene_bounds[1]) / 2.0
    scene_size = scene_bounds[1] - scene_bounds[0]

    # Set camera position to be above the center of the scene
    # We use the Y-axis as "up" in Habitat's default coordinate system
    camera_height = scene_bounds[1][1] + 2.0 # Place camera 2 meters above the highest point
    camera_position = mn.Vector3(scene_center[0], camera_height, scene_center[2])

    # Set the agent's state to move the camera
    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = camera_position
    # Point the camera straight down - use numpy array format for quaternion [x, y, z, w]
    # This represents a -90 degree rotation around X-axis
    agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068])
    agent.set_state(agent_state)

    # --- 5. Set the orthographic projection scale ---
    # The scale should encompass the entire scene
    ortho_scale = max(scene_size[0], scene_size[2]) / 2.0
    ortho_sensor_spec.ortho_scale = ortho_scale
    
    # Reconfigure the sensor with the correct scale
    sim.reconfigure(cfg) 
    # Must set agent state again after reconfiguring
    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = camera_position
    agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068])
    agent.set_state(agent_state)

    # --- 6. Get observation and save the image ---
    observations = sim.get_sensor_observations()
    rgba_img = observations["ortho_rgba_sensor"]

    # Convert to PIL image (drop alpha channel) and save
    image = Image.fromarray(rgba_img[..., :3], "RGB")
    image.save(output_filepath)
    print(f"Top-down view saved to {output_filepath}")

    sim.close()


if __name__ == '__main__':
    # --- USAGE EXAMPLE ---
    # Replace with the actual path to your HM3D scene file
    # You need to download HM3D first from https://aihabitat.org/datasets/hm3d/
    hm3d_scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    output_image_path = "topdown_projection_1.png"
    
    generate_topdown_view(hm3d_scene_path, output_image_path)