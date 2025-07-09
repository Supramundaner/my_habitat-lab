import os
import cv2
import math
import tqdm
import glob
import imageio
import argparse
import numpy as np
import habitat_sim
import multiprocessing as mp
from sklearn.cluster import DBSCAN
def get_floor_navigable_extents(
    hsim: habitat_sim.Simulator, num_points_to_sample: int = 20000
) :
    """
    Function to estimate the number of floors in a 3D scene and the Y extents
    of the navigable space on each floor. It samples a random number
    of navigable points and clusters them based on their height coordinate.
    Each cluster corresponds to a floor, and the points within a cluster
    determine the extents of the navigable surfaces in a floor.
    """
    # randomly sample navigable points
    random_navigable_points = []
    for _i in range(num_points_to_sample):
        point = hsim.pathfinder.get_random_navigable_point()
        if np.isnan(point).any() or np.isinf(point).any():
            continue
        random_navigable_points.append(point)
    random_navigable_points = np.array(random_navigable_points)
    # cluster the rounded y_coordinates using DBScan
    y_coors = np.around(random_navigable_points[:, 1], decimals=1)
    clustering = DBSCAN(eps=0.2, min_samples=500).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    # estimate floor extents
    floor_extents = []
    core_sample_y = y_coors[clustering.core_sample_indices_]
    core_sample_labels = c_labels[clustering.core_sample_indices_]
    for i in range(n_clusters):
        floor_min = core_sample_y[core_sample_labels == i].min().item()
        floor_max = core_sample_y[core_sample_labels == i].max().item()
        floor_mean = core_sample_y[core_sample_labels == i].mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})

    return floor_extents


TOPDOWN_WIDTH = 1280

def calculate_scene_bounds(hsim: habitat_sim.Simulator):
    """
    Calculate the bounds of the navigable area in the scene to determine
    appropriate orthographic scale for rendering.
    """
    navmesh_vertices = np.array(hsim.pathfinder.build_navmesh_vertices())
    
    # Calculate bounds in X and Z dimensions (Y is height)
    x_min, x_max = navmesh_vertices[:, 0].min(), navmesh_vertices[:, 0].max()
    z_min, z_max = navmesh_vertices[:, 2].min(), navmesh_vertices[:, 2].max()
    
    # Calculate scene dimensions
    scene_width = x_max - x_min
    scene_depth = z_max - z_min
    scene_size = max(scene_width, scene_depth)
    
    return {
        'width': scene_width,
        'depth': scene_depth,
        'max_dimension': scene_size,
        'bounds': {
            'x_min': x_min, 'x_max': x_max,
            'z_min': z_min, 'z_max': z_max
        }
    }


def calculate_ortho_scale(scene_size, target_coverage=0.9):
    """
    Calculate appropriate orthographic scale based on scene size.
    
    Args:
        scene_size: Maximum dimension of the scene in meters
        target_coverage: Fraction of the image that should be covered by the scene (0.0-1.0)
    
    Returns:
        ortho_scale: Orthographic scale value
    """
    # The orthographic scale determines how much of the world space fits in the view
    # For a 25m scene with ortho_scale=0.05, we want to maintain this proportion
    # Base calculation: if 25m scene uses 0.05 scale
    base_scene_size = 18.0  # meters
    base_ortho_scale = 0.05
    
    # Scale proportionally, but account for target coverage
    calculated_scale = (base_ortho_scale * base_scene_size) / (scene_size / target_coverage)
    
    # Add some safety margin and clamp to reasonable bounds
    safety_margin = 1.2
    ortho_scale = calculated_scale / safety_margin
    
    # Clamp to reasonable bounds
    ortho_scale = max(0.01, min(0.2, ortho_scale))
    
    return ortho_scale


def make_ortho_habitat_configuration(scene_path, ortho_scale=None):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    # If no ortho_scale provided, use default
    if ortho_scale is None:
        ortho_scale = 0.05

    if habitat_sim.__version__ == "0.1.7":
        # agent configuration
        sensor_cfg = habitat_sim.SensorSpec()
        sensor_cfg.resolution = [4096, 4096]
        # Reference: src/esp/bindings/SensorBindings.cpp
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        sensor_cfg.parameters['far'] = '1000'
        sensor_cfg.parameters['near'] = '0.01'
        sensor_cfg.parameters['fov'] = '90'
        sensor_cfg.parameters['ortho_scale'] = str(ortho_scale)
    else:
        sensor_cfg = habitat_sim.CameraSensorSpec()
        sensor_cfg.resolution = [4096, 4096]
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        sensor_cfg.far = 1000.0
        sensor_cfg.near = 0.01
        sensor_cfg.hfov = 90
        sensor_cfg.ortho_scale = ortho_scale
        sensor_cfg.clear_color = [0., 0., 0., 0.]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_ortho_sim(scene_path, ortho_scale=None):
    sim_cfg = make_ortho_habitat_configuration(scene_path, ortho_scale)
    hsim = habitat_sim.Simulator(sim_cfg)
    if not hsim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings)
    return hsim


def get_downward_quaternion():
    """
    Given a unit vector u = ux i + uy j + uz k, and a rotation angle theta
    the corresponding quaternion is as defined follows:
    q = cos(theta/2) + (ux i + uy j + uz k) sin(theta/2)

    To get downward rotation, rotate about either i or k
    since j is upward in habitat.

    By default, the agent faces -Z. To convert this vector to downward,
    rotate by -90 degree along X.

    Output format: 
        q - [x, y, z, w] elements of a unit quaternion

    Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """
    # -90 degree rotation about x
    q = [-0.7071067, 0.0, 0.0, 0.7071067]
    return q

def render_topdown_views(glb_path):
    # First, load simulator with default scale to calculate scene bounds
    temp_sim = robust_load_ortho_sim(glb_path)
    
    # Calculate scene bounds and appropriate orthographic scale
    scene_info = calculate_scene_bounds(temp_sim)
    optimal_ortho_scale = calculate_ortho_scale(scene_info['max_dimension'])
    
    print(f"Scene dimensions: {scene_info['width']:.2f}m x {scene_info['depth']:.2f}m")
    print(f"Maximum dimension: {scene_info['max_dimension']:.2f}m")
    print(f"Calculated orthographic scale: {optimal_ortho_scale:.4f}")
    
    temp_sim.close()
    
    # Now load simulator with optimal orthographic scale
    sim = robust_load_ortho_sim(glb_path, optimal_ortho_scale)
    
    # Get floor extents
    floor_extents = get_floor_navigable_extents(sim)
    floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
    navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
    floor_images = []
    scene_cent = navmesh_vertices.mean(axis=0).tolist()
    for fext in floor_extents:
        # Get navmesh vertices from current floor
        mask = (
            (navmesh_vertices[:, 1] <= fext['max'] + 0.25) & \
            (navmesh_vertices[:, 1] >= fext['min'] - 0.25)
        )
        fcent = np.median(navmesh_vertices[mask, :], axis=0).tolist() # (3, )
        # Set agent state
        agent_position = [scene_cent[0], fcent[1] + 1.0, scene_cent[2]]
        agent_rotation = get_downward_quaternion()
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        new_state.sensor_states = {}
        agent.set_state(new_state, True)
        # Get observations
        obs = sim.get_sensor_observations()
        floor_images.append(obs['rgba_camera'])

    # Concatenate images vertically
    floor_images = np.concatenate(floor_images, axis=0)
    
    sim.close()

    return floor_images


def render_topdown_views_with_custom_scale(glb_path, custom_ortho_scale=None, target_coverage=0.9):
    """
    Render topdown views with custom orthographic scale or target coverage.
    
    Args:
        glb_path: Path to the scene file
        custom_ortho_scale: If provided, use this specific orthographic scale
        target_coverage: Fraction of image that should be covered by scene (0.0-1.0)
    """
    if custom_ortho_scale is not None:
        print(f"Using custom orthographic scale: {custom_ortho_scale:.4f}")
        sim = robust_load_ortho_sim(glb_path, custom_ortho_scale)
    else:
        # Calculate optimal scale based on scene size
        temp_sim = robust_load_ortho_sim(glb_path)
        scene_info = calculate_scene_bounds(temp_sim)
        optimal_ortho_scale = calculate_ortho_scale(scene_info['max_dimension'], target_coverage)
        
        print(f"Scene dimensions: {scene_info['width']:.2f}m x {scene_info['depth']:.2f}m")
        print(f"Maximum dimension: {scene_info['max_dimension']:.2f}m")
        print(f"Target coverage: {target_coverage:.1%}")
        print(f"Calculated orthographic scale: {optimal_ortho_scale:.4f}")
        
        temp_sim.close()
        sim = robust_load_ortho_sim(glb_path, optimal_ortho_scale)
    
    # Get floor extents
    floor_extents = get_floor_navigable_extents(sim)
    floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
    navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
    floor_images = []
    scene_cent = navmesh_vertices.mean(axis=0).tolist()
    
    for fext in floor_extents:
        # Get navmesh vertices from current floor
        mask = (
            (navmesh_vertices[:, 1] <= fext['max'] + 0.25) & \
            (navmesh_vertices[:, 1] >= fext['min'] - 0.25)
        )
        fcent = np.median(navmesh_vertices[mask, :], axis=0).tolist() # (3, )
        # Set agent state
        agent_position = [scene_cent[0], fcent[1] + 1.0, scene_cent[2]]
        agent_rotation = get_downward_quaternion()
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        new_state.sensor_states = {}
        agent.set_state(new_state, True)
        # Get observations
        obs = sim.get_sensor_observations()
        floor_images.append(obs['rgba_camera'])

    # Concatenate images vertically
    floor_images = np.concatenate(floor_images, axis=0)
    
    sim.close()

    return floor_images


if __name__ == '__main__':
    # --- USAGE EXAMPLE ---
    # Replace with the actual path to your HM3D scene file
    # You need to download HM3D first from https://aihabitat.org/datasets/hm3d/
    #hm3d_scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    #hm3d_scene_path = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.glb"
    hm3d_scene_path = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.glb"
    output_path = "/home/yaoaa/habitat-lab/big_topdown_floors.png"
    
    # Option 1: Automatic dynamic scaling (recommended)
    print("=== Rendering with automatic dynamic scaling ===")
    result = render_topdown_views(hm3d_scene_path)
    print(f"Generated topdown views with shape: {result.shape}")
    
    # Option 2: Custom coverage (e.g., 80% of image should be covered by scene)
    # result = render_topdown_views_with_custom_scale(hm3d_scene_path, target_coverage=0.8)
    
    # Option 3: Manual orthographic scale override
    # result = render_topdown_views_with_custom_scale(hm3d_scene_path, custom_ortho_scale=0.03)
    
    # Save the image
    # Convert RGBA to RGB by dropping alpha channel
    if result.shape[-1] == 4:
        result_rgb = result[..., :3]
    else:
        result_rgb = result
    
    # Save using cv2
    cv2.imwrite(output_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    print(f"Topdown views saved to: {output_path}")