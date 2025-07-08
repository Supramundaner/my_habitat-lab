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

def make_ortho_habitat_configuration(scene_path):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

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
        sensor_cfg.parameters['ortho_scale'] = '0.05'
    else:
        sensor_cfg = habitat_sim.CameraSensorSpec()
        sensor_cfg.resolution = [4096, 4096]
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        sensor_cfg.far = 1000.0
        sensor_cfg.near = 0.01
        sensor_cfg.hfov = 90
        sensor_cfg.ortho_scale = 0.05
        sensor_cfg.clear_color = [0., 0., 0., 0.]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_ortho_sim(scene_path):
    sim_cfg = make_ortho_habitat_configuration(scene_path)
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
    sim = robust_load_ortho_sim(glb_path)
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
    hm3d_scene_path = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.glb"
    output_path = "/home/yaoaa/habitat-lab/topdown_floors.png"
    
    result = render_topdown_views(hm3d_scene_path)
    print(f"Generated topdown views with shape: {result.shape}")
    
    # Save the image
    # Convert RGBA to RGB by dropping alpha channel
    if result.shape[-1] == 4:
        result_rgb = result[..., :3]
    else:
        result_rgb = result
    
    # Save using cv2
    cv2.imwrite(output_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    print(f"Topdown views saved to: {output_path}")