"""
多楼层 Topdown 渲染模块
基于 topdown_renderer.py 实现，支持针对特定楼层的单独渲染
"""

import os
import math
import numpy as np
import habitat_sim
from sklearn.cluster import DBSCAN
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any
import json


def get_floor_navigable_extents(hsim: habitat_sim.Simulator, num_points_to_sample: int = 20000):
    """
    估计3D场景中的楼层数量和每层可导航空间的Y范围。
    通过采样随机可导航点并基于高度坐标进行聚类。
    """
    random_navigable_points = []
    for _i in range(num_points_to_sample):
        point = hsim.pathfinder.get_random_navigable_point()
        if np.isnan(point).any() or np.isinf(point).any():
            continue
        random_navigable_points.append(point)
    random_navigable_points = np.array(random_navigable_points)
    
    y_coors = np.around(random_navigable_points[:, 1], decimals=1)
    y_range = y_coors.max() - y_coors.min()
    y_std = y_coors.std()
    
    print(f"Y coordinates range: {y_coors.min():.2f} to {y_coors.max():.2f} (range: {y_range:.2f}m)")
    print(f"Y coordinates std: {y_std:.3f}")
    
    if y_range < 1.5 and y_std < 0.3:
        eps = max(0.5, y_range * 0.8)
        min_samples = max(200, len(y_coors) // 20)
        print(f"Single floor detected: using eps={eps:.2f}, min_samples={min_samples}")
    else:
        eps = 0.45
        min_samples = 500
        print(f"Multi-floor scene: using eps={eps:.2f}, min_samples={min_samples}")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    
    print(f"DBSCAN detected {n_clusters} clusters")
    
    if n_clusters > 1:
        cluster_means = []
        for i in range(n_clusters):
            if i in c_labels:
                mask = c_labels == i
                cluster_mean = y_coors[mask].mean()
                cluster_means.append((i, cluster_mean))
        
        cluster_means.sort(key=lambda x: x[1])
        
        height_threshold = 1.0
        merged_clusters = []
        current_group = [cluster_means[0][0]]
        
        for i in range(1, len(cluster_means)):
            height_diff = cluster_means[i][1] - cluster_means[i-1][1]
            if height_diff < height_threshold:
                current_group.append(cluster_means[i][0])
            else:
                merged_clusters.append(current_group)
                current_group = [cluster_means[i][0]]
        merged_clusters.append(current_group)
        
        print(f"After merging nearby clusters: {len(merged_clusters)} floors")
    
    floor_extents = []
    if clustering.core_sample_indices_.size > 0:
        core_sample_y = y_coors[clustering.core_sample_indices_]
        core_sample_labels = c_labels[clustering.core_sample_indices_]
        
        if n_clusters > 1 and 'merged_clusters' in locals() and len(merged_clusters) < n_clusters:
            for group_idx, cluster_group in enumerate(merged_clusters):
                group_mask = np.isin(core_sample_labels, cluster_group)
                if group_mask.any():
                    group_y = core_sample_y[group_mask]
                    floor_min = group_y.min().item()
                    floor_max = group_y.max().item()
                    floor_mean = group_y.mean().item()
                    floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
                    print(f"Merged Floor {group_idx}: min={floor_min:.2f}, max={floor_max:.2f}, mean={floor_mean:.2f}")
        else:
            for i in range(n_clusters):
                mask = core_sample_labels == i
                if mask.any():
                    cluster_y = core_sample_y[mask]
                    floor_min = cluster_y.min().item()
                    floor_max = cluster_y.max().item()
                    floor_mean = cluster_y.mean().item()
                    floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
                    print(f"Floor {i}: min={floor_min:.2f}, max={floor_max:.2f}, mean={floor_mean:.2f}")
    
    if len(floor_extents) == 0:
        print("No floors detected, creating default floor from all points")
        floor_min = y_coors.min().item()
        floor_max = y_coors.max().item()
        floor_mean = y_coors.mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})

    return floor_extents


def calculate_scene_bounds_from_visuals(sim: habitat_sim.Simulator):
    """
    计算场景的视觉边界框，而不是仅限于可导航区域。
    这是解决坐标偏移问题的关键。
    """
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_bb = scene_root_node.cumulative_bb
    
    x_min, x_max = scene_bb.min[0], scene_bb.max[0]
    z_min, z_max = scene_bb.min[2], scene_bb.max[2]
    
    scene_width = x_max - x_min
    scene_depth = z_max - z_min
    scene_size = max(scene_width, scene_depth)
    
    x_center = (x_min + x_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    return {
        'width': scene_width,
        'depth': scene_depth,
        'max_dimension': scene_size,
        'center': (x_center, z_center),
        'bounds': {
            'x_min': x_min, 'x_max': x_max,
            'z_min': z_min, 'z_max': z_max
        }
    }


def calculate_ortho_scale(scene_size, target_coverage=0.9, safety_margin=1.1):
    """
    根据场景尺寸计算合适的ortho_scale。
    ortho_scale与视野大小成反比。视野大小 ≈ 1.0 / ortho_scale。
    我们希望视野能覆盖大部分场景。
    """
    desired_view_size = scene_size / target_coverage * safety_margin
    calculated_scale = 1.0 / desired_view_size
    
    print(f"  - Target view size: {desired_view_size:.2f}m")
    print(f"  - Calculated ortho_scale: {calculated_scale:.4f}")

    return max(0.01, calculated_scale)


def make_ortho_habitat_configuration(scene_path, ortho_scale=1.0, near=0.01, far=100.0):
    """创建正交投影的Habitat-Sim配置，支持自定义 near 和 far 参数"""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.random_seed = 1

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.uuid = "rgba_camera"
    sensor_cfg.resolution = [4096, 4096]  # 方形传感器，简化计算
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    sensor_cfg.far = far
    sensor_cfg.near = near
    sensor_cfg.hfov = 90
    sensor_cfg.ortho_scale = ortho_scale
    sensor_cfg.clear_color = [0., 0., 0., 0.]
    sensor_cfg.position = [0.0, 0.0, 0.0] 

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_ortho_sim(scene_path, ortho_scale=1.0, near=0.01, far=100.0):
    """稳健地加载正交投影模拟器"""
    sim_cfg = make_ortho_habitat_configuration(scene_path, ortho_scale, near, far)
    hsim = None
    try:
        hsim = habitat_sim.Simulator(sim_cfg)
        if not hsim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            if not hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings):
                raise RuntimeError("Failed to compute navmesh for top-down render")
        return hsim
    except Exception:
        if hsim is not None:
            hsim.close()
        raise


def get_downward_quaternion():
    """获取向下看的四元数旋转 (绕X轴旋转-90度)"""
    return [-0.7071067, 0.0, 0.0, 0.7071067]


def find_target_floor(position: np.ndarray, floor_extents: List[Dict]) -> Optional[int]:
    """
    根据3D位置找到对应的楼层索引
    
    Args:
        position: 3D位置 [x, y, z]
        floor_extents: 楼层范围信息列表
    
    Returns:
        楼层索引，如果未找到则返回最近的楼层
    """
    point_y = position[1]
    closest_floor_idx = None
    min_distance = float('inf')
    
    for i, fext in enumerate(floor_extents):
        # 检查是否在楼层范围内
        if fext['min'] <= point_y <= fext['max']:
            print(f"Position Y ({point_y:.2f}) is within floor {i} range Y=[{fext['min']:.2f}, {fext['max']:.2f}]")
            return i
        
        # 计算到楼层最小高度的距离
        distance_to_min = abs(point_y - fext['min'])
        if distance_to_min < min_distance:
            min_distance = distance_to_min
            closest_floor_idx = i
    
    # 如果找到最近楼层
    if closest_floor_idx is not None:
        closest_fext = floor_extents[closest_floor_idx]
        print(f"Warning: Position Y ({point_y:.2f}) not in any floor range")
        print(f"  - Assigning to closest floor {closest_floor_idx} (Y-min={closest_fext['min']:.2f}, distance={min_distance:.2f}m)")
        return closest_floor_idx
    
    # 没有找到任何楼层的情况
    print(f"Error: Position Y ({point_y:.2f}) not in any floor range and no floors available")
    return None


def render_single_floor_topdown(
    glb_path: str, 
    target_position: np.ndarray,
    custom_ortho_scale: Optional[float] = None, 
    target_coverage: float = 0.9
) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Dict]]:
    """
    为指定位置所在的楼层渲染 topdown 视图
    
    Args:
        glb_path: 场景的 .glb 文件路径
        target_position: 目标位置 [x, y, z]，用于确定渲染哪个楼层
        custom_ortho_scale: 手动指定正交投影比例
        target_coverage: 自动计算 ortho_scale 时，场景在图像中的覆盖率
    
    Returns:
        tuple: (final_image, unprojected_coords, meta_data)
    """
    print("--- Step 1: Initial Scene Analysis ---")
    temp_sim = robust_load_ortho_sim(glb_path, ortho_scale=1.0)
    try:
        scene_info = calculate_scene_bounds_from_visuals(temp_sim)

        if custom_ortho_scale is not None:
            print(f"Using custom orthographic scale: {custom_ortho_scale:.4f}")
            optimal_ortho_scale = custom_ortho_scale
        else:
            optimal_ortho_scale = calculate_ortho_scale(
                scene_info['max_dimension'], target_coverage
            )

        print("\nDetecting all floors...")
        floor_extents = sorted(
            get_floor_navigable_extents(temp_sim), key=lambda x: x['mean']
        )
    finally:
        temp_sim.close()

    if not floor_extents:
        print("Error: No floors were detected in the scene. Cannot render.")
        return None, None, None

    # 查找目标楼层
    target_floor_idx = find_target_floor(target_position, floor_extents)
    if target_floor_idx is None:
        print("Error: Cannot determine target floor from position")
        return None, None, None
    
    target_fext = floor_extents[target_floor_idx]
    x_center, z_center = scene_info['center']
    
    print(f"\n--- Step 2: Rendering Target Floor {target_floor_idx} ---")
    print(f"Floor Y-range: {target_fext['min']:.2f}m to {target_fext['max']:.2f}m")

    # 设置渲染范围
    ceiling_margin = 0.8
    floor_margin = 0.3
    camera_offset = 0.01

    render_volume_top_y = target_fext['max'] + ceiling_margin
    render_volume_bottom_y = target_fext['min'] - floor_margin
    camera_y = render_volume_top_y + camera_offset
    new_near = camera_y - render_volume_top_y
    new_far = camera_y - render_volume_bottom_y
    if new_near <= 0.0:
        new_near = 0.01
    
    print(f"Expanded render volume to: Y={render_volume_bottom_y:.2f}m to {render_volume_top_y:.2f}m")
    print(f"Camera at Y={camera_y:.2f}m, Clipping [near: {new_near:.2f}m, far: {new_far:.2f}m]")

    sim = None
    try:
        print("Creating new simulator instance for this floor...")
        sim = robust_load_ortho_sim(glb_path, optimal_ortho_scale, new_near, new_far)
        
        # 设置智能体位置
        agent_position = [x_center, camera_y, z_center]
        agent_rotation = get_downward_quaternion()
        
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        agent.set_state(new_state, True)
        
        obs = sim.get_sensor_observations()
        final_image = obs['rgba_camera']
        print(f"Floor rendered successfully.")

        # 计算unprojected坐标信息
        unprojected_coords = get_unprojected_world_coords(sim, x_center, z_center, optimal_ortho_scale)
        meta_data = calculate_metadata(unprojected_coords) if unprojected_coords else {}
        meta_data["selected_floor"] = {
            "index": int(target_floor_idx),
            "min": float(target_fext["min"]),
            "max": float(target_fext["max"]),
            "mean": float(target_fext["mean"]),
        }
        
        return final_image, unprojected_coords, meta_data

    finally:
        if sim is not None:
            sim.close()
            print("Simulator instance closed.")


def get_unprojected_world_coords(sim: habitat_sim.Simulator, x_center: float, z_center: float, ortho_scale: float):
    """
    计算正交相机视图的世界坐标范围
    """
    agent = sim.get_agent(0)
    sensor_spec = agent.agent_config.sensor_specifications[0]

    width, height = sensor_spec.resolution
    cam_pos = [x_center, 0, z_center]  # Y不重要，只用XZ

    aspect_ratio = width / height
    view_height_meters = 1.0 / ortho_scale
    view_width_meters = view_height_meters * aspect_ratio
    
    view_half_width = view_width_meters / 2.0
    view_half_height = view_height_meters / 2.0
    
    tl_x = cam_pos[0] - view_half_width
    tl_z = cam_pos[2] - view_half_height
    
    br_x = cam_pos[0] + view_half_width
    br_z = cam_pos[2] + view_half_height

    corner_coords = {
        'top_left': (tl_x, tl_z),
        'top_right': (br_x, tl_z),
        'bottom_left': (tl_x, br_z),
        'bottom_right': (br_x, br_z),
        'center': (cam_pos[0], cam_pos[2]),
        'view_range': (view_width_meters, view_height_meters),
        'image_size': (width, height)
    }
    
    return corner_coords


def calculate_metadata(corner_coords: Dict) -> Dict:
    """根据正确的unprojected坐标计算元数据（像素间距和原点位置）"""
    tl_x, tl_z = corner_coords['top_left']
    view_width_meters, view_height_meters = corner_coords['view_range']
    img_width, img_height = corner_coords['image_size']
    
    spacing_x = view_width_meters / img_width
    spacing_y = view_height_meters / img_height
    
    origin_pixel_x = (0.0 - tl_x) / spacing_x
    origin_pixel_y = (0.0 - tl_z) / spacing_y
    
    meta_data = {
        "image_size": [img_width, img_height],
        "origin_in_pixels": [origin_pixel_x, origin_pixel_y],
        "spacing_in_meters_per_pixel": spacing_x
    }
    return meta_data


class MultiFloorTopdownRenderer:
    """多楼层 Topdown 渲染器类"""
    
    def __init__(self, scene_path: str):
        """
        初始化渲染器
        
        Args:
            scene_path: 场景文件路径
        """
        self.scene_path = scene_path
        self.floor_extents = None
        self.scene_info = None
        self.optimal_ortho_scale = None
        
    def analyze_scene(self, custom_ortho_scale: Optional[float] = None, target_coverage: float = 0.9):
        """
        分析场景并检测楼层
        
        Args:
            custom_ortho_scale: 自定义正交投影比例
            target_coverage: 目标覆盖率
        """
        print("--- Analyzing scene for multi-floor rendering ---")
        temp_sim = robust_load_ortho_sim(self.scene_path, ortho_scale=1.0)
        try:
            self.scene_info = calculate_scene_bounds_from_visuals(temp_sim)

            if custom_ortho_scale is not None:
                self.optimal_ortho_scale = custom_ortho_scale
            else:
                self.optimal_ortho_scale = calculate_ortho_scale(
                    self.scene_info['max_dimension'], target_coverage
                )

            self.floor_extents = sorted(
                get_floor_navigable_extents(temp_sim), key=lambda x: x['mean']
            )
        finally:
            temp_sim.close()
        
        print(f"Scene analysis complete: {len(self.floor_extents)} floors detected")
        print(f"Optimal ortho_scale: {self.optimal_ortho_scale:.4f}")
    
    def render_floor_by_position(self, position: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Dict]]:
        """
        根据位置渲染对应楼层
        
        Args:
            position: 3D位置 [x, y, z]
        
        Returns:
            tuple: (final_image, unprojected_coords, meta_data)
        """
        if self.floor_extents is None:
            self.analyze_scene()
        
        return render_single_floor_topdown(self.scene_path, position, self.optimal_ortho_scale)
    
    def render_floor_by_index(self, floor_index: int) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Dict]]:
        """
        根据楼层索引渲染
        
        Args:
            floor_index: 楼层索引
        
        Returns:
            tuple: (final_image, unprojected_coords, meta_data)
        """
        if self.floor_extents is None:
            self.analyze_scene()
        
        if not (0 <= floor_index < len(self.floor_extents)):
            raise ValueError(f"Invalid floor index {floor_index}. Available floors: 0-{len(self.floor_extents)-1}")
        
        # 创建一个在该楼层内的虚拟位置
        target_fext = self.floor_extents[floor_index]
        dummy_position = np.array([self.scene_info['center'][0], target_fext['mean'], self.scene_info['center'][1]])
        
        return render_single_floor_topdown(self.scene_path, dummy_position, self.optimal_ortho_scale)
    
    def get_floor_count(self) -> int:
        """获取楼层数量"""
        if self.floor_extents is None:
            self.analyze_scene()
        return len(self.floor_extents)
    
    def get_floor_info(self) -> List[Dict]:
        """获取楼层信息"""
        if self.floor_extents is None:
            self.analyze_scene()
        return self.floor_extents.copy()
