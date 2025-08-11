import os
import cv2
import math
import numpy as np
import habitat_sim
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont
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


def make_ortho_habitat_configuration(scene_path, ortho_scale=1.0):
    """创建正交投影的Habitat-Sim配置"""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [4096, 4096] # 方形传感器，简化计算
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    # 【注意】这里的 near 和 far 只是初始值，稍后会为每个楼层动态修改
    sensor_cfg.far = 100.0
    sensor_cfg.near = 0.01
    sensor_cfg.hfov = 90
    sensor_cfg.ortho_scale = ortho_scale
    sensor_cfg.clear_color = [0., 0., 0., 0.]
    sensor_cfg.position = [0.0, 0.0, 0.0] 

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_ortho_sim(scene_path, ortho_scale=1.0):
    """稳健地加载正交投影模拟器"""
    sim_cfg = make_ortho_habitat_configuration(scene_path, ortho_scale)
    hsim = habitat_sim.Simulator(sim_cfg)
    if not hsim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings)
    return hsim


def get_downward_quaternion():
    """获取向下看的四元数旋转 (绕X轴旋转-90度)"""
    return [-0.7071067, 0.0, 0.0, 0.7071067]


def get_unprojected_world_coords(sim: habitat_sim.Simulator):
    """
    使用正确的公式计算正交相机视图的世界坐标范围。
    """
    agent = sim.get_agent(0)
    camera_sensor = agent.scene_node.node_sensor_suite.get("rgba_camera")
    sensor_spec = agent.agent_config.sensor_specifications[0]

    width, height = sensor_spec.resolution
    ortho_scale = sensor_spec.ortho_scale
    cam_pos = camera_sensor.render_camera.node.absolute_translation

    aspect_ratio = width / height
    view_height_meters = 1.0 / ortho_scale
    view_width_meters = view_height_meters * aspect_ratio
    
    view_half_width = view_width_meters / 2.0
    view_half_height = view_height_meters / 2.0
    
    tl_x = cam_pos[0] - view_half_width
    tl_z = cam_pos[2] - view_half_height
    
    br_x = cam_pos[0] + view_half_width
    br_z = cam_pos[2] + view_half_height

    grid_interval = get_coordinate_grid_interval(max(view_width_meters, view_height_meters))

    corner_coords = {
        'top_left': (tl_x, tl_z),
        'top_right': (br_x, tl_z),
        'bottom_left': (tl_x, br_z),
        'bottom_right': (br_x, br_z),
        'center': (cam_pos[0], cam_pos[2]),
        'view_range': (view_width_meters, view_height_meters),
        'grid_interval': grid_interval,
        'image_size': (width, height)
    }
    
    return corner_coords


def get_coordinate_grid_interval(scene_size):
    """根据场景大小确定合适的坐标网格间隔"""
    if scene_size <= 3: return 0.5
    elif scene_size <= 8: return 1.0
    elif scene_size <= 15: return 2.0
    elif scene_size <= 30: return 5.0
    else: return 10.0


def print_corner_coordinates(corner_coords):
    """输出图像角坐标信息"""
    print("\n=== Unprojected World Coordinate Info ===")
    print(f"Top-Left     : X={corner_coords['top_left'][0]:.2f}m, Z={corner_coords['top_left'][1]:.2f}m")
    print(f"Bottom-Right : X={corner_coords['bottom_right'][0]:.2f}m, Z={corner_coords['bottom_right'][1]:.2f}m")
    print(f"Image Center : X={corner_coords['center'][0]:.2f}m, Z={corner_coords['center'][1]:.2f}m")
    print(f"View Range   : {corner_coords['view_range'][0]:.2f}m × {corner_coords['view_range'][1]:.2f}m")
    print(f"Grid Interval: {corner_coords['grid_interval']:.1f}m")
    print("=" * 40)


def draw_coordinate_system(image: np.ndarray, corner_coords: dict, ortho_scale: float):
    """
    在俯视图上绘制基于unprojected坐标的世界坐标系。
    此函数现在完全依赖于 corner_coords 提供的精确边界。
    """
    pil_image = Image.fromarray(image[..., :3], "RGB")
    tl_x, tl_z = corner_coords['top_left']
    tr_x, _ = corner_coords['top_right']
    _, bl_z = corner_coords['bottom_left']
    grid_interval = corner_coords['grid_interval']
    
    img_width, img_height = pil_image.size
    margin_left, margin_bottom, margin_top, margin_right = 80, 60, 40, 40
    new_width = img_width + margin_left + margin_right
    new_height = img_height + margin_top + margin_bottom
    new_image = Image.new('RGB', (new_width, new_height), (20, 20, 20))
    new_image.paste(pil_image, (margin_left, margin_top))
    
    draw = ImageDraw.Draw(new_image)
    try:
        font_large = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_medium = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 10)
    except IOError:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    grid_color, major_grid_color, border_color, text_color = (80, 80, 80), (120, 120, 120), (255, 255, 255), (255, 255, 255)
    
    img_area_x0, img_area_y0 = margin_left, margin_top
    img_area_x1, img_area_y1 = margin_left + img_width, margin_top + img_height
    
    def world_to_pixel(world_x, world_z):
        fx = (world_x - tl_x) / (tr_x - tl_x)
        fz = (world_z - tl_z) / (bl_z - tl_z)
        pixel_x = img_area_x0 + fx * img_width
        pixel_y = img_area_y0 + fz * img_height
        return int(pixel_x), int(pixel_y)

    x_start = math.ceil(tl_x / grid_interval) * grid_interval
    x_current = x_start
    while x_current <= tr_x:
        pixel_x, _ = world_to_pixel(x_current, tl_z)
        if img_area_x0 <= pixel_x <= img_area_x1:
            is_major = abs(x_current % (grid_interval * 2)) < 0.01
            draw.line([(pixel_x, img_area_y0), (pixel_x, img_area_y1)], fill=major_grid_color if is_major else grid_color, width=1)
            draw.line([(pixel_x, img_area_y1), (pixel_x, img_area_y1 + 5)], fill=border_color, width=1)
            label_text = f"{x_current:.1f}"
            bbox = draw.textbbox((0, 0), label_text, font=font_medium)
            draw.text((pixel_x - (bbox[2]-bbox[0])/2, img_area_y1 + 8), label_text, fill=text_color, font=font_medium)
        x_current += grid_interval
    
    z_start = math.ceil(tl_z / grid_interval) * grid_interval
    z_current = z_start
    while z_current <= bl_z:
        _, pixel_y = world_to_pixel(tl_x, z_current)
        if img_area_y0 <= pixel_y <= img_area_y1:
            is_major = abs(z_current % (grid_interval * 2)) < 0.01
            draw.line([(img_area_x0, pixel_y), (img_area_x1, pixel_y)], fill=major_grid_color if is_major else grid_color, width=1)
            draw.line([(img_area_x0 - 5, pixel_y), (img_area_x0, pixel_y)], fill=border_color, width=1)
            label_text = f"{z_current:.1f}"
            bbox = draw.textbbox((0, 0), label_text, font=font_medium)
            draw.text((img_area_x0 - (bbox[2]-bbox[0]) - 8, pixel_y - (bbox[3]-bbox[1])/2), label_text, fill=text_color, font=font_medium)
        z_current += grid_interval
    
    draw.rectangle([img_area_x0-1, img_area_y0-1, img_area_x1, img_area_y1], outline=border_color, width=1)
    
    x_label = "X (meters)"
    bbox = draw.textbbox((0, 0), x_label, font=font_large)
    draw.text(((new_width - (bbox[2]-bbox[0])) / 2, new_height - 25), x_label, fill=text_color, font=font_large)
    
    z_label = "Z (meters)"
    temp_img = Image.new('RGBA', (200, 30), (0,0,0,0)); temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((0, 0), z_label, fill=text_color, font=font_large)
    rotated = temp_img.rotate(90, expand=True)
    new_image.paste(rotated, (15, int((new_height - rotated.height) / 2)), rotated)
    
    origin_pixel_x, origin_pixel_y = world_to_pixel(0, 0)
    if img_area_x0 <= origin_pixel_x <= img_area_x1 and img_area_y0 <= origin_pixel_y <= img_area_y1:
        draw.ellipse([origin_pixel_x-5, origin_pixel_y-5, origin_pixel_x+5, origin_pixel_y+5], fill=(255, 255, 0), outline=border_color, width=1)
        draw.text((origin_pixel_x+8, origin_pixel_y-8), "Origin (0,0)", fill=(255, 255, 0), font=font_small)
    
    scale_info = f"Grid: {grid_interval}m | Ortho Scale: {ortho_scale:.4f} | View: {corner_coords['view_range'][0]:.1f}m × {corner_coords['view_range'][1]:.1f}m"
    draw.text((img_area_x0 + 5, 5), scale_info, fill=text_color, font=font_small)
    
    return new_image


def calculate_metadata(corner_coords: dict):
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


# ==============================================================================
# 主渲染流程 (已修改)
# ==============================================================================

def render_topdown_view(glb_path, target_floor, custom_ortho_scale=None, target_coverage=0.9, draw_coordinates=False):
    """
    【修改版】为指定的单个楼层创建俯视图。
    可以通过楼层索引（整数）或场景中的一个3D坐标点来指定目标楼层。

    Args:
        glb_path (str): 场景的 .glb 文件路径。
        target_floor (int or list/tuple): 目标楼层。
            - 如果是 int: 楼层的索引 (0 代表第一层, 1 代表第二层, etc.)。
            - 如果是 list/tuple: 场景中的一个3D坐标 `[x, y, z]`，函数会自动找到该点所在的楼层。
        custom_ortho_scale (float, optional): 手动指定正交投影比例。默认为 None (自动计算)。
        target_coverage (float, optional): 自动计算 ortho_scale 时，场景在图像中的覆盖率。
        draw_coordinates (bool, optional): 是否在最终图像上绘制坐标系。
    
    Returns:
        tuple: (final_image, unprojected_coords, meta_data)
               - final_image: 渲染出的图像 (numpy array)。
               - unprojected_coords: 图像的世界坐标信息 (dict)。
               - meta_data: 图像的元数据 (dict)。
    """
    print("--- Step 1: Initial Scene Analysis (One-Time) ---")
    temp_sim = robust_load_ortho_sim(glb_path, ortho_scale=1.0) 
    scene_info = calculate_scene_bounds_from_visuals(temp_sim)
    if custom_ortho_scale is not None:
        print(f"Using custom orthographic scale: {custom_ortho_scale:.4f}")
        optimal_ortho_scale = custom_ortho_scale
    else:
        print(f"Scene visual dimensions: {scene_info['width']:.2f}m x {scene_info['depth']:.2f}m")
        print(f"Maximum visual dimension: {scene_info['max_dimension']:.2f}m")
        print(f"Scene visual center: X={scene_info['center'][0]:.2f}, Z={scene_info['center'][1]:.2f}")
        print(f"Calculating optimal orthographic scale...")
        optimal_ortho_scale = calculate_ortho_scale(scene_info['max_dimension'], target_coverage)
    
    print("\nDetecting all floors...")
    floor_extents = get_floor_navigable_extents(temp_sim)
    floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
    temp_sim.close()

    if not floor_extents:
        print("Error: No floors were detected in the scene. Cannot render.")
        return None, None, None

    # --- 新增：选择目标楼层 ---
    target_fext = None
    if isinstance(target_floor, int):
        if 0 <= target_floor < len(floor_extents):
            target_fext = floor_extents[target_floor]
            print(f"\nSuccessfully targeted floor by index: {target_floor}")
        else:
            raise ValueError(f"Invalid floor index {target_floor}. Scene has {len(floor_extents)} floors (indices 0 to {len(floor_extents)-1}).")
    elif isinstance(target_floor, (list, tuple, np.ndarray)) and len(target_floor) == 3:
        point_y = target_floor[1]
        for i, fext in enumerate(floor_extents):
            # 使用楼层的 min 和 max Y坐标来判断点是否在该楼层内
            if fext['min'] <= point_y <= fext['max']:
                target_fext = fext
                print(f"\nSuccessfully targeted floor {i} using coordinate {target_floor}.")
                print(f"Point Y ({point_y:.2f}) is within floor range Y=[{fext['min']:.2f}, {fext['max']:.2f}]")
                break
        if target_fext is None:
            raise ValueError(f"Point {target_floor} (Y={point_y:.2f}) does not fall within any detected navigable floor extents.")
    else:
        raise TypeError("`target_floor` must be an integer (floor index) or a 3-element list/tuple (world coordinate).")
    # --- 楼层选择结束 ---
    
    x_center, z_center = scene_info['center']
    final_image = None
    unprojected_coords = None
    
    print(f"\n--- Step 2: Rendering Selected Floor ---")

    ceiling_margin = 0.8
    floor_margin = 0.3
    camera_offset = 0.01

    sim = None
    try:
        fext = target_fext # 使用我们选择的楼层
        print(f"--- Processing Target Floor (Navigable Y-range: {fext['min']:.2f}m to {fext['max']:.2f}m) ---")

        render_volume_top_y = fext['max'] + ceiling_margin
        render_volume_bottom_y = fext['min'] - floor_margin
        camera_y = render_volume_top_y + camera_offset
        new_near = camera_y - render_volume_top_y
        new_far = camera_y - render_volume_bottom_y
        if new_near <= 0.0: new_near = 0.01
        
        print(f"  - Expanded render volume to: Y={render_volume_bottom_y:.2f}m to {render_volume_top_y:.2f}m")
        print(f"  - Camera at Y={camera_y:.2f}m, Clipping [near: {new_near:.2f}m, far: {new_far:.2f}m]")

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = glb_path
        sensor_cfg = habitat_sim.CameraSensorSpec()
        sensor_cfg.resolution = [4096, 4096]
        sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
        sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        sensor_cfg.ortho_scale = optimal_ortho_scale
        sensor_cfg.clear_color = [0., 0., 0., 0.]
        sensor_cfg.position = [0.0, 0.0, 0.0] 
        sensor_cfg.near = new_near
        sensor_cfg.far = new_far
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_cfg]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        print("  - Creating new simulator instance for this floor...")
        sim = habitat_sim.Simulator(sim_cfg)
        agent_position = [x_center, camera_y, z_center]
        agent_rotation = get_downward_quaternion()
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        agent.set_state(new_state, True)
        obs = sim.get_sensor_observations()
        
        # 直接获取图像，不再需要列表和拼接
        final_image = obs['rgba_camera']
        print(f"  - Floor rendered successfully.")

        unprojected_coords = get_unprojected_world_coords(sim)
        print_corner_coordinates(unprojected_coords)

    finally:
        if sim is not None:
            sim.close()
            print("  - Simulator instance closed.")

    print("\n--- Step 3: Processing Final Image and Metadata ---")
    if final_image is None:
        print("Warning: Failed to render the target floor.")
        res = [4096, 4096]
        final_image = np.zeros((res[0], res[1], 4), dtype=np.uint8)

    meta_data = calculate_metadata(unprojected_coords) if unprojected_coords else {}
    if meta_data:
      print(f"\nCalculated Metadata:")
      print(f"  Pixel spacing: {meta_data['spacing_in_meters_per_pixel']:.6f} m/pixel")
      print(f"  World Origin (0,0) at pixel: (x={meta_data['origin_in_pixels'][0]:.2f}, y={meta_data['origin_in_pixels'][1]:.2f})")
    
    if draw_coordinates and unprojected_coords:
        print("\n--- Step 4: Drawing Coordinate System ---")
        annotated_pil_image = draw_coordinate_system(final_image, unprojected_coords, optimal_ortho_scale)
        # Convert PIL image back to numpy array with correct channel order
        final_image = np.array(annotated_pil_image)
        # If it's RGB from PIL, we keep it as RGB (don't convert to BGR here)
        print(f"  - Coordinate system drawn, image shape: {final_image.shape}")

    return final_image, unprojected_coords, meta_data


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 配置 ---
    try:
        # 请确保这里的路径是正确的
        # hm3d_scene_path = "/path/to/your/scene.glb" 
        hm3d_scene_path = "/home/yaoaa/habitat-lab/processed_data/scenes_subset/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
        if not os.path.exists(hm3d_scene_path):
             raise FileNotFoundError(f"Scene file not found at: {hm3d_scene_path}. Please check the path.")
    except Exception as e:
        print(e)
        exit(1)

    output_dir = "output_single_floor"
    os.makedirs(output_dir, exist_ok=True)
    
    scene_id = os.path.basename(os.path.dirname(hm3d_scene_path))
    
    # --- 执行示例 1: 按楼层索引渲染 (渲染第一层, index=0) ---
    print("\n" + "="*80)
    print("### RUNNING EXAMPLE 1: RENDER BY FLOOR INDEX ###")
    print("="*80)
    try:
        target_floor_index = 0  # 0 代表第一层
        result_image_idx, _, meta_data_idx = render_topdown_view(
            hm3d_scene_path, 
            target_floor=target_floor_index,
            draw_coordinates=True
        )
        
        if result_image_idx is not None and result_image_idx.size > 0:
            output_image_path = os.path.join(output_dir, f"{scene_id}_floor_{target_floor_index}.png")
            output_metadata_path = os.path.join(output_dir, f"{scene_id}_floor_{target_floor_index}_metadata.json")
            
            result_bgr = cv2.cvtColor(result_image_idx, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, result_bgr)
            print(f"\nFloor {target_floor_index} view saved to: {output_image_path}")
            
            with open(output_metadata_path, "w") as f:
                json.dump(meta_data_idx, f, indent=4)
            print(f"Metadata for floor {target_floor_index} saved to: {output_metadata_path}")
    except (ValueError, TypeError) as e:
        print(f"Error during rendering by index: {e}")

 
    # --- 执行示例 2: 按3D坐标点渲染 ---
    # 我们使用一个在场景中可能存在的点。对于 TEEsavR23oF.glb, y=-3.0 应该在地下室。
    # 如果您的场景不同，请修改此坐标。
    print("### RUNNING EXAMPLE 2: RENDER BY 3D COORDINATE ###")
    try:
        target_point = [
      0,0.81575,0
    ]
        result_image_pt, _, meta_data_pt = render_topdown_view(
            hm3d_scene_path, 
            target_floor=target_point,
            draw_coordinates=True
        )

        if result_image_pt is not None and result_image_pt.size > 0:
            output_image_path = os.path.join(output_dir, f"{scene_id}_floor_by_point.png")
            output_metadata_path = os.path.join(output_dir, f"{scene_id}_floor_by_point_metadata.json")
            
            result_bgr = cv2.cvtColor(result_image_pt, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, result_bgr)
            print(f"\nFloor view (from point {target_point}) saved to: {output_image_path}")
            
            with open(output_metadata_path, "w") as f:
                json.dump(meta_data_pt, f, indent=4)
            print(f"Metadata for floor (from point) saved to: {output_metadata_path}")
    except (ValueError, TypeError) as e:
        print(f"Error during rendering by point: {e}")