import os
import cv2
import math
import numpy as np
import habitat_sim
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont
import json

# ==============================================================================
# 楼层和场景边界计算 (未修改，这些函数是前置步骤，逻辑正确)
# ==============================================================================

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
        eps = 0.3
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


# ==============================================================================
# 核心修正：使用视觉边界框代替NavMesh边界框
# ==============================================================================

def calculate_scene_bounds_from_visuals(sim: habitat_sim.Simulator):
    """
    【已修正】计算场景的视觉边界框，而不是仅限于可导航区域。
    这是解决坐标偏移问题的关键。
    """
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_bb = scene_root_node.cumulative_bb
    
    # --- 核心修正点 ---
    # 旧代码 (错误): if not scene_bb.is_valid():
    # 新代码 (正确):


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

# --- 核心修正 1: 重写 calculate_ortho_scale 使其更直观 ---
def calculate_ortho_scale(scene_size, target_coverage=0.9, safety_margin=1.1):
    """
    根据场景尺寸计算合适的ortho_scale。
    ortho_scale与视野大小成反比。视野大小 ≈ 1.0 / ortho_scale。
    我们希望视野能覆盖大部分场景。
    """
    # 目标视野大小应略大于场景尺寸
    desired_view_size = scene_size / target_coverage * safety_margin
    
    # 从公式: desired_view_size = 1.0 / ortho_scale 推导出 ortho_scale
    # 注意：这里假设为方形传感器，宽度和高度的计算是相同的
    calculated_scale = 1.0 / desired_view_size
    
    print(f"  - Target view size: {desired_view_size:.2f}m")
    print(f"  - Calculated ortho_scale: {calculated_scale:.4f}")

    # 返回一个合理范围内的值
    return max(0.01, calculated_scale)


def make_ortho_habitat_configuration(scene_path, ortho_scale=1.0):
    """创建正交投影的Habitat-Sim配置"""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [2048, 2048] # 方形传感器，简化计算
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    sensor_cfg.far = 1000.0
    sensor_cfg.near = 0.01
    # hfov 在正交投影中不使用，但最好设置
    sensor_cfg.hfov = 90
    sensor_cfg.ortho_scale = ortho_scale
    sensor_cfg.clear_color = [0., 0., 0., 0.]

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

# ==============================================================================
# 核心修改部分：使用正确的公式获取世界坐标
# ==============================================================================

# --- 核心修正 2: 重写 get_unprojected_world_coords ---
def get_unprojected_world_coords(sim: habitat_sim.Simulator):
    """
    使用正确的公式计算正交相机视图的世界坐标范围。
    """
    agent = sim.get_agent(0)
    camera_sensor = agent.scene_node.node_sensor_suite.get("rgba_camera")
    sensor_spec = agent.agent_config.sensor_specifications[0]

    # 获取相机参数
    width, height = sensor_spec.resolution
    ortho_scale = sensor_spec.ortho_scale
    cam_pos = camera_sensor.render_camera.node.absolute_translation

    # 正确的计算方法：
    # 视野高度(米) = 1.0 / ortho_scale
    # 视野宽度(米) = (1.0 / ortho_scale) * (图像宽度 / 图像高度)
    aspect_ratio = width / height
    view_height_meters = 1.0 / ortho_scale
    view_width_meters = view_height_meters * aspect_ratio
    
    # 视野范围的一半
    view_half_width = view_width_meters / 2.0
    view_half_height = view_height_meters / 2.0
    
    # 计算世界坐标中的四个角点
    # 俯视图中，图像的Y轴对应世界的Z轴，图像的X轴对应世界的X轴
    # 图像上方是Z值较小的方向
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


# --- 核心修正 3: 更新 draw_coordinate_system 以使用新坐标 ---
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
    
    # --- 核心：基于unprojected角点进行坐标转换 ---
    def world_to_pixel(world_x, world_z):
        # 计算在世界坐标范围内的相对位置 (0.0 to 1.0)
        fx = (world_x - tl_x) / (tr_x - tl_x)
        fz = (world_z - tl_z) / (bl_z - tl_z) # Z轴方向是从上到下增加
        
        # 将相对位置映射到画布上的像素坐标
        pixel_x = img_area_x0 + fx * img_width
        pixel_y = img_area_y0 + fz * img_height
        return int(pixel_x), int(pixel_y)
    
    # ... (其余绘图代码逻辑不变，因为它依赖于正确的world_to_pixel)
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

# --- 核心修正 4: 重写 calculate_metadata ---
def calculate_metadata(corner_coords: dict):
    """根据正确的unprojected坐标计算元数据（像素间距和原点位置）"""
    tl_x, tl_z = corner_coords['top_left']
    view_width_meters, view_height_meters = corner_coords['view_range']
    img_width, img_height = corner_coords['image_size']
    
    # 正确的像素间距 (米/像素)
    spacing_x = view_width_meters / img_width
    spacing_y = view_height_meters / img_height # 理论上 spacing_x 和 spacing_y 应该相等
    
    # 世界坐标原点(0,0)在图像中的像素位置
    # (0 - tl_x) 是世界原点距离图像左边缘的物理距离
    # 除以间距，得到像素距离
    origin_pixel_x = (0.0 - tl_x) / spacing_x
    origin_pixel_y = (0.0 - tl_z) / spacing_y
    
    meta_data = {
        "image_size": [img_width, img_height],
        "origin_in_pixels": [origin_pixel_x, origin_pixel_y],
        "spacing_in_meters_per_pixel": spacing_x # 假设x和y间距相同
    }
    return meta_data


# ==============================================================================
# 主渲染流程
# ==============================================================================

def render_topdown_views(glb_path, custom_ortho_scale=None, target_coverage=0.9, draw_coordinates=False):
    """
    【已修正】主渲染流程，现在使用视觉中心来定位相机。
    """
    print("--- Step 1: Analyzing Scene Visual Boundaries ---")
    # 使用一个临时的ortho_scale加载模拟器，以便分析场景
    temp_sim = robust_load_ortho_sim(glb_path, ortho_scale=1.0) 
    
    # 【修改点】调用新的函数获取视觉边界和中心
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
    
    temp_sim.close()
    
    print(f"\n--- Step 2: Loading Simulator with Final Ortho Scale: {optimal_ortho_scale:.4f} ---")
    sim = robust_load_ortho_sim(glb_path, optimal_ortho_scale)
    
    print("\n--- Step 3: Detecting Floors ---")
    floor_extents = get_floor_navigable_extents(sim)
    floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
    
    # 【修改点】直接使用计算好的视觉中心点来定位相机
    x_center, z_center = scene_info['center']
    
    floor_images = []
    unprojected_coords = None
    
    print("\n--- Step 4: Rendering Floors and Calculating Coordinates ---")
    for i, fext in enumerate(floor_extents):
        print(f"Rendering floor {i} at height ~{fext['mean']:.2f}m...")
        # 将相机放在视觉中心上方
        agent_position = [x_center, fext['mean'] + 1.5, z_center] # 增加高度以避免被高处物体遮挡
        agent_rotation = get_downward_quaternion()
        
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        agent.set_state(new_state, True)
        
        obs = sim.get_sensor_observations()
        floor_images.append(obs['rgba_camera'])
        
        if i == 0:
            unprojected_coords = get_unprojected_world_coords(sim)
            print_corner_coordinates(unprojected_coords)
    
    sim.close()
    
    print("\n--- Step 5: Processing Final Image and Metadata ---")
    if floor_images:
        final_image = np.concatenate(floor_images, axis=0)
    else:
        final_image = np.zeros((2048, 2048, 4), dtype=np.uint8)
        print("Warning: No floors rendered.")

    meta_data = calculate_metadata(unprojected_coords) if unprojected_coords else {}
    if meta_data:
      print(f"\nCalculated Metadata:")
      print(f"  Pixel spacing: {meta_data['spacing_in_meters_per_pixel']:.6f} m/pixel")
      print(f"  World Origin (0,0) at pixel: (x={meta_data['origin_in_pixels'][0]:.2f}, y={meta_data['origin_in_pixels'][1]:.2f})")
    
    if draw_coordinates and unprojected_coords:
        print("\n--- Step 6: Drawing Coordinate System ---")
        annotated_pil_image = draw_coordinate_system(final_image, unprojected_coords, optimal_ortho_scale)
        final_image = np.array(annotated_pil_image)

    return final_image, unprojected_coords, meta_data

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 配置 ---
    try:
        # 尝试使用环境变量，方便在不同机器上运行
        hm3d_scene_path = os.environ.get("HM3D_SCENE_PATH", "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.glb")
        if not os.path.exists(hm3d_scene_path):
             raise FileNotFoundError(f"Scene file not found at: {hm3d_scene_path}. Please check the path or set the HM3D_SCENE_PATH environment variable.")
    except Exception as e:
        print(e)
        exit(1)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, "topdown_view_with_coords.png")
    output_metadata_path = os.path.join(output_dir, "topdown_metadata.json")
    
    # --- 执行 ---
    result_image, corner_info, meta_data = render_topdown_views(
        hm3d_scene_path, 
        custom_ortho_scale=0.02, # 取消注释以手动覆盖自动计算的值
        draw_coordinates=True
    )
    
    print(f"\nGenerated annotated top-down view with shape: {result_image.shape}")
    
    # --- 保存结果 ---
    if result_image.shape[-1] == 4:
        result_rgb = result_image[..., :3]
    else:
        result_rgb = result_image
    
    cv2.imwrite(output_image_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    print(f"Top-down view with coordinate system saved to: {output_image_path}")

    with open(output_metadata_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    print(f"Metadata (spacing, origin) saved to: {output_metadata_path}")