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
    # 随机采样可导航点
    random_navigable_points = []
    for _i in range(num_points_to_sample):
        point = hsim.pathfinder.get_random_navigable_point()
        if np.isnan(point).any() or np.isinf(point).any():
            continue
        random_navigable_points.append(point)
    random_navigable_points = np.array(random_navigable_points)
    
    # 使用DBSCAN对Y坐标进行聚类
    y_coors = np.around(random_navigable_points[:, 1], decimals=1)
    y_range = y_coors.max() - y_coors.min()
    y_std = y_coors.std()
    
    print(f"Y coordinates range: {y_coors.min():.2f} to {y_coors.max():.2f} (range: {y_range:.2f}m)")
    print(f"Y coordinates std: {y_std:.3f}")
    
    # 动态调整DBSCAN参数
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
    
    # 合并相近的簇
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
    
    # 估计楼层范围
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
    
    # 如果没有检测到楼层，创建默认楼层
    if len(floor_extents) == 0:
        print("No floors detected, creating default floor from all points")
        floor_min = y_coors.min().item()
        floor_max = y_coors.max().item()
        floor_mean = y_coors.mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})

    return floor_extents


def calculate_scene_bounds(hsim: habitat_sim.Simulator):
    """计算场景可导航区域的边界以确定合适的正交投影比例"""
    navmesh_vertices = np.array(hsim.pathfinder.build_navmesh_vertices())
    
    x_min, x_max = navmesh_vertices[:, 0].min(), navmesh_vertices[:, 0].max()
    z_min, z_max = navmesh_vertices[:, 2].min(), navmesh_vertices[:, 2].max()
    
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
    """根据场景大小计算合适的正交投影比例"""
    base_scene_size = 20.0
    base_ortho_scale = 0.05
    
    calculated_scale = (base_ortho_scale * base_scene_size) / (scene_size / target_coverage)
    safety_margin = 1.2
    ortho_scale = calculated_scale / safety_margin
    
    return max(0.01, min(0.2, ortho_scale))


def make_ortho_habitat_configuration(scene_path, ortho_scale=None):
    """创建正交投影的Habitat-Sim配置"""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    if ortho_scale is None:
        ortho_scale = 0.05

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [2048, 2048]
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
# 核心修改部分：使用unproject获取真实世界坐标
# ==============================================================================

def get_unprojected_world_coords(sim: habitat_sim.Simulator):
    """
    计算正交投影相机视图的世界坐标范围。
    对于正交投影，我们需要根据相机参数计算视图边界。
    """
    # 获取agent和相机传感器
    agent = sim.get_agent(0)
    camera_sensor = agent.scene_node.node_sensor_suite.get("rgba_camera")
    
    # 获取相机分辨率和规格
    sensor_spec = agent.agent_config.sensor_specifications[0]  # 假设第一个传感器是相机
    width, height = sensor_spec.resolution[0], sensor_spec.resolution[1]
    ortho_scale = sensor_spec.ortho_scale
    
    # 获取相机位置
    render_camera = camera_sensor.render_camera
    cam_pos = render_camera.node.absolute_translation
    
    # 对于正交投影，视图范围由ortho_scale决定
    # ortho_scale表示从相机中心到视图边缘的距离
    view_half_width = ortho_scale * width / 2
    view_half_height = ortho_scale * height / 2
    
    # 计算世界坐标中的四个角点
    # 对于俯视图（向下看），X轴是水平方向，Z轴是垂直方向
    tl_world = np.array([cam_pos[0] - view_half_width, cam_pos[1], cam_pos[2] - view_half_height])
    tr_world = np.array([cam_pos[0] + view_half_width, cam_pos[1], cam_pos[2] - view_half_height])
    bl_world = np.array([cam_pos[0] - view_half_width, cam_pos[1], cam_pos[2] + view_half_height])
    br_world = np.array([cam_pos[0] + view_half_width, cam_pos[1], cam_pos[2] + view_half_height])
    
    # 场景尺寸
    scene_width = tr_world[0] - tl_world[0]
    scene_depth = bl_world[2] - tl_world[2]
    scene_size = max(scene_width, scene_depth)
    grid_interval = get_coordinate_grid_interval(scene_size)

    corner_coords = {
        'top_left': (tl_world[0], tl_world[2]),
        'top_right': (tr_world[0], tr_world[2]),
        'bottom_left': (bl_world[0], bl_world[2]),
        'bottom_right': (br_world[0], br_world[2]),
        'center': (cam_pos[0], cam_pos[2]),
        'view_range': (scene_width, scene_depth),
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
    print(f"Top-Right    : X={corner_coords['top_right'][0]:.2f}m, Z={corner_coords['top_right'][1]:.2f}m")
    print(f"Bottom-Left  : X={corner_coords['bottom_left'][0]:.2f}m, Z={corner_coords['bottom_left'][1]:.2f}m")
    print(f"Bottom-Right : X={corner_coords['bottom_right'][0]:.2f}m, Z={corner_coords['bottom_right'][1]:.2f}m")
    print(f"Image Center : X={corner_coords['center'][0]:.2f}m, Z={corner_coords['center'][1]:.2f}m")
    print(f"View Range   : {corner_coords['view_range'][0]:.2f}m × {corner_coords['view_range'][1]:.2f}m")
    print(f"Grid Interval: {corner_coords['grid_interval']:.1f}m")
    print("=" * 40)


def draw_coordinate_system(image: np.ndarray, corner_coords: dict, ortho_scale: float):
    """
    在俯视图上绘制基于unprojected坐标的世界坐标系。
    
    Args:
        image: numpy数组格式的俯视图图像 (H, W, C)
        corner_coords: 由get_unprojected_world_coords生成的角点坐标字典。
        ortho_scale: 仅用于在图上显示信息。
    
    Returns:
        annotated_image: 绘制坐标系后的PIL图像。
    """
    # 转换为PIL图像
    pil_image = Image.fromarray(image[..., :3], "RGB")
    
    # 从字典中提取坐标和参数
    tl_x, tl_z = corner_coords['top_left']
    tr_x, _ = corner_coords['top_right']
    _, bl_z = corner_coords['bottom_left']
    grid_interval = corner_coords['grid_interval']
    
    img_width, img_height = pil_image.size
    
    # 设置边距
    margin_left, margin_bottom, margin_top, margin_right = 80, 60, 40, 40
    
    # 创建带边距的新画布
    new_width = img_width + margin_left + margin_right
    new_height = img_height + margin_top + margin_bottom
    new_image = Image.new('RGB', (new_width, new_height), (20, 20, 20))
    new_image.paste(pil_image, (margin_left, margin_top))
    
    draw = ImageDraw.Draw(new_image)
    
    # 加载字体
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except IOError:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # 颜色定义
    grid_color, major_grid_color, border_color, text_color = (80, 80, 80), (120, 120, 120), (255, 255, 255), (255, 255, 255)
    
    # 原始图像在新画布上的区域
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
    
    # 绘制垂直网格线和X轴标注
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
    
    # 绘制水平网格线和Z轴标注
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
    
    # 绘制边框
    draw.rectangle([img_area_x0-1, img_area_y0-1, img_area_x1, img_area_y1], outline=border_color, width=1)
    
    # 添加轴标签
    x_label = "X (meters)"
    bbox = draw.textbbox((0, 0), x_label, font=font_large)
    draw.text(((new_width - (bbox[2]-bbox[0])) / 2, new_height - 25), x_label, fill=text_color, font=font_large)
    
    z_label = "Z (meters)"
    temp_img = Image.new('RGBA', (200, 30), (0,0,0,0)); temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((0, 0), z_label, fill=text_color, font=font_large)
    rotated = temp_img.rotate(90, expand=True)
    new_image.paste(rotated, (15, int((new_height - rotated.height) / 2)), rotated)
    
    # 添加原点标记
    origin_pixel_x, origin_pixel_y = world_to_pixel(0, 0)
    if img_area_x0 <= origin_pixel_x <= img_area_x1 and img_area_y0 <= origin_pixel_y <= img_area_y1:
        draw.ellipse([origin_pixel_x-5, origin_pixel_y-5, origin_pixel_x+5, origin_pixel_y+5], fill=(255, 255, 0), outline=border_color, width=1)
        draw.text((origin_pixel_x+8, origin_pixel_y-8), "Origin (0,0)", fill=(255, 255, 0), font=font_small)
    
    # 添加比例尺信息
    scale_info = f"Grid: {grid_interval}m | Ortho Scale: {ortho_scale:.4f} | View: {corner_coords['view_range'][0]:.1f}m × {corner_coords['view_range'][1]:.1f}m"
    draw.text((img_area_x0 + 5, 5), scale_info, fill=text_color, font=font_small)
    
    return new_image


def calculate_metadata(corner_coords: dict):
    """根据unprojected坐标计算元数据（像素间距和原点位置）"""
    tl_x, tl_z = corner_coords['top_left']
    tr_x, _ = corner_coords['top_right']
    img_width, img_height = corner_coords['image_size']
    
    # 像素间距 (米/像素)
    spacing = (tr_x - tl_x) / img_width
    
    # 世界坐标原点(0,0)在图像中的像素位置 (相对于图像左上角)
    # (0 - tl_x) gives the world distance from the left edge to the origin.
    # Dividing by spacing gives this distance in pixels.
    origin_pixel_x = (0.0 - tl_x) / spacing
    origin_pixel_y = (0.0 - tl_z) / spacing # For Z axis
    
    meta_data = {
        "image_size": [img_width, img_height],
        "origin_x_pixel": origin_pixel_x,
        "origin_y_pixel": origin_pixel_y,
        "spacing_in_meters_per_pixel": spacing
    }
    return meta_data

# ==============================================================================
# 主渲染流程
# ==============================================================================

def render_topdown_views(glb_path, custom_ortho_scale=None, target_coverage=0.9, draw_coordinates=False):
    """
    渲染俯视图，使用unproject获取真实坐标，并支持绘制。
    """
    # 步骤 1: 加载临时模拟器以计算场景边界和最佳ortho_scale
    temp_sim = robust_load_ortho_sim(glb_path)
    
    if custom_ortho_scale is not None:
        print(f"Using custom orthographic scale: {custom_ortho_scale:.4f}")
        optimal_ortho_scale = custom_ortho_scale
    else:
        scene_info = calculate_scene_bounds(temp_sim)
        optimal_ortho_scale = calculate_ortho_scale(scene_info['max_dimension'], target_coverage)
        print(f"Scene dimensions: {scene_info['width']:.2f}m x {scene_info['depth']:.2f}m")
        print(f"Maximum dimension: {scene_info['max_dimension']:.2f}m")
        print(f"Calculated optimal orthographic scale: {optimal_ortho_scale:.4f}")
    
    temp_sim.close()
    
    # 步骤 2: 使用最佳比例加载主模拟器
    sim = robust_load_ortho_sim(glb_path, optimal_ortho_scale)
    
    floor_extents = get_floor_navigable_extents(sim)
    floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
    
    # 计算场景中心以定位相机
    navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
    x_center = (navmesh_vertices[:, 0].min() + navmesh_vertices[:, 0].max()) / 2.0
    z_center = (navmesh_vertices[:, 2].min() + navmesh_vertices[:, 2].max()) / 2.0
    
    floor_images = []
    unprojected_coords = None # 将在此处存储坐标信息
    
    # 步骤 3: 渲染每个楼层
    for i, fext in enumerate(floor_extents):
        # 定位相机在当前楼层上方
        agent_position = [x_center, fext['mean'] + 5.0, z_center] # 放在楼层平均高度上方
        agent_rotation = get_downward_quaternion()
        
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        agent.set_state(new_state, True)
        
        # 步骤 4: 获取观察结果和精确坐标
        obs = sim.get_sensor_observations()
        floor_images.append(obs['rgba_camera'])
        
        # 只需要在第一次渲染后获取坐标，因为X,Z平面对于所有楼层都是一样的
        if i == 0:
            unprojected_coords = get_unprojected_world_coords(sim)
            print_corner_coordinates(unprojected_coords)
    
    # 步骤 5: 关闭模拟器并处理图像
    sim.close()
    
    # 垂直拼接所有楼层的图像
    if floor_images:
        final_image = np.concatenate(floor_images, axis=0)
    else:
        # 如果没有楼层，创建一个空白图像以避免错误
        final_image = np.zeros((2048, 2048, 4), dtype=np.uint8)
        print("Warning: No floors rendered.")

    # 计算元数据
    meta_data = calculate_metadata(unprojected_coords) if unprojected_coords else {}
    if meta_data:
      print(f"\nCalculated Metadata:")
      print(f"  Pixel spacing: {meta_data['spacing_in_meters_per_pixel']:.6f} m/pixel")
      print(f"  World Origin (0,0) at pixel: (x={meta_data['origin_x_pixel']:.2f}, y={meta_data['origin_y_pixel']:.2f})")
    
    # 步骤 6: (可选) 绘制坐标系
    if draw_coordinates and unprojected_coords:
        print("\nDrawing coordinate system...")
        annotated_pil_image = draw_coordinate_system(final_image, unprojected_coords, optimal_ortho_scale)
        # 将PIL图像转换回numpy数组以便保存
        final_image = np.array(annotated_pil_image)

    return final_image, unprojected_coords, meta_data

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 配置 ---
    # 请将此路径替换为您的GLB文件路径
    # hm3d_scene_path = "/path/to/your/scene.glb" 
    hm3d_scene_path = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.glb"
    
    # 输出目录和文件名
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, "topdown_view_with_coords.png")
    output_metadata_path = os.path.join(output_dir, "topdown_metadata.json")
    
    # --- 执行 ---
    result_image, corner_info, meta_data = render_topdown_views(
        hm3d_scene_path, 
        draw_coordinates=True  # 设置为True以绘制坐标系
    )
    
    print(f"\nGenerated annotated top-down view with shape: {result_image.shape}")
    
    # --- 保存结果 ---
    # 保存图像 (将RGB转为BGR给OpenCV)
    if result_image.shape[-1] == 4: # RGBA
        result_rgb = result_image[..., :3]
    else: # RGB
        result_rgb = result_image
    
    cv2.imwrite(output_image_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    print(f"Top-down view with coordinate system saved to: {output_image_path}")

    # 保存元数据
    with open(output_metadata_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    print(f"Metadata (spacing, origin) saved to: {output_metadata_path}")