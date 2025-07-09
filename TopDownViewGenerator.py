import os
import cv2
import math
import numpy as np
import habitat_sim
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageFont


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

    if habitat_sim.__version__ == "0.1.7":
        sensor_cfg = habitat_sim.SensorSpec()
        sensor_cfg.resolution = [4096, 4096]
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


def get_coordinate_grid_interval(scene_size):
    """根据场景大小确定合适的坐标网格间隔"""
    if scene_size <= 3:
        return 0.5
    elif scene_size <= 8:
        return 1.0
    elif scene_size <= 15:
        return 2.0
    elif scene_size <= 30:
        return 5.0
    else:
        return 10.0


def calculate_corner_coordinates(scene_bounds, ortho_scale):
    """
    计算图像四角对应的世界坐标信息
    
    Args:
        scene_bounds: 场景边界 ((min_x, min_y, min_z), (max_x, max_y, max_z))
        ortho_scale: 正交投影缩放因子
    
    Returns:
        corner_coords: 图像四角的世界坐标信息字典
    """
    min_bounds, max_bounds = scene_bounds
    world_min_x, world_min_z = min_bounds[0], min_bounds[2]
    world_max_x, world_max_z = max_bounds[0], max_bounds[2]
    
    world_width = world_max_x - world_min_x
    world_height = world_max_z - world_min_z
    scene_size = max(world_width, world_height)
    
    # 计算世界坐标中心
    world_center_x = (world_min_x + world_max_x) / 2
    world_center_z = (world_min_z + world_max_z) / 2
    
    # 计算视野范围
    base_scene_size = 18.0
    base_ortho_scale = 0.05
    current_view_size = base_scene_size * (base_ortho_scale / ortho_scale)
    view_half_width = current_view_size / 2.0
    view_half_height = view_half_width
    
    # 计算图像四角坐标
    top_left_x = world_center_x - view_half_width
    top_left_z = world_center_z - view_half_height
    top_right_x = world_center_x + view_half_width
    top_right_z = world_center_z - view_half_height
    bottom_left_x = world_center_x - view_half_width
    bottom_left_z = world_center_z + view_half_height
    bottom_right_x = world_center_x + view_half_width
    bottom_right_z = world_center_z + view_half_height
    
    grid_interval = get_coordinate_grid_interval(scene_size)
    
    corner_coords = {
        'top_left': (top_left_x, top_left_z),
        'top_right': (top_right_x, top_right_z),
        'bottom_left': (bottom_left_x, bottom_left_z),
        'bottom_right': (bottom_right_x, bottom_right_z),
        'center': (world_center_x, world_center_z),
        'view_range': (view_half_width * 2, view_half_height * 2),
        'grid_interval': grid_interval
    }
    
    return corner_coords


def print_corner_coordinates(corner_coords):
    """输出图像角坐标信息"""
    print("=== 图像角坐标信息 ===")
    print(f"左上角 (Top-Left): X={corner_coords['top_left'][0]:.2f}m, Z={corner_coords['top_left'][1]:.2f}m")
    print(f"右上角 (Top-Right): X={corner_coords['top_right'][0]:.2f}m, Z={corner_coords['top_right'][1]:.2f}m")
    print(f"左下角 (Bottom-Left): X={corner_coords['bottom_left'][0]:.2f}m, Z={corner_coords['bottom_left'][1]:.2f}m")
    print(f"右下角 (Bottom-Right): X={corner_coords['bottom_right'][0]:.2f}m, Z={corner_coords['bottom_right'][1]:.2f}m")
    print(f"图像中心: X={corner_coords['center'][0]:.2f}m, Z={corner_coords['center'][1]:.2f}m")
    print(f"视野范围: {corner_coords['view_range'][0]:.2f}m × {corner_coords['view_range'][1]:.2f}m")
    print(f"网格间隔: {corner_coords['grid_interval']:.1f}m")
    print("=" * 30)


def draw_coordinate_system(image, scene_bounds, ortho_scale):
    """
    在俯视图上绘制世界坐标系
    
    Args:
        image: numpy数组格式的俯视图图像 (H, W, C)
        scene_bounds: 场景边界 ((min_x, min_y, min_z), (max_x, max_y, max_z))
        ortho_scale: 正交投影缩放因子
    
    Returns:
        annotated_image: 绘制坐标系后的PIL图像
        corner_coords: 图像四角的世界坐标信息
    """
    # 转换为PIL图像
    if image.shape[2] == 4:  # RGBA
        pil_image = Image.fromarray(image[..., :3], "RGB")
    else:
        pil_image = Image.fromarray(image, "RGB")
    
    # 计算坐标信息
    corner_coords = calculate_corner_coordinates(scene_bounds, ortho_scale)
    
    min_bounds, max_bounds = scene_bounds
    world_min_x, world_min_z = min_bounds[0], min_bounds[2]
    world_max_x, world_max_z = max_bounds[0], max_bounds[2]
    
    world_center_x = corner_coords['center'][0]
    world_center_z = corner_coords['center'][1]
    view_half_width = corner_coords['view_range'][0] / 2
    view_half_height = corner_coords['view_range'][1] / 2
    grid_interval = corner_coords['grid_interval']
    
    # 获取图像尺寸
    img_width, img_height = pil_image.size
    
    # 设置边距
    margin_left = 80
    margin_bottom = 60
    margin_top = 40
    margin_right = 40
    
    # 创建带边距的新画布
    new_width = img_width + margin_left + margin_right
    new_height = img_height + margin_top + margin_bottom
    new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    new_image.paste(pil_image, (margin_left, margin_top))
    
    draw = ImageDraw.Draw(new_image)
    
    # 加载字体
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 颜色定义
    grid_color = (100, 100, 100)
    major_grid_color = (150, 150, 150)
    border_color = (255, 255, 255)
    text_color = (255, 255, 255)
    
    # 原始图像在新画布上的区域
    img_area_x0 = margin_left
    img_area_y0 = margin_top
    img_area_x1 = margin_left + img_width
    img_area_y1 = margin_top + img_height
    
    # 世界坐标到图像像素的转换函数
    def world_to_pixel(world_x, world_z):
        rel_x = world_x - world_center_x
        rel_z = world_z - world_center_z
        pixel_x = img_area_x0 + img_width / 2 + (rel_x / view_half_width) * (img_width / 2)
        pixel_y = img_area_y0 + img_height / 2 + (rel_z / view_half_height) * (img_height / 2)
        return int(pixel_x), int(pixel_y)
    
    # 绘制垂直网格线和X轴标注
    x_start = math.ceil((world_center_x - view_half_width) / grid_interval) * grid_interval
    x_current = x_start
    
    while x_current <= world_center_x + view_half_width:
        pixel_x, _ = world_to_pixel(x_current, world_center_z)
        
        if img_area_x0 <= pixel_x <= img_area_x1:
            is_major = abs(x_current - round(x_current)) < 0.01
            line_color = major_grid_color if is_major else grid_color
            line_width = 2 if is_major else 1
            
            draw.line([(pixel_x, img_area_y0), (pixel_x, img_area_y1)], 
                     fill=line_color, width=line_width)
            
            tick_length = 8 if is_major else 5
            draw.line([(pixel_x, img_area_y1), (pixel_x, img_area_y1 + tick_length)], 
                     fill=(255, 255, 255), width=2)
            
            label_text = f"{x_current:.1f}"
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font_medium)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(label_text) * 8
            
            label_x = pixel_x - text_width / 2
            label_y = img_area_y1 + tick_length + 5
            draw.text((label_x, label_y), label_text, fill=text_color, font=font_medium)
        
        x_current += grid_interval
    
    # 绘制水平网格线和Z轴标注
    z_start = math.ceil((world_center_z - view_half_height) / grid_interval) * grid_interval
    z_current = z_start
    
    while z_current <= world_center_z + view_half_height:
        _, pixel_y = world_to_pixel(world_center_x, z_current)
        
        if img_area_y0 <= pixel_y <= img_area_y1:
            is_major = abs(z_current - round(z_current)) < 0.01
            line_color = major_grid_color if is_major else grid_color
            line_width = 2 if is_major else 1
            
            draw.line([(img_area_x0, pixel_y), (img_area_x1, pixel_y)], 
                     fill=line_color, width=line_width)
            
            tick_length = 8 if is_major else 5
            draw.line([(img_area_x0 - tick_length, pixel_y), (img_area_x0, pixel_y)], 
                     fill=(255, 255, 255), width=2)
            
            label_text = f"{z_current:.1f}"
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font_medium)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(label_text) * 8
                text_height = 12
            
            label_x = img_area_x0 - tick_length - text_width - 5
            label_y = pixel_y - text_height / 2
            draw.text((label_x, label_y), label_text, fill=text_color, font=font_medium)
        
        z_current += grid_interval
    
    # 绘制边框
    draw.rectangle([img_area_x0-1, img_area_y0-1, img_area_x1+1, img_area_y1+1], 
                  outline=border_color, width=2)
    
    # 添加轴标签
    x_label = "X (meters)"
    try:
        bbox = draw.textbbox((0, 0), x_label, font=font_large)
        x_label_width = bbox[2] - bbox[0]
    except:
        x_label_width = len(x_label) * 12
    
    x_label_x = (new_width - x_label_width) / 2
    x_label_y = new_height - 25
    draw.text((x_label_x, x_label_y), x_label, fill=text_color, font=font_large)
    
    # Z轴标签（垂直）
    z_label = "Z (meters)"
    temp_img = Image.new('RGBA', (200, 30), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((0, 0), z_label, fill=text_color, font=font_large)
    rotated = temp_img.rotate(90, expand=True)
    
    z_label_x = 15
    z_label_y = (new_height - rotated.height) / 2
    new_image.paste(rotated, (int(z_label_x), int(z_label_y)), rotated)
    
    # 添加原点标记
    origin_pixel_x, origin_pixel_y = world_to_pixel(0, 0)
    if (img_area_x0 <= origin_pixel_x <= img_area_x1 and 
        img_area_y0 <= origin_pixel_y <= img_area_y1):
        draw.ellipse([origin_pixel_x-6, origin_pixel_y-6, origin_pixel_x+6, origin_pixel_y+6], 
                    fill=(255, 255, 0), outline=(255, 255, 255), width=2)
        draw.text((origin_pixel_x+10, origin_pixel_y-10), "Origin (0,0)", 
                 fill=(255, 255, 0), font=font_small)
    
    # 添加比例尺信息
    scale_info = f"Grid: {grid_interval}m | Ortho Scale: {ortho_scale:.4f} | View: {view_half_width*2:.1f}m × {view_half_height*2:.1f}m"
    draw.text((img_area_x0, 5), scale_info, fill=text_color, font=font_small)
    
    return new_image, corner_coords


def render_topdown_views(glb_path, custom_ortho_scale=None, target_coverage=0.9, draw_coordinates=False):
    """
    渲染俯视图，支持自定义正交投影比例和坐标系绘制
    
    Args:
        glb_path: 场景文件路径
        custom_ortho_scale: 自定义正交投影比例（可选）
        target_coverage: 目标覆盖范围（当不使用自定义比例时）
        draw_coordinates: 是否绘制坐标系
    
    Returns:
        floor_images: 渲染的俯视图像
        corner_coords: 图像角坐标信息（总是返回）
    """
    # 第一步：加载临时模拟器计算场景边界
    temp_sim = robust_load_ortho_sim(glb_path)
    
    if custom_ortho_scale is not None:
        print(f"Using custom orthographic scale: {custom_ortho_scale:.4f}")
        optimal_ortho_scale = custom_ortho_scale
    else:
        scene_info = calculate_scene_bounds(temp_sim)
        optimal_ortho_scale = calculate_ortho_scale(scene_info['max_dimension'], target_coverage)
        
        print(f"Scene dimensions: {scene_info['width']:.2f}m x {scene_info['depth']:.2f}m")
        print(f"Maximum dimension: {scene_info['max_dimension']:.2f}m")
        if custom_ortho_scale is None:
            print(f"Target coverage: {target_coverage:.1%}")
        print(f"Calculated orthographic scale: {optimal_ortho_scale:.4f}")
    
    # 保存场景边界信息
    navmesh_vertices = np.array(temp_sim.pathfinder.build_navmesh_vertices())
    if len(navmesh_vertices) > 0:
        min_bounds = navmesh_vertices.min(axis=0).tolist()
        max_bounds = navmesh_vertices.max(axis=0).tolist()
        scene_bounds = (min_bounds, max_bounds)
    else:
        scene_bounds = ([-5.0, 0.0, -5.0], [5.0, 3.0, 5.0])
    
    temp_sim.close()
    
    # 第二步：使用最优比例加载模拟器
    sim = robust_load_ortho_sim(glb_path, optimal_ortho_scale)
    
    # 获取楼层信息
    floor_extents = get_floor_navigable_extents(sim)
    floor_extents = sorted(floor_extents, key=lambda x: x['mean'])
    navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
    
    # 计算场景中心
    x_center = (navmesh_vertices[:, 0].min() + navmesh_vertices[:, 0].max()) / 2.0
    z_center = (navmesh_vertices[:, 2].min() + navmesh_vertices[:, 2].max()) / 2.0
    y_center = (navmesh_vertices[:, 1].min() + navmesh_vertices[:, 1].max()) / 2.0
    scene_cent = [x_center, y_center, z_center]
    
    # 渲染各楼层
    floor_images = []
    for fext in floor_extents:
        mask = (
            (navmesh_vertices[:, 1] <= fext['max'] + 0.25) & \
            (navmesh_vertices[:, 1] >= fext['min'] - 0.25)
        )
        fcent = np.median(navmesh_vertices[mask, :], axis=0).tolist()
        
        # 设置智能体状态
        agent_position = [scene_cent[0], fcent[1] + 1.0, scene_cent[2]]
        agent_rotation = get_downward_quaternion()
        agent = sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = agent_position
        new_state.rotation = agent_rotation
        new_state.sensor_states = {}
        agent.set_state(new_state, True)
        
        # 获取观察
        obs = sim.get_sensor_observations()
        floor_images.append(obs['rgba_camera'])

    # 垂直拼接图像
    floor_images = np.concatenate(floor_images, axis=0)
    sim.close()
    
    # 计算角坐标信息（总是计算）
    corner_coords = calculate_corner_coordinates(scene_bounds, optimal_ortho_scale)
    
    # 输出角坐标信息（总是输出）
    print_corner_coordinates(corner_coords)
    
    # 如果需要绘制坐标系
    if draw_coordinates:
        print("Drawing coordinate system...")
        annotated_image, corner_coords = draw_coordinate_system(floor_images, scene_bounds, optimal_ortho_scale)
        
        # 将PIL图像转换回numpy数组
        floor_images = np.array(annotated_image)
        if len(floor_images.shape) == 3 and floor_images.shape[2] == 3:
            # 添加alpha通道以保持一致性
            alpha_channel = np.ones((floor_images.shape[0], floor_images.shape[1], 1), dtype=floor_images.dtype) * 255
            floor_images = np.concatenate([floor_images, alpha_channel], axis=2)

    return floor_images, corner_coords


if __name__ == '__main__':
    # 使用示例
    #hm3d_scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    hm3d_scene_path ="/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.glb"
    output_path_with_coords = "/home/yaoaa/habitat-lab/TopDownVIew.png"
    result_with_coords, corner_info = render_topdown_views(
        hm3d_scene_path, 
        draw_coordinates=True
    )
    print(f"生成的带注释俯视图尺寸: {result_with_coords.shape}")
    
    # 保存图像
    if result_with_coords.shape[-1] == 4:
        result_coords_rgb = result_with_coords[..., :3]
    else:
        result_coords_rgb = result_with_coords
    cv2.imwrite(output_path_with_coords, cv2.cvtColor(result_coords_rgb, cv2.COLOR_RGB2BGR))
    print(f"带坐标系的俯视图已保存到: {output_path_with_coords}")
