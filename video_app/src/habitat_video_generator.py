#!/usr/bin/env python3
"""
Habitat视频生成器 - 核心逻辑
复用interactive_app的HabitatSimulator逻辑，添加视频生成功能
"""

import sys
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional, Union, Any
import time
from datetime import datetime
import cv2

# 添加interactive_app的src路径以复用代码
interactive_app_src = os.path.join(os.path.dirname(__file__), '../../interactive_app/src')
sys.path.insert(0, interactive_app_src)

# Habitat相关导入
import habitat_sim
import magnum as mn

class HabitatSimulator:
    """封装Habitat-sim相关逻辑的类"""
    
    # 地图padding参数常量 - 确保所有相关函数使用相同的值
    MAP_PADDING_LEFT = 80    # 为Y轴标签留出空间
    MAP_PADDING_BOTTOM = 60  # 为X轴标签留出空间
    MAP_PADDING_TOP = 40     # 顶部边距
    MAP_PADDING_RIGHT = 40   # 右侧边距
    
    def __init__(self, scene_filepath: str, resolution: Tuple[int, int] = (512, 512)):
        self.scene_filepath = scene_filepath
        self.resolution = resolution
        self.sim = None
        self.agent = None
        self.scene_bounds = None
        self.scene_center = None
        self.scene_size = None
        self.ortho_scale = None
        self.base_map_image = None
        
        self._initialize_simulator()
        self._generate_base_map()
    
    def _initialize_simulator(self):
        """初始化Habitat模拟器"""
        # 配置后端 - 启用GPU加速和物理
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_filepath
        backend_cfg.enable_physics = True  # 启用物理以支持实体智能体
        backend_cfg.gpu_device_id = 0  # 使用第一张GPU (RTX 4090)
        backend_cfg.random_seed = 1
        
        # 配置FPV传感器 - 设置1.5米高度（正常人眼高度）
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [self.resolution[1], self.resolution[0]]
        fpv_sensor_spec.position = mn.Vector3(0, 1.5, 0)  # 1.5米高度
        fpv_sensor_spec.hfov = 90.0  # 90度水平视野
        
        # 临时获取场景边界以计算正交传感器分辨率
        # 创建一个最小的传感器配置用于临时模拟器
        temp_sensor = habitat_sim.CameraSensorSpec()
        temp_sensor.uuid = "temp_sensor"
        temp_sensor.sensor_type = habitat_sim.SensorType.COLOR
        temp_sensor.resolution = [64, 64]  # 最小分辨率
        
        temp_agent_cfg = habitat_sim.agent.AgentConfiguration()
        temp_agent_cfg.sensor_specifications = [temp_sensor]
        
        temp_sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [temp_agent_cfg]))
        
        # 确保导航网格已加载
        if not temp_sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            temp_sim.recompute_navmesh(temp_sim.pathfinder, navmesh_settings)
        
        # 使用导航网格顶点计算场景边界（更可靠）
        navmesh_vertices = np.array(temp_sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) == 0:
            print("警告：无法获取导航网格顶点，使用默认场景边界")
            world_size_x = 10.0
            world_size_z = 10.0
        else:
            min_bounds = navmesh_vertices.min(axis=0)
            max_bounds = navmesh_vertices.max(axis=0)
            world_size_x = max_bounds[0] - min_bounds[0]
            world_size_z = max_bounds[2] - min_bounds[2]
            print(f"场景导航区域尺寸: {world_size_x:.2f} x {world_size_z:.2f}")
        
        temp_sim.close()
        
        # 确保尺寸不为零
        if world_size_x <= 0:
            world_size_x = 10.0
        if world_size_z <= 0:
            world_size_z = 10.0
        
        # 使用TopV.py的成功配置 - 固定4096x4096分辨率
        print(f"场景尺寸: {world_size_x:.2f} x {world_size_z:.2f}")
        
        # 配置正交传感器（完全模仿TopV.py的配置）
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.resolution = [4096, 4096]  # TopV.py使用的固定高分辨率
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        ortho_sensor_spec.far = 1000.0
        ortho_sensor_spec.near = 0.01
        ortho_sensor_spec.hfov = 90
        ortho_sensor_spec.ortho_scale = 0.05  # TopV.py的关键参数
        ortho_sensor_spec.clear_color = [0., 0., 0., 0.]
        # 不设置uuid，使用默认传感器名称（与TopV.py一致）
        
        print(f"正交传感器配置: 4096x4096, ortho_scale=0.05")
        
        # 配置智能体 - 设置基本参数
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, ortho_sensor_spec]
        
        # 配置动作空间（正常人的行走和转向速度）
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)  # 每步25cm
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)  # 每次转10度
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        # 创建完整配置
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        # 实例化模拟器
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        
        # 获取场景边界信息
        # 使用导航网格顶点计算边界（更可靠）
        if not self.sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        
        navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) == 0:
            print("警告：无法获取导航网格顶点，使用默认场景边界")
            # 使用默认边界
            min_bounds = [-5.0, 0.0, -5.0]
            max_bounds = [5.0, 3.0, 5.0]
        else:
            min_bounds = navmesh_vertices.min(axis=0).tolist()
            max_bounds = navmesh_vertices.max(axis=0).tolist()
        
        self.scene_bounds = (min_bounds, max_bounds)
        self.scene_center = ((np.array(min_bounds) + np.array(max_bounds)) / 2.0).tolist()
        self.scene_size = (np.array(max_bounds) - np.array(min_bounds)).tolist()
        self.ortho_scale = max(self.scene_size[0], self.scene_size[2]) / 2.0
        
        print(f"场景边界: {self.scene_bounds}")
        print(f"场景中心: {self.scene_center}")
        print(f"场景尺寸: {self.scene_size}")
    
    def _generate_base_map(self):
        """生成带坐标系的基础俯视地图"""
        # 参考TopV.py的相机位置设置方式
        # 使用场景中心位置，高度设置为合理的俯视高度
        navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) > 0:
            # 使用导航网格的平均高度 + 适当的俯视距离
            mean_height = navmesh_vertices[:, 1].mean()
            camera_height = mean_height + 5.0  # 在平均高度上方5米
        else:
            # 备选方案：使用场景边界
            camera_height = self.scene_bounds[1][1] + 5.0
        
        camera_position = mn.Vector3(self.scene_center[0], camera_height, self.scene_center[2])
        
        # 设置智能体状态以获取俯视图
        agent_state = habitat_sim.AgentState()
        agent_state.position = camera_position
        agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068])  # 朝下看
        self.agent.set_state(agent_state)
        
        print(f"正交相机位置: {camera_position}")
        print(f"相机高度: {camera_height:.2f}m")
        
        # 正交传感器已在初始化时配置为ORTHOGRAPHIC类型
        # 这里不需要重新配置sensor_subtype
        
        # 获取俯视图 - 使用默认传感器名称（与TopV.py一致）
        observations = self.sim.get_sensor_observations()
        # 由于我们没有设置uuid，使用默认的COLOR传感器名称
        # 在多传感器配置中，需要确定正确的传感器名称
        print(f"Available sensors: {list(observations.keys())}")
        
        # 尝试不同的传感器名称
        if "rgba_camera" in observations:
            ortho_img = observations["rgba_camera"]
        elif "color_sensor" in observations:
            ortho_img = observations["color_sensor"]
        else:
            # 使用第一个可用的传感器
            sensor_name = list(observations.keys())[0]
            ortho_img = observations[sensor_name]
            print(f"Using sensor: {sensor_name}")
        
        # 转换为PIL图像
        base_image = Image.fromarray(ortho_img[..., :3], "RGB")
        
        # 在图像上绘制坐标系
        self.base_map_image = self._draw_coordinate_system(base_image)
    
    def _draw_coordinate_system(self, image: Image.Image) -> Image.Image:
        """在地图上绘制坐标系 - 参考add_grid.py的实现方式"""
        original_width, original_height = image.size
        
        # 计算实际坐标范围
        world_min_x = self.scene_bounds[0][0]
        world_max_x = self.scene_bounds[1][0]
        world_min_z = self.scene_bounds[0][2] 
        world_max_z = self.scene_bounds[1][2]
        
        x_range = world_max_x - world_min_x
        z_range = world_max_z - world_min_z
        
        # 设置边距参数 - 使用类常量确保一致性
        padding_left = self.MAP_PADDING_LEFT
        padding_bottom = self.MAP_PADDING_BOTTOM
        padding_top = self.MAP_PADDING_TOP
        padding_right = self.MAP_PADDING_RIGHT
        
        # 创建带边距的新画布
        new_width = original_width + padding_left + padding_right
        new_height = original_height + padding_top + padding_bottom
        
        # 根据原图模式创建新画布
        if image.mode == 'RGBA':
            new_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 255))  # 黑色背景
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))  # 黑色背景
        
        # 将原始图像粘贴到新画布上
        image_paste_x = padding_left
        image_paste_y = padding_top
        new_image.paste(image, (image_paste_x, image_paste_y))
        
        # 创建绘图对象
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
        grid_color = (100, 100, 100)     # 深灰色网格线
        major_grid_color = (150, 150, 150)  # 主网格线（稍亮）
        border_color = (255, 255, 255)   # 白色边框
        tick_color = (255, 255, 255)     # 白色刻度线
        text_color = (255, 255, 255)     # 白色文字
        
        # 原始图像在新画布上的区域边界
        img_area_x0 = padding_left
        img_area_y0 = padding_top
        img_area_x1 = padding_left + original_width
        img_area_y1 = padding_top + original_height
        
        # 计算合适的网格间隔（世界坐标）
        def get_nice_interval(range_val):
            """获取合适的刻度间隔"""
            if range_val <= 2:
                return 0.5
            elif range_val <= 5:
                return 1.0
            elif range_val <= 10:
                return 2.0
            elif range_val <= 20:
                return 5.0
            else:
                return 10.0
        
        x_interval = get_nice_interval(x_range)
        z_interval = get_nice_interval(z_range)
        
        # 计算像素间隔
        x_pixel_interval = x_interval / x_range * original_width
        z_pixel_interval = z_interval / z_range * original_height
        
        # 绘制垂直网格线和X轴标注
        x_start = math.ceil(world_min_x / x_interval) * x_interval
        x_current = x_start
        
        while x_current <= world_max_x:
            # 计算在原图中的像素位置
            x_pixel_in_orig = (x_current - world_min_x) / x_range * original_width
            x_pixel_on_canvas = img_area_x0 + x_pixel_in_orig
            
            if 0 <= x_pixel_in_orig <= original_width:
                # 判断是否为主网格线（整数值）
                is_major = abs(x_current - round(x_current)) < 0.01
                line_color = major_grid_color if is_major else grid_color
                line_width = 2 if is_major else 1
                
                # 绘制垂直网格线
                draw.line([(x_pixel_on_canvas, img_area_y0), 
                          (x_pixel_on_canvas, img_area_y1)], 
                         fill=line_color, width=line_width)
                
                # 绘制X轴刻度线
                tick_length = 8 if is_major else 5
                # 底部刻度线
                draw.line([(x_pixel_on_canvas, img_area_y1), 
                          (x_pixel_on_canvas, img_area_y1 + tick_length)], 
                         fill=tick_color, width=2)
                # 顶部刻度线  
                draw.line([(x_pixel_on_canvas, img_area_y0 - tick_length), 
                          (x_pixel_on_canvas, img_area_y0)], 
                         fill=tick_color, width=2)
                
                # 绘制X轴标签
                label_text = f"{x_current:.1f}"
                try:
                    bbox = draw.textbbox((0, 0), label_text, font=font_medium)
                    text_width = bbox[2] - bbox[0]
                except AttributeError:
                    text_width, _ = draw.textsize(label_text, font=font_medium)
                
                # 底部标签
                label_x = x_pixel_on_canvas - text_width / 2
                label_y = img_area_y1 + tick_length + 5
                draw.text((label_x, label_y), label_text, fill=text_color, font=font_medium)
                
                # 顶部标签（可选）
                if is_major:
                    label_y_top = img_area_y0 - tick_length - 20
                    draw.text((label_x, label_y_top), label_text, fill=text_color, font=font_small)
            
            x_current += x_interval
        
        # 绘制水平网格线和Z轴标注
        z_start = math.ceil(world_min_z / z_interval) * z_interval
        z_current = z_start
        
        while z_current <= world_max_z:
            # 计算在原图中的像素位置
            z_pixel_in_orig = (z_current - world_min_z) / z_range * original_height
            z_pixel_on_canvas = img_area_y0 + z_pixel_in_orig
            
            if 0 <= z_pixel_in_orig <= original_height:
                # 判断是否为主网格线
                is_major = abs(z_current - round(z_current)) < 0.01
                line_color = major_grid_color if is_major else grid_color
                line_width = 2 if is_major else 1
                
                # 绘制水平网格线
                draw.line([(img_area_x0, z_pixel_on_canvas), 
                          (img_area_x1, z_pixel_on_canvas)], 
                         fill=line_color, width=line_width)
                
                # 绘制Z轴刻度线
                tick_length = 8 if is_major else 5
                # 左侧刻度线
                draw.line([(img_area_x0 - tick_length, z_pixel_on_canvas), 
                          (img_area_x0, z_pixel_on_canvas)], 
                         fill=tick_color, width=2)
                # 右侧刻度线
                draw.line([(img_area_x1, z_pixel_on_canvas), 
                          (img_area_x1 + tick_length, z_pixel_on_canvas)], 
                         fill=tick_color, width=2)
                
                # 绘制Z轴标签
                label_text = f"{z_current:.1f}"
                try:
                    bbox = draw.textbbox((0, 0), label_text, font=font_medium)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(label_text, font=font_medium)
                
                # 左侧标签
                label_x = img_area_x0 - tick_length - text_width - 5
                label_y = z_pixel_on_canvas - text_height / 2
                draw.text((label_x, label_y), label_text, fill=text_color, font=font_medium)
                
                # 右侧标签（可选）
                if is_major:
                    label_x_right = img_area_x1 + tick_length + 5
                    draw.text((label_x_right, label_y), label_text, fill=text_color, font=font_small)
            
            z_current += z_interval
        
        # 绘制原始图像区域的边框
        draw.rectangle([img_area_x0-1, img_area_y0-1, img_area_x1+1, img_area_y1+1], 
                      outline=border_color, width=2)
        
        # 添加坐标轴标签
        # X轴标签（底部中央）
        x_label = "X (meters)"
        try:
            bbox = draw.textbbox((0, 0), x_label, font=font_large)
            x_label_width = bbox[2] - bbox[0]
        except AttributeError:
            x_label_width, _ = draw.textsize(x_label, font=font_large)
        
        x_label_x = (new_width - x_label_width) / 2
        x_label_y = new_height - 25
        draw.text((x_label_x, x_label_y), x_label, fill=text_color, font=font_large)
        
        # Z轴标签（左侧中央，垂直）
        z_label = "Z (meters)"
        temp_img = Image.new('RGBA', (200, 30), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((0, 0), z_label, fill=text_color, font=font_large)
        rotated = temp_img.rotate(90, expand=True)
        
        # 计算Z轴标签位置
        z_label_x = 15
        z_label_y = (new_height - rotated.height) / 2
        new_image.paste(rotated, (int(z_label_x), int(z_label_y)), rotated)
        
        # 添加原点标记（如果在范围内）
        if world_min_x <= 0 <= world_max_x and world_min_z <= 0 <= world_max_z:
            origin_x = img_area_x0 + (0 - world_min_x) / x_range * original_width
            origin_z = img_area_y0 + (0 - world_min_z) / z_range * original_height
            
            # 绘制原点标记
            draw.ellipse([origin_x-6, origin_z-6, origin_x+6, origin_z+6], 
                        fill=(255, 255, 0), outline=(255, 255, 255), width=2)
            draw.text((origin_x+10, origin_z-10), "Origin (0,0)", fill=(255, 255, 0), font=font_small)
        
        # 添加比例尺信息
        scale_info = f"Scene: {x_range:.1f}m × {z_range:.1f}m | Grid: {x_interval}m × {z_interval}m"
        draw.text((img_area_x0, 5), scale_info, fill=text_color, font=font_small)
        
        # 添加指北针
        compass_x = new_width - 60
        compass_y = 50
        
        # 指北针背景
        draw.rectangle([compass_x-25, compass_y-25, compass_x+25, compass_y+25], 
                      outline=border_color, width=1, fill=(50, 50, 50))
        
        # 绘制指北针箭头 - 修复方向
        # X轴（红色）：水平向右
        draw.line([(compass_x-15, compass_y), (compass_x+15, compass_y)], fill=(255, 0, 0), width=3)
        # 箭头头部
        draw.line([(compass_x+15, compass_y), (compass_x+10, compass_y-5)], fill=(255, 0, 0), width=2)
        draw.line([(compass_x+15, compass_y), (compass_x+10, compass_y+5)], fill=(255, 0, 0), width=2)
        
        # Z轴（绿色）：垂直向下
        draw.line([(compass_x, compass_y-15), (compass_x, compass_y+15)], fill=(0, 255, 0), width=3)
        # 箭头头部
        draw.line([(compass_x, compass_y+15), (compass_x-5, compass_y+10)], fill=(0, 255, 0), width=2)
        draw.line([(compass_x, compass_y+15), (compass_x+5, compass_y+10)], fill=(0, 255, 0), width=2)
        
        # 指北针标签
        draw.text((compass_x+18, compass_y-5), "+X", fill=(255, 0, 0), font=font_small)
        draw.text((compass_x-8, compass_y+18), "+Z", fill=(0, 255, 0), font=font_small)
        
        return new_image
    
    def world_to_map_coords(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """将3D世界坐标转换为2D地图像素坐标"""
        if self.base_map_image is None:
            return (0, 0)
        
        # 获取带padding的地图尺寸
        padded_width, padded_height = self.base_map_image.size
        
        # 计算padding参数（使用类常量确保一致性）
        padding_left = self.MAP_PADDING_LEFT
        padding_bottom = self.MAP_PADDING_BOTTOM
        padding_top = self.MAP_PADDING_TOP
        padding_right = self.MAP_PADDING_RIGHT
        
        # 计算原始图像尺寸（未padding的尺寸）
        original_width = padded_width - padding_left - padding_right
        original_height = padded_height - padding_top - padding_bottom
        
        # 世界坐标范围
        world_min_x = self.scene_bounds[0][0]
        world_max_x = self.scene_bounds[1][0]
        world_min_z = self.scene_bounds[0][2]
        world_max_z = self.scene_bounds[1][2]
        
        # 线性映射到原始图像像素坐标
        px_in_original = (world_pos[0] - world_min_x) / (world_max_x - world_min_x) * original_width
        py_in_original = (world_pos[2] - world_min_z) / (world_max_z - world_min_z) * original_height
        
        # 转换到带padding的图像坐标
        px = int(px_in_original + padding_left)
        py = int(py_in_original + padding_top)
        
        # 确保坐标在图像范围内
        px = max(0, min(px, padded_width - 1))
        py = max(0, min(py, padded_height - 1))
        
        return (px, py)
    
    def map_coords_to_world(self, map_x: int, map_y: int) -> np.ndarray:
        """将2D地图像素坐标转换为3D世界坐标（反向转换）"""
        if self.base_map_image is None:
            return np.array([0.0, 0.0, 0.0])
        
        # 获取带padding的地图尺寸
        padded_width, padded_height = self.base_map_image.size
        
        # 计算padding参数（使用类常量确保一致性）
        padding_left = self.MAP_PADDING_LEFT
        padding_bottom = self.MAP_PADDING_BOTTOM
        padding_top = self.MAP_PADDING_TOP
        padding_right = self.MAP_PADDING_RIGHT
        
        # 计算原始图像尺寸
        original_width = padded_width - padding_left - padding_right
        original_height = padded_height - padding_top - padding_bottom
        
        # 转换到原始图像坐标
        px_in_original = map_x - padding_left
        py_in_original = map_y - padding_top
        
        # 世界坐标范围
        world_min_x = self.scene_bounds[0][0]
        world_max_x = self.scene_bounds[1][0]
        world_min_z = self.scene_bounds[0][2]
        world_max_z = self.scene_bounds[1][2]
        
        # 反向映射到世界坐标
        world_x = world_min_x + (px_in_original / original_width) * (world_max_x - world_min_x)
        world_z = world_min_z + (py_in_original / original_height) * (world_max_z - world_min_z)
        
        # 获取对应的Y坐标，但保持X和Z坐标精确
        # 不使用snap_point避免X,Z坐标偏移
        try:
            test_point = mn.Vector3(world_x, 0.0, world_z)
            snapped_point = self.sim.pathfinder.snap_point(test_point)
            # 只使用snapped_point的Y坐标，保持计算出的X,Z坐标不变
            world_y = snapped_point.y
        except Exception:
            # 如果获取Y坐标失败，使用默认值
            world_y = 0.0
        
        return np.array([world_x, world_y, world_z])
    
    def is_navigable(self, x: float, z: float) -> bool:
        """检查指定的(x,z)位置是否可导航"""
        # 使用pathfinder的snap_point来找到有效的3D点
        test_point = mn.Vector3(x, 0.0, z)
        snapped_point = self.sim.pathfinder.snap_point(test_point)
        return self.sim.pathfinder.is_navigable(snapped_point)
    
    def snap_to_navigable(self, x: float, z: float) -> Optional[np.ndarray]:
        """将2D坐标对齐到可导航的3D点"""
        test_point = mn.Vector3(x, 0.0, z)
        snapped_point = self.sim.pathfinder.snap_point(test_point)
        if self.sim.pathfinder.is_navigable(snapped_point):
            return np.array([snapped_point.x, snapped_point.y, snapped_point.z])
        return None
    
    def move_agent_to(self, world_pos: np.ndarray, rotation: Optional[Any] = None):
        """移动智能体到指定位置"""
        try:
            agent_state = habitat_sim.AgentState()
            
            # 处理不同类型的world_pos
            if hasattr(world_pos, 'x'):
                # magnum Vector3类型
                position_array = np.array([world_pos.x, world_pos.y, world_pos.z], dtype=np.float32)
            elif isinstance(world_pos, np.ndarray):
                position_array = world_pos.astype(np.float32)
            else:
                position_array = np.array(world_pos, dtype=np.float32)
            
            agent_state.position = position_array
            
            if rotation is not None:
                # 将所有rotation转换为[x, y, z, w]的numpy数组格式
                rotation_array = None
                
                if hasattr(rotation, 'vector') and hasattr(rotation, 'scalar'):
                    # magnum Quaternion类型
                    vec = rotation.vector
                    rotation_array = np.array([vec.x, vec.y, vec.z, rotation.scalar], dtype=np.float32)
                elif hasattr(rotation, 'x') and hasattr(rotation, 'y') and hasattr(rotation, 'z') and hasattr(rotation, 'w'):
                    # 如果是quaternion.quaternion类型或其他quaternion对象
                    rotation_array = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32)
                elif isinstance(rotation, np.ndarray) and len(rotation) == 4:
                    # numpy数组格式的quaternion [x, y, z, w]
                    rotation_array = rotation.astype(np.float32)
                elif hasattr(rotation, '__len__') and len(rotation) == 4:
                    # 其他类型的4元素数组
                    rotation_array = np.array(rotation, dtype=np.float32)
                else:
                    print(f"Warning: Unknown rotation type {type(rotation)}, using default")
                    rotation_array = np.array([0, 0, 0, 1], dtype=np.float32)
                
                # 归一化四元数
                if rotation_array is not None:
                    quat_norm = np.linalg.norm(rotation_array)
                    if quat_norm > 0:
                        rotation_array = rotation_array / quat_norm
                    agent_state.rotation = rotation_array
                else:
                    agent_state.rotation = np.array([0, 0, 0, 1], dtype=np.float32)
            else:
                # 默认朝向 - 使用numpy数组格式
                agent_state.rotation = np.array([0, 0, 0, 1], dtype=np.float32)
            
            self.agent.set_state(agent_state)
            
        except Exception as e:
            print(f"move_agent_to 失败: {e}")
            print(f"  world_pos: {world_pos}, type: {type(world_pos)}")
            print(f"  rotation: {rotation}, type: {type(rotation) if rotation is not None else None}")
            raise
    
    def get_agent_state(self) -> habitat_sim.AgentState:
        """获取当前智能体状态"""
        return self.agent.get_state()
    
    def find_path(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """寻找从起点到终点的路径"""
        start_vec = mn.Vector3(start[0], start[1], start[2])
        end_vec = mn.Vector3(end[0], end[1], end[2])
        
        # 使用ShortestPath对象进行寻路
        path_obj = habitat_sim.ShortestPath()
        path_obj.requested_start = start_vec
        path_obj.requested_end = end_vec
        
        found = self.sim.pathfinder.find_path(path_obj)
        if found and len(path_obj.points) > 0:
            # 转换为numpy数组列表
            return [np.array([p.x, p.y, p.z]) for p in path_obj.points]
        return []
    
    def get_fpv_observation(self) -> np.ndarray:
        """获取第一人称视角图像"""
        observations = self.sim.get_sensor_observations()
        return observations["color_sensor"]
    
    def rotate_sensor(self, pitch_deg: float = 0, yaw_deg: float = 0):
        """旋转传感器视角（不改变智能体身体朝向）"""
        agent_state = self.agent.get_state()
        
        # 获取当前传感器状态
        sensor_state = agent_state.sensor_states["color_sensor"]
        
        # 将角度转换为弧度
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        
        # 创建旋转四元数 - 使用正确的magnum四元数构造方式
        pitch_quat = mn.Quaternion.rotation(mn.Rad(pitch_rad), mn.Vector3.x_axis())
        yaw_quat = mn.Quaternion.rotation(mn.Rad(yaw_rad), mn.Vector3.y_axis())
        
        # 组合旋转
        rotation_quat = yaw_quat * pitch_quat
        
        # 从当前传感器旋转构造magnum四元数 - 修复构造方式
        current_rotation_array = sensor_state.rotation
        
        # 处理不同类型的旋转数据
        if hasattr(current_rotation_array, 'x'):
            # 如果是quaternion.quaternion类型
            rotation_values = np.array([current_rotation_array.x, current_rotation_array.y, current_rotation_array.z, current_rotation_array.w], dtype=np.float32)
        elif isinstance(current_rotation_array, np.ndarray):
            rotation_values = current_rotation_array.astype(np.float32)
        else:
            rotation_values = np.array(current_rotation_array, dtype=np.float32)
        
        # magnum四元数构造: [vector(x,y,z), scalar(w)]
        try:
            current_rotation = mn.Quaternion(
                mn.Vector3(float(rotation_values[0]), float(rotation_values[1]), float(rotation_values[2])),
                float(rotation_values[3])
            )
        except Exception as e:
            print(f"传感器旋转四元数构造失败: {e}")
            print(f"  current_rotation_array: {current_rotation_array}, type: {type(current_rotation_array)}")
            print(f"  rotation_values: {rotation_values}")
            raise
        
        new_rotation = current_rotation * rotation_quat
        
        # 更新传感器状态 - 确保正确的数据类型
        sensor_state.rotation = np.array([new_rotation.vector.x, 
                                        new_rotation.vector.y, 
                                        new_rotation.vector.z, 
                                        new_rotation.scalar], dtype=np.float32)
        
        # 重新设置智能体状态
        self.agent.set_state(agent_state)
    
    def close(self):
        """关闭模拟器"""
        if self.sim:
            self.sim.close()
    
    def verify_coordinate_conversion(self, world_pos: np.ndarray) -> dict:
        """验证世界坐标与地图坐标之间的转换是否正确（仅测试X和Z坐标）"""
        # 正向转换：世界坐标 -> 地图坐标
        map_x, map_y = self.world_to_map_coords(world_pos)
        
        # 反向转换：地图坐标 -> 世界坐标
        converted_world_pos = self.map_coords_to_world(map_x, map_y)
        
        # 只计算X和Z坐标的误差（忽略Y坐标，因为地图是2D的）
        original_xz = np.array([world_pos[0], world_pos[2]])
        converted_xz = np.array([converted_world_pos[0], converted_world_pos[2]])
        position_error = np.linalg.norm(original_xz - converted_xz)
        
        return {
            'original_world': world_pos,
            'map_coords': (map_x, map_y),
            'converted_world': converted_world_pos,
            'position_error': position_error,
            'error_acceptable': position_error < 0.1,  # 10cm以内认为是可接受的
            'note': 'Y坐标误差已排除（地图为2D）'
        }
    
    def get_position_with_navmesh_height(self, x: float, z: float) -> Optional[np.ndarray]:
        """获取指定(x,z)位置对应的navmesh上的3D点，不进行snap操作"""
        try:
            # 直接使用用户指定的x,z坐标，仅从navmesh获取对应的Y坐标
            test_point = mn.Vector3(x, 0.0, z)
            
            # 检查该点是否在navmesh表面上
            # 使用snap_point获取最近的navmesh点，但我们只使用其Y坐标
            snapped_point = self.sim.pathfinder.snap_point(test_point)
            
            if self.sim.pathfinder.is_navigable(snapped_point):
                # 返回用户指定的x,z坐标和navmesh的y坐标
                return np.array([x, snapped_point.y, z])
            else:
                # 如果snap_point后的位置不可导航，说明用户指定的位置不在有效区域
                return None
                
        except Exception as e:
            print(f"Error getting position with navmesh height: {e}")
            return None


class HabitatVideoGenerator:
    """Habitat视频生成器"""
    
    def __init__(self, scene_filepath: str, gpu_device_id: int = 0, 
                 fps: int = 30, output_dir: str = "./outputs"):
        self.scene_filepath = scene_filepath
        self.gpu_device_id = gpu_device_id
        self.fps = fps
        self.output_dir = output_dir
        
        # 动画参数 - 调整为更慢的速度
        self.rotation_step = 2.0  # 每2度旋转生成一帧（减慢旋转速度）
        self.movement_step = 0.1   # 每0.1米移动生成一帧（减慢移动速度）
        self.interpolation_steps = 30  # 路径段之间的插值步数（来自interactive_app）
        
        # 视频参数 - 提高精度
        self.video_width = 2048  # 左右各1024 (提高分辨率)
        self.video_height = 1024
        
        # 初始化模拟器
        self.simulator = None
        self.current_frames = []
        self.agent_initialized = False  # 标记代理是否已初始化位置
        self._initialize_simulator()
        
        # 验证坐标转换精度
        self._verify_coordinate_accuracy()
        
        print(f"Video generator initialized with {fps} FPS")
        print(f"Animation steps: {self.rotation_step}°/frame, {self.movement_step}m/frame")
        print("Agent will be positioned at the first command location")
    def _initialize_simulator(self):
        """初始化Habitat模拟器"""
        try:
            # 创建自定义的HabitatSimulator实例，指定GPU设备和更高分辨率
            self.simulator = CustomHabitatSimulator(
                scene_filepath=self.scene_filepath,
                gpu_device_id=self.gpu_device_id,
                resolution=(1024, 1024)  # 提高FPV分辨率
            )
            
            # 不立即设置代理位置，等待第一个指令来决定初始位置
            print("Simulator initialized. Agent position will be set with first command.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize simulator: {e}")
    
    def _reset_agent_to_position(self, x: float, z: float):
        """将代理重置到指定位置的可导航位置"""
        # 尝试对齐到可导航位置
        navigable_pos = self.simulator.snap_to_navigable(x, z)
        if navigable_pos is not None:
            self.simulator.move_agent_to(navigable_pos)
            print(f"Agent initialized at position ({navigable_pos[0]:.2f}, {navigable_pos[2]:.2f})")
            return True
        else:
            print(f"ERROR: Position ({x:.2f}, {z:.2f}) is not navigable")
            # 尝试找到最近的可导航点
            try:
                random_point = self.simulator.sim.pathfinder.get_random_navigable_point()
                self.simulator.move_agent_to(np.array([random_point.x, random_point.y, random_point.z]))
                print(f"Agent fallback to random navigable position ({random_point.x:.2f}, {random_point.z:.2f})")
                return True
            except Exception as e:
                print(f"ERROR: Could not find any navigable position: {e}")
                return False
    
    def process_command_sequence(self, commands: List[List[Union[str, float]]]) -> Optional[str]:
        """处理指令序列并生成视频"""
        self.current_frames = []
        start_time = time.time()
        
        try:
            # 如果代理还未初始化位置，使用第一个指令来设置初始位置
            if not self.agent_initialized and len(commands) > 0:
                first_command = commands[0]
                
                # 检查第一个指令是否是移动指令（包含坐标）
                if not isinstance(first_command[0], str):
                    # 第一个指令是移动指令 [x, z]
                    target_x = float(first_command[0])
                    target_z = float(first_command[1])
                    
                    # 将代理初始化到第一个指令的位置
                    success = self._reset_agent_to_position(target_x, target_z)
                    if not success:
                        print("ERROR: Failed to initialize agent at first command position")
                        return None
                    
                    self.agent_initialized = True
                    
                    # 添加初始帧
                    self._capture_frame()
                    
                    # 跳过第一个指令（因为代理已经在目标位置）
                    commands = commands[1:]
                    print(f"Agent initialized at first command position ({target_x:.2f}, {target_z:.2f})")
                else:
                    # 第一个指令是旋转指令，使用场景中心作为初始位置
                    center_x = (self.simulator.scene_bounds[0][0] + self.simulator.scene_bounds[1][0]) / 2
                    center_z = (self.simulator.scene_bounds[0][2] + self.simulator.scene_bounds[1][2]) / 2
                    
                    success = self._reset_agent_to_position(center_x, center_z)
                    if not success:
                        print("ERROR: Failed to initialize agent at scene center")
                        return None
                    
                    self.agent_initialized = True
                    
                    # 添加初始帧
                    self._capture_frame()
                    print("Agent initialized at scene center (first command is rotation)")
            else:
                # 代理已初始化，直接添加起始帧
                self._capture_frame()
            
            for i, command in enumerate(commands):
                print(f"  Executing command {i+1}/{len(commands)}: {command}")
                
                success = self._execute_command(command)
                if not success:
                    print(f"  Command {i+1} failed, stopping sequence")
                    break
            
            # 如果有帧，生成视频
            if len(self.current_frames) > 0:
                output_path = self._save_video()
                
                execution_time = time.time() - start_time
                print(f"  Generated {len(self.current_frames)} frames in {execution_time:.2f}s")
                
                return output_path
            else:
                print("  No frames captured")
                return None
                
        except Exception as e:
            print(f"ERROR: Command processing failed: {e}")
            # 即使出错，也尝试保存已有的帧
            if len(self.current_frames) > 0:
                return self._save_video()
            return None
    
    def _execute_command(self, command: List[Union[str, float]]) -> bool:
        """执行单个指令"""
        try:
            if isinstance(command[0], str):
                # 旋转指令 ["left"|"right", angle]
                direction = command[0]
                angle = float(command[1])
                return self._execute_rotation(direction, angle)
            else:
                # 移动指令 [x, z]
                target_x = float(command[0])
                target_z = float(command[1])
                return self._execute_movement(target_x, target_z)
                
        except Exception as e:
            print(f"    ERROR: Command execution failed: {e}")
            return False
    
    def _execute_rotation(self, direction: str, angle: float) -> bool:
        """执行旋转指令（平滑动画）"""
        try:
            total_steps = int(abs(angle) / self.rotation_step)
            if total_steps == 0:
                return True
                
            step_angle = self.rotation_step if angle > 0 else -self.rotation_step
            if direction == "left":
                step_angle = abs(step_angle)
            else:  # right
                step_angle = -abs(step_angle)
            
            for step in range(total_steps):
                # 执行一小步旋转
                self._rotate_agent(step_angle)
                
                # 捕获帧
                self._capture_frame()
            
            # 处理剩余的小数角度
            remaining_angle = angle - (total_steps * self.rotation_step)
            if abs(remaining_angle) > 0.1:  # 只有大于0.1度才执行
                final_step = remaining_angle if direction == "left" else -remaining_angle
                self._rotate_agent(final_step)
                self._capture_frame()
            
            return True
            
        except Exception as e:
            print(f"    Rotation failed: {e}")
            return False
    
    def _execute_movement(self, target_x: float, target_z: float) -> bool:
        """执行移动指令（直线移动，并在每一步检查碰撞）。
        
        该函数严格遵循指令，不使用pathfinder进行自动寻路。
        它会直接朝目标点移动，并在动画的每一步检查碰撞。
        如果检测到碰撞或目标点不可达，将停止执行并返回False。
        """
        try:
            # 检查目标位置是否可导航，主要是为了获取正确的Y坐标
            target_pos = self.simulator.snap_to_navigable(target_x, target_z)
            if target_pos is None:
                print(f"    ERROR: Target position ({target_x:.2f}, {target_z:.2f}) is not on a navigable surface. Halting.")
                return False  # 返回False将停止指令序列

            # 验证目标位置的坐标转换精度
            coord_check = self.simulator.verify_coordinate_conversion(target_pos)
            if not coord_check['error_acceptable']:
                print(f"    Warning: Target position coordinate conversion error {coord_check['position_error']:.3f}m")
            # 获取当前位置
            current_state = self.simulator.get_agent_state()
            current_pos = current_state.position

            # 计算距离
            distance = np.linalg.norm(target_pos - current_pos)

            # 如果距离很近，直接瞬移
            if distance < 0.1:
                self.simulator.move_agent_to(target_pos)
                self._capture_frame()
                return True

            # 直接执行直线移动。_execute_direct_movement函数内部包含碰撞检测。
            # 这样就实现了严格跟随指令，并在碰撞时停止。
            print(f"    Executing direct movement to ({target_x:.2f}, {target_z:.2f})")
            return self._execute_direct_movement(current_pos, target_pos)

        except Exception as e:
            print(f"    Movement failed: {e}")
            return False
    
    def _execute_direct_movement(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """直线移动（先转向再移动，避免漂移效果）"""
        try:
            # 计算移动方向和目标朝向
            direction = end_pos - start_pos
            distance = np.linalg.norm(direction)
            
            if distance < 0.01:  # 距离太近，直接移动
                self.simulator.move_agent_to(end_pos)
                self._capture_frame()
                return True
            
            # 归一化方向向量
            direction = direction / distance
            
            # 计算目标朝向角度
            angle = math.atan2(direction[0], direction[2])  # 使用+Z计算
            angle += math.pi  # 加180度修正
            
            # 创建目标旋转四元数
            rotation = mn.Quaternion.rotation(mn.Rad(angle), mn.Vector3.y_axis())
            target_rotation = np.array([rotation.vector.x, rotation.vector.y, 
                                      rotation.vector.z, rotation.scalar], dtype=np.float32)
            
            # 获取当前旋转
            current_state = self.simulator.get_agent_state()
            if hasattr(current_state.rotation, 'x'):
                start_rotation = np.array([
                    current_state.rotation.x, current_state.rotation.y, 
                    current_state.rotation.z, current_state.rotation.w
                ], dtype=np.float32)
            else:
                start_rotation = current_state.rotation.astype(np.float32)
            
            # 第一阶段：先执行视角转向（保持位置不变）
            rotation_steps = 15  # 转向帧数
            for step in range(rotation_steps):
                t = step / rotation_steps
                
                # 旋转插值
                try:
                    start_quat = mn.Quaternion(
                        mn.Vector3(start_rotation[0], start_rotation[1], start_rotation[2]),
                        start_rotation[3]
                    )
                    end_quat = mn.Quaternion(
                        mn.Vector3(target_rotation[0], target_rotation[1], target_rotation[2]),
                        target_rotation[3]
                    )
                    
                    # 球面线性插值
                    interpolated_quat = mn.Math.slerp(start_quat, end_quat, t)
                    interpolated_rotation = np.array([
                        interpolated_quat.vector.x, interpolated_quat.vector.y,
                        interpolated_quat.vector.z, interpolated_quat.scalar
                    ], dtype=np.float32)
                    
                except Exception:
                    # 如果球面插值失败，使用线性插值
                    interpolated_rotation = start_rotation + t * (target_rotation - start_rotation)
                    norm = np.linalg.norm(interpolated_rotation)
                    if norm > 0:
                        interpolated_rotation = interpolated_rotation / norm
                
                # 只改变旋转，保持当前位置
                self.simulator.move_agent_to(start_pos, interpolated_rotation)
                self._capture_frame()
            
            # 确保转向完成
            self.simulator.move_agent_to(start_pos, target_rotation)
            self._capture_frame()
            
            # 第二阶段：再执行位置移动（保持目标朝向）
            total_steps = max(1, int(distance / self.movement_step))
            direction_vector = (end_pos - start_pos) / total_steps
            
            for step in range(total_steps):
                # 计算下一个位置
                next_pos = start_pos + direction_vector * (step + 1)
                
                # 碰撞检测
                if not self.simulator.is_navigable(next_pos[0], next_pos[2]):
                    print(f"    ERROR: Collision detected at step {step+1}/{total_steps}")
                    return False
                
                # 移动代理（保持目标朝向）
                self.simulator.move_agent_to(next_pos, target_rotation)
                self._capture_frame()
            
            return True
            
        except Exception as e:
            print(f"    Direct movement失败: {e}")
            return False
    
    def _execute_path_movement(self, path: List[np.ndarray]) -> bool:
        """执行路径移动（先转向再移动，避免漂移效果）"""
        try:
            # 为每个路径段生成平滑动画
            for i in range(len(path) - 1):
                start_pos = path[i]
                end_pos = path[i + 1]
                
                # 计算朝向 - 完全复刻interactive_app的角度计算
                direction = end_pos - start_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    
                    # 在Habitat中，-Z轴是前方，复刻interactive_app的修正
                    angle = math.atan2(direction[0], direction[2])  # 使用+Z计算
                    angle += math.pi  # 加180度修正（复刻interactive_app）
                    
                    # 创建朝向目标的旋转四元数
                    rotation = mn.Quaternion.rotation(mn.Rad(angle), mn.Vector3.y_axis())
                    target_rotation = np.array([rotation.vector.x, rotation.vector.y, 
                                              rotation.vector.z, rotation.scalar], dtype=np.float32)
                else:
                    target_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
                
                # 获取当前旋转
                current_state = self.simulator.get_agent_state()
                if hasattr(current_state.rotation, 'x'):
                    start_rotation = np.array([
                        current_state.rotation.x, current_state.rotation.y, 
                        current_state.rotation.z, current_state.rotation.w
                    ], dtype=np.float32)
                else:
                    start_rotation = current_state.rotation.astype(np.float32)
                
                # 第一阶段：先执行视角转向（保持位置不变）
                rotation_steps = self.interpolation_steps // 2  # 转向用一半的帧数
                for step in range(rotation_steps):
                    t = step / rotation_steps
                    
                    # 旋转插值（球面线性插值）
                    try:
                        start_quat = mn.Quaternion(
                            mn.Vector3(start_rotation[0], start_rotation[1], start_rotation[2]),
                            start_rotation[3]
                        )
                        end_quat = mn.Quaternion(
                            mn.Vector3(target_rotation[0], target_rotation[1], target_rotation[2]),
                            target_rotation[3]
                        )
                        
                        # 球面线性插值
                        interpolated_quat = mn.Math.slerp(start_quat, end_quat, t)
                        interpolated_rotation = np.array([
                            interpolated_quat.vector.x, interpolated_quat.vector.y,
                            interpolated_quat.vector.z, interpolated_quat.scalar
                        ], dtype=np.float32)
                        
                    except Exception:
                        # 如果球面插值失败，使用线性插值
                        interpolated_rotation = start_rotation + t * (target_rotation - start_rotation)
                        norm = np.linalg.norm(interpolated_rotation)
                        if norm > 0:
                            interpolated_rotation = interpolated_rotation / norm
                    
                    # 只改变旋转，保持当前位置
                    self.simulator.move_agent_to(start_pos, interpolated_rotation)
                    self._capture_frame()
                
                # 确保转向完成
                self.simulator.move_agent_to(start_pos, target_rotation)
                self._capture_frame()
                
                # 第二阶段：再执行位置移动（保持目标朝向）
                movement_steps = self.interpolation_steps - rotation_steps  # 移动用剩余的帧数
                for step in range(movement_steps):
                    t = step / movement_steps
                    
                    # 位置插值
                    interpolated_pos = start_pos + t * (end_pos - start_pos)
                    
                    # 保持目标旋转不变
                    self.simulator.move_agent_to(interpolated_pos, target_rotation)
                    self._capture_frame()
                
                # 确保到达精确的路径点
                self.simulator.move_agent_to(end_pos, target_rotation)
                self._capture_frame()
            
            return True
            
        except Exception as e:
            print(f"    Path movement failed: {e}")
            return False
    
    def _rotate_agent(self, angle_degrees: float):
        """旋转代理（基于interactive_app的实现）"""
        agent_state = self.simulator.agent.get_state()
        
        # 将角度转换为弧度
        angle_rad = math.radians(angle_degrees)
        
        # 创建绕Y轴的旋转四元数
        rotation_quat = mn.Quaternion.rotation(mn.Rad(angle_rad), mn.Vector3.y_axis())
        
        # 获取当前旋转
        current_rotation = agent_state.rotation
        if hasattr(current_rotation, 'x'):
            current_quat = mn.Quaternion(
                mn.Vector3(current_rotation.x, current_rotation.y, current_rotation.z),
                current_rotation.w
            )
        else:
            current_quat = mn.Quaternion(
                mn.Vector3(current_rotation[0], current_rotation[1], current_rotation[2]),
                current_rotation[3]
            )
        
        # 应用旋转
        new_rotation = rotation_quat * current_quat
        
        # 更新代理状态
        agent_state.rotation = np.array([
            new_rotation.vector.x, new_rotation.vector.y,
            new_rotation.vector.z, new_rotation.scalar
        ], dtype=np.float32)
        
        self.simulator.agent.set_state(agent_state)
    
    def _capture_frame(self):
        """捕获当前帧（左右分屏，修复坐标转换问题，提高分辨率）"""
        try:
            # 检查代理是否已初始化
            if not self.agent_initialized:
                print("    Warning: Attempting to capture frame before agent initialization")
                return
            
            # 获取FPV图像
            fpv_image = self.simulator.get_fpv_observation()
            fpv_pil = Image.fromarray(fpv_image[..., :3], "RGB")
            
            # 获取俯视图（复用基础地图）
            map_image = self.simulator.base_map_image.copy()
            
            # 在原始地图上绘制代理（使用正确的坐标系）
            agent_state = self.simulator.get_agent_state()
            self._draw_agent_on_original_map(map_image, agent_state.position, agent_state.rotation)
            
            # 然后调整地图大小，保持纵横比，提高分辨率到1024x1024
            map_resized = self._resize_map_with_aspect_ratio(map_image, 1024, 1024)
            
            # 调整FPV图像大小到1024x1024
            fpv_resized = fpv_pil.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # 创建左右分屏图像 (2048x1024)
            combined = Image.new('RGB', (2048, 1024))
            combined.paste(fpv_resized, (0, 0))
            combined.paste(map_resized, (1024, 0))
            
            # 转换为numpy数组并添加到帧列表
            frame_array = np.array(combined)
            self.current_frames.append(frame_array)
            
        except Exception as e:
            print(f"    Failed to capture frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _resize_map_with_aspect_ratio(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """调整地图大小同时保持纵横比，多余空间用黑色填充"""
        original_width, original_height = image.size
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        if original_aspect > target_aspect:
            # 原图更宽，按宽度缩放
            new_width = target_width
            new_height = int(target_width / original_aspect)
        else:
            # 原图更高，按高度缩放
            new_height = target_height
            new_width = int(target_height * original_aspect)
        
        # 缩放图像
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建黑色背景
        result = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # 居中粘贴缩放后的图像
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(resized_image, (x_offset, y_offset))
        
        return result
    
    def _draw_agent_on_original_map(self, image: Image.Image, agent_pos: np.ndarray, 
                                   agent_rotation: Optional[np.ndarray] = None):
        """在原始地图上绘制代理位置和朝向（使用修复后的坐标系）"""
        draw = ImageDraw.Draw(image)
        
        # 使用修复后的HabitatSimulator的world_to_map_coords方法
        # 该方法已经修复了padding偏移问题，会正确处理坐标转换
        map_x, map_y = self.simulator.world_to_map_coords(agent_pos)
        
        # 验证坐标转换精度（用于调试）
        coord_check = self.simulator.verify_coordinate_conversion(agent_pos)
        if not coord_check['error_acceptable']:
            print(f"    Warning: Coordinate conversion error {coord_check['position_error']:.3f}m for agent position")
        
        # 确保坐标在原始地图范围内
        original_width, original_height = image.size
        map_x = max(0, min(map_x, original_width - 1))
        map_y = max(0, min(map_y, original_height - 1))
        
        # 绘制代理位置（红点）
        dot_radius = 8  # 固定大小，因为是在原始地图上绘制
        draw.ellipse([
            map_x - dot_radius, map_y - dot_radius,
            map_x + dot_radius, map_y + dot_radius
        ], fill=(255, 0, 0))
        
        # 绘制朝向箭头
        if agent_rotation is not None:
            try:
                if hasattr(agent_rotation, 'x'):
                    rotation_array = np.array([agent_rotation.x, agent_rotation.y, agent_rotation.z, agent_rotation.w], dtype=np.float32)
                elif isinstance(agent_rotation, np.ndarray):
                    rotation_array = agent_rotation.astype(np.float32)
                else:
                    rotation_array = np.array(agent_rotation, dtype=np.float32)
                
                if len(rotation_array) == 4:
                    quat = mn.Quaternion(
                        mn.Vector3(float(rotation_array[0]), float(rotation_array[1]), float(rotation_array[2])),
                        float(rotation_array[3])
                    )
                    
                    # 在Habitat中，-Z轴是前方
                    forward_vec = quat.transform_vector(mn.Vector3(0, 0, -1))
                    
                    # 计算箭头终点（固定长度）
                    arrow_length = 20
                    arrow_end_x = map_x + int(forward_vec.x * arrow_length)
                    arrow_end_y = map_y + int(forward_vec.z * arrow_length)
                    
                    # 确保箭头终点在图像范围内
                    arrow_end_x = max(0, min(arrow_end_x, original_width - 1))
                    arrow_end_y = max(0, min(arrow_end_y, original_height - 1))
                    
                    # 绘制箭头线
                    draw.line([(map_x, map_y), (arrow_end_x, arrow_end_y)], 
                             fill=(255, 0, 0), width=3)
                    
                    # 绘制箭头头部
                    angle = math.atan2(forward_vec.z, forward_vec.x)
                    arrow_head_length = 10
                    
                    head_angle1 = angle + math.pi * 0.8
                    head_angle2 = angle - math.pi * 0.8
                    
                    head_x1 = arrow_end_x + int(math.cos(head_angle1) * arrow_head_length)
                    head_y1 = arrow_end_y + int(math.sin(head_angle1) * arrow_head_length)
                    head_x2 = arrow_end_x + int(math.cos(head_angle2) * arrow_head_length)
                    head_y2 = arrow_end_y + int(math.sin(head_angle2) * arrow_head_length)
                    
                    # 确保箭头头部在图像范围内
                    head_x1 = max(0, min(head_x1, original_width - 1))
                    head_y1 = max(0, min(head_y1, original_height - 1))
                    head_x2 = max(0, min(head_x2, original_width - 1))
                    head_y2 = max(0, min(head_y2, original_height - 1))
                    
                    draw.line([(arrow_end_x, arrow_end_y), (head_x1, head_y1)], 
                             fill=(255, 0, 0), width=2)
                    draw.line([(arrow_end_x, arrow_end_y), (head_x2, head_y2)], 
                             fill=(255, 0, 0), width=2)
            except Exception as e:
                # 如果箭头绘制失败，只显示点
                print(f"    Warning: Failed to draw arrow: {e}")
                pass

    def _draw_agent_on_map(self, image: Image.Image, agent_pos: np.ndarray, 
                          agent_rotation: Optional[np.ndarray] = None):
        """在地图上绘制代理位置和朝向（使用修复后的坐标转换）"""
        draw = ImageDraw.Draw(image)
        
        # 使用修复后的坐标转换方法获取原始地图坐标
        original_map_coords = self.simulator.world_to_map_coords(agent_pos)
        
        # 验证坐标转换精度
        coord_check = self.simulator.verify_coordinate_conversion(agent_pos)
        if not coord_check['error_acceptable']:
            print(f"    Warning: Coordinate conversion error {coord_check['position_error']:.3f}m")
        
        # 获取原始地图和当前图像的尺寸
        original_map_width, original_map_height = self.simulator.base_map_image.size
        current_width, current_height = image.size
        
        # 计算缩放和偏移
        # 假设当前图像是通过_resize_map_with_aspect_ratio处理的
        original_aspect = original_map_width / original_map_height
        current_aspect = current_width / current_height
        
        if original_aspect > current_aspect:
            # 原图更宽，按宽度缩放
            scale = current_width / original_map_width
            scaled_width = current_width
            scaled_height = int(original_map_height * scale)
            x_offset = 0
            y_offset = (current_height - scaled_height) // 2
        else:
            # 原图更高，按高度缩放
            scale = current_height / original_map_height
            scaled_width = int(original_map_width * scale)
            scaled_height = current_height
            x_offset = (current_width - scaled_width) // 2
            y_offset = 0
        
        # 转换坐标到当前图像坐标系
        map_x = int(original_map_coords[0] * scale + x_offset)
        map_y = int(original_map_coords[1] * scale + y_offset)
        
        # 确保坐标在图像范围内
        map_x = max(0, min(map_x, current_width - 1))
        map_y = max(0, min(map_y, current_height - 1))
        
        # 绘制代理位置（红点）
        dot_radius = max(4, int(8 * scale))  # 根据缩放调整点的大小
        draw.ellipse([
            map_x - dot_radius, map_y - dot_radius,
            map_x + dot_radius, map_y + dot_radius
        ], fill=(255, 0, 0))
        
        # 绘制朝向箭头
        if agent_rotation is not None:
            try:
                if hasattr(agent_rotation, 'x'):
                    rotation_array = np.array([agent_rotation.x, agent_rotation.y, agent_rotation.z, agent_rotation.w], dtype=np.float32)
                elif isinstance(agent_rotation, np.ndarray):
                    rotation_array = agent_rotation.astype(np.float32)
                else:
                    rotation_array = np.array(agent_rotation, dtype=np.float32)
                
                if len(rotation_array) == 4:
                    quat = mn.Quaternion(
                        mn.Vector3(float(rotation_array[0]), float(rotation_array[1]), float(rotation_array[2])),
                        float(rotation_array[3])
                    )
                    
                    # 在Habitat中，-Z轴是前方
                    forward_vec = quat.transform_vector(mn.Vector3(0, 0, -1))
                    
                    # 计算箭头终点（根据缩放调整长度）
                    arrow_length = max(10, int(20 * scale))
                    arrow_end_x = map_x + int(forward_vec.x * arrow_length)
                    arrow_end_y = map_y + int(forward_vec.z * arrow_length)
                    
                    # 确保箭头终点在图像范围内
                    arrow_end_x = max(0, min(arrow_end_x, current_width - 1))
                    arrow_end_y = max(0, min(arrow_end_y, current_height - 1))
                    
                    # 绘制箭头线
                    line_width = max(2, int(3 * scale))
                    draw.line([(map_x, map_y), (arrow_end_x, arrow_end_y)], 
                             fill=(255, 0, 0), width=line_width)
                    
                    # 绘制箭头头部
                    angle = math.atan2(forward_vec.z, forward_vec.x)
                    arrow_head_length = max(5, int(10 * scale))
                    
                    head_angle1 = angle + math.pi * 0.8
                    head_angle2 = angle - math.pi * 0.8
                    
                    head_x1 = arrow_end_x + int(math.cos(head_angle1) * arrow_head_length)
                    head_y1 = arrow_end_y + int(math.sin(head_angle1) * arrow_head_length)
                    head_x2 = arrow_end_x + int(math.cos(head_angle2) * arrow_head_length)
                    head_y2 = arrow_end_y + int(math.sin(head_angle2) * arrow_head_length)
                    
                    # 确保箭头头部在图像范围内
                    head_x1 = max(0, min(head_x1, current_width - 1))
                    head_y1 = max(0, min(head_y1, current_height - 1))
                    head_x2 = max(0, min(head_x2, current_width - 1))
                    head_y2 = max(0, min(head_y2, current_height - 1))
                    
                    head_width = max(1, int(2 * scale))
                    draw.line([(arrow_end_x, arrow_end_y), (head_x1, head_y1)], 
                             fill=(255, 0, 0), width=head_width)
                    draw.line([(arrow_end_x, arrow_end_y), (head_x2, head_y2)], 
                             fill=(255, 0, 0), width=head_width)
            except Exception as e:
                # 如果箭头绘制失败，只显示点
                print(f"    Warning: Failed to draw arrow: {e}")
                pass
    
    def _save_video(self) -> str:
        """保存视频文件"""
        if len(self.current_frames) == 0:
            raise ValueError("No frames to save")
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        
        # 使用cv2.VideoWriter保存视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.video_width, self.video_height))
        
        try:
            for frame in self.current_frames:
                # 转换RGB到BGR（OpenCV格式）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            return output_path
            
        except Exception as e:
            writer.release()
            # 如果保存失败，尝试删除不完整的文件
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to save video: {e}")
    
    def get_agent_position(self) -> Tuple[float, float, float]:
        """获取代理当前位置"""
        if self.simulator and self.agent_initialized:
            state = self.simulator.get_agent_state()
            pos = state.position
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return (0.0, 0.0, 0.0)
    
    def get_agent_2d_map_coordinates(self) -> Tuple[float, float]:
        """获取代理在2D地图中的坐标"""
        if self.simulator and self.agent_initialized:
            state = self.simulator.get_agent_state()
            pos = state.position
            map_x, map_y = self.simulator.world_to_map_coords(pos)
            return (float(map_x), float(map_y))
        return (0.0, 0.0)
    
    def get_agent_coordinates_info(self) -> dict:
        """获取代理的完整坐标信息"""
        if self.simulator and self.agent_initialized:
            state = self.simulator.get_agent_state()
            pos = state.position
            map_x, map_y = self.simulator.world_to_map_coords(pos)
            
            return {
                '3d_world': {
                    'x': float(pos[0]),
                    'y': float(pos[1]), 
                    'z': float(pos[2])
                },
                '2d_map': {
                    'x': float(map_x),
                    'y': float(map_y)
                }
            }
        return {
            '3d_world': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            '2d_map': {'x': 0.0, 'y': 0.0}
        }
    
    def close(self):
        """关闭模拟器"""
        if self.simulator:
            self.simulator.close()
            self.simulator = None
    
    def _verify_coordinate_accuracy(self):
        """验证坐标转换精度，确保修复生效"""
        try:
            # 测试几个关键点的坐标转换精度
            test_points = [
                self.simulator.scene_center,  # 场景中心
                self.simulator.scene_bounds[0],  # 最小角
                self.simulator.scene_bounds[1],  # 最大角
            ]
            
            total_error = 0.0
            max_error = 0.0
            acceptable_count = 0
            
            print("=== 坐标转换精度验证 ===")
            for i, test_point in enumerate(test_points):
                result = self.simulator.verify_coordinate_conversion(test_point)
                total_error += result['position_error']
                max_error = max(max_error, result['position_error'])
                if result['error_acceptable']:
                    acceptable_count += 1
                
                print(f"  测试点{i+1}: 误差 {result['position_error']:.6f}m {'✓' if result['error_acceptable'] else '⚠'}")
            
            avg_error = total_error / len(test_points)
            success_rate = acceptable_count / len(test_points) * 100
            
            print(f"  平均误差: {avg_error:.6f}m")
            print(f"  最大误差: {max_error:.6f}m") 
            print(f"  精度可接受率: {success_rate:.1f}%")
            
            if avg_error < 0.1 and success_rate >= 80:
                print("  ✅ 坐标转换精度验证通过")
            else:
                print("  ⚠️ 坐标转换精度可能需要进一步优化")
                
        except Exception as e:
            print(f"  ❌ 坐标转换精度验证失败: {e}")
    
    def get_agent_coordinate_info(self) -> dict:
        """获取代理的详细坐标信息，包括转换精度"""
        if not self.simulator or not self.agent_initialized:
            return {
                'error': 'Agent not initialized yet',
                'scene_info': {
                    'bounds': {
                        'min': [float(self.simulator.scene_bounds[0][i]) for i in range(3)] if self.simulator else [],
                        'max': [float(self.simulator.scene_bounds[1][i]) for i in range(3)] if self.simulator else []
                    },
                    'center': [float(self.simulator.scene_center[i]) for i in range(3)] if self.simulator else []
                }
            }
        
        try:
            agent_state = self.simulator.get_agent_state()
            world_pos = agent_state.position
            
            # 获取地图坐标
            map_coords = self.simulator.world_to_map_coords(world_pos)
            
            # 验证坐标转换精度
            coord_check = self.simulator.verify_coordinate_conversion(world_pos)
            
            return {
                'world_position': {
                    'x': float(world_pos[0]),
                    'y': float(world_pos[1]), 
                    'z': float(world_pos[2])
                },
                'map_coordinates': {
                    'x': int(map_coords[0]),
                    'y': int(map_coords[1])
                },
                'coordinate_accuracy': {
                    'error': coord_check['position_error'],
                    'acceptable': coord_check['error_acceptable']
                },
                'scene_info': {
                    'bounds': {
                        'min': [float(self.simulator.scene_bounds[0][i]) for i in range(3)],
                        'max': [float(self.simulator.scene_bounds[1][i]) for i in range(3)]
                    },
                    'center': [float(self.simulator.scene_center[i]) for i in range(3)]
                },
                'agent_initialized': self.agent_initialized
            }
            
        except Exception as e:
            return {'error': str(e)}


class CustomHabitatSimulator(HabitatSimulator):
    """扩展的Habitat模拟器，支持物理智能体和多智能体场景"""
    
    def __init__(self, scene_filepath: str, resolution: Tuple[int, int] = (512, 512), 
                 gpu_device_id: int = 0, enable_physics: bool = True,
                 agent_model_path: Optional[str] = None):
        self.gpu_device_id = gpu_device_id
        self.enable_physics = enable_physics
        self.agent_model_path = agent_model_path
        
        # 调用父类初始化，但会重写_initialize_simulator方法
        super().__init__(scene_filepath, resolution)
        
    def _initialize_simulator(self):
        """重写父类方法，支持物理智能体"""
        # 配置后端
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_filepath
        backend_cfg.enable_physics = self.enable_physics
        backend_cfg.gpu_device_id = self.gpu_device_id
        backend_cfg.random_seed = 1
        
        # 配置FPV传感器
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [self.resolution[1], self.resolution[0]]
        fpv_sensor_spec.position = mn.Vector3(0, 1.5, 0)
        fpv_sensor_spec.hfov = 90.0
        
        # 配置深度传感器
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [self.resolution[1], self.resolution[0]]
        depth_sensor_spec.position = mn.Vector3(0, 1.5, 0)
        depth_sensor_spec.hfov = 90.0
        
        # 获取场景边界以计算正交传感器分辨率
        temp_sensor = habitat_sim.CameraSensorSpec()
        temp_sensor.uuid = "temp_sensor"
        temp_sensor.sensor_type = habitat_sim.SensorType.COLOR
        temp_sensor.resolution = [64, 64]
        
        temp_agent_cfg = habitat_sim.agent.AgentConfiguration()
        temp_agent_cfg.sensor_specifications = [temp_sensor]
        
        temp_sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [temp_agent_cfg]))
        
        # 确保导航网格已加载
        if not temp_sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            temp_sim.recompute_navmesh(temp_sim.pathfinder, navmesh_settings)
        
        # 使用导航网格顶点计算场景边界（更可靠）
        navmesh_vertices = np.array(temp_sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) == 0:
            print("警告：无法获取导航网格顶点，使用默认场景边界")
            world_size_x = 10.0
            world_size_z = 10.0
        else:
            min_bounds = navmesh_vertices.min(axis=0)
            max_bounds = navmesh_vertices.max(axis=0)
            world_size_x = max_bounds[0] - min_bounds[0]
            world_size_z = max_bounds[2] - min_bounds[2]
            print(f"场景导航区域尺寸: {world_size_x:.2f} x {world_size_z:.2f}")
        
        temp_sim.close()
        
        # 确保尺寸不为零
        if world_size_x <= 0:
            world_size_x = 10.0
        if world_size_z <= 0:
            world_size_z = 10.0
        
        # 使用TopV.py的成功配置 - 固定4096x4096分辨率
        print(f"场景尺寸: {world_size_x:.2f} x {world_size_z:.2f}")
        
        # 配置正交传感器（完全模仿TopV.py的配置）
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.resolution = [4096, 4096]  # TopV.py使用的固定高分辨率
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        ortho_sensor_spec.far = 1000.0
        ortho_sensor_spec.near = 0.01
        ortho_sensor_spec.hfov = 90
        ortho_sensor_spec.ortho_scale = 0.05  # TopV.py的关键参数
        ortho_sensor_spec.clear_color = [0., 0., 0., 0.]
        # 不设置uuid，使用默认传感器名称（与TopV.py一致）
        
        print(f"正交传感器配置: 4096x4096, ortho_scale=0.05")
        
        # 配置智能体
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, depth_sensor_spec, ortho_sensor_spec]
        
        # 如果指定了智能体模型，尝试加载物理智能体
        if self.agent_model_path and os.path.exists(self.agent_model_path):
            try:
                # 验证URDF文件存在
                if not os.path.isfile(self.agent_model_path):
                    raise FileNotFoundError(f"URDF file not found: {self.agent_model_path}")
                
                # 在Habitat-Sim中加载物理智能体需要通过ArticulatedObject
                # 但智能体本身仍然使用标准配置
                print(f"✓ Physical agent model found: {self.agent_model_path}")
                print(f"✓ Will load URDF robot after simulator initialization")
                
                # 设置标准智能体参数
                agent_cfg.height = 1.5
                agent_cfg.radius = 0.4
                
            except Exception as e:
                print(f"Warning: Failed to validate physical agent, using default: {e}")
                agent_cfg.height = 1.5
                agent_cfg.radius = 0.4
        else:
            # 使用默认虚拟智能体
            agent_cfg.height = 1.5
            agent_cfg.radius = 0.4
            if self.agent_model_path:
                print(f"Warning: Agent model not found: {self.agent_model_path}")
                print(f"Using default virtual agent instead")
        
        # 配置动作空间
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        # 创建完整配置
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        # 实例化模拟器
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        
        # 如果指定了物理智能体模型，在此处加载它
        self.robot_object = None
        if self.agent_model_path and os.path.exists(self.agent_model_path):
            try:
                # 加载URDF机器人作为ArticulatedObject
                obj_mgr = self.sim.get_object_template_manager()
                
                # 创建机器人模板
                robot_template_id = obj_mgr.load_configs(self.agent_model_path)[0]
                
                # 实例化机器人对象
                articulated_obj_mgr = self.sim.get_articulated_object_manager()
                self.robot_object = articulated_obj_mgr.add_articulated_object_from_template(
                    robot_template_id
                )
                
                # 设置机器人初始位置
                robot_initial_transform = mn.Matrix4.translation(mn.Vector3(0, 0, 0))
                self.robot_object.transformation = robot_initial_transform
                
                print(f"✓ Successfully loaded physical robot: {self.agent_model_path}")
                print(f"✓ Robot object ID: {self.robot_object.object_id}")
                print(f"✓ Robot has {len(self.robot_object.joint_positions)} joints")
                
            except Exception as e:
                print(f"Warning: Failed to load physical robot as ArticulatedObject: {e}")
                print(f"Continuing with virtual agent only")
                self.robot_object = None
        
        # 获取场景信息
        # 使用导航网格顶点计算边界（更可靠）
        if not self.sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        
        navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) == 0:
            print("警告：无法获取导航网格顶点，使用默认场景边界")
            # 使用默认边界
            min_bounds = [-5.0, 0.0, -5.0]
            max_bounds = [5.0, 3.0, 5.0]
        else:
            min_bounds = navmesh_vertices.min(axis=0).tolist()
            max_bounds = navmesh_vertices.max(axis=0).tolist()
        
        self.scene_bounds = (min_bounds, max_bounds)
        self.scene_center = ((np.array(min_bounds) + np.array(max_bounds)) / 2.0).tolist()
        self.scene_size = (np.array(max_bounds) - np.array(min_bounds)).tolist()
        self.ortho_scale = max(self.scene_size[0], self.scene_size[2]) / 2.0
        
        print(f"Custom simulator initialized with GPU device {self.gpu_device_id}")
        print(f"Physics enabled: {self.enable_physics}")
        print(f"Agent height: {agent_cfg.height}m, radius: {agent_cfg.radius}m")
        print(f"Scene bounds: {self.scene_bounds}")
        
        # 报告物理智能体状态
        if self.robot_object is not None:
            print(f"✓ Physical robot successfully loaded and active")
            print(f"  - Model: {os.path.basename(self.agent_model_path)}")
            print(f"  - Object ID: {self.robot_object.object_id}")
            print(f"  - Joint count: {len(self.robot_object.joint_positions)}")
        else:
            print(f"→ Using virtual agent (no physical robot loaded)")
    
    def check_agent_collision(self, position: np.ndarray, radius: float = 0.4) -> bool:
        """检查智能体在指定位置是否会发生碰撞"""
        try:
            # 检查中心点是否可导航
            center_point = mn.Vector3(position[0], position[1], position[2])
            if not self.sim.pathfinder.is_navigable(center_point):
                return True
            
            # 检查智能体边界是否可导航
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                check_x = position[0] + radius * np.cos(angle)
                check_z = position[2] + radius * np.sin(angle)
                check_point = mn.Vector3(check_x, position[1], check_z)
                
                if not self.sim.pathfinder.is_navigable(check_point):
                    return True
            
            return False
            
        except Exception as e:
            print(f"Collision check failed: {e}")
            return True  # 安全起见，认为有碰撞
    
    def get_agent_bounding_box(self, position: np.ndarray, radius: float = 0.4) -> List[np.ndarray]:
        """获取智能体的边界框点"""
        points = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x = position[0] + radius * np.cos(angle)
            z = position[2] + radius * np.sin(angle)
            points.append(np.array([x, position[1], z]))
        return points
    
    def sync_robot_with_agent(self):
        """同步物理机器人与虚拟智能体的位置"""
        if self.robot_object is not None and self.agent is not None:
            try:
                # 获取虚拟智能体的状态
                agent_state = self.agent.get_state()
                
                # 将虚拟智能体的位置和旋转应用到物理机器人
                transform = mn.Matrix4.from_(
                    agent_state.rotation.to_matrix(),
                    agent_state.position
                )
                self.robot_object.transformation = transform
                
                return True
            except Exception as e:
                print(f"Warning: Failed to sync robot with agent: {e}")
                return False
        return False
    
    def get_robot_status(self) -> dict:
        """获取物理机器人状态信息"""
        if self.robot_object is None:
            return {
                "has_robot": False,
                "model_path": self.agent_model_path,
                "status": "No physical robot loaded"
            }
        
        try:
            return {
                "has_robot": True,
                "model_path": self.agent_model_path,
                "object_id": self.robot_object.object_id,
                "joint_count": len(self.robot_object.joint_positions),
                "position": self.robot_object.translation.tolist(),
                "rotation": self.robot_object.rotation.to_matrix().tolist(),
                "status": "Physical robot active"
            }
        except Exception as e:
            return {
                "has_robot": True,
                "model_path": self.agent_model_path,
                "status": f"Error reading robot status: {e}"
            }
    
    def move_robot_to_position(self, position: np.ndarray, rotation: Optional[np.ndarray] = None):
        """直接移动物理机器人到指定位置"""
        if self.robot_object is None:
            return False
            
        try:
            # 设置位置
            if rotation is not None:
                # 使用提供的旋转
                quat = mn.Quaternion.from_matrix(mn.Matrix3x3(rotation.reshape(3, 3)))
                transform = mn.Matrix4.from_(quat.to_matrix(), mn.Vector3(position))
            else:
                # 只设置位置，保持当前旋转
                current_rotation = self.robot_object.rotation
                transform = mn.Matrix4.from_(current_rotation.to_matrix(), mn.Vector3(position))
            
            self.robot_object.transformation = transform
            
            # 同步虚拟智能体位置
            agent_state = habitat_sim.AgentState()
            agent_state.position = position
            if rotation is not None:
                agent_state.rotation = mn.Quaternion.from_matrix(mn.Matrix3x3(rotation.reshape(3, 3)))
            else:
                agent_state.rotation = self.robot_object.rotation
            
            self.agent.set_state(agent_state)
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to move robot to position: {e}")
            return False

    def set_agent_state(self, position: np.ndarray, rotation: np.ndarray):
        """设置智能体状态（位置和旋转）"""
        try:
            # 创建AgentState对象
            agent_state = habitat_sim.AgentState()
            agent_state.position = position
            
            # 处理旋转 - 确保转换为正确格式
            try:
                if isinstance(rotation, mn.Quaternion):
                    # 如果是Magnum Quaternion对象，转换为numpy数组
                    rotation_array = np.array([
                        rotation.vector.x, rotation.vector.y, rotation.vector.z, rotation.scalar
                    ], dtype=np.float32)
                    agent_state.rotation = rotation_array
                elif hasattr(rotation, '__len__') and len(rotation) == 4:
                    # 四元数格式 [x, y, z, w] - 确保转换为numpy数组
                    rotation_array = np.array([
                        float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])
                    ], dtype=np.float32)
                    agent_state.rotation = rotation_array
                else:
                    # 默认旋转 - 单位四元数
                    agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            except Exception as rot_e:
                print(f"  Rotation conversion error: {rot_e}")
                # 使用默认旋转作为回退
                agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # 设置智能体状态
            self.agent.set_state(agent_state)
            
        except Exception as e:
            print(f"Warning: Failed to set agent state: {e}")
            # 添加更多调试信息
            print(f"  Position type: {type(position)}, value: {position}")
            print(f"  Rotation type: {type(rotation)}, value: {rotation}")
            print(f"Warning: Failed to set agent state: {e}")

    def move_agent_to(self, position: np.ndarray, rotation: np.ndarray):
        """移动智能体到指定位置（兼容性方法）"""
        self.set_agent_state(position, rotation)
