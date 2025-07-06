#!/usr/bin/env python3
"""
交互式Habitat-sim导航应用程序
使用PyQt5创建GUI界面，左侧显示第一人称视角，右侧显示动态俯视地图
"""

import sys
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional
import time

# GUI相关导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                           QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QStatusBar, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

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
        
        # 手动设置场景边界（不使用pathfinder）
        # 这里使用一个合理的默认范围，您可以根据实际场景调整
        world_size_x = 20.0  # 假设场景X方向20米
        world_size_z = 20.0  # 假设场景Z方向20米
        temp_sim.close()
        
        # 计算保持纵横比的地图分辨率
        max_resolution = 1024
        aspect_ratio = world_size_x / world_size_z
        
        if aspect_ratio > 1:
            # 宽度大于高度
            map_width = max_resolution
            map_height = int(max_resolution / aspect_ratio)
        else:
            # 高度大于宽度
            map_height = max_resolution
            map_width = int(max_resolution * aspect_ratio)
        
        print(f"场景尺寸比: {world_size_x:.2f} x {world_size_z:.2f} (比例: {aspect_ratio:.2f})")
        print(f"地图分辨率: {map_width} x {map_height}")
        
        # 配置正交传感器（用于生成俯视地图）
        # 注意：habitat-sim的分辨率格式是[height, width]
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.resolution = [map_height, map_width]  # [height, width]
        ortho_sensor_spec.position = mn.Vector3(0, 0, 0)
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        
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
        
        # 手动设置场景边界信息（不使用pathfinder）
        # 您可以根据实际场景调整这些值
        self.scene_bounds = [
            np.array([-10.0, -5.0, -10.0]),  # 最小边界 [x, y, z]
            np.array([10.0, 5.0, 10.0])     # 最大边界 [x, y, z]
        ]
        self.scene_center = (self.scene_bounds[0] + self.scene_bounds[1]) / 2.0
        self.scene_size = self.scene_bounds[1] - self.scene_bounds[0]
        self.ortho_scale = max(self.scene_size[0], self.scene_size[2]) / 2.0
        
        print(f"场景边界: {self.scene_bounds}")
        print(f"场景中心: {self.scene_center}")
        print(f"场景尺寸: {self.scene_size}")
    
    def _generate_base_map(self):
        """生成带坐标系的基础俯视地图"""
        # 设置正交相机位置和参数
        camera_height = self.scene_bounds[1][1] + 5.0
        camera_position = mn.Vector3(self.scene_center[0], camera_height, self.scene_center[2])
        
        # 设置智能体状态以获取俯视图
        agent_state = habitat_sim.AgentState()
        agent_state.position = camera_position
        agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068])  # 朝下看
        self.agent.set_state(agent_state)
        
        # 正交传感器已在初始化时配置为ORTHOGRAPHIC类型
        # 这里不需要重新配置sensor_subtype
        
        # 获取俯视图
        observations = self.sim.get_sensor_observations()
        ortho_img = observations["ortho_sensor"]
        
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
            return np.array([0.0, 1.5, 0.0])
        
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
        
        # 使用固定的Y坐标，不调用pathfinder
        return np.array([world_x, 1.5, world_z])
    
    def is_navigable(self, x: float, z: float) -> bool:
        """直接返回True，不进行导航检查"""
        return True
    
    def snap_to_navigable(self, x: float, z: float) -> Optional[np.ndarray]:
        """直接使用用户输入的坐标，不进行对齐"""
        # 使用固定的Y坐标（地面高度）
        return np.array([x, 1.5, z])
    
    def move_agent_to(self, world_pos: np.ndarray, rotation: Optional[np.ndarray] = None):
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
                # 处理不同类型的rotation
                if hasattr(rotation, 'x'):
                    # 如果是quaternion.quaternion类型
                    rotation_array = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32)
                elif isinstance(rotation, np.ndarray):
                    rotation_array = rotation.astype(np.float32)
                else:
                    rotation_array = np.array(rotation, dtype=np.float32)
                
                # 验证四元数格式
                if len(rotation_array) != 4:
                    raise ValueError(f"四元数必须有4个元素，得到 {len(rotation_array)} 个")
                
                agent_state.rotation = rotation_array
            else:
                # 默认朝向 - 使用正确的四元数格式 [x, y, z, w]
                agent_state.rotation = np.array([0, 0, 0, 1], dtype=np.float32)
            
            self.agent.set_state(agent_state)
            
        except Exception as e:
            print(f"move_agent_to 失败: {e}")
            print(f"  world_pos: {world_pos}, type: {type(world_pos)}")
            print(f"  rotation: {rotation}, type: {type(rotation) if rotation is not None else None}")
            if rotation is not None and hasattr(rotation, 'x'):
                print(f"  quaternion components: x={rotation.x}, y={rotation.y}, z={rotation.z}, w={rotation.w}")
            raise
    
    def get_agent_state(self) -> habitat_sim.AgentState:
        """获取当前智能体状态"""
        return self.agent.get_state()
    
    def find_path(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """直接返回起点和终点，不进行路径规划"""
        return [start, end]
    
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
        """验证世界坐标与地图坐标之间的转换是否正确"""
        # 正向转换：世界坐标 -> 地图坐标
        map_x, map_y = self.world_to_map_coords(world_pos)
        
        # 反向转换：地图坐标 -> 世界坐标
        converted_world_pos = self.map_coords_to_world(map_x, map_y)
        
        # 计算误差
        position_error = np.linalg.norm(world_pos - converted_world_pos)
        
        return {
            'original_world': world_pos,
            'map_coords': (map_x, map_y),
            'converted_world': converted_world_pos,
            'position_error': position_error,
            'error_acceptable': position_error < 0.1  # 10cm以内认为是可接受的
        }


class HabitatNavigatorApp(QMainWindow):
    """主应用程序类"""
    
    def __init__(self, scene_filepath: str):
        super().__init__()
        self.scene_filepath = scene_filepath
        self.simulator = None
        self.current_map_image = None
        self.is_moving = False
        self.path_waypoints = []
        self.current_waypoint_index = 0
        self.animation_timer = QTimer()
        
        # 平滑动画相关变量
        self.interpolation_steps = 20  # 每两个路径点之间的插值步数
        self.current_interpolation_step = 0
        self.animation_start_pos = None
        self.animation_end_pos = None
        self.animation_start_rotation = None
        self.animation_end_rotation = None
        
        self.init_ui()
        self.init_simulator()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Habitat-sim 导航应用程序")
        self.setGeometry(100, 100, 1200, 600)
        
        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主水平布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：第一人称视角
        self.fpv_label = QLabel()
        self.fpv_label.setFixedSize(512, 512)
        self.fpv_label.setStyleSheet("border: 1px solid black;")
        self.fpv_label.setText("第一人称视角\n(FPV)")
        self.fpv_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.fpv_label)
        
        # 右侧：垂直布局
        right_layout = QVBoxLayout()
        
        # 俯视地图
        self.map_label = QLabel()
        self.map_label.setFixedSize(512, 512)
        self.map_label.setStyleSheet("border: 1px solid black;")
        self.map_label.setText("俯视地图\n(正在生成...)")
        self.map_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.map_label)
        
        # 输入框
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("输入坐标 (x, z) 直接移动或视角命令 (如: right 30)")
        self.command_input.returnPressed.connect(self.process_command)
        right_layout.addWidget(self.command_input)
        
        # 执行按钮
        self.execute_button = QPushButton("执行")
        self.execute_button.clicked.connect(self.process_command)
        right_layout.addWidget(self.execute_button)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc;")
        self.status_label.setWordWrap(True)  # 允许文本换行
        self.status_label.setMinimumHeight(80)  # 设置最小高度以显示多行文本
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 文本对齐到左上角
        right_layout.addWidget(self.status_label)
        
        main_layout.addLayout(right_layout)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("正在初始化...")
        
        # 设置动画定时器
        self.animation_timer.timeout.connect(self.animate_movement)
        
    def init_simulator(self):
        """初始化模拟器"""
        try:
            self.status_bar.showMessage("正在初始化Habitat模拟器...")
            self.simulator = HabitatSimulator(self.scene_filepath)
            
            # 显示基础地图
            self.update_map_display()
            self.status_bar.showMessage("模拟器初始化完成")
            
            # 提供更好的初始化引导
            bounds = self.simulator.scene_bounds
            center_x = (bounds[0][0] + bounds[1][0]) / 2
            center_z = (bounds[0][2] + bounds[1][2]) / 2
            
            guide_text = (f"初始化完成！\n"
                         f"请输入坐标开始移动 (格式: x, z)\n"
                         f"建议起始坐标: {center_x:.1f}, {center_z:.1f}\n"
                         f"注意：智能体会直接移动到指定坐标，使用直线动画\n"
                         f"场景范围: X({bounds[0][0]:.1f}~{bounds[1][0]:.1f}), "
                         f"Z({bounds[0][2]:.1f}~{bounds[1][2]:.1f})")
            
            self.status_label.setText(guide_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化模拟器失败: {str(e)}")
            self.close()
    
    def update_map_display(self, agent_pos: Optional[np.ndarray] = None, 
                          agent_rotation: Optional[np.ndarray] = None):
        """更新地图显示"""
        if not self.simulator or not self.simulator.base_map_image:
            return
        
        # 复制基础地图
        map_image = self.simulator.base_map_image.copy()
        
        # 如果有智能体位置，绘制智能体标记
        if agent_pos is not None:
            self.draw_agent_on_map(map_image, agent_pos, agent_rotation)
        
        # 转换为QPixmap并显示
        self.current_map_image = map_image
        qimage = self.pil_to_qimage(map_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # 缩放到适合标签大小
        scaled_pixmap = pixmap.scaled(self.map_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.map_label.setPixmap(scaled_pixmap)
    
    def draw_agent_on_map(self, image: Image.Image, agent_pos: np.ndarray, 
                         agent_rotation: Optional[np.ndarray] = None):
        """在地图上绘制智能体位置和朝向"""
        draw = ImageDraw.Draw(image)
        
        # 转换世界坐标到地图坐标
        map_x, map_y = self.simulator.world_to_map_coords(agent_pos)
        
        # 绘制智能体位置（红点）
        dot_radius = 8
        draw.ellipse([
            map_x - dot_radius, map_y - dot_radius,
            map_x + dot_radius, map_y + dot_radius
        ], fill=(255, 0, 0))
        
        # 绘制朝向箭头
        if agent_rotation is not None:
            try:
                # 处理不同类型的agent_rotation
                if hasattr(agent_rotation, 'x'):
                    # 如果是quaternion.quaternion类型
                    rotation_array = np.array([agent_rotation.x, agent_rotation.y, agent_rotation.z, agent_rotation.w], dtype=np.float32)
                elif isinstance(agent_rotation, np.ndarray):
                    # 如果是numpy数组
                    rotation_array = agent_rotation.astype(np.float32)
                else:
                    # 尝试转换为numpy数组
                    rotation_array = np.array(agent_rotation, dtype=np.float32)
                
                if len(rotation_array) != 4:
                    print(f"警告: agent_rotation长度不正确: {len(rotation_array)}")
                    return  # 跳过箭头绘制
                
                # 从四元数计算朝向角度 - 修复前方向量
                # agent_rotation格式: [x, y, z, w]
                quat = mn.Quaternion(
                    mn.Vector3(float(rotation_array[0]), float(rotation_array[1]), float(rotation_array[2])),
                    float(rotation_array[3])
                )
                # 在Habitat中，-Z轴是前方
                forward_vec = quat.transform_vector(mn.Vector3(0, 0, -1))
                
                # 计算箭头终点
                arrow_length = 20
                arrow_end_x = map_x + int(forward_vec.x * arrow_length)
                arrow_end_y = map_y + int(forward_vec.z * arrow_length)
                
                # 绘制箭头线
                draw.line([(map_x, map_y), (arrow_end_x, arrow_end_y)], 
                         fill=(255, 0, 0), width=3)
                
                # 绘制箭头头部
                angle = math.atan2(forward_vec.z, forward_vec.x)
                arrow_head_length = 10
                
                # 计算箭头头部的两个点
                head_angle1 = angle + math.pi * 0.8
                head_angle2 = angle - math.pi * 0.8
                
                head_x1 = arrow_end_x + int(math.cos(head_angle1) * arrow_head_length)
                head_y1 = arrow_end_y + int(math.sin(head_angle1) * arrow_head_length)
                head_x2 = arrow_end_x + int(math.cos(head_angle2) * arrow_head_length)
                head_y2 = arrow_end_y + int(math.sin(head_angle2) * arrow_head_length)
                
                draw.line([(arrow_end_x, arrow_end_y), (head_x1, head_y1)], 
                         fill=(255, 0, 0), width=2)
                draw.line([(arrow_end_x, arrow_end_y), (head_x2, head_y2)], 
                         fill=(255, 0, 0), width=2)
            except Exception as e:
                # 如果四元数处理失败，只绘制点，不绘制箭头
                print(f"箭头绘制失败: {e}")
                print(f"  agent_rotation: {agent_rotation}, type: {type(agent_rotation)}")
                if hasattr(agent_rotation, 'x'):
                    print(f"  quaternion components: x={agent_rotation.x}, y={agent_rotation.y}, z={agent_rotation.z}, w={agent_rotation.w}")
                if isinstance(agent_rotation, np.ndarray):
                    print(f"  shape: {agent_rotation.shape}, dtype: {agent_rotation.dtype}")
    
    def update_fpv_display(self):
        """更新第一人称视角显示"""
        if not self.simulator:
            return
        
        try:
            # 获取FPV图像
            fpv_image = self.simulator.get_fpv_observation()
            
            # 检查图像格式并进行适当转换
            if len(fpv_image.shape) == 3:
                height, width, channels = fpv_image.shape
                
                if channels == 4:  # RGBA格式
                    # 转换RGBA到RGB
                    fpv_rgb = fpv_image[:, :, :3].copy()  # 只取RGB通道并复制
                elif channels == 3:  # RGB格式
                    fpv_rgb = fpv_image.copy()
                else:
                    print(f"不支持的通道数: {channels}")
                    return
                
                # 确保数据是连续的并转换为bytes
                fpv_rgb = np.ascontiguousarray(fpv_rgb)
                bytes_per_line = width * 3
                
                qimage = QImage(fpv_rgb.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                print(f"不支持的图像形状: {fpv_image.shape}")
                return
            
            pixmap = QPixmap.fromImage(qimage)
            
            # 缩放到适合标签大小
            scaled_pixmap = pixmap.scaled(self.fpv_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.fpv_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"FPV显示更新失败: {e}")
            print(f"  FPV图像形状: {fpv_image.shape if 'fpv_image' in locals() else 'N/A'}")
            print(f"  FPV图像类型: {fpv_image.dtype if 'fpv_image' in locals() else 'N/A'}")
    
    def pil_to_qimage(self, pil_image: Image.Image) -> QImage:
        """将PIL图像转换为QImage"""
        rgb_image = pil_image.convert('RGB')
        width, height = rgb_image.size
        
        qimage = QImage(width, height, QImage.Format_RGB888)
        
        for y in range(height):
            for x in range(width):
                r, g, b = rgb_image.getpixel((x, y))
                qimage.setPixel(x, y, (r << 16) | (g << 8) | b)
        
        return qimage
    
    def process_command(self):
        """处理用户输入的命令"""
        if self.is_moving:
            self.status_label.setText("智能体正在移动中，请稍候...")
            return
        
        command = self.command_input.text().strip()
        if not command:
            return
        
        self.command_input.clear()
        
        # 解析命令
        if ',' in command:
            # 坐标命令
            self.process_coordinate_command(command)
        else:
            # 视角命令
            self.process_view_command(command)
    
    def process_coordinate_command(self, command: str):
        """处理坐标命令"""
        try:
            parts = command.split(',')
            if len(parts) != 2:
                self.status_label.setText("坐标格式错误，请使用: x, z")
                return
            
            x = float(parts[0].strip())
            z = float(parts[1].strip())
            
            self.status_label.setText(f"处理坐标 ({x:.1f}, {z:.1f})...")
            
            # 直接创建目标位置，不进行导航检查
            target_pos = np.array([x, 1.5, z], dtype=np.float32)  # 使用固定的1.5米高度
            
            self.status_label.setText(f"目标位置: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
            
            # 获取当前智能体状态
            try:
                current_state = self.simulator.get_agent_state()
                current_pos = current_state.position
                
                # 检查是否是第一次设置位置
                scene_center = self.simulator.scene_center
                is_first_placement = (
                    np.allclose(current_pos, [0, 0, 0], atol=0.1) or  # 默认原点
                    np.allclose(current_pos, scene_center, atol=0.5) or  # 场景中心附近
                    np.linalg.norm(current_pos) < 0.1  # 接近原点
                )
                
                # 计算距离，决定是直接瞬移还是动画移动
                distance = np.linalg.norm(target_pos - current_pos)
                if distance < 0.1:  # 如果距离很近，直接瞬移
                    self.simulator.move_agent_to(target_pos)
                    self.update_displays()
                    
                    # 验证坐标转换精度
                    coord_check = self.simulator.verify_coordinate_conversion(target_pos)
                    if coord_check['error_acceptable']:
                        coord_status = f"坐标转换精度: {coord_check['position_error']:.3f}m ✓"
                    else:
                        coord_status = f"坐标转换精度: {coord_check['position_error']:.3f}m ⚠"
                    
                    self.status_label.setText(f"智能体已移动到 ({target_pos[0]:.1f}, {target_pos[2]:.1f})\n{coord_status}")
                else:
                    # 使用动画移动，但不使用pathfinder路径规划
                    self.start_direct_animation(current_pos, target_pos)
                    
                    # 验证坐标转换精度
                    coord_check = self.simulator.verify_coordinate_conversion(target_pos)
                    if coord_check['error_acceptable']:
                        coord_status = f"坐标转换精度: {coord_check['position_error']:.3f}m ✓"
                    else:
                        coord_status = f"坐标转换精度: {coord_check['position_error']:.3f}m ⚠"
                    
                    self.status_label.setText(f"开始移动到 ({target_pos[0]:.1f}, {target_pos[2]:.1f}) (距离: {distance:.1f}m)\n{coord_status}")
                        
            except Exception as e:
                self.status_label.setText(f"移动智能体失败: {str(e)}")
                return
                
        except ValueError as e:
            self.status_label.setText(f"坐标格式错误: {str(e)}")
        except Exception as e:
            self.status_label.setText(f"处理命令时发生错误: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
    
    def process_view_command(self, command: str):
        """处理视角命令 - 修复实现"""
        parts = command.lower().split()
        if len(parts) != 2:
            self.status_label.setText("视角命令格式: 方向 角度 (如: right 30)")
            return
        
        try:
            direction = parts[0]
            angle = float(parts[1])
            
            # 获取当前智能体状态
            current_state = self.simulator.get_agent_state()
            current_rotation = current_state.rotation
            
            # 创建当前四元数
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
            
            # 计算旋转变化
            rotation_quat = mn.Quaternion()  # 默认构造函数创建单位四元数
            
            if direction == "left":
                # 绕Y轴左转
                rotation_quat = mn.Quaternion.rotation(mn.Rad(math.radians(angle)), mn.Vector3.y_axis())
            elif direction == "right":
                # 绕Y轴右转
                rotation_quat = mn.Quaternion.rotation(mn.Rad(math.radians(-angle)), mn.Vector3.y_axis())
            elif direction == "up":
                # 绕X轴上看
                rotation_quat = mn.Quaternion.rotation(mn.Rad(math.radians(-angle)), mn.Vector3.x_axis())
            elif direction == "down":
                # 绕X轴下看
                rotation_quat = mn.Quaternion.rotation(mn.Rad(math.radians(angle)), mn.Vector3.x_axis())
            else:
                self.status_label.setText("无效方向，请使用: left, right, up, down")
                return
            
            # 应用旋转 - 修复四元数乘法顺序
            # 对于局部旋转，应该用旋转四元数左乘当前四元数
            new_rotation = rotation_quat * current_quat
            
            # 创建新的智能体状态
            new_state = habitat_sim.AgentState()
            new_state.position = current_state.position
            new_state.rotation = np.array([new_rotation.vector.x, new_rotation.vector.y, 
                                         new_rotation.vector.z, new_rotation.scalar], dtype=np.float32)
            
            # 应用新状态
            self.simulator.agent.set_state(new_state)
            
            # 更新所有显示（包括FPV和地图上的朝向箭头）
            self.update_displays()
            
            self.status_label.setText(f"视角已调整: {direction} {angle}度")
            
        except Exception as e:
            self.status_label.setText(f"视角调整失败: {str(e)}")
            print(f"视角调整详细错误: {e}")
            import traceback
            traceback.print_exc()
    
    def start_direct_animation(self, start_pos: np.ndarray, end_pos: np.ndarray):
        """开始直线动画移动（不使用pathfinder路径规划）"""
        try:
            # 直接创建从起点到终点的路径（只有两个点）
            path = [start_pos.copy(), end_pos.copy()]
            
            self.path_waypoints = path
            self.current_waypoint_index = 0
            self.current_interpolation_step = 0  # 重置插值步数
            self.is_moving = True
            
            distance = np.linalg.norm(end_pos - start_pos)
            self.status_label.setText(f"开始直线移动，距离: {distance:.1f}m")
            
            # 开始动画定时器 - 更快的更新频率以实现平滑动画
            self.animation_timer.start(50)  # 每50ms更新一次，实现更平滑的动画
            
        except Exception as e:
            self.status_label.setText(f"动画初始化出错: {str(e)[:50]}...，执行直接瞬移")
            self.simulator.move_agent_to(end_pos)
            self.update_displays()

    def start_path_animation(self, start_pos: np.ndarray, end_pos: np.ndarray):
        """开始路径动画（保留原函数用于兼容性，但现在也使用直接路径）"""
        # 直接调用新的直线动画函数
        self.start_direct_animation(start_pos, end_pos)
    
    def animate_movement(self):
        """平滑动画移动函数"""
        if not self.is_moving or self.current_waypoint_index >= len(self.path_waypoints) - 1:
            # 动画完成
            self.animation_timer.stop()
            self.is_moving = False
            self.current_interpolation_step = 0
            self.status_label.setText("移动完成")
            return
        
        # 如果是新的路径段，初始化插值
        if self.current_interpolation_step == 0:
            self.animation_start_pos = self.path_waypoints[self.current_waypoint_index].copy()
            self.animation_end_pos = self.path_waypoints[self.current_waypoint_index + 1].copy()
            
            # 计算朝向 - 修复角度计算（180度错误）
            direction = self.animation_end_pos - self.animation_start_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
                # 在Habitat中，-Z轴是前方，修复角度计算
                # 之前的计算导致了180度的错误，现在修正
                angle = math.atan2(direction[0], direction[2])  # 使用+Z计算，然后旋转180度
                angle += math.pi  # 加180度修正
                
                # 创建朝向目标的旋转四元数
                rotation = mn.Quaternion.rotation(mn.Rad(angle), mn.Vector3.y_axis())
                self.animation_end_rotation = np.array([rotation.vector.x, rotation.vector.y, 
                                                      rotation.vector.z, rotation.scalar], dtype=np.float32)
            else:
                self.animation_end_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
            
            # 获取当前旋转作为起始旋转
            current_state = self.simulator.get_agent_state()
            if hasattr(current_state.rotation, 'x'):
                self.animation_start_rotation = np.array([
                    current_state.rotation.x, current_state.rotation.y, 
                    current_state.rotation.z, current_state.rotation.w
                ], dtype=np.float32)
            else:
                self.animation_start_rotation = current_state.rotation.astype(np.float32)
        
        # 计算插值参数 (0.0 到 1.0)
        t = self.current_interpolation_step / self.interpolation_steps
        
        # 位置插值 (线性插值)
        interpolated_pos = self.animation_start_pos + t * (self.animation_end_pos - self.animation_start_pos)
        
        # 旋转插值 (球面线性插值)
        try:
            start_quat = mn.Quaternion(
                mn.Vector3(self.animation_start_rotation[0], self.animation_start_rotation[1], self.animation_start_rotation[2]),
                self.animation_start_rotation[3]
            )
            end_quat = mn.Quaternion(
                mn.Vector3(self.animation_end_rotation[0], self.animation_end_rotation[1], self.animation_end_rotation[2]),
                self.animation_end_rotation[3]
            )
            
            # 球面线性插值
            interpolated_quat = mn.Math.slerp(start_quat, end_quat, t)
            interpolated_rotation = np.array([
                interpolated_quat.vector.x, interpolated_quat.vector.y,
                interpolated_quat.vector.z, interpolated_quat.scalar
            ], dtype=np.float32)
            
        except Exception as e:
            # 如果插值失败，使用线性插值
            interpolated_rotation = self.animation_start_rotation + t * (self.animation_end_rotation - self.animation_start_rotation)
            # 归一化四元数
            norm = np.linalg.norm(interpolated_rotation)
            if norm > 0:
                interpolated_rotation = interpolated_rotation / norm
        
        # 移动智能体到插值位置
        self.simulator.move_agent_to(interpolated_pos, interpolated_rotation)
        
        # 更新显示
        self.update_displays()
        
        # 更新插值步数
        self.current_interpolation_step += 1
        
        # 检查是否完成当前路径段
        if self.current_interpolation_step >= self.interpolation_steps:
            # 移动到下一个路径点
            self.current_waypoint_index += 1
            self.current_interpolation_step = 0
            
            # 更新状态信息
            remaining_waypoints = len(self.path_waypoints) - self.current_waypoint_index - 1
            if remaining_waypoints > 0:
                self.status_label.setText(f"移动中... 剩余{remaining_waypoints}个路径点")
            else:
                self.status_label.setText("即将完成移动...")
    
    def update_displays(self):
        """更新所有显示"""
        if not self.simulator:
            return
        
        # 获取当前智能体状态
        agent_state = self.simulator.get_agent_state()
        
        # 更新地图显示
        self.update_map_display(agent_state.position, agent_state.rotation)
        
        # 更新FPV显示
        self.update_fpv_display()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.simulator:
            self.simulator.close()
        event.accept()


def main():
    """主函数"""
    # HM3D场景文件路径 - 请根据实际情况修改
    # 优先尝试相对路径，然后尝试绝对路径
    possible_paths = [
        "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
    ]
    
    scene_filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            scene_filepath = path
            break
    
    # 检查场景文件是否存在
    if scene_filepath is None:
        print("错误: 找不到场景文件。请检查以下可能的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\n请确保已下载HM3D数据集或修改scene_filepath变量")
        print("或者在命令行中指定场景文件路径:")
        print(f"  python {sys.argv[0]} <scene_file_path>")
        
        # 允许通过命令行参数指定场景文件
        if len(sys.argv) > 1:
            scene_filepath = sys.argv[1]
            if not os.path.exists(scene_filepath):
                print(f"指定的场景文件不存在: {scene_filepath}")
                sys.exit(1)
        else:
            sys.exit(1)
    
    print(f"使用场景文件: {scene_filepath}")
    
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = HabitatNavigatorApp(scene_filepath)
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
