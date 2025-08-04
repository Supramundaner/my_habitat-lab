"""
VideoComposer - 视频合成和绘制类 (View)
处理所有图像操作和视频文件写入
"""

import cv2
import numpy as np
import magnum as mn
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional
import math

from .simulator import HabitatSimulator
from .utils import euler_from_quaternion, convert_to_magnum_quat


class VideoComposer:
    """处理视频合成和图像绘制的类"""
    
    def __init__(self, simulator: HabitatSimulator, config: Dict[str, Any], output_path: str):
        """
        初始化视频合成器
        
        Args:
            simulator: HabitatSimulator实例
            config: 配置字典
            output_path: 输出视频路径
        """
        self.simulator = simulator
        self.config = config
        self.output_path = output_path
        
        # 视频配置
        self.fps = config['video']['fps']
        self.video_width = config['video']['resolution']['width']
        self.video_height = config['video']['resolution']['height']
        self.fpv_width = config['video']['fpv_width']
        self.map_width = config['video']['map_width']
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, 
            (self.video_width, self.video_height)
        )
        
        # 获取并处理基础地图
        base_map = simulator.get_base_map()
        if base_map is not None:
            self.base_map = self._resize_map_for_video(base_map)
        else:
            # 创建黑色占位图
            self.base_map = Image.new('RGB', (self.map_width, self.video_height), (0, 0, 0))
            print("警告: 无法获取基础地图，使用黑色占位图")
        
        self.frame_count = 0
        self.map_builder = None
        print(f"视频合成器初始化完成: {self.video_width}x{self.video_height} @ {self.fps}fps")
    
    def _resize_map_for_video(self, map_image: Image.Image) -> Image.Image:
        """
        调整地图尺寸以适配视频右侧
        保持纵横比，多余空间用黑色填充
        
        Args:
            map_image: 原始地图图像
        
        Returns:
            调整后的地图图像
        """
        original_width, original_height = map_image.size
        target_width = self.map_width
        target_height = self.video_height
        
        # 计算缩放比例，保持纵横比
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)
        
        # 计算新尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 缩放图像
        resized_image = map_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建黑色背景并居中放置
        result = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(resized_image, (x_offset, y_offset))
        
        # 保存缩放参数用于坐标转换
        self.map_scale = scale
        self.map_x_offset = x_offset
        self.map_y_offset = y_offset
        self.map_scaled_width = new_width
        self.map_scaled_height = new_height
        
        return result
    
    def add_frame(self, robot_state: Optional[Dict[str, Any]] = None, 
                  observation: Optional[Dict[str, Any]] = None):
        """
        添加一帧到视频
        
        Args:
            robot_state: 机器人状态
            observation: 模拟器观察结果
        """
        if robot_state is None:
            robot_state = self.simulator.get_robot_state()
        
        if observation is None:
            observation = self.simulator.get_observation()
        
        # 1. 获取第一人称视角
        fpv_image = self._process_fpv_image(observation['rgb'])
        
        # 2. 获取全局鸟瞰图
        top_down_map = self._create_annotated_map(robot_state)

        # 3. 获取并更新占用地图
        if self.map_builder:
            self.map_builder.update_map(
                observation['depth'],
                robot_state,
                self.config['OCCUPANCY_MAP']['HFOV']
            )
            occupancy_map_img = self.map_builder.get_map_image(
                robot_state,
                (self.map_width, self.map_width)
            )
            # Convert to PIL for consistency
            occupancy_map_pil = Image.fromarray(cv2.cvtColor(occupancy_map_img, cv2.COLOR_BGR2RGB))
        else:
            occupancy_map_pil = Image.new('RGB', (self.map_width, self.map_width), (10, 10, 10))

        # 4. 调整图像尺寸
        right_panel_height = self.video_height // 2
        occupancy_map_resized = occupancy_map_pil.resize((self.map_width, right_panel_height))
        top_down_map_resized = top_down_map.resize((self.map_width, right_panel_height))

        # 5. 拼接右侧面板
        right_panel = Image.new('RGB', (self.map_width, self.video_height))
        right_panel.paste(occupancy_map_resized, (0, 0))
        right_panel.paste(top_down_map_resized, (0, right_panel_height))

        # 6. 拼接最终帧
        final_frame = self._compose_final_frame(fpv_image, right_panel)
        
        # 7. 写入视频
        self.video_writer.write(cv2.cvtColor(np.array(final_frame), cv2.COLOR_RGB2BGR))
        self.frame_count += 1
        #print(f"已添加第 {self.frame_count} 帧")
    
    def _process_fpv_image(self, fpv_array: np.ndarray) -> Image.Image:
        """
        处理FPV图像
        
        Args:
            fpv_array: FPV图像数组
        
        Returns:
            处理后的PIL图像
        """
        # 转换为PIL图像
        if fpv_array.shape[2] == 4:  # RGBA
            fpv_pil = Image.fromarray(fpv_array[..., :3], "RGB")
        else:  # RGB
            fpv_pil = Image.fromarray(fpv_array, "RGB")
        
        # 调整到目标尺寸
        fpv_resized = fpv_pil.resize((self.fpv_width, self.video_height), Image.Resampling.LANCZOS)
        
        return fpv_resized
    
    def _create_annotated_map(self, robot_state: Dict[str, np.ndarray]) -> Image.Image:
        """
        创建带智能体标记的地图
        
        Args:
            robot_state: 机器人状态字典
        
        Returns:
            带标记的地图图像
        """
        # 复制基础地图
        annotated_map = self.base_map.copy()
        
        # 绘制智能体标记
        self._draw_agent_marker(
            annotated_map, 
            robot_state['position'], 
            robot_state['rotation']
        )
        
        return annotated_map
    
    def _draw_agent_marker(self, map_image: Image.Image, position: np.ndarray, 
                          rotation_quat: np.ndarray):
        """
        在地图上绘制智能体位置和朝向
        
        Args:
            map_image: 地图图像
            position: 3D位置
            rotation_quat: 四元数旋转
        """
        draw = ImageDraw.Draw(map_image)
        #print(f"绘制智能体位置: {position}, 朝向: {rotation_quat}")
        # 转换世界坐标到地图坐标
        original_map_x, original_map_y = self.simulator.world_to_map_coords(position)
        # 转换到当前缩放的地图坐标系
        scaled_map_x = int(original_map_x * self.map_scale + self.map_x_offset)
        scaled_map_y = int(original_map_y * self.map_scale + self.map_y_offset)
        #print(f"map_x_offset: {self.map_x_offset}, map_y_offset: {self.map_y_offset}")
        # 确保坐标在图像范围内
        scaled_map_x = max(0, min(scaled_map_x, self.map_width - 1))
        scaled_map_y = max(0, min(scaled_map_y, self.video_height - 1))
        
        # 绘制位置点（红色圆点）
        dot_radius = max(4, int(8 * self.map_scale))
        draw.ellipse([
            scaled_map_x - dot_radius, scaled_map_y - dot_radius,
            scaled_map_x + dot_radius, scaled_map_y + dot_radius
        ], fill=(255, 0, 0))
        
        # 绘制朝向箭头
        self._draw_direction_arrow(
            draw, scaled_map_x, scaled_map_y, 
            rotation_quat, dot_radius * 2
        )
    
    def _draw_direction_arrow(self, draw: ImageDraw.Draw, center_x: int, center_y: int,
                            rotation_quat: np.ndarray, arrow_length: int):
        """
        绘制朝向箭头
        
        Args:
            draw: ImageDraw对象
            center_x: 中心X坐标
            center_y: 中心Y坐标
            rotation_quat: 四元数旋转
            arrow_length: 箭头长度
        """
        try:
            # 使用Magnum四元数来获取正确的前向向量
            quat = convert_to_magnum_quat(rotation_quat)
            
            # 在Habitat中，-Z轴是前方，计算前向向量
            forward_vec = quat.transform_vector(mn.Vector3(0, 0, -1))
            
            # 转换到地图坐标系：X轴向右，Z轴向下
            # 在地图上：X对应水平向右，Z对应垂直向下
            dx = forward_vec.x * arrow_length
            dz = forward_vec.z * arrow_length  # 注意：这里是Z，不是Y
            
            end_x = center_x + int(dx)
            end_y = center_y + int(dz)  # Z轴对应地图的Y轴
            
            # 绘制主箭头线
            draw.line([(center_x, center_y), (end_x, end_y)], fill=(255, 255, 0), width=3)
            
            # 计算箭头头部的方向
            arrow_angle = math.atan2(dz, dx)
            arrow_head_length = arrow_length * 0.3
            arrow_head_angle = math.radians(30)
            
            # 左侧箭头线
            left_angle = arrow_angle + math.pi - arrow_head_angle
            left_x = end_x + int(math.cos(left_angle) * arrow_head_length)
            left_y = end_y + int(math.sin(left_angle) * arrow_head_length)
            draw.line([(end_x, end_y), (left_x, left_y)], fill=(255, 255, 0), width=2)
            
            # 右侧箭头线
            right_angle = arrow_angle + math.pi + arrow_head_angle
            right_x = end_x + int(math.cos(right_angle) * arrow_head_length)
            right_y = end_y + int(math.sin(right_angle) * arrow_head_length)
            draw.line([(end_x, end_y), (right_x, right_y)], fill=(255, 255, 0), width=2)
            
        except Exception as e:
            print(f"绘制朝向箭头失败: {e}")
    
    def _compose_final_frame(self, fpv_image: Image.Image, map_image: Image.Image) -> Image.Image:
        """
        合成最终视频帧
        
        Args:
            fpv_image: FPV图像
            map_image: 地图图像
        
        Returns:
            合成后的图像
        """
        # 创建最终画布
        final_frame = Image.new('RGB', (self.video_width, self.video_height), (0, 0, 0))
        
        # 左侧放置FPV图像
        final_frame.paste(fpv_image, (0, 0))
        
        # 右侧放置地图
        final_frame.paste(map_image, (self.fpv_width, 0))
        
        # 可选：添加分隔线
        draw = ImageDraw.Draw(final_frame)
        draw.line([(self.fpv_width, 0), (self.fpv_width, self.video_height)], 
                 fill=(255, 255, 255), width=2)
        
        return final_frame
    
    def save_and_close(self):
        """保存并关闭视频文件"""
        if self.video_writer:
            self.video_writer.release()
            print(f"视频已保存: {self.output_path}")
            print(f"总帧数: {self.frame_count}")
    
    def get_frame_count(self) -> int:
        """获取当前帧数"""
        return self.frame_count
    
    def set_map_builder(self, map_builder):
        """设置地图构建器"""
        self.map_builder = map_builder
