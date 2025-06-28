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
from typing import Tuple, List, Optional, Union
import time
from datetime import datetime
import cv2

# 添加interactive_app的src路径以复用代码
interactive_app_src = os.path.join(os.path.dirname(__file__), '../../interactive_app/src')
sys.path.insert(0, interactive_app_src)

# Habitat相关导入
import habitat_sim
import magnum as mn

# 复用interactive_app的HabitatSimulator类
from habitat_navigator_app import HabitatSimulator


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
            print(f"    Direct movement failed: {e}")
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
    
    def get_agent_rotation(self) -> Tuple[float, float, float, float]:
        """获取代理当前旋转（四元数）"""
        if self.simulator and self.agent_initialized:
            state = self.simulator.get_agent_state()
            rot = state.rotation
            if hasattr(rot, 'x'):
                return (float(rot.x), float(rot.y), float(rot.z), float(rot.w))
            else:
                return (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        return (0.0, 0.0, 0.0, 1.0)
    
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
    """自定义Habitat模拟器，支持指定GPU设备，继承修复后的坐标转换功能"""
    
    def __init__(self, scene_filepath: str, resolution: Tuple[int, int] = (512, 512), 
                 gpu_device_id: int = 0):
        self.gpu_device_id = gpu_device_id
        # 继承父类的所有修复，包括坐标转换和padding常量
        super().__init__(scene_filepath, resolution)
    
    def _initialize_simulator(self):
        """重写初始化方法以支持GPU设备选择，保持父类的修复功能"""
        # 配置后端 - 指定GPU设备
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_filepath
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = self.gpu_device_id  # 指定GPU设备
        backend_cfg.random_seed = 1
        
        print(f"Initializing simulator with GPU device {self.gpu_device_id}")
        
        # 配置人类agent模型
        try:
            # 尝试使用内置的人类模型
            backend_cfg.default_agent_id = 0
            backend_cfg.create_renderer = True
        except Exception as e:
            print(f"Warning: Could not load human agent model: {e}")
        
        # 配置FPV传感器 - 基于人类视角高度
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [self.resolution[1], self.resolution[0]]
        fpv_sensor_spec.position = mn.Vector3(0, 1.7, 0)  # 人类平均视角高度1.7米
        fpv_sensor_spec.hfov = 90.0
        
        # 获取场景边界以计算正交传感器分辨率
        temp_sensor = habitat_sim.CameraSensorSpec()
        temp_sensor.uuid = "temp_sensor"
        temp_sensor.sensor_type = habitat_sim.SensorType.COLOR
        temp_sensor.resolution = [64, 64]
        
        temp_agent_cfg = habitat_sim.agent.AgentConfiguration()
        temp_agent_cfg.sensor_specifications = [temp_sensor]
        
        temp_sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [temp_agent_cfg]))
        temp_bounds = temp_sim.pathfinder.get_bounds()
        world_size_x = temp_bounds[1][0] - temp_bounds[0][0]
        world_size_z = temp_bounds[1][2] - temp_bounds[0][2]
        temp_sim.close()
        
        # 计算地图分辨率 - 提高地图质量
        max_resolution = 2048  # 提高地图分辨率
        aspect_ratio = world_size_x / world_size_z
        
        if aspect_ratio > 1:
            map_width = max_resolution
            map_height = int(max_resolution / aspect_ratio)
        else:
            map_height = max_resolution
            map_width = int(max_resolution * aspect_ratio)
        
        print(f"地图分辨率: {map_width} x {map_height}")
        
        # 配置正交传感器
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.resolution = [map_height, map_width]
        ortho_sensor_spec.position = mn.Vector3(0, 0, 0)
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        
        # 配置智能体 - 加载人类模型
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, ortho_sensor_spec]
        
        # 设置人类agent的形状和大小
        agent_cfg.height = 1.7  # 人类平均身高
        agent_cfg.radius = 0.3  # 人类身体半径（用于碰撞检测）
        
        # 尝试加载人类模型文件 - 使用URDF配置
        try:
            # 检查是否有Fetch机器人模型（可以作为人形agent的替代）
            fetch_urdf_path = "/home/yaoaa/habitat-lab/data/robots/hab_fetch/robots/hab_fetch.urdf"
            
            if os.path.exists(fetch_urdf_path):
                print(f"Found robot model: {fetch_urdf_path}")
                # 注意：由于这是机器人模型而非人类模型，我们只使用其物理属性
            else:
                print("No specific agent model found, using default capsule shape with human dimensions")
        except Exception as e:
            print(f"Warning: Could not check agent model: {e}")
        
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
        
        # 获取场景信息
        self.scene_bounds = self.sim.pathfinder.get_bounds()
        self.scene_center = (self.scene_bounds[0] + self.scene_bounds[1]) / 2.0
        self.scene_size = self.scene_bounds[1] - self.scene_bounds[0]
        self.ortho_scale = max(self.scene_size[0], self.scene_size[2]) / 2.0
        
        print(f"Simulator initialized with GPU device {self.gpu_device_id}")
        print(f"Agent height: {agent_cfg.height}m, radius: {agent_cfg.radius}m")
        print(f"Scene bounds: {self.scene_bounds}")
        print(f"Scene center: {self.scene_center}")
        
        # 验证坐标转换修复是否生效
        print("=== 坐标转换修复验证 ===")
        print(f"MAP_PADDING_LEFT: {self.MAP_PADDING_LEFT}")
        print(f"MAP_PADDING_TOP: {self.MAP_PADDING_TOP}")
        print(f"MAP_PADDING_RIGHT: {self.MAP_PADDING_RIGHT}")
        print(f"MAP_PADDING_BOTTOM: {self.MAP_PADDING_BOTTOM}")
        
        # 测试场景中心的坐标转换
        if hasattr(self, 'verify_coordinate_conversion'):
            center_check = self.verify_coordinate_conversion(self.scene_center)
            print(f"场景中心坐标转换误差: {center_check['position_error']:.6f}m {'✓' if center_check['error_acceptable'] else '⚠'}")
        else:
            print("⚠️ 坐标转换验证方法不可用")
