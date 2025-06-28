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
        
        # 动画参数
        self.rotation_step = 1.0  # 每度旋转生成一帧
        self.movement_step = 0.05  # 每0.05米移动生成一帧
        
        # 视频参数
        self.video_width = 1024  # 左右各512
        self.video_height = 512
        
        # 初始化模拟器
        self.simulator = None
        self.current_frames = []
        self._initialize_simulator()
        
        print(f"Video generator initialized with {fps} FPS")
        print(f"Animation steps: {self.rotation_step}°/frame, {self.movement_step}m/frame")
    
    def _initialize_simulator(self):
        """初始化Habitat模拟器"""
        try:
            # 创建自定义的HabitatSimulator实例，指定GPU设备
            self.simulator = CustomHabitatSimulator(
                scene_filepath=self.scene_filepath,
                gpu_device_id=self.gpu_device_id
            )
            
            # 设置代理到场景中心的可导航位置
            self._reset_agent_to_center()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize simulator: {e}")
    
    def _reset_agent_to_center(self):
        """将代理重置到场景中心的可导航位置"""
        center_x = (self.simulator.scene_bounds[0][0] + self.simulator.scene_bounds[1][0]) / 2
        center_z = (self.simulator.scene_bounds[0][2] + self.simulator.scene_bounds[1][2]) / 2
        
        # 尝试对齐到可导航位置
        navigable_pos = self.simulator.snap_to_navigable(center_x, center_z)
        if navigable_pos is not None:
            self.simulator.move_agent_to(navigable_pos)
        else:
            # 如果中心不可导航，使用pathfinder找一个随机可导航点
            random_point = self.simulator.sim.pathfinder.get_random_navigable_point()
            self.simulator.move_agent_to(np.array([random_point.x, random_point.y, random_point.z]))
    
    def process_command_sequence(self, commands: List[List[Union[str, float]]]) -> Optional[str]:
        """处理指令序列并生成视频"""
        self.current_frames = []
        start_time = time.time()
        
        try:
            # 添加起始帧
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
        """执行移动指令（平滑动画，带碰撞检测和视角调整）"""
        try:
            # 检查目标位置是否可导航
            target_pos = self.simulator.snap_to_navigable(target_x, target_z)
            if target_pos is None:
                print(f"    ERROR: Target position ({target_x:.2f}, {target_z:.2f}) is not navigable")
                return False
            
            # 获取当前位置和旋转
            current_state = self.simulator.get_agent_state()
            current_pos = current_state.position
            current_rotation = current_state.rotation
            
            # 如果距离很近，直接瞬移
            distance = np.linalg.norm(target_pos - current_pos)
            if distance < self.movement_step:
                # 计算朝向目标的旋转
                target_rotation = self._calculate_rotation_towards_target(current_pos, target_pos)
                self.simulator.move_agent_to(target_pos, target_rotation)
                self._capture_frame()
                return True
            
            # 计算移动步数
            total_steps = int(distance / self.movement_step)
            
            # 计算每步的位置增量
            direction_vector = (target_pos - current_pos) / total_steps
            
            # 计算目标旋转（朝向目标点）
            target_rotation = self._calculate_rotation_towards_target(current_pos, target_pos)
            
            # 开始移动，每步都调整视角
            for step in range(total_steps):
                # 计算下一个位置
                next_pos = current_pos + direction_vector * (step + 1)
                
                # 碰撞检测
                if not self.simulator.is_navigable(next_pos[0], next_pos[2]):
                    print(f"    ERROR: Collision detected while moving to [{target_x:.2f}, {target_z:.2f}]. Aborting.")
                    return False
                
                # 计算插值旋转（平滑过渡到目标朝向）
                t = (step + 1) / total_steps
                interpolated_rotation = self._interpolate_rotation(current_rotation, target_rotation, t)
                
                # 移动代理并设置旋转
                self.simulator.move_agent_to(next_pos, interpolated_rotation)
                
                # 捕获帧
                self._capture_frame()
            
            # 确保到达精确位置和旋转
            self.simulator.move_agent_to(target_pos, target_rotation)
            self._capture_frame()
            
            return True
            
        except Exception as e:
            print(f"    Movement failed: {e}")
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
        """捕获当前帧（左右分屏，保持地图比例）"""
        try:
            # 获取FPV图像
            fpv_image = self.simulator.get_fpv_observation()
            fpv_pil = Image.fromarray(fpv_image[..., :3], "RGB")
            
            # 获取俯视图（复用基础地图并绘制代理）
            map_image = self.simulator.base_map_image.copy()
            agent_state = self.simulator.get_agent_state()
            self._draw_agent_on_map(map_image, agent_state.position, agent_state.rotation)
            
            # 调整FPV图像大小到512x512
            fpv_resized = fpv_pil.resize((512, 512), Image.Resampling.LANCZOS)
            
            # 处理地图图像，保持纵横比
            map_resized = self._resize_map_with_aspect_ratio(map_image, 512, 512)
            
            # 创建左右分屏图像
            combined = Image.new('RGB', (1024, 512))
            combined.paste(fpv_resized, (0, 0))
            combined.paste(map_resized, (512, 0))
            
            # 转换为numpy数组并添加到帧列表
            frame_array = np.array(combined)
            self.current_frames.append(frame_array)
            
        except Exception as e:
            print(f"    Failed to capture frame: {e}")
    
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
    
    def _draw_agent_on_map(self, image: Image.Image, agent_pos: np.ndarray, 
                          agent_rotation: Optional[np.ndarray] = None):
        """在地图上绘制代理位置和朝向（复用interactive_app逻辑）"""
        draw = ImageDraw.Draw(image)
        
        # 转换世界坐标到地图坐标
        map_x, map_y = self.simulator.world_to_map_coords(agent_pos)
        
        # 绘制代理位置（红点）
        dot_radius = 8
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
            except:
                pass  # 如果箭头绘制失败，只显示点
    
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
        if self.simulator:
            state = self.simulator.get_agent_state()
            pos = state.position
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return (0.0, 0.0, 0.0)
    
    def get_agent_rotation(self) -> Tuple[float, float, float, float]:
        """获取代理当前旋转（四元数）"""
        if self.simulator:
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
    
    def _calculate_rotation_towards_target(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """计算从当前位置朝向目标位置的旋转四元数"""
        # 计算方向向量
        direction = target_pos - current_pos
        direction[1] = 0  # 忽略Y轴差异，只考虑水平方向
        
        # 如果距离太近，返回当前旋转
        if np.linalg.norm(direction) < 0.01:
            current_state = self.simulator.get_agent_state()
            return current_state.rotation
        
        # 归一化方向向量
        direction = direction / np.linalg.norm(direction)
        
        # 在Habitat中，-Z轴是前方，计算需要的旋转角度
        # 参考interactive_app的实现
        angle = math.atan2(direction[0], -direction[2])  # 注意这里使用-Z
        
        # 创建朝向目标的旋转四元数
        rotation_quat = mn.Quaternion.rotation(mn.Rad(angle), mn.Vector3.y_axis())
        
        return np.array([
            rotation_quat.vector.x, rotation_quat.vector.y,
            rotation_quat.vector.z, rotation_quat.scalar
        ], dtype=np.float32)
    
    def _interpolate_rotation(self, start_rotation: np.ndarray, end_rotation: np.ndarray, t: float) -> np.ndarray:
        """在两个旋转之间进行插值"""
        # 处理不同类型的旋转数据
        if hasattr(start_rotation, 'x'):
            start_array = np.array([start_rotation.x, start_rotation.y, start_rotation.z, start_rotation.w], dtype=np.float32)
        else:
            start_array = start_rotation.astype(np.float32)
        
        if hasattr(end_rotation, 'x'):
            end_array = np.array([end_rotation.x, end_rotation.y, end_rotation.z, end_rotation.w], dtype=np.float32)
        else:
            end_array = end_rotation.astype(np.float32)
        
        # 检查四元数的点积，确保选择较短的路径
        dot_product = np.dot(start_array, end_array)
        if dot_product < 0:
            end_array = -end_array  # 反向四元数代表相同的旋转
        
        # 线性插值
        interpolated = start_array + t * (end_array - start_array)
        
        # 归一化四元数
        norm = np.linalg.norm(interpolated)
        if norm > 0:
            interpolated = interpolated / norm
        
        return interpolated


class CustomHabitatSimulator(HabitatSimulator):
    """自定义Habitat模拟器，支持指定GPU设备"""
    
    def __init__(self, scene_filepath: str, resolution: Tuple[int, int] = (512, 512), 
                 gpu_device_id: int = 0):
        self.gpu_device_id = gpu_device_id
        super().__init__(scene_filepath, resolution)
    
    def _initialize_simulator(self):
        """重写初始化方法以支持GPU设备选择"""
        # 配置后端 - 指定GPU设备
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_filepath
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = self.gpu_device_id  # 指定GPU设备
        backend_cfg.random_seed = 1
        
        # 其余配置保持不变（复用父类逻辑）
        # 配置FPV传感器
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [self.resolution[1], self.resolution[0]]
        fpv_sensor_spec.position = mn.Vector3(0, 1.5, 0)
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
        
        # 计算地图分辨率
        max_resolution = 1024
        aspect_ratio = world_size_x / world_size_z
        
        if aspect_ratio > 1:
            map_width = max_resolution
            map_height = int(max_resolution / aspect_ratio)
        else:
            map_height = max_resolution
            map_width = int(max_resolution * aspect_ratio)
        
        # 配置正交传感器
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.resolution = [map_height, map_width]
        ortho_sensor_spec.position = mn.Vector3(0, 0, 0)
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        
        # 配置智能体
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, ortho_sensor_spec]
        
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
        print(f"Scene bounds: {self.scene_bounds}")
        print(f"Scene center: {self.scene_center}")
