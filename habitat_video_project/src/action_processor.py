"""
ActionProcessor - 动作处理和动画逻辑类 (Controller)
协调整个模拟过程，处理动作序列和动画
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional

from .simulator import HabitatSimulator
from .video_composer import VideoComposer
from .utils import (
    slerp, 
    quaternion_to_direction_yaw, 
    quaternion_from_euler,
    euler_from_quaternion
)


class ActionProcessor:
    """动作处理和动画控制的核心类"""
    
    def __init__(self, simulator: HabitatSimulator, composer: VideoComposer, config: Dict[str, Any]):
        """
        初始化动作处理器
        
        Args:
            simulator: HabitatSimulator实例
            composer: VideoComposer实例
            config: 配置字典
        """
        self.simulator = simulator
        self.composer = composer
        self.config = config
        
        # 动画参数
        self.linear_speed = config['agent']['linear_speed']  # m/s
        self.angular_speed = config['agent']['angular_speed']  # deg/s
        self.fps = config['video']['fps']
        self.time_step = 1.0 / self.fps  # 每帧时间间隔
        
        print(f"动作处理器初始化完成")
        print(f"线性速度: {self.linear_speed} m/s")
        print(f"角速度: {self.angular_speed} deg/s")
        print(f"视频帧率: {self.fps} fps")
    
    def execute_sequence(self, action_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行动作序列
        
        Args:
            action_sequence: 动作序列列表
        
        Returns:
            执行结果报告
        """
        completed_actions = []
        collision_action = None
        
        print(f"开始执行动作序列，共 {len(action_sequence)} 个动作")
        
        for i, action in enumerate(action_sequence):
            print(f"执行动作 {i+1}/{len(action_sequence)}: {action}")
            
            success = self._execute_single_action(action)
            
            if not success:
                collision_action = {
                    'index': i,
                    'action': action,
                    'reason': 'collision_detected'
                }
                print(f"在第 {i+1} 个动作处检测到碰撞，停止执行")
                break
            else:
                completed_actions.append(action)
                print(f"动作 {i+1} 执行完成")
        
        print(f"动作序列执行完成，成功执行 {len(completed_actions)} 个动作")
        
        return {
            'completed_actions': completed_actions,
            'collision_action': collision_action
        }
    
    def _execute_single_action(self, action: Dict[str, Any]) -> bool:
        """
        执行单个动作
        
        Args:
            action: 动作字典
        
        Returns:
            True表示成功，False表示失败（碰撞）
        """
        action_type = action['type']
        params = action['params']
        
        if action_type == 'move_to':
            return self._handle_move_to(params)
        elif action_type == 'turn_left':
            return self._handle_turn_left(params)
        elif action_type == 'turn_right':
            return self._handle_turn_right(params)
        elif action_type == 'pause':
            return self._handle_pause(params)
        else:
            print(f"未知动作类型: {action_type}")
            return False
    
    def _handle_move_to(self, params: Dict[str, Any]) -> bool:
        """
        处理移动到指定位置的动作
        
        Args:
            params: 参数字典，包含x和z坐标
        
        Returns:
            True表示成功，False表示碰撞
        """
        target_x = params['x']
        target_z = params['z']
        
        # 1. 获取当前机器人状态
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        current_rot = current_state['rotation']
        
        # 2. 计算目标3D位置
        target_y = self.simulator.get_navigable_y(target_x, target_z)
        if target_y is None:
            print(f"目标位置 ({target_x}, {target_z}) 不可导航")
            return False
        
        target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)
        
        # 3. 碰撞预检
        if self.simulator.check_straight_path_collision(current_pos, target_pos):
            print(f"检测到从 {current_pos} 到 {target_pos} 的路径会发生碰撞")
            return False
        
        # 4. 计算朝向目标的旋转
        target_rotation = quaternion_to_direction_yaw(current_pos, target_pos)
        
        # 5. 执行转向动画
        self._animate_rotation(current_rot, target_rotation)
        
        # 6. 执行移动动画
        self._animate_movement(current_pos, target_pos)
        
        return True
    
    def _handle_turn_left(self, params: Dict[str, Any]) -> bool:
        """
        处理左转动作
        
        Args:
            params: 参数字典，包含angle角度
        
        Returns:
            True表示成功
        """
        angle = params['angle']
        return self._handle_turn(angle)  # 左转为正角度（修复方向）
    
    def _handle_turn_right(self, params: Dict[str, Any]) -> bool:
        """
        处理右转动作
        
        Args:
            params: 参数字典，包含angle角度
        
        Returns:
            True表示成功
        """
        angle = params['angle']
        return self._handle_turn(-angle)  # 右转为负角度（修复方向）
    
    def _handle_turn(self, angle_degrees: float) -> bool:
        """
        处理转向动作
        
        Args:
            angle_degrees: 转向角度（正数为右转，负数为左转）
        
        Returns:
            True表示成功
        """
        # 获取当前旋转
        current_state = self.simulator.get_robot_state()
        current_rot = current_state['rotation']
        
        # 计算目标旋转
        # 从当前四元数提取欧拉角
        roll, pitch, yaw = euler_from_quaternion(current_rot)
        
        # 更新偏航角
        new_yaw = yaw + angle_degrees
        
        # 创建新的四元数
        target_rot = quaternion_from_euler(roll, pitch, new_yaw)
        
        # 执行旋转动画
        self._animate_rotation(current_rot, target_rot)
        
        return True
    
    def _handle_pause(self, params: Dict[str, Any]) -> bool:
        """
        处理暂停动作
        
        Args:
            params: 参数字典，包含duration秒数
        
        Returns:
            True表示成功
        """
        duration = params['duration']
        num_frames = int(duration * self.fps)
        
        print(f"暂停 {duration} 秒 ({num_frames} 帧)")
        
        for _ in range(num_frames):
            self.composer.add_frame()
        
        return True
    
    def _animate_rotation(self, start_rot: np.ndarray, end_rot: np.ndarray):
        """
        执行旋转动画
        
        Args:
            start_rot: 起始四元数
            end_rot: 结束四元数
        """
        # 计算旋转角度差
        start_roll, start_pitch, start_yaw = euler_from_quaternion(start_rot)
        end_roll, end_pitch, end_yaw = euler_from_quaternion(end_rot)
        
        angle_diff = abs(end_yaw - start_yaw)
        
        # 处理角度跨越180度的情况
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 计算动画时长和帧数
        duration = angle_diff / self.angular_speed
        num_frames = max(1, int(duration * self.fps))
        
        print(f"旋转动画: {angle_diff:.1f}度, {duration:.2f}秒, {num_frames}帧")
        
        # 获取当前位置（保持不变）
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        
        # 逐帧插值
        for frame in range(num_frames + 1):
            t = frame / num_frames if num_frames > 0 else 1.0
            
            # 使用球面线性插值
            interpolated_rot = slerp(start_rot, end_rot, t)
            
            # 更新机器人姿态
            self.simulator.set_robot_pose(current_pos, interpolated_rot)
            
            # 添加视频帧
            self.composer.add_frame()
    
    def _animate_movement(self, start_pos: np.ndarray, end_pos: np.ndarray):
        """
        执行移动动画
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
        """
        # 计算距离和动画时长
        distance = np.linalg.norm(end_pos - start_pos)
        duration = distance / self.linear_speed
        num_frames = max(1, int(duration * self.fps))
        
        print(f"移动动画: {distance:.2f}米, {duration:.2f}秒, {num_frames}帧")
        
        # 获取当前旋转（保持不变）
        current_state = self.simulator.get_robot_state()
        current_rot = current_state['rotation']
        
        # 逐帧插值
        for frame in range(num_frames + 1):
            t = frame / num_frames if num_frames > 0 else 1.0
            
            # 线性插值位置
            interpolated_pos = start_pos + (end_pos - start_pos) * t
            
            # 更新机器人姿态
            self.simulator.set_robot_pose(interpolated_pos, current_rot)
            
            # 添加视频帧
            self.composer.add_frame()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_frames': self.composer.get_frame_count(),
            'total_duration': self.composer.get_frame_count() / self.fps,
            'linear_speed': self.linear_speed,
            'angular_speed': self.angular_speed,
            'fps': self.fps
        }
