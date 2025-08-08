"""
ActionProcessor - 动作处理和动画逻辑类 (Controller)
协调整个模拟过程，处理动作序列和动画
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional

from .simulator import HabitatSimulator
from .video_composer import VideoComposer
from .vfh_star import VFHStar
from .utils import (
    slerp, 
    quaternion_to_direction_yaw, 
    quaternion_from_euler,
    euler_from_quaternion,
    get_device,
    use_mixed_precision,
    to_numpy
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
        
        # GPU设置
        self.use_gpu = config.get('gpu', {}).get('enabled', False)
        
        print(f"动作处理器初始化完成")
        print(f"线性速度: {self.linear_speed} m/s")
        print(f"角速度: {self.angular_speed} deg/s")
        print(f"视频帧率: {self.fps} fps")
        print(f"GPU加速: {'启用' if self.use_gpu else '禁用'}")
    
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
        处理移动到指定位置的动作（使用VFH*算法）。
        
        Args:
            params: 参数字典，包含x和z坐标
        
        Returns:
            True表示成功，False表示失败（目标不可达或无路径）
        """
        target_x = params['x']
        target_z = params['z']
        
        # 1. 获取当前机器人状态
        current_state = self.simulator.get_robot_state()
        start_pos = current_state['position']
        
        # 2. 检查目标点是否可导航并获取其3D坐标
        target_y = self.simulator.get_navigable_y(target_x, target_z)
        if target_y is None:
            print(f"错误: 目标位置 ({target_x}, {target_z}) 不在可导航区域。")
            return False
        
        # 3. 初始化VFH*算法
        target_pos_2d = np.array([target_x, target_z])
        vfh_config = self.config.get('vfh', {})
        vfh_star = VFHStar(target_pos_2d, vfh_config)
        
        # 4. 获取占用地图构建器
        map_builder = self.composer.get_map_builder()
        if map_builder is None:
            print("错误: 无法获取占用地图构建器")
            return False
        
        # 5. VFH*导航循环
        max_iterations = 1000  # 防止无限循环
        iteration = 0
        prev_direction = None
        
        while iteration < max_iterations:
            iteration += 1
            
            # 获取当前机器人状态
            current_state = self.simulator.get_robot_state()
            current_pos = current_state['position']
            current_rot = current_state['rotation']
            
            # 计算到目标的距离
            dist_to_target = np.sqrt((current_pos[0] - target_x)**2 + (current_pos[2] - target_z)**2)
            
            # 检查是否到达目标
            if dist_to_target < 0.5:  # 0.5米阈值
                print(f"成功到达目标位置 ({target_x}, {target_z})")
                return True
            
            # 获取当前占用地图
            observation = self.simulator.get_observation()
            depth_observation = observation.get('depth')
            if depth_observation is None:
                print("错误: 无法获取深度传感器数据")
                return False
            
            # 更新占用地图
            agent_pose = {'position': current_pos, 'rotation': current_rot}
            map_builder.update_map(depth_observation, agent_pose, 90.0)  # 90度FOV
            
            # 从占用地图提取局部障碍物
            robot_pos_2d = np.array([current_pos[0], current_pos[2]])
            obstacles = map_builder.get_obstacles_from_map(robot_pos_2d, vfh_star.sensor_range)
            
            # 获取机器人朝向角度
            robot_theta = map_builder.get_robot_theta_from_quaternion(current_rot)
            
            # VFH*计算最佳方向
            ideal_direction = vfh_star.get_best_direction(robot_pos_2d, robot_theta, obstacles, prev_direction)

            action_name, action_value = vfh_star.get_discrete_action(ideal_direction, robot_theta)
            params = {'angle': action_value, 'type': action_name}
            if action_name == "turn_left":
                self._handle_turn_left(params)
            elif action_name == "turn_right":
                self._handle_turn_right(params)
            else:
                self._execute_move_forward()
            
            
            current_state = self.simulator.get_robot_state()
            current_pos = current_state['position']
            current_rot = current_state['rotation']
            
            # 重新获取深度传感器数据并更新地图
            observation = self.simulator.get_observation()
            depth_observation = observation.get('depth')
            if depth_observation is not None:
                agent_pose = {'position': current_pos, 'rotation': current_rot}
                map_builder.update_map(depth_observation, agent_pose, 90.0)
                
            # 更新前一个方向
            prev_direction = ideal_direction
            
            # 添加视频帧（传入当前机器人状态和观察数据）
            current_state = self.simulator.get_robot_state()
            current_observation = self.simulator.get_observation()
            self.composer.add_frame(robot_state=current_state, observation=current_observation)
        
        print(f"达到最大迭代次数 {max_iterations}，导航失败")
        return False
    
    def _execute_move_forward(self):
        """执行前进动作"""
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        current_rot = current_state['rotation']
        
        # 计算前进距离
        forward_distance = 0.25
        
        # 计算前进方向
        from .utils import euler_from_quaternion
        roll, pitch, yaw = euler_from_quaternion(current_rot)
        forward_direction = yaw
        
        # 计算新位置
        new_pos = current_pos + np.array([
            forward_distance * np.cos(forward_direction),
            0,
            forward_distance * np.sin(forward_direction)
        ])
        
        # 更新机器人位置
        self.simulator.set_robot_pose(new_pos, current_rot)
    
    def _execute_turn_left(self, angle_degrees: float):
        """执行左转动作"""
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        current_rot = current_state['rotation']
        
        # 计算旋转角度（弧度）
        angle_radians = np.deg2rad(angle_degrees)
        
        # 计算新的旋转
        roll, pitch, yaw = euler_from_quaternion(current_rot)
        new_yaw = yaw + angle_radians
        new_rot = quaternion_from_euler(roll, pitch, new_yaw, use_gpu=self.use_gpu)
        
        if hasattr(new_rot, 'cpu'):
            new_rot = to_numpy(new_rot)
        
        # 更新机器人姿态
        self.simulator.set_robot_pose(current_pos, new_rot)
    
    def _execute_turn_right(self, angle_degrees: float):
        """执行右转动作"""
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        current_rot = current_state['rotation']
        
        # 计算旋转角度（弧度）
        angle_radians = np.deg2rad(-angle_degrees)  # 右转为负角度
        
        # 计算新的旋转
        roll, pitch, yaw = euler_from_quaternion(current_rot)
        new_yaw = yaw + angle_radians
        new_rot = quaternion_from_euler(roll, pitch, new_yaw, use_gpu=self.use_gpu)
        
        if hasattr(new_rot, 'cpu'):
            new_rot = to_numpy(new_rot)
        
        # 更新机器人姿态
        self.simulator.set_robot_pose(current_pos, new_rot)
    
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
        
        # 更新偏航角（将度数转换为弧度）
        angle_radians = np.deg2rad(angle_degrees)
        new_yaw = yaw + angle_radians
        

        
        # 创建新的四元数
        target_rot = quaternion_from_euler(roll, pitch, new_yaw, use_gpu=self.use_gpu)
        
        # 确保target_rot是numpy数组
        if hasattr(target_rot, 'cpu'):
            target_rot = to_numpy(target_rot)
        
        # 执行旋转动画 - 直接使用传入的角度值
        self._animate_rotation(current_rot, target_rot, abs(angle_degrees))
        
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
            # 添加视频帧（传入当前机器人状态和观察数据）
            current_state = self.simulator.get_robot_state()
            current_observation = self.simulator.get_observation()
            self.composer.add_frame(robot_state=current_state, observation=current_observation)
        
        return True
    
    def _animate_rotation(self, start_rot: np.ndarray, end_rot: np.ndarray, target_angle: float):
        """
        执行旋转动画
        
        Args:
            start_rot: 起始四元数
            end_rot: 结束四元数
            target_angle: 目标旋转角度（用于计算动画时长）
        """
        # 直接使用传入的目标角度计算动画时长和帧数
        duration = abs(target_angle) / self.angular_speed
        min_frames = max(20, int(duration * self.fps))  # 最少20帧，确保旋转明显可见
        num_frames = max(min_frames, round(duration * self.fps))
        
        print(f"旋转动画: {target_angle:.1f}度, {duration:.2f}秒, {num_frames}帧")
        
        # 获取当前位置（保持不变）
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        
        # 调试：显示起始和结束的欧拉角
        start_roll, start_pitch, start_yaw = euler_from_quaternion(start_rot)
        end_roll, end_pitch, end_yaw = euler_from_quaternion(end_rot)
        print(f"调试 - 起始欧拉角: roll={np.rad2deg(start_roll):.1f}°, pitch={np.rad2deg(start_pitch):.1f}°, yaw={np.rad2deg(start_yaw):.1f}°")
        print(f"调试 - 结束欧拉角: roll={np.rad2deg(end_roll):.1f}°, pitch={np.rad2deg(end_pitch):.1f}°, yaw={np.rad2deg(end_yaw):.1f}°")
        
        # 逐帧插值 - 使用 range(1, num_frames + 1) 避免重复起始帧
        for frame in range(1, num_frames + 1):
            t = frame / num_frames
            
            # 使用球面线性插值
            interpolated_rot = slerp(start_rot, end_rot, t)
            
            # 确保返回numpy数组（如果是torch张量则转换）
            if hasattr(interpolated_rot, 'cpu'):
                interpolated_rot = to_numpy(interpolated_rot)
            
            # 调试：显示当前帧的欧拉角
            if frame % 5 == 0:  # 每5帧显示一次
                curr_roll, curr_pitch, curr_yaw = euler_from_quaternion(interpolated_rot)
                print(f"调试 - 帧{frame}: yaw={np.rad2deg(curr_yaw):.1f}° (t={t:.2f})")
            
            # 更新机器人姿态
            self.simulator.set_robot_pose(current_pos, interpolated_rot)
            
            # 添加视频帧（传入当前机器人状态和观察数据）
            current_state = self.simulator.get_robot_state()
            current_observation = self.simulator.get_observation()
            self.composer.add_frame(robot_state=current_state, observation=current_observation)
    
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
        min_frames = max(20, int(duration * self.fps))  # 最少20帧，确保移动明显可见
        num_frames = max(min_frames, round(duration * self.fps))  # 使用 round 而不是 int 来避免截断误差
        
        print(f"移动动画: {distance:.2f}米, {duration:.2f}秒, {num_frames}帧")
        
        # 获取当前旋转（保持不变）
        current_state = self.simulator.get_robot_state()
        current_rot = current_state['rotation']
        
        # 逐帧插值 - 使用 range(1, num_frames + 1) 避免重复起始帧
        for frame in range(1, num_frames + 1):
            t = frame / num_frames
            
            # 线性插值位置
            interpolated_pos = start_pos + (end_pos - start_pos) * t
            
            # 更新机器人姿态
            self.simulator.set_robot_pose(interpolated_pos, current_rot)
            
            # 添加视频帧（传入当前机器人状态和观察数据）
            current_state = self.simulator.get_robot_state()
            current_observation = self.simulator.get_observation()
            self.composer.add_frame(robot_state=current_state, observation=current_observation)
    
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
