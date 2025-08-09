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
    
    def __init__(self, simulator: HabitatSimulator, composer: VideoComposer, config: Dict[str, Any], map_builder=None):
        """
        初始化动作处理器
        
        Args:
            simulator: HabitatSimulator实例
            composer: VideoComposer实例
            config: 配置字典
            map_builder: 占用地图构建器实例
        """
        self.simulator = simulator
        self.composer = composer
        self.config = config
        self.map_builder = map_builder
        
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
        """
        处理移动到指定位置的动作
        
        Args:
            params: 参数字典，包含x和z坐标
        
        Returns:
            True表示成功，False表示碰撞
        """
    """def _handle_move_to(self, params: Dict[str, Any]) -> bool:

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

        
        # 4. 计算朝向目标的旋转
        target_rotation = quaternion_to_direction_yaw(current_pos, target_pos)
        
        # 5. 执行转向动画
        self._animate_rotation(current_rot, target_rotation)
        if self.simulator.check_straight_path_collision(current_pos, target_pos):
            print(f"检测到从 {current_pos} 到 {target_pos} 的路径会发生碰撞")
            return False
        # 6. 执行移动动画
        self._animate_movement(current_pos, target_pos)
        
        return True"""
    def _handle_move_to(self, params: Dict[str, Any]) -> bool:
        """
        处理移动到指定位置的动作（使用路径规划）。
        
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
            return False # 动作失败，但不是碰撞，所以返回True让序列继续？这里我们定义为False，表示动作无法执行
        
        end_pos = np.array([target_x, target_y, target_z], dtype=np.float32)

        
        target_pos_2d = np.array([target_x, target_z])
        vfh_config = self.config.get('vfh', {})

        
        vfh_star = VFHStar(target_pos_2d, vfh_config)
        
        # 将VFH实例传递给video_composer以启用histogram可视化
        self.composer.set_vfh_instance(vfh_star)
        
        map_builder = self.composer.get_map_builder()
        
        if map_builder is None:
            print("[ERROR] 无法获取地图构建器！")
            return False
            


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
                print(f"[SUCCESS] 成功到达目标位置 ({target_x}, {target_z})")
                return True
            
            # 获取当前占用地图
            observation = self.simulator.get_observation()
            depth_observation = observation.get('depth')
            if depth_observation is None:
                print("[ERROR] 无法获取深度传感器数据")
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
            
            if ideal_direction is None:
                print("[ERROR] VFH*无法找到可行方向！")
                return False
                
            action_name, action_value = vfh_star.get_discrete_action(ideal_direction, robot_theta)
            
            params = {'angle': action_value, 'type': action_name}

            # 执行动作
            if action_name == "turn_left":
                self._handle_turn_left(params)
            elif action_name == "turn_right":
                self._handle_turn_right(params)
            else:
                # 获取当前机器人状态
                current_state = self.simulator.get_robot_state()
                current_pos = current_state['position']
                current_rot = current_state['rotation']
                
                # 从四元数提取偏航角
                from .utils import euler_from_quaternion, yaw_to_unified_angle, unified_angle_to_direction_vector
                roll, pitch, yaw = euler_from_quaternion(current_rot)
                yaw_rad = np.radians(yaw)
                
                # 转换为统一角度系统
                unified_angle = yaw_to_unified_angle(yaw_rad)
                
                # 计算前进方向向量（使用统一角度系统）
                forward_direction = unified_angle_to_direction_vector(unified_angle)
                
                # 计算向前0.25米的目标位置
                distance = 0.25  # 前进距离
                end_pos = current_pos + forward_direction * distance
                
                print(f"[DEBUG] 前进: 从 {current_pos} 到 {end_pos}")
                
                # 执行移动到目标位置
                self._animate_movement(current_pos, end_pos)
            
            # 获取执行动作后的状态
            current_state = self.simulator.get_robot_state()
            current_pos = current_state['position']
            current_rot = current_state['rotation']
            
            print(f"[DEBUG] 动作执行后位置: {current_pos}")
            
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
            
            # 检查是否卡住（位置没有变化）
            if iteration > 1:
                pos_change = np.linalg.norm(current_pos - start_pos)
                if pos_change < 0.01:  # 如果位置变化小于1cm
                    print(f"[WARNING] 机器人可能卡住，位置变化: {pos_change:.3f}m")
                    if iteration > 10:  # 如果连续10次迭代都卡住
                        print(f"[ERROR] 机器人卡住，停止导航")
                        return False
        
        print(f"[ERROR] 达到最大迭代次数 {max_iterations}，导航失败")
        return False
        
    
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
        
        # 调试信息：输出转向详情
        print(f"[DEBUG] 转向动作详情:")
        print(f"  - 当前角度: {yaw:.2f}°")
        print(f"  - 转向角度: {angle_degrees:.2f}°")
        print(f"  - 目标角度: {new_yaw:.2f}°")
        
        # 创建新的四元数
        target_rot = quaternion_from_euler(roll, pitch, new_yaw, use_gpu=self.use_gpu)
        
        # 确保target_rot是numpy数组
        if hasattr(target_rot, 'cpu'):
            target_rot = to_numpy(target_rot)
        
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
        num_frames = max(1, round(duration * self.fps))  # 使用 round 而不是 int 来避免截断误差
        
        # 调试信息：输出旋转详情
        print(f"[DEBUG] 旋转动画详情:")
        print(f"  - 起始角度: {start_yaw:.2f}° (四元数: {start_rot})")
        print(f"  - 目标角度: {end_yaw:.2f}° (四元数: {end_rot})")
        print(f"  - 角度差: {angle_diff:.1f}度")
        print(f"  - 动画时长: {duration:.2f}秒, {num_frames}帧")
        
        # 获取当前位置（保持不变）
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        
        # 逐帧插值 - 修复：使用 range(1, num_frames + 1) 避免重复起始帧
        for frame in range(1, num_frames + 1):
            t = frame / num_frames
            
            # 使用球面线性插值
            interpolated_rot = slerp(start_rot, end_rot, t)
            
            # 确保返回numpy数组（如果是torch张量则转换）
            if hasattr(interpolated_rot, 'cpu'):
                interpolated_rot = to_numpy(interpolated_rot)
            
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
        num_frames = max(1, round(duration * self.fps))  # 使用 round 而不是 int 来避免截断误差
        
        # 调试信息：输出移动详情
        print(f"[DEBUG] 移动动画详情:")
        print(f"  - 起始位置: {start_pos}")
        print(f"  - 目标位置: {end_pos}")
        print(f"  - 移动距离: {distance:.3f}米")
        print(f"  - 移动方向: {end_pos - start_pos}")
        print(f"  - 动画时长: {duration:.2f}秒, {num_frames}帧")
        
        # 获取当前旋转（保持不变）
        current_state = self.simulator.get_robot_state()
        current_rot = current_state['rotation']
        
        # 逐帧插值 - 修复：使用 range(1, num_frames + 1) 避免重复起始帧
        for frame in range(1, num_frames + 1):
            t = frame / num_frames
            
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