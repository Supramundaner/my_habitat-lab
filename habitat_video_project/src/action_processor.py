"""
ActionProcessor - 动作处理和动画逻辑类 (Controller)
协调整个模拟过程，处理目标导航和动画
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Union

from .simulator import HabitatSimulator
from .video_composer import VideoComposer
from .vfh_star import VFHStar
from .object_detector import ObjectDetector
from .utils import (
    slerp, 
    quaternion_to_direction_yaw, 
    quaternion_from_euler,
    euler_from_quaternion,
    get_device,
    use_mixed_precision,
    to_numpy
)


class NavigationPhase:
    """导航阶段枚举"""
    HYBRID_NAVIGATION = "hybrid"           # A* + VFH* 混合导航
    OBJECT_DETECTION_NAVIGATION = "object_detection"  # 对象检测导航


class NavigationConfig:
    """导航配置类，集中管理所有导航相关参数"""
    
    def __init__(self, config: Dict[str, Any]):
        # A*路径规划参数
        self.a_star_interval = 5  # 每5次行动重新规划A*
        
        # A*距离变换参数
        # self.a_star_weight_w = config.get('navigation', {}).get('a_star_weight_w', 10)  # 距离惩罚权重
        # self.a_star_weight_w = 200
        self.a_star_epsilon = config.get('navigation', {}).get('a_star_epsilon', 0.01)  # 防止除零的小正数
        self.unknown_region_distance = config.get('navigation', {}).get('unknown_region_distance', 5.0)  # 未知区域距离值
        
        # 目标点距离参数
        self.intermediate_distance = 1.5  # 中间目标点距离
        self.final_target_threshold = 1.5  # 切换到最终目标的阈值
        self.final_stop_threshold = 0.8  # 最终停止阈值
        
        self.max_search_radius = 500  # 最大搜索半径（像素）
        
        # 前进参数
        self.forward_distance = 0.25  # 前进距离（米）
        
        # 最大迭代次数
        self.max_iterations = 1000  # 防止无限循环


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
        
        # 初始化导航配置
        self.nav_config = NavigationConfig(config)
        
        # 初始化对象检测器
        self.object_detector = ObjectDetector(config)
        
        # 动画参数
        self.linear_speed = config['agent']['linear_speed']  # m/s
        self.angular_speed = config['agent']['angular_speed']  # deg/s
        self.fps = config['video']['fps']
        self.time_step = 1.0 / self.fps  # 每帧时间间隔
        
        # GPU设置
        self.use_gpu = config.get('gpu', {}).get('enabled', False)
        
        # 距离跟踪
        self.total_distance_traveled = 0.0  # 累计行走距离
        self.previous_position = None  # 上一个位置，用于计算距离
        
        # 导航距离阈值配置
        self.waypoint_distance = config.get('navigation', {}).get('waypoint_distance', 1.5)
        self.destination_distance = config.get('navigation', {}).get('destination_distance', 0.8)
        
        # Time step跟踪参数
        self.time_steps_num = config.get('agent', {}).get('time_steps_num', 10)
        self.min_displacement = config.get('agent', {}).get('min_displacement', 1.0)
        
        print(f"动作处理器初始化完成")
        print(f"线性速度: {self.linear_speed} m/s")
        print(f"角速度: {self.angular_speed} deg/s")
        print(f"视频帧率: {self.fps} fps")
        print(f"中间waypoint距离: {self.waypoint_distance} m")
        print(f"最终目标距离: {self.destination_distance} m")
        print(f"Time step检测窗口: {self.time_steps_num} steps")
        print(f"最小位移阈值: {self.min_displacement} m")
        print(f"GPU加速: {'启用' if self.use_gpu else '禁用'}")
    
    def _get_robot_state(self) -> Dict[str, Any]:
        """
        获取当前机器人状态
        
        Returns:
            包含位置、旋转和2D位置的字典
        """
        current_state = self.simulator.get_robot_state()
        current_pos = current_state['position']
        current_rot = current_state['rotation']
        current_pos_2d = np.array([current_pos[0], current_pos[2]])
        
        return {
            'position': current_pos,
            'rotation': current_rot,
            'position_2d': current_pos_2d,
            'raw_state': current_state
        }
    
    def _should_switch_to_object_detection(self, current_pos_2d: np.ndarray, current_path: List[np.ndarray], target_pos_2d: np.ndarray) -> bool:
        """
        判断是否应该切换到对象检测阶段
        
        条件：
        1. 有有效的A*路径
        2. 沿着路径到目标的距离 < 1.5m
        3. 对象检测模块可用
        
        Args:
            current_pos_2d: 当前位置 [x, z]
            current_path: A*路径点列表
            target_pos_2d: 目标位置 [x, z]
            
        Returns:
            bool: 是否应该切换到对象检测阶段
        """
        if current_path is None or len(current_path) < 2:
            return False
        
        if not self.object_detector.is_enabled():
            return False
        
        path_distance = self._calculate_path_distance_to_target(
            current_pos_2d, current_path, target_pos_2d
        )
        
        return path_distance < 1.5
    
    def _get_camera_params(self) -> Dict[str, float]:
        """获取相机参数"""
        # 从simulator或配置中获取相机参数
        return {
            'fx': 512.0,  # 焦距x
            'fy': 512.0,  # 焦距y
            'cx': 256.0,  # 主点x
            'cy': 256.0   # 主点y
        }
    
    def _downsample_map(self, map: np.ndarray, factor: int) -> np.ndarray:
        """
        对地图进行下采样
        
        Args:
            map: 原始地图
            factor: 下采样因子（宽高都缩小到1/factor）
            
        Returns:
            下采样后的地图
        """
        # 使用最大池化进行下采样，确保障碍物信息不丢失
        from scipy import ndimage
        
        # 将地图值转换为二值：0=障碍物，1=可行走
        binary_map = (map != 0).astype(np.uint8)
        
        # 使用最大池化下采样
        downsampled_binary = ndimage.maximum_filter(binary_map, size=factor)
        downsampled_binary = downsampled_binary[::factor, ::factor]
        
        # 转换回原始值：0=障碍物，255=可行走，128=未知
        downsampled_map = np.zeros_like(downsampled_binary, dtype=np.uint8)
        downsampled_map[downsampled_binary == 1] = 255
        
        return downsampled_map
    
    def _camera_to_world_coords(self, camera_coords: np.ndarray) -> np.ndarray:
        """
        将相机坐标系转换为世界坐标系
        
        Args:
            camera_coords: 相机坐标系下的3D坐标 [x, y, z]
            
        Returns:
            世界坐标系下的3D坐标 [x, y, z]
        """
        try:
            # 获取机器人当前状态
            robot_state = self.simulator.get_robot_state()
            robot_position = robot_state['position']
            robot_rotation = robot_state['rotation']
            
            # 基于map_builder.py中的_transform_points_to_world函数实现
            # 1. 相机到智能体坐标系的转换（绕X轴旋转180度）
            cam_to_agent_rot = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ], dtype=np.float32)
            
            # 将相机坐标转换到智能体坐标系
            points_agent_frame = camera_coords @ cam_to_agent_rot.T
            
            # 2. 四元数到旋转矩阵的转换
            x, y, z, w = robot_rotation
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            xw, yw, zw = x*w, y*w, z*w
            
            rot_mat = np.array([
                [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
                [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
                [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
            ], dtype=np.float32)
            
            # 3. 智能体坐标系到世界坐标系的转换：旋转 + 平移
            world_coords = points_agent_frame @ rot_mat.T + robot_position
            
            return world_coords
            
        except Exception as e:
            print(f"坐标系转换错误: {e}")
            return camera_coords  # 转换失败时返回原始坐标
    
    def _select_vfh_target(self, current_pos_2d: np.ndarray, adjusted_target_pos: np.ndarray, 
                          current_path: List[np.ndarray], dist_to_final_target: float) -> Dict[str, Any]:
        """
        选择VFH*的目标点
        
        Args:
            current_pos_2d: 当前位置 [x, z]
            adjusted_target_pos: 调整后的目标位置 [x, z]
            current_path: A*路径点列表
            dist_to_final_target: 到最终目标的距离（直线距离，用于兼容性）
        
        Returns:
            包含目标点和类型的字典
        """
        # 计算沿着路径到目标的弧长
        path_distance_to_target = self._calculate_path_distance_to_target(current_pos_2d, current_path, adjusted_target_pos)
        
        if path_distance_to_target < self.nav_config.final_target_threshold:
            # 沿着路径到最终目标的距离很近，直接以调整后的最终目标为VFH*目标
            vfh_target = adjusted_target_pos
            target_type = "最终目标"
        else:
            # 使用A*路径上的中间目标点
            vfh_target = self._get_intermediate_target(current_pos_2d, current_path, self.nav_config.intermediate_distance)
            if vfh_target is None:
                return {
                    'success': False,
                    'reason': 'no_intermediate_target',
                    'message': '无法找到中间目标点'
                }
            target_type = "中间目标"
        
        return {
            'success': True,
            'target': vfh_target,
            'type': target_type
        }
    
    def _execute_vfh_action(self, action_name: str, action_value: float, current_pos: np.ndarray, current_rot: np.ndarray) -> None:
        """
        执行VFH*计算出的动作
        
        Args:
            action_name: 动作名称
            action_value: 动作值
            current_pos: 当前位置
            current_rot: 当前旋转
        """
        if action_name == "turn_left":
            self._handle_turn_left({'angle': action_value})
        elif action_name == "turn_right":
            self._handle_turn_right({'angle': action_value})
        else:
            # 前进动作
            # 从四元数提取偏航角
            from .utils import euler_from_quaternion, yaw_to_unified_angle, unified_angle_to_direction_vector
            roll, pitch, yaw = euler_from_quaternion(current_rot)
            yaw_rad = np.radians(yaw)
            
            # 转换为统一角度系统
            unified_angle = yaw_to_unified_angle(yaw_rad)
            
            # 计算前进方向向量（使用统一角度系统）
            forward_direction = unified_angle_to_direction_vector(unified_angle)

            
            # 计算目标位置的x,z坐标
            # forward_direction是[x, y, z]格式，所以使用[0]和[2]分别对应x和z
            target_x = current_pos[0] + forward_direction[0] * self.nav_config.forward_distance
            target_z = current_pos[2] + forward_direction[2] * self.nav_config.forward_distance
            
            # 获取目标位置对应的可导航y坐标
            target_y = self.simulator.get_navigable_y(target_x, target_z)
            
            if target_y is not None:
                # 构建完整的3D目标位置（包含正确的y坐标）
                end_pos = np.array([target_x, target_y, target_z])
                
                # 执行移动到目标位置
                self._animate_movement(current_pos, end_pos)
            else:
                print(f"[WARNING] 目标位置 ({target_x:.2f}, {target_z:.2f}) 不可导航，跳过前进动作")
    
    def execute_sequence(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行目标导航序列
        
        Args:
            action_data: 动作数据字典，包含target_info和wall_mask_path
        
        Returns:
            执行结果报告
        """
        # 重置距离跟踪
        self.total_distance_traveled = 0.0
        current_state = self.simulator.get_robot_state()
        self.previous_position = current_state['position'].copy()
        
        completed_actions = []
        collision_action = None
        
        # 从action_data中提取target_info
        target_info = action_data.get('target_info', None)
        
        if target_info is None:
            print("错误: 未找到target_info")
            return {
                'completed_actions': [],
                'collision_action': {'reason': 'no_target_info', 'message': '未找到目标信息'},
                'target_found': False
            }
        
        print(f"开始执行目标导航，目标信息: {target_info}")
        
        # 执行目标导航
        result = self._execute_target_navigation(target_info)
        
        # 检查是否成功
        success = result.get('success', False)
        
        if success:
            completed_actions.append({'type': 'target_navigation', 'target': target_info})
            print(f"目标导航完成")
        else:
            collision_action = {
                'reason': result.get('reason', 'navigation_failed'),
                'message': result.get('message', '目标导航失败')
            }
            print(f"目标导航失败: {collision_action['message']}")
        
        return {
            'completed_actions': completed_actions,
            'collision_action': collision_action,
            'target_found': success
        }
    
    def _execute_target_navigation(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行目标导航
        
        Args:
            target_info: 目标信息字典，包含coordinate和name
        
        Returns:
            导航结果字典
        """
        # 提取目标坐标
        target_coord = target_info.get('coordinate', None)
        target_name = target_info.get('name', 'unknown')
        
        if target_coord is None:
            return {
                'success': False,
                'reason': 'invalid_target_coordinate',
                'message': '目标坐标无效'
            }
        
        # 确保target_coord是[x, z]格式
        if len(target_coord) != 2:
            return {
                'success': False,
                'reason': 'invalid_target_coordinate',
                'message': f'目标坐标格式错误，期望[x, z]，实际{target_coord}'
            }
        
        target_x, target_z = target_coord
        
        print(f"开始导航到目标: {target_name} 位置: ({target_x}, {target_z})")
        
        # 检查目标点是否可导航并获取其3D坐标
        target_y = self.simulator.get_navigable_y(target_x, target_z)
        if target_y is None:
            return {
                'success': False,
                'reason': 'target_unreachable',
                'message': f'目标位置 ({target_x}, {target_z}) 不在可导航区域'
            }
        
        # 初始化VFH*算法
        target_pos_2d = np.array([target_x, target_z])
        vfh_config = self.config.get('vfh', {})
        vfh_star = VFHStar(target_pos_2d, vfh_config)
        
        # 执行混合导航（A* + VFH*）
        return self._execute_hybrid_navigation(target_pos_2d, target_name, vfh_star)
    
    def _execute_hybrid_phase(self, current_pos_2d: np.ndarray, current_rot: np.ndarray, 
                             target_pos_2d: np.ndarray, vfh_star: VFHStar, 
                             current_path: List[np.ndarray], adjusted_target_pos: np.ndarray,
                             action_count: int, prev_direction: Optional[float]) -> Dict[str, Any]:
        """
        执行混合导航阶段（A* + VFH*）
        
        Args:
            current_pos_2d: 当前位置 [x, z]
            current_rot: 当前旋转
            target_pos_2d: 目标位置 [x, z]
            vfh_star: VFH*算法实例
            current_path: 当前A*路径
            adjusted_target_pos: 调整后的目标位置
            action_count: 行动计数器
            prev_direction: 前一个方向
            
        Returns:
            执行结果字典
        """
        # 检查是否到达最终目标
        dist_to_final_target = self._calculate_path_distance_to_target(current_pos_2d, current_path, target_pos_2d)
        
        if dist_to_final_target < self.nav_config.final_stop_threshold:
            return {'success': True}
        
        # 检查目标点是否在障碍物内
        if self.map_builder.grid_map is not None:
            # 将世界坐标转换为地图坐标
            adjusted_target_map_coords = self.map_builder._world_to_map_coords(np.array([[adjusted_target_pos[0], 0, adjusted_target_pos[1]]]))
            adjusted_target_pixel = (int(adjusted_target_map_coords[0, 0]), int(adjusted_target_map_coords[0, 1]))
            
            target_value = self.map_builder.grid_map[adjusted_target_pixel[1], adjusted_target_pixel[0]]
            if target_value == 0:  # 障碍物
                print(f"[WARNING] adjusted_target_pos在障碍物内，寻找最近的可行走点")
                nearest_walkable_pixel = self._find_nearest_walkable_point(adjusted_target_pixel, self.map_builder.grid_map)
                
                if nearest_walkable_pixel is not None:
                    # 将像素坐标转换为世界坐标
                    nearest_walkable_world = self._pixel_to_world_coord(nearest_walkable_pixel)
                    print(f"[INFO] adjusted_target_pos已更新: 原位置 {adjusted_target_pos} -> 新位置 {nearest_walkable_world}")
                    adjusted_target_pos = nearest_walkable_world
                else:
                    print(f"[ERROR] 无法找到adjusted_target_pos附近的可行走区域")
                    return {'failed': True, 'reason': 'no_walkable_area_near_target'}
        
        # 每5次行动重新规划A*路径
        if action_count % self.nav_config.a_star_interval == 0:
            current_path_result = self._plan_a_star_path(current_pos_2d, target_pos_2d)
            if current_path_result is None:
                return {'failed': True, 'reason': 'no_a_star_path'}
            
            current_path = current_path_result['path']
            if current_path_result['was_adjusted']:
                adjusted_target_pos = current_path_result['adjusted_goal']
        
        # 选择VFH*目标
        vfh_target_result = self._select_vfh_target(
            current_pos_2d, adjusted_target_pos, current_path, dist_to_final_target
        )
        line_distance = np.linalg.norm(current_pos_2d - adjusted_target_pos)
        # 计算并显示路径距离信息
        if current_path is not None:
            path_distance_to_target = self._calculate_path_distance_to_target(current_pos_2d, current_path, adjusted_target_pos)
            print(f"距离信息 - 直线距离: {line_distance:.2f}m, 路径距离: {path_distance_to_target:.2f}m")
        
        if not vfh_target_result['success']:
            return {'failed': True, 'reason': vfh_target_result['reason']}
        
        # 执行VFH*导航
        result = self._execute_vfh_navigation(
            current_pos_2d, current_rot, vfh_star, vfh_target_result['target'], prev_direction
        )
        
        return {
            'success': False,
            'failed': False,
            'current_path': current_path,
            'adjusted_target_pos': adjusted_target_pos,
            'action_count': action_count + 1,
            'prev_direction': result.get('prev_direction', prev_direction)
        }
    
    def _execute_object_detection_phase(self, current_pos_2d: np.ndarray, current_rot: np.ndarray, 
                                       target_name: str, vfh_star: VFHStar, 
                                       prev_direction: Optional[float], original_target_pos: np.ndarray,
                                       detected_target_pos: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        执行对象检测导航阶段
        
        Args:
            current_pos_2d: 当前位置 [x, z]
            current_rot: 当前旋转
            target_name: 目标名称
            vfh_star: VFH*算法实例
            prev_direction: 前一个方向
            original_target_pos: 原始目标位置（当检测失败时使用）
            detected_target_pos: 已检测到的目标位置（如果为None则进行检测）
            
        Returns:
            执行结果字典
        """
        # 如果还没有检测到目标，则进行检测
        if detected_target_pos is None:
            # 获取当前观测
            observation = self.simulator.get_observation()
            rgb_image = observation.get('rgb')
            depth_image = observation.get('depth')
            
            if rgb_image is None or depth_image is None:
                return {'failed': True, 'reason': 'no_observation_data'}
            
            # 相机参数
            camera_params = self._get_camera_params()
            
            # 对象检测
            detected_target_pos_camera = self.object_detector.detect_and_get_target_coords(
                rgb_image, depth_image, target_name, camera_params
            )
            
            if detected_target_pos_camera is None:
                print(f"[OBJECT DETECTION] 未能检测到目标: {target_name}，使用原始目标位置")
                # 使用原始目标位置进行导航
                target_pos_2d = original_target_pos
                detected_target_pos = None
            else:
                print(f"[OBJECT DETECTION] 检测到目标相机坐标: {detected_target_pos_camera}")
                # 将相机坐标转换为世界坐标
                detected_target_pos = self._camera_to_world_coords(detected_target_pos_camera)
                print(f"[OBJECT DETECTION] 转换为世界坐标: {detected_target_pos}")
                # 将3D坐标转换为2D导航坐标
                target_pos_2d = np.array([detected_target_pos[0], detected_target_pos[2]])
        else:
            # 已经检测到目标，直接使用
            print(f"[OBJECT DETECTION] 使用已检测到的目标位置: {detected_target_pos}")
            # 安全地处理2D/3D坐标转换
            if len(detected_target_pos) >= 3:
                # 3D坐标 [x, y, z]
                target_pos_2d = np.array([detected_target_pos[0], detected_target_pos[2]])
            elif len(detected_target_pos) == 2:
                # 2D坐标 [x, z]
                target_pos_2d = detected_target_pos.copy()
            else:
                print(f"[ERROR] 意外的坐标维度: {len(detected_target_pos)}")
                return {'failed': True, 'reason': 'invalid_coordinates'}
        
        # 检查目标点是否在障碍物内
        if self.map_builder.grid_map is not None:
            # 将世界坐标转换为地图坐标
            target_map_coords = self.map_builder._world_to_map_coords(np.array([[target_pos_2d[0], 0, target_pos_2d[1]]]))
            target_pixel = (int(target_map_coords[0, 0]), int(target_map_coords[0, 1]))
            
            target_value = self.map_builder.grid_map[target_pixel[1], target_pixel[0]]
            if target_value == 0:  # 障碍物
                print(f"[WARNING] 目标点在障碍物内，寻找最近的可行走点")
                nearest_walkable_pixel = self._find_nearest_walkable_point(target_pixel, self.map_builder.grid_map)
                
                if nearest_walkable_pixel is not None:
                    # 将像素坐标转换为世界坐标
                    nearest_walkable_world = self._pixel_to_world_coord(nearest_walkable_pixel)
                    print(f"[INFO] 目标点已调整: 原位置 {target_pos_2d} -> 新位置 {nearest_walkable_world}")
                    target_pos_2d = nearest_walkable_world
                    if detected_target_pos is not None:
                        detected_target_pos = nearest_walkable_world
                else:
                    print(f"[ERROR] 无法找到目标点附近的可行走区域")
                    return {'failed': True, 'reason': 'no_walkable_area_near_target'}
        
        # 检查是否足够接近目标
        current_path_result = self._plan_a_star_path(current_pos_2d, target_pos_2d)
        current_path = current_path_result['path']
        distance_to_target = self._calculate_path_distance_to_target(current_pos_2d, current_path, target_pos_2d)
        
        if distance_to_target < self.nav_config.final_stop_threshold:
            print(f"[SUCCESS] 成功导航到目标")
            return {'success': True}
        
        # 使用VFH*导航到目标
        vfh_star.update_target(target_pos_2d)
        
        result = self._execute_vfh_navigation(
            current_pos_2d, current_rot, vfh_star, target_pos_2d, prev_direction
        )
        
        # 根据是否检测到目标，更新相应的位置信息
        if detected_target_pos is not None:
            # 如果检测到目标，更新检测到的目标位置（包含障碍物调整）
            updated_detected_target_pos = detected_target_pos.copy()
            updated_detected_target_pos[0] = target_pos_2d[0]  # X坐标
            
            # 处理Z坐标：检查数组维度
            if len(detected_target_pos) >= 3:
                # 3D坐标 [x, y, z]，更新Z坐标
                updated_detected_target_pos[2] = target_pos_2d[1]  # Z坐标
            elif len(detected_target_pos) == 2:
                # 2D坐标 [x, z]，更新第二个元素
                updated_detected_target_pos[1] = target_pos_2d[1]  # Z坐标
            
            updated_original_target_pos = original_target_pos
        else:
            # 如果没有检测到目标，更新原始目标位置（包含障碍物调整）
            updated_detected_target_pos = None
            updated_original_target_pos = target_pos_2d.copy()
        
        return {
            'success': False,
            'failed': False,
            'prev_direction': result.get('prev_direction', prev_direction),
            'detected_target_pos': updated_detected_target_pos,
            'original_target_pos': updated_original_target_pos
        }
    
    def _execute_vfh_navigation(self, current_pos_2d: np.ndarray, current_rot: np.ndarray,
                               vfh_star: VFHStar, vfh_target: np.ndarray, 
                               prev_direction: Optional[float]) -> Dict[str, Any]:
        """
        执行VFH*导航
        
        Args:
            current_pos_2d: 当前位置 [x, z]
            current_rot: 当前旋转
            vfh_star: VFH*算法实例
            vfh_target: VFH*目标位置
            prev_direction: 前一个方向
            
        Returns:
            执行结果字典
        """
        # 更新VFH*目标
        vfh_star.update_target(vfh_target)

        
        # 获取当前占用地图
        observation = self.simulator.get_observation()
        depth_observation = observation.get('depth')
        if depth_observation is None:
            return {'failed': True, 'reason': 'no_depth_data'}
        
        # 更新占用地图
        # current_pos = np.array([current_pos_2d[0], 0, current_pos_2d[1]])
        current_pos = self._get_robot_state()['position']
        agent_pose = {'position': current_pos, 'rotation': current_rot}
        self.map_builder.update_map(depth_observation, agent_pose, 90.0)  # 90度FOV
        
        # 从占用地图提取局部障碍物
        obstacles = self.map_builder.get_obstacles_from_map(current_pos_2d, vfh_star.sensor_range)
        
        # 获取机器人朝向角度
        robot_theta = self.map_builder.get_robot_theta_from_quaternion(current_rot)
        
        # VFH*计算最佳方向
        ideal_direction = vfh_star.get_best_direction(current_pos_2d, robot_theta, obstacles, prev_direction)
        
        if ideal_direction is None:
            return {'failed': True, 'reason': 'no_feasible_path'}
            
        action_name, action_value = vfh_star.get_discrete_action(ideal_direction, robot_theta)
        
        # 执行低层动作
        self._execute_vfh_action(action_name, action_value, current_pos, current_rot)
        
        # 添加视频帧
        current_state = self.simulator.get_robot_state()
        current_observation = self.simulator.get_observation()
        self.composer.add_frame(robot_state=current_state, observation=current_observation)
        
        return {
            'success': False,
            'failed': False,
            'prev_direction': ideal_direction
        }
    
    def _execute_hybrid_navigation(self, target_pos_2d: np.ndarray, target_name: str, vfh_star: VFHStar) -> Dict[str, Any]:
        """
        主导航控制器 - 支持混合导航和对象检测导航两阶段
        
        Args:
            target_pos_2d: 目标位置 [x, z]
            target_name: 目标名称
            vfh_star: VFH*算法实例
        
        Returns:
            导航结果
        """
        print(f"开始主导航控制器，目标: {target_name}")
        
        # 初始化导航状态
        navigation_phase = NavigationPhase.HYBRID_NAVIGATION
        current_path = None
        adjusted_target_pos = target_pos_2d
        detected_target_pos = None  # 检测到的目标位置
        
        iteration = 0
        action_count = 0
        prev_direction = None
        
        while iteration < self.nav_config.max_iterations:
            iteration += 1
            
            # 获取当前机器人状态
            robot_state = self._get_robot_state()
            current_pos = robot_state['position']
            current_rot = robot_state['rotation']
            current_pos_2d = robot_state['position_2d']
            
            # 检查阶段切换
            if (navigation_phase == NavigationPhase.HYBRID_NAVIGATION and 
                self._should_switch_to_object_detection(current_pos_2d, current_path, target_pos_2d)):
                
                navigation_phase = NavigationPhase.OBJECT_DETECTION_NAVIGATION
                print(f"[PHASE SWITCH] 切换到对象检测导航阶段")
                if current_path is not None:
                    path_distance = self._calculate_path_distance_to_target(current_pos_2d, current_path, target_pos_2d)
                    print(f"当前路径距离: {path_distance:.2f}m")
            
            # 根据当前阶段执行相应的导航逻辑
            if navigation_phase == NavigationPhase.HYBRID_NAVIGATION:
                print(f"迭代 {iteration}: 执行混合导航阶段")
                result = self._execute_hybrid_phase(
                    current_pos_2d, current_rot, target_pos_2d, 
                    vfh_star, current_path, adjusted_target_pos, 
                    action_count, prev_direction
                )
                
                if result['success']:
                    return result
                elif result['failed']:
                    return result
                
                # 更新状态
                current_path = result.get('current_path', current_path)
                adjusted_target_pos = result.get('adjusted_target_pos', adjusted_target_pos)
                action_count = result.get('action_count', action_count)
                prev_direction = result.get('prev_direction', prev_direction)
                
            elif navigation_phase == NavigationPhase.OBJECT_DETECTION_NAVIGATION:
                print(f"迭代 {iteration}: 执行对象检测导航阶段")
                result = self._execute_object_detection_phase(
                    current_pos_2d, current_rot, target_name, vfh_star, prev_direction,
                    adjusted_target_pos, detected_target_pos
                )
                
                if result['success']:
                    return result
                elif result['failed']:
                    return result
                
                prev_direction = result.get('prev_direction', prev_direction)
                detected_target_pos = result.get('detected_target_pos', detected_target_pos)
                adjusted_target_pos = result.get('target_pos_2d', adjusted_target_pos)
        
        return {
            'success': False,
            'reason': 'max_iterations_exceeded',
            'message': f'达到最大迭代次数{self.nav_config.max_iterations}，导航失败'
        }
    
    def _plan_a_star_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        使用A*算法规划路径
        
        Args:
            start_pos: 起始位置 [x, z]
            goal_pos: 目标位置 [x, z]
        
        Returns:
            包含路径信息的字典，如果无路径则返回None
            字典格式: {
                'path': List[np.ndarray],  # A*路径点列表（世界坐标）
                'adjusted_goal': np.ndarray,  # 调整后的目标点（如果原目标在障碍物内）
                'was_adjusted': bool  # 是否调整了目标点
            }
        """
        try:
            # 获取占用地图
            if self.map_builder.grid_map is None:
                print("[ERROR] 占用地图未初始化")
                return None
            
            map = self.map_builder.grid_map.copy()
            
            
            # 下采样地图以提高A*性能（宽高都缩小到1/4）
            downsample_factor = 4
            downsampled_map = self._downsample_map(map, downsample_factor)
            
            # 将世界坐标转换为地图坐标
            start_map_coords = self.map_builder._world_to_map_coords(np.array([[start_pos[0], 0, start_pos[1]]]))
            goal_map_coords = self.map_builder._world_to_map_coords(np.array([[goal_pos[0], 0, goal_pos[1]]]))
            
            start_pixel = (int(start_map_coords[0, 0]), int(start_map_coords[0, 1]))
            goal_pixel = (int(goal_map_coords[0, 0]), int(goal_map_coords[0, 1]))
            
            # 将像素坐标转换为下采样地图的坐标
            start_pixel_downsampled = (start_pixel[0] // downsample_factor, start_pixel[1] // downsample_factor)
            goal_pixel_downsampled = (goal_pixel[0] // downsample_factor, goal_pixel[1] // downsample_factor)

            
            # 检查下采样像素坐标是否在地图范围内
            if (start_pixel_downsampled[0] < 0 or start_pixel_downsampled[0] >= downsampled_map.shape[1] or 
                start_pixel_downsampled[1] < 0 or start_pixel_downsampled[1] >= downsampled_map.shape[0]):
                print(f"[ERROR] 起始下采样像素坐标超出地图范围: {start_pixel_downsampled}, 下采样地图尺寸: {downsampled_map.shape}")
                return None
            
            if (goal_pixel_downsampled[0] < 0 or goal_pixel_downsampled[0] >= downsampled_map.shape[1] or 
                goal_pixel_downsampled[1] < 0 or goal_pixel_downsampled[1] >= downsampled_map.shape[0]):
                print(f"[ERROR] 目标下采样像素坐标超出地图范围: {goal_pixel_downsampled}, 下采样地图尺寸: {downsampled_map.shape}")
                return None
            
            # 检查起点和终点是否可行走（在下采样地图上）
            start_value = downsampled_map[start_pixel_downsampled[1], start_pixel_downsampled[0]]
            goal_value = downsampled_map[goal_pixel_downsampled[1], goal_pixel_downsampled[0]]
            
            if start_value == 0:
                print(f"[ERROR] 起始点不可行走 (地图值: {start_value})")
                return None
            
            # 检查目标点是否在障碍物内
            adjusted_goal = goal_pos
            was_adjusted = False
            
            if goal_value == 0:
                print(f"[WARNING] 目标点在障碍物内，寻找最近的可行走点")
                adjusted_goal_pixel_downsampled = self._find_nearest_walkable_point(goal_pixel_downsampled, downsampled_map)
                
                if adjusted_goal_pixel_downsampled is None:
                    print(f"[ERROR] 无法找到目标点附近的可行走区域")
                    return None
                
                # 将下采样的像素坐标转换回原始像素坐标
                adjusted_goal_pixel = (adjusted_goal_pixel_downsampled[0] * downsample_factor, 
                                     adjusted_goal_pixel_downsampled[1] * downsample_factor)
                
                # 将调整后的像素坐标转换为世界坐标
                adjusted_goal = self._pixel_to_world_coord(adjusted_goal_pixel)
                was_adjusted = True
                
                # 更新goal_pixel_downsampled为调整后的下采样像素坐标
                goal_pixel_downsampled = adjusted_goal_pixel_downsampled
                goal_value = downsampled_map[goal_pixel_downsampled[1], goal_pixel_downsampled[0]]
                

            
            # 在下采样地图上运行A*算法
            path_pixels_downsampled = self._a_star_pathfinding(start_pixel_downsampled, goal_pixel_downsampled, downsampled_map)
            
            if not path_pixels_downsampled:
                print("[ERROR] A*算法未找到路径")
                return None
            
            # 将下采样的像素坐标转换回原始分辨率
            path_pixels = []
            for pixel_downsampled in path_pixels_downsampled:
                pixel_original = (pixel_downsampled[0] * downsample_factor, pixel_downsampled[1] * downsample_factor)
                path_pixels.append(pixel_original)
            
            # 将像素坐标转换回世界坐标
            path_world = []
            for pixel in path_pixels:
                world_coord = self._pixel_to_world_coord(pixel)
                path_world.append(world_coord)

            return {
                'path': path_world,
                'adjusted_goal': adjusted_goal,
                'was_adjusted': was_adjusted
            }
            
        except Exception as e:
            print(f"[ERROR] A*路径规划失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _add_robot_padding(self) -> np.ndarray:
        """
        为占用地图添加机器人半径padding
        
        Returns:
            添加padding后的地图
        """
        from scipy import ndimage
        
        # 复制原地图
        map = self.map_builder.grid_map.copy()
        
        # 机器人半径（像素）
        robot_radius_pixels = max(1, int(0.14 / self.map_builder.map_resolution))  # 0.14m机器人半径
        
        
        # 统计padding前的地图状态
        total_cells = map.size
        free_cells_before = np.sum(map == 255)
        occupied_cells_before = np.sum(map == 0)
        unknown_cells_before = np.sum(map == 128)
 
        # 对障碍物进行膨胀操作
        kernel = np.ones((2 * robot_radius_pixels + 1, 2 * robot_radius_pixels + 1), dtype=np.uint8)
        dilated = ndimage.binary_dilation(map == 0, structure=kernel)
        
        # 更新地图：膨胀后的区域标记为障碍物
        map[dilated] = 0
        
        
        return map
    

    
    def _a_star_pathfinding(self, start: tuple, goal: tuple, wall_mask: np.ndarray) -> List[tuple]:
        """
        A*路径规划算法 - 修改版，倾向于远离障碍物
        
        Args:
            start: 起始位置 (x, y) 像素坐标
            goal: 目标位置 (x, y) 像素坐标
            wall_mask: 占用地图，0=障碍物，255=可行走
        
        Returns:
            路径点列表，如果无路径则返回空列表
        """
        import heapq
        from scipy import ndimage
        
        # 检查起点和终点是否可行走（允许未知区域）
        if wall_mask[start[1], start[0]] == 0 or wall_mask[goal[1], goal[0]] == 0:
            return []
        
        # 1. 预处理：生成距离变换图
        distance_map = self._compute_distance_transform(wall_mask)
        
        # 调试信息：显示距离变换的统计信息
        min_dist = np.min(distance_map)
        max_dist = np.max(distance_map)
        mean_dist = np.mean(distance_map)
        
        # 2. 设置参数

        # weight_w = self.nav_config.a_star_weight_w  # 权重系数，控制远离障碍物的程度
        constant = 0.009
        resolution = self.map_builder.map_resolution
        weight_w = constant / (resolution**2)
        epsilon = self.nav_config.a_star_epsilon  # 防止除零的小正数
        
        # 优先级队列: (f_score, g_score, position)
        open_set = [(0.0, 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self._calculate_f_score(start, 0.0, goal, distance_map, weight_w, epsilon)}
        
        visited = set()
        iterations = 0
        max_iterations = 2000000 # 大幅增加最大迭代次数

        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            if iterations % 20000 == 0:
                # 如果开放集过大，可能路径不存在
                if len(open_set) > 10000:
                    break
            
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                return path
            
            # 8连通邻居
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # 检查边界
                    if (0 <= neighbor[0] < wall_mask.shape[1] and 
                        0 <= neighbor[1] < wall_mask.shape[0]):
                        
                        # 检查是否可行走
                        if wall_mask[neighbor[1], neighbor[0]] in [255, 128]:  # 空闲区域或未知区域都可行走
                            
                            # 计算移动成本
                            movement_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                            tentative_g_score = g_score[current] + movement_cost
                            
                            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                                came_from[neighbor] = current
                                g_score[neighbor] = tentative_g_score
                                f_score[neighbor] = self._calculate_f_score(
                                    neighbor, tentative_g_score, goal, distance_map, weight_w, epsilon
                                )
                                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
        
        return []
    
    def _compute_distance_transform(self, wall_mask: np.ndarray) -> np.ndarray:
        """
        计算距离变换图
        
        Args:
            wall_mask: 占用地图，0=障碍物，255=可行走，128=未知
            
        Returns:
            距离变换图，每个像素值表示到最近障碍物的距离
        """
        from scipy import ndimage
        
        # 创建二值地图：0=障碍物，1=自由空间
        binary_map = (wall_mask != 0).astype(np.uint8)
        
        # 使用scipy的distance_transform_edt计算欧几里得距离变换
        # 这个函数计算每个像素到最近障碍物的欧几里得距离
        distance_map = ndimage.distance_transform_edt(binary_map)
        
        # 对于未知区域（128），给予中等距离值，避免过度惩罚
        unknown_mask = (wall_mask == 128)
        if np.any(unknown_mask):
            # 未知区域给予一个合理的距离值
            unknown_distance = self.nav_config.unknown_region_distance
            distance_map[unknown_mask] = np.maximum(distance_map[unknown_mask], unknown_distance)
        
        return distance_map
    
    def _calculate_f_score(self, node: tuple, g_score: float, goal: tuple, 
                          distance_map: np.ndarray, weight_w: float, epsilon: float) -> float:
        """
        计算修改后的f分数：f(n) = g(n) + h(n) + w * C(n)
        
        Args:
            node: 当前节点 (x, y)
            g_score: 从起点到当前节点的实际成本
            goal: 目标节点 (x, y)
            distance_map: 距离变换图
            weight_w: 权重系数
            epsilon: 防止除零的小正数
            
        Returns:
            修改后的f分数
        """
        # 计算h(n) - 启发式成本（欧几里得距离）
        h_n = self._euclidean_distance(node, goal)
        
        # 获取D(n) - 当前节点到最近障碍物的距离
        d_n = distance_map[node[1], node[0]]
        
        # 计算C(n) - 基于距离的惩罚项（反比关系）
        c_n = 1.0 / (d_n + epsilon)
        
        # 计算总f分数：f(n) = g(n) + h(n) + w * C(n)
        return g_score + h_n + weight_w * c_n
    
    def _euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        """计算欧几里得距离"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def _calculate_path_distance_to_target(self, current_pos: np.ndarray, path: List[np.ndarray], target_pos: np.ndarray) -> float:
        """
        计算沿着路径到目标位置的弧长
        
        Args:
            current_pos: 当前位置 [x, z]
            path: A*路径点列表
            target_pos: 目标位置 [x, z]
        
        Returns:
            沿着路径到目标的弧长
        """
        if not path or len(path) < 2:
            # 如果路径为空，返回直线距离
            return np.sqrt((current_pos[0] - target_pos[0])**2 + (current_pos[1] - target_pos[1])**2)
        
        # 找到路径上距离当前位置最近的点
        min_dist = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(path):
            dist = np.sqrt((current_pos[0] - path_point[0])**2 + (current_pos[1] - path_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 从最近点开始，沿路径计算弧长到目标
        accumulated_distance = 0.0
        
        # 首先计算从当前位置到最近路径点的距离
        if closest_idx < len(path):
            accumulated_distance += np.sqrt(
                (current_pos[0] - path[closest_idx][0])**2 + (current_pos[1] - path[closest_idx][1])**2
            )
        
        # 然后沿着路径计算到目标的弧长
        for i in range(closest_idx, len(path) - 1):
            # 计算当前段长度
            segment_length = np.sqrt(
                (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
            )
            accumulated_distance += segment_length
            
            # 检查是否到达目标（如果路径点接近目标）
            if np.sqrt((path[i+1][0] - target_pos[0])**2 + (path[i+1][1] - target_pos[1])**2) < 0.1:
                break
        
        return accumulated_distance

    def _get_intermediate_target(self, current_pos: np.ndarray, path: List[np.ndarray], target_distance: float) -> Optional[np.ndarray]:
        """
        从A*路径上找到距离当前位置指定距离的中间目标点
        
        Args:
            current_pos: 当前位置 [x, z]
            path: A*路径点列表
            target_distance: 目标距离
        
        Returns:
            中间目标点 [x, z]，如果找不到则返回None
        """
        if not path or len(path) < 2:
            return None
        
        # 找到路径上距离当前位置最近的点
        min_dist = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(path):
            dist = np.sqrt((current_pos[0] - path_point[0])**2 + (current_pos[1] - path_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 从最近点开始，沿路径计算弧长
        accumulated_distance = 0.0
        
        for i in range(closest_idx, len(path) - 1):
            # 计算当前段长度
            segment_length = np.sqrt(
                (path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2
            )
            
            # 如果加上这段长度超过目标距离
            if accumulated_distance + segment_length >= target_distance:
                # 计算插值比例
                remaining_distance = target_distance - accumulated_distance
                ratio = remaining_distance / segment_length
                
                # 插值得到中间目标点
                intermediate_x = path[i][0] + ratio * (path[i+1][0] - path[i][0])
                intermediate_z = path[i][1] + ratio * (path[i+1][1] - path[i][1])
                
                return np.array([intermediate_x, intermediate_z])
            
            accumulated_distance += segment_length
        
        # 如果路径总长度小于目标距离，返回最后一个点
        return np.array(path[-1])
    
    def _pixel_to_world_coord(self, pixel: tuple) -> np.ndarray:
        """
        将像素坐标转换为世界坐标
        
        Args:
            pixel: 像素坐标 (x, y)
        
        Returns:
            世界坐标 [x, z]
        """
        # 使用map_builder的逆变换
        world_x = (pixel[0] + 0.5) * self.map_builder.map_resolution + self.map_builder.topdown_map_bounds['top_left'][0]
        world_z = (pixel[1] + 0.5) * self.map_builder.map_resolution + self.map_builder.topdown_map_bounds['top_left'][1]
        
        return np.array([world_x, world_z])
    
    def _is_safe_walkable_point(self, pixel: tuple, map: np.ndarray, safety_radius_m: float = 0.1) -> bool:
        """
        检查一个点是否为安全的可行走点（周围指定半径内都是非障碍物）
        
        Args:
            pixel: 像素坐标 (x, y)
            map: 占用地图，0=障碍物，255=可行走，128=未知
            safety_radius_m: 安全半径（米）
        
        Returns:
            True表示该点周围安全区域内都是非障碍物
        """
        x, y = pixel
        height, width = map.shape
        
        # 将安全半径转换为像素单位
        safety_radius_pixels = int(np.ceil(safety_radius_m / self.map_builder.map_resolution))
        
        # 检查安全区域内的所有点
        for dx in range(-safety_radius_pixels, safety_radius_pixels + 1):
            for dy in range(-safety_radius_pixels, safety_radius_pixels + 1):
                # 计算实际距离（欧几里得距离）
                distance_pixels = np.sqrt(dx * dx + dy * dy)
                
                # 只检查在安全半径内的点
                if distance_pixels <= safety_radius_pixels:
                    nx, ny = x + dx, y + dy
                    
                    # 检查边界
                    if 0 <= nx < width and 0 <= ny < height:
                        # 如果发现障碍物，则不安全
                        if map[ny, nx] == 0:
                            return False
                    else:
                        # 超出边界也视为不安全
                        return False
        
        return True

    def _find_nearest_walkable_point(self, obstacle_pixel: tuple, map: np.ndarray) -> Optional[tuple]:
        """
        找到距离障碍物点最近的安全可行走点（确保周围0.1米内都是非障碍物）
        
        Args:
            obstacle_pixel: 障碍物像素坐标 (x, y)
            map: 占用地图，0=障碍物，255=可行走，128=未知
        
        Returns:
            最近的安全可行走像素坐标，如果找不到则返回None
        """
        x, y = obstacle_pixel
        height, width = map.shape
        
        # 从最小半径开始搜索
        for radius in range(0, self.nav_config.max_search_radius + 1):
            candidates = []
            
            # 在指定半径内搜索所有点
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # 只检查半径边界上的点（避免重复检查内层）
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = x + dx, y + dy
                        
                        # 检查边界
                        if 0 <= nx < width and 0 <= ny < height:
                            # 检查是否为可行走区域（空闲或未知）
                            if map[ny, nx] in [255, 128]:
                                # 进一步检查是否为安全的可行走点
                                if self._is_safe_walkable_point((nx, ny), map):
                                    # 计算到原障碍物点的距离
                                    distance = np.sqrt(dx*dx + dy*dy)
                                    candidates.append((nx, ny, distance))
            
            # 如果找到候选点，返回最近的一个
            if candidates:
                # 按距离排序
                candidates.sort(key=lambda c: c[2])
                nearest_pixel = (candidates[0][0], candidates[0][1])
                return nearest_pixel
        
        print(f"[ERROR] 在半径{self.nav_config.max_search_radius}内未找到安全可行走点")
        return None
    
    def _check_connectivity(self, point: tuple, wall_mask: np.ndarray) -> int:
        """
        检查一个点的连通性（周围可行走区域的数量）
        
        Args:
            point: 检查的点 (x, y)
            wall_mask: 地图
            
        Returns:
            周围可行走区域的数量
        """
        x, y = point
        accessible_count = 0
        
        # 检查8连通邻居
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                
                # 检查边界
                if (0 <= nx < wall_mask.shape[1] and 0 <= ny < wall_mask.shape[0]):
                    if wall_mask[ny, nx] in [255, 128]:
                        accessible_count += 1
        
        return accessible_count
    
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
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_frames': self.composer.get_frame_count(),
            'total_duration': self.composer.get_frame_count() / self.fps,
            'total_distance': self.total_distance_traveled,
            'linear_speed': self.linear_speed,
            'angular_speed': self.angular_speed,
            'fps': self.fps
        }
    
    def _update_distance_tracking(self, current_position: np.ndarray):
        """
        更新累计行走距离
        
        Args:
            current_position: 当前位置
        """
        if self.previous_position is not None:
            distance = np.linalg.norm(current_position - self.previous_position)
            self.total_distance_traveled += distance
        self.previous_position = current_position.copy()
    
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
        
        # 获取当前旋转（保持不变）
        current_state = self.simulator.get_robot_state()
        current_rot = current_state['rotation']
        
        # 更新距离跟踪 - 在开始移动前记录起始位置
        self._update_distance_tracking(start_pos)
        
        # 逐帧插值 - 修复：使用 range(1, num_frames + 1) 避免重复起始帧
        for frame in range(1, num_frames + 1):
            t = frame / num_frames
            
            # 线性插值位置
            interpolated_pos = start_pos + (end_pos - start_pos) * t
            
            # 更新机器人姿态
            self.simulator.set_robot_pose(interpolated_pos, current_rot)
            
            # 更新距离跟踪
            self._update_distance_tracking(interpolated_pos)
            
            # 添加视频帧
            self.composer.add_frame()
    
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
