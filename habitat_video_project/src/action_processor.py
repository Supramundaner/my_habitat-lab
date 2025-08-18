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
    
    def execute_sequence(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行目标导航序列
        
        Args:
            action_data: 动作数据字典，包含target_info和wall_mask_path
        
        Returns:
            执行结果报告
        """
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
        
        # 将VFH实例传递给video_composer以启用histogram可视化
        self.composer.set_vfh_instance(vfh_star)
        
        # 执行混合导航（A* + VFH*）
        return self._execute_hybrid_navigation(target_pos_2d, target_name, vfh_star)
    
    def _execute_hybrid_navigation(self, target_pos_2d: np.ndarray, target_name: str, vfh_star: VFHStar) -> Dict[str, Any]:
        """
        执行混合导航（A* + VFH*）
        
        Args:
            target_pos_2d: 目标位置 [x, z]
            target_name: 目标名称
            vfh_star: VFH*算法实例
        
        Returns:
            导航结果
        """
        print(f"开始A* + VFH*混合导航到目标: {target_name}")
        
        # 导航参数
        a_star_interval = 5  # 每5次行动重新规划A*
        intermediate_distance = 1.5  # 中间目标点距离
        final_target_threshold = 1.5  # 切换到最终目标的阈值
        final_stop_threshold = 0.8  # 最终停止阈值
        
        # 跟踪变量
        action_count = 0  # 行动计数器
        current_path = None  # 当前A*路径
        current_path_points = None  # 当前路径点（世界坐标）
        
        max_iterations = 1000  # 防止无限循环
        iteration = 0
        prev_direction = None
        
        while iteration < max_iterations:
            iteration += 1
            
            # 获取当前机器人状态
            current_state = self.simulator.get_robot_state()
            current_pos = current_state['position']
            current_rot = current_state['rotation']
            current_pos_2d = np.array([current_pos[0], current_pos[2]])
            
            # 计算到最终目标的距离
            dist_to_final_target = np.sqrt((current_pos[0] - target_pos_2d[0])**2 + (current_pos[2] - target_pos_2d[1])**2)
            
            print(f"迭代 {iteration}: 到最终目标距离: {dist_to_final_target:.2f}m")
            
            # 检查是否到达最终目标
            if dist_to_final_target < final_stop_threshold:
                print(f"[SUCCESS] 成功到达最终目标位置 ({target_pos_2d[0]}, {target_pos_2d[1]})")
                return {'success': True}
            
            # 每5次行动（包括第一次）重新运行A*算法
            if action_count % a_star_interval == 0:
                print(f"重新规划A*路径 (行动次数: {action_count})")
                current_path = self._plan_a_star_path(current_pos_2d, target_pos_2d)
                
                if current_path is None:
                    return {
                        'success': False,
                        'reason': 'no_a_star_path',
                        'message': 'A*算法无法找到路径'
                    }
                
                print(f"A*路径规划成功，路径点数: {len(current_path)}")
            
            # 确定当前VFH*目标
            if dist_to_final_target < final_target_threshold:
                # 距离最终目标很近，直接以最终目标为VFH*目标
                vfh_target = target_pos_2d
                target_type = "最终目标"
            else:
                # 使用A*路径上的中间目标点
                vfh_target = self._get_intermediate_target(current_pos_2d, current_path, intermediate_distance)
                if vfh_target is None:
                    return {
                        'success': False,
                        'reason': 'no_intermediate_target',
                        'message': '无法找到中间目标点'
                    }
                target_type = "中间目标"
            
            # 更新VFH*目标
            vfh_star.update_target(vfh_target)
            print(f"VFH*目标更新为: {target_type} ({vfh_target[0]:.2f}, {vfh_target[1]:.2f})")
            
            # 获取当前占用地图
            observation = self.simulator.get_observation()
            depth_observation = observation.get('depth')
            if depth_observation is None:
                return {
                    'success': False,
                    'reason': 'no_depth_data',
                    'message': '无法获取深度传感器数据'
                }
            
            # 更新占用地图
            agent_pose = {'position': current_pos, 'rotation': current_rot}
            self.map_builder.update_map(depth_observation, agent_pose, 90.0)  # 90度FOV
            
            # 从占用地图提取局部障碍物
            robot_pos_2d = np.array([current_pos[0], current_pos[2]])
            obstacles = self.map_builder.get_obstacles_from_map(robot_pos_2d, vfh_star.sensor_range)
            
            # 获取机器人朝向角度
            robot_theta = self.map_builder.get_robot_theta_from_quaternion(current_rot)
            
            # VFH*计算最佳方向
            ideal_direction = vfh_star.get_best_direction(robot_pos_2d, robot_theta, obstacles, prev_direction)
            
            if ideal_direction is None:
                return {
                    'success': False,
                    'reason': 'no_feasible_path',
                    'message': 'VFH*无法找到可行方向'
                }
                
            action_name, action_value = vfh_star.get_discrete_action(ideal_direction, robot_theta)
            
            # 执行低层动作
            if action_name == "turn_left":
                self._handle_turn_left({'angle': action_value})
            elif action_name == "turn_right":
                self._handle_turn_right({'angle': action_value})
            else:
                # 前进动作
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
                
                # 执行移动到目标位置
                self._animate_movement(current_pos, end_pos)
            
            # 增加行动计数器
            action_count += 1
            
            # 更新前一个方向
            prev_direction = ideal_direction
            
            # 添加视频帧
            current_state = self.simulator.get_robot_state()
            current_observation = self.simulator.get_observation()
            self.composer.add_frame(robot_state=current_state, observation=current_observation)
        
        return {
            'success': False,
            'reason': 'max_iterations_exceeded',
            'message': f'达到最大迭代次数{max_iterations}，导航失败'
        }
    
    def _plan_a_star_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        使用A*算法规划路径
        
        Args:
            start_pos: 起始位置 [x, z]
            goal_pos: 目标位置 [x, z]
        
        Returns:
            A*路径点列表（世界坐标），如果无路径则返回None
        """
        try:
            print(f"[DEBUG] A*路径规划开始")
            print(f"[DEBUG] 起始位置 (世界坐标): {start_pos}")
            print(f"[DEBUG] 目标位置 (世界坐标): {goal_pos}")
            
            # 获取占用地图
            if self.map_builder.grid_map is None:
                print("[ERROR] 占用地图未初始化")
                return None
            
            print(f"[DEBUG] 地图尺寸: {self.map_builder.grid_map.shape}")
            print(f"[DEBUG] 地图分辨率: {self.map_builder.map_resolution}")
            print(f"[DEBUG] 地图边界: {self.map_builder.topdown_map_bounds}")
            
            # 添加机器人半径padding
            # padded_map = self._add_robot_padding()
            padded_map = self.map_builder.grid_map.copy()
            print(f"[DEBUG] Padding后地图尺寸: {padded_map.shape}")
            
            # 临时测试：尝试不使用padding的地图
            print(f"[DEBUG] 临时测试：尝试不使用padding的地图")
            
            # no_padding_map = self.map_builder.grid_map.copy()
            # no_padding_free_cells = np.sum(no_padding_map == 255)
            #no_padding_unknown_cells = np.sum(no_padding_map == 128)
            #print(f"[DEBUG] 无padding地图 - 空闲: {no_padding_free_cells}, 未知: {no_padding_unknown_cells}")
            
            # 统计地图状态
            total_cells = padded_map.size
            free_cells = np.sum(padded_map == 255)
            occupied_cells = np.sum(padded_map == 0)
            unknown_cells = np.sum(padded_map == 128)
            print(f"[DEBUG] 地图统计 - 总单元格: {total_cells}, 空闲: {free_cells} ({free_cells/total_cells*100:.1f}%), 占用: {occupied_cells} ({occupied_cells/total_cells*100:.1f}%), 未知: {unknown_cells} ({unknown_cells/total_cells*100:.1f}%)")
            
            # 将世界坐标转换为地图坐标
            start_map_coords = self.map_builder._world_to_map_coords(np.array([[start_pos[0], 0, start_pos[1]]]))
            goal_map_coords = self.map_builder._world_to_map_coords(np.array([[goal_pos[0], 0, goal_pos[1]]]))
            
            start_pixel = (int(start_map_coords[0, 0]), int(start_map_coords[0, 1]))
            goal_pixel = (int(goal_map_coords[0, 0]), int(goal_map_coords[0, 1]))
            
            print(f"[DEBUG] 起始位置 (像素坐标): {start_pixel}")
            print(f"[DEBUG] 目标位置 (像素坐标): {goal_pixel}")
            
            # 检查像素坐标是否在地图范围内
            if (start_pixel[0] < 0 or start_pixel[0] >= padded_map.shape[1] or 
                start_pixel[1] < 0 or start_pixel[1] >= padded_map.shape[0]):
                print(f"[ERROR] 起始像素坐标超出地图范围: {start_pixel}, 地图尺寸: {padded_map.shape}")
                return None
            
            if (goal_pixel[0] < 0 or goal_pixel[0] >= padded_map.shape[1] or 
                goal_pixel[1] < 0 or goal_pixel[1] >= padded_map.shape[0]):
                print(f"[ERROR] 目标像素坐标超出地图范围: {goal_pixel}, 地图尺寸: {padded_map.shape}")
                return None
            
            # 检查起点和终点是否可行走
            start_value = padded_map[start_pixel[1], start_pixel[0]]
            goal_value = padded_map[goal_pixel[1], goal_pixel[0]]
            
            print(f"[DEBUG] 起始点地图值: {start_value} (255=可行走, 0=障碍物, 128=未知)")
            print(f"[DEBUG] 目标点地图值: {goal_value} (255=可行走, 0=障碍物, 128=未知)")
            
            if start_value == 0:
                print(f"[ERROR] 起始点不可行走 (地图值: {start_value})")
                return None
            
            if goal_value == 0:
                print(f"[ERROR] 目标点不可行走 (地图值: {goal_value})")
                return None
            
            # 运行A*算法
            print(f"[DEBUG] 开始运行A*算法...")
            
            # 检查起点和终点之间的连通性
            print(f"[DEBUG] 检查连通性...")
            start_accessible = self._check_connectivity(start_pixel, padded_map)
            goal_accessible = self._check_connectivity(goal_pixel, padded_map)
            print(f"[DEBUG] 起点连通性: {start_accessible}, 终点连通性: {goal_accessible}")
            
            path_pixels = self._a_star_pathfinding(start_pixel, goal_pixel, padded_map)
            
            if not path_pixels:
                print("[ERROR] A*算法未找到路径")
                return None
            
            print(f"[DEBUG] A*算法找到路径，像素点数: {len(path_pixels)}")
            
            # 将像素坐标转换回世界坐标
            path_world = []
            for pixel in path_pixels:
                world_coord = self._pixel_to_world_coord(pixel)
                path_world.append(world_coord)
            
            print(f"[DEBUG] 路径规划完成，世界坐标点数: {len(path_world)}")
            return path_world
            
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
        padded_map = self.map_builder.grid_map.copy()
        
        # 机器人半径（像素）
        robot_radius_pixels = max(1, int(0.14 / self.map_builder.map_resolution))  # 0.14m机器人半径
        
        print(f"[DEBUG] 机器人半径: 0.14m")
        print(f"[DEBUG] 地图分辨率: {self.map_builder.map_resolution}m/pixel")
        print(f"[DEBUG] 机器人半径 (像素): {robot_radius_pixels}")
        
        # 统计padding前的地图状态
        total_cells = padded_map.size
        free_cells_before = np.sum(padded_map == 255)
        occupied_cells_before = np.sum(padded_map == 0)
        unknown_cells_before = np.sum(padded_map == 128)
        
        print(f"[DEBUG] Padding前 - 空闲: {free_cells_before} ({free_cells_before/total_cells*100:.1f}%), 占用: {occupied_cells_before} ({occupied_cells_before/total_cells*100:.1f}%), 未知: {unknown_cells_before} ({unknown_cells_before/total_cells*100:.1f}%)")
        
        # 对障碍物进行膨胀操作
        kernel = np.ones((2 * robot_radius_pixels + 1, 2 * robot_radius_pixels + 1), dtype=np.uint8)
        dilated = ndimage.binary_dilation(padded_map == 0, structure=kernel)
        
        # 更新地图：膨胀后的区域标记为障碍物
        padded_map[dilated] = 0
        
        # 统计padding后的地图状态
        free_cells_after = np.sum(padded_map == 255)
        occupied_cells_after = np.sum(padded_map == 0)
        unknown_cells_after = np.sum(padded_map == 128)
        
        print(f"[DEBUG] Padding后 - 空闲: {free_cells_after} ({free_cells_after/total_cells*100:.1f}%), 占用: {occupied_cells_after} ({occupied_cells_after/total_cells*100:.1f}%), 未知: {unknown_cells_after} ({unknown_cells_after/total_cells*100:.1f}%)")
        print(f"[DEBUG] Padding减少了 {free_cells_before - free_cells_after} 个空闲单元格")
        
        return padded_map
    

    
    def _a_star_pathfinding(self, start: tuple, goal: tuple, wall_mask: np.ndarray) -> List[tuple]:
        """
        A*路径规划算法
        
        Args:
            start: 起始位置 (x, y) 像素坐标
            goal: 目标位置 (x, y) 像素坐标
            wall_mask: 占用地图，0=障碍物，255=可行走
        
        Returns:
            路径点列表，如果无路径则返回空列表
        """
        import heapq
        
        # 检查起点和终点是否可行走（允许未知区域）
        if wall_mask[start[1], start[0]] == 0 or wall_mask[goal[1], goal[0]] == 0:
            return []
        
        # 优先级队列: (f_score, g_score, position)
        open_set = [(0.0, 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self._euclidean_distance(start, goal)}
        
        visited = set()
        iterations = 0
        max_iterations = 200000000000  # 大幅增加最大迭代次数
        
        print(f"[DEBUG] A*算法开始，最大迭代次数: {max_iterations}")
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            if iterations % 20000 == 0:
                print(f"[DEBUG] A*迭代次数: {iterations}, 开放集大小: {len(open_set)}, 已访问节点: {len(visited)}")
                # 如果开放集过大，可能路径不存在
                if len(open_set) > 10000:
                    print(f"[DEBUG] 开放集过大({len(open_set)})，可能路径不存在，提前终止")
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
                
                print(f"[DEBUG] A*算法成功找到路径，迭代次数: {iterations}, 访问节点数: {len(visited)}")
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
                                f_score[neighbor] = tentative_g_score + self._euclidean_distance(neighbor, goal)
                                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
        
        print(f"[DEBUG] A*算法未找到路径，迭代次数: {iterations}, 访问节点数: {len(visited)}")
        print(f"[DEBUG] 开放集大小: {len(open_set)}")
        return []
    
    def _euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        """计算欧几里得距离"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
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
            'linear_speed': self.linear_speed,
            'angular_speed': self.angular_speed,
            'fps': self.fps
        }
    
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
        
        # 逐帧插值 - 修复：使用 range(1, num_frames + 1) 避免重复起始帧
        for frame in range(1, num_frames + 1):
            t = frame / num_frames
            
            # 线性插值位置
            interpolated_pos = start_pos + (end_pos - start_pos) * t
            
            # 更新机器人姿态
            self.simulator.set_robot_pose(interpolated_pos, current_rot)
            
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
