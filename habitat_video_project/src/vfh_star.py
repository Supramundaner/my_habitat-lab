"""
VFH*算法模块 - 基于Vector Field Histogram的导航算法
适配Habitat项目的占用地图和坐标系
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Any
from .utils import (
    get_target_angle_unified, 
    normalize_unified_angle, 
    delta_unified_angle,
    unified_angle_to_direction_vector,
    cartesian_to_unified_angle
)


class VFHStar:
    """
    VFH*算法实现，用于基于局部占用地图的导航
    """
    
    def __init__(self, target: np.ndarray, config: Dict[str, Any] = None):
        """
        初始化VFH*算法
        
        Args:
            target: 目标位置 [x, z] (Habitat坐标系)
            config: 配置参数
        """
        self.target = target
        self.config = config or {}
        
        # 算法参数
        self.mu1 = self.config.get('mu1', 5.0)
        self.mu2 = self.config.get('mu2', 2.0) 
        self.mu3 = self.config.get('mu3', 2.0)
        self.mu1_prime = self.config.get('mu1_prime', 5.0)
        self.mu2_prime = self.config.get('mu2_prime', 1.0)
        self.mu3_prime = self.config.get('mu3_prime', 1.0)
        self.lambda_param = self.config.get('lambda', 0.8)
        
        # 机器人参数
        self.robot_radius = self.config.get('robot_radius', 0.14)
        self.sensor_range = self.config.get('sensor_range', 2.0)
        self.ds = self.robot_radius * 0.5  # 投影距离
        
        # 直方图参数
        self.histogram_alpha = np.deg2rad(5)  # 5度
        self.num_histogram_bins = int(2 * np.pi / self.histogram_alpha)
        self.smax = 16  # 最大安全扇区大小
        
        # 对齐容差
        self.alignment_tolerance = np.deg2rad(17)
        
        # 离散动作 - 使用统一角度系统
        self.discrete_actions = {
            "turn_left_30": np.deg2rad(30),    # 正角度 = 左转
            "turn_right_30": -np.deg2rad(30),  # 负角度 = 右转
            "no_turn": 0.0
        }
        
        # 可视化相关
        self.ax_hist = None
        self.ax_tree = None
        
        print(f"VFH*算法初始化完成")
        print(f"目标位置: {target}")
        print(f"机器人半径: {self.robot_radius}m")
        print(f"传感器范围: {self.sensor_range}m")
    
    def get_best_direction(self, robot_pos: np.ndarray, robot_theta: float, 
                          obstacles: List[Tuple[float, float, float]] = None, 
                          prev_direction: float = None, map_builder = None) -> Optional[float]:
        """
        获取最佳移动方向
        
        Args:
            robot_pos: 机器人位置 [x, z]
            robot_theta: 机器人朝向角度（弧度，统一角度系统）
            obstacles: 障碍物列表 [(x, y, radius), ...]（可选，如果提供map_builder则忽略）
            prev_direction: 前一个选择的方向
            map_builder: MapBuilder实例（可选，如果提供则直接从地图计算）
            
        Returns:
            最佳方向角度（弧度，统一角度系统），如果无解则返回None
        """

        
        if prev_direction is None:
            prev_direction = robot_theta

            
        # 获取候选方向
        primary_candidates = self._get_candidate_directions(robot_pos, obstacles, map_builder)
        
        if not primary_candidates:
            print("[VFH* ERROR] 没有找到可行的候选方向")
            return None
            
        if len(primary_candidates) == 1:
            # 只有一个候选方向，直接返回
            return primary_candidates[0]
            
        # 使用A*搜索最佳路径
        ng = self.config.get('search_depth', 5)
        best_direction = self._a_star_search(robot_pos, robot_theta, primary_candidates, 
                                 ng, obstacles, prev_direction)
        return best_direction
    

    
    def _get_polar_histogram(self, pos: np.ndarray, obstacles: List[Tuple[float, float, float]] = None, map_builder = None) -> np.ndarray:
        """
        计算极坐标直方图
        
        Args:
            pos: 机器人位置 
            obstacles: 障碍物列表（可选，如果提供map_builder则忽略）
            map_builder: MapBuilder实例（可选，如果提供则直接从地图计算）
            
        Returns:
            极坐标直方图数组
        """
        # 如果提供了map_builder，使用高效的直接地图计算方法
        if map_builder is not None:
            return self._get_polar_histogram_from_map(pos, map_builder)
        
        # 否则使用原有的障碍物列表方法
        histogram = np.zeros(self.num_histogram_bins)
        
        for i, (ox, oy, orad) in enumerate(obstacles):
            dx, dy = ox - pos[0], oy - pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            

            
            if dist < self.sensor_range:
                # 使用统一角度系统计算障碍物角度
                angle = cartesian_to_unified_angle(dx, dy)
                gamma = np.arcsin(min(1.0, (self.robot_radius + orad) / dist))
                
                start_angle = normalize_unified_angle(angle - gamma)
                end_angle = normalize_unified_angle(angle + gamma)
                
                start_bin = int(normalize_unified_angle(start_angle) / self.histogram_alpha + 
                               self.num_histogram_bins/2) % self.num_histogram_bins
                end_bin = int(normalize_unified_angle(end_angle) / self.histogram_alpha + 
                             self.num_histogram_bins/2) % self.num_histogram_bins
                

                
                curr = start_bin
                while curr != end_bin:
                    histogram[curr] = 1.0
                    curr = (curr + 1) % self.num_histogram_bins
                histogram[end_bin] = 1.0
            else:
                pass
        
        
        return histogram
    
    def _get_polar_histogram_from_map(self, pos: np.ndarray, map_builder) -> np.ndarray:
        """
        从地图直接计算极坐标直方图（内部方法）
        
        Args:
            pos: 机器人位置 [x, z]
            map_builder: MapBuilder实例
            
        Returns:
            极坐标直方图数组
        """
        histogram = np.zeros(self.num_histogram_bins)
        
        # 将机器人位置转换为地图坐标
        agent_map_coords = map_builder._world_to_map_coords(np.array([[pos[0], 0, pos[1]]]))
        agent_map_x, agent_map_y = int(agent_map_coords[0, 0]), int(agent_map_coords[0, 1])
        
        # 计算传感器范围内的地图像素半径
        sensor_radius_pixels = int(self.sensor_range / map_builder.map_resolution)
        robot_radius_pixels = max(1, int(self.robot_radius / map_builder.map_resolution))
        
        # 创建搜索区域的坐标网格
        y_range = np.arange(-sensor_radius_pixels, sensor_radius_pixels + 1)
        x_range = np.arange(-sensor_radius_pixels, sensor_radius_pixels + 1)
        y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
        
        # 转换为世界坐标的相对位置
        rel_world_x = x_grid * map_builder.map_resolution
        rel_world_z = y_grid * map_builder.map_resolution  # 注意：y对应z轴
        distances = np.sqrt(rel_world_x**2 + rel_world_z**2)
        
        # 计算绝对地图坐标
        abs_x = agent_map_x + x_grid
        abs_y = agent_map_y + y_grid
        
        # 应用掩码
        valid_mask = ((abs_x >= 0) & (abs_x < map_builder.map_shape[1]) & 
                      (abs_y >= 0) & (abs_y < map_builder.map_shape[0]))
        distance_mask = (distances >= self.robot_radius) & (distances <= self.sensor_range)
        combined_mask = valid_mask & distance_mask
        
        if np.any(combined_mask):
            # 获取有效像素的占用状态
            valid_occupancy = map_builder.grid_map[abs_y[combined_mask], abs_x[combined_mask]]
            obstacle_indices = np.where(valid_occupancy == 0)[0]  # 0表示占用
            
            if len(obstacle_indices) > 0:
                # 获取障碍物像素的相对坐标
                flat_combined = combined_mask.flatten()
                valid_indices = np.where(flat_combined)[0]
                obstacle_flat_indices = valid_indices[obstacle_indices]
                
                # 转换回2D索引
                obstacle_y_indices, obstacle_x_indices = np.unravel_index(
                    obstacle_flat_indices, combined_mask.shape)
                
                # 获取障碍物的世界坐标相对位置
                obs_rel_x = rel_world_x[obstacle_y_indices, obstacle_x_indices]
                obs_rel_z = rel_world_z[obstacle_y_indices, obstacle_x_indices]
                
                # 向量化计算统一角度
                angles = np.arctan2(-obs_rel_x, -obs_rel_z)  # 基于cartesian_to_unified_angle的逻辑
                
                # 标准化角度并转换为bin索引
                normalized_angles = (angles + np.pi) % (2 * np.pi) - np.pi  # normalize_unified_angle
                bin_indices = ((normalized_angles / self.histogram_alpha + 
                               self.num_histogram_bins/2) % self.num_histogram_bins).astype(int)
                
                # 设置直方图
                histogram[bin_indices] = 1.0
        
        return histogram
    
    def _get_candidate_directions(self, pos: np.ndarray, obstacles: List[Tuple[float, float, float]] = None, map_builder = None) -> List[float]:
        """
        获取候选方向
        
        Args:
            pos: 机器人位置
            obstacles: 障碍物列表（可选，如果提供map_builder则忽略）
            map_builder: MapBuilder实例（可选，如果提供则直接从地图计算）
            
        Returns:
            候选方向列表（统一角度系统）
        """

        
        histogram = self._get_polar_histogram(pos, obstacles, map_builder)
        
        # 如果没有障碍物，直接朝向目标
        if np.all(histogram == 0):
            target_angle = get_target_angle_unified(
                np.array([pos[0], 0, pos[1]]),  # 转换为3D坐标
                np.array([self.target[0], 0, self.target[1]])
            )
            normalized_angle = normalize_unified_angle(target_angle)
            return [normalized_angle]
        
        # 找到自由扇区
        free_bins = np.where(histogram == 0)[0]
        
        
        if len(free_bins) == 0:
            print(f"[VFH* ERROR] 没有自由扇区，所有方向都被阻塞")
            return []
        
        # 分割连续的自由扇区
        diffs = np.diff(free_bins)
        split_indices = np.where(diffs > 1)[0] + 1
        
        # 分割扇区
        sectors = []
        start_idx = 0
        for split_idx in split_indices:
            sectors.append(free_bins[start_idx:split_idx])
            start_idx = split_idx
        sectors.append(free_bins[start_idx:])
        

        
        # 计算每个扇区的候选方向
        candidates = []
        target_angle = get_target_angle_unified(
                np.array([pos[0], 0, pos[1]]),  # 转换为3D坐标
                np.array([self.target[0], 0, self.target[1]])
            )
        
        for sector in sectors:
            if len(sector) == 0:
                continue
                
            if len(sector) < self.smax:
                # 小扇区：只取中心方向
                center_bin = sector[len(sector) // 2]
                center_angle = (center_bin - self.num_histogram_bins // 2) * self.histogram_alpha
                normalized_center = normalize_unified_angle(center_angle)
                candidates.append(normalized_center)
            else:
                # 大扇区：取两个安全边界方向，以及可能的目标方向
                safe_margin = self.smax // 2
                
                # 左边界方向
                left_bin = sector[safe_margin]
                left_angle = (left_bin - self.num_histogram_bins // 2) * self.histogram_alpha
                normalized_left = normalize_unified_angle(left_angle)
                candidates.append(normalized_left)
                
                # 右边界方向
                right_bin = sector[-1 - safe_margin]
                right_angle = (right_bin - self.num_histogram_bins // 2) * self.histogram_alpha
                normalized_right = normalize_unified_angle(right_angle)
                candidates.append(normalized_right)
                
                # 检查目标方向是否在当前扇区内，如果是则添加
                target_bin = int(normalize_unified_angle(target_angle) / self.histogram_alpha + self.num_histogram_bins // 2) % self.num_histogram_bins
                if target_bin in sector:
                    candidates.append(normalize_unified_angle(target_angle))

        

        
        return candidates
    

    
    def _a_star_search(self, robot_pos: np.ndarray, robot_theta: float, 
                       primary_candidates: List[float], ng: int, 
                       obstacles: List[Tuple[float, float, float]], 
                       prev_direction: float) -> float:
        """
        A*搜索最佳路径
        
        Args:
            robot_pos: 机器人位置
            robot_theta: 机器人朝向（统一角度系统）
            primary_candidates: 主要候选方向
            ng: 搜索深度
            obstacles: 障碍物列表
            prev_direction: 前一个方向
            
        Returns:
            最佳方向（统一角度系统）
        """
        open_set = []
        
        # 初始化开放集
        for cand in primary_candidates:
            g = self._cost_g0(cand, robot_pos, robot_theta, prev_direction)
            h = self._heuristic_h(cand, robot_theta, prev_direction, 1)
            
            new_pos, new_theta = self._project_robot(robot_pos, robot_theta, cand)
            
            heapq.heappush(open_set, (g + h, g, 1, (float(new_pos[0]), float(new_pos[1]), new_theta, cand), 
                                     [(float(robot_pos[0]), float(robot_pos[1])), new_pos.tolist()]))
        
        expanded_nodes = {}
        
        while open_set:
            f, g, depth, node, path = heapq.heappop(open_set)
            
            if depth not in expanded_nodes:
                expanded_nodes[depth] = []
            expanded_nodes[depth].append(path)
            
            if depth >= ng:
                # 达到最大深度，返回最佳候选
                best_candidate = min(primary_candidates, 
                                   key=lambda c: self._cost_g0(c, robot_pos, robot_theta, prev_direction))
                return best_candidate
            
            # 扩展节点
            pos_x, pos_z, theta, direction = node
            
            # 获取新的候选方向
            new_candidates = self._get_candidate_directions(np.array([pos_x, pos_z]), obstacles)
            
            for new_cand in new_candidates:
                new_g = g + self._cost_gi(new_cand, theta, direction, depth + 1)
                new_h = self._heuristic_h(new_cand, theta, direction, depth + 1)
                
                new_pos, new_theta = self._project_robot(np.array([pos_x, pos_z]), theta, new_cand)
                
                heapq.heappush(open_set, (new_g + new_h, new_g, depth + 1, 
                                         (float(new_pos[0]), float(new_pos[1]), new_theta, new_cand), 
                                         path + [new_pos.tolist()]))
        
        # 如果没有找到路径，返回最佳候选
        best_candidate = min(primary_candidates, 
                           key=lambda c: self._cost_g0(c, robot_pos, robot_theta, prev_direction))
        return best_candidate
    

    
    def _project_robot(self, pos: np.ndarray, theta: float, direction: float) -> Tuple[np.ndarray, float]:
        """
        投影机器人位置和朝向
        
        Args:
            pos: 当前位置
            theta: 当前朝向（统一角度系统）
            direction: 目标方向（统一角度系统）
            
        Returns:
            (新位置, 新朝向)
        """
        # 计算方向向量
        direction_vec = unified_angle_to_direction_vector(direction)
        
        # 投影距离
        projection_distance = 0.25  # 25cm
        
        # 新位置
        new_pos = pos + direction_vec[:2] * projection_distance  # 只使用X和Z分量
        
        # 新朝向
        new_theta = normalize_unified_angle(theta + delta_unified_angle(direction, theta))
        
        return new_pos, new_theta
    
    def _cost_g0(self, c0: float, robot_pos: np.ndarray, robot_theta: float, prev_direction: float) -> float:
        """
        计算初始成本
        
        Args:
            c0: 候选方向（统一角度系统）
            robot_pos: 机器人位置
            robot_theta: 机器人朝向（统一角度系统）
            prev_direction: 前一个方向（统一角度系统）
            
        Returns:
            成本值
        """
        target_angle = get_target_angle_unified(
            np.array([robot_pos[0], 0, robot_pos[1]]),  # 转换为3D坐标
            np.array([self.target[0], 0, self.target[1]])
        )
        
        return (self.mu1 * abs(delta_unified_angle(c0, target_angle)) + 
                self.mu2 * abs(delta_unified_angle(c0, robot_theta)) + 
                self.mu3 * abs(delta_unified_angle(c0, prev_direction)))
    
    def _cost_gi(self, ci: float, theta_i: float, ci_minus_1: float, i: int) -> float:
        """
        计算第i步的成本
        
        Args:
            ci: 当前方向（统一角度系统）
            theta_i: 当前朝向（统一角度系统）
            ci_minus_1: 前一个方向（统一角度系统）
            i: 步数
            
        Returns:
            成本值
        """
        target_angle = get_target_angle_unified(
            np.array([0, 0, 0]),  # 原点
            np.array([self.target[0], 0, self.target[1]])  # 目标位置
        )
        
        return (self.lambda_param**i) * (
            self.mu1_prime * abs(delta_unified_angle(ci, target_angle)) + 
            self.mu2_prime * abs(delta_unified_angle(ci, theta_i)) + 
            self.mu3_prime * abs(delta_unified_angle(ci, ci_minus_1))
        )
    
    def _heuristic_h(self, c: float, theta: float, prev_dir: float, depth: int) -> float:
        """
        计算启发式函数
        
        Args:
            c: 候选方向（统一角度系统）
            theta: 当前朝向（统一角度系统）
            prev_dir: 前一个方向（统一角度系统）
            depth: 搜索深度
            
        Returns:
            启发式值
        """
        target_angle = get_target_angle_unified(
            np.array([0, 0, 0]),  # 原点
            np.array([self.target[0], 0, self.target[1]])  # 目标位置
        )
        h = 0
        
        for i in range(depth, depth + 3):
            h += (self.lambda_param**i) * (
                self.mu1_prime * abs(delta_unified_angle(target_angle, theta)) + 
                self.mu2_prime * abs(delta_unified_angle(theta, prev_dir)) + 
                self.mu3_prime * abs(delta_unified_angle(target_angle, prev_dir))
            )
        
        return h
    
    def get_discrete_action(self, ideal_direction: float, current_theta: float) -> Tuple[str, float]:
        """
        将连续方向转换为离散动作
        
        Args:
            ideal_direction: 理想方向（统一角度系统）
            current_theta: 当前朝向（统一角度系统）
            
        Returns:
            (动作名称, 动作值)
        """
        direction_error = delta_unified_angle(ideal_direction, current_theta)
        
        if abs(direction_error) < self.alignment_tolerance:
            return "move_forward", 0.0
        
        # 找到最接近的离散动作
        best_action, min_err = None, float('inf')
        
        for name, val in self.discrete_actions.items():
            err = abs(delta_unified_angle(ideal_direction, normalize_unified_angle(current_theta + val)))
            if err < min_err:
                min_err, best_action = err, val
        
        action_name = "move_forward"
        action_value = 0.0
        
        if best_action == self.discrete_actions["turn_left_30"]:
            action_name = "turn_left"
            action_value = 30.0
        elif best_action == self.discrete_actions["turn_right_30"]:
            action_name = "turn_right"
            action_value = 30.0
        

        
        return action_name, action_value
    
    def set_visualization_axes(self, ax_hist=None, ax_tree=None):
        """
        设置可视化轴
        
        Args:
            ax_hist: 直方图轴
            ax_tree: 搜索树轴
        """
        self.ax_hist = ax_hist
        self.ax_tree = ax_tree
    
    def visualize_polar_histogram(self, pos: np.ndarray, obstacles: List[Tuple[float, float, float]]):
        """
        可视化极坐标直方图
        
        Args:
            pos: 机器人位置
            obstacles: 障碍物列表
        """
        if not self.ax_hist:
            return
            
        self.ax_hist.clear()
        histogram = self._get_polar_histogram(pos, obstacles)
        
        angles = np.linspace(-np.pi, np.pi, self.num_histogram_bins, endpoint=False)
        self.ax_hist.bar(angles, histogram, width=self.histogram_alpha, 
                         color='blue', alpha=0.7, align='center')
        self.ax_hist.set_theta_zero_location('E')
        self.ax_hist.set_title("Polar Histogram (VFH*)") 