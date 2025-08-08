"""
VFH*算法模块 - 基于Vector Field Histogram的导航算法
适配Habitat项目的占用地图和坐标系
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Any


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
        self.sensor_range = self.config.get('sensor_range', 0.4)
        self.ds = self.robot_radius * 0.5  # 投影距离
        
        # 直方图参数
        self.histogram_alpha = np.deg2rad(5)  # 5度
        self.num_histogram_bins = int(2 * np.pi / self.histogram_alpha)
        self.smax = 16  # 最大安全扇区大小
        
        # 对齐容差
        self.alignment_tolerance = np.deg2rad(17)
        
        # 离散动作
        self.discrete_actions = {
            "turn_left_30": np.deg2rad(30),
            "turn_right_30": -np.deg2rad(30),
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
                          obstacles: List[Tuple[float, float, float]], 
                          prev_direction: float = None) -> Optional[float]:
        """
        获取最佳移动方向
        
        Args:
            robot_pos: 机器人位置 [x, z]
            robot_theta: 机器人朝向角度（弧度）
            obstacles: 障碍物列表 [(x, y, radius), ...]
            prev_direction: 前一个选择的方向
            
        Returns:
            最佳方向角度（弧度），如果无解则返回None
        """
        if prev_direction is None:
            prev_direction = robot_theta
            
        # 获取候选方向
        primary_candidates = self._get_candidate_directions(robot_pos, obstacles)
        
        if not primary_candidates:
            print("警告: 没有找到可行的候选方向")
            return None
            
        if len(primary_candidates) == 1:
            # 只有一个候选方向，直接返回
            return primary_candidates[0]
            
        # 使用A*搜索最佳路径
        ng = self.config.get('search_depth', 5)
        return self._a_star_search(robot_pos, robot_theta, primary_candidates, 
                                 ng, obstacles, prev_direction)
    
    def _get_polar_histogram(self, pos: np.ndarray, obstacles: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        计算极坐标直方图
        
        Args:
            pos: 机器人位置 [x, z]
            obstacles: 障碍物列表
            
        Returns:
            极坐标直方图数组
        """
        histogram = np.zeros(self.num_histogram_bins)
        
        for ox, oy, orad in obstacles:
            dx, dy = ox - pos[0], oy - pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < self.sensor_range:
                angle = np.arctan2(dy, dx)
                gamma = np.arcsin(min(1.0, (self.robot_radius + orad) / dist))
                
                start_angle = self._normalize_angle(angle - gamma)
                end_angle = self._normalize_angle(angle + gamma)
                
                start_bin = int(self._normalize_angle(start_angle) / self.histogram_alpha + 
                               self.num_histogram_bins/2) % self.num_histogram_bins
                end_bin = int(self._normalize_angle(end_angle) / self.histogram_alpha + 
                             self.num_histogram_bins/2) % self.num_histogram_bins
                
                curr = start_bin
                while curr != end_bin:
                    histogram[curr] = 1.0
                    curr = (curr + 1) % self.num_histogram_bins
                histogram[end_bin] = 1.0
                
        return histogram
    
    def _get_candidate_directions(self, pos: np.ndarray, obstacles: List[Tuple[float, float, float]]) -> List[float]:
        """
        获取候选方向
        
        Args:
            pos: 机器人位置
            obstacles: 障碍物列表
            
        Returns:
            候选方向列表
        """
        histogram = self._get_polar_histogram(pos, obstacles)
        
        # 如果没有障碍物，直接朝向目标
        if np.all(histogram == 0):
            target_angle = np.arctan2(self.target[1] - pos[1], self.target[0] - pos[0])
            return [self._normalize_angle(target_angle)]
        
        # 找到自由扇区
        free_bins = np.where(histogram == 0)[0]
        if len(free_bins) == 0:
            return []
        
        # 分割连续的自由扇区
        diffs = np.diff(free_bins)
        split_indices = np.where(diffs > 1)[0] + 1
        valleys_bins = np.split(free_bins, split_indices)
        
        # 处理跨越0度的情况
        if len(valleys_bins) > 1 and free_bins[0] == 0 and free_bins[-1] == self.num_histogram_bins - 1:
            valleys_bins[-1] = np.concatenate((valleys_bins[-1], valleys_bins[0]))
            valleys_bins.pop(0)
        
        candidates = []
        target_angle = np.arctan2(self.target[1] - pos[1], self.target[0] - pos[0])
        
        for valley in valleys_bins:
            if len(valley) == 0:
                continue
                
            if len(valley) < self.smax:
                # 小扇区，选择中点
                mid_bin = valley[len(valley) // 2]
                candidates.append(self._normalize_angle(
                    (mid_bin - self.num_histogram_bins/2) * self.histogram_alpha
                ))
            else:
                # 大扇区，选择边界点
                safe_margin = self.smax // 2
                candidates.append(self._normalize_angle(
                    (valley[safe_margin] - self.num_histogram_bins/2) * self.histogram_alpha
                ))
                candidates.append(self._normalize_angle(
                    (valley[-1 - safe_margin] - self.num_histogram_bins/2) * self.histogram_alpha
                ))
                
                # 如果目标在扇区内，也加入候选
                target_bin = int(self._normalize_angle(target_angle) / self.histogram_alpha + 
                                self.num_histogram_bins/2) % self.num_histogram_bins
                if target_bin in valley:
                    candidates.append(self._normalize_angle(target_angle))
        
        return list(set(candidates))
    
    def _a_star_search(self, robot_pos: np.ndarray, robot_theta: float, 
                       primary_candidates: List[float], ng: int, 
                       obstacles: List[Tuple[float, float, float]], 
                       prev_direction: float) -> float:
        """
        A*搜索最佳路径
        
        Args:
            robot_pos: 机器人位置
            robot_theta: 机器人朝向
            primary_candidates: 主要候选方向
            ng: 搜索深度
            obstacles: 障碍物列表
            prev_direction: 前一个方向
            
        Returns:
            最佳方向
        """
        open_set = []
        
        # 初始化开放集
        for cand in primary_candidates:
            g = self._cost_g0(cand, robot_pos, robot_theta, prev_direction)
            h = self._heuristic_h(cand, robot_theta, prev_direction, 1)
            
            new_pos, new_theta = self._project_robot(robot_pos, robot_theta, cand)
            
            heapq.heappush(open_set, (g + h, g, 1, (new_pos[0], new_pos[1], new_theta, cand), 
                                     [(robot_pos[0], robot_pos[1]), new_pos]))
        
        expanded_nodes = {}
        
        while open_set:
            f, g, depth, node, path = heapq.heappop(open_set)
            
            if depth not in expanded_nodes:
                expanded_nodes[depth] = []
            expanded_nodes[depth].append(path)
            
            if depth >= ng:
                # 达到搜索深度，返回最佳主要候选
                for pc in primary_candidates:
                    p_pos, _ = self._project_robot(robot_pos, robot_theta, pc)
                    if np.allclose(p_pos, path[1]):
                        return pc
            
            x, y, theta, prev_dir = node
            
            # 获取投影位置的候选方向
            projected_candidates = self._get_candidate_directions(np.array([x, y]), obstacles)
            
            for cand in projected_candidates:
                new_pos, _ = self._project_robot(np.array([x, y]), theta, cand)
                
                # 检查碰撞
                is_collision = any(
                    np.sqrt((new_pos[0] - ox)**2 + (new_pos[1] - oy)**2) < self.robot_radius + orad 
                    for ox, oy, orad in obstacles
                )
                
                if is_collision:
                    continue
                
                new_g = g + self._cost_gi(cand, theta, prev_dir, depth)
                new_h = self._heuristic_h(cand, theta, prev_dir, depth + 1)
                _, new_theta = self._project_robot(np.array([x, y]), theta, cand)
                
                heapq.heappush(open_set, (new_g + new_h, new_g, depth + 1, 
                                        (new_pos[0], new_pos[1], new_theta, cand), 
                                        path + [new_pos]))
        
        # 如果没有找到路径，返回成本最低的主要候选
        costs = [self._cost_g0(c, robot_pos, robot_theta, prev_direction) for c in primary_candidates]
        best_dir = primary_candidates[np.argmin(costs)]
        
        return best_dir
    
    def _project_robot(self, pos: np.ndarray, theta: float, direction: float) -> Tuple[np.ndarray, float]:
        """
        投影机器人位置
        
        Args:
            pos: 当前位置
            theta: 当前朝向
            direction: 目标方向
            
        Returns:
            (新位置, 新朝向)
        """
        new_theta = self._normalize_angle(direction)
        new_pos = np.array([
            pos[0] + self.ds * np.cos(new_theta),
            pos[1] + self.ds * np.sin(new_theta)
        ])
        return new_pos, new_theta
    
    def _cost_g0(self, c0: float, robot_pos: np.ndarray, robot_theta: float, prev_direction: float) -> float:
        """
        计算初始成本
        
        Args:
            c0: 候选方向
            robot_pos: 机器人位置
            robot_theta: 机器人朝向
            prev_direction: 前一个方向
            
        Returns:
            成本值
        """
        target_angle = np.arctan2(self.target[1] - robot_pos[1], self.target[0] - robot_pos[0])
        
        return (self.mu1 * abs(self._delta(c0, target_angle)) + 
                self.mu2 * abs(self._delta(c0, robot_theta)) + 
                self.mu3 * abs(self._delta(c0, prev_direction)))
    
    def _cost_gi(self, ci: float, theta_i: float, ci_minus_1: float, i: int) -> float:
        """
        计算第i步的成本
        
        Args:
            ci: 当前方向
            theta_i: 当前朝向
            ci_minus_1: 前一个方向
            i: 步数
            
        Returns:
            成本值
        """
        target_angle = np.arctan2(self.target[1], self.target[0])
        
        return (self.lambda_param**i) * (
            self.mu1_prime * abs(self._delta(ci, target_angle)) + 
            self.mu2_prime * abs(self._delta(ci, theta_i)) + 
            self.mu3_prime * abs(self._delta(ci, ci_minus_1))
        )
    
    def _heuristic_h(self, c: float, theta: float, prev_dir: float, depth: int) -> float:
        """
        计算启发式函数
        
        Args:
            c: 候选方向
            theta: 当前朝向
            prev_dir: 前一个方向
            depth: 搜索深度
            
        Returns:
            启发式值
        """
        target_angle = np.arctan2(self.target[1], self.target[0])
        h = 0
        
        for i in range(depth, depth + 3):
            h += (self.lambda_param**i) * (
                self.mu2_prime * abs(self._delta(target_angle, theta)) + 
                self.mu3_prime * abs(self._delta(target_angle, prev_dir))
            )
        
        return h
    
    def _normalize_angle(self, angle: float) -> float:
        """
        标准化角度到[-π, π]
        
        Args:
            angle: 输入角度
            
        Returns:
            标准化后的角度
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _delta(self, a1: float, a2: float) -> float:
        """
        计算角度差
        
        Args:
            a1: 角度1
            a2: 角度2
            
        Returns:
            角度差
        """
        return self._normalize_angle(a1 - a2)
    
    def get_discrete_action(self, ideal_direction: float, current_theta: float) -> Tuple[str, float]:
        """
        将连续方向转换为离散动作
        
        Args:
            ideal_direction: 理想方向
            current_theta: 当前朝向
            
        Returns:
            (动作名称, 动作值)
        """
        direction_error = self._normalize_angle(ideal_direction - current_theta)
        
        if abs(direction_error) < self.alignment_tolerance:
            return "move_forward", 0.0
        
        # 找到最接近的离散动作
        best_action, min_err = None, float('inf')
        
        for name, val in self.discrete_actions.items():
            err = abs(self._normalize_angle(ideal_direction - self._normalize_angle(current_theta + val)))
            if err < min_err:
                min_err, best_action = err, val
        
        if best_action == self.discrete_actions["turn_left_30"]:
            return "turn_left", 30.0
        elif best_action == self.discrete_actions["turn_right_30"]:
            return "turn_right", 30.0
        else:
            return "move_forward", 0.0
    
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