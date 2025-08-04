# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

import magnum as mn
from PIL import Image

class OccupancyMapBuilder:
    """
    根据智能体的深度感知和位姿，实时构建和可视化鸟瞰占用地图。
    该类的实现参考了 UniGoal 项目中的方法。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化地图构建器。

        Args:
            config: 包含地图参数的配置字典，例如：
                    MAP_SIZE, MAP_SCALE, VISION_RANGE, MIN_DEPTH, MAX_DEPTH,
                    CAMERA_HEIGHT, MIN_H, MAX_H, HFOV
        """
        # MAP_SIZE 以米为单位，MAP_SCALE 以米/像素为单位
        self.map_size_m = config['MAP_SIZE']  # 地图大小（米）
        self.map_resolution = config['MAP_SCALE']  # 分辨率（米/像素）
        self.map_shape = (
            int(self.map_size_m / self.map_resolution),
            int(self.map_size_m / self.map_resolution)
        )
        
        self.camera_height = config['CAMERA_HEIGHT']
        self.height_filter_range = (config['MIN_H'], config['MAX_H'])
        
        # 初始化一个空的地图数组。 128: 未知, 255: 空闲, 0: 占用
        self.grid_map = np.full(self.map_shape, 128, dtype=np.uint8)
        self.agent_map_coords = None

    def update_map(
        self, 
        depth_observation: np.ndarray, 
        agent_pose: Dict[str, np.ndarray],
        hfov: float,
    ):
        """
        使用新的传感器数据更新占用地图。

        Args:
            depth_observation: 来自模拟器的深度图 (H, W)。
            agent_pose: 包含 'position' 和 'rotation' 的字典。
            hfov: 水平视场角 (度)。
        """
        # 1. 将深度图转换为相机坐标系下的点云
        point_cloud_camera = self._depth_to_point_cloud(depth_observation, hfov)

        # 2. 将点云从相机坐标系转换到世界坐标系
        point_cloud_world = self._transform_points_to_world(
            point_cloud_camera, agent_pose['position'], agent_pose['rotation']
        )

        # 3. 将世界坐标系下的点云投影到地图上
        self._project_to_map(point_cloud_world, agent_pose['position'])

    def get_map_image(self, agent_pose: Dict[str, np.ndarray], output_size: Tuple[int, int]) -> np.ndarray:
        """
        生成用于显示的可视化地图图像，并在其上绘制智能体。

        Args:
            agent_pose: 智能体位姿，用于绘制其在地图上的位置。
            output_size: 输出图像的尺寸 (width, height)。

        Returns:
            一个 BGR 格式的 NumPy 图像数组。
        """
        vis_map = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)

        # 绘制智能体
        if self.agent_map_coords:
            cv2.circle(vis_map, self.agent_map_coords, 5, (0, 0, 255), -1) # 红色圆点

            # 绘制朝向 - 使用手动实现的四元数旋转
            forward_vec = self._quaternion_rotate_vector(agent_pose['rotation'], np.array([0, 0, -1]))
            endpoint_x = self.agent_map_coords[0] + int(forward_vec[0] * 15)
            endpoint_y = self.agent_map_coords[1] - int(forward_vec[2] * 15) # 地图Y轴与世界Z轴方向相反
            cv2.line(vis_map, self.agent_map_coords, (endpoint_x, endpoint_y), (0, 255, 0), 2)

        if output_size:
            vis_map = cv2.resize(vis_map, output_size, interpolation=cv2.INTER_NEAREST)

        return vis_map

    def _depth_to_point_cloud(self, depth: np.ndarray, hfov: float) -> np.ndarray:
        """从深度图计算相机坐标系下的点云。"""
        h, w = depth.shape
        hfov_rad = np.deg2rad(hfov)
        
        # 获取相机内参矩阵
        K = get_camera_matrix(w, h, hfov_rad)
        K_inv = np.linalg.inv(K)

        # 创建像素坐标网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w))
        pixel_coords = np.stack((x, y, ones), axis=2).reshape(-1, 3) # (H*W, 3)
        
        # 滤除无效深度值
        depth_flat = depth.flatten()
        valid_depth_mask = (depth_flat > 0.1) & (depth_flat < 5.0)

        # 计算相机坐标
        camera_coords = K_inv @ pixel_coords[valid_depth_mask].T # (3, N)
        camera_coords = camera_coords * depth_flat[valid_depth_mask] # (3, N)
        
        # 转换为 (N, 3) 格式
        return camera_coords.T

    def _transform_points_to_world(self, points: np.ndarray, agent_pos: np.ndarray, agent_rot: np.ndarray) -> np.ndarray:
        """将点云从相机坐标系转换到世界坐标系。"""
        # Habitat-sim 相机坐标系: +Y向上, +X向右, -Z向前
        # 需要旋转以匹配世界坐标系: +Y向上, +X向右, +Z向后
        # 这个旋转是固定的，绕X轴旋转180度
        cam_to_agent_rot = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        points_agent_frame = (cam_to_agent_rot @ points.T).T

        # 应用智能体的旋转和平移
        # 使用四元数到旋转矩阵的转换，避免直接构造 Magnum 四元数
        # agent_rot 格式为 [x, y, z, w]
        x, y, z, w = agent_rot
        
        # 手动计算旋转矩阵
        rot_mat = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        
        world_points = (rot_mat @ points_agent_frame.T).T
        world_points += agent_pos
        
        return world_points

    def _project_to_map(self, points_world: np.ndarray, agent_pos: np.ndarray):
        """将世界坐标系的点云投影到2D栅格地图上。"""
        # 高度过滤
        min_h, max_h = self.height_filter_range
        height_mask = (points_world[:, 1] > min_h) & (points_world[:, 1] < max_h)
        points_to_project = points_world[height_mask]

        if len(points_to_project) == 0:
            return

        # 转换到地图坐标
        map_coords = self._world_to_map_coords(points_to_project)
        
        # 标记占用栅格
        for x, y in map_coords:
            if 0 <= x < self.map_shape[1] and 0 <= y < self.map_shape[0]:
                self.grid_map[y, x] = 0  # 黑色: 占用

        # 更新智能体位置
        agent_map_coords = self._world_to_map_coords(agent_pos.reshape(1, 3))[0]
        self.agent_map_coords = (agent_map_coords[0], agent_map_coords[1])
        
        # 简单的光线追踪标记空闲区域
        self._trace_rays_to_obstacles(agent_map_coords, map_coords)

    def _trace_rays_to_obstacles(self, agent_coords: np.ndarray, obstacle_coords: np.ndarray):
        """从智能体位置到障碍物之间的光线追踪，标记空闲区域"""
        agent_x, agent_y = agent_coords
        
        for obs_x, obs_y in obstacle_coords:
            if 0 <= obs_x < self.map_shape[1] and 0 <= obs_y < self.map_shape[0]:
                # 使用 Bresenham 算法绘制从智能体到障碍物的直线
                self._draw_line(agent_x, agent_y, obs_x, obs_y)
    
    def _draw_line(self, x0: int, y0: int, x1: int, y1: int):
        """使用 Bresenham 算法在地图上绘制直线，标记空闲区域"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        
        while True:
            # 只标记未知区域为空闲，不覆盖已知的占用区域
            if (0 <= x < self.map_shape[1] and 0 <= y < self.map_shape[0] and 
                self.grid_map[y, x] == 128):  # 只标记未知区域
                self.grid_map[y, x] = 255  # 白色: 空闲
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc

    def _world_to_map_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """将世界坐标 (X, Z) 转换为地图栅格坐标 (map_x, map_y)。"""
        map_center = self.map_shape[0] // 2
        
        map_x = np.clip(
            np.floor(world_coords[:, 0] / self.map_resolution) + map_center, 0, self.map_shape[1] - 1
        ).astype(int)
        
        # 世界Z轴对应地图的-Y轴
        map_y = np.clip(
            np.floor(-world_coords[:, 2] / self.map_resolution) + map_center, 0, self.map_shape[0] - 1
        ).astype(int)
        
        return np.stack([map_x, map_y], axis=1)

    def _quaternion_rotate_vector(self, quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        使用四元数旋转向量
        
        Args:
            quat: 四元数 [x, y, z, w]
            vector: 要旋转的向量 [x, y, z]
            
        Returns:
            旋转后的向量
        """
        # 四元数分量
        qx, qy, qz, qw = quat
        
        # 向量分量
        vx, vy, vz = vector
        
        # 四元数旋转公式: v' = q * v * q^(-1)
        # 这里使用展开的公式直接计算
        
        # 计算 q * v (四元数乘法的第一部分)
        qv_w = -qx * vx - qy * vy - qz * vz
        qv_x = qw * vx + qy * vz - qz * vy
        qv_y = qw * vy + qz * vx - qx * vz
        qv_z = qw * vz + qx * vy - qy * vx
        
        # 计算 (q * v) * q^(-1) (四元数乘法的第二部分)
        # q^(-1) = [-qx, -qy, -qz, qw] (对于单位四元数)
        result_x = qv_w * (-qx) + qv_x * qw + qv_y * (-qz) - qv_z * (-qy)
        result_y = qv_w * (-qy) + qv_y * qw + qv_z * (-qx) - qv_x * (-qz)
        result_z = qv_w * (-qz) + qv_z * qw + qv_x * (-qy) - qv_y * (-qx)
        
        return np.array([result_x, result_y, result_z])

def get_camera_matrix(width: int, height: int, hfov_rad: float) -> np.ndarray:
    """计算相机内参矩阵。"""
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx # 假设像素是正方形
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
