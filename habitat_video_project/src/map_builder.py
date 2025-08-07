# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import cv2
from typing import Tuple, Dict, Any

# 假设这些工具函数在其他地方定义
from .utils import get_device, to_torch, to_numpy

class OccupancyMapBuilder:
    """
    根据智能体的深度感知和位姿，实时构建和可视化鸟瞰占用地图。
    该类实现了统一的API，可根据配置自动选择CPU或GPU进行加速计算。
    """
    def __init__(self, use_gpu: bool = True, config: Dict[str, Any] = None):
        """
        初始化地图构建器。
        注意：初始化后必须调用 set_global_reference 来设置坐标系。

        Args:
            use_gpu: 是否启用GPU加速。
            config: 配置字典，主要用于GPU内存管理。
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = get_device() if self.use_gpu else torch.device('cpu')
        self.dtype = torch.float32  # GPU计算统一使用float32以提高效率

        # 从配置中读取GPU内存管理参数
        gpu_config = config.get('gpu', {}) if config else {}
        self.memory_efficient = gpu_config.get('memory_efficient', True)
        self.max_chunk_size = gpu_config.get('max_chunk_size', 50000)

        # 缓存的相机和坐标系参数
        self.camera_height: Optional[float] = None
        self.height_filter_range: Optional[Tuple[float, float]] = None
        self.map_resolution: Optional[float] = None
        self.map_shape: Optional[Tuple[int, int]] = None
        self.scene_center: Optional[np.ndarray] = None
        self.topdown_map_bounds: Optional[Dict[str, float]] = None

        # 占用地图和智能体状态
        self.grid_map: Optional[np.ndarray] = None
        self.agent_map_coords: Optional[Tuple[int, int]] = None

        # 预计算的变换矩阵，避免重复创建
        # 相机坐标系到智能体坐标系的固定旋转（绕X轴旋转180度）
        _cam_to_agent_rot_np = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32)
        self._cam_to_agent_rot_torch = torch.from_numpy(_cam_to_agent_rot_np).to(self.device)
        self._cam_to_agent_rot_np = _cam_to_agent_rot_np

        # 缓存的相机内参，避免重复计算
        self._camera_intrinsics_inv: Optional[torch.Tensor] = None

        print(f"OccupancyMapBuilder已初始化，使用 {'GPU' if self.use_gpu else 'CPU'} 进行计算。")
        if self.use_gpu:
            print(f"  - 内存效率模式: {'启用' if self.memory_efficient else '禁用'}")
            print(f"  - 最大数据块大小: {self.max_chunk_size}")

    def set_global_reference(self, scene_center: np.ndarray, topdown_map_bounds: Dict[str, float],
                           topdown_spacing: float, topdown_map_size: Tuple[int, int],
                           camera_height: float, height_filter_range: Tuple[float, float]):
        """
        设置全局坐标系参考，通常与一个顶视图(TopDownView)保持一致。

        Args:
            scene_center: 场景中心的世界坐标 [x, y, z]。
            topdown_map_bounds: 顶视图的世界坐标边界。
            topdown_spacing: 地图分辨率 (米/像素)。
            topdown_map_size: 地图尺寸 (宽度, 高度)。
            camera_height: 相机物理高度。
            height_filter_range: 有效障碍物的高度范围 (min_h, max_h)。
        """
        self.scene_center = scene_center
        self.topdown_map_bounds = topdown_map_bounds
        self.map_resolution = topdown_spacing
        self.map_shape = topdown_map_size
        self.camera_height = camera_height
        self.height_filter_range = height_filter_range

        # 初始化占用地图 (128: 未知, 0: 占据, 255: 空闲)
        self.grid_map = np.full(self.map_shape, 128, dtype=np.uint8)
        self._camera_intrinsics_inv = None # 清除相机内参缓存，因为地图/图像尺寸可能已改变

        print(f"占用地图坐标系已设置：分辨率={self.map_resolution:.4f} m/pixel, 尺寸={self.map_shape}")

    def update_map(self, depth_observation: np.ndarray, agent_pose: Dict[str, np.ndarray], hfov: float):
        """
        使用新的传感器数据更新占用地图。

        Args:
            depth_observation: 深度图像 (H, W)。
            agent_pose: 智能体位姿字典，包含 'position' 和 'rotation'。
            hfov: 水平视场角 (度)。
        """
        # 1. 深度图 -> 相机坐标系点云
        point_cloud_camera = self._depth_to_point_cloud(depth_observation, hfov)

        # 2. 相机坐标系点云 -> 世界坐标系点云
        point_cloud_world = self._transform_points_to_world(
            point_cloud_camera, agent_pose['position'], agent_pose['rotation']
        )

        # 3. 世界坐标系点云 -> 更新地图
        self._project_to_map(point_cloud_world, agent_pose['position'])

    def get_map_image(self, agent_pose: Dict[str, np.ndarray], output_size: Tuple[int, int]) -> np.ndarray:
        """
        生成用于显示的可视化地图图像，并在其上绘制智能体。

        Args:
            agent_pose: 智能体位姿，用于绘制其在地图上的位置和朝向。
            output_size: 输出图像的目标尺寸 (宽度, 高度)。

        Returns:
            一个 BGR 格式的 NumPy 图像数组。
        """
        vis_map = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)

        # 在地图上绘制智能体的位置和朝向
        if self.agent_map_coords:
            # 根据地图分辨率动态计算智能体标记的大小
            agent_physical_size = 0.4  # 智能体物理尺寸（米）
            dot_radius = max(2, int(agent_physical_size / (2 * self.map_resolution)))
            arrow_length = dot_radius * 3

            # 绘制位置（红色圆点）
            cv2.circle(vis_map, self.agent_map_coords, dot_radius, (0, 0, 255), -1)
            # 绘制朝向（绿色箭头）
            self._draw_agent_direction_arrow(vis_map, agent_pose['rotation'], arrow_length)

        if output_size and output_size != self.map_shape:
            vis_map = cv2.resize(vis_map, output_size, interpolation=cv2.INTER_NEAREST)

        return vis_map

    def _preprocess_depth_data(self, depth: np.ndarray) -> np.ndarray:
        """
        预处理深度数据，过滤由透明或反光表面引起的噪声。
        """
        depth_float = depth.astype(np.float32)
        # 使用中值滤波去除椒盐噪声
        depth_filtered = cv2.medianBlur(depth_float, 5)

        # 计算深度梯度以检测异常的深度跳变
        grad_x = cv2.Sobel(depth_filtered, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_filtered, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 标记梯度过大的区域为异常（可能是伪影）
        anomaly_mask = gradient_magnitude > 0.5  # 梯度阈值
        kernel = np.ones((5, 5), np.uint8)
        # 扩大异常区域以完全覆盖噪声
        anomaly_mask = cv2.dilate(anomaly_mask.astype(np.uint8), kernel, iterations=1) > 0

        # 将异常区域的深度值置零，使其在后续处理中被忽略
        depth_filtered[anomaly_mask] = 0
        return depth_filtered

    def _depth_to_point_cloud(self, depth: np.ndarray, hfov: float) -> np.ndarray:
        """
        将深度图转换为相机坐标系下的点云。
        自动选择CPU或GPU，并为大型图像启用分块处理以节省GPU内存。
        """
        depth = self._preprocess_depth_data(depth)
        h, w = depth.shape
        hfov_rad = np.deg2rad(hfov)

        # --- GPU 加速路径 ---
        if self.use_gpu and depth.size > 10000:
            depth_tensor = to_torch(depth, device=self.device, dtype=torch.float32)

            # 缓存或重新计算逆相机内参矩阵
            if self._camera_intrinsics_inv is None:
                K = get_camera_matrix(w, h, hfov_rad)
                K_inv = np.linalg.inv(K)
                self._camera_intrinsics_inv = to_torch(K_inv, device=self.device, dtype=self.dtype)

            # 如果图像过大，则分块处理以避免GPU内存溢出
            use_chunking = self.memory_efficient and h * w > self.max_chunk_size
            if use_chunking:
                num_chunks = (h * w + self.max_chunk_size - 1) // self.max_chunk_size
                ys, xs = torch.meshgrid(
                    torch.arange(h, device=self.device, dtype=self.dtype),
                    torch.arange(w, device=self.device, dtype=self.dtype),
                    indexing='ij'
                )
                pixel_coords = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).view(-1, 3)
                chunks = []
                for i in range(num_chunks):
                    start = i * self.max_chunk_size
                    end = start + self.max_chunk_size
                    chunk_pixels = pixel_coords[start:end]
                    chunk_depths_flat = depth_tensor.view(-1)[start:end]
                    
                    valid_mask = (chunk_depths_flat > 0.01) & (chunk_depths_flat < 10.0)
                    if not valid_mask.any(): continue

                    cam_coords = self._camera_intrinsics_inv @ chunk_pixels[valid_mask].T
                    cam_coords *= chunk_depths_flat[valid_mask]
                    chunks.append(cam_coords.T)
                
                return to_numpy(torch.cat(chunks, dim=0)) if chunks else np.empty((0, 3))

            # --- 非分块GPU处理 ---
            ys, xs = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=self.dtype),
                torch.arange(w, device=self.device, dtype=self.dtype),
                indexing='ij'
            )
            pixel_coords = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).view(-1, 3)
            depth_flat = depth_tensor.view(-1)

            valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
            if not valid_mask.any(): return np.empty((0, 3))

            cam_coords = self._camera_intrinsics_inv @ pixel_coords[valid_mask].T
            cam_coords *= depth_flat[valid_mask]
            return to_numpy(cam_coords.T)

        # --- CPU 路径 ---
        else:
            K = get_camera_matrix(w, h, hfov_rad)
            K_inv = np.linalg.inv(K)
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            pixel_coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
            depth_flat = depth.flatten()

            valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
            if not valid_mask.any(): return np.empty((0, 3))

            cam_coords = K_inv @ pixel_coords[valid_mask].T
            cam_coords *= depth_flat[valid_mask]
            return cam_coords.T

    def _transform_points_to_world(self, points: np.ndarray, agent_pos: np.ndarray, agent_rot_quat: np.ndarray) -> np.ndarray:
        """
        将点云从相机坐标系转换到世界坐标系。自动选择CPU或GPU。
        """
        if points.shape[0] == 0:
            return points

        # --- GPU 加速路径 ---
        if self.use_gpu and points.shape[0] > 100:
            points_tensor = to_torch(points, device=self.device, dtype=self.dtype)
            agent_pos_tensor = to_torch(agent_pos, device=self.device, dtype=self.dtype)
            
            # 1. 从相机坐标系转换到智能体坐标系
            points_agent_frame = points_tensor @ self._cam_to_agent_rot_torch.T
            
            # 2. 从智能体坐标系转换到世界坐标系
            # 四元数 -> 旋转矩阵 (PyTorch实现)
            x, y, z, w = agent_rot_quat
            rot_mat = torch.tensor([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], device=self.device, dtype=self.dtype)
            
            world_points = points_agent_frame @ rot_mat.T + agent_pos_tensor
            return to_numpy(world_points)

        # --- CPU 路径 ---
        else:
            # 1. 从相机坐标系转换到智能体坐标系
            points_agent_frame = points @ self._cam_to_agent_rot_np.T
            
            # 2. 从智能体坐标系转换到世界坐标系
            # 四元数 -> 旋转矩阵 (Numpy实现)
            x, y, z, w = agent_rot_quat
            rot_mat = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)

            world_points = points_agent_frame @ rot_mat.T + agent_pos
            return world_points

    def _world_to_map_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """
        将世界坐标批量转换为地图栅格坐标。自动选择CPU或GPU。
        """
        if world_coords.shape[0] == 0:
            return np.empty((0, 2), dtype=int)
            
        tl_x = self.topdown_map_bounds['top_left'][0]
        tl_z = self.topdown_map_bounds['top_left'][1]
        map_w, map_h = self.map_shape

        # --- GPU 加速路径 ---
        if self.use_gpu and world_coords.shape[0] > 1000:
            world_coords_tensor = to_torch(world_coords, device=self.device, dtype=self.dtype)
            map_x = (world_coords_tensor[:, 0] - tl_x) / self.map_resolution
            map_y = (world_coords_tensor[:, 2] - tl_z) / self.map_resolution # 世界Z轴对应地图Y轴
            
            map_x = torch.clamp(map_x, 0, map_w - 1)
            map_y = torch.clamp(map_y, 0, map_h - 1)
            
            return to_numpy(torch.stack([map_x, map_y], dim=1).int())

        # --- CPU 路径 ---
        else:
            map_x = (world_coords[:, 0] - tl_x) / self.map_resolution
            map_y = (world_coords[:, 2] - tl_z) / self.map_resolution
            
            map_x = np.clip(map_x, 0, map_w - 1).astype(int)
            map_y = np.clip(map_y, 0, map_h - 1).astype(int)

            return np.stack([map_x, map_y], axis=1)

    def _project_to_map(self, points_world: np.ndarray, agent_pos: np.ndarray):
        """
        将世界坐标系中的点云投影到2D占用地图上。
        采用三步法更新：1. 标记障碍物；2. 标记空闲区域；3. 移除被遮挡的射线。
        """
        if points_world.shape[0] == 0:
            return

        # 更新智能体在地图上的坐标
        agent_map_coords = self._world_to_map_coords(agent_pos.reshape(1, 3))[0]
        self.agent_map_coords = (agent_map_coords[0], agent_map_coords[1])
        
        # 转换点云到地图坐标并过滤无效点
        map_coords = self._world_to_map_coords(points_world)
        
        # 1. 标记障碍物
        min_h, max_h = self.height_filter_range
        obstacle_mask = (points_world[:, 1] >= min_h) & (points_world[:, 1] <= max_h)
        obstacle_coords = map_coords[obstacle_mask]
        if obstacle_coords.shape[0] > 0:
            # 使用np.unique避免重复写入，提高效率
            unique_obstacle_coords = np.unique(obstacle_coords, axis=0)
            self.grid_map[unique_obstacle_coords[:, 1], unique_obstacle_coords[:, 0]] = 0 # 0: 占据

        # 2. 标记空闲区域（使用Bresenham's line的思想，通过cv2.line实现）
        temp_free_map = np.zeros_like(self.grid_map)
        agent_pt = self.agent_map_coords
        
        # 批量绘制从智能体到所有可见点的射线
        for x, y in map_coords:
            cv2.line(temp_free_map, agent_pt, (x, y), 255, 1)
        
        # 3. 移除穿越障碍物的射线部分，确保视线不穿墙
        self._vectorized_remove_blocked_rays(temp_free_map, agent_pt)
        
        # 4. 将未被遮挡且当前未知的区域更新为空闲
        free_mask = (temp_free_map == 255) & (self.grid_map == 128)
        self.grid_map[free_mask] = 255 # 255: 空闲

    def _vectorized_remove_blocked_rays(self, temp_map: np.ndarray, agent_pt: tuple):
        """
        使用向量化操作，高效地移除被障碍物遮挡的射线部分。
        原理：对每个方向（角度），找到最近的障碍物，并擦除该方向上比此障碍物更远的射线。
        """
        agent_x, agent_y = agent_pt
        
        # 1. 获取所有射线点和障碍物点的坐标
        ray_y, ray_x = np.where(temp_map == 255)
        obs_y, obs_x = np.where(self.grid_map == 0)

        if ray_y.size == 0 or obs_y.size == 0:
            return # 没有射线或没有障碍物，无需处理

        # 2. 向量化计算所有点相对于智能体的距离和角度
        ray_dx, ray_dy = ray_x - agent_x, ray_y - agent_y
        ray_dist = np.sqrt(ray_dx**2 + ray_dy**2)
        ray_angles = np.arctan2(ray_dy, ray_dx)

        obs_dx, obs_dy = obs_x - agent_x, obs_y - agent_y
        obs_dist = np.sqrt(obs_dx**2 + obs_dy**2)
        obs_angles = np.arctan2(obs_dy, obs_dx)

        # 3. 对每个角度，找到最近的障碍物距离
        # 使用排序和查找来替代显式循环，以提高效率
        # 角度相近的障碍物可以遮挡同一方向的射线
        angle_resolution = np.deg2rad(1.0) # 1度的角度分辨率
        
        # 对障碍物按角度排序
        sort_indices = np.argsort(obs_angles)
        sorted_obs_angles = obs_angles[sort_indices]
        sorted_obs_dist = obs_dist[sort_indices]

        # 4. 向量化查找每个射线点是否被遮挡
        # `searchsorted` 找到每个射线角度在排序后的障碍物角度数组中的插入位置
        # 这使我们能快速找到每个射线方向上“附近”的障碍物
        indices = np.searchsorted(sorted_obs_angles, ray_angles)
        
        blocked = np.zeros_like(ray_dist, dtype=bool)
        # 检查前后两个角度的障碍物，以处理角度量化带来的边界问题
        for offset in [-2, -1, 0, 1, 2]:
            check_indices = np.clip(indices + offset, 0, len(sorted_obs_angles) - 1)
            angle_diff = np.abs(ray_angles - sorted_obs_angles[check_indices])
            
            # 如果角度足够近且射线比障碍物远，则认为被遮挡
            is_close_angle = angle_diff < angle_resolution
            is_further = ray_dist > sorted_obs_dist[check_indices]
            blocked |= (is_close_angle & is_further)
        
        # 5. 批量清除被遮挡的射线点
        if np.any(blocked):
            temp_map[ray_y[blocked], ray_x[blocked]] = 0

    def _draw_agent_direction_arrow(self, vis_map: np.ndarray, rotation_quat: np.ndarray, arrow_length: int):
        """
        在地图上绘制表示智能体朝向的箭头。

        Args:
            vis_map: 用于绘制的地图图像。
            rotation_quat: 智能体的四元数旋转 [x, y, z, w]。
            arrow_length: 箭头的像素长度。
        """
        center_x, center_y = self.agent_map_coords
        
        try:
            # 使用旋转矩阵计算前向向量，更标准且不易出错
            x, y, z, w = rotation_quat
            rot_mat = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ])
            # Habitat中的前向是-Z轴
            forward_vec = rot_mat @ np.array([0, 0, -1])

            # 将前向向量投影到地图平面（X-Z平面）
            end_x = center_x + int(forward_vec[0] * arrow_length)
            end_y = center_y + int(forward_vec[2] * arrow_length) # 世界Z对应地图Y

            # 绘制箭头主干（绿色）
            cv2.arrowedLine(vis_map, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
        except Exception as e:
            # 绘图失败不应中断程序
            print(f"绘制智能体朝向箭头时发生错误: {e}")


def get_camera_matrix(width: int, height: int, hfov_rad: float) -> np.ndarray:
    """
    根据图像尺寸和水平视场角计算相机内参矩阵K。
    """
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    # 假设像素是正方形的，即 fx = fy
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)