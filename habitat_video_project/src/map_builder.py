# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import cv2
from typing import Tuple, Dict, Any, Optional
from scipy import ndimage
from numba import jit, prange
import torch.nn.functional as F

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
        self.max_chunk_size = gpu_config.get('max_chunk_size', 100000)  # 增大块大小
        self.enable_numba = config.get('enable_numba', True) if config else True  # 启用Numba加速

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
        
        # 预分配GPU内存缓存，避免频繁分配
        if self.use_gpu:
            self._gpu_cache = {
                'pixel_coords': None,
                'depth_buffer': None,
                'point_buffer': None,
                'transform_buffer': None
            }
            # 预热GPU
            torch.cuda.empty_cache()

        print(f"OccupancyMapBuilder已初始化，使用 {'GPU' if self.use_gpu else 'CPU'} 进行计算。")
        if self.use_gpu:
            print(f"  - 内存效率模式: {'启用' if self.memory_efficient else '禁用'}")
            print(f"  - 最大数据块大小: {self.max_chunk_size}")
            print(f"  - Numba加速: {'启用' if self.enable_numba else '禁用'}")

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
        优化的深度数据预处理，支持多种加速方式。
        """
        depth_float = depth.astype(np.float32)
        
        # 如果启用Numba且数据量较大，使用JIT编译的快速滤波
        if self.enable_numba and depth.size > 100000:
            try:
                return numba_fast_depth_filter(depth_float, threshold=0.3)
            except Exception as e:
                print(f"Numba深度滤波失败，回退到标准实现: {e}")
        
        # GPU路径：使用PyTorch实现的快速滤波
        if self.use_gpu and depth.size > 50000:
            depth_tensor = torch.from_numpy(depth_float).to(self.device)
            
            # 使用PyTorch的卷积实现快速中值滤波替代
            kernel = torch.ones((1, 1, 3, 3), device=self.device, dtype=torch.float32) / 9.0
            depth_4d = depth_tensor.unsqueeze(0).unsqueeze(0)
            depth_filtered = F.conv2d(depth_4d, kernel, padding=1).squeeze()
            
            # 快速异常检测：使用梯度幅度（修复张量尺寸不匹配问题）
            grad_kernel_x = torch.tensor([[[[-1, 0, 1]]]], device=self.device, dtype=torch.float32)
            grad_kernel_y = torch.tensor([[[[-1], [0], [1]]]], device=self.device, dtype=torch.float32)
            
            grad_x = F.conv2d(depth_4d, grad_kernel_x, padding=1).squeeze()
            grad_y = F.conv2d(depth_4d, grad_kernel_y, padding=1).squeeze()
            
            # 确保两个梯度张量尺寸一致
            min_h = min(grad_x.shape[0], grad_y.shape[0], depth_filtered.shape[0])
            min_w = min(grad_x.shape[1], grad_y.shape[1], depth_filtered.shape[1])
            
            grad_x = grad_x[:min_h, :min_w]
            grad_y = grad_y[:min_h, :min_w]
            depth_filtered = depth_filtered[:min_h, :min_w]
            
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            
            # 标记异常区域
            anomaly_mask = gradient_magnitude > 0.3  # 降低阈值减少过度滤波
            depth_filtered[anomaly_mask] = 0
            
            return depth_filtered.cpu().numpy()
        else:
            # CPU路径：快速简化处理
            # 使用scipy的快速滤波替代opencv
            depth_filtered = ndimage.uniform_filter(depth_float, size=3)
            
            # 简化的异常检测
            grad_x = ndimage.sobel(depth_filtered, axis=1)
            grad_y = ndimage.sobel(depth_filtered, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 简化的异常处理
            anomaly_mask = gradient_magnitude > 0.3
            depth_filtered[anomaly_mask] = 0
            
            return depth_filtered

    def _depth_to_point_cloud(self, depth: np.ndarray, hfov: float) -> np.ndarray:
        """
        高性能深度图转点云，完全向量化实现。
        """
        depth = self._preprocess_depth_data(depth)
        h, w = depth.shape
        hfov_rad = np.deg2rad(hfov)

        # --- 高性能GPU路径 ---
        if self.use_gpu and depth.size > 5000:  # 降低GPU使用阈值
            depth_tensor = to_torch(depth, device=self.device, dtype=torch.float32)

            # 缓存或重新计算逆相机内参矩阵
            if self._camera_intrinsics_inv is None or self._camera_intrinsics_inv.shape[-1] != w:
                K = get_camera_matrix(w, h, hfov_rad)
                K_inv = np.linalg.inv(K)
                self._camera_intrinsics_inv = to_torch(K_inv, device=self.device, dtype=self.dtype)

            # 重用或创建像素坐标缓存
            cache_key = f"{h}x{w}"
            if (self._gpu_cache.get('pixel_coords') is None or 
                self._gpu_cache['pixel_coords'].shape[0] != h * w):
                
                # 预计算像素坐标网格，避免每次重新计算
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(h, device=self.device, dtype=self.dtype),
                    torch.arange(w, device=self.device, dtype=self.dtype),
                    indexing='ij'
                )
                pixel_coords = torch.stack([
                    x_coords.flatten(), 
                    y_coords.flatten(), 
                    torch.ones(h * w, device=self.device, dtype=self.dtype)
                ], dim=1)
                self._gpu_cache['pixel_coords'] = pixel_coords
            
            pixel_coords = self._gpu_cache['pixel_coords']
            depth_flat = depth_tensor.flatten()

            # 向量化有效性检查
            valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
            valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
            
            if valid_indices.numel() == 0:
                return np.empty((0, 3))

            # 批量处理：直接使用向量化矩阵乘法
            valid_pixels = pixel_coords[valid_indices]
            valid_depths = depth_flat[valid_indices]
            
            # 单次矩阵乘法完成所有点的投影变换
            cam_coords = (self._camera_intrinsics_inv @ valid_pixels.T) * valid_depths.unsqueeze(0)
            
            return to_numpy(cam_coords.T)

        # --- 优化的CPU路径 ---
        else:
            # 预计算网格坐标，避免重复计算
            K = get_camera_matrix(w, h, hfov_rad)
            K_inv = np.linalg.inv(K)
            
            # 向量化创建像素坐标
            y_indices, x_indices = np.mgrid[0:h, 0:w]
            pixel_coords = np.stack([
                x_indices.flatten(), 
                y_indices.flatten(), 
                np.ones(h * w)
            ], axis=1).astype(np.float32)
            
            depth_flat = depth.flatten().astype(np.float32)
            
            # 向量化有效性检查
            valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
            if not np.any(valid_mask):
                return np.empty((0, 3))

            valid_pixels = pixel_coords[valid_mask]
            valid_depths = depth_flat[valid_mask]
            
            # 单次矩阵乘法完成投影变换
            cam_coords = (K_inv @ valid_pixels.T) * valid_depths
            return cam_coords.T

    def _transform_points_to_world(self, points: np.ndarray, agent_pos: np.ndarray, agent_rot_quat: np.ndarray) -> np.ndarray:
        """
        高性能点云坐标系转换，完全向量化实现。
        """
        if points.shape[0] == 0:
            return points

        # 预计算四元数到旋转矩阵的转换，避免重复计算
        x, y, z, w = agent_rot_quat
        
        # --- 优化的GPU加速路径 ---
        if self.use_gpu and points.shape[0] > 50:  # 降低GPU使用阈值
            points_tensor = to_torch(points, device=self.device, dtype=self.dtype)
            agent_pos_tensor = to_torch(agent_pos, device=self.device, dtype=self.dtype)
            
            # 1. 相机到智能体坐标系（预计算的旋转矩阵）
            points_agent_frame = points_tensor @ self._cam_to_agent_rot_torch.T
            
            # 2. 四元数到旋转矩阵的向量化计算
            # 避免逐个元素计算，使用向量化操作
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            xw, yw, zw = x*w, y*w, z*w
            
            rot_mat = torch.tensor([
                [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
                [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
                [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
            ], device=self.device, dtype=self.dtype)
            
            # 3. 批量变换：旋转 + 平移
            world_points = points_agent_frame @ rot_mat.T + agent_pos_tensor
            return to_numpy(world_points)

        # --- 优化的CPU路径 ---
        else:
            # 1. 相机到智能体坐标系
            points_agent_frame = points @ self._cam_to_agent_rot_np.T
            
            # 2. 四元数到旋转矩阵（向量化计算）
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            xw, yw, zw = x*w, y*w, z*w
            
            rot_mat = np.array([
                [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
                [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
                [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
            ], dtype=np.float32)

            # 3. 批量变换
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
        高性能地图投影，包含射线遮挡检测。
        """
        if points_world.shape[0] == 0:
            return

        # 更新智能体在地图上的坐标
        agent_map_coords = self._world_to_map_coords(agent_pos.reshape(1, 3))[0]
        self.agent_map_coords = (int(agent_map_coords[0]), int(agent_map_coords[1]))
        
        # 转换点云到地图坐标
        map_coords = self._world_to_map_coords(points_world)
        
        # 1. 向量化标记障碍物
        min_h, max_h = self.height_filter_range
        obstacle_mask = (points_world[:, 1] >= min_h) & (points_world[:, 1] <= max_h)
        
        if np.any(obstacle_mask):
            obstacle_coords = map_coords[obstacle_mask]
            # 使用advanced indexing批量更新
            self.grid_map[obstacle_coords[:, 1], obstacle_coords[:, 0]] = 0
        
        # 2. 高性能射线绘制（包含遮挡检测）
        if len(map_coords) > 0:
            self._vectorized_ray_casting_with_occlusion(map_coords, self.agent_map_coords)

    def _vectorized_ray_casting_with_occlusion(self, target_points: np.ndarray, agent_point: Tuple[int, int]):
        """
        高性能射线追踪实现，包含遮挡检测，支持Numba加速。
        """
        agent_x, agent_y = agent_point
        
        # 如果启用Numba且点数较多，使用JIT编译的函数
        if self.enable_numba and len(target_points) > 100:
            try:
                # 使用Numba加速的射线追踪（包含遮挡检测）
                grid_map_copy = self.grid_map.copy()  # Numba需要可写的数组
                numba_ray_casting_with_occlusion(
                    target_points.astype(np.int32), 
                    np.int32(agent_x), 
                    np.int32(agent_y), 
                    (self.map_shape[0], self.map_shape[1]), 
                    grid_map_copy
                )
                self.grid_map = grid_map_copy
                return
            except Exception as e:
                print(f"Numba加速失败，回退到标准实现: {e}")
        
        # 标准的向量化实现（包含遮挡检测）
        self._standard_ray_casting_with_occlusion(target_points, agent_point)

    def _standard_ray_casting_with_occlusion(self, target_points: np.ndarray, agent_point: Tuple[int, int]):
        """
        标准的射线追踪实现，包含遮挡检测。
        """
        agent_x, agent_y = agent_point
        
        # 按距离排序目标点，优先处理近距离点
        distances = np.sqrt((target_points[:, 0] - agent_x)**2 + (target_points[:, 1] - agent_y)**2)
        sorted_indices = np.argsort(distances)
        sorted_targets = target_points[sorted_indices]
        sorted_distances = distances[sorted_indices]
        
        # 创建临时地图记录本次射线
        temp_ray_map = np.zeros_like(self.grid_map, dtype=np.uint8)
        
        for i, (target_point, dist) in enumerate(zip(sorted_targets, sorted_distances)):
            target_x, target_y = target_point
            
            if dist == 0:
                continue
                
            # 使用Bresenham算法生成射线路径
            ray_x, ray_y = self._bresenham_line(agent_x, agent_y, target_x, target_y)
            
            # 检查射线是否被遮挡
            hit_obstacle = False
            valid_ray_points = []
            
            for rx, ry in zip(ray_x, ray_y):
                # 检查边界
                if not (0 <= rx < self.map_shape[0] and 0 <= ry < self.map_shape[1]):
                    break
                    
                # 如果遇到障碍物，停止射线
                if self.grid_map[ry, rx] == 0:  # 障碍物
                    hit_obstacle = True
                    break
                    
                valid_ray_points.append((rx, ry))
            
            # 标记有效的射线路径为空闲（只有未被遮挡的部分）
            if valid_ray_points:
                for rx, ry in valid_ray_points:
                    if self.grid_map[ry, rx] == 128:  # 只更新未知区域
                        self.grid_map[ry, rx] = 255  # 空闲
                        temp_ray_map[ry, rx] = 255

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bresenham直线算法，生成从(x0,y0)到(x1,y1)的像素坐标。
        """
        points_x = []
        points_y = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x, y = x0, y0
        
        x_inc = 1 if x0 < x1 else -1
        y_inc = 1 if y0 < y1 else -1
        
        error = dx - dy
        
        while True:
            points_x.append(x)
            points_y.append(y)
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * error
            
            if e2 > -dy:
                error -= dy
                x += x_inc
                
            if e2 < dx:
                error += dx
                y += y_inc
        
        return np.array(points_x), np.array(points_y)

    def _vectorized_ray_casting(self, target_points: np.ndarray, agent_point: Tuple[int, int]):
        """
        高性能射线追踪实现，支持Numba加速。
        """
        agent_x, agent_y = agent_point
        
        # 如果启用Numba且点数较多，使用JIT编译的函数
        if self.enable_numba and len(target_points) > 100:
            try:
                # 使用Numba加速的射线追踪
                grid_map_copy = self.grid_map.copy()  # Numba需要可写的数组
                numba_vectorized_ray_casting(
                    target_points.astype(np.int32), 
                    np.int32(agent_x), 
                    np.int32(agent_y), 
                    (self.map_shape[0], self.map_shape[1]), 
                    grid_map_copy
                )
                self.grid_map = grid_map_copy
                return
            except Exception as e:
                print(f"Numba加速失败，回退到标准实现: {e}")
        
        # 标准的向量化实现
        dx = target_points[:, 0] - agent_x
        dy = target_points[:, 1] - agent_y
        distances = np.sqrt(dx**2 + dy**2)
        
        # 过滤掉距离为0的点
        valid_mask = distances > 0
        if not np.any(valid_mask):
            return
            
        dx = dx[valid_mask]
        dy = dy[valid_mask]
        distances = distances[valid_mask]
        target_points = target_points[valid_mask]
        
        # 归一化方向向量
        dx_norm = dx / distances
        dy_norm = dy / distances
        
        # 向量化生成所有射线上的点
        all_x_coords = []
        all_y_coords = []
        
        for i, (dx_n, dy_n, dist) in enumerate(zip(dx_norm, dy_norm, distances)):
            steps = np.arange(0, int(dist) + 1)
            ray_x = agent_x + (steps * dx_n).astype(np.int32)
            ray_y = agent_y + (steps * dy_n).astype(np.int32)
            
            # 确保坐标在地图范围内
            valid_coords = ((ray_x >= 0) & (ray_x < self.map_shape[0]) & 
                           (ray_y >= 0) & (ray_y < self.map_shape[1]))
            
            if np.any(valid_coords):
                all_x_coords.append(ray_x[valid_coords])
                all_y_coords.append(ray_y[valid_coords])
        
        if all_x_coords:
            x_coords = np.concatenate(all_x_coords)
            y_coords = np.concatenate(all_y_coords)
            
            # 批量更新空闲区域
            free_mask = self.grid_map[y_coords, x_coords] == 128
            if np.any(free_mask):
                self.grid_map[y_coords[free_mask], x_coords[free_mask]] = 255

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


@jit(nopython=True, parallel=True)
def numba_ray_casting_with_occlusion(target_points, agent_x, agent_y, map_shape, grid_map):
    """
    使用Numba JIT编译的高性能射线追踪，包含遮挡检测。
    """
    map_width, map_height = map_shape
    
    for i in prange(len(target_points)):
        target_x, target_y = target_points[i]
        
        # Bresenham直线算法实现
        dx = abs(target_x - agent_x)
        dy = abs(target_y - agent_y)
        
        x, y = agent_x, agent_y
        x_inc = 1 if agent_x < target_x else -1
        y_inc = 1 if agent_y < target_y else -1
        
        error = dx - dy
        hit_obstacle = False
        
        while True:
            # 检查边界
            if not (0 <= x < map_width and 0 <= y < map_height):
                break
                
            # 检查是否遇到障碍物
            if grid_map[y, x] == 0:  # 障碍物
                hit_obstacle = True
                break
                
            # 只更新未知区域为空闲
            if grid_map[y, x] == 128:
                grid_map[y, x] = 255
            
            # 到达目标点
            if x == target_x and y == target_y:
                break
                
            # Bresenham步进
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc


@jit(nopython=True, parallel=True)
def numba_vectorized_ray_casting(target_points, agent_x, agent_y, map_shape, grid_map):
    """
    使用Numba JIT编译的高性能射线追踪。
    """
    map_width, map_height = map_shape
    
    for i in prange(len(target_points)):
        target_x, target_y = target_points[i]
        
        # Bresenham直线算法的向量化实现
        dx = abs(target_x - agent_x)
        dy = abs(target_y - agent_y)
        
        x, y = agent_x, agent_y
        x_inc = 1 if agent_x < target_x else -1
        y_inc = 1 if agent_y < target_y else -1
        
        error = dx - dy
        
        while True:
            # 检查边界并更新地图
            if 0 <= x < map_width and 0 <= y < map_height:
                if grid_map[y, x] == 128:  # 只更新未知区域
                    grid_map[y, x] = 255  # 设置为空闲
            
            if x == target_x and y == target_y:
                break
                
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc


@jit(nopython=True, parallel=True)
def numba_fast_depth_filter(depth_array, threshold=0.3):
    """
    使用Numba加速的深度数据滤波。
    """
    h, w = depth_array.shape
    filtered = depth_array.copy()
    
    # 简化的梯度计算和异常检测
    for i in prange(1, h-1):
        for j in prange(1, w-1):
            if depth_array[i, j] > 0:
                # 计算局部梯度
                grad_x = abs(depth_array[i, j+1] - depth_array[i, j-1])
                grad_y = abs(depth_array[i+1, j] - depth_array[i-1, j])
                gradient = (grad_x + grad_y) / 2.0
                
                if gradient > threshold:
                    filtered[i, j] = 0.0
    
    return filtered