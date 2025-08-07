# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import cv2
from typing import Tuple, Optional, Dict, Any

import magnum as mn
from PIL import Image

from .utils import get_device, get_dtype, to_torch, to_numpy, use_mixed_precision

class OccupancyMapBuilder:
    """
    根据智能体的深度感知和位姿，实时构建和可视化鸟瞰占用地图。
    该类的实现参考了 UniGoal 项目中的方法，支持GPU加速计算。
    """
    def __init__(self, use_gpu: bool = True, config: Dict[str, Any] = None):
        """
        初始化地图构建器。
        注意：初始化后必须调用set_global_reference来设置坐标系。
        
        Args:
            use_gpu: 是否使用GPU加速计算
            config: 配置字典，用于GPU内存管理
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = get_device() if self.use_gpu else torch.device('cpu')
        self.dtype = torch.float32  # 为了内存效率，强制使用float32
        
        # GPU内存管理参数
        if config and 'gpu' in config:
            gpu_config = config['gpu']
            self.memory_efficient = gpu_config.get('memory_efficient', True)
            self.max_chunk_size = gpu_config.get('max_chunk_size', 50000)
        else:
            self.memory_efficient = True
            self.max_chunk_size = 50000
        
        self.camera_height = None
        self.height_filter_range = None
        
        # 地图参数（将在set_global_reference中设置）
        self.map_resolution = None
        self.map_shape = None
        self.grid_map = None
        self.agent_map_coords = None
        
        # 全局坐标系参数
        self.scene_center = None
        self.topdown_map_bounds = None
        self.topdown_spacing = None
        self.topdown_map_size = None
        
        # GPU缓存张量，避免重复分配内存
        self._camera_intrinsics = None
        self._camera_intrinsics_inv = None
        
        print(f"OccupancyMapBuilder初始化: {'GPU' if self.use_gpu else 'CPU'}加速")
        if self.use_gpu:
            print(f"内存效率模式: {self.memory_efficient}, 最大块大小: {self.max_chunk_size}")

    def set_global_reference(self, scene_center: np.ndarray, topdown_map_bounds: Dict[str, float], 
                           topdown_spacing: float, topdown_map_size: Tuple[int, int],
                           camera_height: float, height_filter_range: Tuple[float, float]):
        """
        设置与topdown view相同的全局坐标系参考
        
        Args:
            scene_center: 场景中心坐标 [x, y, z]
            topdown_map_bounds: topdown地图的世界坐标边界
            topdown_spacing: topdown地图的像素间距（米/像素）
            topdown_map_size: topdown地图尺寸 (width, height)
            camera_height: 摄像头高度
            height_filter_range: 高度过滤范围 (min_h, max_h)
        """
        self.scene_center = scene_center
        self.topdown_map_bounds = topdown_map_bounds
        self.topdown_spacing = topdown_spacing
        self.topdown_map_size = topdown_map_size
        self.camera_height = camera_height
        self.height_filter_range = height_filter_range
        
        # 使用与topdown相同的分辨率和地图尺寸
        self.map_resolution = topdown_spacing
        self.map_shape = topdown_map_size
        
        # 初始化地图数组
        self.grid_map = np.full(self.map_shape, 128, dtype=np.uint8)
        
        print(f"Occupancy map坐标系已设置: 中心={scene_center}, 分辨率={topdown_spacing:.6f}m/pixel, 尺寸={topdown_map_size}")

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
        使用与topdown view相同的agent标注逻辑和大小。

        Args:
            agent_pose: 智能体位姿，用于绘制其在地图上的位置。
            output_size: 输出图像的尺寸 (width, height)。

        Returns:
            一个 BGR 格式的 NumPy 图像数组。
        """
        vis_map = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)

        # 绘制智能体 - 使用基于分辨率的动态大小调整
        if self.agent_map_coords:
            # 计算agent标注的合适大小，考虑地图分辨率和物理尺寸
            # 目标：agent标注应该代表现实中约0.3-0.5米的尺寸（agent的肩宽）
            
            # 方法1：基于物理尺寸计算
            agent_physical_size = 0.4  # 米，agent的典型肩宽
            dot_radius_physical = int(agent_physical_size / (2 * self.map_resolution))  # 半径对应物理尺寸
            
            # 方法2：基于地图尺寸的相对比例（作为备选）
            map_min_size = min(self.map_shape)
            dot_radius_relative = max(2, int(map_min_size * 0.01))  # 地图最小边的1%
            
            # 使用两种方法的较小值，确保标注不会过大
            dot_radius = min(max(2, dot_radius_physical), dot_radius_relative, 100)  # 限制在2-8像素
            
            # 箭头长度：相对于agent大小，表示agent的朝向
            arrow_length = max(dot_radius * 2, 12)  # 至少12像素长的箭头
            
            #print(f"Debug: 地图分辨率={self.map_resolution:.3f}m/px, 物理半径={dot_radius_physical}px, "
            #      f"相对半径={dot_radius_relative}px, 最终半径={dot_radius}px")
            
            # 绘制位置点（红色圆点）
            cv2.circle(vis_map, self.agent_map_coords, dot_radius, (0, 0, 255), -1)
            
            # 绘制朝向箭头 - 参考topdown view的朝向计算
            self._draw_agent_direction_arrow(vis_map, agent_pose['rotation'], arrow_length)

        if output_size:
            vis_map = cv2.resize(vis_map, output_size, interpolation=cv2.INTER_NEAREST)

        return vis_map
    
    def _draw_agent_direction_arrow(self, vis_map: np.ndarray, rotation_quat: np.ndarray, arrow_length: int):
        """
        绘制智能体朝向箭头，完全参考topdown view的实现逻辑
        
        Args:
            vis_map: 可视化地图图像
            rotation_quat: 四元数旋转 [x, y, z, w]
            arrow_length: 箭头长度
        """
        if not self.agent_map_coords:
            return
            
        center_x, center_y = self.agent_map_coords
        
        try:
            # 使用与topdown view相同的朝向计算方法
            # 在Habitat中，-Z轴是前方，计算前向向量
            forward_vec = self._quaternion_rotate_vector(rotation_quat, np.array([0, 0, -1]))
            
            # 转换到地图坐标系：X轴向右，Z轴向下
            # 在地图上：X对应水平向右，Z对应垂直向下
            dx = forward_vec[0] * arrow_length  # X分量
            dz = forward_vec[2] * arrow_length  # Z分量（注意是Z，不是Y）
            
            end_x = center_x + int(dx)
            end_y = center_y + int(dz)  # Z轴对应地图的Y轴
            
            # 绘制主箭头线（绿色）
            cv2.line(vis_map, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
            
            # 计算箭头头部的方向
            arrow_angle = np.arctan2(dz, dx)
            arrow_head_length = arrow_length * 0.3
            arrow_head_angle = np.radians(30)
            
            # 左侧箭头线
            left_angle = arrow_angle + np.pi - arrow_head_angle
            left_x = end_x + int(np.cos(left_angle) * arrow_head_length)
            left_y = end_y + int(np.sin(left_angle) * arrow_head_length)
            cv2.line(vis_map, (end_x, end_y), (left_x, left_y), (0, 255, 255), 2)
            
            # 右侧箭头线
            right_angle = arrow_angle + np.pi + arrow_head_angle
            right_x = end_x + int(np.cos(right_angle) * arrow_head_length)
            right_y = end_y + int(np.sin(right_angle) * arrow_head_length)
            cv2.line(vis_map, (end_x, end_y), (right_x, right_y), (0, 255, 255), 2)
            
        except Exception as e:
            print(f"绘制朝向箭头失败: {e}")
            # 回退到简单的线条绘制
            forward_vec = self._quaternion_rotate_vector(rotation_quat, np.array([0, 0, -1]))
            endpoint_x = center_x + int(forward_vec[0] * arrow_length)
            endpoint_y = center_y + int(forward_vec[2] * arrow_length)
            cv2.line(vis_map, (center_x, center_y), (endpoint_x, endpoint_y), (0, 255, 0), 2)
    
    def _preprocess_depth_data(self, depth: np.ndarray) -> np.ndarray:
        """预处理深度数据，过滤异常值"""
        # 1. 中值滤波 - 去除孤立噪声点
        depth_filtered = cv2.medianBlur(depth.astype(np.float32), 5)
        
        # 2. 检测异常跳变 - 识别透视/反射
        # 计算深度梯度
        grad_x = cv2.Sobel(depth_filtered, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_filtered, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. 过滤异常梯度区域（可能是透视或反射）
        gradient_threshold = 0.5  # 米/像素
        anomaly_mask = gradient_magnitude > gradient_threshold
        
        # 4. 形态学操作，扩大异常区域
        kernel = np.ones((5,5), np.uint8)
        anomaly_mask = cv2.dilate(anomaly_mask.astype(np.uint8), kernel, iterations=1)
        
        # 5. 将异常区域设为无效
        depth_filtered[anomaly_mask > 0] = 0
        
        return depth_filtered



    def _depth_to_point_cloud(self, depth: np.ndarray, hfov: float) -> np.ndarray:
        """从深度图计算相机坐标系下的点云，使用GPU加速。"""
        if self.use_gpu and self.memory_efficient and depth.size > 10000:  # 只有足够大的深度图才使用GPU
            return self._depth_to_point_cloud_gpu(depth, hfov)
        else:
            return self._depth_to_point_cloud_cpu(depth, hfov)
    
    def _depth_to_point_cloud_gpu(self, depth: np.ndarray, hfov: float) -> np.ndarray:
        """GPU加速版本的深度图到点云转换"""
        depth = self._preprocess_depth_data(depth)
        h, w = depth.shape
        hfov_rad = np.deg2rad(hfov)
        
        # 为大深度图使用分块处理，避免GPU内存不足
        max_pixels_per_chunk = self.max_chunk_size if self.memory_efficient else 200000
        
        if h * w > max_pixels_per_chunk:
            return self._depth_to_point_cloud_chunked_gpu(depth, hfov, max_pixels_per_chunk)
        
        # 转换为torch张量（使用float32以节省内存）
        depth_tensor = torch.from_numpy(depth).to(device=self.device, dtype=torch.float32)
        
        # 获取或计算相机内参矩阵
        if self._camera_intrinsics is None or self._camera_intrinsics.shape != (3, 3):
            K = get_camera_matrix(w, h, hfov_rad)
            self._camera_intrinsics = torch.from_numpy(K).to(device=self.device, dtype=torch.float32)
            self._camera_intrinsics_inv = torch.inverse(self._camera_intrinsics)
        
        # 创建像素坐标网格（GPU上）
        x_coords = torch.arange(w, device=self.device, dtype=torch.float32)
        y_coords = torch.arange(h, device=self.device, dtype=torch.float32)
        x, y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        ones = torch.ones_like(x)
        
        # 堆叠为齐次坐标 (H*W, 3)
        pixel_coords = torch.stack((x, y, ones), dim=2).reshape(-1, 3)
        
        # 滤除无效深度值
        depth_flat = depth_tensor.flatten()
        valid_depth_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
        
        if not valid_depth_mask.any():
            return np.array([]).reshape(0, 3)
        
        # 计算相机坐标
        valid_pixels = pixel_coords[valid_depth_mask]  # (N, 3)
        valid_depths = depth_flat[valid_depth_mask]    # (N,)
        
        # 矩阵乘法: K^(-1) * pixels^T
        camera_coords = torch.matmul(self._camera_intrinsics_inv, valid_pixels.T)  # (3, N)
        camera_coords = camera_coords * valid_depths.unsqueeze(0)  # 广播乘法
        
        # 转换为 (N, 3) 格式并返回numpy
        return to_numpy(camera_coords.T)
    
    def _depth_to_point_cloud_chunked_gpu(self, depth: np.ndarray, hfov: float, max_pixels_per_chunk: int) -> np.ndarray:
        """分块处理大深度图的GPU版本"""
        h, w = depth.shape
        hfov_rad = np.deg2rad(hfov)
        
        # 获取或计算相机内参矩阵
        if self._camera_intrinsics is None:
            K = get_camera_matrix(w, h, hfov_rad)
            self._camera_intrinsics = torch.from_numpy(K).to(device=self.device, dtype=torch.float32)
            self._camera_intrinsics_inv = torch.inverse(self._camera_intrinsics)
        
        # 将深度图分成块
        total_pixels = h * w
        num_chunks = (total_pixels + max_pixels_per_chunk - 1) // max_pixels_per_chunk
        
        results = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_pixels_per_chunk
            end_idx = min(start_idx + max_pixels_per_chunk, total_pixels)
            
            # 计算当前块的行列范围
            start_row = start_idx // w
            end_row = (end_idx - 1) // w + 1
            
            if start_row >= h:
                break
                
            # 提取当前块
            chunk_depth = depth[start_row:end_row, :]
            chunk_h, chunk_w = chunk_depth.shape
            
            # 转换为torch张量
            depth_tensor = torch.from_numpy(chunk_depth).to(device=self.device, dtype=torch.float32)
            
            # 创建块内像素坐标（调整偏移量）
            x_coords = torch.arange(chunk_w, device=self.device, dtype=torch.float32)
            y_coords = torch.arange(start_row, start_row + chunk_h, device=self.device, dtype=torch.float32)
            x, y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            ones = torch.ones_like(x)
            
            pixel_coords = torch.stack((x, y, ones), dim=2).reshape(-1, 3)
            
            # 滤除无效深度值
            depth_flat = depth_tensor.flatten()
            valid_depth_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
            
            if valid_depth_mask.any():
                valid_pixels = pixel_coords[valid_depth_mask]
                valid_depths = depth_flat[valid_depth_mask]
                
                # 计算相机坐标
                camera_coords = torch.matmul(self._camera_intrinsics_inv, valid_pixels.T)
                camera_coords = camera_coords * valid_depths.unsqueeze(0)
                
                # 添加到结果列表
                results.append(to_numpy(camera_coords.T))
            
            # 清理中间张量以释放GPU内存
            del depth_tensor, pixel_coords, depth_flat
            torch.cuda.empty_cache()
        
        # 合并所有块的结果
        if results:
            return np.vstack(results)
        else:
            return np.array([]).reshape(0, 3)
    
    def _depth_to_point_cloud_cpu(self, depth: np.ndarray, hfov: float) -> np.ndarray:
        """CPU版本的深度图到点云转换（保持原有逻辑）"""
        depth = self._preprocess_depth_data(depth)
        h, w = depth.shape
        hfov_rad = np.deg2rad(hfov)
        
        # 获取相机内参矩阵
        K = get_camera_matrix(w, h, hfov_rad)
        K_inv = np.linalg.inv(K)

        # 创建像素坐标网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w))
        pixel_coords = np.stack((x, y, ones), axis=2).reshape(-1, 3) # (H*W, 3)
        
        # 滤除无效深度值 - 放宽范围以获取更多可见信息
        depth_flat = depth.flatten()
        valid_depth_mask = (depth_flat > 0.01) & (depth_flat < 10.0)  # 扩大有效深度范围

        # 计算相机坐标
        camera_coords = K_inv @ pixel_coords[valid_depth_mask].T # (3, N)
        camera_coords = camera_coords * depth_flat[valid_depth_mask] # (3, N)
        
        # 转换为 (N, 3) 格式
        return camera_coords.T

    def _transform_points_to_world(self, points: np.ndarray, agent_pos: np.ndarray, agent_rot: np.ndarray) -> np.ndarray:
        """将点云从相机坐标系转换到世界坐标系，使用GPU加速。"""
        if self.use_gpu and len(points) > 100:  # 只有点云足够大才使用GPU
            return self._transform_points_to_world_gpu(points, agent_pos, agent_rot)
        else:
            return self._transform_points_to_world_cpu(points, agent_pos, agent_rot)
    
    def _transform_points_to_world_gpu(self, points: np.ndarray, agent_pos: np.ndarray, agent_rot: np.ndarray) -> np.ndarray:
        """GPU加速版本的坐标变换"""
        if len(points) == 0:
            return points
        
        # 转换为torch张量
        points_tensor = to_torch(points)
        agent_pos_tensor = to_torch(agent_pos)
        agent_rot_tensor = to_torch(agent_rot)
        
        # 相机到智能体坐标系的固定旋转矩阵
        cam_to_agent_rot = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], device=self.device, dtype=self.dtype)
        
        # 批量矩阵乘法
        points_agent_frame = torch.matmul(points_tensor, cam_to_agent_rot.T)
        
        # 四元数到旋转矩阵的转换（GPU上）
        x, y, z, w = agent_rot_tensor[0], agent_rot_tensor[1], agent_rot_tensor[2], agent_rot_tensor[3]
        
        # 手动计算旋转矩阵（向量化）
        rot_mat = torch.zeros(3, 3, device=self.device, dtype=self.dtype)
        rot_mat[0, 0] = 1 - 2*(y*y + z*z)
        rot_mat[0, 1] = 2*(x*y - z*w)
        rot_mat[0, 2] = 2*(x*z + y*w)
        rot_mat[1, 0] = 2*(x*y + z*w)
        rot_mat[1, 1] = 1 - 2*(x*x + z*z)
        rot_mat[1, 2] = 2*(y*z - x*w)
        rot_mat[2, 0] = 2*(x*z - y*w)
        rot_mat[2, 1] = 2*(y*z + x*w)
        rot_mat[2, 2] = 1 - 2*(x*x + y*y)
        
        # 批量旋转和平移
        world_points = torch.matmul(points_agent_frame, rot_mat.T) + agent_pos_tensor.unsqueeze(0)
        
        return to_numpy(world_points)
    
    def _transform_points_to_world_cpu(self, points: np.ndarray, agent_pos: np.ndarray, agent_rot: np.ndarray) -> np.ndarray:
        """CPU版本的坐标变换（保持原有逻辑）"""
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
        """OpenCV划线 + 障碍物放置 + 向量化后处理移除穿越射线"""
        if len(points_world) == 0:
            return

        # 更新智能体位置 - 使用统一的坐标转换
        agent_map_coords = self._world_to_map_coords(agent_pos.reshape(1, 3))[0]
        self.agent_map_coords = (agent_map_coords[0], agent_map_coords[1])
        
        # 坐标转换和边界检查
        map_coords = self._world_to_map_coords(points_world)
        valid_mask = ((map_coords[:, 0] >= 0) & (map_coords[:, 0] < self.map_shape[1]) &
                    (map_coords[:, 1] >= 0) & (map_coords[:, 1] < self.map_shape[0]))
        
        valid_map_coords = map_coords[valid_mask]
        valid_points_world = points_world[valid_mask]
        
        if len(valid_map_coords) == 0:
            return
        
        # === Step 1: 先用OpenCV批量划线 ===
        temp_map = np.zeros_like(self.grid_map)
        agent_pt = (agent_map_coords[0], agent_map_coords[1])
        
        for x, y in valid_map_coords:
            cv2.line(temp_map, agent_pt, (x, y), 255, 1)
        
        # === Step 2: 放置障碍物 ===
        min_h, max_h = self.height_filter_range
        heights = valid_points_world[:, 1]
        obstacle_mask = (heights >= min_h) & (heights <= max_h)
        
        # 累积障碍物点（解决覆盖问题）
        occupancy_accum = np.zeros(self.map_shape, dtype=np.float32)
        for i, (x, y) in enumerate(valid_map_coords):
            if obstacle_mask[i]:
                occupancy_accum[y, x] += 1.0
        
        # 标记障碍物
        occupied_mask = occupancy_accum >= 1.0
        self.grid_map[occupied_mask] = 0
        
        # === Step 3: 向量化后处理 - 移除穿越障碍物的射线 ===
        self._vectorized_remove_blocked_rays(temp_map, agent_pt)
        
        # === Step 4: 更新空闲区域 ===
        free_from_rays = (temp_map == 255) & (self.grid_map == 128)
        self.grid_map[free_from_rays] = 255

    def _vectorized_remove_blocked_rays(self, temp_map: np.ndarray, agent_pt: tuple):
        """向量化移除被障碍物阻挡的射线段"""
        agent_x, agent_y = agent_pt
        
        # === 向量化处理所有射线点 ===
        
        # 1. 获取所有射线点的坐标
        ray_coords = np.where(temp_map == 255)
        ray_y, ray_x = ray_coords[0], ray_coords[1]
        
        if len(ray_x) == 0:
            return
        
        # 2. 向量化计算所有点的距离和角度
        dx = ray_x - agent_x
        dy = ray_y - agent_y
        distances = np.sqrt(dx*dx + dy*dy)
        angles = np.arctan2(dy, dx)
        
        # 3. 向量化找出所有障碍物点
        obstacle_coords = np.where(self.grid_map == 0)
        if len(obstacle_coords[0]) == 0:
            return  # 没有障碍物，不需要处理
        
        obs_y, obs_x = obstacle_coords[0], obstacle_coords[1]
        obs_dx = obs_x - agent_x
        obs_dy = obs_y - agent_y
        obs_distances = np.sqrt(obs_dx*obs_dx + obs_dy*obs_dy)
        obs_angles = np.arctan2(obs_dy, obs_dx)
        
        # 4. 向量化处理：对每个角度区间，找到最近的障碍物距离
        angle_resolution = np.radians(3)  # 3度角度精度
        quantized_angles = np.round(angles / angle_resolution) * angle_resolution
        obs_quantized_angles = np.round(obs_angles / angle_resolution) * angle_resolution
        
        # 获取所有唯一角度
        unique_angles = np.unique(quantized_angles)
        
        # 5. 向量化标记需要清除的射线点
        points_to_clear = np.zeros(len(ray_x), dtype=bool)
        
        for angle in unique_angles:
            # 找到这个角度的所有射线点
            ray_angle_mask = np.abs(quantized_angles - angle) < angle_resolution/2
            if not np.any(ray_angle_mask):
                continue
                
            # 找到这个角度的所有障碍物点
            obs_angle_mask = np.abs(obs_quantized_angles - angle) < angle_resolution/2
            
            if np.any(obs_angle_mask):
                # 找到这个角度最近的障碍物距离
                min_obstacle_distance = np.min(obs_distances[obs_angle_mask])
                
                # 标记这个角度上距离大于最近障碍物的射线点
                ray_indices = np.where(ray_angle_mask)[0]
                blocked_mask = distances[ray_indices] > min_obstacle_distance
                points_to_clear[ray_indices[blocked_mask]] = True
        
        # 6. 批量清除被阻挡的射线点
        if np.any(points_to_clear):
            blocked_y = ray_y[points_to_clear]
            blocked_x = ray_x[points_to_clear]
            temp_map[blocked_y, blocked_x] = 0

    def _world_to_map_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """将世界坐标转换为地图栅格坐标，使用GPU加速。"""
        if self.use_gpu and len(world_coords) > 1000:  # 大量点云时使用GPU
            return self._world_to_map_coords_gpu(world_coords)
        else:
            return self._world_to_map_coords_cpu(world_coords)
    
    def _world_to_map_coords_gpu(self, world_coords: np.ndarray) -> np.ndarray:
        """GPU加速版本的世界坐标到地图坐标转换"""
        world_coords_tensor = to_torch(world_coords)
        
        # 获取topdown地图的世界坐标边界
        tl_x = self.topdown_map_bounds['top_left'][0]
        tl_z = self.topdown_map_bounds['top_left'][1]
        
        # 向量化坐标转换
        map_x = (world_coords_tensor[:, 0] - tl_x) / self.map_resolution
        map_y = (world_coords_tensor[:, 2] - tl_z) / self.map_resolution  # Z对应地图Y
        
        # 裁剪到有效范围
        map_x = torch.clamp(map_x, 0, self.map_shape[1] - 1)
        map_y = torch.clamp(map_y, 0, self.map_shape[0] - 1)
        
        # 转换为整数并返回numpy
        result = torch.stack([map_x, map_y], dim=1).int()
        return to_numpy(result)
    
    def _world_to_map_coords_cpu(self, world_coords: np.ndarray) -> np.ndarray:
        """CPU版本的世界坐标到地图坐标转换（保持原有逻辑）"""
        # 使用与topdown view完全相同的坐标转换
        # 基于topdown.py中的calculate_metadata和world_to_pixel逻辑
        
        # 获取topdown地图的世界坐标边界
        tl_x = self.topdown_map_bounds['top_left'][0]
        tl_z = self.topdown_map_bounds['top_left'][1] 
        
        # 将世界坐标转换为像素坐标
        map_x = (world_coords[:, 0] - tl_x) / self.map_resolution
        map_y = (world_coords[:, 2] - tl_z) / self.map_resolution  # Z对应地图Y
        
        # 裁剪到有效范围
        map_x = np.clip(map_x, 0, self.map_shape[1] - 1).astype(int)
        map_y = np.clip(map_y, 0, self.map_shape[0] - 1).astype(int)
        
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
