"""
ObjectDetector - 物体检测和分割类
封装Grounding DINO和Mobile SAM客户端，提供统一的物体检测接口
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import sys
import os

# 添加本地VLM路径
vlm_path = os.path.join(os.path.dirname(__file__), '..', 'vlm')
sys.path.insert(0, vlm_path)

try:
    from vlm.grounding_dino import GroundingDINOClient
    from vlm.sam import MobileSAMClient
    from vlm.detections import ObjectDetections
except ImportError as e:
    print(f"Warning: Could not import VLM modules: {e}")
    print("Object detection functionality will be disabled.")
    GroundingDINOClient = None
    MobileSAMClient = None
    ObjectDetections = None


class ObjectDetector:
    """物体检测和分割的核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化物体检测器
        
        Args:
            config: 包含object_detection配置的字典
        """
        self.config = config.get('object_detection', {})
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            print("Object detection is disabled in config")
            return
            
        # 检查VLFM模块是否可用
        if GroundingDINOClient is None or MobileSAMClient is None:
            print("VLFM modules not available, object detection disabled")
            self.enabled = False
            return
        
        # 初始化客户端
        try:
            grounding_dino_port = self.config.get('grounding_dino_port', 12181)
            mobile_sam_port = self.config.get('mobile_sam_port', 12184)
            
            self.grounding_dino = GroundingDINOClient(port=grounding_dino_port)
            self.mobile_sam = MobileSAMClient(port=mobile_sam_port)
            
            print(f"Object detector initialized successfully")
            print(f"Grounding DINO port: {grounding_dino_port}")
            print(f"Mobile SAM port: {mobile_sam_port}")
            
        except Exception as e:
            print(f"Failed to initialize object detector: {e}")
            print("Object detection will be disabled")
            self.enabled = False
    
    def detect_object(
        self, 
        rgb_image: np.ndarray, 
        target_object: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        检测目标物体并返回分割结果
        
        Args:
            rgb_image: RGB图像 (H, W, 3)
            depth_image: 深度图像 (H, W)
            target_object: 目标物体名称
            camera_params: 相机参数，包含fx, fy, cx, cy等
            
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: 
            - 第一个元素是物体掩码 (H, W)
            - 第二个元素是边界框 [x1, y1, x2, y2]
            如果检测失败返回None
        """
        if not self.enabled:
            return None
            
        try:
            # 1. 使用Grounding DINO检测物体
            # 使用更精确的检测策略：同时检测多个物体，然后选择目标物体
            caption = target_object + " ."
            detections = self.grounding_dino.predict(rgb_image, caption=caption)
            
            if detections.num_detections == 0:
                print(f"No {target_object} detected")
                return None
            
            # 2. 找到目标物体的检测结果
            target_found = False
            best_idx = -1
            best_confidence = 0.0
            
            for i, phrase in enumerate(detections.phrases):
                # 检查检测到的物体是否匹配目标物体
                if phrase.lower().strip() == target_object.lower().strip():
                    confidence = detections.logits[i]
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_idx = i
                        target_found = True
            
            if not target_found:
                print(f"No {target_object} found in detections. Available: {detections.phrases}")
                return None
            
            confidence = detections.logits[best_idx]
            
            # 打印调试信息
            print(f"Found {target_object} at index {best_idx} with confidence {confidence:.3f}")
            print(f"All detections: {list(zip(detections.phrases, detections.logits.tolist()))}")
            
            # 检查置信度阈值
            threshold = self.config.get('detection_threshold', 0.4)
            if confidence < threshold:
                print(f"Detection confidence {confidence:.3f} below threshold {threshold}")
                return None
            
            # 3. 获取边界框
            # 直接使用detections中的bbox，让detections.py中的annotate函数自动处理归一化
            bbox = detections.boxes[best_idx]  # [x1, y1, x2, y2]
            
            # 将Tensor转换为NumPy数组
            if hasattr(bbox, 'cpu'):
                bbox = bbox.cpu().numpy()
            
            # 检查bbox是否已经归一化，如果是则进行反归一化
            if bbox.max() <= 1:
                height, width = rgb_image.shape[:2]
                bbox_denorm = bbox * np.array([width, height, width, height])
                bbox_denorm = bbox_denorm.astype(int)
            else:
                # 如果已经是像素坐标，直接使用
                bbox_denorm = bbox.astype(int)
            
            # 4. 使用Mobile SAM进行分割
            object_mask = self.mobile_sam.segment_bbox(rgb_image, bbox_denorm.tolist())
            
            print(f"Successfully detected {target_object} with confidence {confidence:.3f}")
            print(f"Bounding box: {bbox_denorm}")
            
            return object_mask, bbox_denorm
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return None
    
    def project_to_3d(
        self, 
        object_mask: np.ndarray, 
        depth_image: np.ndarray,
        camera_params: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        将2D物体掩码投影到3D空间
        
        Args:
            object_mask: 物体掩码 (H, W)
            depth_image: 深度图像 (H, W)
            camera_params: 相机参数
            
        Returns:
            Optional[np.ndarray]: 3D点云坐标 (N, 3) 或 None
        """
        try:
            # 获取相机参数
            fx = camera_params.get('fx', 512.0)  # 焦距x
            fy = camera_params.get('fy', 512.0)  # 焦距y
            cx = camera_params.get('cx', 256.0)  # 主点x
            cy = camera_params.get('cy', 256.0)  # 主点y
            
            # 找到掩码中的有效像素
            mask_indices = np.where(object_mask > 0)
            if len(mask_indices[0]) == 0:
                return None
            
            y_coords = mask_indices[0]
            x_coords = mask_indices[1]
            
            # 获取对应的深度值
            depths = depth_image[y_coords, x_coords]
            
            # 过滤无效深度值
            valid_depth_mask = (depths > 0) & (depths < np.inf)
            if not np.any(valid_depth_mask):
                return None
            
            x_coords = x_coords[valid_depth_mask]
            y_coords = y_coords[valid_depth_mask]
            depths = depths[valid_depth_mask]
            
            # 计算3D坐标
            # Z = depth
            # X = (u - cx) * depth / fx
            # Y = (v - cy) * depth / fy
            x_3d = (x_coords - cx) * depths / fx
            y_3d = (y_coords - cy) * depths / fy
            z_3d = depths
            
            # 组合成点云
            point_cloud = np.column_stack([x_3d, y_3d, z_3d])
            
            return point_cloud
            
        except Exception as e:
            print(f"Error during 3D projection: {e}")
            return None
    
    def calculate_target_position(
        self, 
        point_cloud: np.ndarray,
        max_distance: float = 10.0
    ) -> Optional[np.ndarray]:
        """
        计算导航目标位置
        
        Args:
            point_cloud: 3D点云 (N, 3)
            max_distance: 最大检测距离
            
        Returns:
            Optional[np.ndarray]: 目标位置 [x, y, z] 或 None
        """
        try:
            if point_cloud is None or len(point_cloud) == 0:
                return None
            
            # 计算点云的中心点
            center = np.mean(point_cloud, axis=0)
            
            # 检查距离
            distance = np.linalg.norm(center)
            if distance > max_distance:
                print(f"Object too far: {distance:.2f}m > {max_distance}m")
                return None
            
            print(f"Target position calculated: {center}")
            print(f"Distance: {distance:.2f}m")
            
            return center
            
        except Exception as e:
            print(f"Error calculating target position: {e}")
            return None
    
    def detect_and_get_target_coords(
        self, 
        rgb_image: np.ndarray, 
        depth_image: np.ndarray,
        target_object: str,
        camera_params: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        完整的物体检测和坐标计算流程
        
        Args:
            rgb_image: RGB图像
            depth_image: 深度图像
            target_object: 目标物体名称
            camera_params: 相机参数
            
        Returns:
            Optional[np.ndarray]: 目标3D坐标 [x, y, z] 或 None
        """
        if not self.enabled:
            return None
        
        # 1. 物体检测和分割
        detection_result = self.detect_object(rgb_image, target_object)
        if detection_result is None:
            return None
        
        object_mask, bbox = detection_result
        
        # 2. 3D投影
        point_cloud = self.project_to_3d(object_mask, depth_image, camera_params)
        if point_cloud is None:
            return None
        
        # 3. 计算目标位置
        max_distance = self.config.get('max_detection_distance', 10.0)
        target_position = self.calculate_target_position(point_cloud, max_distance)
        
        return target_position
    
    def is_enabled(self) -> bool:
        """检查物体检测是否启用"""
        return self.enabled 