"""
工具函数模块
提供项目中使用的通用工具函数
"""

import json
import numpy as np
import magnum as mn
from datetime import datetime
from typing import Dict, Tuple, Any
import os


def load_json_config(filepath: str) -> Dict[str, Any]:
    """安全地加载和解析JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误 in {filepath}: {e}")


def write_json_report(filepath: str, data: Dict[str, Any]) -> None:
    """将最终报告写入JSON文件"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=str)
    except Exception as e:
        raise RuntimeError(f"写入报告文件失败 {filepath}: {e}")


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值，用于平滑旋转动画
    
    Args:
        q0: 起始四元数 [x, y, z, w]
        q1: 结束四元数 [x, y, z, w]
        t: 插值参数 [0, 1]
    
    Returns:
        插值后的四元数
    """
    # 确保四元数已归一化
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    # 计算两个四元数的点积
    dot = np.dot(q0, q1)
    
    # 如果点积为负，取反其中一个四元数以选择更短的路径
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # 如果两个四元数非常接近，使用线性插值
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # 计算插值角度
    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    # 计算插值系数
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q0 + s1 * q1


def convert_to_magnum_quat(rotation: np.ndarray) -> mn.Quaternion:
    """将numpy数组形式的四元数转换为Magnum四元数
    
    Args:
        rotation: numpy四元数 [x, y, z, w]
    
    Returns:
        Magnum四元数对象
    """
    # Magnum四元数构造: [vector(x,y,z), scalar(w)]
    return mn.Quaternion(
        mn.Vector3(rotation[0], rotation[1], rotation[2]),
        rotation[3]
    )


def convert_to_numpy_quat(quat: mn.Quaternion) -> np.ndarray:
    """将Magnum四元数转换为numpy数组
    
    Args:
        quat: Magnum四元数对象
    
    Returns:
        numpy四元数 [x, y, z, w]
    """
    return np.array([
        quat.vector.x,
        quat.vector.y, 
        quat.vector.z,
        quat.scalar
    ], dtype=np.float32)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """从欧拉角创建四元数
    
    Args:
        roll: 滚转角 (绕X轴旋转)
        pitch: 俯仰角 (绕Y轴旋转)  
        yaw: 偏航角 (绕Z轴旋转)
    
    Returns:
        四元数 [x, y, z, w]
    """
    # 转换为弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # 计算各轴的半角
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # 计算四元数分量
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w], dtype=np.float32)


def euler_from_quaternion(quat: np.ndarray) -> Tuple[float, float, float]:
    """从四元数提取欧拉角
    
    Args:
        quat: 四元数 [x, y, z, w]
    
    Returns:
        (roll, pitch, yaw) 欧拉角（度）
    """
    x, y, z, w = quat
    
    # Roll (绕X轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (绕Y轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 使用90度如果超出范围
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (绕Z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def generate_output_paths(output_dir: str) -> Dict[str, str]:
    """生成输出文件路径
    
    Args:
        output_dir: 输出目录
    
    Returns:
        包含视频和报告路径的字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'video': os.path.join(output_dir, f"output_{timestamp}.mp4"),
        'report': os.path.join(output_dir, f"report_{timestamp}.json")
    }


def calculate_direction_vector(start_pos: np.ndarray, end_pos: np.ndarray) -> np.ndarray:
    """计算从起始点到终点的方向向量（已归一化）
    
    Args:
        start_pos: 起始位置 [x, y, z]
        end_pos: 终点位置 [x, y, z]
    
    Returns:
        归一化的方向向量 [x, y, z]
    """
    direction = end_pos - start_pos
    distance = np.linalg.norm(direction)
    
    if distance == 0:
        return np.array([0.0, 0.0, 1.0])  # 默认朝向Z轴正方向
    
    return direction / distance


def quaternion_to_direction_yaw(position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    """计算朝向目标位置所需的偏航角四元数
    
    Args:
        position: 当前位置 [x, y, z]
        target_position: 目标位置 [x, y, z]
    
    Returns:
        表示偏航角的四元数 [x, y, z, w]
    """
    # 计算XZ平面上的方向向量
    dx = target_position[0] - position[0]
    dz = target_position[2] - position[2]
    
    # 计算偏航角
    yaw = np.arctan2(dx, dz)
    
    # 转换为四元数（只绕Y轴旋转）
    return quaternion_from_euler(0, 0, np.degrees(yaw))


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的完整性和有效性
    
    Args:
        config: 配置字典
    
    Returns:
        是否有效
    """
    required_keys = [
        'video.fps', 'video.resolution.width', 'video.resolution.height',
        'agent.linear_speed', 'agent.angular_speed',
        'scene.scene_file', 'scene.robot_urdf',
        'simulation.gpu_device_id'
    ]
    
    def check_nested_key(data: Dict[str, Any], key_path: str) -> bool:
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        
        return True
    
    missing_keys = [key for key in required_keys if not check_nested_key(config, key)]
    
    if missing_keys:
        print(f"配置文件缺少必需的键: {missing_keys}")
        return False
    
    return True
