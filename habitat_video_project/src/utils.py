"""
工具函数模块
提供项目中使用的通用工具函数，支持GPU加速
"""

import json
import numpy as np
import torch
import magnum as mn
from datetime import datetime
from typing import Dict, Tuple, Any, Union
import os

# GPU设置全局变量
_device = None
_dtype = None
_use_mixed_precision = False

def initialize_gpu(config: Dict[str, Any]) -> None:
    """初始化GPU设置"""
    global _device, _dtype, _use_mixed_precision
    
    gpu_config = config.get('gpu', {})
    
    if gpu_config.get('enabled', False) and torch.cuda.is_available():
        _device = torch.device(gpu_config.get('device', 'cuda:0'))
        _use_mixed_precision = gpu_config.get('mixed_precision', True)
        _dtype = torch.float16 if _use_mixed_precision else torch.float32
        
        print(f"GPU加速已启用: {_device}, 混合精度: {_use_mixed_precision}")
        print(f"GPU内存: {torch.cuda.get_device_properties(_device).total_memory / 1024**3:.1f}GB")
    else:
        _device = torch.device('cpu')
        _dtype = torch.float32
        _use_mixed_precision = False
        print("使用CPU计算")

def get_device() -> torch.device:
    """获取当前设备"""
    global _device
    return _device if _device is not None else torch.device('cpu')

def get_dtype() -> torch.dtype:
    """获取当前数据类型"""
    global _dtype
    return _dtype if _dtype is not None else torch.float32

def use_mixed_precision() -> bool:
    """是否使用混合精度"""
    global _use_mixed_precision
    return _use_mixed_precision

def to_torch(arr: np.ndarray, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """将numpy数组转换为torch tensor，自动处理大数组"""
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()
    
    # 如果使用GPU且数组很大，使用float32以节省内存
    if device.type == 'cuda' and arr.nbytes > 100 * 1024 * 1024:  # 100MB
        dtype = torch.float32 if dtype == torch.float16 else dtype
        print(f"警告: 大数组({arr.nbytes/1024/1024:.1f}MB)，使用float32以节省GPU内存")
    
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将torch tensor转换为numpy数组"""
    return tensor.detach().cpu().numpy()

def clear_gpu_cache() -> None:
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def slerp(q0: Union[np.ndarray, torch.Tensor], q1: Union[np.ndarray, torch.Tensor], t: float) -> Union[np.ndarray, torch.Tensor]:
    """球面线性插值，用于平滑旋转动画，支持GPU加速
    
    Args:
        q0: 起始四元数 [x, y, z, w]
        q1: 结束四元数 [x, y, z, w]
        t: 插值参数 [0, 1]
    
    Returns:
        插值后的四元数
    """
    # 检测输入类型并决定使用numpy还是torch
    use_torch = isinstance(q0, torch.Tensor) or isinstance(q1, torch.Tensor)
    
    if use_torch:
        # GPU加速版本
        if not isinstance(q0, torch.Tensor):
            q0 = to_torch(q0)
        if not isinstance(q1, torch.Tensor):
            q1 = to_torch(q1)
        
        # 确保四元数已归一化
        q0 = q0 / torch.norm(q0)
        q1 = q1 / torch.norm(q1)
        
        # 计算两个四元数的点积
        dot = torch.dot(q0, q1)
        
        # 如果点积为负，取反其中一个四元数以选择更短的路径
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        
        # 如果两个四元数非常接近，使用线性插值
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / torch.norm(result)
        
        # 计算插值角度
        theta_0 = torch.arccos(torch.abs(dot))
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        
        # 计算插值系数
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q0 + s1 * q1
    else:
        # CPU版本（保持原有逻辑）
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


def quaternion_from_euler(roll: float, pitch: float, yaw: float, use_gpu: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """从欧拉角创建四元数（适配Habitat坐标系），支持GPU加速
    
    Args:
        roll: 滚转角 (绕X轴旋转，度)
        pitch: 俯仰角 (绕X轴旋转，度) - 注意：在Habitat中主要使用偏航角
        yaw: 偏航角 (绕Y轴旋转，度) - 这是机器人导航中的主要旋转轴
        use_gpu: 是否使用GPU计算
    
    Returns:
        四元数 [x, y, z, w]
    """
    if use_gpu and get_device().type == 'cuda':
        # GPU加速版本
        yaw_rad = torch.tensor(np.radians(yaw), device=get_device(), dtype=get_dtype())
        half_yaw = yaw_rad / 2.0
        
        result = torch.zeros(4, device=get_device(), dtype=get_dtype())
        result[0] = 0.0  # x
        result[1] = torch.sin(half_yaw)  # y
        result[2] = 0.0  # z
        result[3] = torch.cos(half_yaw)  # w
        
        return result
    else:
        # CPU版本（保持原有逻辑）
        # 转换为弧度
        yaw_rad = np.radians(yaw)
        
        # 对于机器人导航，主要使用绕Y轴的偏航角
        # 简化为只处理偏航角，与habitat_video_generator.py中的实现一致
        return np.array([
            0.0,
            np.sin(yaw_rad / 2.0),
            0.0,
            np.cos(yaw_rad / 2.0)
        ], dtype=np.float32)


def euler_from_quaternion(quat: Union[np.ndarray, torch.Tensor]) -> Tuple[float, float, float]:
    """从四元数提取欧拉角，支持GPU加速
    
    Args:
        quat: 四元数 [x, y, z, w]
    
    Returns:
        (roll, pitch, yaw) 欧拉角（度）
        roll: 绕X轴旋转
        pitch: 绕Z轴旋转
        yaw: 绕Y轴旋转（机器人导航中的主要旋转轴）
    """
    if isinstance(quat, torch.Tensor):
        # GPU加速版本
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (绕X轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (绕Z轴旋转)
        sinp = 2 * (w * z - x * y)
        pitch = torch.where(torch.abs(sinp) >= 1, 
                          torch.copysign(torch.tensor(np.pi / 2, device=quat.device), sinp),
                          torch.asin(sinp))
        
        # Yaw (绕Y轴旋转) - 这是机器人导航中的主要旋转轴
        siny_cosp = 2 * (w * y + x * z)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return (torch.rad2deg(roll).item(), 
                torch.rad2deg(pitch).item(), 
                torch.rad2deg(yaw).item())
    else:
        # CPU版本（保持原有逻辑）
        x, y, z, w = quat
        
        # Roll (绕X轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (绕Z轴旋转)
        sinp = 2 * (w * z - x * y)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 使用90度如果超出范围
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (绕Y轴旋转) - 这是机器人导航中的主要旋转轴
        siny_cosp = 2 * (w * y + x * z)
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


def calculate_direction_vector(start_pos: np.ndarray, end_pos: np.ndarray, use_gpu: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """计算从起始点到终点的方向向量（已归一化），支持GPU加速
    
    Args:
        start_pos: 起始位置 [x, y, z]
        end_pos: 终点位置 [x, y, z]
        use_gpu: 是否使用GPU计算
    
    Returns:
        归一化的方向向量 [x, y, z]
    """
    if use_gpu and get_device().type == 'cuda':
        # GPU加速版本
        if not isinstance(start_pos, torch.Tensor):
            start_pos = to_torch(start_pos)
        if not isinstance(end_pos, torch.Tensor):
            end_pos = to_torch(end_pos)
        
        direction = end_pos - start_pos
        distance = torch.norm(direction)
        
        if distance == 0:
            return torch.tensor([0.0, 0.0, 1.0], device=get_device(), dtype=get_dtype())
        
        return direction / distance
    else:
        # CPU版本（保持原有逻辑）
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            return np.array([0.0, 0.0, 1.0])  # 默认朝向Z轴正方向
        
        return direction / distance




def quaternion_to_direction_yaw(position: Union[np.ndarray, torch.Tensor], 
                               target_position: Union[np.ndarray, torch.Tensor], 
                               use_gpu: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """计算朝向目标位置所需的偏航角四元数，支持GPU加速。
    
    Args:
        position: 当前位置 [x, y, z]
        target_position: 目标位置 [x, y, z]
        use_gpu: 是否使用GPU计算
    
    Returns:
        表示偏航角的四元数 [x, y, z, w]
    """
    if use_gpu and get_device().type == 'cuda':
        # GPU加速版本
        if not isinstance(position, torch.Tensor):
            position = to_torch(position)
        if not isinstance(target_position, torch.Tensor):
            target_position = to_torch(target_position)
        
        # 使用统一角度系统计算目标角度
        dx = target_position[0] - position[0]
        dz = target_position[2] - position[2]
        
        if torch.isclose(dx, torch.tensor(0.0, device=dx.device)) and torch.isclose(dz, torch.tensor(0.0, device=dz.device)):
            return torch.tensor([0., 0., 0., 1.], device=get_device(), dtype=get_dtype())
        
        # 使用统一角度系统
        unified_angle = torch.atan2(dx, -dz)
        half_angle = unified_angle / 2.0
        
        q = torch.zeros(4, device=get_device(), dtype=get_dtype())
        q[0] = 0                           # x
        q[1] = torch.sin(half_angle)       # y (注意：这里使用sin而不是-sin)
        q[2] = 0                           # z
        q[3] = torch.cos(half_angle)       # w
        
        return q
    else:
        # CPU版本（保持原有逻辑）
        # 使用统一角度系统计算目标角度
        dx = target_position[0] - position[0]
        dz = target_position[2] - position[2]
        
        if np.isclose(dx, 0) and np.isclose(dz, 0):
            return np.array([0., 0., 0., 1.])
        
        # 使用统一角度系统
        unified_angle = np.arctan2(dx, -dz)
        half_angle = unified_angle / 2.0

        q = np.array([
            0,                 
            np.sin(half_angle),  # 注意：这里使用sin而不是-sin
            0,                 
            np.cos(half_angle)   
        ])
        
        return q


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
        'scene.scene_file', 'scene.robot_urdf'
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

def unified_angle_system():
    """
    统一角度系统说明：
    - 0度：指向-Z轴（Habitat的前进方向）
    - 正角度：逆时针旋转
    - 负角度：顺时针旋转
    - 角度范围：[-π, π]
    """
    pass

def cartesian_to_unified_angle(dx: float, dz: float) -> float:
    """
    将笛卡尔坐标差转换为统一角度系统
    
    Args:
        dx: X方向差值 (target_x - current_x)
        dz: Z方向差值 (target_z - current_z)
    
    Returns:
        统一角度系统中的角度（弧度）
    """
    # 在统一系统中：
    # 0度 = -Z轴方向
    # 90度 = -X轴方向  
    # 180度 = +Z轴方向
    # -90度 = +X轴方向
    
    # 使用atan2(dx, -dz)来获得正确的角度
    # 这样：当dz<0时（目标在-Z方向），角度为0
    # 当dx>0时（目标在+X方向），角度为-90度
    angle_radians = np.arctan2(-dx, -dz)
    return angle_radians

def unified_angle_to_direction_vector(angle: float) -> np.ndarray:
    """
    将统一角度转换为方向向量
    
    Args:
        angle: 统一角度系统中的角度（弧度）
    
    Returns:
        方向向量 [x, y, z]
    """
    return np.array([
        -np.sin(angle),  # X分量
        0,               # Y分量
        -np.cos(angle)   # Z分量
    ])

def yaw_to_unified_angle(yaw_rad: float) -> float:
    """
    将四元数yaw角转换为统一角度系统
    
    Args:
        yaw_rad: 四元数yaw角（弧度）
    
    Returns:
        统一角度系统中的角度（弧度）
    """
    # 四元数yaw角与统一角度系统是一致的
    # 因为我们的四元数转换已经使用了相同的约定
    return yaw_rad

def unified_angle_to_yaw(unified_angle: float) -> float:
    """
    将统一角度转换为四元数yaw角
    
    Args:
        unified_angle: 统一角度系统中的角度（弧度）
    
    Returns:
        四元数yaw角（弧度）
    """
    return unified_angle

def get_target_angle_unified(current_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    计算到目标的统一角度
    
    Args:
        current_pos: 当前位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]
    
    Returns:
        统一角度系统中的角度（弧度）
    """
    dx = target_pos[0] - current_pos[0]
    dz = target_pos[2] - current_pos[2]  # 注意使用Z轴
    
    return cartesian_to_unified_angle(dx, dz)

def normalize_unified_angle(angle: float) -> float:
    """
    标准化统一角度到[-π, π]
    
    Args:
        angle: 输入角度（弧度）
    
    Returns:
        标准化后的角度（弧度）
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def delta_unified_angle(angle1: float, angle2: float) -> float:
    """
    计算统一角度系统中两个角度的差值
    
    Args:
        angle1: 角度1（弧度）
        angle2: 角度2（弧度）
    
    Returns:
        角度差值（弧度）
    """
    return normalize_unified_angle(angle1 - angle2)
