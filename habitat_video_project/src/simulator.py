"""
HabitatSimulator - 核心模拟器类 (Model)
处理Habitat-Sim的底层交互和状态管理
"""

import numpy as np
import magnum as mn
import habitat_sim
from PIL import Image
from typing import Dict, Tuple, Optional, Any
import math
import os

from .utils import convert_to_magnum_quat, convert_to_numpy_quat


class HabitatSimulator:
    """封装Habitat-sim相关逻辑的核心模拟器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.sim = None
        self.agent = None
        self.robot_object = None
        
        # 场景信息
        self.scene_bounds = None
        self.scene_center = None
        self.scene_size = None
        self.base_map_image = None
        
        # 坐标转换参数
        self.map_width = None
        self.map_height = None
        self.world_to_map_scale_x = None
        self.world_to_map_scale_z = None
        
        self._initialize_sim()
    
    def _initialize_sim(self):
        """配置并创建habitat_sim.Simulator实例"""
        # 后端配置
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.config['scene']['scene_file']
        backend_cfg.enable_physics = self.config['simulation']['enable_physics']
        backend_cfg.gpu_device_id = self.config['simulation']['gpu_device_id']
        backend_cfg.random_seed = 1
        
        # FPV传感器配置
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [512, 512]  # 临时分辨率，后续会调整
        fpv_sensor_spec.position = mn.Vector3(0, self.config['agent']['sensor_height'], 0)
        fpv_sensor_spec.hfov = 90.0
        
        # 正交传感器配置（用于生成topdown地图）
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"
        ortho_sensor_spec.resolution = [4096, 4096]  # 高分辨率用于地图生成
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        ortho_sensor_spec.far = 1000.0
        ortho_sensor_spec.near = 0.01
        ortho_sensor_spec.hfov = 90
        ortho_sensor_spec.ortho_scale = 0.05  # 参考TopV.py的配置
        ortho_sensor_spec.clear_color = [0., 0., 0., 0.]
        
        # 智能体配置
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, ortho_sensor_spec]
        
        # 动作空间配置
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        # 创建模拟器
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.get_agent(0)
        
        print("模拟器初始化完成")
    
    def setup_scene_and_agent(self, initial_state: Dict[str, Any]):
        """
        设置场景和智能体
        
        Args:
            initial_state: 初始状态配置，包含position和rotation
        """
        # 1. 生成topdown地图
        self._generate_topdown_map()
        
        # 2. 加载物理机器人
        if os.path.exists(self.config['scene']['robot_urdf']):
            initial_3d_pos = self._convert_2d_to_3d(
                initial_state['position'][0], 
                initial_state['position'][1]
            )
            if initial_3d_pos is not None:
                self._load_physical_robot(self.config['scene']['robot_urdf'], initial_3d_pos)
            else:
                print("警告: 初始位置不可导航，未加载物理机器人")
        else:
            print(f"警告: URDF文件不存在: {self.config['scene']['robot_urdf']}")
        
        # 3. 设置初始位置和朝向
        initial_3d_pos = self._convert_2d_to_3d(
            initial_state['position'][0], 
            initial_state['position'][1]
        )
        
        if initial_3d_pos is not None:
            initial_rotation = self._yaw_to_quaternion(initial_state['rotation'])
            self.set_robot_pose(initial_3d_pos, initial_rotation)
            print(f"智能体初始化到位置: {initial_3d_pos}, 朝向: {initial_state['rotation']}度")
        else:
            print("错误: 无法将智能体放置到指定初始位置")
    
    def _generate_topdown_map(self) -> None:
        """生成topdown地图 - 参考TopViewGenerator.py的实现"""
        # 确保导航网格已加载
        if not self.sim.pathfinder.is_loaded:
            print("警告: 导航网格未加载")
            return
        
        # 使用导航网格顶点计算场景边界
        navmesh_vertices = np.array(self.sim.pathfinder.build_navmesh_vertices())
        if len(navmesh_vertices) == 0:
            print("错误: 无法获取导航网格顶点")
            return
        
        # 计算场景边界
        min_bounds = navmesh_vertices.min(axis=0)
        max_bounds = navmesh_vertices.max(axis=0)
        
        self.scene_bounds = (min_bounds.tolist(), max_bounds.tolist())
        self.scene_center = ((min_bounds + max_bounds) / 2.0).tolist()
        self.scene_size = (max_bounds - min_bounds).tolist()
        
        print(f"场景边界: {self.scene_bounds}")
        print(f"场景中心: {self.scene_center}")
        print(f"场景尺寸: {self.scene_size}")
        
        # 设置相机位置 - 使用场景中心上方的合理高度
        camera_height = max_bounds[1] + max(self.scene_size[0], self.scene_size[2]) * 0.5
        camera_position = mn.Vector3(self.scene_center[0], camera_height, self.scene_center[2])
        
        # 设置智能体状态以获取俯视图
        agent_state = habitat_sim.AgentState()
        agent_state.position = camera_position
        agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068])  # 朝下看
        self.agent.set_state(agent_state)
        
        print(f"正交相机位置: {camera_position}")
        print(f"相机高度: {camera_height:.2f}m")
        
        # 获取俯视图
        observations = self.sim.get_sensor_observations()
        
        if "ortho_sensor" in observations:
            ortho_img = observations["ortho_sensor"]
        else:
            print(f"可用传感器: {list(observations.keys())}")
            print("错误: 未找到正交传感器")
            return
        
        # 转换为PIL图像
        self.base_map_image = Image.fromarray(ortho_img[..., :3], "RGB")
        
        # 保存地图尺寸和坐标转换参数
        self.map_width, self.map_height = self.base_map_image.size
        
        # 计算世界坐标到地图坐标的缩放因子
        world_width = self.scene_bounds[1][0] - self.scene_bounds[0][0]
        world_height = self.scene_bounds[1][2] - self.scene_bounds[0][2]
        
        self.world_to_map_scale_x = self.map_width / world_width
        self.world_to_map_scale_z = self.map_height / world_height
        
        print(f"地图尺寸: {self.map_width} x {self.map_height}")
        print(f"坐标缩放因子: X={self.world_to_map_scale_x:.2f}, Z={self.world_to_map_scale_z:.2f}")
    
    def _load_physical_robot(self, model_path: str, initial_position: np.ndarray) -> Optional[Any]:
        """
        加载物理机器人URDF模型
        
        Args:
            model_path: URDF文件路径
            initial_position: 初始3D位置
        
        Returns:
            机器人对象或None
        """
        try:
            # 获取关节对象管理器
            articulated_obj_mgr = self.sim.get_articulated_object_manager()
            
            # 从URDF文件加载机器人
            self.robot_object = articulated_obj_mgr.add_articulated_object_from_urdf(
                filepath=model_path,
                fixed_base=False,  # 允许机器人移动
                global_scale=1.0,
                mass_scale=1.0
            )
            
            # 设置初始位置
            robot_initial_transform = mn.Matrix4.translation(mn.Vector3(
                initial_position[0], initial_position[1], initial_position[2]
            ))
            self.robot_object.transformation = robot_initial_transform
            
            print(f"物理机器人加载成功: {model_path}")
            return self.robot_object
            
        except Exception as e:
            print(f"加载物理机器人失败: {e}")
            return None
    
    def _convert_2d_to_3d(self, x: float, z: float) -> Optional[np.ndarray]:
        """
        将2D坐标转换为3D坐标，通过navmesh获取Y坐标
        
        Args:
            x: X坐标
            z: Z坐标
        
        Returns:
            3D坐标[x, y, z]或None（如果不可导航）
        """
        y = self.get_navigable_y(x, z)
        if y is not None:
            return np.array([x, y, z], dtype=np.float32)
        return None
    
    def get_navigable_y(self, x: float, z: float) -> Optional[float]:
        """
        获取指定(x,z)位置对应的可导航Y坐标
        
        Args:
            x: X坐标
            z: Z坐标
        
        Returns:
            Y坐标或None（如果不可导航）
        """
        try:
            test_point = mn.Vector3(x, 0.0, z)
            snapped_point = self.sim.pathfinder.snap_point(test_point)
            
            if self.sim.pathfinder.is_navigable(snapped_point):
                return float(snapped_point.y)
            else:
                return None
                
        except Exception as e:
            print(f"获取导航高度失败: {e}")
            return None
    
    def check_straight_path_collision(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                    step_size: float = 0.1) -> bool:
        """
        检查直线路径是否会发生碰撞
        
        Args:
            start_pos: 起始位置 [x, y, z]
            end_pos: 终点位置 [x, y, z]
            step_size: 检测步长（米）
        
        Returns:
            True表示会发生碰撞，False表示路径安全
        """
        try:
            # 计算路径方向和总距离
            direction = end_pos - start_pos
            total_distance = np.linalg.norm(direction)
            
            if total_distance == 0:
                return False
            
            # 归一化方向向量
            direction = direction / total_distance
            
            # 计算检测步数
            num_steps = max(1, int(total_distance / step_size))
            
            # 逐步检查路径
            for i in range(1, num_steps + 1):
                # 计算当前检测点
                progress = min(i * step_size, total_distance)
                current_pos = start_pos + direction * progress
                next_pos = start_pos + direction * min((i + 1) * step_size, total_distance)
                
                # 使用pathfinder检查这一步是否可行
                current_point = mn.Vector3(current_pos[0], current_pos[1], current_pos[2])
                next_point = mn.Vector3(next_pos[0], next_pos[1], next_pos[2])
                
                # 尝试从当前点移动到下一点
                filtered_end = self.sim.pathfinder.try_step(current_point, next_point)
                
                # 计算实际移动距离和期望移动距离
                expected_move = np.linalg.norm(next_pos - current_pos)
                actual_move = np.linalg.norm(
                    [filtered_end.x - current_pos[0], 
                     filtered_end.y - current_pos[1], 
                     filtered_end.z - current_pos[2]]
                )
                
                # 如果实际移动距离明显小于期望距离，说明遇到障碍物
                if actual_move + 1e-5 < expected_move:
                    return True
            
            return False
            
        except Exception as e:
            print(f"碰撞检测失败: {e}")
            return True  # 保守策略，检测失败时认为会碰撞
    
    def set_robot_pose(self, position: np.ndarray, rotation_quat: np.ndarray):
        """
        设置机器人姿态（同时更新物理机器人和虚拟智能体）
        
        Args:
            position: 3D位置 [x, y, z]
            rotation_quat: 四元数旋转 [x, y, z, w]
        """
        try:
            # 更新物理机器人
            if self.robot_object is not None:
                pos_vec = mn.Vector3(position[0], position[1], position[2])
                quat = convert_to_magnum_quat(rotation_quat)
                
                # 创建变换矩阵
                rotation_matrix = mn.Matrix4.from_(quat.to_matrix(), mn.Vector3())
                translation_matrix = mn.Matrix4.translation(pos_vec)
                
                # 设置机器人变换
                self.robot_object.transformation = translation_matrix @ rotation_matrix
            
            # 同步更新虚拟智能体（用于传感器）
            agent_state = habitat_sim.AgentState()
            agent_state.position = position
            agent_state.rotation = rotation_quat
            self.agent.set_state(agent_state)
            
        except Exception as e:
            print(f"设置机器人姿态失败: {e}")
    
    def get_fpv_observation(self) -> np.ndarray:
        """
        获取第一人称视角图像
        
        Returns:
            RGB图像数组
        """
        try:
            # 如果有物理机器人，从其传感器位置获取观察
            if self.robot_object is not None:
                return self._get_robot_sensor_observation()
            else:
                # 否则直接从虚拟智能体获取
                observations = self.sim.get_sensor_observations()
                return observations["color_sensor"]
                
        except Exception as e:
            print(f"获取FPV观察失败: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def _get_robot_sensor_observation(self) -> np.ndarray:
        """从物理机器人的传感器位置获取观察"""
        try:
            # 获取机器人当前变换
            robot_transform = self.robot_object.transformation
            robot_position = robot_transform.translation
            robot_rotation = mn.Quaternion.from_matrix(robot_transform.rotation())
            
            # 计算传感器位置（假设传感器在机器人头部）
            sensor_offset = mn.Vector3(0, self.config['agent']['sensor_height'], 0)
            sensor_position = robot_position + robot_transform.transform_vector(sensor_offset)
            
            # 临时设置虚拟智能体到传感器位置
            temp_agent_state = habitat_sim.AgentState()
            temp_agent_state.position = np.array([sensor_position.x, sensor_position.y, sensor_position.z])
            temp_agent_state.rotation = convert_to_numpy_quat(robot_rotation)
            
            # 保存当前状态
            original_state = self.agent.get_state()
            
            # 设置临时状态并获取观察
            self.agent.set_state(temp_agent_state)
            observations = self.sim.get_sensor_observations()
            observation = observations["color_sensor"]
            
            # 恢复原状态
            self.agent.set_state(original_state)
            
            return observation
            
        except Exception as e:
            print(f"从机器人传感器获取观察失败: {e}")
            # 回退到虚拟智能体
            observations = self.sim.get_sensor_observations()
            return observations["color_sensor"]
    
    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        获取机器人当前状态
        
        Returns:
            包含position和rotation的字典
        """
        try:
            if self.robot_object is not None:
                # 从物理机器人获取状态
                transform = self.robot_object.transformation
                position = transform.translation
                rotation_quat = mn.Quaternion.from_matrix(transform.rotation())
                
                return {
                    'position': np.array([position.x, position.y, position.z]),
                    'rotation': convert_to_numpy_quat(rotation_quat)
                }
            else:
                # 从虚拟智能体获取状态
                agent_state = self.agent.get_state()
                return {
                    'position': agent_state.position,
                    'rotation': agent_state.rotation
                }
                
        except Exception as e:
            print(f"获取机器人状态失败: {e}")
            return {
                'position': np.array([0.0, 0.0, 0.0]),
                'rotation': np.array([0.0, 0.0, 0.0, 1.0])
            }
    
    def get_base_map(self) -> Optional[Image.Image]:
        """
        获取基础topdown地图
        
        Returns:
            PIL图像对象
        """
        return self.base_map_image
    
    def world_to_map_coords(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """
        将3D世界坐标转换为2D地图像素坐标
        
        Args:
            world_pos: 世界坐标 [x, y, z]
        
        Returns:
            地图像素坐标 (map_x, map_y)
        """
        if self.base_map_image is None:
            return (0, 0)
        
        # 将世界坐标映射到地图坐标
        map_x = (world_pos[0] - self.scene_bounds[0][0]) * self.world_to_map_scale_x
        map_y = (world_pos[2] - self.scene_bounds[0][2]) * self.world_to_map_scale_z
        
        # 转换为整数像素坐标并确保在范围内
        map_x = max(0, min(int(map_x), self.map_width - 1))
        map_y = max(0, min(int(map_y), self.map_height - 1))
        
        return (map_x, map_y)
    
    def _yaw_to_quaternion(self, yaw_degrees: float) -> np.ndarray:
        """
        将偏航角（度）转换为四元数
        
        Args:
            yaw_degrees: 偏航角（度）
        
        Returns:
            四元数 [x, y, z, w]
        """
        yaw_rad = math.radians(yaw_degrees)
        
        # 绕Y轴旋转的四元数
        return np.array([
            0.0,
            math.sin(yaw_rad / 2.0),
            0.0,
            math.cos(yaw_rad / 2.0)
        ], dtype=np.float32)
    
    def close(self):
        """关闭模拟器"""
        if self.sim:
            self.sim.close()
            print("模拟器已关闭")
