#!/usr/bin/env python3
"""
Multi-Agent Habitat Navigation System
支持多智能体同步导航的复杂系统，具备预测性碰撞检测和状态持久化功能
"""

import sys
import os
import json
import yaml
import logging
import time
import math
import numpy as np
import cv2
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

# 添加interactive_app的src路径以复用代码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Habitat相关导入
import habitat_sim
import magnum as mn
from habitat_sim.utils import viz_utils as vut

# 导入基础HabitatSimulator类（复用已有的实现）
from habitat_video_generator import HabitatSimulator, CustomHabitatSimulator


@dataclass
class AgentState:
    """智能体状态数据类"""
    position: np.ndarray
    rotation: np.ndarray  # quaternion [x, y, z, w]
    velocity: float = 0.0
    angular_velocity: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式以便序列化"""
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentState':
        """从字典格式恢复状态"""
        return cls(
            position=np.array(data["position"]),
            rotation=np.array(data["rotation"]),
            velocity=data.get("velocity", 0.0),
            angular_velocity=data.get("angular_velocity", 0.0)
        )


@dataclass
class ActionCommand:
    """动作命令数据类"""
    action: str
    target: Optional[List[float]] = None
    angle: Optional[float] = None
    distance: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ActionCommand':
        """从字典格式创建动作命令"""
        return cls(
            action=data["action"],
            target=data.get("target"),
            angle=data.get("angle"),
            distance=data.get("distance")
        )


class DummyVideoWriter:
    """虚拟视频写入器，用于处理编码器失败的情况"""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.frame_count = 0
        
    def write(self, frame):
        """虚拟写入帧"""
        self.frame_count += 1
        
    def release(self):
        """虚拟释放资源"""
        logging.info(f"Dummy video writer released for {self.output_path} ({self.frame_count} frames)")
        
    def isOpened(self):
        """总是返回True"""
        return True


class CollisionDetector:
    """碰撞检测器"""
    
    def __init__(self, config: Dict):
        self.enabled = config.get("enabled", True)
        self.agent_radius = config.get("agent_radius", 0.4)
        self.height_threshold = config.get("height_threshold", 0.3)
        self.prediction_steps = config.get("prediction_steps", 3)
        self.min_agent_distance = config.get("min_agent_distance", 0.8)
        
    def check_collision_with_environment(self, sim: habitat_sim.Simulator, 
                                       position: np.ndarray) -> bool:
        """检查与环境的碰撞"""
        if not self.enabled:
            return False
            
        try:
            # 检查位置是否可导航
            test_point = mn.Vector3(position[0], position[1], position[2])
            is_navigable = sim.pathfinder.is_navigable(test_point)
            
            if not is_navigable:
                return True
                
            # 检查周围一圈的可导航性（考虑智能体半径）
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                check_x = position[0] + self.agent_radius * np.cos(angle)
                check_z = position[2] + self.agent_radius * np.sin(angle)
                check_point = mn.Vector3(check_x, position[1], check_z)
                
                if not sim.pathfinder.is_navigable(check_point):
                    return True
                    
            return False
            
        except Exception as e:
            logging.warning(f"Environment collision check failed: {e}")
            return True  # 安全起见，认为有碰撞
    
    def check_collision_between_agents(self, agent_positions: Dict[str, np.ndarray]) -> Tuple[bool, List[Tuple[str, str]]]:
        """检查智能体之间的碰撞"""
        if not self.enabled or len(agent_positions) < 2:
            return False, []
            
        collisions = []
        agents = list(agent_positions.items())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1_id, pos1 = agents[i]
                agent2_id, pos2 = agents[j]
                
                # 计算距离（忽略Y坐标）
                distance = np.linalg.norm(pos1[[0, 2]] - pos2[[0, 2]])
                
                if distance < self.min_agent_distance:
                    collisions.append((agent1_id, agent2_id))
        
        return len(collisions) > 0, collisions
    
    def predict_collision(self, sim: habitat_sim.Simulator,
                         agent_positions: Dict[str, np.ndarray],
                         planned_movements: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """预测执行动作后是否会发生碰撞"""
        if not self.enabled:
            return False, ""
            
        # 预测每个智能体的未来位置
        future_positions = {}
        for agent_id, current_pos in agent_positions.items():
            if agent_id in planned_movements:
                future_pos = current_pos + planned_movements[agent_id]
                future_positions[agent_id] = future_pos
            else:
                future_positions[agent_id] = current_pos
        
        # 检查与环境的碰撞
        for agent_id, future_pos in future_positions.items():
            if self.check_collision_with_environment(sim, future_pos):
                return True, f"Agent {agent_id} will collide with environment"
        
        # 检查智能体之间的碰撞
        has_collision, collision_pairs = self.check_collision_between_agents(future_positions)
        if has_collision:
            pair_str = ", ".join([f"{a1}-{a2}" for a1, a2 in collision_pairs])
            return True, f"Agent collision predicted: {pair_str}"
        
        return False, ""


class MultiAgentSimulator:
    """多智能体模拟器，基于HabitatSimulator扩展"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scene_config = config["scene"]
        self.sim_config = config["simulator"]
        self.agent_configs = config["agents"]
        self.collision_detector = CollisionDetector(config["collision_detection"])
        
        # 初始化日志
        self._setup_logging()
        
        # 状态管理
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_robots: Dict[str, Any] = {}  # 存储每个智能体的物理机器人对象
        self.agent_configs_dict: Dict[str, Dict] = {}  # 存储智能体配置
        self.current_actions: Dict[str, Optional[ActionCommand]] = {}
        
        # 单个模拟器实例管理所有智能体
        self.simulator: Optional[CustomHabitatSimulator] = None
        
        # 为每个智能体配置建立索引
        for agent_config in self.agent_configs:
            agent_id = agent_config["id"]
            self.agent_configs_dict[agent_id] = agent_config
        
        # 运动参数
        self.movement_config = config["movement"]
        self.linear_speed = self.movement_config["linear_speed"]
        self.angular_speed = self.movement_config["angular_speed"]
        self.time_step = self.movement_config["time_step"]
        
        # 视频输出配置
        self.video_config = config["video_output"]
        self.map_config = config["map_config"]
        
        # 初始化模拟器和智能体
        self._initialize_single_simulator()
        self._initialize_all_agents()
        self._generate_clean_map()
        
        logging.info("Multi-agent simulator initialized successfully")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config["logging"]
        
        # 创建输出目录
        os.makedirs(os.path.dirname(log_config["log_file"]), exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_config["log_level"]),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config["log_file"]),
                logging.StreamHandler() if log_config["console_output"] else logging.NullHandler()
            ]
        )
    
    def _initialize_single_simulator(self):
        """初始化单个模拟器实例来管理所有智能体"""
        scene_path = self._resolve_scene_path()
        
        try:
            # 创建单个模拟器实例
            # 使用第一个智能体的配置作为基础配置
            first_agent_config = self.agent_configs[0]
            
            self.simulator = CustomHabitatSimulator(
                scene_filepath=scene_path,
                resolution=tuple(first_agent_config["sensors"]["color_sensor"]["resolution"]),
                gpu_device_id=self.sim_config.get("gpu_device_id", 0),
                enable_physics=self.sim_config.get("enable_physics", True),
                agent_model_path=None  # 物理机器人将单独加载
            )
            
            logging.info(f"Single simulator initialized for {len(self.agent_configs)} agents")
            
        except Exception as e:
            logging.error(f"Failed to initialize simulator: {e}")
            raise
    
    def _initialize_all_agents(self):
        """初始化所有智能体和其物理机器人"""
        for agent_config in self.agent_configs:
            agent_id = agent_config["id"]
            
            try:
                # 初始化智能体状态
                initial_pos = agent_config.get("initial_position")
                initial_rot = np.array(agent_config.get("initial_rotation", [0, 0, 0, 1]))
                
                if initial_pos is not None:
                    initial_pos = np.array(initial_pos)
                else:
                    # 使用场景中心作为默认位置，为不同智能体添加偏移
                    bounds = self.simulator.scene_bounds
                    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
                    center_z = (bounds[0][2] + bounds[1][2]) / 2.0
                    
                    # 为不同智能体添加偏移以避免重叠
                    agent_index = list(self.agent_configs_dict.keys()).index(agent_id)
                    offset_x = agent_index * 2.0  # 每个智能体间隔2米
                    offset_z = agent_index * 1.0  # Z方向也有偏移
                    
                    initial_pos = self.simulator.get_position_with_navmesh_height(
                        center_x + offset_x, center_z + offset_z
                    )
                    
                    if initial_pos is None:
                        # 如果计算的位置不可导航，使用随机可导航点
                        random_point = self.simulator.sim.pathfinder.get_random_navigable_point()
                        initial_pos = np.array([random_point[0], random_point[1], random_point[2]])
                
                # 创建智能体状态
                self.agent_states[agent_id] = AgentState(
                    position=initial_pos,
                    rotation=initial_rot
                )
                
                # 加载物理机器人（如果指定）
                agent_model_path = agent_config.get("agent_model_path")
                logging.info(f"Agent {agent_id} model path: {agent_model_path}")
                
                if agent_model_path:
                    # 检查文件是否存在
                    if os.path.exists(agent_model_path):
                        logging.info(f"URDF file found: {agent_model_path}")
                        robot_obj = self._load_physical_robot(agent_id, agent_model_path, initial_pos)
                        if robot_obj:
                            self.agent_robots[agent_id] = robot_obj
                            logging.info(f"✓ Agent {agent_id} loaded with physical robot: {os.path.basename(agent_model_path)}")
                        else:
                            logging.warning(f"⚠ Agent {agent_id} failed to load physical robot, using virtual agent")
                    else:
                        logging.warning(f"⚠ Agent {agent_id} URDF file not found: {agent_model_path}")
                        logging.info(f"→ Agent {agent_id} using virtual agent (URDF file missing)")
                else:
                    logging.info(f"→ Agent {agent_id} using virtual agent (no physical robot specified)")
                
            except Exception as e:
                logging.error(f"Failed to initialize agent {agent_id}: {e}")
                raise
    
    def _load_physical_robot(self, agent_id: str, model_path: str, initial_position: np.ndarray) -> Optional[Any]:
        """为指定智能体加载物理机器人"""
        try:
            # 在Habitat-Sim中，URDF机器人需要通过add_articulated_object_from_urdf直接加载
            articulated_obj_mgr = self.simulator.sim.get_articulated_object_manager()
            
            # 直接从URDF文件加载机器人
            robot_object = articulated_obj_mgr.add_articulated_object_from_urdf(
                filepath=model_path,
                fixed_base=False,  # 允许机器人移动
                global_scale=1.0,
                mass_scale=1.0
            )
            
            # 设置机器人初始位置
            robot_initial_transform = mn.Matrix4.translation(mn.Vector3(
                initial_position[0], initial_position[1], initial_position[2]
            ))
            robot_object.transformation = robot_initial_transform
            
            logging.info(f"✓ Physical robot loaded for {agent_id}")
            logging.info(f"  - URDF path: {model_path}")
            logging.info(f"  - Object ID: {robot_object.object_id}")
            logging.info(f"  - Joint count: {len(robot_object.joint_positions)}")
            logging.info(f"  - Initial position: {initial_position}")
            
            return robot_object
            
        except Exception as e:
            logging.error(f"Failed to load physical robot for {agent_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def _resolve_scene_path(self) -> str:
        """解析场景路径"""
        scene_dataset_path = self.scene_config["scene_dataset_path"]
        scene_id = self.scene_config.get("scene_id")
        
        if scene_id:
            return scene_id
        
        # 检查场景路径是否直接是一个GLB文件
        if scene_dataset_path.endswith('.glb'):
            # 如果是绝对路径，直接返回
            if os.path.isabs(scene_dataset_path):
                if os.path.exists(scene_dataset_path):
                    return scene_dataset_path
                else:
                    logging.warning(f"Scene file not found: {scene_dataset_path}")
            
            # 如果是相对路径，尝试不同的基础路径
            possible_bases = [
                os.getcwd(),  # 当前工作目录
                os.path.dirname(os.path.abspath(__file__)),  # 脚本所在目录
                "/home/yaoaa/habitat-lab",  # Habitat-Lab根目录
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")  # 项目根目录
            ]
            
            for base_path in possible_bases:
                full_path = os.path.join(base_path, scene_dataset_path)
                if os.path.exists(full_path):
                    absolute_path = os.path.abspath(full_path)
                    logging.info(f"Found scene file at: {absolute_path}")
                    return absolute_path
            
            # 如果仍未找到，返回原路径让Habitat-Sim处理
            logging.warning(f"Scene file not found, using original path: {scene_dataset_path}")
            return scene_dataset_path
        
        # 如果是数据集配置文件，按原逻辑处理
        else:
            try:
                if not os.path.exists(scene_dataset_path):
                    logging.error(f"Dataset config file not found: {scene_dataset_path}")
                    # 回退到测试场景
                    return self._get_fallback_scene()
                
                with open(scene_dataset_path, 'r') as f:
                    dataset_config = json.load(f)
                
                if "stages" in dataset_config and len(dataset_config["stages"]) > 0:
                    return dataset_config["stages"][0]["filepath"]
                else:
                    raise ValueError("No stages found in scene dataset")
                    
            except Exception as e:
                logging.error(f"Failed to resolve scene path: {e}")
                return self._get_fallback_scene()
    
    def _get_fallback_scene(self) -> str:
        """获取回退场景路径"""
        # 尝试常见的测试场景路径
        fallback_scenes = [
            "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
            "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        ]
        
        for scene_path in fallback_scenes:
            if os.path.exists(scene_path):
                logging.info(f"Using fallback scene: {scene_path}")
                return os.path.abspath(scene_path)
        
        # 如果所有回退都失败，返回第一个路径让Habitat-Sim报告具体错误
        return fallback_scenes[0]
    
    def _generate_clean_map(self):
        """生成干净的静态地图（无智能体标记）"""
        # 使用主模拟器生成基础地图
        self.clean_map = self.simulator.base_map_image.copy()
        logging.info("Clean map generated")
    
    def load_actions_from_file(self, actions_file: str) -> Dict[str, List[ActionCommand]]:
        """从JSON文件加载动作序列"""
        try:
            with open(actions_file, 'r') as f:
                actions_data = json.load(f)
            
            actions = {}
            for agent_id, agent_actions in actions_data.items():
                actions[agent_id] = [ActionCommand.from_dict(action) for action in agent_actions]
            
            logging.info(f"Loaded actions for {len(actions)} agents from {actions_file}")
            return actions
            
        except Exception as e:
            logging.error(f"Failed to load actions from {actions_file}: {e}")
            raise
    
    def execute_actions_sequence(self, actions: Dict[str, List[Any]]) -> bool:
        """执行动作序列，支持状态持久化和碰撞检测"""
        try:
            # 转换actions为ActionCommand对象
            converted_actions = {}
            for agent_id, agent_actions in actions.items():
                converted_actions[agent_id] = []
                for action_data in agent_actions:
                    if isinstance(action_data, dict):
                        # 将字典转换为ActionCommand对象
                        action_cmd = ActionCommand(
                            action=action_data["action"],
                            target=action_data.get("target"),
                            distance=action_data.get("distance"),
                            angle=action_data.get("angle")
                        )
                        converted_actions[agent_id].append(action_cmd)
                    else:
                        # 已经是ActionCommand对象
                        converted_actions[agent_id].append(action_data)
            
            # 初始化智能体位置（从第一个move_to动作获取）
            self._initialize_agent_positions_from_actions(converted_actions)
            
            # 创建视频写入器
            video_writers = self._create_video_writers()
            self._current_video_writers = video_writers  # 存储供动画函数使用
            
            # 写入初始帧
            self._write_initial_frames(video_writers)
            
            # 执行动作序列
            max_actions = max(len(agent_actions) for agent_actions in converted_actions.values())
            
            for step in range(max_actions):
                logging.info(f"Executing step {step + 1}/{max_actions}")
                
                # 收集当前步骤的动作
                current_step_actions = {}
                for agent_id, agent_actions in converted_actions.items():
                    if step < len(agent_actions):
                        current_step_actions[agent_id] = agent_actions[step]
                
                # 预测碰撞
                collision_detected, collision_reason = self._predict_step_collisions(current_step_actions)
                
                if collision_detected:
                    logging.warning(f"Collision predicted at step {step + 1}: {collision_reason}")
                    logging.info("Stopping all agents to avoid collision")
                    break
                
                # 执行动作
                success = self._execute_step_actions(current_step_actions, video_writers)
                
                if not success:
                    logging.error(f"Failed to execute step {step + 1}")
                    break
                
                # 保存状态（如果配置要求）
                if self.config["state_persistence"]["save_after_each_action"]:
                    self._save_current_states()
            
            # 关闭视频写入器
            self._close_video_writers(video_writers)
            
            # 保存最终状态
            if self.config["state_persistence"]["save_final_state"]:
                self._save_current_states()
            
            logging.info("Action sequence execution completed")
            return True
            
        except Exception as e:
            logging.error(f"Action sequence execution failed: {e}")
            return False
    
    def _initialize_agent_positions_from_actions(self, actions: Dict[str, List[ActionCommand]]):
        """从第一个move_to动作初始化智能体位置"""
        for agent_id, agent_actions in actions.items():
            if len(agent_actions) > 0 and agent_actions[0].action == "move_to":
                target = agent_actions[0].target
                if target and len(target) >= 2:
                    position = self.simulator.get_position_with_navmesh_height(target[0], target[1])
                    
                    if position is not None:
                        self.agent_states[agent_id].position = position
                        self.simulator.move_agent_to(position, self.agent_states[agent_id].rotation)
                        
                        # 如果有物理机器人，也移动它
                        if agent_id in self.agent_robots:
                            robot_obj = self.agent_robots[agent_id]
                            transform = mn.Matrix4.translation(mn.Vector3(position[0], position[1], position[2]))
                            robot_obj.transformation = transform
                        
                        logging.info(f"Agent {agent_id} positioned at {target}")
                    else:
                        logging.warning(f"Failed to position agent {agent_id} at {target}")
    
    def _predict_step_collisions(self, step_actions: Dict[str, ActionCommand]) -> Tuple[bool, str]:
        """预测当前步骤的碰撞"""
        if not self.collision_detector.enabled:
            return False, ""
        
        # 计算每个智能体的计划移动
        planned_movements = {}
        current_positions = {}
        
        for agent_id, agent_state in self.agent_states.items():
            current_positions[agent_id] = agent_state.position
            
            if agent_id in step_actions:
                action = step_actions[agent_id]
                movement = self._calculate_action_movement(action, agent_state)
                planned_movements[agent_id] = movement
            else:
                planned_movements[agent_id] = np.array([0.0, 0.0, 0.0])
        
        # 使用主模拟器进行碰撞检测
        return self.collision_detector.predict_collision(
            self.simulator.sim, current_positions, planned_movements
        )
    
    def _calculate_action_movement(self, action: ActionCommand, agent_state: AgentState) -> np.ndarray:
        """计算动作产生的移动向量"""
        if action.action == "move_to" and action.target:
            # 计算到目标的直线移动
            target_pos = np.array([action.target[0], agent_state.position[1], action.target[1]])
            return target_pos - agent_state.position
        
        elif action.action == "move_forward" and action.distance:
            # 计算前进移动
            rotation = agent_state.rotation
            # 从四元数获取前进方向
            quat = mn.Quaternion(mn.Vector3(rotation[0], rotation[1], rotation[2]), rotation[3])
            forward_dir = quat.transform_vector(mn.Vector3(0, 0, -1))  # 负Z为前进方向
            movement = np.array([forward_dir.x, 0, forward_dir.z]) * action.distance
            return movement
        
        else:
            # 旋转动作不产生位置移动
            return np.array([0.0, 0.0, 0.0])
    
    def _execute_step_actions(self, step_actions: Dict[str, ActionCommand], 
                            video_writers: Dict[str, cv2.VideoWriter]) -> bool:
        """执行当前步骤的所有动作"""
        try:
            for agent_id, action in step_actions.items():
                success = self._execute_single_action(agent_id, action)
                if not success:
                    logging.error(f"Failed to execute action for agent {agent_id}: {action}")
                    return False
                
                # 写入视频帧
                self._write_video_frame(agent_id, video_writers[agent_id])
            
            return True
            
        except Exception as e:
            logging.error(f"Step action execution failed: {e}")
            return False
    
    def _execute_single_action(self, agent_id: str, action: ActionCommand) -> bool:
        """执行单个智能体的动作"""
        try:
            agent_state = self.agent_states[agent_id]
            
            if action.action == "move_to":
                return self._execute_move_to(agent_id, action.target, self.simulator, agent_state)
            
            elif action.action == "move_forward":
                return self._execute_move_forward(agent_id, action.distance, self.simulator, agent_state)
            
            elif action.action == "turn_left":
                return self._execute_turn(agent_id, action.angle, True, self.simulator, agent_state)
            
            elif action.action == "turn_right":
                return self._execute_turn(agent_id, action.angle, False, self.simulator, agent_state)
            
            else:
                logging.warning(f"Unknown action: {action.action}")
                return False
                
        except Exception as e:
            logging.error(f"Single action execution failed for {agent_id}: {e}")
            return False
    
    def _execute_move_to(self, agent_id: str, target: List[float], 
                        simulator: HabitatSimulator, agent_state: AgentState) -> bool:
        """执行移动到指定坐标的动作"""
        if not target or len(target) < 2:
            return False
        
        target_pos = simulator.get_position_with_navmesh_height(target[0], target[1])
        if target_pos is None:
            logging.warning(f"Agent {agent_id}: Target position {target} not navigable")
            return False
        
        # 使用平滑移动动画
        return self._animate_movement(agent_id, target_pos, simulator, agent_state)
    
    def _execute_move_forward(self, agent_id: str, distance: float,
                            simulator: HabitatSimulator, agent_state: AgentState) -> bool:
        """执行前进动作"""
        # 计算前进目标位置
        rotation = agent_state.rotation
        quat = mn.Quaternion(mn.Vector3(rotation[0], rotation[1], rotation[2]), rotation[3])
        forward_dir = quat.transform_vector(mn.Vector3(0, 0, -1))
        
        target_pos = agent_state.position + np.array([forward_dir.x, 0, forward_dir.z]) * distance
        target_pos[1] = agent_state.position[1]  # 保持Y坐标
        
        # 检查目标位置是否可导航
        test_point = mn.Vector3(target_pos[0], target_pos[1], target_pos[2])
        snapped_point = simulator.sim.pathfinder.snap_point(test_point)
        
        if not simulator.sim.pathfinder.is_navigable(snapped_point):
            logging.warning(f"Agent {agent_id}: Forward movement blocked")
            return False
        
        target_pos = np.array([snapped_point.x, snapped_point.y, snapped_point.z])
        return self._animate_movement(agent_id, target_pos, simulator, agent_state)
    
    def _execute_turn(self, agent_id: str, angle: float, turn_left: bool,
                     simulator: HabitatSimulator, agent_state: AgentState) -> bool:
        """执行转向动作"""
        if not angle:
            return True
        
        # 计算转向角度（左转为正，右转为负）
        turn_angle = angle if turn_left else -angle
        
        # 使用平滑转向动画
        return self._animate_rotation(agent_id, turn_angle, simulator, agent_state)
    
    def _animate_movement(self, agent_id: str, target_pos: np.ndarray,
                         simulator: HabitatSimulator, agent_state: AgentState) -> bool:
        """执行平滑移动动画"""
        start_pos = agent_state.position
        distance = np.linalg.norm(target_pos - start_pos)
        
        if distance < 0.01:  # 距离太小，直接到达
            return True
        
        # 计算动画步数
        duration = distance / self.linear_speed
        num_steps = max(10, int(duration / self.time_step))  # 最少10步
        
        for step in range(num_steps + 1):
            t = step / num_steps if num_steps > 0 else 1.0
            
            # 线性插值
            current_pos = start_pos + (target_pos - start_pos) * t
            
            # 更新智能体位置（同时更新虚拟智能体和物理机器人）
            self._update_agent_pose(agent_id, current_pos)
            
            # 在运动过程中生成视频帧（每几步生成一帧）
            if step % max(1, num_steps // 10) == 0:  # 生成约10个中间帧
                try:
                    video_writers = getattr(self, '_current_video_writers', None)
                    if video_writers and agent_id in video_writers:
                        self._write_video_frame(agent_id, video_writers[agent_id])
                except Exception as e:
                    logging.debug(f"Failed to write intermediate frame: {e}")
        
        logging.info(f"Agent {agent_id} moved to {target_pos}")
        return True
    
    def _animate_rotation(self, agent_id: str, angle_deg: float,
                         simulator: HabitatSimulator, agent_state: AgentState) -> bool:
        """执行平滑旋转动画"""
        if abs(angle_deg) < 0.1:
            return True
        
        # 计算动画步数
        duration = abs(angle_deg) / self.angular_speed
        num_steps = max(10, int(duration / self.time_step))  # 最少10步
        
        # 将当前旋转转换为欧拉角
        current_rotation = agent_state.rotation
        quat = mn.Quaternion(mn.Vector3(current_rotation[0], current_rotation[1], current_rotation[2]), current_rotation[3])
        
        # 计算旋转增量
        angle_step = angle_deg / num_steps
        
        for step in range(num_steps):
            # 应用旋转增量
            rotation_delta = mn.Quaternion.rotation(mn.Rad(math.radians(angle_step)), mn.Vector3.y_axis())
            quat = quat * rotation_delta
            
            # 更新智能体旋转
            new_rotation = np.array([quat.vector.x, quat.vector.y, quat.vector.z, quat.scalar])
            self._update_agent_pose(agent_id, agent_state.position, new_rotation)
            
            # 在旋转过程中生成视频帧（每几步生成一帧）
            if step % max(1, num_steps // 15) == 0:  # 生成约15个中间帧
                try:
                    video_writers = getattr(self, '_current_video_writers', None)
                    if video_writers and agent_id in video_writers:
                        self._write_video_frame(agent_id, video_writers[agent_id])
                except Exception as e:
                    logging.debug(f"Failed to write intermediate frame: {e}")
        
        logging.info(f"Agent {agent_id} rotated {angle_deg} degrees")
        return True
    
    def _create_video_writers(self) -> Dict[str, cv2.VideoWriter]:
        """为每个智能体创建视频写入器"""
        video_writers = {}
        
        # 创建输出目录
        output_dir = self.video_config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # 视频参数
        fps = self.video_config["fps"]
        height, width = self.video_config["resolution"]
        
        # 使用更兼容的编码器
        # 首先尝试H.264，如果失败则回退到XVID
        codecs_to_try = ['H264', 'XVID', 'MP4V', 'MJPG']
        
        for agent_id in self.agent_configs_dict.keys():
            output_path = os.path.join(output_dir, f"{agent_id}_output.mp4")
            
            writer = None
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # 测试写入器是否有效
                    if writer.isOpened():
                        logging.info(f"Video writer created for {agent_id} with {codec} codec: {output_path}")
                        break
                    else:
                        writer.release()
                        writer = None
                        logging.warning(f"Failed to create video writer with {codec} codec")
                        
                except Exception as e:
                    logging.warning(f"Error creating video writer with {codec} codec: {e}")
                    if writer:
                        writer.release()
                        writer = None
            
            if writer is None:
                # 如果所有编码器都失败，创建一个回退的写入器
                logging.error(f"Failed to create video writer for {agent_id} with any codec")
                # 尝试不指定编码器
                try:
                    writer = cv2.VideoWriter(output_path, -1, fps, (width, height))
                except Exception as e:
                    logging.error(f"Failed to create fallback video writer: {e}")
                    # 创建一个虚拟写入器以避免错误
                    writer = DummyVideoWriter(output_path)
            
            video_writers[agent_id] = writer
        
        return video_writers
    
    def _write_initial_frames(self, video_writers: Dict[str, cv2.VideoWriter]):
        """写入初始帧"""
        for agent_id, writer in video_writers.items():
            self._write_video_frame(agent_id, writer)
    
    def _write_video_frame(self, agent_id: str, video_writer: cv2.VideoWriter):
        """为指定智能体写入视频帧"""
        try:
            agent_state = self.agent_states[agent_id]
            
            # 获取FPV图像 - 从物理机器人的传感器位置
            fpv_image = self._get_robot_sensor_observation(agent_id, agent_state)
            
            # 验证图像有效性
            if fpv_image is None or fpv_image.size == 0:
                logging.warning(f"Invalid FPV image for {agent_id}, creating black frame")
                # 创建黑色帧
                resolution = self.agent_configs[0]["sensors"]["color_sensor"]["resolution"]
                fpv_image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
                
            # 确保图像是RGB格式
            if len(fpv_image.shape) == 3:
                if fpv_image.shape[2] == 4:
                    # RGBA格式，转换为RGB
                    fpv_pil = Image.fromarray(fpv_image[..., :3].astype(np.uint8), "RGB")
                elif fpv_image.shape[2] == 3:
                    # RGB格式
                    fpv_pil = Image.fromarray(fpv_image.astype(np.uint8), "RGB")
                else:
                    logging.warning(f"Unexpected image channels for {agent_id}: {fpv_image.shape[2]}")
                    resolution = self.agent_configs[0]["sensors"]["color_sensor"]["resolution"]
                    fpv_image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
                    fpv_pil = Image.fromarray(fpv_image, "RGB")
            else:
                logging.warning(f"Unexpected image format for {agent_id}: {fpv_image.shape}")
                resolution = self.agent_configs[0]["sensors"]["color_sensor"]["resolution"]
                fpv_image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
                fpv_pil = Image.fromarray(fpv_image, "RGB")
            
            # 创建带智能体标记的地图
            map_with_agent = self._create_map_with_agent_marker(agent_id, agent_state)
            
            # 创建分屏视频帧
            frame = self._create_split_screen_frame(fpv_pil, map_with_agent)
            
            # 验证帧尺寸
            expected_height, expected_width = self.video_config["resolution"]
            if frame.size != (expected_width, expected_height):
                logging.warning(f"Frame size mismatch for {agent_id}: expected {expected_width}x{expected_height}, got {frame.size}")
                frame = frame.resize((expected_width, expected_height), Image.Resampling.LANCZOS)
            
            # 转换为OpenCV格式并写入
            frame_array = np.array(frame)
            if frame_array.shape[2] == 3:  # RGB
                frame_cv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                # 检查是否是虚拟写入器
                if hasattr(video_writer, 'frame_count'):
                    # 虚拟写入器
                    video_writer.write(frame_cv)
                elif video_writer.isOpened():
                    # 真实写入器
                    video_writer.write(frame_cv)
                    logging.debug(f"Video frame written for {agent_id}")
                else:
                    logging.warning(f"Video writer not opened for {agent_id}")
            else:
                logging.warning(f"Invalid frame format for {agent_id}: {frame_array.shape}")
            
        except Exception as e:
            logging.error(f"Failed to write video frame for {agent_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _create_map_with_agent_marker(self, agent_id: str, agent_state: AgentState) -> Image.Image:
        """创建带有智能体位置标记的地图"""
        map_image = self.clean_map.copy()
        draw = ImageDraw.Draw(map_image)
        
        # 转换世界坐标到地图坐标
        map_x, map_y = self.simulator.world_to_map_coords(agent_state.position)
        
        # 绘制智能体位置标记
        marker_size = self.map_config["agent_marker_size"]
        marker_color = tuple(self.map_config["agent_marker_color"])
        
        # 绘制圆形标记
        draw.ellipse([
            map_x - marker_size, map_y - marker_size,
            map_x + marker_size, map_y + marker_size
        ], fill=marker_color, outline=(255, 255, 255), width=2)
        
        # 绘制方向箭头
        arrow_length = self.map_config["direction_arrow_length"]
        rotation = agent_state.rotation
        quat = mn.Quaternion(mn.Vector3(rotation[0], rotation[1], rotation[2]), rotation[3])
        forward_dir = quat.transform_vector(mn.Vector3(0, 0, -1))
        
        arrow_end_x = map_x + forward_dir.x * arrow_length
        arrow_end_y = map_y + forward_dir.z * arrow_length
        
        draw.line([(map_x, map_y), (arrow_end_x, arrow_end_y)], 
                 fill=marker_color, width=3)
        
        # 绘制箭头头部
        arrow_head_size = 5
        angle = math.atan2(forward_dir.z, forward_dir.x)
        
        for offset in [-0.5, 0.5]:
            head_angle = angle + offset * math.pi
            head_x = arrow_end_x - arrow_head_size * math.cos(head_angle)
            head_y = arrow_end_y - arrow_head_size * math.sin(head_angle)
            draw.line([(arrow_end_x, arrow_end_y), (head_x, head_y)], 
                     fill=marker_color, width=2)
        
        # 添加智能体ID标签
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw.text((map_x + marker_size + 5, map_y - marker_size), 
                 agent_id, fill=(255, 255, 255), font=font)
        
        return map_image
    
    def _create_split_screen_frame(self, fpv_image: Image.Image, map_image: Image.Image) -> Image.Image:
        """创建分屏视频帧（左侧FPV，右侧地图）"""
        height, width = self.video_config["resolution"]
        half_width = width // 2
        
        # 创建新画布
        frame = Image.new('RGB', (width, height), (0, 0, 0))
        
        # 调整FPV图像大小并放置在左侧
        fpv_resized = fpv_image.resize((half_width, height), Image.Resampling.LANCZOS)
        frame.paste(fpv_resized, (0, 0))
        
        # 调整地图图像大小并放置在右侧
        map_resized = map_image.resize((half_width, height), Image.Resampling.LANCZOS)
        frame.paste(map_resized, (half_width, 0))
        
        return frame
    
    def _close_video_writers(self, video_writers: Dict[str, Any]):
        """关闭所有视频写入器"""
        for agent_id, writer in video_writers.items():
            try:
                writer.release()
                logging.info(f"Video writer closed for {agent_id}")
            except Exception as e:
                logging.warning(f"Error closing video writer for {agent_id}: {e}")
    
    def _save_current_states(self):
        """保存当前所有智能体状态"""
        try:
            states_data = {}
            for agent_id, agent_state in self.agent_states.items():
                states_data[agent_id] = agent_state.to_dict()
            
            state_file = self.config["state_persistence"]["state_file"]
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(states_data, f, indent=2)
            
            logging.info(f"Agent states saved to {state_file}")
            
        except Exception as e:
            logging.error(f"Failed to save agent states: {e}")
    
    def load_states_from_file(self, state_file: str) -> bool:
        """从文件加载智能体状态"""
        try:
            with open(state_file, 'r') as f:
                states_data = json.load(f)
            
            for agent_id, state_data in states_data.items():
                if agent_id in self.agent_states:
                    restored_state = AgentState.from_dict(state_data)
                    self.agent_states[agent_id] = restored_state
                    
                    # 在模拟器中恢复位置（同时更新虚拟智能体和物理机器人）
                    self._update_agent_pose(agent_id, restored_state.position, restored_state.rotation)
                    
                    logging.info(f"Agent {agent_id} state restored from {state_file}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load agent states from {state_file}: {e}")
            return False
    
    def close(self):
        """关闭模拟器"""
        if self.simulator:
            try:
                self.simulator.close()
                logging.info("Simulator closed")
            except Exception as e:
                # OpenGL上下文关闭错误通常不影响核心功能
                if "GL::Context::current()" in str(e):
                    logging.warning(f"OpenGL context warning during cleanup (non-critical)")
                else:
                    logging.warning(f"Warning during cleanup: {e}")
        else:
            logging.info("No simulator to close")
    
    def get_agent_status_report(self) -> Dict[str, Any]:
        """获取所有智能体的状态报告"""
        report = {
            "total_agents": len(self.agent_configs),
            "simulator_status": "active" if self.simulator else "inactive",
            "agents": {}
        }
        
        for agent_id in self.agent_configs_dict.keys():
            agent_config = self.agent_configs_dict[agent_id]
            agent_state = self.agent_states.get(agent_id)
            robot_obj = self.agent_robots.get(agent_id)
            
            agent_report = {
                "id": agent_id,
                "has_physical_robot": robot_obj is not None,
                "model_path": agent_config.get("agent_model_path"),
                "current_position": agent_state.position.tolist() if agent_state else None,
                "current_rotation": agent_state.rotation.tolist() if agent_state else None,
            }
            
            if robot_obj:
                try:
                    # 安全地获取机器人属性
                    robot_id = getattr(robot_obj, 'object_id', 'N/A')
                    joint_count = len(getattr(robot_obj, 'joint_positions', []))
                    
                    # 安全地获取机器人位置
                    try:
                        robot_pos = robot_obj.translation
                        robot_position = [float(robot_pos.x), float(robot_pos.y), float(robot_pos.z)]
                    except Exception:
                        robot_position = "N/A"
                    
                    agent_report.update({
                        "robot_object_id": robot_id,
                        "robot_joint_count": joint_count,
                        "robot_position": robot_position,
                        "robot_status": "active"
                    })
                except Exception as e:
                    agent_report.update({
                        "robot_object_id": "error",
                        "robot_joint_count": "error",
                        "robot_position": "error",
                        "robot_status": f"error: {e}"
                    })
            else:
                agent_report.update({
                    "robot_object_id": None,
                    "robot_joint_count": None,
                    "robot_position": None,
                    "robot_status": "virtual agent only"
                })
            
            report["agents"][agent_id] = agent_report
        
        return report
    
    def print_agent_status(self):
        """打印所有智能体的状态"""
        report = self.get_agent_status_report()
        
        print("\n" + "="*60)
        print("MULTI-AGENT SYSTEM STATUS")
        print("="*60)
        print(f"Total agents: {report['total_agents']}")
        print(f"Simulator: {report['simulator_status']}")
        print()
        
        for agent_id, agent_info in report["agents"].items():
            print(f"Agent: {agent_id}")
            print(f"  Physical robot: {'✓ Yes' if agent_info['has_physical_robot'] else '→ No (virtual agent)'}")
            if agent_info["model_path"]:
                print(f"  Model: {os.path.basename(agent_info['model_path'])}")
            if agent_info["has_physical_robot"]:
                print(f"  Robot ID: {agent_info.get('robot_object_id', 'N/A')}")
                print(f"  Joints: {agent_info.get('robot_joint_count', 'N/A')}")
                print(f"  Status: {agent_info.get('robot_status', 'N/A')}")
            print()
        
        print("="*60)
    
    def _update_agent_pose(self, agent_id: str, position: np.ndarray, rotation: Optional[np.ndarray] = None):
        """更新智能体位置和姿态，同时同步虚拟智能体和物理机器人"""
        # 更新智能体状态
        agent_state = self.agent_states[agent_id]
        agent_state.position = position
        if rotation is not None:
            agent_state.rotation = rotation
        
        # 更新虚拟智能体
        # 安全地处理rotation用于虚拟智能体
        virtual_rotation = agent_state.rotation
        if hasattr(virtual_rotation, '__len__') and len(virtual_rotation) == 4:
            # 将numpy数组转换为Quaternion
            quat_array = np.array(virtual_rotation)
            quat_norm = np.linalg.norm(quat_array)
            if quat_norm > 0:
                quat_array = quat_array / quat_norm
            virtual_rotation = mn.Quaternion(
                mn.Vector3(quat_array[0], quat_array[1], quat_array[2]),
                quat_array[3]
            )
        elif not hasattr(virtual_rotation, 'vector'):
            # 如果不是有效的Quaternion，使用默认
            virtual_rotation = mn.Quaternion()
        
        self.simulator.move_agent_to(position, virtual_rotation)
        
        # 如果有物理机器人，同步更新
        if agent_id in self.agent_robots:
            try:
                robot_obj = self.agent_robots[agent_id]
                
                # 构建变换矩阵 - 安全地处理rotation
                if rotation is not None and hasattr(rotation, '__len__') and len(rotation) == 4:
                    # rotation是quaternion [x, y, z, w]
                    quat_array = np.array(rotation)
                    quat_norm = np.linalg.norm(quat_array)
                    if quat_norm > 0:
                        quat_array = quat_array / quat_norm
                    quat = mn.Quaternion(mn.Vector3(quat_array[0], quat_array[1], quat_array[2]), quat_array[3])
                    transform = mn.Matrix4.from_(quat.to_matrix(), mn.Vector3(position))
                elif hasattr(rotation, 'to_matrix'):
                    # rotation已经是Quaternion对象
                    transform = mn.Matrix4.from_(rotation.to_matrix(), mn.Vector3(position))
                else:
                    # 只有位置变换
                    transform = mn.Matrix4.translation(mn.Vector3(position))
                
                robot_obj.transformation = transform
                
            except Exception as e:
                logging.warning(f"Failed to sync robot {agent_id}: {e}")
    
    def _get_robot_transform_matrix(self, robot_obj) -> Optional[np.ndarray]:
        """安全地获取机器人的4x4变换矩阵"""
        try:
            # 方法1：尝试直接从transformation属性获取
            transform_matrix = robot_obj.transformation
            
            # 尝试不同的方法来提取矩阵数据
            try:
                # 方法1：使用to_numpy()方法（如果存在）
                if hasattr(transform_matrix, 'to_numpy'):
                    return transform_matrix.to_numpy()
                
                # 方法2：手动提取矩阵元素
                elif hasattr(transform_matrix, '__getitem__'):
                    # Magnum Matrix4是4x4矩阵，按行主序或列主序存储
                    matrix_data = np.zeros((4, 4))
                    for i in range(4):
                        for j in range(4):
                            matrix_data[i, j] = transform_matrix[i, j]
                    return matrix_data
                
                # 方法3：尝试直接转换为numpy数组
                else:
                    matrix_array = np.array(transform_matrix)
                    if matrix_array.shape == (4, 4):
                        return matrix_array
                    elif matrix_array.shape == (16,):
                        # 假设是列主序存储，需要reshape并转置
                        return matrix_array.reshape(4, 4).T
                    else:
                        logging.warning(f"Unexpected matrix shape: {matrix_array.shape}")
                        return None
                        
            except Exception as matrix_extract_error:
                logging.warning(f"Failed to extract matrix data: {matrix_extract_error}")
                
                # 方法4：手动构建变换矩阵（从位置和旋转）
                try:
                    position = robot_obj.translation
                    rotation = robot_obj.rotation
                    
                    # 构建变换矩阵
                    transform = np.eye(4)
                    
                    # 设置旋转部分（3x3子矩阵）
                    if hasattr(rotation, 'to_matrix'):
                        rot_matrix = np.array(rotation.to_matrix())
                        if rot_matrix.shape == (3, 3):
                            transform[:3, :3] = rot_matrix
                    
                    # 设置平移部分
                    if hasattr(position, 'x') and hasattr(position, 'y') and hasattr(position, 'z'):
                        transform[0, 3] = position.x
                        transform[1, 3] = position.y
                        transform[2, 3] = position.z
                    
                    return transform
                    
                except Exception as manual_build_error:
                    logging.error(f"Failed to manually build transform matrix: {manual_build_error}")
                    return None
        
        except Exception as e:
            logging.error(f"Failed to get robot transform matrix: {e}")
            return None
    
    def _get_robot_camera_transform(self, agent_id: str, robot_transform: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """计算机器人摄像头的位置和朝向"""
        try:
            # 获取智能体配置中的URDF路径
            agent_config = self.agent_configs_dict.get(agent_id)
            if not agent_config:
                return None, None
            
            urdf_path = agent_config.get("agent_model_path")
            if not urdf_path or not os.path.exists(urdf_path):
                return None, None
            
            # 创建URDF解析器
            urdf_parser = URDFCameraParser(urdf_path)
            
            # 获取摄像头的位置和朝向
            position, orientation = urdf_parser.get_camera_position_and_orientation(robot_transform)
            
            return position, orientation
            
        except Exception as e:
            logging.warning(f"Failed to get robot camera transform for {agent_id}: {e}")
            return None, None
    
    def _get_robot_sensor_observation(self, agent_id: str, agent_state: AgentState) -> np.ndarray:
        """从物理机器人的传感器获取第一人称观察"""
        try:
            # 获取物理机器人对象
            robot_obj = self.agent_robots.get(agent_id)
            if robot_obj is None:
                logging.warning(f"No physical robot for {agent_id}, using fallback")
                return self._get_virtual_agent_observation(agent_id, agent_state)
            
            # 获取机器人的变换矩阵
            robot_transform_matrix = self._get_robot_transform_matrix(robot_obj)
            if robot_transform_matrix is None:
                logging.warning(f"Failed to get robot transform matrix for {agent_id}")
                return self._get_robot_sensor_observation_fallback(agent_id, robot_obj)
            
            # 获取摄像头的位置和朝向
            camera_position, camera_orientation = self._get_robot_camera_transform(agent_id, robot_transform_matrix)
            
            if camera_position is None or camera_orientation is None:
                logging.warning(f"Failed to calculate camera transform for {agent_id}")
                return self._get_robot_sensor_observation_fallback(agent_id, robot_obj)
            
            # 设置虚拟智能体到摄像头位置并获取观察
            agent_node = self.simulator.get_agent(0)._body.object  # 获取默认智能体
            
            # 保存原始位置和朝向
            original_translation = agent_node.translation
            original_rotation = agent_node.rotation
            
            try:
                # 临时设置智能体位置到摄像头位置
                agent_node.translation = mn.Vector3(camera_position[0], camera_position[1], camera_position[2])
                
                # 设置朝向（quaternion转换为rotation）
                quat = mn.Quaternion(mn.Vector3(camera_orientation[0], camera_orientation[1], camera_orientation[2]), camera_orientation[3])
                agent_node.rotation = quat
                
                # 获取观察
                observations = self.simulator.sim.get_sensor_observations()
                color_obs = observations.get("color_sensor", observations.get("rgb", None))
                
                if color_obs is not None:
                    # 确保图像格式正确
                    if len(color_obs.shape) == 3 and color_obs.shape[2] == 3:
                        # RGB format
                        fpv_image = color_obs
                    elif len(color_obs.shape) == 3 and color_obs.shape[2] == 4:
                        # RGBA format, 移除alpha通道
                        fpv_image = color_obs[:, :, :3]
                    else:
                        # 异常格式，创建默认图像
                        fpv_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        logging.warning(f"Unexpected observation format for {agent_id}: {color_obs.shape}")
                    
                    logging.info(f"✓ Got robot sensor observation for {agent_id}: {fpv_image.shape}")
                    return fpv_image
                else:
                    logging.warning(f"No color observation available for {agent_id}")
                    return self._get_robot_sensor_observation_fallback(agent_id, robot_obj)
                    
            finally:
                # 恢复原始位置和朝向
                agent_node.translation = original_translation
                agent_node.rotation = original_rotation
                
        except Exception as e:
            logging.error(f"Failed to get robot sensor observation for {agent_id}: {e}")
            logging.error(traceback.format_exc())
            return self._get_robot_sensor_observation_fallback(agent_id, robot_obj)
    
    def _get_robot_sensor_observation_fallback(self, agent_id: str, robot_obj) -> np.ndarray:
        """回退方法：使用简单的头部偏移来估计摄像头位置"""
        try:
            # 获取机器人的当前位置和朝向
            robot_position = robot_obj.translation
            robot_rotation = robot_obj.rotation
            
            # 将Magnum对象转换为numpy
            base_position = np.array([robot_position.x, robot_position.y, robot_position.z])
            
            # 估计头部摄像头相对于机器人基座的偏移
            # 对于Fetch机器人，摄像头大约在基座上方1.3米，前方0.1米
            head_offset_local = np.array([0.1, 1.3, 0.0])  # [前方, 上方, 右方]
            
            # 将局部偏移转换到世界坐标系
            if hasattr(robot_rotation, 'transform_vector'):
                head_offset_world = robot_rotation.transform_vector(mn.Vector3(head_offset_local))
                head_offset_world = np.array([head_offset_world.x, head_offset_world.y, head_offset_world.z])
            else:
                # 如果无法获取旋转，假设机器人朝向前方
                head_offset_world = head_offset_local
            
            # 计算摄像头位置
            camera_position = base_position + head_offset_world
            
            # 设置智能体状态并获取观察
            sensor_agent_state = habitat_sim.AgentState()
            sensor_agent_state.position = mn.Vector3(camera_position)
            
            # 使用机器人的朝向作为摄像头朝向
            if hasattr(robot_rotation, 'vector') and hasattr(robot_rotation, 'scalar'):
                vec = robot_rotation.vector
                sensor_orientation = np.array([vec.x, vec.y, vec.z, robot_rotation.scalar], dtype=np.float32)
                quat_norm = np.linalg.norm(sensor_orientation)
                if quat_norm > 0:
                    sensor_orientation = sensor_orientation / quat_norm
                sensor_agent_state.rotation = sensor_orientation
            else:
                # 使用默认朝向
                sensor_agent_state.rotation = np.array([0, 0, 0, 1], dtype=np.float32)
            
            # 临时设置智能体状态并获取观察
            original_state = self.simulator.agent.get_state()
            self.simulator.agent.set_state(sensor_agent_state)
            observation = self.simulator.get_fpv_observation()
            self.simulator.agent.set_state(original_state)
            
            logging.info(f"Got fallback robot sensor observation for {agent_id} from estimated head position: {camera_position}")
            return observation
            
        except Exception as e:
            logging.error(f"Failed to get fallback robot sensor observation for {agent_id}: {e}")
            logging.debug(traceback.format_exc())
            
            # 最终回退：使用智能体状态
            agent_state = self.agent_states.get(agent_id)
            if agent_state:
                return self._get_virtual_agent_observation(agent_state)
            else:
                # 返回黑色图像
                resolution = self.agent_configs[0]["sensors"]["color_sensor"]["resolution"]
                return np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
    

class URDFCameraParser:
    """解析URDF文件以获取机器人头部摄像头的真实位置和朝向"""
    
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        self.joint_transforms = {}
        self._parse_joints()
    
    def _parse_joints(self):
        """解析所有关节的变换"""
        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0').split()
                rpy = origin.get('rpy', '0 0 0').split()
                translation = np.array([float(x) for x in xyz])
                rotation = np.array([float(x) for x in rpy])
                
                # 将RPY转换为旋转矩阵
                rot_matrix = self._rpy_to_rotation_matrix(rotation)
                
                # 构建4x4变换矩阵
                transform = np.eye(4)
                transform[:3, :3] = rot_matrix
                transform[:3, 3] = translation
                
                self.joint_transforms[joint_name] = {
                    'transform': transform,
                    'parent': joint.find('parent').get('link') if joint.find('parent') is not None else None,
                    'child': joint.find('child').get('link') if joint.find('child') is not None else None
                }
    
    def _rpy_to_rotation_matrix(self, rpy):
        """将RPY角度转换为旋转矩阵"""
        roll, pitch, yaw = rpy
        
        # 旋转矩阵 = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def get_camera_transform_chain(self, camera_link='head_camera_rgb_frame'):
        """获取从base_link到摄像头的完整变换链"""
        # 定义Fetch机器人的运动链：base_link -> torso_lift_link -> head_pan_link -> head_tilt_link -> head_camera_link -> head_camera_rgb_frame
        chain = [
            'torso_lift_joint',    # base_link -> torso_lift_link
            'head_pan_joint',      # torso_lift_link -> head_pan_link  
            'head_tilt_joint',     # head_pan_link -> head_tilt_link
            'head_camera_joint',   # head_tilt_link -> head_camera_link
            'head_camera_rgb_joint' # head_camera_link -> head_camera_rgb_frame
        ]
        
        # 计算累积变换
        cumulative_transform = np.eye(4)
        for joint_name in chain:
            if joint_name in self.joint_transforms:
                cumulative_transform = cumulative_transform @ self.joint_transforms[joint_name]['transform']
            else:
                logging.warning(f"Joint '{joint_name}' not found in URDF")
        
        return cumulative_transform
    
    def get_camera_position_and_orientation(self, robot_transform):
        """
        获取摄像头在世界坐标系中的位置和朝向
        
        Args:
            robot_transform: 机器人base_link在世界坐标系中的4x4变换矩阵
            
        Returns:
            tuple: (position, orientation_quaternion)
        """
        try:
            # 获取从base_link到摄像头的变换
            camera_local_transform = self.get_camera_transform_chain()
            
            # 计算摄像头在世界坐标系中的变换
            camera_world_transform = robot_transform @ camera_local_transform
            
            # 提取位置
            position = camera_world_transform[:3, 3]
            
            # 提取旋转矩阵并转换为四元数
            rotation_matrix = camera_world_transform[:3, :3]
            
            try:
                # 尝试使用scipy进行旋转矩阵到四元数的转换
                from scipy.spatial.transform import Rotation as R
                rotation = R.from_matrix(rotation_matrix)
                orientation_quaternion = rotation.as_quat()  # [x, y, z, w]
            except ImportError:
                # 如果scipy不可用，使用手动计算四元数的方法
                logging.warning("scipy not available, using manual quaternion calculation")
                orientation_quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
            
            return position, orientation_quaternion
            
        except Exception as e:
            logging.error(f"Failed to get camera position and orientation: {e}")
            return None, None
    
    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        """手动将旋转矩阵转换为四元数 [x, y, z, w]"""
        try:
            # 从旋转矩阵计算四元数的Shepperd方法
            R = rotation_matrix
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            # 返回[x, y, z, w]格式
            quaternion = np.array([qx, qy, qz, qw])
            
            # 归一化四元数
            quat_norm = np.linalg.norm(quaternion)
            if quat_norm > 0:
                quaternion = quaternion / quat_norm
            
            return quaternion
            
        except Exception as e:
            logging.error(f"Failed to convert rotation matrix to quaternion: {e}")
            # 返回单位四元数
            return np.array([0, 0, 0, 1])
