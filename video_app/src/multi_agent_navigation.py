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
import traceback
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

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


class AdvancedCollisionDetector:
    """高级碰撞检测器 - 支持动态碰撞检测和物理引擎集成"""
    
    def __init__(self, config: Dict):
        self.enabled = config.get("enabled", True)
        self.agent_radius = config.get("agent_radius", 0.1)
        self.height_threshold = config.get("height_threshold", 0.3)
        self.prediction_steps = config.get("prediction_steps", 10)  # 增加预测步数
        self.min_agent_distance = config.get("min_agent_distance", 0.4)
        self.raycast_steps = config.get("raycast_steps", 20)  # 射线检测步数
        self.contact_threshold = config.get("contact_threshold", 0.01)  # 接触阈值
        self.physics_enabled = False  # 物理引擎状态标记
        
        # 射线投射配置
        self.ray_directions = self._generate_ray_directions()
        
        # 碰撞历史记录
        self.collision_history = {}
        self.contact_history = []
        
        logging.info(f"Advanced collision detector initialized with {len(self.ray_directions)} ray directions")
    
    def _generate_ray_directions(self) -> List[mn.Vector3]:
        """生成360度射线方向用于环境碰撞检测"""
        directions = []
        
        # 水平方向的射线（8个方向）
        for i in range(8):
            angle = i * 2 * math.pi / 8
            directions.append(mn.Vector3(math.cos(angle), 0, math.sin(angle)))
        
        # 斜向上的射线（8个方向）
        for i in range(8):
            angle = i * 2 * math.pi / 8
            directions.append(mn.Vector3(math.cos(angle) * 0.7, 0.7, math.sin(angle) * 0.7))
        
        # 斜向下的射线（8个方向）
        for i in range(8):
            angle = i * 2 * math.pi / 8
            directions.append(mn.Vector3(math.cos(angle) * 0.7, -0.7, math.sin(angle) * 0.7))
        
        return directions
    
    def set_physics_enabled(self, enabled: bool):
        """设置物理引擎状态"""
        self.physics_enabled = enabled
        logging.info(f"Physics engine enabled: {enabled}")
    
    def check_physics_contacts(self, sim: habitat_sim.Simulator, 
                              agent_robot_objects: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """使用物理引擎检查接触点碰撞"""
        if not self.enabled or not self.physics_enabled:
            return False, []
        
        try:
            # 获取物理接触点
            contact_points = sim.get_physics_contact_points()
            
            # 记录当前帧的接触历史
            self.contact_history.append({
                'timestamp': time.time(),
                'contact_count': len(contact_points),
                'contacts': contact_points
            })
            
            # 保持历史记录在合理范围内
            if len(self.contact_history) > 100:
                self.contact_history = self.contact_history[-50:]
            
            collision_agents = []
            active_collisions = False
            
            for contact in contact_points:
                # 检查接触是否活跃且距离小于阈值
                if contact.is_active and abs(contact.contact_distance) < self.contact_threshold:
                    active_collisions = True
                    
                    # 检查是否涉及我们的智能体机器人
                    for agent_id, robot_obj in agent_robot_objects.items():
                        if hasattr(robot_obj, 'object_id'):
                            if (contact.object_id_a == robot_obj.object_id or 
                                contact.object_id_b == robot_obj.object_id):
                                if agent_id not in collision_agents:
                                    collision_agents.append(agent_id)
                                    
                                    # 记录碰撞详情
                                    logging.debug(f"Physics collision detected for {agent_id}: "
                                                f"distance={contact.contact_distance:.4f}, "
                                                f"force={contact.normal_force:.4f}")
            
            return active_collisions, collision_agents
            
        except Exception as e:
            logging.warning(f"Physics contact check failed: {e}")
            return False, []
    
    def check_environment_collision_with_raycast(self, sim: habitat_sim.Simulator, 
                                                position: np.ndarray, 
                                                direction: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """使用射线投射检查环境碰撞"""
        if not self.enabled:
            return False, ""
        
        try:
            collisions_detected = []
            ray_origin = mn.Vector3(position[0], position[1] + 0.1, position[2])  # 稍微抬高起点
            
            # 如果指定了方向，只检查该方向
            if direction is not None:
                test_directions = [mn.Vector3(direction[0], direction[1], direction[2])]
            else:
                test_directions = self.ray_directions
            
            for ray_dir in test_directions:
                try:
                    # 创建射线
                    ray = habitat_sim.geo.Ray(ray_origin, ray_dir)
                    
                    # 投射射线，检查距离在智能体半径范围内
                    raycast_results = sim.cast_ray(ray, max_distance=self.agent_radius * 1.5)
                    
                    if raycast_results.has_hits():
                        hit_info = raycast_results.hits[0]  # 获取最近的碰撞
                        if hit_info.ray_distance < self.agent_radius:
                            collisions_detected.append(f"raycast-{hit_info.ray_distance:.3f}m")
                            
                except Exception as e:
                    logging.debug(f"Raycast failed for direction {ray_dir}: {e}")
                    continue
            
            if collisions_detected:
                return True, f"Environment collision: {', '.join(collisions_detected)}"
            
            # 回退到导航网格检查
            return self._check_navmesh_collision(sim, position)
            
        except Exception as e:
            logging.warning(f"Raycast collision check failed: {e}")
            return self._check_navmesh_collision(sim, position)
    
    def _check_navmesh_collision(self, sim: habitat_sim.Simulator, 
                                position: np.ndarray) -> Tuple[bool, str]:
        """使用导航网格检查碰撞（回退方案）"""
        try:
            test_point = mn.Vector3(position[0], position[1], position[2])
            
            # 检查是否可导航
            is_navigable = sim.pathfinder.is_navigable(test_point)
            
            if not is_navigable:
                # 尝试捕捉到最近的可导航点
                snapped_point = sim.pathfinder.snap_point(test_point)
                snap_distance = np.linalg.norm(
                    np.array([test_point.x, test_point.y, test_point.z]) - 
                    np.array([snapped_point.x, snapped_point.y, snapped_point.z])
                )
                
                if snap_distance > self.agent_radius:
                    return True, f"NavMesh collision: snap distance {snap_distance:.3f}m"
            
            # 检查周围区域
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                check_x = position[0] + self.agent_radius * 0.9 * np.cos(angle)
                check_z = position[2] + self.agent_radius * 0.9 * np.sin(angle)
                check_point = mn.Vector3(check_x, position[1], check_z)
                
                if not sim.pathfinder.is_navigable(check_point):
                    return True, f"NavMesh collision: surrounding area blocked"
            
            return False, ""
            
        except Exception as e:
            logging.warning(f"NavMesh collision check failed: {e}")
            return True, f"NavMesh check error: {str(e)}"
    
    def check_dynamic_path_collision(self, sim: habitat_sim.Simulator,
                                   start_pos: np.ndarray, 
                                   end_pos: np.ndarray,
                                   agent_robot_objects: Dict[str, Any],
                                   current_agent_id: str) -> Tuple[bool, str]:
        """检查路径上的动态碰撞"""
        if not self.enabled:
            return False, ""
        
        try:
            # 计算路径向量
            path_vector = end_pos - start_pos
            path_length = np.linalg.norm(path_vector)
            
            if path_length < 0.01:  # 路径太短
                return False, ""
            
            # 归一化路径方向
            path_direction = path_vector / path_length
            
            # 沿路径检查多个点
            collision_details = []
            
            for step in range(self.raycast_steps + 1):
                t = step / self.raycast_steps
                check_pos = start_pos + t * path_vector
                
                # 1. 检查环境碰撞（使用方向性射线投射）
                env_collision, env_reason = self.check_environment_collision_with_raycast(
                    sim, check_pos, path_direction
                )
                if env_collision:
                    collision_details.append(f"step {step}: {env_reason}")
                
                # 2. 检查与其他智能体的碰撞
                agent_collision, agent_reason = self._check_agent_collision_at_position(
                    check_pos, agent_robot_objects, current_agent_id, sim
                )
                if agent_collision:
                    collision_details.append(f"step {step}: {agent_reason}")
            
            if collision_details:
                return True, f"Dynamic path collision: {'; '.join(collision_details[:3])}"  # 只报告前3个
            
            return False, ""
            
        except Exception as e:
            logging.warning(f"Dynamic path collision check failed: {e}")
            return False, ""
    
    def _check_agent_collision_at_position(self, position: np.ndarray,
                                         agent_robot_objects: Dict[str, Any],
                                         current_agent_id: str,
                                         sim: habitat_sim.Simulator) -> Tuple[bool, str]:
        """检查特定位置与其他智能体的碰撞"""
        try:
            collision_agents = []
            
            for other_agent_id, robot_obj in agent_robot_objects.items():
                if other_agent_id == current_agent_id:
                    continue
                
                # 获取其他智能体的当前位置
                try:
                    if hasattr(robot_obj, 'translation'):
                        other_pos = robot_obj.translation
                        other_position = np.array([other_pos.x, other_pos.y, other_pos.z])
                    else:
                        continue  # 无法获取位置，跳过
                    
                    # 计算距离
                    distance = np.linalg.norm(position[[0, 2]] - other_position[[0, 2]])
                    
                    if distance < self.min_agent_distance:
                        collision_agents.append(f"{other_agent_id}({distance:.2f}m)")
                        
                except Exception as e:
                    logging.debug(f"Failed to get position for {other_agent_id}: {e}")
                    continue
            
            if collision_agents:
                return True, f"Agent collision with: {', '.join(collision_agents)}"
            
            return False, ""
            
        except Exception as e:
            logging.warning(f"Agent collision check failed: {e}")
            return False, ""
    
    def predict_comprehensive_collision(self, sim: habitat_sim.Simulator,
                                      agent_positions: Dict[str, np.ndarray],
                                      planned_movements: Dict[str, np.ndarray],
                                      agent_robot_objects: Dict[str, Any]) -> Tuple[bool, str]:
        """全面的碰撞预测，结合多种检测方法"""
        if not self.enabled:
            return False, ""
        
        try:
            all_collision_reports = []
            
            # 1. 检查当前物理接触点
            if self.physics_enabled:
                has_physics_collision, physics_agents = self.check_physics_contacts(sim, agent_robot_objects)
                if has_physics_collision:
                    all_collision_reports.append(f"Physics contacts: {', '.join(physics_agents)}")
            
            # 2. 为每个智能体检查计划的移动路径
            for agent_id, current_pos in agent_positions.items():
                if agent_id not in planned_movements:
                    continue
                
                movement = planned_movements[agent_id]
                if np.linalg.norm(movement) < 0.01:  # 移动太小
                    continue
                
                target_pos = current_pos + movement
                
                # 动态路径碰撞检查
                path_collision, path_reason = self.check_dynamic_path_collision(
                    sim, current_pos, target_pos, agent_robot_objects, agent_id
                )
                
                if path_collision:
                    all_collision_reports.append(f"{agent_id}: {path_reason}")
            
            # 3. 检查所有智能体的最终位置是否会重叠
            future_positions = {}
            for agent_id, current_pos in agent_positions.items():
                if agent_id in planned_movements:
                    future_positions[agent_id] = current_pos + planned_movements[agent_id]
                else:
                    future_positions[agent_id] = current_pos
            
            final_collision, final_agents = self._check_final_position_collisions(future_positions)
            if final_collision:
                all_collision_reports.append(f"Final position collisions: {', '.join(final_agents)}")
            
            # 汇总结果
            if all_collision_reports:
                return True, "; ".join(all_collision_reports[:2])  # 只报告前2个最重要的
            
            return False, ""
            
        except Exception as e:
            logging.error(f"Comprehensive collision prediction failed: {e}")
            return False, ""
    
    def _check_final_position_collisions(self, positions: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
        """检查最终位置的智能体间碰撞"""
        if len(positions) < 2:
            return False, []
        
        collision_pairs = []
        agents = list(positions.items())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1_id, pos1 = agents[i]
                agent2_id, pos2 = agents[j]
                
                # 只考虑水平距离
                distance = np.linalg.norm(pos1[[0, 2]] - pos2[[0, 2]])
                
                if distance < self.min_agent_distance:
                    collision_pairs.append(f"{agent1_id}-{agent2_id}({distance:.2f}m)")
        
        return len(collision_pairs) > 0, collision_pairs
    
    def step_physics_with_collision_monitoring(self, sim: habitat_sim.Simulator, 
                                             dt: float,
                                             agent_robot_objects: Dict[str, Any]) -> Tuple[bool, str]:
        """执行物理步进并监控碰撞"""
        if not self.enabled or not self.physics_enabled:
            sim.step_physics(dt)
            return False, ""
        
        try:
            # 执行物理步进
            sim.step_physics(dt)
            
            # 检查步进后的碰撞状态
            has_collision, collision_agents = self.check_physics_contacts(sim, agent_robot_objects)
            
            if has_collision:
                # 获取碰撞摘要
                collision_summary = sim.get_physics_step_collision_summary()
                active_contacts = sim.get_physics_num_active_contact_points()
                
                return True, f"Physics step collision: {len(collision_agents)} agents, {active_contacts} contacts"
            
            return False, ""
            
        except Exception as e:
            logging.warning(f"Physics step with collision monitoring failed: {e}")
            # 即使监控失败，也要确保物理步进执行
            try:
                sim.step_physics(dt)
            except:
                pass
            return False, ""
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """获取碰撞统计信息"""
        if not self.contact_history:
            return {"status": "no_data"}
        
        recent_contacts = self.contact_history[-10:]  # 最近10帧
        
        total_contacts = sum(entry['contact_count'] for entry in recent_contacts)
        avg_contacts = total_contacts / len(recent_contacts) if recent_contacts else 0
        
        return {
            "total_frames": len(self.contact_history),
            "recent_avg_contacts": avg_contacts,
            "last_contact_count": recent_contacts[-1]['contact_count'] if recent_contacts else 0,
            "physics_enabled": self.physics_enabled,
            "detection_enabled": self.enabled
        }
    
    def enable_collision_visualization(self):
        """启用碰撞可视化功能"""
        if hasattr(self.collision_detector, 'enable_visualization'):
            self.collision_detector.enable_visualization = True
            logging.info("Collision visualization enabled")
    
    def get_collision_visualization_data(self) -> Dict[str, Any]:
        """获取碰撞可视化数据"""
        try:
            if hasattr(self.collision_detector, 'contact_history') and self.collision_detector.contact_history:
                recent_contacts = self.collision_detector.contact_history[-5:]  # 最近5帧
                
                visualization_data = {
                    "contact_points": [],
                    "collision_pairs": [],
                    "raycast_hits": []
                }
                
                for frame_data in recent_contacts:
                    for contact in frame_data.get('contacts', []):
                        if hasattr(contact, 'position_on_a_in_ws') and hasattr(contact, 'position_on_b_in_ws'):
                            pos_a = contact.position_on_a_in_ws
                            pos_b = contact.position_on_b_in_ws
                            
                            visualization_data["contact_points"].extend([
                                [pos_a.x, pos_a.y, pos_a.z],
                                [pos_b.x, pos_b.y, pos_b.z]
                            ])
                            
                            # 添加碰撞对信息
                            if hasattr(contact, 'object_id_a') and hasattr(contact, 'object_id_b'):
                                visualization_data["collision_pairs"].append({
                                    "object_a": contact.object_id_a,
                                    "object_b": contact.object_id_b,
                                    "force": getattr(contact, 'normal_force', 0),
                                    "distance": getattr(contact, 'contact_distance', 0)
                                })
                
                return visualization_data
            
            return {"status": "no_data"}
            
        except Exception as e:
            logging.warning(f"Failed to get collision visualization data: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_collision_report(self) -> str:
        """生成详细的碰撞检测报告"""
        try:
            stats = self.collision_detector.get_collision_statistics()
            report_lines = [
                "=" * 50,
                "COLLISION DETECTION REPORT",
                "=" * 50,
                f"Detection Status: {'Enabled' if stats.get('detection_enabled') else 'Disabled'}",
                f"Physics Engine: {'Active' if stats.get('physics_enabled') else 'Inactive'}",
                f"Total Monitoring Frames: {stats.get('total_frames', 0)}",
                f"Recent Average Contacts: {stats.get('recent_avg_contacts', 0):.2f}",
                f"Last Frame Contacts: {stats.get('last_contact_count', 0)}",
                "",
                "Agent Configuration:",
            ]
            
            # 添加智能体信息
            for agent_id, agent_state in self.agent_states.items():
                has_robot = agent_id in self.agent_robots
                report_lines.extend([
                    f"  {agent_id}:",
                    f"    Type: {'Physical Robot' if has_robot else 'Virtual Agent'}",
                    f"    Position: [{agent_state.position[0]:.3f}, {agent_state.position[1]:.3f}, {agent_state.position[2]:.3f}]",
                ])
                
                if has_robot:
                    try:
                        robot_obj = self.agent_robots[agent_id]
                        robot_pos = robot_obj.translation
                        report_lines.append(f"    Robot Position: [{robot_pos.x:.3f}, {robot_pos.y:.3f}, {robot_pos.z:.3f}]")
                    except:
                        report_lines.append(f"    Robot Position: [Error getting position]")
            
            # 添加碰撞检测配置
            report_lines.extend([
                "",
                "Collision Detection Configuration:",
                f"  Agent Radius: {self.collision_detector.agent_radius}m",
                f"  Min Agent Distance: {self.collision_detector.min_agent_distance}m",
                f"  Prediction Steps: {self.collision_detector.prediction_steps}",
                f"  Raycast Steps: {self.collision_detector.raycast_steps}",
                f"  Contact Threshold: {self.collision_detector.contact_threshold}m",
                "=" * 50
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Error generating collision report: {e}"

    def benchmark_collision_detection(self, num_tests: int = 100) -> Dict[str, float]:
        """基准测试碰撞检测性能"""
        if not self.collision_detector.enabled:
            return {"error": "Collision detection disabled"}
        
        import time
        
        # 准备测试数据
        test_positions = {}
        test_movements = {}
        
        for agent_id, agent_state in self.agent_states.items():
            test_positions[agent_id] = agent_state.position.copy()
            test_movements[agent_id] = np.array([0.1, 0.0, 0.1])  # 小的移动向量
        
        results = {
            "total_tests": num_tests,
            "physics_contacts_time": 0.0,
            "raycast_time": 0.0,
            "comprehensive_prediction_time": 0.0,
            "average_contacts_per_test": 0.0
        }
        
        contact_counts = []
        
        try:
            # 测试物理接触点检测
            start_time = time.time()
            for _ in range(num_tests):
                has_collision, agents = self.collision_detector.check_physics_contacts(
                    self.simulator.sim, self.agent_robots
                )
                contact_counts.append(len(agents))
            results["physics_contacts_time"] = (time.time() - start_time) / num_tests
            
            # 测试射线投射
            start_time = time.time()
            test_position = list(test_positions.values())[0]
            for _ in range(num_tests):
                has_collision, reason = self.collision_detector.check_environment_collision_with_raycast(
                    self.simulator.sim, test_position
                )
            results["raycast_time"] = (time.time() - start_time) / num_tests
            
            # 测试综合预测
            start_time = time.time()
            for _ in range(num_tests):
                has_collision, reason = self.collision_detector.predict_comprehensive_collision(
                    self.simulator.sim, test_positions, test_movements, self.agent_robots
                )
            results["comprehensive_prediction_time"] = (time.time() - start_time) / num_tests
            
            results["average_contacts_per_test"] = sum(contact_counts) / len(contact_counts) if contact_counts else 0
            
            logging.info(f"Collision detection benchmark completed: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            return {"error": str(e)}

    def save_collision_debug_data(self, filename: str = "collision_debug.json"):
        """保存碰撞检测调试数据"""
        try:
            debug_data = {
                "timestamp": datetime.now().isoformat(),
                "collision_statistics": self.collision_detector.get_collision_statistics(),
                "agent_states": {
                    agent_id: {
                        "position": agent_state.position.tolist(),
                        "rotation": agent_state.rotation.tolist(),
                        "has_robot": agent_id in self.agent_robots
                    }
                    for agent_id, agent_state in self.agent_states.items()
                },
                "configuration": {
                    "agent_radius": self.collision_detector.agent_radius,
                    "min_agent_distance": self.collision_detector.min_agent_distance,
                    "prediction_steps": self.collision_detector.prediction_steps,
                    "raycast_steps": self.collision_detector.raycast_steps,
                    "physics_enabled": self.collision_detector.physics_enabled
                }
            }
            
            # 添加最近的接触历史
            if hasattr(self.collision_detector, 'contact_history'):
                debug_data["recent_contacts"] = []
                for entry in self.collision_detector.contact_history[-10:]:  # 最近10帧
                    contact_data = {
                        "timestamp": entry.get("timestamp", 0),
                        "contact_count": entry.get("contact_count", 0)
                    }
                    debug_data["recent_contacts"].append(contact_data)
            
            # 保存到文件
            output_dir = self.video_config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(debug_data, f, indent=2)
            
            logging.info(f"Collision debug data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Failed to save collision debug data: {e}")
            return None

    # ...existing code...
