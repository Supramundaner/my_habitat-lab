#!/usr/bin/env python3
"""
Multi-Agent Collision Detection Test Suite
测试多智能体碰撞检测系统的各种场景
"""

import os
import sys
import json
import numpy as np
import math
import logging
import traceback
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_navigation import MultiAgentSimulator, CollisionDetector, AgentState, ActionCommand


@dataclass
class CollisionTestCase:
    """碰撞检测测试用例"""
    name: str
    description: str
    agent_positions: Dict[str, np.ndarray]
    planned_movements: Dict[str, np.ndarray]
    expected_collision: bool
    expected_collision_type: str  # "environment" 或 "agent"


class CollisionDetectionTester:
    """碰撞检测测试器"""
    
    def __init__(self, test_config_path: str):
        """初始化测试器"""
        self.test_config_path = test_config_path
        self.simulator = None
        self.collision_detector = None
        self.test_results = []
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_test_environment(self):
        """设置测试环境"""
        try:
            # 加载测试配置
            with open(self.test_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 创建多智能体模拟器
            self.simulator = MultiAgentSimulator(config)
            self.collision_detector = self.simulator.collision_detector
            
            logging.info("✓ Test environment setup completed")
            logging.info(f"  - Collision detection enabled: {self.collision_detector.enabled}")
            logging.info(f"  - Agent radius: {self.collision_detector.agent_radius}")
            logging.info(f"  - Min agent distance: {self.collision_detector.min_agent_distance}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup test environment: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def create_test_cases(self) -> List[CollisionTestCase]:
        """创建测试用例"""
        test_cases = []
        
        # 获取场景中的一些可导航位置作为测试基础
        try:
            # 获取几个随机的可导航点
            nav_point1 = self.simulator.simulator.sim.pathfinder.get_random_navigable_point()
            nav_point2 = self.simulator.simulator.sim.pathfinder.get_random_navigable_point()
            nav_point3 = self.simulator.simulator.sim.pathfinder.get_random_navigable_point()
            
            # 转换为numpy数组
            base_pos1 = np.array([nav_point1.x, nav_point1.y, nav_point1.z])
            base_pos2 = np.array([nav_point2.x, nav_point2.y, nav_point2.z])
            base_pos3 = np.array([nav_point3.x, nav_point3.y, nav_point3.z])
            
            logging.info(f"Using navigable positions as test base:")
            logging.info(f"  Position 1: {base_pos1}")
            logging.info(f"  Position 2: {base_pos2}")
            logging.info(f"  Position 3: {base_pos3}")
            
        except Exception as e:
            logging.warning(f"Failed to get navigable points, using default positions: {e}")
            # 回退到默认位置
            base_pos1 = np.array([0.0, 0.0, 0.0])
            base_pos2 = np.array([2.0, 0.0, 0.0])
            base_pos3 = np.array([0.0, 0.0, 3.0])
        
        # 测试用例1：智能体之间的碰撞（距离过近）
        # 使用第一个可导航点的附近位置
        close_pos1 = base_pos1.copy()
        close_pos2 = base_pos1.copy()
        close_pos2[0] += 0.5  # X方向偏移0.5米
        
        test_cases.append(CollisionTestCase(
            name="agent_collision_close",
            description="两个智能体距离过近，应该检测到碰撞",
            agent_positions={
                "agent_1": close_pos1,
                "agent_2": close_pos2
            },
            planned_movements={
                "agent_1": np.array([0.0, 0.0, 0.0]),
                "agent_2": np.array([0.0, 0.0, 0.0])
            },
            expected_collision=True,
            expected_collision_type="agent"
        ))
        
        # 测试用例2：智能体之间的碰撞（移动后会碰撞）
        # 使用两个不同的可导航点
        test_cases.append(CollisionTestCase(
            name="agent_collision_movement",
            description="两个智能体移动后会相撞",
            agent_positions={
                "agent_1": base_pos1.copy(),
                "agent_2": base_pos2.copy()
            },
            planned_movements={
                "agent_1": (base_pos2 - base_pos1) * 0.6,  # 朝向agent_2移动60%的距离
                "agent_2": (base_pos1 - base_pos2) * 0.6   # 朝向agent_1移动60%的距离
            },
            expected_collision=True,
            expected_collision_type="agent"
        ))
        
        # 测试用例3：安全距离，无碰撞
        test_cases.append(CollisionTestCase(
            name="no_collision_safe_distance",
            description="智能体保持安全距离，无碰撞",
            agent_positions={
                "agent_1": base_pos1.copy(),
                "agent_2": base_pos2.copy()
            },
            planned_movements={
                "agent_1": np.array([0.0, 0.0, 0.0]),
                "agent_2": np.array([0.0, 0.0, 0.0])
            },
            expected_collision=False,
            expected_collision_type=""
        ))
        
        # 测试用例4：三个智能体，其中两个会碰撞
        # 将第三个智能体移动靠近第一个
        test_cases.append(CollisionTestCase(
            name="three_agents_partial_collision",
            description="三个智能体中两个会碰撞",
            agent_positions={
                "agent_1": base_pos1.copy(),
                "agent_2": base_pos2.copy(),
                "agent_3": base_pos3.copy()
            },
            planned_movements={
                "agent_1": np.array([0.2, 0.0, 0.0]),  # 向右移动0.2米
                "agent_2": np.array([0.0, 0.0, 0.0]),  # 不移动
                "agent_3": (base_pos1 - base_pos3) * 0.9  # 朝向agent_1移动90%距离，会很近
            },
            expected_collision=True,
            expected_collision_type="agent"
        ))
        
        # 测试用例5：相向运动但不会碰撞
        safe_distance = np.linalg.norm(base_pos2 - base_pos1)
        if safe_distance > 2.0:  # 只有当两点距离足够远时才做这个测试
            test_cases.append(CollisionTestCase(
                name="approaching_but_safe",
                description="两个智能体相向运动但保持安全距离",
                agent_positions={
                    "agent_1": base_pos1.copy(),
                    "agent_2": base_pos2.copy()
                },
                planned_movements={
                    "agent_1": (base_pos2 - base_pos1) * 0.2,  # 朝向agent_2移动20%距离
                    "agent_2": (base_pos1 - base_pos2) * 0.2   # 朝向agent_1移动20%距离
                },
                expected_collision=False,
                expected_collision_type=""
            ))
        
        # 测试用例6：智能体在不同高度（测试Y轴忽略）
        height_pos1 = base_pos1.copy()
        height_pos2 = base_pos1.copy()
        height_pos2[0] += 0.3  # X方向偏移0.3米，距离很近
        height_pos2[1] += 2.0  # Y方向偏移2米，不同高度
        
        test_cases.append(CollisionTestCase(
            name="different_heights",
            description="智能体在不同高度但XZ平面距离过近",
            agent_positions={
                "agent_1": height_pos1,
                "agent_2": height_pos2
            },
            planned_movements={
                "agent_1": np.array([0.0, 0.0, 0.0]),
                "agent_2": np.array([0.0, 0.0, 0.0])
            },
            expected_collision=True,  # 应该检测到碰撞，因为只考虑XZ平面
            expected_collision_type="agent"
        ))
        
        return test_cases
    
    def run_collision_detection_test(self, test_case: CollisionTestCase) -> bool:
        """运行单个碰撞检测测试"""
        try:
            logging.info(f"\n--- Testing: {test_case.name} ---")
            logging.info(f"Description: {test_case.description}")
            
            # 执行碰撞预测
            has_collision, collision_reason = self.collision_detector.predict_collision(
                self.simulator.simulator.sim,
                test_case.agent_positions,
                test_case.planned_movements
            )
            
            # 检查结果
            test_passed = has_collision == test_case.expected_collision
            
            if test_passed:
                logging.info(f"✓ PASS - Collision detection result: {has_collision}")
                if has_collision:
                    logging.info(f"  Collision reason: {collision_reason}")
                    # 检查碰撞类型是否正确
                    if test_case.expected_collision_type == "agent" and "Agent collision" in collision_reason:
                        logging.info(f"  ✓ Correct collision type detected: agent collision")
                    elif test_case.expected_collision_type == "environment" and "environment" in collision_reason:
                        logging.info(f"  ✓ Correct collision type detected: environment collision")
                    elif test_case.expected_collision_type:
                        logging.warning(f"  ⚠ Expected {test_case.expected_collision_type} collision, but got: {collision_reason}")
            else:
                logging.error(f"✗ FAIL - Expected: {test_case.expected_collision}, Got: {has_collision}")
                if has_collision:
                    logging.error(f"  Unexpected collision reason: {collision_reason}")
            
            # 记录详细信息
            logging.info(f"  Agent positions: {test_case.agent_positions}")
            logging.info(f"  Planned movements: {test_case.planned_movements}")
            
            # 计算最终位置和距离
            final_positions = {}
            for agent_id, pos in test_case.agent_positions.items():
                movement = test_case.planned_movements.get(agent_id, np.array([0.0, 0.0, 0.0]))
                final_positions[agent_id] = pos + movement
            
            logging.info(f"  Final positions: {final_positions}")
            
            # 计算智能体间距离
            if len(final_positions) >= 2:
                agents = list(final_positions.items())
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        agent1_id, pos1 = agents[i]
                        agent2_id, pos2 = agents[j]
                        distance = np.linalg.norm(pos1[[0, 2]] - pos2[[0, 2]])  # XZ平面距离
                        logging.info(f"  Distance {agent1_id}-{agent2_id}: {distance:.3f}m (min required: {self.collision_detector.min_agent_distance}m)")
            
            return test_passed
            
        except Exception as e:
            logging.error(f"✗ ERROR in test {test_case.name}: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def run_action_movement_tests(self):
        """测试动作移动计算的准确性"""
        logging.info(f"\n{'='*60}")
        logging.info("TESTING ACTION MOVEMENT CALCULATIONS")
        logging.info(f"{'='*60}")
        
        # 创建测试智能体状态
        agent_state = AgentState(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0, 1.0])  # 朝向-Z方向（前方）
        )
        
        test_cases = [
            {
                "name": "move_to_action",
                "action": ActionCommand(action="move_to", target=[3.0, 2.0]),
                "expected_movement": np.array([3.0, 0.0, 2.0])
            },
            {
                "name": "move_forward_action", 
                "action": ActionCommand(action="move_forward", distance=2.0),
                "expected_movement": np.array([0.0, 0.0, -2.0])  # 前方是-Z方向
            },
            {
                "name": "turn_left_action",
                "action": ActionCommand(action="turn_left", angle=45.0),
                "expected_movement": np.array([0.0, 0.0, 0.0])  # 旋转不产生位移
            }
        ]
        
        for test_case in test_cases:
            try:
                logging.info(f"\n--- Testing: {test_case['name']} ---")
                
                movement = self.simulator._calculate_action_movement(
                    test_case["action"], agent_state
                )
                
                expected = test_case["expected_movement"]
                difference = np.linalg.norm(movement - expected)
                
                if difference < 0.01:  # 允许小误差
                    logging.info(f"✓ PASS - Movement calculation: {movement}")
                else:
                    logging.error(f"✗ FAIL - Expected: {expected}, Got: {movement}")
                    logging.error(f"  Difference: {difference}")
                
            except Exception as e:
                logging.error(f"✗ ERROR in {test_case['name']}: {e}")
    
    def run_environment_collision_tests(self):
        """测试环境碰撞检测"""
        logging.info(f"\n{'='*60}")
        logging.info("TESTING ENVIRONMENT COLLISION DETECTION")
        logging.info(f"{'='*60}")
        
        if not self.simulator or not self.simulator.simulator:
            logging.error("Simulator not available for environment collision tests")
            return
        
        # 获取一些测试位置
        try:
            # 获取可导航点作为安全位置
            safe_point = self.simulator.simulator.sim.pathfinder.get_random_navigable_point()
            safe_position = np.array([safe_point[0], safe_point[1], safe_point[2]])
            
            # 测试安全位置
            logging.info(f"\n--- Testing safe position ---")
            logging.info(f"Position: {safe_position}")
            
            collision = self.collision_detector.check_collision_with_environment(
                self.simulator.simulator.sim, safe_position
            )
            
            if not collision:
                logging.info("✓ PASS - Safe position correctly identified as navigable")
            else:
                logging.error("✗ FAIL - Safe position incorrectly identified as collision")
            
            # 测试明显不安全的位置（超出场景边界）
            logging.info(f"\n--- Testing unsafe position ---")
            bounds = self.simulator.simulator.scene_bounds
            unsafe_position = np.array([
                bounds[1][0] + 10.0,  # 超出X边界
                bounds[0][1],
                bounds[1][2] + 10.0   # 超出Z边界
            ])
            logging.info(f"Position: {unsafe_position}")
            
            collision = self.collision_detector.check_collision_with_environment(
                self.simulator.simulator.sim, unsafe_position
            )
            
            if collision:
                logging.info("✓ PASS - Unsafe position correctly identified as collision")
            else:
                logging.warning("⚠ WARNING - Unsafe position not detected as collision (might be outside pathfinder bounds)")
            
        except Exception as e:
            logging.error(f"Environment collision test failed: {e}")
            logging.error(traceback.format_exc())
    
    def run_all_tests(self):
        """运行所有测试"""
        logging.info(f"{'='*60}")
        logging.info("MULTI-AGENT COLLISION DETECTION TEST SUITE")
        logging.info(f"{'='*60}")
        
        if not self.setup_test_environment():
            logging.error("Failed to setup test environment")
            return False
        
        # 1. 测试碰撞检测基础功能
        logging.info(f"\n{'='*60}")
        logging.info("TESTING COLLISION DETECTION")
        logging.info(f"{'='*60}")
        
        test_cases = self.create_test_cases()
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            if self.run_collision_detection_test(test_case):
                passed_tests += 1
            self.test_results.append({
                "name": test_case.name,
                "passed": self.run_collision_detection_test(test_case)
            })
        
        # 2. 测试动作移动计算
        self.run_action_movement_tests()
        
        # 3. 测试环境碰撞检测
        self.run_environment_collision_tests()
        
        # 4. 总结测试结果
        logging.info(f"\n{'='*60}")
        logging.info("TEST SUMMARY")
        logging.info(f"{'='*60}")
        logging.info(f"Collision Detection Tests: {passed_tests}/{total_tests} passed")
        
        if passed_tests == total_tests:
            logging.info("🎉 ALL COLLISION DETECTION TESTS PASSED!")
        else:
            logging.warning(f"⚠ {total_tests - passed_tests} tests failed")
        
        # 打印智能体状态报告
        self.simulator.print_agent_status()
        
        return passed_tests == total_tests
    
    def cleanup(self):
        """清理资源"""
        if self.simulator:
            self.simulator.close()


def create_test_config():
    """创建测试配置文件"""
    test_config = {
        "scene": {
            "scene_dataset_path": "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
        },
        "simulator": {
            "gpu_device_id": 0,
            "enable_physics": True
        },
        "agents": [
            {
                "id": "agent_1",
                "initial_position": [0.0, 0.0, 0.0],
                "initial_rotation": [0, 0, 0, 1],
                "sensors": {
                    "color_sensor": {
                        "resolution": [256, 256]
                    }
                }
            },
            {
                "id": "agent_2", 
                "initial_position": [2.0, 0.0, 0.0],
                "initial_rotation": [0, 0, 0, 1],
                "sensors": {
                    "color_sensor": {
                        "resolution": [256, 256]
                    }
                }
            },
            {
                "id": "agent_3",
                "initial_position": [0.0, 0.0, 3.0],
                "initial_rotation": [0, 0, 0, 1],
                "sensors": {
                    "color_sensor": {
                        "resolution": [256, 256]
                    }
                }
            }
        ],
        "collision_detection": {
            "enabled": True,
            "agent_radius": 0.4,
            "height_threshold": 0.3,
            "prediction_steps": 3,
            "min_agent_distance": 0.8
        },
        "movement": {
            "linear_speed": 1.0,
            "angular_speed": 45.0,
            "time_step": 0.033
        },
        "video_output": {
            "output_dir": "outputs/collision_test",
            "fps": 30,
            "resolution": [600, 400]
        },
        "map_config": {
            "agent_marker_size": 8,
            "agent_marker_color": [255, 0, 0],
            "direction_arrow_length": 20
        },
        "state_persistence": {
            "save_after_each_action": False,
            "save_final_state": False,
            "state_file": "outputs/collision_test/agent_states.json"
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "outputs/collision_test/collision_test.log",
            "console_output": True
        }
    }
    
    return test_config


def main():
    """主函数"""
    # 创建测试配置
    test_config = create_test_config()
    
    # 创建输出目录
    os.makedirs("outputs/collision_test", exist_ok=True)
    
    # 保存测试配置
    config_path = "outputs/collision_test/test_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)
    
    # 运行测试
    tester = CollisionDetectionTester(config_path)
    
    try:
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 所有碰撞检测测试通过！")
            return 0
        else:
            print("\n⚠ 部分测试失败，请检查日志")
            return 1
            
    except Exception as e:
        logging.error(f"测试运行失败: {e}")
        logging.error(traceback.format_exc())
        return 1
        
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit(main())
