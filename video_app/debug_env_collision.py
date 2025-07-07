#!/usr/bin/env python3
"""
调试环境碰撞检测的脚本
"""

import os
import sys
import json
import numpy as np
import logging

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_navigation import MultiAgentSimulator, CollisionDetector

def debug_environment_collision():
    """调试环境碰撞检测"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 使用与测试相同的配置
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
            }
        ],
        "collision_detection": {
            "enabled": True,
            "agent_radius": 0.4,
            "height_threshold": 0.3,
            "prediction_steps": 3,
            "min_agent_distance": 0.8
        },
        "movement": {"linear_speed": 1.0, "angular_speed": 45.0, "time_step": 0.033},
        "video_output": {"output_dir": "outputs/debug", "fps": 30, "resolution": [600, 400]},
        "map_config": {"agent_marker_size": 8, "agent_marker_color": [255, 0, 0], "direction_arrow_length": 20},
        "state_persistence": {"save_after_each_action": False, "save_final_state": False, "state_file": "outputs/debug/states.json"},
        "logging": {"log_level": "INFO", "log_file": "outputs/debug/debug.log", "console_output": True}
    }
    
    # 创建模拟器
    simulator = MultiAgentSimulator(test_config)
    collision_detector = simulator.collision_detector
    
    # 获取一些测试位置
    nav_point1 = simulator.simulator.sim.pathfinder.get_random_navigable_point()
    nav_point2 = simulator.simulator.sim.pathfinder.get_random_navigable_point()
    
    pos1 = np.array([nav_point1.x, nav_point1.y, nav_point1.z])
    pos2 = np.array([nav_point2.x, nav_point2.y, nav_point2.z])
    
    print(f"Testing positions:")
    print(f"  Position 1: {pos1}")
    print(f"  Position 2: {pos2}")
    print(f"  Distance: {np.linalg.norm(pos2[[0, 2]] - pos1[[0, 2]]):.3f}m")
    
    # 单独测试每个位置的环境碰撞
    print(f"\n--- Testing Environment Collision ---")
    
    for i, pos in enumerate([pos1, pos2], 1):
        print(f"\nPosition {i}: {pos}")
        
        # 检查原始位置
        import magnum as mn
        test_point = mn.Vector3(pos[0], pos[1], pos[2])
        is_navigable = simulator.simulator.sim.pathfinder.is_navigable(test_point)
        print(f"  Original position navigable: {is_navigable}")
        
        # 检查捕捉
        snapped_point = simulator.simulator.sim.pathfinder.snap_point(test_point)
        snapped_pos = np.array([snapped_point.x, snapped_point.y, snapped_point.z])
        snap_distance = np.linalg.norm(pos - snapped_pos)
        is_snapped_navigable = simulator.simulator.sim.pathfinder.is_navigable(snapped_point)
        
        print(f"  Snapped position: {snapped_pos}")
        print(f"  Snap distance: {snap_distance:.3f}m")
        print(f"  Snapped position navigable: {is_snapped_navigable}")
        
        # 使用碰撞检测器
        has_env_collision = collision_detector.check_collision_with_environment(
            simulator.simulator.sim, pos
        )
        print(f"  Environment collision detected: {has_env_collision}")
        
        # 检查周围点
        print(f"  Checking surrounding points (radius: {collision_detector.agent_radius}):")
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            check_x = pos[0] + collision_detector.agent_radius * 0.8 * np.cos(angle)
            check_z = pos[2] + collision_detector.agent_radius * 0.8 * np.sin(angle)
            check_point = mn.Vector3(check_x, pos[1], check_z)
            check_navigable = simulator.simulator.sim.pathfinder.is_navigable(check_point)
            print(f"    Angle {np.degrees(angle):6.1f}°: navigable={check_navigable}")
    
    # 测试完整的碰撞预测
    print(f"\n--- Testing Complete Collision Prediction ---")
    
    agent_positions = {"agent_1": pos1, "agent_2": pos2}
    planned_movements = {"agent_1": np.array([0.0, 0.0, 0.0]), "agent_2": np.array([0.0, 0.0, 0.0])}
    
    has_collision, reason = collision_detector.predict_collision(
        simulator.simulator.sim, agent_positions, planned_movements
    )
    
    print(f"  Collision detected: {has_collision}")
    print(f"  Reason: {reason}")
    
    simulator.close()

if __name__ == "__main__":
    debug_environment_collision()
