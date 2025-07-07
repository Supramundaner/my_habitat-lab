#!/usr/bin/env python3
"""
测试动态碰撞检测功能
"""

import os
import sys
import json
import logging
import numpy as np

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_navigation import MultiAgentSimulator, ActionCommand

def create_test_config():
    """创建测试配置"""
    config = {
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
                        "resolution": [480, 640]
                    }
                }
            },
            {
                "id": "agent_2", 
                "initial_position": [2.0, 0.0, 0.0],
                "initial_rotation": [0, 0, 0, 1],
                "sensors": {
                    "color_sensor": {
                        "resolution": [480, 640]
                    }
                }
            }
        ],
        "collision_detection": {
            "enabled": True,
            "agent_radius": 0.2,
            "min_agent_distance": 0.5,
            "prediction_steps": 5
        },
        "movement": {
            "linear_speed": 1.0,
            "angular_speed": 45.0,
            "time_step": 0.1,
            "collision_check_steps": 10
        },
        "video_output": {
            "output_dir": "outputs/collision_test",
            "fps": 30,
            "resolution": [480, 640]
        },
        "map_config": {
            "agent_marker_size": 5,
            "agent_marker_color": [255, 0, 0],
            "direction_arrow_length": 10
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "outputs/collision_test/test.log",
            "console_output": True
        },
        "state_persistence": {
            "save_after_each_action": False,
            "save_final_state": True,
            "state_file": "outputs/collision_test/final_states.json"
        }
    }
    return config

def test_collision_scenarios():
    """测试不同的碰撞场景"""
    print("="*60)
    print("动态碰撞检测测试")
    print("="*60)
    
    # 创建配置
    config = create_test_config()
    
    try:
        # 初始化模拟器
        print("初始化多智能体模拟器...")
        simulator = MultiAgentSimulator(config)
        
        print("✓ 模拟器初始化成功")
        simulator.print_agent_status()
        
        # 测试场景1：智能体相互靠近的碰撞
        print("\n" + "="*40)
        print("测试场景1：智能体相互靠近")
        print("="*40)
        
        collision_actions = {
            "agent_1": [
                ActionCommand("move_to", target=[1.0, 0.0]),  # 向右移动
            ],
            "agent_2": [
                ActionCommand("move_to", target=[1.0, 0.0]),  # 向左移动，会与agent_1碰撞
            ]
        }
        
        # 执行碰撞测试
        print("执行可能导致碰撞的动作...")
        result = simulator.execute_actions_sequence(collision_actions)
        print(f"结果: {'成功' if result else '失败/碰撞检测生效'}")
        
        # 测试场景2：智能体朝向墙壁移动
        print("\n" + "="*40)
        print("测试场景2：智能体朝向边界移动") 
        print("="*40)
        
        boundary_actions = {
            "agent_1": [
                ActionCommand("move_to", target=[-10.0, -10.0]),  # 尝试移动到场景边界外
            ]
        }
        
        print("执行朝向边界的动作...")
        result = simulator.execute_actions_sequence(boundary_actions)
        print(f"结果: {'成功' if result else '失败/边界碰撞检测生效'}")
        
        # 测试场景3：安全的移动
        print("\n" + "="*40)
        print("测试场景3：安全的移动")
        print("="*40)
        
        safe_actions = {
            "agent_1": [
                ActionCommand("move_to", target=[0.5, 1.0]),  # 安全的移动
            ],
            "agent_2": [
                ActionCommand("move_to", target=[3.0, 1.0]),  # 保持距离的移动
            ]
        }
        
        print("执行安全的动作...")
        result = simulator.execute_actions_sequence(safe_actions)
        print(f"结果: {'成功' if result else '失败'}")
        
        print("\n" + "="*40)
        print("最终智能体状态:")
        print("="*40)
        simulator.print_agent_status()
        
        # 清理
        simulator.close()
        print("\n✓ 测试完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def test_path_sampling():
    """测试路径采样功能"""
    print("\n" + "="*60)
    print("路径采样测试")
    print("="*60)
    
    config = create_test_config()
    
    try:
        simulator = MultiAgentSimulator(config)
        
        # 测试不同类型的动作路径采样
        from multi_agent_navigation import AgentState
        
        test_state = AgentState(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0, 0, 0, 1])
        )
        
        # 测试move_to动作的路径采样
        move_to_action = ActionCommand("move_to", target=[2.0, 0.0])
        path = simulator._sample_path_for_action(move_to_action, test_state)
        print(f"Move_to路径采样点数: {len(path)}")
        
        # 测试move_forward动作的路径采样
        move_forward_action = ActionCommand("move_forward", distance=1.5)
        path = simulator._sample_path_for_action(move_forward_action, test_state)
        print(f"Move_forward路径采样点数: {len(path)}")
        
        print("✓ 路径采样测试成功")
        
        simulator.close()
        
    except Exception as e:
        print(f"✗ 路径采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def main():
    """主测试函数"""
    print("开始动态碰撞检测系统测试...")
    
    # 确保输出目录存在
    os.makedirs("outputs/collision_test", exist_ok=True)
    
    # 运行测试
    tests = [
        ("基础碰撞检测", test_collision_scenarios),
        ("路径采样", test_path_sampling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n正在运行: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\n总体结果: {'✓ 所有测试通过' if all_passed else '✗ 部分测试失败'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
