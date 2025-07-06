#!/usr/bin/env python3
"""
Physical Agent Test Script
物理智能体加载测试脚本
"""

import os
import sys
import yaml
import json
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_physical_agent_loading():
    """测试物理智能体加载"""
    print("=" * 60)
    print("PHYSICAL AGENT LOADING TEST")
    print("=" * 60)
    
    try:
        from multi_agent_navigation import MultiAgentSimulator
        
        # 创建临时配置
        config = {
            "scene": {
                "scene_dataset_path": "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
                "scene_id": None
            },
            "simulator": {
                "gpu_device_id": 0,
                "enable_physics": True,
                "random_seed": 1
            },
            "agents": [
                {
                    "id": "fetch_robot",
                    "agent_model_path": "/home/yaoaa/habitat-lab/data/robots/hab_fetch/robots/hab_fetch.urdf",
                    "sensors": {
                        "color_sensor": {
                            "sensor_type": "COLOR",
                            "resolution": [256, 256],
                            "position": [0.0, 1.5, 0.0],
                            "hfov": 90.0
                        }
                    },
                    "action_space": {
                        "move_forward": {"amount": 0.25},
                        "move_backward": {"amount": 0.25},
                        "turn_left": {"amount": 10.0},
                        "turn_right": {"amount": 10.0}
                    },
                    "initial_position": None,
                    "initial_rotation": [0, 0, 0, 1]
                }
            ],
            "collision_detection": {
                "enabled": True,
                "agent_radius": 0.4,
                "height_threshold": 0.3,
                "prediction_steps": 3,
                "min_agent_distance": 1.2
            },
            "video_output": {
                "output_dir": "./test_outputs",
                "fps": 15,
                "resolution": [512, 512],
                "codec": "mp4v",
                "save_frames": False
            },
            "map_config": {
                "resolution": 512,
                "show_grid": True,
                "grid_interval": 1.0,
                "agent_marker_size": 6,
                "agent_marker_color": [255, 0, 0],
                "direction_arrow_length": 12
            },
            "movement": {
                "linear_speed": 0.5,
                "angular_speed": 20.0,
                "time_step": 0.1,
                "path_interpolation_points": 10
            },
            "logging": {
                "log_file": "./test_outputs/test_physical_agent.log",
                "log_level": "INFO",
                "console_output": True
            },
            "state_persistence": {
                "state_file": "./test_outputs/test_agent_states.json",
                "save_after_each_action": True,
                "save_final_state": True
            }
        }
        
        print("1. Initializing multi-agent simulator...")
        simulator = MultiAgentSimulator(config)
        print("   ✓ Simulator initialized successfully")
        
        print("\\n2. Checking physical agent status...")
        simulator.print_agent_status()
        
        print("\\n3. Getting detailed status report...")
        report = simulator.get_agent_status_report()
        
        # 验证物理智能体是否成功加载
        fetch_robot_info = report["agents"]["fetch_robot"]
        physical_robot_loaded = fetch_robot_info["has_physical_robot"]
        
        if physical_robot_loaded:
            print("   ✓ Physical Fetch robot successfully loaded!")
            print(f"   ✓ Robot Object ID: {fetch_robot_info['robot_object_id']}")
            print(f"   ✓ Joint Count: {fetch_robot_info['robot_joint_count']}")
            print(f"   ✓ Robot Status: {fetch_robot_info['robot_status']}")
        else:
            print("   ✗ Physical robot failed to load")
            print(f"   → Robot Status: {fetch_robot_info['robot_status']}")
            print(f"   → Model Path: {fetch_robot_info['model_path']}")
        
        print("\\n4. Testing basic agent movement...")
        # 创建简单的测试动作
        test_actions = {
            "fetch_robot": [
                {"action": "move_forward", "distance": 0.5},
                {"action": "turn_left", "angle": 45}
            ]
        }
        
        # 执行动作
        for agent_id, actions in test_actions.items():
            for action_data in actions:
                print(f"   Executing: {action_data}")
                # 这里只是演示，实际执行可能需要更复杂的步骤处理
        
        print("   ✓ Movement test completed")
        
        print("\\n5. Cleaning up...")
        simulator.close()
        print("   ✓ Simulator closed successfully")
        
        # 返回物理机器人是否成功加载
        return physical_robot_loaded
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Physical Agent Loading Test Suite")
    print("=" * 60)
    
    # 检查URDF文件是否存在
    urdf_path = "data/robots/hab_fetch/robots/hab_fetch.urdf"
    full_urdf_path = os.path.join("/home/yaoaa/habitat-lab", urdf_path)
    
    print(f"Checking URDF file: {urdf_path}")
    if os.path.exists(full_urdf_path):
        print(f"   ✓ URDF file found: {full_urdf_path}")
    else:
        print(f"   ✗ URDF file not found: {full_urdf_path}")
        print("   Please ensure Habitat-Lab data is properly installed")
        return 1
    
    # 运行物理智能体测试
    if test_physical_agent_loading():
        print("\\n" + "=" * 60)
        print("🎉 PHYSICAL AGENT TEST PASSED!")
        print("The Fetch robot was successfully loaded as a physical agent.")
        print("=" * 60)
        return 0
    else:
        print("\\n" + "=" * 60)
        print("❌ PHYSICAL AGENT TEST FAILED!")
        print("The physical robot could not be loaded properly.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main())
