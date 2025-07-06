#!/usr/bin/env python3
"""
Physical Robot Sensor FPV Test
物理机器人传感器第一人称视角测试
"""

import os
import sys
import yaml
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_simple_test_config():
    """创建简单的测试配置"""
    config = {
        "scene": {
            "scene_dataset_path": "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
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
                        "resolution": [480, 640],  # 较小的分辨率便于测试
                        "position": [0.0, 1.5, 0.0],  # 传感器相对于机器人基座的位置
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
            "enabled": False,
            "agent_radius": 0.4,
            "height_threshold": 0.3,
            "prediction_steps": 3,
            "min_agent_distance": 1.5
        },
        "video_output": {
            "output_dir": "./outputs/robot_sensor_test",
            "fps": 10,  # 较低的帧率用于测试
            "resolution": [480, 960],  # [height, width] - 小分辨率
            "codec": "XVID",
            "save_frames": False
        },
        "map_config": {
            "resolution": 512,
            "show_grid": True,
            "grid_interval": 1.0,
            "agent_marker_size": 8,
            "agent_marker_color": [255, 0, 0],
            "direction_arrow_length": 15
        },
        "movement": {
            "linear_speed": 0.5,
            "angular_speed": 20.0,
            "time_step": 0.1,
            "path_interpolation_points": 10
        },
        "logging": {
            "log_file": "./outputs/robot_sensor_test/robot_sensor_test.log",
            "log_level": "INFO",
            "console_output": True
        },
        "state_persistence": {
            "state_file": "./outputs/robot_sensor_test/robot_states.json",
            "save_after_each_action": True,
            "save_final_state": True
        }
    }
    return config

def create_simple_actions():
    """创建简单的测试动作"""
    actions = {
        "fetch_robot": [
            {"action": "move_to", "target": [1.0, 1.0]},
            {"action": "turn_left", "angle": 90},
            {"action": "move_forward", "distance": 0.5},
            {"action": "turn_right", "angle": 180}
        ]
    }
    return actions

def main():
    """主函数"""
    print("=" * 70)
    print("PHYSICAL ROBOT SENSOR FPV TEST")
    print("=" * 70)
    print("Testing Fetch robot sensor-based FPV and video generation")
    print()
    
    try:
        from multi_agent_navigation import MultiAgentSimulator
        
        # 创建配置和动作
        config = create_simple_test_config()
        actions = create_simple_actions()
        
        # 创建输出目录
        output_dir = config["video_output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        print("1. Initializing simulator with physical robot...")
        simulator = MultiAgentSimulator(config)
        print("   ✓ Simulator initialized")
        
        print("\\n2. Checking robot sensor status...")
        simulator.print_agent_status()
        
        # 验证物理机器人
        report = simulator.get_agent_status_report()
        robot_info = report["agents"]["fetch_robot"]
        
        if not robot_info["has_physical_robot"]:
            print("   ❌ Physical robot not loaded - cannot test sensor FPV")
            return 1
        
        print(f"   ✓ Physical robot loaded with {robot_info['robot_joint_count']} joints")
        
        print("\\n3. Testing robot sensor observation...")
        try:
            # 测试从机器人传感器获取观察
            agent_state = simulator.agent_states["fetch_robot"]
            observation = simulator._get_robot_sensor_observation("fetch_robot", agent_state)
            
            print(f"   ✓ Robot sensor observation: {observation.shape}")
            print(f"   ✓ Image data type: {observation.dtype}")
            print(f"   ✓ Min/Max values: {observation.min()}/{observation.max()}")
            
        except Exception as e:
            print(f"   ❌ Robot sensor test failed: {e}")
            return 1
        
        print("\\n4. Executing actions with video recording...")
        success = simulator.execute_actions_sequence(actions)
        
        if success:
            print("   ✓ Actions executed successfully")
        else:
            print("   ⚠ Actions completed with warnings")
        
        print("\\n5. Checking output files...")
        video_file = os.path.join(output_dir, "fetch_robot_output.mp4")
        
        if os.path.exists(video_file):
            file_size = os.path.getsize(video_file)
            print(f"   ✓ Video file created: {file_size} bytes")
            
            if file_size > 1000:  # At least 1KB
                print("   ✓ Video file appears to contain data")
            else:
                print("   ⚠ Video file is very small - may be corrupted")
        else:
            print("   ❌ Video file not found")
        
        # 显示所有输出文件
        print("\\n6. Generated files:")
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    print(f"   📁 {filename}: {size} bytes")
        
        print("\\n7. Cleaning up...")
        simulator.close()
        print("   ✓ Simulator closed")
        
        print("\\n" + "=" * 70)
        print("🎉 PHYSICAL ROBOT SENSOR FPV TEST COMPLETED!")
        print("=" * 70)
        print(f"📁 Check outputs in: {output_dir}")
        print("📹 Video should show FPV from robot's head sensor")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
