#!/usr/bin/env python3
"""
安全的双代理测试脚本
使用更安全的初始位置和动作序列来避免过早的碰撞检测
"""

import sys
import os
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from multi_agent_navigation import MultiAgentSimulator

def create_safe_test_config():
    """创建安全的测试配置"""
    # 获取项目根目录
    habitat_root = Path(__file__).parent.parent
    
    config = {
        "scene": {
            "scene_dataset_path": str(habitat_root / "data/scene_datasets/habitat-test-scenes/apartment_1.glb")
        },
        "simulator": {
            "gpu_device_id": 0,
            "enable_physics": True
        },
        "agents": [
            {
                "id": "agent_walker",
                "initial_position": [5.0, 0.0, 4.0],  # 更远的起始位置
                "initial_rotation": [0, 0, 0, 1],
                "agent_model_path": str(habitat_root / "data/robots/hab_fetch/robots/hab_fetch.urdf"),
                "sensors": {
                    "color_sensor": {
                        "resolution": [480, 640]
                    }
                }
            },
            {
                "id": "agent_spinner", 
                "initial_position": [1.0, 0.0, 1.0],  # 更远的起始位置
                "initial_rotation": [0, 0, 0, 1],
                "agent_model_path": str(habitat_root / "data/robots/hab_fetch/robots/hab_fetch.urdf"),
                "sensors": {
                    "color_sensor": {
                        "resolution": [480, 640]
                    }
                }
            }
        ],
        "collision_detection": {
            "enabled": True,
            "agent_radius": 0.3,
            "min_agent_distance": 0.8,
            "prediction_steps": 20
        },
        "movement": {
            "linear_speed": 1.0,
            "angular_speed": 45.0,
            "time_step": 0.1,
            "collision_check_steps": 20
        },
        "video_output": {
            "output_dir": "outputs/safe_two_agents_test",
            "fps": 30,
            "resolution": [480, 640]
        },
        "map_config": {
            "agent_marker_size": 8,
            "agent_marker_color": [255, 100, 100],
            "direction_arrow_length": 15
        },
        "logging": {
            "log_file": "outputs/safe_two_agents_test/navigation.log",
            "log_level": "INFO",
            "console_output": True
        },
        "state_persistence": {
            "save_after_each_action": False,
            "save_final_state": True,
            "state_file": "outputs/safe_two_agents_test/final_states.json"
        }
    }
    return config

def create_safe_action_sequences():
    """创建安全的动作序列"""
    action_sequences = {
        "agent_walker": [
            {"action": "move_to", "target": [4.0, 3.5]},   # 向左上移动
            {"action": "move_to", "target": [3.0, 3.0]},   # 继续向左移动
            {"action": "move_to", "target": [2.5, 4.0]},   # 向上移动
            {"action": "turn_left", "angle": 180},         # 转身
            {"action": "move_to", "target": [3.5, 3.5]},   # 回程
        ],
        "agent_spinner": [
            {"action": "move_to", "target": [1.5, 1.5]},   # 向右下移动
            {"action": "turn_right", "angle": 90},         # 右转
            {"action": "move_to", "target": [2.0, 0.5]},   # 向下移动
            {"action": "turn_left", "angle": 180},         # 转身
            {"action": "move_to", "target": [1.2, 1.2]},   # 回程
        ]
    }
    return action_sequences

def test_safe_collision_avoidance():
    """测试安全的碰撞避免功能"""
    print("=== 开始安全的双代理导航测试 ===")
    
    # 创建配置
    config = create_safe_test_config()
    action_sequences = create_safe_action_sequences()
    
    # 创建导航系统
    navigation = MultiAgentSimulator(config)
    
    try:
        # 执行导航
        navigation.execute_actions_sequence(action_sequences)
        
        # 检查输出文件
        output_dir = "outputs/safe_two_agents_test"
        
        print(f"\n=== 测试完成 ===")
        print(f"输出目录: {output_dir}")
        
        # 检查生成的文件
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.endswith('.mp4'):
                    file_path = os.path.join(output_dir, filename)
                    file_size = os.path.getsize(file_path)
                    print(f"视频文件: {filename} ({file_size} bytes)")
        
        # 检查日志文件
        log_path = os.path.join(output_dir, "navigation.log")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                print(f"日志文件: {len(lines)} 行")
                # 显示最后几行
                print("最后几行日志:")
                for line in lines[-10:]:
                    print(f"  {line.strip()}")
        
        # 检查最终状态
        final_states_path = os.path.join(output_dir, "final_states.json")
        if os.path.exists(final_states_path):
            with open(final_states_path, 'r') as f:
                states = json.load(f)
                print(f"最终状态:")
                for agent_id, state in states.items():
                    pos = state.get('position', [0, 0, 0])
                    print(f"  {agent_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        navigation.close()

if __name__ == "__main__":
    test_safe_collision_avoidance()
