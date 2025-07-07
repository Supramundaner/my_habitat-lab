#!/usr/bin/env python3
"""
Walker Spinner 测试
- walker在[1.0, 0.0]和[2.0, 0.5]之间来回移动3次
- spinner在[2.0, 1.3]位置旋转3个完整的360度
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_navigation import MultiAgentSimulator

def create_walker_spinner_config():
    """创建walker和spinner的配置"""
    
    # 获取Habitat-Lab根目录的绝对路径
    habitat_root = Path(__file__).parent.parent.absolute()
    
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
                "agent_model_path": str(habitat_root / "data/robots/hab_fetch/robots/hab_fetch.urdf"),
                "initial_position": [1.0, 0.0, 0.0],
                "initial_rotation": [0, 0, 0, 1],
                "sensors": {
                    "color_sensor": {
                        "resolution": [512, 512]
                    }
                }
            },
            {
                "id": "agent_spinner",
                "agent_model_path": str(habitat_root / "data/robots/hab_fetch/robots/hab_fetch.urdf"),
                "initial_position": [2.0, 1.3, 0.0],
                "initial_rotation": [0, 0, 0, 1],
                "sensors": {
                    "color_sensor": {
                        "resolution": [512, 512]
                    }
                }
            }
        ],
        "collision_detection": {
            "enabled": True,
            "agent_radius": 0.2,
            "min_agent_distance": 0.6
        },
        "movement": {
            "linear_speed": 30.0,     # 度/秒
            "angular_speed": 45.0,    # 度/秒
            "time_step": 0.05,        # 时间步长
            "collision_check_steps": 5
        },
        "video_output": {
            "output_dir": "outputs/walker_spinner_test",
            "fps": 30,
            "resolution": [1280, 720]
        },
        "map_config": {
            "agent_marker_size": 8,
            "agent_marker_color": [255, 0, 0],
            "direction_arrow_length": 20
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "outputs/walker_spinner_test/navigation.log",
            "console_output": True
        },
        "state_persistence": {
            "save_after_each_action": False,
            "save_final_state": True,
            "state_file": "outputs/walker_spinner_test/final_states.json"
        }
    }
    
    return config

def create_walker_spinner_actions():
    """创建walker来回移动和spinner旋转的动作序列"""
    
    walker_actions = []
    spinner_actions = []
    
    # Walker: 在[1.0, 0.0]和[2.0, 0.5]之间来回移动3次
    # 总共需要6个移动（来回3次 = 3次去 + 3次回）
    for cycle in range(3):
        # 移动到[2.0, 0.5]
        walker_actions.append({
            "action": "move_to",
            "target": [2.0, 0.5]
        })
        
        # 移动回[1.0, 0.0]
        walker_actions.append({
            "action": "move_to", 
            "target": [1.0, 0.0]
        })
    
    # Spinner: 在[2.0, 1.3]位置旋转3个完整的360度
    # Spinner is already at the target location, so we just need to rotate
    for rotation in range(3):
        spinner_actions.append({
            "action": "turn_left",
            "angle": 360.0
        })
    
    return {
        "agent_walker": walker_actions,
        "agent_spinner": spinner_actions
    }

def main():
    print("=== 开始Walker Spinner测试 ===")
    
    try:
        # 创建配置
        config = create_walker_spinner_config()
        
        # 创建动作序列
        actions = create_walker_spinner_actions()
        
        # 保存动作序列到文件（可选，用于调试）
        os.makedirs("outputs/walker_spinner_test", exist_ok=True)
        with open("outputs/walker_spinner_test/actions.json", "w") as f:
            json.dump(actions, f, indent=2)
        
        # 初始化多智能体模拟器
        simulator = MultiAgentSimulator(config)
        
        # 显示智能体状态
        simulator.print_agent_status()
        
        # 执行动作序列
        success = simulator.execute_actions_sequence(actions)
        
        if success:
            print("\n✓ 动作序列执行成功!")
        else:
            print("\n✗ 动作序列执行失败!")
        
        # 关闭模拟器
        simulator.close()
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 显示结果
    output_dir = "outputs/walker_spinner_test"
    print(f"\n=== 测试完成 ===")
    print(f"输出目录: {output_dir}")
    
    # 检查生成的文件
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                if filename.endswith('.mp4'):
                    print(f"视频文件: {filename} ({size} bytes)")
                elif filename.endswith('.log'):
                    # 显示日志行数
                    with open(filepath, 'r') as f:
                        lines = len(f.readlines())
                    print(f"日志文件: {lines} 行")
                    
                    # 显示最后几行日志
                    with open(filepath, 'r') as f:
                        log_lines = f.readlines()
                    if len(log_lines) > 5:
                        print("最后几行日志:")
                        for line in log_lines[-10:]:
                            print(f"  {line.strip()}")
                elif filename.endswith('.json'):
                    print(f"状态文件: {filename}")
                    
                    # 如果是最终状态文件，显示内容
                    if filename == "final_states.json":
                        try:
                            with open(filepath, 'r') as f:
                                states = json.load(f)
                            print("最终状态:")
                            for agent_id, state in states.items():
                                pos = state.get('position', [0, 0, 0])
                                print(f"  {agent_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                        except Exception as e:
                            print(f"  无法读取状态文件: {e}")
    
    return True

if __name__ == "__main__":
    main()
