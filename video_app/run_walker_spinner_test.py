#!/usr/bin/env python3
"""
简化的Walker Spinner测试 - 专门验证绝对时间轴同步功能
- walker: 简单的前进动作
- spinner: 简单的旋转动作
- 重点验证时间同步而非复杂的导航
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_navigation import MultiAgentSimulator

def create_simple_test_config():
    """创建简化的测试配置"""
    
    # 获取Habitat-Lab根目录的绝对路径
    habitat_root = Path(__file__).parent.parent.absolute()
    
    config = {
        "scene": {
            # 使用真实的测试场景
            "scene_dataset_path": str(habitat_root / "data/scene_datasets/habitat-test-scenes/apartment_1.glb")
        },
        "simulator": {
            "gpu_device_id": 0,
            "enable_physics": True  # 启用物理模拟以支持真实机器人
        },
        "agents": [
            {
                "id": "walker",
                "initial_position": [0.0, 0.0, 0.0],  # 默认起始位置，会通过move_to移动到实际起始点
                "initial_rotation": [0, 0, 0, 1],
                "agent_model_path": str(habitat_root / "data/robots/hab_fetch/robots/hab_fetch.urdf"),  # 使用真实机器人
                "sensors": {
                    "color_sensor": {
                        "resolution": [256, 256]  # 降低分辨率以提高速度
                    }
                }
            },
            {
                "id": "spinner",
                "initial_position": [0.0, 0.0, 0.0],  # 默认起始位置，会通过move_to移动到实际起始点
                "initial_rotation": [0, 0, 0, 1],
                "agent_model_path": str(habitat_root / "data/robots/hab_fetch/robots/hab_fetch.urdf"),  # 使用真实机器人
                "sensors": {
                    "color_sensor": {
                        "resolution": [256, 256]  # 降低分辨率以提高速度
                    }
                }
            }
        ],
        "collision_detection": {
            "enabled": True,  # 启用碰撞检测以获得更真实的行为
            "agent_radius": 0.2,
            "min_agent_distance": 0.6
        },
        "movement": {
            "linear_speed": 1.0,      # 1米/秒
            "angular_speed": 90.0,    # 90度/秒
            "time_step": 0.033,       # 30fps对应的时间步长
            "collision_check_steps": 1
        },
        "video_output": {
            "output_dir": "outputs/simple_time_sync_test",
            "fps": 30,
            "resolution": [512, 256]  # 较小的分辨率
        },
        "map_config": {
            "agent_marker_size": 6,
            "agent_marker_color": [255, 0, 0],
            "direction_arrow_length": 15
        },
        "logging": {
            "log_level": "DEBUG",  # 详细日志以便调试
            "log_file": "outputs/simple_time_sync_test/debug.log",
            "console_output": True
        },
        "state_persistence": {
            "save_after_each_action": True,  # 保存每个动作后的状态
            "save_final_state": True,
            "state_file": "outputs/simple_time_sync_test/states.json"
        }
    }
    
    return config

def create_simple_test_actions():
    """创建复杂的测试动作序列"""
    
    # Walker: 从[1.0, 0.0]到[2.0, 0.5]来回移动3次
    walker_actions = []
    
    # 首先移动到起始位置[1.0, 0.0]
    walker_actions.append({
        "action": "move_to",
        "target": [1.0, 0.0]
    })
    
    # 然后在[1.0, 0.0]和[2.0, 0.5]之间来回移动3次
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
    
    # Spinner: 移动到[2.0, 1.3]位置，然后向右旋转180度连续3次
    spinner_actions = []
    
    # 首先移动到指定位置[2.0, 1.3]
    spinner_actions.append({
        "action": "move_to",
        "target": [2.0, 1.3]
    })
    
    # 然后向右旋转180度，连续3次
    for rotation in range(3):
        spinner_actions.append({
            "action": "turn_right",
            "angle": 180.0
        })
    
    actions = {
        "walker": walker_actions,
        "spinner": spinner_actions
    }
    
    return actions

def analyze_expected_timing(actions, config):
    """分析预期的时间安排"""
    
    linear_speed = config["movement"]["linear_speed"]
    angular_speed = config["movement"]["angular_speed"]
    fps = config["video_output"]["fps"]
    
    print("\n" + "="*60)
    print("预期时间分析（绝对时间轴同步）")
    print("="*60)
    
    # Walker分析
    walker_actions = actions["walker"]
    walker_total_time = 0.0
    print("\nWalker动作分析：")
    for i, action in enumerate(walker_actions):
        if action["action"] == "move_to":
            if i == 0:
                # 第一个动作：从[0,0,0]移动到[1.0, 0.0]
                distance = ((1.0-0)**2 + (0.0-0)**2)**0.5
                duration = distance / linear_speed
                print(f"  动作{i+1}: 移动到起始位置[1.0, 0.0] → 距离{distance:.2f}米，耗时{duration:.2f}秒")
            else:
                # 计算移动距离
                if action["target"] == [2.0, 0.5]:
                    distance = ((2.0-1.0)**2 + (0.5-0.0)**2)**0.5  # 从[1.0,0.0]到[2.0,0.5]
                    print(f"  动作{i+1}: 移动到[2.0, 0.5] → 距离{distance:.2f}米，耗时{distance/linear_speed:.2f}秒")
                else:
                    distance = ((1.0-2.0)**2 + (0.0-0.5)**2)**0.5  # 从[2.0,0.5]到[1.0,0.0]
                    print(f"  动作{i+1}: 移动到[1.0, 0.0] → 距离{distance:.2f}米，耗时{distance/linear_speed:.2f}秒")
                duration = distance / linear_speed
            walker_total_time += duration
        elif action["action"] in ["turn_left", "turn_right"]:
            duration = action["angle"] / angular_speed
            print(f"  动作{i+1}: 旋转{action['angle']}度 → 耗时{duration:.2f}秒")
            walker_total_time += duration
    print(f"  Walker总耗时: {walker_total_time:.2f}秒")
    
    # Spinner分析
    spinner_actions = actions["spinner"]
    spinner_total_time = 0.0
    print("\nSpinner动作分析：")
    for i, action in enumerate(spinner_actions):
        if action["action"] == "move_to":
            if i == 0:
                # 第一个动作：从[0,0,0]移动到[2.0, 1.3]
                distance = ((2.0-0)**2 + (1.3-0)**2)**0.5
                duration = distance / linear_speed
                print(f"  动作{i+1}: 移动到起始位置[2.0, 1.3] → 距离{distance:.2f}米，耗时{duration:.2f}秒")
            spinner_total_time += duration
        elif action["action"] in ["turn_left", "turn_right"]:
            duration = action["angle"] / angular_speed
            print(f"  动作{i+1}: 旋转{action['angle']}度 → 耗时{duration:.2f}秒")
            spinner_total_time += duration
    print(f"  Spinner总耗时: {spinner_total_time:.2f}秒")
    
    # 视频分析
    max_time = max(walker_total_time, spinner_total_time)
    total_frames = int(max_time * fps)
    
    print(f"\n视频分析：")
    print(f"  最大耗时: {max_time:.2f}秒")
    print(f"  视频帧数: {total_frames}帧")
    print(f"  帧率: {fps}fps")
    print(f"  每帧时间: {1.0/fps:.4f}秒")
    
    print(f"\n时间轴同步验证：")
    print(f"  t=0.0s (第1帧): 两个智能体同时开始动作")
    if walker_total_time < max_time:
        completion_frame = int(walker_total_time * fps)
        print(f"  t={walker_total_time:.2f}s (第{completion_frame}帧): Walker完成所有动作，Spinner继续")
    if spinner_total_time < max_time:
        completion_frame = int(spinner_total_time * fps)
        print(f"  t={spinner_total_time:.2f}s (第{completion_frame}帧): Spinner完成所有动作，Walker继续")
    print(f"  t={max_time:.2f}s (第{total_frames}帧): 所有动作完成")
    
    print("="*60)

def verify_smooth_motion(log_file):
    """验证动作的平滑性"""
    if not os.path.exists(log_file):
        print("日志文件不存在，无法验证平滑性")
        return
    
    print("\n" + "="*40)
    print("平滑性验证")
    print("="*40)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        frame_updates = []
        for line in lines:
            if "Frame" in line and "Agent" in line:
                frame_updates.append(line.strip())
        
        if frame_updates:
            print(f"检测到 {len(frame_updates)} 个帧更新")
            if len(frame_updates) > 10:
                print("前10个帧更新：")
                for update in frame_updates[:10]:
                    print(f"  {update}")
                print("...")
                print("后5个帧更新：")
                for update in frame_updates[-5:]:
                    print(f"  {update}")
            else:
                print("所有帧更新：")
                for update in frame_updates:
                    print(f"  {update}")
        else:
            print("未找到帧更新记录")
            
    except Exception as e:
        print(f"验证平滑性时出错: {e}")

def main():
    print("=== 简化的时间同步测试 ===")
    
    try:
        # 创建输出目录
        os.makedirs("outputs/simple_time_sync_test", exist_ok=True)
        
        # 创建配置和动作
        config = create_simple_test_config()
        actions = create_simple_test_actions()
        
        # 分析预期时间安排
        analyze_expected_timing(actions, config)
        
        # 保存配置和动作到文件（便于调试）
        with open("outputs/simple_time_sync_test/config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open("outputs/simple_time_sync_test/actions.json", "w") as f:
            json.dump(actions, f, indent=2)
        
        print(f"\n配置和动作已保存到 outputs/simple_time_sync_test/")
        
        # 初始化模拟器
        print("\n初始化模拟器...")
        simulator = MultiAgentSimulator(config)
        
        # 显示同步方式解释
        simulator.explain_synchronization_difference()
        
        # 显示智能体初始状态
        print("\n" + "="*40)
        print("智能体初始状态")
        print("="*40)
        simulator.print_agent_status()
        
        # 执行动作序列
        print("\n" + "="*40)
        print("执行动作序列")
        print("="*40)
        success = simulator.execute_actions_sequence(actions)
        
        # 显示最终状态
        print("\n" + "="*40)
        print("智能体最终状态")
        print("="*40)
        simulator.print_agent_status()
        
        # 关闭模拟器
        simulator.close()
        
        if success:
            print("\n✓ 测试执行成功!")
            
            # 验证平滑性
            log_file = "outputs/simple_time_sync_test/debug.log"
            verify_smooth_motion(log_file)
            
        else:
            print("\n✗ 测试执行失败!")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 显示结果
    output_dir = "outputs/simple_time_sync_test"
    print(f"\n=== 测试完成 ===")
    print(f"输出目录: {output_dir}")
    
    # 检查生成的文件
    if os.path.exists(output_dir):
        print("\n生成的文件：")
        for filename in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                if filename.endswith('.mp4'):
                    print(f"  📹 {filename} ({size:,} bytes)")
                elif filename.endswith('.log'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                    print(f"  📄 {filename} ({lines} 行)")
                elif filename.endswith('.json'):
                    print(f"  📊 {filename} ({size:,} bytes)")
                else:
                    print(f"  📁 {filename} ({size:,} bytes)")
    
    print(f"\n请检查生成的视频文件，验证：")
    print(f"1. Walker是否按照计划路径移动：[0,0] → [1.0,0.0] → [2.0,0.5] → [1.0,0.0] (重复3次)")
    print(f"2. Spinner是否移动到[2.0,1.3]然后向右旋转180度（重复3次）")
    print(f"3. 两个智能体的动作是否按照绝对时间轴同步")
    print(f"4. 视频中没有突然的跳跃或传送，动作连续平滑")
    
    return True

if __name__ == "__main__":
    main()
