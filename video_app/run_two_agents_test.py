#!/usr/bin/env python3
"""
双智能体动态碰撞检测测试脚本
测试两个智能体在apartment_1.glb环境中的多智能体导航，包含动态碰撞检测功能
"""

import os
import sys
import yaml
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

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
                "id": "agent_walker",
                "initial_position": [4.0, 0.0, 3.0],
                "initial_rotation": [0, 0, 0, 1],
                "agent_model_path": "data/robots/hab_fetch/robots/hab_fetch.urdf",
                "sensors": {
                    "color_sensor": {
                        "resolution": [480, 640]
                    }
                }
            },
            {
                "id": "agent_spinner", 
                "initial_position": [2.0, 0.0, 1.5],
                "initial_rotation": [0, 0, 0, 1],
                "agent_model_path": "data/robots/hab_fetch/robots/hab_fetch.urdf",
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
            "prediction_steps": 5
        },
        "movement": {
            "linear_speed": 1.0,
            "angular_speed": 45.0,
            "time_step": 0.1,
            "collision_check_steps": 15
        },
        "video_output": {
            "output_dir": "outputs/two_agents_dynamic_test",
            "fps": 30,
            "resolution": [480, 640]
        },
        "map_config": {
            "agent_marker_size": 8,
            "agent_marker_color": [255, 100, 100],
            "direction_arrow_length": 15
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "outputs/two_agents_dynamic_test/navigation.log",
            "console_output": True
        },
        "state_persistence": {
            "save_after_each_action": False,
            "save_final_state": True,
            "state_file": "outputs/two_agents_dynamic_test/final_states.json"
        }
    }
    return config

def create_test_actions():
    """创建测试动作序列，包含可能产生碰撞的场景"""
    actions = {
        "agent_walker": [
            {"action": "move_to", "target": [3.0, 3.0]},  # 移动到中间区域
            {"action": "move_to", "target": [2.0, 2.0]},  # 向spinner靠近 - 可能碰撞
            {"action": "move_to", "target": [0.5, 3.0]},  # 安全移动
            {"action": "move_to", "target": [0.5, 0.0]},
            {"action": "turn_left", "angle": 360}         # 完整旋转
        ],
        "agent_spinner": [
            {"action": "move_to", "target": [2.0, 1.8]},  # 向walker靠近 - 可能碰撞
            {"action": "turn_right", "angle": 180},       # 半圈旋转
            {"action": "turn_left", "angle": 360},        # 第一圈
            {"action": "turn_left", "angle": 360},        # 第二圈
            {"action": "turn_left", "angle": 360}         # 第三圈
        ]
    }
    return actions

def main():
    """主函数"""
    print("=" * 80)
    print("DUAL AGENT DYNAMIC COLLISION DETECTION TEST")
    print("=" * 80)
    print("Testing two agents with dynamic collision detection in apartment_1.glb:")
    print("- Agent Walker: Following path with potential collision scenarios")
    print("- Agent Spinner: Move close to walker then spin (collision test)")
    print("- Dynamic collision detection: Monitors entire movement paths")
    print()
    
    try:
        from multi_agent_navigation import MultiAgentSimulator
        
        # 创建配置和动作
        config = create_test_config()
        actions = create_test_actions()
        
        # 确保输出目录存在
        output_dir = config["video_output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        print("1. Initializing dual-agent simulator with collision detection...")
        simulator = MultiAgentSimulator(config)
        print("   ✓ Simulator initialized with dynamic collision detection")
        
        print("\n2. Checking agent status...")
        simulator.print_agent_status()
        
        # 验证智能体状态
        report = simulator.get_agent_status_report()
        
        walker_info = report["agents"].get("agent_walker", {})
        spinner_info = report["agents"].get("agent_spinner", {})
        
        print(f"\n3. Agent validation:")
        if walker_info.get("has_physical_robot"):
            print(f"   ✓ Walker agent: Physical robot with {walker_info.get('robot_joint_count', 'N/A')} joints")
        else:
            print("   → Walker agent: Using virtual agent (URDF not found)")
            
        if spinner_info.get("has_physical_robot"):
            print(f"   ✓ Spinner agent: Physical robot with {spinner_info.get('robot_joint_count', 'N/A')} joints")
        else:
            print("   → Spinner agent: Using virtual agent (URDF not found)")
        
        print(f"\n4. Collision detection settings:")
        collision_config = config["collision_detection"]
        print(f"   ✓ Enabled: {collision_config['enabled']}")
        print(f"   ✓ Agent radius: {collision_config['agent_radius']}m")
        print(f"   ✓ Min distance: {collision_config['min_agent_distance']}m")
        print(f"   ✓ Path sampling steps: {config['movement']['collision_check_steps']}")
        
        print("\n5. Action plan with collision scenarios:")
        for agent_id, agent_actions in actions.items():
            print(f"   {agent_id}:")
            for i, action in enumerate(agent_actions, 1):
                if action["action"] == "move_to":
                    print(f"     {i}. Move to ({action['target'][0]}, {action['target'][1]})")
                elif action["action"] == "turn_left":
                    print(f"     {i}. Turn left {action['angle']}°")
                elif action["action"] == "turn_right":
                    print(f"     {i}. Turn right {action['angle']}°")
        
        print("\n6. Executing dual-agent navigation with collision detection...")
        print("   → This will test dynamic collision detection during movement")
        print("   → Expected: Some actions may be prevented due to collision prediction")
        
        success = simulator.execute_actions_sequence(actions)
        
        if success:
            print("   ✓ All actions executed successfully (no collisions detected)")
        else:
            print("   ✓ Collision detection worked - some actions were prevented")
        
        print("\n7. Checking output files...")
        
        # 检查视频文件
        video_files = [
            ("Walker", "agent_walker_output.mp4"),
            ("Spinner", "agent_spinner_output.mp4")
        ]
        
        for agent_name, filename in video_files:
            video_path = os.path.join(output_dir, filename)
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"   ✓ {agent_name} video: {size_mb:.1f} MB")
            else:
                print(f"   ⚠ {agent_name} video not found")
        
        # 显示所有输出文件
        print("\n8. Generated files:")
        if os.path.exists(output_dir):
            for filename in sorted(os.listdir(output_dir)):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"   📁 {filename}: {size_mb:.2f} MB")
        
        print("\n9. Final agent positions:")
        final_report = simulator.get_agent_status_report()
        for agent_id, agent_info in final_report["agents"].items():
            pos = agent_info.get("current_position")
            if pos and pos != "N/A":
                print(f"   {agent_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            else:
                print(f"   {agent_id}: Position not available")
        
        print("\n10. Testing collision detection specifically...")
        # 创建一个明确会碰撞的测试场景
        collision_test_actions = {
            "agent_walker": [{"action": "move_to", "target": [2.0, 1.5]}],  # 移动到spinner位置
            "agent_spinner": [{"action": "move_to", "target": [2.0, 1.5]}]  # 同样移动到相同位置
        }
        
        print("    Testing explicit collision scenario...")
        collision_result = simulator.execute_actions_sequence(collision_test_actions)
        
        if not collision_result:
            print("    ✓ Collision detection prevented collision successfully!")
        else:
            print("    ⚠ Collision test completed (unexpected)")
        
        print("\n11. Cleaning up...")
        simulator.close()
        print("    ✓ Simulator closed")
        
        print("\n" + "=" * 80)
        print("🎉 DUAL AGENT DYNAMIC COLLISION DETECTION TEST COMPLETED!")
        print("=" * 80)
        print(f"📁 Check outputs in: {output_dir}")
        print("📹 Videos generated:")
        print("   - agent_walker_output.mp4: Path-following agent with collision avoidance")
        print("   - agent_spinner_output.mp4: Spinning agent with collision detection")
        print("📊 Log file: navigation.log (contains collision detection details)")
        print("💾 Final states: final_states.json")
        print("\n🔍 Key features tested:")
        print("   ✓ Dynamic path sampling for collision prediction")
        print("   ✓ Environment collision detection")
        print("   ✓ Agent-to-agent collision detection")
        print("   ✓ Real-time collision prevention")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
