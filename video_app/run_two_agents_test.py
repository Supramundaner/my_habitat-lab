#!/usr/bin/env python3
"""
双智能体测试脚本
测试两个Fetch机器人在apartment_1.glb环境中的多智能体导航
"""

import os
import sys
import yaml
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """主函数"""
    print("=" * 70)
    print("DUAL AGENT NAVIGATION TEST")
    print("=" * 70)
    print("Testing two Fetch robots in apartment_1.glb:")
    print("- Agent Walker: Following path [4.0,3.0] → [0.5,3.0] → [0.5,0.0] → [2.0,1.0] + 360° turn")
    print("- Agent Spinner: Move to [2.0,1.5] then spin 3 times (3 × 360°)")
    print()
    
    try:
        from multi_agent_navigation import MultiAgentSimulator
        
        # 配置文件路径
        config_path = "config/two_agents_test_config.yaml"
        actions_path = "two_agents_actions.json"
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return 1
            
        if not os.path.exists(actions_path):
            print(f"❌ Actions file not found: {actions_path}")
            return 1
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        with open(actions_path, 'r') as f:
            actions = json.load(f)
        
        print("1. Initializing dual-agent simulator...")
        simulator = MultiAgentSimulator(config)
        print("   ✓ Simulator initialized")
        
        print("\\n2. Checking agent status...")
        simulator.print_agent_status()
        
        # 验证两个物理机器人都已加载
        report = simulator.get_agent_status_report()
        
        walker_info = report["agents"].get("agent_walker", {})
        spinner_info = report["agents"].get("agent_spinner", {})
        
        if not walker_info.get("has_physical_robot"):
            print("   ⚠ Walker agent: Physical robot not loaded")
        else:
            print(f"   ✓ Walker agent: {walker_info['robot_joint_count']} joints")
            
        if not spinner_info.get("has_physical_robot"):
            print("   ⚠ Spinner agent: Physical robot not loaded")
        else:
            print(f"   ✓ Spinner agent: {spinner_info['robot_joint_count']} joints")
        
        print("\\n3. Executing dual-agent action sequence...")
        print("   → Walker: Following path trajectory")
        print("   → Spinner: Moving to position then spinning")
        
        success = simulator.execute_actions_sequence(actions)
        
        if success:
            print("   ✓ All actions executed successfully")
        else:
            print("   ⚠ Some actions may have failed")
        
        print("\\n4. Checking output files...")
        output_dir = config["video_output"]["output_dir"]
        
        # 检查两个视频文件
        walker_video = os.path.join(output_dir, "agent_walker_output.mp4")
        spinner_video = os.path.join(output_dir, "agent_spinner_output.mp4")
        
        for agent_name, video_path in [("Walker", walker_video), ("Spinner", spinner_video)]:
            if os.path.exists(video_path):
                size = os.path.getsize(video_path)
                print(f"   ✓ {agent_name} video: {size} bytes")
            else:
                print(f"   ❌ {agent_name} video not found")
        
        # 显示所有输出文件
        print("\\n5. Generated files:")
        if os.path.exists(output_dir):
            for filename in sorted(os.listdir(output_dir)):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    print(f"   📁 {filename}: {size} bytes")
        
        print("\\n6. Final agent positions:")
        final_report = simulator.get_agent_status_report()
        for agent_id, agent_info in final_report["agents"].items():
            pos = agent_info.get("current_position", "N/A")
            if pos != "N/A":
                print(f"   {agent_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        print("\\n7. Cleaning up...")
        simulator.close()
        print("   ✓ Simulator closed")
        
        print("\\n" + "=" * 70)
        print("🎉 DUAL AGENT NAVIGATION TEST COMPLETED!")
        print("=" * 70)
        print(f"📁 Check outputs in: {output_dir}")
        print("📹 Two videos generated:")
        print("   - agent_walker_output.mp4: Path-following agent")
        print("   - agent_spinner_output.mp4: Spinning agent")
        print("=" * 70)
        
        return 0
        config_file = "config/two_physical_agents_config.yaml"
        actions_file = "test_two_agents_actions.json"
        
        # 检查文件是否存在
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            return 1
            
        if not os.path.exists(actions_file):
            print(f"❌ Actions file not found: {actions_file}")
            return 1
        
        # 加载配置
        print("1. Loading configuration...")
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Config loaded from: {config_file}")
        
        # 创建输出目录
        output_dir = config["video_output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        print(f"   ✓ Output directory: {output_dir}")
        
        # 初始化模拟器
        print("\\n2. Initializing multi-agent simulator...")
        simulator = MultiAgentSimulator(config)
        print("   ✓ Simulator initialized")
        
        # 打印智能体状态
        print("\\n3. Physical agents status:")
        simulator.print_agent_status()
        
        # 验证物理机器人是否成功加载
        report = simulator.get_agent_status_report()
        physical_agents_count = sum(1 for agent_info in report["agents"].values() 
                                  if agent_info["has_physical_robot"])
        
        print(f"\\n4. Physical agents validation:")
        print(f"   Total agents: {report['total_agents']}")
        print(f"   Physical robots loaded: {physical_agents_count}")
        
        if physical_agents_count != 2:
            print("   ⚠ Warning: Not all agents have physical robots loaded!")
            print("   → Continuing with available agents...")
        else:
            print("   ✓ All agents have physical robots loaded!")
        
        # 加载动作序列
        print("\\n5. Loading action sequences...")
        actions = simulator.load_actions_from_file(actions_file)
        
        total_actions = sum(len(agent_actions) for agent_actions in actions.values())
        print(f"   ✓ Loaded {len(actions)} agent action sequences")
        print(f"   ✓ Total actions: {total_actions}")
        
        # 显示动作计划
        print("\\n6. Action plan:")
        for agent_id, agent_actions in actions.items():
            print(f"   {agent_id}:")
            for i, action in enumerate(agent_actions, 1):
                if action.action == "move_to":
                    print(f"     {i}. Move to ({action.target[0]}, {action.target[1]})")
                elif action.action == "turn_left":
                    print(f"     {i}. Turn left {action.angle}°")
                elif action.action == "turn_right":
                    print(f"     {i}. Turn right {action.angle}°")
        
        # 执行模拟
        print("\\n7. Executing navigation simulation...")
        print("   (This may take several minutes depending on action complexity)")
        
        success = simulator.execute_actions_sequence(actions)
        
        if success:
            print("   ✓ Simulation completed successfully!")
        else:
            print("   ⚠ Simulation completed with warnings")
        
        # 显示输出文件
        print("\\n8. Generated outputs:")
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"   📁 {filename} ({size_mb:.1f} MB)")
        
        # 清理
        print("\\n9. Cleaning up...")
        simulator.close()
        print("   ✓ Simulator closed")
        
        print("\\n" + "=" * 70)
        print("🎉 TWO PHYSICAL AGENTS TEST COMPLETED!")
        print("=" * 70)
        print(f"📁 Check outputs in: {output_dir}")
        print("📹 Video files: agent_0_output.mp4, agent_1_output.mp4")
        print("📊 Log file: two_agents_navigation.log")
        print("💾 State file: agent_states.json")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
