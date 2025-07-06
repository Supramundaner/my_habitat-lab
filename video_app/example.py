#!/usr/bin/env python3
"""
Simple Multi-Agent Navigation Example
简单的多智能体导航示例
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def run_simple_example():
    """运行简单示例"""
    print("=" * 60)
    print("Simple Multi-Agent Navigation Example")
    print("=" * 60)
    
    # 创建临时输出目录
    output_dir = "./example_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用默认配置文件
    config_file = project_root / "config" / "multi_agent_config.yaml"
    actions_file = project_root / "config" / "actions_example.json"
    
    # 检查文件是否存在
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        print("Please run: python launcher.py --create-sample")
        return False
    
    if not actions_file.exists():
        print(f"Error: Actions file not found: {actions_file}")
        print("Please run: python launcher.py --create-sample")
        return False
    
    try:
        # 导入模块
        from multi_agent_navigation import MultiAgentSimulator
        import yaml
        
        # 加载配置
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 修改配置以使用较小的设置（适合示例）
        config["video_output"]["output_dir"] = output_dir
        config["video_output"]["fps"] = 15
        config["video_output"]["resolution"] = [512, 1024]
        config["logging"]["log_file"] = os.path.join(output_dir, "example.log")
        config["state_persistence"]["state_file"] = os.path.join(output_dir, "states.json")
        
        # 使用可靠的测试场景路径
        possible_scenes = [
            "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        ]
        
        scene_path = None
        for scene in possible_scenes:
            if os.path.exists(scene):
                scene_path = os.path.abspath(scene)
                break
        
        if scene_path:
            config["scene"]["scene_dataset_path"] = scene_path
            print(f"   Using scene: {scene_path}")
        else:
            print("   Warning: No test scenes found, using default path")
            config["scene"]["scene_dataset_path"] = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
        
        # 创建模拟器
        print("1. Initializing multi-agent simulator...")
        simulator = MultiAgentSimulator(config)
        print("   ✓ Simulator initialized")
        
        # 加载动作
        print("2. Loading actions...")
        actions = simulator.load_actions_from_file(str(actions_file))
        print(f"   ✓ Loaded actions for {len(actions)} agents")
        
        # 执行模拟
        print("3. Executing simulation...")
        success = simulator.execute_actions_sequence(actions)
        
        if success:
            print("   ✓ Simulation completed successfully")
            print(f"\n📁 Output files:")
            print(f"   - Videos: {output_dir}/agent_*_output.mp4")
            print(f"   - Log: {output_dir}/example.log")
            print(f"   - States: {output_dir}/states.json")
        else:
            print("   ⚠ Simulation completed with issues")
        
        # 清理
        simulator.close()
        print("4. Simulator closed")
        
        return success
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("Multi-Agent Habitat Navigation - Simple Example")
    
    # 运行示例
    success = run_simple_example()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated videos in ./example_outputs/")
        print("2. Modify config/actions_example.json to create your own scenarios")
        print("3. Run: python launcher.py -c your_config.yaml -a your_actions.json")
    else:
        print("\n❌ Example failed!")
        print("\nTroubleshooting:")
        print("1. Make sure Habitat-Sim is properly installed")
        print("2. Check that test scenes are available")
        print("3. Run the test script: python test_system.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
