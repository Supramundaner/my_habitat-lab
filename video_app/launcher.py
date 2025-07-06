#!/usr/bin/env python3
"""
Multi-Agent Navigation Launcher
多智能体导航系统启动器
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_sample_config():
    """创建示例配置文件"""
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    sample_config = config_dir / "multi_agent_config.yaml"
    sample_actions = config_dir / "actions_example.json"
    
    if sample_config.exists() and sample_actions.exists():
        print(f"Sample files already exist:")
        print(f"  Config: {sample_config}")
        print(f"  Actions: {sample_actions}")
        return str(sample_config), str(sample_actions)
    
    # 配置文件和动作文件已经在之前创建了
    return str(sample_config), str(sample_actions)

def validate_paths(config_path, actions_path):
    """验证路径有效性"""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return False
    
    if not os.path.exists(actions_path):
        print(f"Error: Actions file not found: {actions_path}")
        return False
    
    return True

def run_test_system():
    """运行系统测试"""
    test_script = project_root / "test_system.py"
    if test_script.exists():
        print("Running system tests...")
        result = subprocess.run([sys.executable, str(test_script)], 
                              capture_output=False)
        return result.returncode == 0
    else:
        print("Test script not found.")
        return False

def run_example():
    """运行示例"""
    example_script = project_root / "example.py"
    if example_script.exists():
        print("Running example...")
        result = subprocess.run([sys.executable, str(example_script)], 
                              capture_output=False)
        return result.returncode == 0
    else:
        print("Example script not found.")
        return False

def view_states(state_file):
    """查看智能体状态"""
    state_viewer = project_root / "tools" / "state_viewer.py"
    if state_viewer.exists():
        subprocess.run([sys.executable, str(state_viewer), state_file])
    else:
        print("State viewer not found.")

def check_and_fix_scene_paths():
    """检查并修复场景路径"""
    print("Checking scene file availability...")
    
    # 常见的测试场景路径
    scene_candidates = [
        "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
        "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    ]
    
    available_scenes = []
    for scene_path in scene_candidates:
        if os.path.exists(scene_path):
            available_scenes.append(os.path.abspath(scene_path))
            print(f"  ✓ Found: {scene_path}")
    
    if not available_scenes:
        print("  ⚠ No test scenes found!")
        print("  Please download test scenes:")
        print("    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes")
        return None
    
    return available_scenes[0]  # 返回第一个可用的场景

def fix_config_scene_path(config_path):
    """修复配置文件中的场景路径"""
    try:
        # 检查可用场景
        valid_scene = check_and_fix_scene_paths()
        if not valid_scene:
            return False
        
        # 读取配置文件
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新场景路径
        old_path = config["scene"]["scene_dataset_path"]
        config["scene"]["scene_dataset_path"] = valid_scene
        
        # 写回配置文件
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"  Updated scene path from {old_path} to {valid_scene}")
        return True
        
    except Exception as e:
        print(f"  Failed to fix config: {e}")
        return False

def main_launcher():
    """主启动函数"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Habitat Navigation System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample files
  python launcher.py --create-sample
  
  # Run system tests
  python launcher.py --test
  
  # Run simple example
  python launcher.py --example
  
  # Run with default config and actions
  python launcher.py
  
  # Run with custom config and actions
  python launcher.py -c my_config.yaml -a my_actions.json
  
  # Resume from saved state
  python launcher.py -r ./outputs/agent_states.json
  
  # View saved states
  python launcher.py --view-states ./outputs/agent_states.json
  
  # Interactive state editor
  python launcher.py --edit-states ./outputs/agent_states.json
        """
    )
    
    # 主要功能选项
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample config and actions files")
    parser.add_argument("--test", action="store_true",
                       help="Run system tests")
    parser.add_argument("--example", action="store_true",
                       help="Run simple example")
    parser.add_argument("--setup-scenes", action="store_true",
                       help="Download and setup test scenes")
    parser.add_argument("--check-scenes", action="store_true",
                       help="Check scene availability")
    
    # 运行参数
    parser.add_argument("--config", "-c", 
                       help="Path to configuration YAML file")
    parser.add_argument("--actions", "-a",
                       help="Path to actions JSON file")
    parser.add_argument("--resume-state", "-r",
                       help="Path to saved state file to resume from")
    
    # 工具选项
    parser.add_argument("--view-states", 
                       help="View agent states from file")
    parser.add_argument("--edit-states",
                       help="Interactive state editor")
    
    # 其他选项
    parser.add_argument("--scene", "-s",
                       help="Override scene path in config")
    parser.add_argument("--output-dir", "-o", default="./outputs",
                       help="Output directory for videos and logs")
    
    args = parser.parse_args()
    
    # 创建示例文件
    if args.create_sample:
        config_path, actions_path = create_sample_config()
        print(f"Sample files:")
        print(f"  Config: {config_path}")
        print(f"  Actions: {actions_path}")
        print(f"\nNext steps:")
        print(f"  1. Run tests: python launcher.py --test")
        print(f"  2. Run example: python launcher.py --example")
        print(f"  3. Run full system: python launcher.py")
        return 0
    
    # 运行测试
    if args.test:
        success = run_test_system()
        if success:
            print("\n✓ All tests passed! System is ready.")
        else:
            print("\n✗ Tests failed. Please check the output above.")
        return 0 if success else 1
    
    # 运行示例
    if args.example:
        success = run_example()
        return 0 if success else 1
    
    # 查看状态
    if args.view_states:
        view_states(args.view_states)
        return 0
    
    # 编辑状态
    if args.edit_states:
        state_viewer = project_root / "tools" / "state_viewer.py"
        if state_viewer.exists():
            subprocess.run([sys.executable, str(state_viewer), 
                          args.edit_states, "--interactive"])
        return 0
    
    # 运行主程序
    config_path = args.config or str(project_root / "config" / "multi_agent_config.yaml")
    actions_path = args.actions or str(project_root / "config" / "actions_example.json")
    
    # 验证文件存在
    if not validate_paths(config_path, actions_path):
        print("\nTo create sample files, run:")
        print("  python launcher.py --create-sample")
        print("\nTo run tests first:")
        print("  python launcher.py --test")
        return 1
    
    # 检查并修复场景路径
    print("\n" + "=" * 70)
    print("🔍 Scene Path Validation")
    print("=" * 70)
    
    scene_fixed = fix_config_scene_path(config_path)
    if not scene_fixed:
        print("\n❌ Scene validation failed!")
        print("Please ensure test scenes are available.")
        print("Run: python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes")
        return 1
    
    print("✅ Scene path validated and fixed if needed")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 修复场景路径
    fix_config_scene_path(config_path)
    
    # 准备系统参数
    sys.argv = [
        "multi_agent_navigation.py",
        "--config", config_path,
        "--actions", actions_path
    ]
    
    if args.resume_state:
        sys.argv.extend(["--resume-state", args.resume_state])
    
    print("=" * 70)
    print("🤖 Multi-Agent Habitat Navigation System")
    print("=" * 70)
    print(f"Config file:      {config_path}")
    print(f"Actions file:     {actions_path}")
    print(f"Output directory: {args.output_dir}")
    
    if args.resume_state:
        print(f"Resuming from:    {args.resume_state}")
    
    if args.scene:
        print(f"Scene override:   {args.scene}")
    
    print("=" * 70)
    
    try:
        # 导入并调用主函数
        from multi_agent_navigation import main
        result = main()
        
        print("\n" + "=" * 70)
        if result == 0:
            print("🎉 SUCCESS: Multi-agent navigation completed!")
            print(f"\n📁 Check output files in: {args.output_dir}")
            print("   - agent_*_output.mp4 (videos)")
            print("   - multi_agent_nav.log (execution log)")
            print("   - agent_states.json (final states)")
            print("\n🔧 Useful commands:")
            print(f"   View states: python launcher.py --view-states {args.output_dir}/agent_states.json")
            print(f"   Edit states: python launcher.py --edit-states {args.output_dir}/agent_states.json")
        else:
            print("❌ FAILED: Multi-agent navigation failed!")
            print(f"\n🔍 Check the log file: {args.output_dir}/multi_agent_nav.log")
            print("\n🔧 Debugging tips:")
            print("   1. Run tests: python launcher.py --test")
            print("   2. Try example: python launcher.py --example")
            print("   3. Check Habitat-Sim installation")
        print("=" * 70)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main_launcher())
