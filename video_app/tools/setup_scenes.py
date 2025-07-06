#!/usr/bin/env python3
"""
Scene Setup and Validation Script
场景设置和验证脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_habitat_sim():
    """检查Habitat-Sim安装"""
    try:
        import habitat_sim
        print(f"✓ Habitat-Sim version: {habitat_sim.__version__}")
        return True
    except ImportError:
        print("✗ Habitat-Sim not found!")
        print("Please install: conda install habitat-sim -c conda-forge")
        return False

def check_test_scenes():
    """检查测试场景可用性"""
    print("Checking test scenes...")
    
    # 可能的场景位置
    possible_locations = [
        "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes",
        "data/scene_datasets/habitat-test-scenes",
        "../data/scene_datasets/habitat-test-scenes",
        "../../data/scene_datasets/habitat-test-scenes"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            scenes = []
            for file in os.listdir(location):
                if file.endswith('.glb'):
                    scenes.append(os.path.join(location, file))
            
            if scenes:
                print(f"✓ Found test scenes in: {location}")
                for scene in scenes:
                    print(f"  - {os.path.basename(scene)}")
                return location, scenes
    
    print("✗ No test scenes found!")
    return None, []

def download_test_scenes():
    """下载测试场景"""
    print("Downloading test scenes...")
    try:
        # 尝试使用habitat_sim下载器
        result = subprocess.run([
            sys.executable, "-m", "habitat_sim.utils.datasets_download", 
            "--uids", "habitat_test_scenes"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Test scenes downloaded successfully")
            return True
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def create_simple_scene():
    """创建一个简单的测试场景配置"""
    print("Creating simple scene configuration...")
    
    # 创建一个最小的场景配置
    scene_dir = Path("data/scene_datasets/simple_test")
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建场景数据集配置
    simple_config = {
        "dataset": "simple_test",
        "stages": [
            {
                "filepath": "data/scene_datasets/simple_test/empty_room.glb",
                "attributes": {
                    "name": "empty_room",
                    "gravity": [0, -9.8, 0]
                }
            }
        ]
    }
    
    import json
    config_file = scene_dir / "simple_test.scene_dataset_config.json"
    with open(config_file, 'w') as f:
        json.dump(simple_config, f, indent=2)
    
    print(f"✓ Created simple scene config: {config_file}")
    return str(config_file)

def main():
    """主函数"""
    print("=" * 60)
    print("Habitat Scene Setup and Validation")
    print("=" * 60)
    
    # 1. 检查Habitat-Sim
    if not check_habitat_sim():
        return 1
    
    # 2. 检查测试场景
    scene_location, scenes = check_test_scenes()
    
    if scenes:
        print(f"\n✅ Test scenes are available!")
        print(f"Location: {scene_location}")
        return 0
    
    # 3. 尝试下载测试场景
    print("\n📥 Attempting to download test scenes...")
    if download_test_scenes():
        # 重新检查
        scene_location, scenes = check_test_scenes()
        if scenes:
            print(f"\n✅ Test scenes downloaded and available!")
            return 0
    
    # 4. 如果下载失败，提供手动指导
    print("\n⚠️ Automatic download failed. Manual setup required:")
    print("\n1. Install test scenes manually:")
    print("   git clone https://github.com/facebookresearch/habitat-sim.git")
    print("   cd habitat-sim")
    print("   python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes")
    
    print("\n2. Or use HM3D dataset:")
    print("   python -m habitat_sim.utils.datasets_download --uids hm3d_minival")
    
    print("\n3. Or check existing Habitat-Lab installation:")
    print("   ls /home/yaoaa/habitat-lab/data/scene_datasets/")
    
    return 1

if __name__ == "__main__":
    exit(main())
