#!/usr/bin/env python3
"""
测试脚本：验证Habitat导航应用程序的核心功能
"""

import os
import sys
import numpy as np

def test_dependencies():
    """测试所有依赖是否正确安装"""
    print("测试依赖安装...")
    
    try:
        import habitat_sim
        print("✓ habitat_sim 导入成功")
    except ImportError as e:
        print(f"✗ habitat_sim 导入失败: {e}")
        return False
    
    try:
        import magnum as mn
        print("✓ magnum 导入成功")
    except ImportError as e:
        print(f"✗ magnum 导入失败: {e}")
        return False
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5 导入成功")
    except ImportError as e:
        print(f"✗ PyQt5 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        from PIL import Image
        print("✓ numpy 和 PIL 导入成功")
    except ImportError as e:
        print(f"✗ numpy/PIL 导入失败: {e}")
        return False
    
    return True

def test_scene_file():
    """测试场景文件是否存在"""
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    
    print(f"\n测试场景文件: {scene_path}")
    
    if os.path.exists(scene_path):
        print("✓ 场景文件存在")
        return scene_path
    else:
        print("✗ 场景文件不存在")
        
        # 尝试查找其他可能的场景文件
        possible_paths = [
            "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/",
            "/home/yaoaa/habitat-lab/data/scene_datasets/",
            "/home/yaoaa/habitat-lab/data/"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到目录: {path}")
                try:
                    files = os.listdir(path)
                    glb_files = [f for f in files if f.endswith('.glb')]
                    if glb_files:
                        print(f"发现.glb文件: {glb_files}")
                        return os.path.join(path, glb_files[0])
                except:
                    pass
        
        return None

def test_simulator_basic():
    """测试模拟器基本功能"""
    scene_path = test_scene_file()
    if not scene_path:
        print("跳过模拟器测试（没有可用的场景文件）")
        return False
    
    print(f"\n测试模拟器基本功能...")
    
    try:
        # 导入应用程序类
        from habitat_navigator_app import HabitatSimulator
        
        # 创建模拟器实例
        sim = HabitatSimulator(scene_path, resolution=(256, 256))
        
        print("✓ 模拟器创建成功")
        print(f"✓ 场景边界: {sim.scene_bounds}")
        print(f"✓ 场景中心: {sim.scene_center}")
        
        # 测试坐标转换
        test_pos = np.array([0, 0, 0])
        map_coords = sim.world_to_map_coords(test_pos)
        print(f"✓ 坐标转换测试: {test_pos} -> {map_coords}")
        
        # 测试导航检查
        is_nav = sim.is_navigable(0, 0)
        print(f"✓ 导航检查测试: 原点可导航 = {is_nav}")
        
        # 清理
        sim.close()
        print("✓ 模拟器清理完成")
        
        return True
        
    except Exception as e:
        print(f"✗ 模拟器测试失败: {e}")
        return False

def test_gui_creation():
    """测试GUI创建（不显示窗口）"""
    print("\n测试GUI创建...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        
        # 创建应用程序实例
        app = QApplication(sys.argv if sys.argv else ['test'])
        app.setAttribute(Qt.AA_X11InitThreads)
        
        print("✓ QApplication 创建成功")
        
        # 测试主窗口类导入
        from habitat_navigator_app import HabitatNavigatorApp
        print("✓ 主窗口类导入成功")
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"✗ GUI测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Habitat导航应用程序 - 功能测试")
    print("=" * 50)
    
    all_passed = True
    
    # 测试依赖
    if not test_dependencies():
        all_passed = False
    
    # 测试场景文件
    if not test_scene_file():
        print("警告: 没有找到场景文件，某些功能可能无法使用")
    
    # 测试模拟器
    if not test_simulator_basic():
        all_passed = False
    
    # 测试GUI
    if not test_gui_creation():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！应用程序已准备就绪。")
        print("\n启动应用程序:")
        print("  ./start_app.sh")
        print("  或")
        print("  python3 habitat_navigator_app.py")
    else:
        print("✗ 某些测试失败，请检查安装和配置。")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
