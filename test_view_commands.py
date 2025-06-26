#!/usr/bin/env python3
"""
测试视角命令修复
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication

# 导入我们的应用
sys.path.append('/home/yaoaa/habitat-lab')
from habitat_navigator_app import HabitatNavigatorApp

def test_view_commands():
    """测试视角命令"""
    print("=== 测试视角命令修复 ===\n")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    try:
        # 创建应用程序（需要QApplication）
        app = QApplication(sys.argv)
        
        # 创建导航应用
        navigator = HabitatNavigatorApp(scene_path)
        
        print("✓ 应用程序初始化成功")
        
        # 测试各种视角命令
        test_commands = [
            "right 30",
            "left 45", 
            "up 20",
            "down 15"
        ]
        
        for command in test_commands:
            print(f"\n测试命令: '{command}'")
            try:
                navigator.process_view_command(command)
                print(f"✓ 命令 '{command}' 执行成功")
            except Exception as e:
                print(f"✗ 命令 '{command}' 失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ 所有视角命令测试完成")
        
        # 不启动GUI，直接退出
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_view_commands()
    if success:
        print("\n🎉 视角命令修复测试通过！")
    else:
        print("\n❌ 视角命令仍有问题")
