#!/usr/bin/env python3
"""测试应用程序的坐标输入功能"""

import sys
import os

# 添加habitat-lab到路径
sys.path.append('/home/yaoaa/habitat-lab')

# 测试导入
try:
    from habitat_navigator_app import HabitatNavigatorApp
    print("成功导入HabitatNavigatorApp")
except Exception as e:
    print(f"导入失败: {e}")
    exit(1)

from PyQt5.QtWidgets import QApplication

# 创建应用程序
app = QApplication([])

try:
    # 使用默认场景创建导航器
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/hm3d_v0.1/train/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.glb"
    if not os.path.exists(scene_path):
        print(f"场景文件不存在: {scene_path}")
        # 尝试找到其他场景文件
        import glob
        possible_scenes = glob.glob("/home/yaoaa/habitat-lab/data/scene_datasets/**/*.glb", recursive=True)
        if possible_scenes:
            scene_path = possible_scenes[0]
            print(f"使用场景: {scene_path}")
        else:
            print("没有找到可用的场景文件")
            exit(1)
    
    navigator = HabitatNavigatorApp(scene_path)
    print("成功创建导航应用")
    
    # 测试坐标输入
    print("测试坐标输入...")
    navigator.command_input.setText("2.6, 0.1")
    # 直接调用处理方法而不是点击按钮
    navigator.process_coordinate_command("2.6, 0.1")
    print("坐标输入测试完成")
    
except Exception as e:
    print(f"创建应用失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
