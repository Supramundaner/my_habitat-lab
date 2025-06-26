#!/usr/bin/env python3
"""简单测试FPV显示修复"""

import sys
import os
sys.path.append('/home/yaoaa/habitat-lab')

from habitat_navigator_app import HabitatNavigatorApp
from PyQt5.QtWidgets import QApplication
import numpy as np

# 创建应用程序
app = QApplication([])

try:
    # 使用默认场景创建导航器
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    navigator = HabitatNavigatorApp(scene_path)
    print("成功创建导航应用")
    
    # 移动到一个位置
    test_pos = np.array([1.0, 1.5, 1.0], dtype=np.float32)
    navigator.simulator.move_agent_to(test_pos)
    
    # 更新FPV显示
    print("测试FPV显示更新...")
    navigator.update_fpv_display()
    print("✓ FPV显示更新完成，无错误")
    
    # 测试几个不同位置
    positions = [
        np.array([0.0, 1.5, 0.0], dtype=np.float32),
        np.array([2.0, 1.5, 1.0], dtype=np.float32),
        np.array([-1.0, 1.5, -1.0], dtype=np.float32),
    ]
    
    for i, pos in enumerate(positions):
        navigator.simulator.move_agent_to(pos)
        navigator.update_fpv_display()
        print(f"✓ 位置 {i+1} FPV更新成功")
    
    print("\n所有FPV测试通过！")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
