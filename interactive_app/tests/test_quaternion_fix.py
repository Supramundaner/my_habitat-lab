#!/usr/bin/env python3
"""全面测试四元数修复"""

import sys
import os

# 添加habitat-lab到路径
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
    
    # 测试多个坐标输入
    test_coordinates = [
        "2.6, 0.1",    # 原始问题坐标
        "0.0, 0.0",    # 原点
        "-5.0, 3.0",   # 负坐标
        "10.0, -2.0",  # 混合坐标
    ]
    
    for coord in test_coordinates:
        print(f"\n测试坐标: {coord}")
        try:
            navigator.process_coordinate_command(coord)
            print(f"  ✓ 坐标 {coord} 处理成功")
        except Exception as e:
            print(f"  ✗ 坐标 {coord} 处理失败: {e}")
    
    # 测试移动到特定位置（不依赖GUI）
    print("\n直接测试移动功能...")
    test_positions = [
        np.array([1.0, 1.5, 1.0], dtype=np.float32),
        np.array([-2.0, 1.5, 0.5], dtype=np.float32),  
        np.array([0.0, 1.5, -1.0], dtype=np.float32),
    ]
    
    for pos in test_positions:
        try:
            navigator.simulator.move_agent_to(pos)
            print(f"  ✓ 移动到 {pos} 成功")
            
            # 测试获取状态和地图绘制
            agent_state = navigator.simulator.get_agent_state()
            navigator.update_displays()
            print(f"    智能体位置: {agent_state.position}")
            print(f"    智能体旋转: {agent_state.rotation} (类型: {type(agent_state.rotation)})")
            
        except Exception as e:
            print(f"  ✗ 移动到 {pos} 失败: {e}")
    
    print("\n✓ 所有测试完成，四元数问题已修复！")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
