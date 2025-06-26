#!/usr/bin/env python3
"""测试修复后的坐标对齐"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

# 添加habitat-lab到路径
sys.path.append('/home/yaoaa/habitat-lab')

from habitat_navigator_app import HabitatNavigatorApp
from PyQt5.QtWidgets import QApplication

# 创建应用程序
app = QApplication([])

try:
    # 使用默认场景创建导航器
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    navigator = HabitatNavigatorApp(scene_path)
    print("成功创建导航应用")
    
    # 检查地图尺寸和比例
    if navigator.simulator.base_map_image:
        map_width, map_height = navigator.simulator.base_map_image.size
        print(f"地图尺寸: {map_width} x {map_height}")
        
        # 重新计算比例
        world_bounds = navigator.simulator.scene_bounds
        world_size_x = world_bounds[1][0] - world_bounds[0][0]
        world_size_z = world_bounds[1][2] - world_bounds[0][2]
        
        scale_x = map_width / world_size_x
        scale_z = map_height / world_size_z
        
        print(f"世界尺寸: {world_size_x:.3f} x {world_size_z:.3f}")
        print(f"新比例: X={scale_x:.2f} pixels/unit, Z={scale_z:.2f} pixels/unit")
        print(f"比例差异: {abs(scale_x - scale_z):.2f}")
        
        if abs(scale_x - scale_z) < 1:
            print("✅ 比例修复成功！")
        else:
            print("❌ 比例仍有问题")
        
        # 快速测试几个位置
        test_positions = [
            np.array([0.0, 1.5, 0.0], dtype=np.float32),   # 原点
            np.array([2.6, 1.5, 0.1], dtype=np.float32),   # 用户测试坐标
        ]
        
        fixed_map = navigator.simulator.base_map_image.copy()
        draw = ImageDraw.Draw(fixed_map)
        
        for i, pos in enumerate(test_positions):
            navigator.simulator.move_agent_to(pos)
            actual_state = navigator.simulator.get_agent_state()
            actual_pos = actual_state.position
            
            map_x, map_y = navigator.simulator.world_to_map_coords(actual_pos)
            print(f"位置 {i+1}: 世界({actual_pos[0]:.1f}, {actual_pos[2]:.1f}) -> 地图({map_x}, {map_y})")
            
            # 在地图上标记
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            radius = 10
            draw.ellipse([map_x-radius, map_y-radius, map_x+radius, map_y+radius], 
                        outline=color, width=3)
            draw.text((map_x+radius+5, map_y-10), f"P{i+1}", fill=color)
        
        # 保存修复后的测试地图
        fixed_map.save('/home/yaoaa/habitat-lab/fixed_alignment_test.png')
        print("保存修复后测试地图: fixed_alignment_test.png")
    
    print("✓ 测试完成")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
