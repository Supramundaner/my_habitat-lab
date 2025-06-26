#!/usr/bin/env python3
"""诊断坐标对齐问题"""

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
    
    # 获取场景信息
    print(f"\n场景边界: {navigator.simulator.scene_bounds}")
    print(f"场景中心: {navigator.simulator.scene_center}")
    print(f"场景尺寸: {navigator.simulator.scene_size}")
    
    # 测试几个关键位置的坐标转换
    test_positions = [
        # [x, y, z]
        navigator.simulator.scene_center,  # 场景中心
        navigator.simulator.scene_bounds[0],  # 最小边界
        navigator.simulator.scene_bounds[1],  # 最大边界
        np.array([0, 1.5, 0], dtype=np.float32),  # 原点附近
        np.array([2.6, 1.5, 0.1], dtype=np.float32),  # 用户测试坐标
    ]
    
    print(f"\n=== 坐标转换测试 ===")
    for i, pos in enumerate(test_positions):
        # 移动智能体到该位置
        try:
            navigator.simulator.move_agent_to(pos)
            actual_state = navigator.simulator.get_agent_state()
            actual_pos = actual_state.position
            
            # 计算地图坐标
            map_x, map_y = navigator.simulator.world_to_map_coords(actual_pos)
            
            print(f"位置 {i+1}:")
            print(f"  期望世界坐标: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
            print(f"  实际世界坐标: [{actual_pos[0]:7.3f}, {actual_pos[1]:7.3f}, {actual_pos[2]:7.3f}]")
            print(f"  地图像素坐标: ({map_x:4d}, {map_y:4d})")
            
            # 检查是否在地图范围内
            if navigator.simulator.base_map_image:
                map_width, map_height = navigator.simulator.base_map_image.size
                if 0 <= map_x < map_width and 0 <= map_y < map_height:
                    print(f"  地图状态: 在范围内 (地图尺寸: {map_width}x{map_height})")
                else:
                    print(f"  地图状态: 超出范围 (地图尺寸: {map_width}x{map_height})")
            
        except Exception as e:
            print(f"位置 {i+1} 测试失败: {e}")
    
    # 创建一个测试地图来验证坐标系
    print(f"\n=== 生成测试地图 ===")
    if navigator.simulator.base_map_image:
        test_map = navigator.simulator.base_map_image.copy()
        draw = ImageDraw.Draw(test_map)
        
        # 在地图上标记测试位置
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, pos in enumerate(test_positions):
            try:
                map_x, map_y = navigator.simulator.world_to_map_coords(pos)
                color = colors[i % len(colors)]
                
                # 绘制圆圈和编号
                radius = 10
                draw.ellipse([map_x-radius, map_y-radius, map_x+radius, map_y+radius], 
                           outline=color, width=3)
                draw.text((map_x+radius+5, map_y-10), str(i+1), fill=color)
                
                print(f"  在地图上标记位置 {i+1}: ({map_x}, {map_y})")
                
            except Exception as e:
                print(f"  标记位置 {i+1} 失败: {e}")
        
        # 保存测试地图
        test_map.save('/home/yaoaa/habitat-lab/coordinate_alignment_test.png')
        print("保存测试地图: coordinate_alignment_test.png")
    
    # 测试反向转换（如果存在）
    print(f"\n=== 反向转换测试 ===")
    if navigator.simulator.base_map_image:
        map_width, map_height = navigator.simulator.base_map_image.size
        
        # 测试几个地图像素位置
        test_pixels = [
            (map_width//4, map_height//4),    # 左上区域
            (map_width//2, map_height//2),    # 中心
            (3*map_width//4, 3*map_height//4), # 右下区域
        ]
        
        for i, (px, py) in enumerate(test_pixels):
            # 手动反向计算世界坐标
            world_min_x = navigator.simulator.scene_bounds[0][0]
            world_max_x = navigator.simulator.scene_bounds[1][0]
            world_min_z = navigator.simulator.scene_bounds[0][2]
            world_max_z = navigator.simulator.scene_bounds[1][2]
            
            world_x = world_min_x + (px / map_width) * (world_max_x - world_min_x)
            world_z = world_min_z + (py / map_height) * (world_max_z - world_min_z)
            
            print(f"像素 ({px:4d}, {py:4d}) -> 世界坐标 ({world_x:7.3f}, ?, {world_z:7.3f})")
            
            # 验证正向转换
            test_pos = np.array([world_x, 1.5, world_z], dtype=np.float32)
            back_px, back_py = navigator.simulator.world_to_map_coords(test_pos)
            
            error_x = abs(px - back_px)
            error_y = abs(py - back_py)
            print(f"  反向验证: ({back_px:4d}, {back_py:4d}), 误差: ({error_x}, {error_y})")
    
    # 检查地图生成时的相机设置
    print(f"\n=== 地图生成设置检查 ===")
    print(f"正交相机位置计算:")
    camera_height = navigator.simulator.scene_bounds[1][1] + 5.0
    camera_position = [navigator.simulator.scene_center[0], camera_height, navigator.simulator.scene_center[2]]
    print(f"  相机高度: {camera_height}")
    print(f"  相机位置: {camera_position}")
    
    # 检查当前智能体位置和地图显示
    current_state = navigator.simulator.get_agent_state()
    current_map_pos = navigator.simulator.world_to_map_coords(current_state.position)
    print(f"\n当前智能体:")
    print(f"  世界位置: {current_state.position}")
    print(f"  地图位置: {current_map_pos}")
    
    print("\n✓ 坐标对齐诊断完成")
    
except Exception as e:
    print(f"诊断失败: {e}")
    import traceback
    traceback.print_exc()

print("诊断完成")
