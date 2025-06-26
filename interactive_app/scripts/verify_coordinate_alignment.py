#!/usr/bin/env python3
"""精确的坐标对齐验证"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    
    # 定义测试位置 - 选择易于识别的位置
    test_positions = [
        np.array([0.0, 1.5, 0.0], dtype=np.float32),   # 原点
        np.array([3.0, 1.5, 0.0], dtype=np.float32),   # X轴正方向
        np.array([0.0, 1.5, 2.0], dtype=np.float32),   # Z轴正方向
        np.array([-2.0, 1.5, -1.0], dtype=np.float32), # 负象限
        np.array([2.6, 1.5, 0.1], dtype=np.float32),   # 用户测试坐标
    ]
    
    position_names = ["原点", "X轴正", "Z轴正", "负象限", "用户测试"]
    
    print(f"\n=== 详细坐标对齐测试 ===")
    print(f"场景边界: {navigator.simulator.scene_bounds}")
    
    # 获取基础地图
    base_map = navigator.simulator.base_map_image.copy()
    draw = ImageDraw.Draw(base_map)
    
    # 为每个位置生成详细信息
    for i, (pos, name) in enumerate(zip(test_positions, position_names)):
        print(f"\n--- 位置 {i+1}: {name} ---")
        
        try:
            # 移动智能体到位置
            navigator.simulator.move_agent_to(pos)
            actual_state = navigator.simulator.get_agent_state()
            actual_pos = actual_state.position
            
            print(f"期望位置: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
            print(f"实际位置: [{actual_pos[0]:7.3f}, {actual_pos[1]:7.3f}, {actual_pos[2]:7.3f}]")
            
            # 计算地图坐标
            map_x, map_y = navigator.simulator.world_to_map_coords(actual_pos)
            print(f"地图像素: ({map_x:4d}, {map_y:4d})")
            
            # 检查是否可导航
            navigable = navigator.simulator.is_navigable(actual_pos[0], actual_pos[2])
            print(f"可导航性: {'是' if navigable else '否'}")
            
            # 获取该位置的FPV图像
            fpv_obs = navigator.simulator.get_fpv_observation()
            if len(fpv_obs.shape) == 3 and fpv_obs.shape[2] == 4:
                fpv_rgb = fpv_obs[:, :, :3]
            else:
                fpv_rgb = fpv_obs
            
            # 保存FPV图像
            fpv_image = Image.fromarray(fpv_rgb, 'RGB')
            fpv_image.save(f'/home/yaoaa/habitat-lab/fpv_position_{i+1}_{name}.png')
            print(f"保存FPV图像: fpv_position_{i+1}_{name}.png")
            
            # 在地图上标记位置
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i]
            
            # 绘制大圆圈
            radius = 15
            draw.ellipse([map_x-radius, map_y-radius, map_x+radius, map_y+radius], 
                        outline=color, width=4)
            
            # 绘制中心点
            draw.ellipse([map_x-3, map_y-3, map_x+3, map_y+3], fill=color)
            
            # 添加文字标签
            draw.text((map_x+radius+5, map_y-15), f"{i+1}:{name}", fill=color)
            draw.text((map_x+radius+5, map_y), f"({actual_pos[0]:.1f},{actual_pos[2]:.1f})", fill=color)
            
            # 检查地图坐标是否合理
            map_width, map_height = base_map.size
            if 0 <= map_x < map_width and 0 <= map_y < map_height:
                print(f"地图状态: ✓ 在有效范围内")
            else:
                print(f"地图状态: ✗ 超出范围 (地图: {map_width}x{map_height})")
                
        except Exception as e:
            print(f"位置 {i+1} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存标记了所有位置的地图
    base_map.save('/home/yaoaa/habitat-lab/alignment_verification_map.png')
    print(f"\n保存对齐验证地图: alignment_verification_map.png")
    
    # 检查坐标系方向是否正确
    print(f"\n=== 坐标系方向检查 ===")
    
    # 测试X轴方向：从(0,0)到(1,0)应该在地图上向右移动
    pos1 = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    pos2 = np.array([1.0, 1.5, 0.0], dtype=np.float32)
    
    map1_x, map1_y = navigator.simulator.world_to_map_coords(pos1)
    map2_x, map2_y = navigator.simulator.world_to_map_coords(pos2)
    
    print(f"X轴测试:")
    print(f"  世界坐标 (0,0) -> 地图像素 ({map1_x}, {map1_y})")
    print(f"  世界坐标 (1,0) -> 地图像素 ({map2_x}, {map2_y})")
    print(f"  X方向: {'正确 (向右)' if map2_x > map1_x else '错误 (向左)'}")
    
    # 测试Z轴方向：从(0,0)到(0,1)应该在地图上向下移动（或向上，取决于约定）
    pos3 = np.array([0.0, 1.5, 1.0], dtype=np.float32)
    map3_x, map3_y = navigator.simulator.world_to_map_coords(pos3)
    
    print(f"Z轴测试:")
    print(f"  世界坐标 (0,0) -> 地图像素 ({map1_x}, {map1_y})")
    print(f"  世界坐标 (0,1) -> 地图像素 ({map3_x}, {map3_y})")
    print(f"  Z方向变化: ΔY = {map3_y - map1_y} (正值=向下, 负值=向上)")
    
    # 计算坐标系的比例
    world_bounds = navigator.simulator.scene_bounds
    world_size_x = world_bounds[1][0] - world_bounds[0][0]
    world_size_z = world_bounds[1][2] - world_bounds[0][2]
    map_width, map_height = base_map.size
    
    scale_x = map_width / world_size_x
    scale_z = map_height / world_size_z
    
    print(f"\n=== 比例检查 ===")
    print(f"世界尺寸: {world_size_x:.3f} x {world_size_z:.3f}")
    print(f"地图尺寸: {map_width} x {map_height}")
    print(f"比例: X={scale_x:.2f} pixels/unit, Z={scale_z:.2f} pixels/unit")
    print(f"比例差异: {abs(scale_x - scale_z):.2f} ({'保持比例' if abs(scale_x - scale_z) < 1 else '拉伸变形'})")
    
    print(f"\n✓ 坐标对齐验证完成")
    
    # 给出对齐分析结论
    print(f"\n=== 对齐分析结论 ===")
    if abs(scale_x - scale_z) > 1:
        print("❌ 问题：地图X和Z轴比例不一致，可能导致位置偏差")
    else:
        print("✅ 地图比例正确")
        
    if map2_x > map1_x:
        print("✅ X轴方向正确")
    else:
        print("❌ 问题：X轴方向错误")
        
    print("📌 建议检查生成的FPV图像和地图标记，确认视觉对应关系")
    
except Exception as e:
    print(f"验证失败: {e}")
    import traceback
    traceback.print_exc()

print("验证完成")
