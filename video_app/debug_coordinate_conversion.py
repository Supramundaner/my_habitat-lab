#!/usr/bin/env python3
"""
专门测试坐标转换精度的调试脚本
"""

import sys
import os
sys.path.append('/home/yaoaa/habitat-lab')

# 添加interactive_app的src路径
interactive_app_src = os.path.join('/home/yaoaa/habitat-lab/interactive_app/src')
sys.path.insert(0, interactive_app_src)

from habitat_navigator_app import HabitatSimulator
import numpy as np

def debug_coordinate_conversion():
    """调试坐标转换精度问题"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("调试坐标转换精度问题")
        print("=" * 60)
        
        # 创建模拟器
        simulator = HabitatSimulator(scene_path)
        
        print(f"场景边界: {simulator.scene_bounds}")
        print(f"场景中心: {simulator.scene_center}")
        print(f"地图图像尺寸: {simulator.base_map_image.size}")
        
        # 测试点：使用场景中心
        test_world_pos = simulator.scene_center
        print(f"\n测试世界坐标: {test_world_pos}")
        
        # 正向转换：世界坐标 → 地图坐标
        map_x, map_y = simulator.world_to_map_coords(test_world_pos)
        print(f"转换后的地图坐标: ({map_x}, {map_y})")
        
        # 反向转换：地图坐标 → 世界坐标
        converted_world_pos = simulator.map_coords_to_world(map_x, map_y)
        print(f"反向转换的世界坐标: {converted_world_pos}")
        
        # 计算误差
        position_error = np.linalg.norm(test_world_pos - converted_world_pos)
        print(f"位置误差: {position_error:.6f}m")
        
        # 分别检查X, Y, Z坐标的误差
        x_error = abs(test_world_pos[0] - converted_world_pos[0])
        y_error = abs(test_world_pos[1] - converted_world_pos[1])
        z_error = abs(test_world_pos[2] - converted_world_pos[2])
        
        print(f"X坐标误差: {x_error:.6f}m")
        print(f"Y坐标误差: {y_error:.6f}m")
        print(f"Z坐标误差: {z_error:.6f}m")
        
        # 测试get_position_with_navmesh_height方法
        print(f"\n测试get_position_with_navmesh_height方法:")
        navmesh_pos = simulator.get_position_with_navmesh_height(test_world_pos[0], test_world_pos[2])
        if navmesh_pos is not None:
            print(f"Navmesh位置: {navmesh_pos}")
            navmesh_error = np.linalg.norm(test_world_pos - navmesh_pos)
            print(f"Navmesh方法误差: {navmesh_error:.6f}m")
            
            # 分别检查X, Y, Z坐标的误差
            navmesh_x_error = abs(test_world_pos[0] - navmesh_pos[0])
            navmesh_y_error = abs(test_world_pos[1] - navmesh_pos[1])
            navmesh_z_error = abs(test_world_pos[2] - navmesh_pos[2])
            
            print(f"Navmesh X坐标误差: {navmesh_x_error:.6f}m")
            print(f"Navmesh Y坐标误差: {navmesh_y_error:.6f}m")
            print(f"Navmesh Z坐标误差: {navmesh_z_error:.6f}m")
        else:
            print("Navmesh方法返回None")
        
        # 测试边界点
        print(f"\n测试边界点:")
        boundary_points = [
            simulator.scene_bounds[0],  # 最小角
            simulator.scene_bounds[1],  # 最大角
            np.array([simulator.scene_bounds[0][0], simulator.scene_center[1], simulator.scene_center[2]]),  # X最小
            np.array([simulator.scene_bounds[1][0], simulator.scene_center[1], simulator.scene_center[2]]),  # X最大
        ]
        
        for i, point in enumerate(boundary_points):
            print(f"\n边界点{i+1}: {point}")
            result = simulator.verify_coordinate_conversion(point)
            print(f"  误差: {result['position_error']:.6f}m")
            print(f"  原始: {result['original_world']}")
            print(f"  转换: {result['converted_world']}")
            print(f"  地图坐标: {result['map_coords']}")
        
        # 关闭模拟器
        simulator.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_coordinate_conversion()
