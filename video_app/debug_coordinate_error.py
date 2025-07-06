#!/usr/bin/env python3
"""
专门诊断坐标转换误差巨大问题的调试脚本
"""

import sys
import os
import numpy as np

# 添加interactive_app的src路径
interactive_app_src = os.path.join(os.path.dirname(__file__), '../interactive_app/src')
sys.path.insert(0, interactive_app_src)

from habitat_navigator_app import HabitatSimulator

def debug_coordinate_conversion_error():
    """调试坐标转换误差巨大的问题"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("诊断坐标转换误差巨大问题")
        print("=" * 60)
        
        # 创建模拟器
        simulator = HabitatSimulator(scene_path)
        
        print(f"场景边界: {simulator.scene_bounds}")
        print(f"场景中心: {simulator.scene_center}")
        print(f"地图图像尺寸: {simulator.base_map_image.size}")
        print(f"MAP_PADDING常量: LEFT={simulator.MAP_PADDING_LEFT}, RIGHT={simulator.MAP_PADDING_RIGHT}, TOP={simulator.MAP_PADDING_TOP}, BOTTOM={simulator.MAP_PADDING_BOTTOM}")
        
        # 测试场景中心的坐标转换
        test_pos = simulator.scene_center
        print(f"\n测试位置（场景中心）: {test_pos}")
        
        # 详细的坐标转换步骤分解
        print("\n=== 详细坐标转换步骤 ===")
        
        # 1. 世界坐标 -> 地图坐标（正向转换）
        print("1. 世界坐标 -> 地图坐标（正向转换）:")
        
        # 获取地图尺寸信息
        padded_width, padded_height = simulator.base_map_image.size
        original_width = padded_width - simulator.MAP_PADDING_LEFT - simulator.MAP_PADDING_RIGHT
        original_height = padded_height - simulator.MAP_PADDING_TOP - simulator.MAP_PADDING_BOTTOM
        
        print(f"  - 带padding地图尺寸: {padded_width} x {padded_height}")
        print(f"  - 原始地图尺寸: {original_width} x {original_height}")
        
        # 世界坐标范围
        world_min_x = simulator.scene_bounds[0][0]
        world_max_x = simulator.scene_bounds[1][0]
        world_min_z = simulator.scene_bounds[0][2]
        world_max_z = simulator.scene_bounds[1][2]
        
        print(f"  - 世界X范围: {world_min_x:.3f} ~ {world_max_x:.3f}")
        print(f"  - 世界Z范围: {world_min_z:.3f} ~ {world_max_z:.3f}")
        
        # 计算原始图像坐标
        px_in_original = (test_pos[0] - world_min_x) / (world_max_x - world_min_x) * original_width
        py_in_original = (test_pos[2] - world_min_z) / (world_max_z - world_min_z) * original_height
        
        print(f"  - 原始图像坐标: ({px_in_original:.3f}, {py_in_original:.3f})")
        
        # 加上padding
        px = int(px_in_original + simulator.MAP_PADDING_LEFT)
        py = int(py_in_original + simulator.MAP_PADDING_TOP)
        
        print(f"  - 加上padding后: ({px}, {py})")
        
        # 使用函数验证
        map_x, map_y = simulator.world_to_map_coords(test_pos)
        print(f"  - 函数结果: ({map_x}, {map_y})")
        
        if map_x != px or map_y != py:
            print("  ⚠️ 函数结果与手动计算不一致!")
        else:
            print("  ✓ 函数结果与手动计算一致")
        
        # 2. 地图坐标 -> 世界坐标（反向转换）
        print("\n2. 地图坐标 -> 世界坐标（反向转换）:")
        
        # 减去padding
        px_in_original_back = map_x - simulator.MAP_PADDING_LEFT
        py_in_original_back = map_y - simulator.MAP_PADDING_TOP
        
        print(f"  - 减去padding后: ({px_in_original_back:.3f}, {py_in_original_back:.3f})")
        
        # 转换回世界坐标
        world_x_back = world_min_x + (px_in_original_back / original_width) * (world_max_x - world_min_x)
        world_z_back = world_min_z + (py_in_original_back / original_height) * (world_max_z - world_min_z)
        
        print(f"  - 计算得到的世界坐标: ({world_x_back:.6f}, ?, {world_z_back:.6f})")
        
        # 使用函数验证
        converted_world_pos = simulator.map_coords_to_world(map_x, map_y)
        print(f"  - 函数结果: ({converted_world_pos[0]:.6f}, {converted_world_pos[1]:.6f}, {converted_world_pos[2]:.6f})")
        
        # 3. 计算误差
        print("\n3. 误差分析:")
        
        x_error = abs(test_pos[0] - converted_world_pos[0])
        y_error = abs(test_pos[1] - converted_world_pos[1])
        z_error = abs(test_pos[2] - converted_world_pos[2])
        
        print(f"  - X坐标误差: {x_error:.6f}m")
        print(f"  - Y坐标误差: {y_error:.6f}m")
        print(f"  - Z坐标误差: {z_error:.6f}m")
        
        # 使用verify_coordinate_conversion函数
        result = simulator.verify_coordinate_conversion(test_pos)
        print(f"  - 验证函数报告的误差: {result['position_error']:.6f}m")
        print(f"  - 误差可接受: {'是' if result['error_acceptable'] else '否'}")
        
        # 4. 深入分析Y坐标问题
        print("\n4. Y坐标问题分析:")
        
        print(f"  - 原始Y坐标: {test_pos[1]:.6f}")
        print(f"  - 转换后Y坐标: {converted_world_pos[1]:.6f}")
        print(f"  - Y坐标差异: {y_error:.6f}m")
        
        # 检查navmesh snap的影响
        try:
            import magnum as mn
            test_point = mn.Vector3(test_pos[0], 0.0, test_pos[2])
            snapped_point = simulator.sim.pathfinder.snap_point(test_point)
            print(f"  - Navmesh snap结果: ({snapped_point.x:.6f}, {snapped_point.y:.6f}, {snapped_point.z:.6f})")
            
            snap_x_error = abs(test_pos[0] - snapped_point.x)
            snap_z_error = abs(test_pos[2] - snapped_point.z)
            print(f"  - Navmesh X偏移: {snap_x_error:.6f}m")
            print(f"  - Navmesh Z偏移: {snap_z_error:.6f}m")
            
        except Exception as e:
            print(f"  - Navmesh测试失败: {e}")
        
        # 5. 结论和建议
        print("\n5. 结论和建议:")
        
        if result['position_error'] > 1.0:
            print("  ❌ 误差超过1米，属于严重问题")
            
            # 检查是否Y坐标导致
            xy_error = np.linalg.norm([x_error, z_error])
            print(f"  - 仅考虑X,Z坐标的误差: {xy_error:.6f}m")
            
            if xy_error < 0.1 and y_error > 1.0:
                print("  💡 问题主要由Y坐标引起，X,Z坐标转换正常")
                print("  💡 建议：修改验证函数，只检查X,Z坐标误差")
            else:
                print("  💡 X,Z坐标转换也存在问题，需要进一步调试")
                
        elif result['position_error'] > 0.1:
            print("  ⚠️ 误差在0.1~1米之间，需要优化")
        else:
            print("  ✅ 误差在可接受范围内")
        
        # 6. 测试修正版本的验证
        print("\n6. 测试修正版本的验证:")
        
        # 只计算X,Z坐标的误差
        xz_error = np.linalg.norm([x_error, z_error])
        print(f"  - 仅X,Z坐标误差: {xz_error:.6f}m")
        print(f"  - 修正版本可接受: {'是' if xz_error < 0.1 else '否'}")
        
        # 关闭模拟器
        simulator.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_coordinate_conversion_error()
