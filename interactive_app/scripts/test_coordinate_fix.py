#!/usr/bin/env python3
"""
坐标对齐修复验证脚本

此脚本用于验证坐标转换修复是否解决了0.3m偏移问题。
将测试多个已知世界坐标点的转换精度。
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from habitat_navigator_app import HabitatSimulator

def test_coordinate_accuracy():
    """测试坐标转换精度"""
    print("=== 坐标对齐修复验证 ===\n")
    
    # 使用测试场景
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    
    if not os.path.exists(scene_path):
        print(f"错误: 场景文件不存在: {scene_path}")
        print("请修改scene_path变量或确保场景文件存在")
        return
    
    try:
        # 初始化模拟器
        print("正在初始化模拟器...")
        simulator = HabitatSimulator(scene_path)
        print("模拟器初始化完成\n")
        
        # 获取场景边界信息
        bounds = simulator.scene_bounds
        print(f"场景边界: {bounds}")
        print(f"场景中心: {simulator.scene_center}")
        print(f"场景尺寸: {simulator.scene_size}\n")
        
        # 定义测试点
        test_points = [
            simulator.scene_center,  # 场景中心
            bounds[0],  # 最小角
            bounds[1],  # 最大角
            np.array([bounds[0][0], 0, simulator.scene_center[2]]),  # X最小，Z中心
            np.array([bounds[1][0], 0, simulator.scene_center[2]]),  # X最大，Z中心
            np.array([simulator.scene_center[0], 0, bounds[0][2]]),  # X中心，Z最小
            np.array([simulator.scene_center[0], 0, bounds[1][2]]),  # X中心，Z最大
        ]
        
        print("=== 坐标转换精度测试 ===")
        total_error = 0.0
        max_error = 0.0
        acceptable_count = 0
        
        for i, test_point in enumerate(test_points):
            print(f"\n测试点 {i+1}: ({test_point[0]:.3f}, {test_point[1]:.3f}, {test_point[2]:.3f})")
            
            # 验证坐标转换
            result = simulator.verify_coordinate_conversion(test_point)
            
            print(f"  地图坐标: ({result['map_coords'][0]}, {result['map_coords'][1]})")
            print(f"  转换回世界坐标: ({result['converted_world'][0]:.3f}, {result['converted_world'][1]:.3f}, {result['converted_world'][2]:.3f})")
            print(f"  位置误差: {result['position_error']:.6f}m")
            print(f"  精度可接受: {'是' if result['error_acceptable'] else '否'}")
            
            total_error += result['position_error']
            max_error = max(max_error, result['position_error'])
            if result['error_acceptable']:
                acceptable_count += 1
        
        # 统计结果
        avg_error = total_error / len(test_points)
        success_rate = acceptable_count / len(test_points) * 100
        
        print("\n=== 总体结果 ===")
        print(f"测试点数量: {len(test_points)}")
        print(f"平均误差: {avg_error:.6f}m")
        print(f"最大误差: {max_error:.6f}m")
        print(f"精度可接受率: {success_rate:.1f}% ({acceptable_count}/{len(test_points)})")
        
        # 判断修复是否成功
        if avg_error < 0.1 and max_error < 0.2:
            print("\n✅ 坐标对齐修复成功！误差在可接受范围内。")
        elif avg_error < 0.3:
            print("\n⚠️ 坐标对齐有改善，但仍需进一步优化。")
        else:
            print("\n❌ 坐标对齐问题仍然存在，需要进一步调试。")
        
        # 测试特定的padding相关坐标
        print("\n=== Padding偏移测试 ===")
        
        # 计算padding引起的理论偏移
        padding_offset_x = simulator.MAP_PADDING_LEFT
        padding_offset_y = simulator.MAP_PADDING_TOP
        
        # 获取地图尺寸信息
        padded_width, padded_height = simulator.base_map_image.size
        original_width = padded_width - simulator.MAP_PADDING_LEFT - simulator.MAP_PADDING_RIGHT
        original_height = padded_height - simulator.MAP_PADDING_TOP - simulator.MAP_PADDING_BOTTOM
        
        print(f"地图尺寸 - 原始: {original_width}x{original_height}, 带padding: {padded_width}x{padded_height}")
        print(f"Padding - 左: {simulator.MAP_PADDING_LEFT}, 右: {simulator.MAP_PADDING_RIGHT}")
        print(f"Padding - 上: {simulator.MAP_PADDING_TOP}, 下: {simulator.MAP_PADDING_BOTTOM}")
        
        # 计算理论的像素-世界比例
        world_width = bounds[1][0] - bounds[0][0]
        world_height = bounds[1][2] - bounds[0][2]
        pixel_to_world_x = world_width / original_width
        pixel_to_world_z = world_height / original_height
        
        print(f"像素-世界比例 - X: {pixel_to_world_x:.6f}m/px, Z: {pixel_to_world_z:.6f}m/px")
        
        # 计算padding在世界坐标中的偏移
        padding_world_offset_x = padding_offset_x * pixel_to_world_x
        padding_world_offset_z = padding_offset_y * pixel_to_world_z
        
        print(f"Padding世界偏移 - X: {padding_world_offset_x:.6f}m, Z: {padding_world_offset_z:.6f}m")
        
        if abs(padding_world_offset_x - 0.3) < 0.05 or abs(padding_world_offset_z - 0.3) < 0.05:
            print("✅ 检测到接近0.3m的padding偏移，这确认了之前的分析。")
        else:
            print("ℹ️ Padding偏移与0.3m不完全匹配，可能还有其他因素。")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'simulator' in locals():
            simulator.close()

if __name__ == "__main__":
    test_coordinate_accuracy()
