#!/usr/bin/env python3
"""
检测主要坐标偏移 - 专门用于检测0.3m以上的系统性坐标不一致
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

# 添加路径
sys.path.insert(0, '../interactive_app/src')
from habitat_navigator_app import HabitatSimulator

def test_major_coordinate_offset():
    """测试是否存在0.3m以上的坐标偏移"""
    
    # 使用公寓场景
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/hm3d/train/00001-apartment_1/apartment_1.basis.glb"
    
    if not os.path.exists(scene_path):
        print(f"❌ 场景文件不存在: {scene_path}")
        return
    
    try:
        print("=== 检测主要坐标偏移 (0.3m+) ===")
        simulator = HabitatSimulator(scene_path, resolution=(1024, 1024))
        
        # 获取场景信息
        print(f"场景边界: {simulator.scene_bounds}")
        print(f"场景中心: {simulator.scene_center}")
        print(f"地图尺寸: {simulator.base_map_image.size}")
        
        # 测试多个具有代表性的点
        test_positions = [
            # 场景中心
            simulator.scene_center,
            
            # 场景四个角落
            np.array([simulator.scene_bounds[0][0], simulator.scene_center[1], simulator.scene_bounds[0][2]]),
            np.array([simulator.scene_bounds[1][0], simulator.scene_center[1], simulator.scene_bounds[0][2]]),
            np.array([simulator.scene_bounds[0][0], simulator.scene_center[1], simulator.scene_bounds[1][2]]),
            np.array([simulator.scene_bounds[1][0], simulator.scene_center[1], simulator.scene_bounds[1][2]]),
            
            # 一些随机点
            np.array([-1.5, simulator.scene_center[1], 2.0]),
            np.array([1.0, simulator.scene_center[1], -1.0]),
            np.array([0.5, simulator.scene_center[1], 1.5]),
        ]
        
        major_offsets_found = []
        
        for i, test_pos in enumerate(test_positions):
            print(f"\n--- 测试点 {i+1} ---")
            print(f"世界坐标: ({test_pos[0]:.3f}, {test_pos[2]:.3f})")
            
            # 检查点是否在navmesh上
            navmesh_pos = simulator.get_position_with_navmesh_height(test_pos[0], test_pos[2])
            if navmesh_pos is None:
                print("  ⚠️ 点不在navmesh上，跳过")
                continue
            
            # 实际移动代理到该位置
            simulator.move_agent_to(navmesh_pos)
            actual_state = simulator.get_agent_state()
            actual_pos = actual_state.position
            
            print(f"实际位置: ({actual_pos[0]:.3f}, {actual_pos[2]:.3f})")
            
            # 计算X/Z方向的实际偏移（排除Y坐标）
            offset_x = actual_pos[0] - test_pos[0]
            offset_z = actual_pos[2] - test_pos[2]
            total_2d_offset = np.sqrt(offset_x**2 + offset_z**2)
            
            print(f"2D偏移: X={offset_x:.4f}m, Z={offset_z:.4f}m, 总计={total_2d_offset:.4f}m")
            
            # 转换到地图坐标
            map_coords = simulator.world_to_map_coords(actual_pos)
            print(f"地图坐标: ({map_coords[0]:.1f}, {map_coords[1]:.1f})")
            
            # 反向转换验证
            reverse_world = simulator.map_coords_to_world(map_coords[0], map_coords[1])
            reverse_offset_x = reverse_world[0] - actual_pos[0]
            reverse_offset_z = reverse_world[2] - actual_pos[2]
            reverse_total = np.sqrt(reverse_offset_x**2 + reverse_offset_z**2)
            
            print(f"反向转换偏移: X={reverse_offset_x:.4f}m, Z={reverse_offset_z:.4f}m, 总计={reverse_total:.4f}m")
            
            # 检测主要偏移（0.3m以上）
            if total_2d_offset > 0.3:
                major_offsets_found.append({
                    'test_point': i+1,
                    'target': test_pos,
                    'actual': actual_pos,
                    'offset_2d': total_2d_offset,
                    'offset_x': offset_x,
                    'offset_z': offset_z
                })
                print(f"  🚨 发现主要偏移: {total_2d_offset:.4f}m > 0.3m")
            elif reverse_total > 0.3:
                print(f"  🚨 发现坐标转换主要偏移: {reverse_total:.4f}m > 0.3m")
            else:
                print(f"  ✅ 偏移在可接受范围内: {max(total_2d_offset, reverse_total):.4f}m < 0.3m")
        
        # 总结结果
        print(f"\n=== 检测结果总结 ===")
        print(f"测试点总数: {len(test_positions)}")
        print(f"发现主要偏移(>0.3m): {len(major_offsets_found)}个")
        
        if major_offsets_found:
            print("\n🚨 发现的主要偏移:")
            for offset in major_offsets_found:
                print(f"  点{offset['test_point']}: {offset['offset_2d']:.4f}m (X:{offset['offset_x']:.4f}, Z:{offset['offset_z']:.4f})")
            
            print("\n这表明存在系统性的坐标偏移问题！")
            return False
        else:
            print("\n✅ 未发现0.3m以上的坐标偏移")
            print("可能的原因:")
            print("  1. 偏移出现在特定的运动场景中")
            print("  2. 偏移与地图渲染/缩放相关")
            print("  3. 偏移与视频生成过程中的坐标处理相关")
            return True
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'simulator' in locals():
            simulator.close()

def test_movement_sequence_offset():
    """测试在运动序列中是否出现坐标偏移"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/hm3d/train/00001-apartment_1/apartment_1.basis.glb"
    
    if not os.path.exists(scene_path):
        print(f"❌ 场景文件不存在: {scene_path}")
        return
    
    try:
        print("\n=== 测试运动序列中的坐标偏移 ===")
        simulator = HabitatSimulator(scene_path, resolution=(1024, 1024))
        
        # 模拟一个典型的移动序列
        movement_sequence = [
            (-1.5, 2.0),   # 起始点
            (-1.0, 2.0),   # 小步移动
            (-0.5, 2.0),   # 继续移动
            (-0.5, 1.5),   # 转向移动
            (0.0, 1.5),    # 继续移动
            (0.5, 1.0),    # 对角移动
        ]
        
        previous_pos = None
        cumulative_offset = 0.0
        
        for i, (target_x, target_z) in enumerate(movement_sequence):
            print(f"\n--- 移动步骤 {i+1}: 目标({target_x:.1f}, {target_z:.1f}) ---")
            
            # 获取navmesh高度
            navmesh_pos = simulator.get_position_with_navmesh_height(target_x, target_z)
            if navmesh_pos is None:
                print(f"  ⚠️ 目标位置不在navmesh上，跳过")
                continue
            
            # 移动代理
            simulator.move_agent_to(navmesh_pos)
            actual_state = simulator.get_agent_state()
            actual_pos = actual_state.position
            
            # 计算偏移
            offset_x = actual_pos[0] - target_x
            offset_z = actual_pos[2] - target_z
            total_offset = np.sqrt(offset_x**2 + offset_z**2)
            
            print(f"  目标: ({target_x:.3f}, {target_z:.3f})")
            print(f"  实际: ({actual_pos[0]:.3f}, {actual_pos[2]:.3f})")
            print(f"  偏移: {total_offset:.4f}m (X:{offset_x:.4f}, Z:{offset_z:.4f})")
            
            # 累积偏移分析
            if previous_pos is not None:
                expected_movement = np.array([target_x - movement_sequence[i-1][0], 
                                            target_z - movement_sequence[i-1][1]])
                actual_movement = np.array([actual_pos[0] - previous_pos[0], 
                                          actual_pos[2] - previous_pos[2]])
                movement_error = np.linalg.norm(actual_movement - expected_movement)
                
                print(f"  运动误差: {movement_error:.4f}m")
                cumulative_offset += movement_error
            
            # 地图坐标验证
            map_coords = simulator.world_to_map_coords(actual_pos)
            reverse_pos = simulator.map_coords_to_world(map_coords[0], map_coords[1])
            map_error = np.sqrt((reverse_pos[0] - actual_pos[0])**2 + 
                               (reverse_pos[2] - actual_pos[2])**2)
            
            print(f"  地图转换误差: {map_error:.4f}m")
            
            if total_offset > 0.3:
                print(f"  🚨 发现主要偏移: {total_offset:.4f}m")
            elif map_error > 0.3:
                print(f"  🚨 发现地图转换主要偏移: {map_error:.4f}m")
            else:
                print(f"  ✅ 偏移正常")
            
            previous_pos = actual_pos
        
        print(f"\n累积运动偏移: {cumulative_offset:.4f}m")
        
        if cumulative_offset > 0.3:
            print(f"🚨 累积偏移超过阈值: {cumulative_offset:.4f}m > 0.3m")
            return False
        else:
            print(f"✅ 累积偏移在可接受范围内: {cumulative_offset:.4f}m")
            return True
    
    except Exception as e:
        print(f"❌ 运动序列测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'simulator' in locals():
            simulator.close()

if __name__ == "__main__":
    print("开始检测主要坐标偏移...")
    
    # 测试1: 静态点的坐标偏移
    static_result = test_major_coordinate_offset()
    
    # 测试2: 运动序列中的坐标偏移
    movement_result = test_movement_sequence_offset()
    
    print(f"\n=== 最终结果 ===")
    print(f"静态点测试: {'通过' if static_result else '发现问题'}")
    print(f"运动序列测试: {'通过' if movement_result else '发现问题'}")
    
    if not static_result or not movement_result:
        print("\n🚨 检测到0.3m以上的坐标偏移！")
        print("建议进一步调查:")
        print("1. 检查navmesh snap的实际影响")
        print("2. 验证坐标转换常量的正确性")
        print("3. 分析地图渲染流程")
        print("4. 检查视频生成过程中的坐标处理")
    else:
        print("\n✅ 未检测到主要坐标偏移")
        print("如果视频中仍观察到偏移，可能的原因:")
        print("1. 地图缩放/渲染相关问题")
        print("2. 视频帧合成过程中的偏移")
        print("3. 特定场景或条件下的边缘情况")
