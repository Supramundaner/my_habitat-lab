#!/usr/bin/env python3
"""
测试navmesh snap对坐标精度的影响
"""

import sys
import os
import numpy as np

# 添加interactive_app的src路径
interactive_app_src = os.path.join(os.path.dirname(__file__), '../interactive_app/src')
sys.path.insert(0, interactive_app_src)

from habitat_navigator_app import HabitatSimulator
import magnum as mn

def test_navmesh_snap_impact():
    """测试navmesh snap对坐标精度的影响"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("测试navmesh snap对坐标精度的影响")
        print("=" * 60)
        
        # 创建模拟器
        simulator = HabitatSimulator(scene_path)
        
        # 测试点：场景中心附近的一些精确坐标
        center = simulator.scene_center
        test_coordinates = [
            (center[0], center[2]),  # 场景中心
            (center[0] + 0.1, center[2] + 0.1),  # 微小偏移
            (center[0] - 0.05, center[2] + 0.15),  # 另一个微小偏移
            (center[0] + 0.2, center[2] - 0.1),  # 稍大偏移
        ]
        
        for i, (target_x, target_z) in enumerate(test_coordinates):
            print(f"\n测试点 {i+1}: 目标坐标 ({target_x:.6f}, {target_z:.6f})")
            
            # 方法1：使用当前的get_position_with_navmesh_height方法
            result_current = simulator.get_position_with_navmesh_height(target_x, target_z)
            if result_current is not None:
                x_error_current = abs(result_current[0] - target_x)
                z_error_current = abs(result_current[2] - target_z)
                print(f"  当前方法结果: ({result_current[0]:.6f}, {result_current[1]:.6f}, {result_current[2]:.6f})")
                print(f"  X误差: {x_error_current:.6f}m, Z误差: {z_error_current:.6f}m")
            else:
                print(f"  当前方法: 返回None（不可导航）")
            
            # 方法2：直接测试snap_point的影响
            test_point = mn.Vector3(target_x, 0.0, target_z)
            snapped_point = simulator.sim.pathfinder.snap_point(test_point)
            
            x_error_snap = abs(snapped_point.x - target_x)
            z_error_snap = abs(snapped_point.z - target_z)
            print(f"  Snap_point结果: ({snapped_point.x:.6f}, {snapped_point.y:.6f}, {snapped_point.z:.6f})")
            print(f"  Snap X误差: {x_error_snap:.6f}m, Z误差: {z_error_snap:.6f}m")
            
            # 方法3：纯Y坐标查询（理想方法）
            # 检查是否可以直接获取高度而不snap X,Z坐标
            try:
                # 尝试在精确坐标处获取地面高度
                height_at_point = simulator.sim.pathfinder.get_random_navigable_point_near(
                    mn.Vector3(target_x, 0.0, target_z), 0.01
                )
                if height_at_point is not None:
                    print(f"  Near点查询: ({height_at_point.x:.6f}, {height_at_point.y:.6f}, {height_at_point.z:.6f})")
                    x_error_near = abs(height_at_point.x - target_x)
                    z_error_near = abs(height_at_point.z - target_z)
                    print(f"  Near X误差: {x_error_near:.6f}m, Z误差: {z_error_near:.6f}m")
                else:
                    print(f"  Near点查询: 返回None")
            except Exception as e:
                print(f"  Near点查询失败: {e}")
        
        # 分析结论
        print(f"\n=== 分析结论 ===")
        print("如果看到X或Z坐标的误差 > 0.001m，说明navmesh snap确实在改变坐标")
        print("这会导致地图上显示的位置与实际移动到的位置不一致")
        
        # 关闭模拟器
        simulator.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_navmesh_snap_impact()
