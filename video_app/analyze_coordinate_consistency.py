#!/usr/bin/env python3
"""
深入分析地图坐标与视频坐标不一致的问题
"""

import sys
import os
import numpy as np

# 添加video_app的src路径
video_app_src = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, video_app_src)

from habitat_video_generator import HabitatVideoGenerator

def analyze_coordinate_consistency():
    """分析地图坐标与视频坐标的一致性"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("分析地图坐标与视频坐标的一致性")
        print("=" * 60)
        
        # 创建视频生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir="./test_outputs"
        )
        
        # 测试精确的坐标移动
        center = generator.simulator.scene_center
        test_target = (center[0] + 1.0, center[2] + 0.5)  # 从中心偏移的目标点
        
        print(f"场景中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        print(f"目标坐标: ({test_target[0]:.6f}, {test_target[1]:.6f})")
        
        # 步骤1: 分析目标位置处理过程
        print(f"\n=== 步骤1: 目标位置处理 ===")
        target_pos = generator.simulator.get_position_with_navmesh_height(test_target[0], test_target[1])
        if target_pos is not None:
            print(f"get_position_with_navmesh_height结果: ({target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f})")
            x_drift = abs(target_pos[0] - test_target[0])
            z_drift = abs(target_pos[2] - test_target[1])
            print(f"坐标漂移: X={x_drift:.6f}m, Z={z_drift:.6f}m")
        else:
            print("目标位置无法获取navmesh高度")
            return False
        
        # 步骤2: 分析地图坐标转换
        print(f"\n=== 步骤2: 地图坐标转换 ===")
        map_coords = generator.simulator.world_to_map_coords(target_pos)
        print(f"世界坐标 -> 地图坐标: ({target_pos[0]:.6f}, {target_pos[2]:.6f}) -> ({map_coords[0]}, {map_coords[1]})")
        
        # 反向转换验证
        converted_back = generator.simulator.map_coords_to_world(map_coords[0], map_coords[1])
        print(f"地图坐标 -> 世界坐标: ({map_coords[0]}, {map_coords[1]}) -> ({converted_back[0]:.6f}, {converted_back[1]:.6f}, {converted_back[2]:.6f})")
        
        conversion_error_x = abs(target_pos[0] - converted_back[0])
        conversion_error_z = abs(target_pos[2] - converted_back[2])
        print(f"坐标转换误差: X={conversion_error_x:.6f}m, Z={conversion_error_z:.6f}m")
        
        # 步骤3: 实际移动并检查代理位置
        print(f"\n=== 步骤3: 实际移动测试 ===")
        
        # 初始化代理到场景中心
        if not generator.agent_initialized:
            success = generator._reset_agent_to_position(center[0], center[2])
            if success:
                generator.agent_initialized = True
        
        # 执行移动
        print(f"移动到目标: ({test_target[0]:.6f}, {test_target[1]:.6f})")
        success = generator._execute_movement(test_target[0], test_target[1])
        
        if success:
            # 获取实际代理位置
            actual_agent_state = generator.simulator.get_agent_state()
            actual_pos = actual_agent_state.position
            print(f"实际代理位置: ({actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f})")
            
            # 计算实际位置的地图坐标
            actual_map_coords = generator.simulator.world_to_map_coords(actual_pos)
            print(f"实际位置的地图坐标: ({actual_map_coords[0]}, {actual_map_coords[1]})")
            
            # 对比分析
            print(f"\n=== 对比分析 ===")
            print(f"期望位置: ({test_target[0]:.6f}, {test_target[1]:.6f})")
            print(f"计算位置: ({target_pos[0]:.6f}, {target_pos[2]:.6f})")
            print(f"实际位置: ({actual_pos[0]:.6f}, {actual_pos[2]:.6f})")
            
            expected_vs_actual_x = abs(test_target[0] - actual_pos[0])
            expected_vs_actual_z = abs(test_target[1] - actual_pos[2])
            print(f"期望vs实际误差: X={expected_vs_actual_x:.6f}m, Z={expected_vs_actual_z:.6f}m")
            
            calculated_vs_actual_x = abs(target_pos[0] - actual_pos[0])
            calculated_vs_actual_z = abs(target_pos[2] - actual_pos[2])
            print(f"计算vs实际误差: X={calculated_vs_actual_x:.6f}m, Z={calculated_vs_actual_z:.6f}m")
            
            map_coord_diff = abs(map_coords[0] - actual_map_coords[0]) + abs(map_coords[1] - actual_map_coords[1])
            print(f"地图坐标差异: {map_coord_diff} pixels")
            
            # 判断问题来源
            print(f"\n=== 问题诊断 ===")
            if expected_vs_actual_x > 0.01 or expected_vs_actual_z > 0.01:
                print("❌ 发现坐标不一致问题！")
                
                if calculated_vs_actual_x > 0.01 or calculated_vs_actual_z > 0.01:
                    print("🔍 问题可能来自移动执行过程中的坐标修改")
                else:
                    print("🔍 问题来自get_position_with_navmesh_height方法")
                    
                if map_coord_diff > 1:
                    print("🔍 地图坐标转换也存在问题")
            else:
                print("✅ 坐标一致性良好")
        else:
            print("移动执行失败")
        
        # 关闭生成器
        generator.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_coordinate_consistency()
