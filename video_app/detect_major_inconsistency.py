#!/usr/bin/env python3
"""
严格检查0.3m以上坐标不一致问题
"""

import sys
import os
import numpy as np

# 添加video_app的src路径
video_app_src = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, video_app_src)

from habitat_video_generator import HabitatVideoGenerator

def detect_major_coordinate_inconsistency():
    """检测主要的坐标不一致问题（>0.3m）"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("检测主要坐标不一致问题（>0.3m）")
        print("=" * 60)
        
        # 创建视频生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir="./test_outputs"
        )
        
        # 测试多个精确坐标点
        center = generator.simulator.scene_center
        
        test_coordinates = [
            (center[0], center[2]),  # 场景中心
            (center[0] + 1.0, center[2]),  # X轴偏移1米
            (center[0], center[2] + 1.0),  # Z轴偏移1米
            (center[0] + 2.0, center[2] + 1.5),  # 对角偏移
            (center[0] - 1.5, center[2] - 0.5),  # 负方向偏移
        ]
        
        print(f"场景边界: {generator.simulator.scene_bounds}")
        print(f"场景中心: {center}")
        
        major_inconsistencies = []
        
        for i, (target_x, target_z) in enumerate(test_coordinates):
            print(f"\n=== 测试点 {i+1}: 目标({target_x:.6f}, {target_z:.6f}) ===")
            
            # 步骤1: 处理目标坐标
            target_pos = generator.simulator.get_position_with_navmesh_height(target_x, target_z)
            if target_pos is None:
                print(f"  跳过：目标位置不在navmesh上")
                continue
            
            print(f"  处理后目标位置: ({target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f})")
            
            # 检查第一层不一致：get_position_with_navmesh_height的影响
            navmesh_drift_x = abs(target_pos[0] - target_x)
            navmesh_drift_z = abs(target_pos[2] - target_z)
            total_navmesh_drift = np.sqrt(navmesh_drift_x**2 + navmesh_drift_z**2)
            
            print(f"  Navmesh处理偏移: X={navmesh_drift_x:.6f}m, Z={navmesh_drift_z:.6f}m, 总计={total_navmesh_drift:.6f}m")
            
            if total_navmesh_drift > 0.3:
                print(f"  ❌ 发现重大navmesh偏移: {total_navmesh_drift:.6f}m")
                major_inconsistencies.append(f"Navmesh偏移 {total_navmesh_drift:.6f}m")
            
            # 步骤2: 坐标转换测试
            map_coords = generator.simulator.world_to_map_coords(target_pos)
            converted_back = generator.simulator.map_coords_to_world(map_coords[0], map_coords[1])
            
            conversion_error_x = abs(target_pos[0] - converted_back[0])
            conversion_error_z = abs(target_pos[2] - converted_back[2])
            total_conversion_error = np.sqrt(conversion_error_x**2 + conversion_error_z**2)
            
            print(f"  坐标转换误差: X={conversion_error_x:.6f}m, Z={conversion_error_z:.6f}m, 总计={total_conversion_error:.6f}m")
            
            if total_conversion_error > 0.3:
                print(f"  ❌ 发现重大坐标转换误差: {total_conversion_error:.6f}m")
                major_inconsistencies.append(f"坐标转换误差 {total_conversion_error:.6f}m")
            
            # 步骤3: 实际移动测试
            print(f"  执行实际移动测试...")
            
            # 初始化代理（如果需要）
            if not generator.agent_initialized:
                init_success = generator._reset_agent_to_position(center[0], center[2])
                if init_success:
                    generator.agent_initialized = True
                else:
                    print(f"  无法初始化代理")
                    continue
            
            # 获取移动前位置
            pre_move_state = generator.simulator.get_agent_state()
            pre_move_pos = pre_move_state.position
            
            # 执行移动
            move_success = generator._execute_movement(target_x, target_z)
            
            if move_success:
                # 获取移动后位置
                post_move_state = generator.simulator.get_agent_state()
                actual_pos = post_move_state.position
                
                print(f"  实际到达位置: ({actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f})")
                
                # 计算期望vs实际的误差
                expected_vs_actual_x = abs(target_x - actual_pos[0])
                expected_vs_actual_z = abs(target_z - actual_pos[2])
                total_expected_error = np.sqrt(expected_vs_actual_x**2 + expected_vs_actual_z**2)
                
                print(f"  期望vs实际误差: X={expected_vs_actual_x:.6f}m, Z={expected_vs_actual_z:.6f}m, 总计={total_expected_error:.6f}m")
                
                if total_expected_error > 0.3:
                    print(f"  ❌ 发现重大实际移动误差: {total_expected_error:.6f}m")
                    major_inconsistencies.append(f"实际移动误差 {total_expected_error:.6f}m")
                
                # 计算处理后vs实际的误差
                processed_vs_actual_x = abs(target_pos[0] - actual_pos[0])
                processed_vs_actual_z = abs(target_pos[2] - actual_pos[2])
                total_processed_error = np.sqrt(processed_vs_actual_x**2 + processed_vs_actual_z**2)
                
                print(f"  处理后vs实际误差: X={processed_vs_actual_x:.6f}m, Z={processed_vs_actual_z:.6f}m, 总计={total_processed_error:.6f}m")
                
                if total_processed_error > 0.3:
                    print(f"  ❌ 发现重大处理后误差: {total_processed_error:.6f}m")
                    major_inconsistencies.append(f"处理后误差 {total_processed_error:.6f}m")
                
                # 步骤4: 地图坐标一致性检查
                actual_map_coords = generator.simulator.world_to_map_coords(actual_pos)
                expected_map_coords = generator.simulator.world_to_map_coords(np.array([target_x, target_pos[1], target_z]))
                
                map_pixel_diff_x = abs(actual_map_coords[0] - expected_map_coords[0])
                map_pixel_diff_y = abs(actual_map_coords[1] - expected_map_coords[1])
                total_map_pixel_diff = np.sqrt(map_pixel_diff_x**2 + map_pixel_diff_y**2)
                
                print(f"  地图坐标差异: 像素({map_pixel_diff_x:.1f}, {map_pixel_diff_y:.1f}), 总计={total_map_pixel_diff:.1f}px")
                
                # 将像素差异转换为世界坐标差异的估计
                bounds = generator.simulator.scene_bounds
                world_width = bounds[1][0] - bounds[0][0]
                world_height = bounds[1][2] - bounds[0][2]
                map_width, map_height = generator.simulator.base_map_image.size
                
                # 减去padding
                original_width = map_width - generator.simulator.MAP_PADDING_LEFT - generator.simulator.MAP_PADDING_RIGHT
                original_height = map_height - generator.simulator.MAP_PADDING_TOP - generator.simulator.MAP_PADDING_BOTTOM
                
                pixel_to_world_x = world_width / original_width
                pixel_to_world_z = world_height / original_height
                
                estimated_world_diff = total_map_pixel_diff * max(pixel_to_world_x, pixel_to_world_z)
                print(f"  估计世界坐标差异: {estimated_world_diff:.6f}m")
                
                if estimated_world_diff > 0.3:
                    print(f"  ❌ 发现重大地图坐标差异: {estimated_world_diff:.6f}m")
                    major_inconsistencies.append(f"地图坐标差异 {estimated_world_diff:.6f}m")
                
            else:
                print(f"  移动执行失败")
        
        # 总结
        print(f"\n" + "=" * 60)
        print("检测结果总结")
        print("=" * 60)
        
        if len(major_inconsistencies) > 0:
            print(f"❌ 发现 {len(major_inconsistencies)} 个重大坐标不一致问题（>0.3m）:")
            for inconsistency in major_inconsistencies:
                print(f"  - {inconsistency}")
            
            print(f"\n🔍 问题分析：")
            if any("Navmesh偏移" in inc for inc in major_inconsistencies):
                print("  - get_position_with_navmesh_height方法存在问题")
            if any("坐标转换误差" in inc for inc in major_inconsistencies):
                print("  - world_to_map_coords/map_coords_to_world方法存在问题")
            if any("实际移动误差" in inc for inc in major_inconsistencies):
                print("  - 移动执行过程中存在坐标修改")
            if any("地图坐标差异" in inc for inc in major_inconsistencies):
                print("  - 地图绘制坐标系与实际坐标系不匹配")
        else:
            print("✅ 未发现重大坐标不一致问题（>0.3m）")
            print("所有测试的坐标误差都在可接受范围内")
        
        # 关闭生成器
        generator.close()
        
        return len(major_inconsistencies) == 0
        
    except Exception as e:
        print(f"❌ 检测失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = detect_major_coordinate_inconsistency()
    if success:
        print("\n✅ 坐标一致性检查通过！")
    else:
        print("\n❌ 发现重大坐标不一致问题！需要进一步修复。")
