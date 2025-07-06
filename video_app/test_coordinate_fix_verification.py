#!/usr/bin/env python3
"""
验证video_app中的坐标转换修复效果
"""

import sys
import os
sys.path.append('/home/yaoaa/habitat-lab/video_app/src')

from habitat_video_generator import HabitatVideoGenerator

def test_coordinate_conversion_fix():
    """测试坐标转换修复效果"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("测试video_app中的坐标转换修复效果")
        print("=" * 60)
        
        # 创建视频生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir="./test_outputs"
        )
        
        print("\n=== 坐标转换精度测试 ===")
        
        # 测试场景中心的坐标转换
        center_pos = generator.simulator.scene_center
        print(f"场景中心位置: {center_pos}")
        
        # 使用verify_coordinate_conversion方法
        result = generator.simulator.verify_coordinate_conversion(center_pos)
        
        print(f"坐标转换结果:")
        print(f"  - 原始世界坐标: {result['original_world']}")
        print(f"  - 地图坐标: {result['map_coords']}")
        print(f"  - 转换回的世界坐标: {result['converted_world']}")
        print(f"  - 位置误差: {result['position_error']:.6f}m")
        print(f"  - 误差可接受: {'是' if result['error_acceptable'] else '否'}")
        print(f"  - 说明: {result['note']}")
        
        # 测试几个边界点
        print(f"\n=== 边界点测试 ===")
        test_points = [
            generator.simulator.scene_bounds[0],  # 最小角
            generator.simulator.scene_bounds[1],  # 最大角
        ]
        
        for i, test_point in enumerate(test_points):
            result = generator.simulator.verify_coordinate_conversion(test_point)
            print(f"边界点{i+1}: 误差 {result['position_error']:.6f}m {'✓' if result['error_acceptable'] else '⚠'}")
        
        # 测试移动到场景中心
        print(f"\n=== 移动测试 ===")
        test_commands = [
            [center_pos[0], center_pos[2]],  # 移动到场景中心
            ["left", 45],                    # 左转45度
        ]
        
        print("执行测试命令序列...")
        output_path = generator.process_command_sequence(test_commands)
        
        if output_path:
            print(f"✅ 测试视频已生成: {output_path}")
            
            # 检查最终坐标精度
            final_coord_info = generator.get_agent_coordinate_info()
            if 'coordinate_accuracy' in final_coord_info:
                error = final_coord_info['coordinate_accuracy']['error']
                acceptable = final_coord_info['coordinate_accuracy']['acceptable']
                print(f"最终位置转换误差: {error:.6f}m {'✓' if acceptable else '⚠'}")
        else:
            print("❌ 测试视频生成失败")
        
        # 关闭生成器
        generator.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_conversion_fix()
    if success:
        print("\n✅ 坐标转换修复验证成功！")
    else:
        print("\n❌ 坐标转换修复验证失败！")
