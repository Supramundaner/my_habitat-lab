#!/usr/bin/env python3
"""
视频生成器坐标修复验证脚本

验证video_generator中的坐标转换修复是否正常工作。
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def main():
    print("=== 视频生成器坐标修复验证 ===\n")
    
    # 测试场景路径
    possible_scene_paths = [
        "../../habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
        "../../../habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
        "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    ]
    
    scene_path = None
    for path in possible_scene_paths:
        if os.path.exists(path):
            scene_path = path
            break
    
    if scene_path is None:
        print("错误: 找不到测试场景文件")
        print("请确保场景文件存在或修改脚本中的路径")
        return
    
    try:
        from habitat_video_generator import HabitatVideoGenerator
        
        print(f"使用场景: {scene_path}")
        print("正在初始化视频生成器...")
        
        # 创建视频生成器实例
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir="./test_outputs"
        )
        
        print("\n=== 坐标信息测试 ===")
        
        # 获取代理坐标信息
        coord_info = generator.get_agent_coordinate_info()
        
        if 'error' in coord_info:
            print(f"获取坐标信息失败: {coord_info['error']}")
        else:
            print("当前代理位置:")
            print(f"  世界坐标: ({coord_info['world_position']['x']:.3f}, {coord_info['world_position']['y']:.3f}, {coord_info['world_position']['z']:.3f})")
            print(f"  地图坐标: ({coord_info['map_coordinates']['x']}, {coord_info['map_coordinates']['y']})")
            print(f"  转换误差: {coord_info['coordinate_accuracy']['error']:.6f}m")
            print(f"  精度可接受: {'是' if coord_info['coordinate_accuracy']['acceptable'] else '否'}")
            
            print("\n场景信息:")
            bounds = coord_info['scene_info']['bounds']
            center = coord_info['scene_info']['center']
            print(f"  边界: 最小({bounds['min'][0]:.2f}, {bounds['min'][1]:.2f}, {bounds['min'][2]:.2f})")
            print(f"        最大({bounds['max'][0]:.2f}, {bounds['max'][1]:.2f}, {bounds['max'][2]:.2f})")
            print(f"  中心: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        print("\n=== 移动测试 ===")
        
        # 测试移动到场景中心
        if 'scene_info' in coord_info:
            center = coord_info['scene_info']['center']
            test_commands = [
                [center[0], center[2]],  # 移动到场景中心
                ["left", 90],            # 左转90度
                ["right", 180],          # 右转180度
            ]
            
            print("执行测试命令序列...")
            output_path = generator.process_command_sequence(test_commands)
            
            if output_path:
                print(f"✅ 测试视频已生成: {output_path}")
                
                # 检查最终位置的坐标精度
                final_coord_info = generator.get_agent_coordinate_info()
                if 'coordinate_accuracy' in final_coord_info:
                    error = final_coord_info['coordinate_accuracy']['error']
                    acceptable = final_coord_info['coordinate_accuracy']['acceptable']
                    print(f"最终位置转换误差: {error:.6f}m {'✓' if acceptable else '⚠'}")
                    
                    if error < 0.1:
                        print("✅ 坐标转换修复验证成功！")
                    else:
                        print("⚠️ 坐标转换仍需优化")
                else:
                    print("⚠️ 无法获取最终位置精度信息")
            else:
                print("❌ 测试视频生成失败")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保video_app的src目录在Python路径中")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'generator' in locals():
            generator.close()
            print("\n视频生成器已关闭")

if __name__ == "__main__":
    main()
