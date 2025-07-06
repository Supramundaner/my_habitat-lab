#!/usr/bin/env python3
"""
简单的坐标转换修复验证
"""

import sys
import os
import numpy as np

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator


def test_coordinate_fix():
    """验证坐标转换修复"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("=== 坐标转换修复验证 ===")
    
    output_dir = "./outputs/coordinate_test"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 创建视频生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=25,
            output_dir=output_dir
        )
        
        # 测试场景中心的坐标转换
        if hasattr(generator.simulator, 'verify_coordinate_conversion'):
            center_check = generator.simulator.verify_coordinate_conversion(generator.simulator.scene_center)
            print(f"场景中心坐标转换误差: {center_check['position_error']:.6f}m {'✓' if center_check['error_acceptable'] else '⚠'}")
            
            if center_check['error_acceptable']:
                print("✅ 坐标转换修复成功！")
            else:
                print("❌ 坐标转换仍有问题")
        else:
            print("❌ 坐标转换验证方法不可用")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'generator' in locals():
            generator.close()


if __name__ == "__main__":
    test_coordinate_fix()
