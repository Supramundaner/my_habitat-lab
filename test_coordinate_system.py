#!/usr/bin/env python3
"""
测试改进后的坐标系绘制
"""

import os
import sys
import numpy as np
from PIL import Image

def test_coordinate_system():
    """测试坐标系绘制"""
    print("测试改进后的坐标系绘制...")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    
    if not os.path.exists(scene_path):
        print(f"场景文件不存在: {scene_path}")
        return False
    
    try:
        # 导入主应用程序的模拟器类
        from habitat_navigator_app import HabitatSimulator
        
        # 创建模拟器实例
        print("创建模拟器...")
        sim = HabitatSimulator(scene_path, resolution=(512, 512))
        
        print(f"✓ 场景边界: {sim.scene_bounds}")
        print(f"✓ 场景中心: {sim.scene_center}")
        
        # 保存改进后的地图
        if sim.base_map_image:
            sim.base_map_image.save("improved_topdown_map.png")
            print("✓ 改进的俯视地图已保存: improved_topdown_map.png")
            
            # 测试坐标转换
            test_points = [
                np.array([0, 0, 0]),  # 世界原点
                sim.scene_center,     # 场景中心
                sim.scene_bounds[0],  # 场景最小点
                sim.scene_bounds[1],  # 场景最大点
            ]
            
            print("\n坐标转换测试:")
            for i, point in enumerate(test_points):
                map_coords = sim.world_to_map_coords(point)
                print(f"  点{i+1}: 世界坐标{point} -> 地图坐标{map_coords}")
        
        # 清理
        sim.close()
        print("✓ 模拟器已关闭")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("改进后的坐标系绘制测试")
    print("=" * 40)
    
    if test_coordinate_system():
        print("\n✓ 坐标系改进测试通过！")
        print("请查看生成的 improved_topdown_map.png 文件")
    else:
        print("\n✗ 坐标系测试失败")
    
    return True

if __name__ == '__main__':
    main()
