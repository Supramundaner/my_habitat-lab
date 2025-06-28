#!/usr/bin/env python3
"""
测试新的坐标系绘制功能
参考add_grid.py实现的增强版坐标系
"""

import sys
import os
import numpy as np
from PIL import Image

# 导入我们的应用
sys.path.append('/home/yaoaa/habitat-lab/interactive_app/src')
sys.path.append('/home/yaoaa/habitat-lab')
from habitat_navigator_app import HabitatSimulator

def test_enhanced_coordinate_system():
    """测试增强的坐标系功能"""
    print("=== 测试增强坐标系绘制 ===\n")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    try:
        # 初始化模拟器
        print("1. 初始化模拟器...")
        simulator = HabitatSimulator(scene_path, resolution=(512, 512))
        print("✓ 模拟器初始化成功")
        
        # 获取增强的地图
        print("\n2. 生成增强坐标系地图...")
        enhanced_map = simulator.base_map_image
        
        if enhanced_map:
            # 保存原始大小的地图
            enhanced_map.save("/home/yaoaa/habitat-lab/interactive_app/images/enhanced_coordinate_map.png")
            print("✓ 增强坐标系地图已保存到: enhanced_coordinate_map.png")
            print(f"✓ 地图尺寸: {enhanced_map.size}")
            
            # 检查图像特征
            map_array = np.array(enhanced_map)
            
            # 检查是否有网格线（深灰色像素）
            gray_pixels = np.sum((map_array[:,:,0] >= 80) & (map_array[:,:,0] <= 120) & 
                               (map_array[:,:,1] >= 80) & (map_array[:,:,1] <= 120) & 
                               (map_array[:,:,2] >= 80) & (map_array[:,:,2] <= 120))
            
            # 检查是否有白色标签
            white_pixels = np.sum((map_array[:,:,0] > 200) & 
                                (map_array[:,:,1] > 200) & 
                                (map_array[:,:,2] > 200))
            
            print(f"✓ 检测到 {gray_pixels} 个网格像素")
            print(f"✓ 检测到 {white_pixels} 个白色像素（标签和边框）")
            
            # 显示场景信息
            bounds = simulator.scene_bounds
            x_range = bounds[1][0] - bounds[0][0]
            z_range = bounds[1][2] - bounds[0][2]
            
            print(f"\n3. 场景坐标信息:")
            print(f"✓ X范围: {bounds[0][0]:.2f} ~ {bounds[1][0]:.2f} ({x_range:.2f}m)")
            print(f"✓ Z范围: {bounds[0][2]:.2f} ~ {bounds[1][2]:.2f} ({z_range:.2f}m)")
            print(f"✓ 场景中心: ({simulator.scene_center[0]:.2f}, {simulator.scene_center[2]:.2f})")
            
            # 测试一些坐标转换
            print(f"\n4. 测试坐标转换:")
            test_points = [
                [bounds[0][0], bounds[0][2]],  # 左下角
                [bounds[1][0], bounds[1][2]],  # 右上角
                [simulator.scene_center[0], simulator.scene_center[2]],  # 中心
                [0, 0] if bounds[0][0] <= 0 <= bounds[1][0] and bounds[0][2] <= 0 <= bounds[1][2] else None  # 原点
            ]
            
            for i, point in enumerate(test_points):
                if point is not None:
                    world_pos = np.array([point[0], 0, point[1]])
                    map_coords = simulator.world_to_map_coords(world_pos)
                    print(f"✓ 世界坐标 ({point[0]:.2f}, {point[1]:.2f}) -> 地图像素 {map_coords}")
        
        # 清理
        simulator.close()
        
        print("\n✓ 增强坐标系测试完成！")
        print("\n新功能特性:")
        print("  - 带边距的清晰布局")
        print("  - 主要和次要网格线区分")
        print("  - 精确的刻度标注")
        print("  - 原点特殊标记")
        print("  - 彩色指北针")
        print("  - 详细的比例尺信息")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_coordinate_system()
    if success:
        print("\n🎉 增强坐标系测试通过！")
        print("📁 查看生成的图像:")
        print("   /home/yaoaa/habitat-lab/interactive_app/images/enhanced_coordinate_map.png")
    else:
        print("\n❌ 测试失败，需要检查代码")
