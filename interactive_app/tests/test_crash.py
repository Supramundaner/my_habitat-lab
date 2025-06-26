#!/usr/bin/env python3
"""
简单测试崩溃问题
"""
import numpy as np

def test_crash_issue():
    """测试崩溃问题"""
    try:
        # 导入主应用程序模块
        from habitat_navigator_app import HabitatSimulator
        
        scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
        
        print("创建模拟器...")
        sim = HabitatSimulator(scene_path)
        
        print("测试坐标处理...")
        # 测试坐标 "2.6, 0.1"
        x, z = 2.6, 0.1
        
        print(f"检查坐标 ({x}, {z}) 是否可导航...")
        is_nav = sim.is_navigable(x, z)
        print(f"可导航: {is_nav}")
        
        if is_nav:
            print("获取对齐的3D点...")
            target_pos = sim.snap_to_navigable(x, z)
            print(f"对齐位置: {target_pos}")
            
            if target_pos is not None:
                print("移动智能体...")
                sim.move_agent_to(target_pos)
                print("✓ 移动成功")
        
        print("测试FPV图像获取...")
        fpv_img = sim.get_fpv_observation()
        print(f"✓ FPV图像尺寸: {fpv_img.shape}")
        
        # 清理
        sim.close()
        print("✓ 测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_crash_issue()
