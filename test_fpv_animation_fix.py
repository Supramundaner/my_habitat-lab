#!/usr/bin/env python3
"""测试FPV和动画修复"""

import sys
import os
import time
import numpy as np

# 添加habitat-lab到路径
sys.path.append('/home/yaoaa/habitat-lab')

from habitat_navigator_app import HabitatNavigatorApp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 创建应用程序
app = QApplication([])

try:
    # 使用默认场景创建导航器
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    navigator = HabitatNavigatorApp(scene_path)
    print("成功创建导航应用")
    
    # 测试FPV显示修复
    print("\n=== 测试FPV显示修复 ===")
    
    # 移动到一个位置
    test_pos = np.array([1.0, 1.5, 1.0], dtype=np.float32)
    navigator.simulator.move_agent_to(test_pos)
    
    # 更新FPV显示
    navigator.update_fpv_display()
    print("✓ FPV显示更新完成，检查是否还有乱码")
    
    # 获取并检查FPV图像
    fpv_obs = navigator.simulator.get_fpv_observation()
    print(f"  FPV图像形状: {fpv_obs.shape}")
    print(f"  FPV图像类型: {fpv_obs.dtype}")
    print(f"  FPV图像范围: {fpv_obs.min()} - {fpv_obs.max()}")
    
    # 保存一个测试FPV图像
    from PIL import Image
    if len(fpv_obs.shape) == 3 and fpv_obs.shape[2] == 4:
        # RGBA转RGB
        fpv_rgb = fpv_obs[:, :, :3]
        pil_image = Image.fromarray(fpv_rgb, 'RGB')
        pil_image.save('/home/yaoaa/habitat-lab/test_fpv_fixed.png')
        print("  保存测试FPV图像: test_fpv_fixed.png")
    
    # 测试动画修复
    print("\n=== 测试平滑动画修复 ===")
    
    # 创建一个测试导航
    start_pos = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    end_pos = np.array([3.0, 1.5, 2.0], dtype=np.float32)
    
    # 移动到起始位置
    navigator.simulator.move_agent_to(start_pos)
    navigator.update_displays()
    print(f"起始位置: {start_pos}")
    
    # 开始路径动画
    print("开始路径动画...")
    navigator.start_path_animation(start_pos, end_pos)
    
    # 检查动画状态
    print(f"  路径点数量: {len(navigator.path_waypoints) if navigator.path_waypoints else 0}")
    print(f"  插值步数设置: {navigator.interpolation_steps}")
    print(f"  动画定时器间隔: {navigator.animation_timer.interval()}ms")
    print(f"  是否正在移动: {navigator.is_moving}")
    
    # 等待一小段时间让动画开始
    app.processEvents()
    time.sleep(0.1)
    app.processEvents()
    
    print("  动画已开始，检查是否平滑...")
    
    # 手动触发几次动画更新来测试
    if navigator.is_moving:
        print("  手动测试动画步骤:")
        for i in range(5):
            old_step = navigator.current_interpolation_step
            old_waypoint = navigator.current_waypoint_index
            
            navigator.animate_movement()
            app.processEvents()
            
            new_step = navigator.current_interpolation_step
            new_waypoint = navigator.current_waypoint_index
            
            current_state = navigator.simulator.get_agent_state()
            print(f"    步骤 {i+1}: 插值步数 {old_step}->{new_step}, 路径点 {old_waypoint}->{new_waypoint}")
            print(f"             当前位置: [{current_state.position[0]:.2f}, {current_state.position[1]:.2f}, {current_state.position[2]:.2f}]")
            
            if not navigator.is_moving:
                print("    动画完成")
                break
                
            time.sleep(0.05)  # 小延迟
    else:
        print("  动画未开始或已完成")
    
    print("\n✓ 测试完成")
    
    # 显示主窗口进行最终验证
    print("\n显示主窗口进行视觉验证...")
    navigator.show()
    
    # 设置定时器关闭窗口
    close_timer = QTimer()
    close_timer.timeout.connect(app.quit)
    close_timer.start(3000)  # 3秒后关闭
    
    app.exec_()
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
