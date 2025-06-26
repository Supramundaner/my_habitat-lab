#!/usr/bin/env python3
"""诊断FPV图像问题"""

import sys
import os
import numpy as np
from PIL import Image

# 添加habitat-lab到路径
sys.path.append('/home/yaoaa/habitat-lab')

from habitat_navigator_app import HabitatNavigatorApp
from PyQt5.QtWidgets import QApplication

# 创建应用程序
app = QApplication([])

try:
    # 使用默认场景创建导航器
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    navigator = HabitatNavigatorApp(scene_path)
    print("成功创建导航应用")
    
    # 获取FPV观察
    fpv_obs = navigator.simulator.get_fpv_observation()
    print(f"FPV观察形状: {fpv_obs.shape}")
    print(f"FPV观察数据类型: {fpv_obs.dtype}")
    print(f"FPV观察数值范围: {fpv_obs.min()} - {fpv_obs.max()}")
    print(f"FPV观察前10个像素: {fpv_obs.flatten()[:10]}")
    
    # 检查是否有NaN或无限值
    if np.any(np.isnan(fpv_obs)):
        print("警告: FPV图像包含NaN值")
    if np.any(np.isinf(fpv_obs)):
        print("警告: FPV图像包含无限值")
    
    # 尝试保存原始FPV图像
    if fpv_obs.dtype != np.uint8:
        # 如果不是uint8，需要转换
        if fpv_obs.max() <= 1.0:
            # 假设是0-1范围的浮点数
            fpv_uint8 = (fpv_obs * 255).astype(np.uint8)
            print("转换: 0-1浮点数 -> uint8")
        else:
            # 可能已经是0-255范围
            fpv_uint8 = np.clip(fpv_obs, 0, 255).astype(np.uint8)
            print("转换: 裁剪到0-255 -> uint8")
    else:
        fpv_uint8 = fpv_obs
        print("数据类型已经是uint8")
    
    # 保存图像
    if len(fpv_uint8.shape) == 3 and fpv_uint8.shape[2] == 3:
        # RGB图像
        pil_image = Image.fromarray(fpv_uint8, 'RGB')
        pil_image.save('/home/yaoaa/habitat-lab/debug_fpv_current.png')
        print("保存当前FPV图像: debug_fpv_current.png")
    elif len(fpv_uint8.shape) == 3 and fpv_uint8.shape[2] == 4:
        # RGBA图像
        pil_image = Image.fromarray(fpv_uint8, 'RGBA')
        pil_image.save('/home/yaoaa/habitat-lab/debug_fpv_current.png')
        print("保存当前FPV图像 (RGBA): debug_fpv_current.png")
    else:
        print(f"不支持的图像格式: {fpv_uint8.shape}")
    
    # 测试移动到不同位置
    print("\n测试移动到不同位置...")
    test_positions = [
        np.array([0.0, 1.5, 0.0], dtype=np.float32),
        np.array([2.0, 1.5, 1.0], dtype=np.float32),
        np.array([-1.0, 1.5, -1.0], dtype=np.float32),
    ]
    
    for i, pos in enumerate(test_positions):
        try:
            navigator.simulator.move_agent_to(pos)
            fpv_obs = navigator.simulator.get_fpv_observation()
            
            print(f"位置 {i+1} ({pos}): FPV形状={fpv_obs.shape}, 范围={fpv_obs.min()}-{fpv_obs.max()}")
            
            # 保存这个位置的FPV图像
            if fpv_obs.dtype != np.uint8:
                if fpv_obs.max() <= 1.0:
                    fpv_uint8 = (fpv_obs * 255).astype(np.uint8)
                else:
                    fpv_uint8 = np.clip(fpv_obs, 0, 255).astype(np.uint8)
            else:
                fpv_uint8 = fpv_obs
            
            if len(fpv_uint8.shape) == 3 and fpv_uint8.shape[2] == 3:
                pil_image = Image.fromarray(fpv_uint8, 'RGB')
                pil_image.save(f'/home/yaoaa/habitat-lab/debug_fpv_pos_{i+1}.png')
                print(f"  保存位置{i+1}的图像: debug_fpv_pos_{i+1}.png")
                
        except Exception as e:
            print(f"位置 {i+1} 测试失败: {e}")
    
    print("\n✓ FPV诊断完成")
    
except Exception as e:
    print(f"诊断失败: {e}")
    import traceback
    traceback.print_exc()
    
print("测试完成")
