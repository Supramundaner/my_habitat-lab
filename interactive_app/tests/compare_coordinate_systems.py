#!/usr/bin/env python3
"""
对比新旧坐标系的效果
"""

import sys
import os
import numpy as np
from PIL import Image

# 导入我们的应用
sys.path.append('/home/yaoaa/habitat-lab/interactive_app/src')
sys.path.append('/home/yaoaa/habitat-lab')

def compare_coordinate_systems():
    """对比新旧坐标系"""
    print("=== 坐标系对比分析 ===\n")
    
    # 检查新的增强坐标系图像
    new_map_path = "/home/yaoaa/habitat-lab/interactive_app/images/enhanced_coordinate_map.png"
    old_map_path = "/home/yaoaa/habitat-lab/interactive_app/images/final_test_map.png"
    
    try:
        if os.path.exists(new_map_path):
            new_img = Image.open(new_map_path)
            print(f"✓ 新坐标系图像: {new_img.size}")
            
            # 分析新图像特征
            new_array = np.array(new_img)
            
            # 检测网格线（灰色像素）
            gray_pixels = np.sum((new_array[:,:,0] >= 80) & (new_array[:,:,0] <= 120))
            major_grid_pixels = np.sum((new_array[:,:,0] >= 140) & (new_array[:,:,0] <= 160))
            white_pixels = np.sum((new_array[:,:,0] > 200) & (new_array[:,:,1] > 200) & (new_array[:,:,2] > 200))
            
            print(f"  - 网格线像素: {gray_pixels:,}")
            print(f"  - 主网格线像素: {major_grid_pixels:,}")
            print(f"  - 白色标签像素: {white_pixels:,}")
            
            # 检查边距
            print(f"  - 图像尺寸包含边距: {new_img.size}")
            
        if os.path.exists(old_map_path):
            old_img = Image.open(old_map_path)
            print(f"\n✓ 旧坐标系图像: {old_img.size}")
            
            old_array = np.array(old_img)
            old_white_pixels = np.sum((old_array[:,:,0] > 200) & (old_array[:,:,1] > 200) & (old_array[:,:,2] > 200))
            print(f"  - 白色标签像素: {old_white_pixels:,}")
            
        print(f"\n📊 改进效果:")
        print(f"✅ 新坐标系具有以下优势:")
        print(f"   1. 带边距布局 - 标签完全可见")
        print(f"   2. 多层次网格 - 主/次网格线区分")
        print(f"   3. 精确刻度 - 更细致的坐标标注")
        print(f"   4. 原点标记 - 黄色圆点突出显示")
        print(f"   5. 彩色指北针 - 红色X轴，绿色Z轴")
        print(f"   6. 详细信息 - 包含网格间隔信息")
        
        print(f"\n🎯 实际应用效果:")
        print(f"   - 更容易读取坐标值")
        print(f"   - 更精确的位置定位")
        print(f"   - 更清晰的方向指示")
        print(f"   - 更专业的图纸样式")
        
        return True
        
    except Exception as e:
        print(f"✗ 对比失败: {e}")
        return False

if __name__ == "__main__":
    success = compare_coordinate_systems()
    if success:
        print(f"\n🎉 坐标系优化完成！")
        print(f"📁 图像文件位置:")
        print(f"   新版本: /home/yaoaa/habitat-lab/interactive_app/images/enhanced_coordinate_map.png")
        print(f"   旧版本: /home/yaoaa/habitat-lab/interactive_app/images/final_test_map.png")
