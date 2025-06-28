#!/usr/bin/env python3
"""
解释snap_to_navigable机制的演示脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator
import numpy as np

def explain_snap_to_navigable():
    """解释snap_to_navigable机制"""
    print("=== Habitat-sim的snap_to_navigable机制解释 ===\n")
    
    # 初始化生成器
    generator = HabitatVideoGenerator(
        scene_filepath="/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
        gpu_device_id=0
    )
    
    print("场景边界:")
    bounds = generator.simulator.scene_bounds
    print(f"  X: {bounds[0][0]:.3f} ~ {bounds[1][0]:.3f}")
    print(f"  Z: {bounds[0][2]:.3f} ~ {bounds[1][2]:.3f}")
    print()
    
    # 测试不同类型的点
    test_points = [
        (2.5, 0.5, "室内地板点"),
        (0.0, 0.0, "场景中心附近"),
        (1.0, -1.0, "可能的墙壁或障碍物位置"),
        (5.0, 1.0, "接近边界的点"),
        (10.0, 10.0, "场景外的点"),
        (-5.0, -5.0, "负坐标区域"),
    ]
    
    print("测试不同点的snap_to_navigable结果:")
    print("输入坐标 -> 对齐后坐标 | 是否可导航 | 距离差")
    print("-" * 70)
    
    for x, z, description in test_points:
        # 检查原始点是否可导航
        is_navigable = generator.simulator.is_navigable(x, z)
        
        # 获取对齐后的点
        snapped_point = generator.simulator.snap_to_navigable(x, z)
        
        if snapped_point is not None:
            # 计算距离差
            original = np.array([x, 0, z])
            snapped = np.array([snapped_point[0], 0, snapped_point[2]])
            distance = np.linalg.norm(snapped - original)
            
            print(f"({x:5.1f}, {z:5.1f}) -> ({snapped_point[0]:5.3f}, {snapped_point[2]:5.3f}) | "
                  f"{'可导航' if is_navigable else '不可导航':>6} | {distance:6.3f}m | {description}")
        else:
            print(f"({x:5.1f}, {z:5.1f}) -> None (无法对齐)           | "
                  f"{'可导航' if is_navigable else '不可导航':>6} |   N/A    | {description}")
    
    print("\n=== 机制解释 ===")
    print("1. snap_to_navigable的作用:")
    print("   - 将任意2D坐标(x,z)对齐到最近的可导航3D点")
    print("   - 自动计算正确的Y坐标(高度)")
    print("   - 确保代理不会\"悬浮\"或\"沉入地下\"")
    print()
    
    print("2. 为什么需要这个机制:")
    print("   - 用户输入的是2D坐标(x,z)，但Habitat需要3D坐标(x,y,z)")
    print("   - 地面高度可能不是0，可能有台阶、斜坡等")
    print("   - 确保代理站在\"地面\"上，而不是空中或地下")
    print()
    
    print("3. 什么情况下会产生位置偏移:")
    print("   - 输入点在墙壁内部 -> 对齐到最近的地面")
    print("   - 输入点在家具上方 -> 对齐到可行走的地面") 
    print("   - 输入点在场景边界外 -> 对齐到边界内的点")
    print("   - 输入点高度不对 -> 调整到正确的地面高度")
    print()
    
    print("4. 这不是碰撞检测:")
    print("   - 碰撞检测是移动过程中检查路径是否被阻挡")
    print("   - snap_to_navigable是坐标预处理，确保起始/目标点有效")
    print("   - 即使原始点\"可导航\"，也可能被微调以获得更精确的位置")
    
    # 演示一个具体例子
    print("\n=== 具体例子 ===")
    test_x, test_z = 2.5, 0.5
    is_nav_before = generator.simulator.is_navigable(test_x, test_z)
    snapped = generator.simulator.snap_to_navigable(test_x, test_z)
    
    print(f"输入: ({test_x}, {test_z})")
    print(f"原始点可导航: {is_nav_before}")
    if snapped is not None:
        print(f"对齐后: ({snapped[0]:.6f}, {snapped[1]:.6f}, {snapped[2]:.6f})")
        print(f"高度调整: {snapped[1]:.6f}m (这就是为什么需要snap)")
        distance = np.linalg.norm(np.array([test_x, 0, test_z]) - np.array([snapped[0], 0, snapped[2]]))
        print(f"水平偏移: {distance:.6f}m")
    
    generator.close()

if __name__ == "__main__":
    explain_snap_to_navigable()
