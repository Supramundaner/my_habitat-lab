#!/usr/bin/env python3
"""
简单的VFH*算法测试脚本
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vfh_star import VFHStar

def test_vfh_star():
    """测试VFH*算法的基本功能"""
    print("=" * 50)
    print("VFH*算法测试")
    print("=" * 50)
    
    # 1. 初始化VFH*算法
    target = np.array([10.0, 5.0])  # 目标位置
    config = {
        'robot_radius': 0.14,
        'sensor_range': 0.4,
        'search_depth': 3
    }
    
    vfh = VFHStar(target, config)
    print(f"✅ VFH*算法初始化成功")
    print(f"   目标位置: {target}")
    print(f"   机器人半径: {config['robot_radius']}m")
    print(f"   传感器范围: {config['sensor_range']}m")
    
    # 2. 测试基本参数
    print(f"\n📊 算法参数:")
    print(f"   直方图角度分辨率: {np.rad2deg(vfh.histogram_alpha):.1f}°")
    print(f"   直方图bin数量: {vfh.num_histogram_bins}")
    print(f"   最大安全扇区: {vfh.smax}")
    print(f"   对齐容差: {np.rad2deg(vfh.alignment_tolerance):.1f}°")
    
    # 3. 测试无障碍物情况
    print(f"\n🧪 测试场景1: 无障碍物")
    robot_pos = np.array([0.0, 0.0])
    robot_theta = 0.0
    obstacles = []
    
    direction = vfh.get_best_direction(robot_pos, robot_theta, obstacles)
    if direction is not None:
        print(f"   最佳方向: {np.rad2deg(direction):.1f}°")
        action_name, action_value = vfh.get_discrete_action(direction, robot_theta)
        print(f"   离散动作: {action_name} ({action_value}°)")
    else:
        print("   ❌ 无法找到可行方向")
    
    # 4. 测试有障碍物情况
    print(f"\n🧪 测试场景2: 有障碍物")
    obstacles = [
        (2.0, 1.0, 0.2),  # 右前方障碍物
        (1.0, 2.0, 0.2),  # 左前方障碍物
    ]
    
    direction = vfh.get_best_direction(robot_pos, robot_theta, obstacles)
    if direction is not None:
        print(f"   最佳方向: {np.rad2deg(direction):.1f}°")
        action_name, action_value = vfh.get_discrete_action(direction, robot_theta)
        print(f"   离散动作: {action_name} ({action_value}°)")
    else:
        print("   ❌ 无法找到可行方向")
    
    # 5. 测试极坐标直方图
    print(f"\n📈 极坐标直方图测试:")
    histogram = vfh._get_polar_histogram(robot_pos, obstacles)
    occupied_bins = np.sum(histogram)
    free_bins = len(histogram) - occupied_bins
    print(f"   占用bin数量: {occupied_bins}")
    print(f"   空闲bin数量: {free_bins}")
    print(f"   直方图形状: {histogram.shape}")
    
    # 6. 测试候选方向
    print(f"\n🎯 候选方向测试:")
    candidates = vfh._get_candidate_directions(robot_pos, obstacles)
    print(f"   候选方向数量: {len(candidates)}")
    for i, cand in enumerate(candidates):
        print(f"   候选方向 {i+1}: {np.rad2deg(cand):.1f}°")
    
    print(f"\n✅ VFH*算法测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    test_vfh_star() 