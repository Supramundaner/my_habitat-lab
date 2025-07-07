#!/usr/bin/env python3
"""
调试碰撞检测问题
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_collision_detection():
    """调试碰撞检测"""
    print("调试碰撞检测逻辑...")
    
    # 模拟测试场景的初始位置
    agent_walker_pos = np.array([4.0, -1.6, 3.0])  # 从日志看到的初始位置
    agent_spinner_pos = np.array([2.0, -1.6, 1.8])
    
    print(f"初始位置:")
    print(f"  Walker: {agent_walker_pos}")
    print(f"  Spinner: {agent_spinner_pos}")
    
    initial_distance = np.linalg.norm(agent_walker_pos - agent_spinner_pos)
    print(f"  初始距离: {initial_distance:.2f}m")
    
    # 模拟第二步的动作
    walker_target = np.array([2.0, -1.6, 2.0])  # move_to [2.0, 2.0]
    spinner_target = np.array([2.0, -1.6, 1.8])  # 保持在原位（turn_right不移动）
    
    print(f"\n第二步目标:")
    print(f"  Walker target: {walker_target}")
    print(f"  Spinner target: {spinner_target}")
    
    # 模拟路径采样（15步）
    num_steps = 15
    print(f"\n路径采样 ({num_steps} 步):")
    
    for i in range(1, num_steps + 1):
        t = i / num_steps
        
        # Walker的路径点
        walker_point = agent_walker_pos + t * (walker_target - agent_walker_pos)
        
        # Spinner的路径点（旋转动作，位置不变）
        spinner_point = agent_spinner_pos
        
        distance = np.linalg.norm(walker_point - spinner_point)
        
        print(f"  步骤 {i:2d}: Walker={walker_point}, Spinner={spinner_point}, 距离={distance:.2f}m")
        
        # 检查是否小于阈值
        min_distance = 0.8  # 从配置中
        if distance < min_distance:
            print(f"    ⚠ 碰撞警告！距离 {distance:.2f}m < 阈值 {min_distance}m")
            break
    
    print(f"\n配置参数:")
    print(f"  agent_radius: 0.3m")
    print(f"  min_agent_distance: 0.8m")
    print(f"  collision_check_steps: 15")

def debug_with_safer_positions():
    """用更安全的位置测试"""
    print("\n" + "="*50)
    print("测试更安全的初始位置...")
    
    # 更安全的初始位置
    walker_pos = np.array([4.0, -1.6, 4.0])
    spinner_pos = np.array([1.0, -1.6, 1.0])
    
    print(f"安全初始位置:")
    print(f"  Walker: {walker_pos}")
    print(f"  Spinner: {spinner_pos}")
    
    initial_distance = np.linalg.norm(walker_pos - spinner_pos)
    print(f"  初始距离: {initial_distance:.2f}m")
    
    # 第一个动作：Walker移动到[3.0, 3.0]，Spinner移动到[2.0, 1.8]
    walker_target1 = np.array([3.0, -1.6, 3.0])
    spinner_target1 = np.array([2.0, -1.6, 1.8])
    
    # 检查这个动作是否安全
    num_steps = 15
    collision_detected = False
    
    for i in range(1, num_steps + 1):
        t = i / num_steps
        
        walker_point = walker_pos + t * (walker_target1 - walker_pos)
        spinner_point = spinner_pos + t * (spinner_target1 - spinner_pos)
        
        distance = np.linalg.norm(walker_point - spinner_point)
        
        if i <= 5 or distance < 1.0:  # 只显示前几步或接近的步骤
            print(f"  步骤 {i:2d}: 距离={distance:.2f}m")
        
        if distance < 0.8:
            print(f"    ⚠ 第一个动作就会碰撞！步骤 {i}, 距离={distance:.2f}m")
            collision_detected = True
            break
    
    if not collision_detected:
        print("  ✓ 第一个动作安全")
        
        # 检查第二个动作
        walker_target2 = np.array([2.0, -1.6, 2.0])  # [2.0, 2.0]
        
        print(f"\n第二个动作:")
        print(f"  Walker: {walker_target1} → {walker_target2}")
        print(f"  Spinner: 原地旋转")
        
        for i in range(1, num_steps + 1):
            t = i / num_steps
            
            walker_point = walker_target1 + t * (walker_target2 - walker_target1)
            spinner_point = spinner_target1  # 旋转不移动
            
            distance = np.linalg.norm(walker_point - spinner_point)
            
            if i <= 5 or distance < 1.0:
                print(f"    步骤 {i:2d}: 距离={distance:.2f}m")
            
            if distance < 0.8:
                print(f"      ⚠ 第二个动作碰撞！步骤 {i}, 距离={distance:.2f}m")
                break

if __name__ == "__main__":
    debug_collision_detection()
    debug_with_safer_positions()
