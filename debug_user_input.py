#!/usr/bin/env python3
"""模拟实际的应用运行情况"""

import sys
import os
import numpy as np
import magnum as mn
import habitat_sim

# 创建一个最小的测试环境
def test_coordinate_processing():
    """测试坐标处理逻辑"""
    
    # 模拟用户输入
    user_input = "2.6, 0.1"
    
    # 解析坐标
    try:
        coords = [float(x.strip()) for x in user_input.split(',')]
        if len(coords) != 2:
            raise ValueError("需要两个坐标值")
        
        x, z = coords
        print(f"解析坐标: x={x}, z={z}")
        
        # 创建目标位置 (x, y, z) - y通常是高度，设为1.5
        target_pos = np.array([x, 1.5, z], dtype=np.float32)
        print(f"目标位置: {target_pos}")
        
        # 测试可能的四元数创建
        # 模拟默认旋转
        default_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
        print(f"默认旋转: {default_rotation}")
        
        # 测试可能的错误构造
        try:
            # 这可能是错误发生的地方
            quat = mn.Quaternion(
                mn.Vector3(default_rotation[0], default_rotation[1], default_rotation[2]),
                default_rotation[3]
            )
            print(f"四元数构造成功: {quat}")
        except Exception as e:
            print(f"四元数构造失败: {e}")
        
        # 测试 AgentState
        try:
            agent_state = habitat_sim.AgentState()
            agent_state.position = target_pos
            agent_state.rotation = default_rotation
            print("AgentState创建成功")
        except Exception as e:
            print(f"AgentState创建失败: {e}")
        
        # 测试一个可能的错误情况：用户输入导致的问题
        # 如果某个地方误用了用户输入的坐标作为四元数
        try:
            # 这可能是错误的来源
            wrong_quat = mn.Quaternion(1, 0, 0, 0)  # 这会失败
        except Exception as e:
            print(f"错误的四元数构造（预期失败）: {e}")
            
    except Exception as e:
        print(f"坐标处理失败: {e}")

if __name__ == "__main__":
    test_coordinate_processing()
