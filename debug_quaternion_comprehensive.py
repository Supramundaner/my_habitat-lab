#!/usr/bin/env python3
"""尝试重现四元数构造错误"""

import numpy as np
import magnum as mn
import habitat_sim

# 测试不同的四元数值
test_values = [
    [0, 0, 0, 1],       # 默认朝向
    [1, 0, 0, 0],       # 可能导致问题的值
    [0, 1, 0, 0],       # 可能导致问题的值
    [0, 0, 1, 0],       # 可能导致问题的值
    [0.1, 0.2, 0.3, 0.9], # 正常值
]

print("测试不同的四元数值...")

for i, values in enumerate(test_values):
    print(f"\n测试 {i+1}: {values}")
    
    # 转换为numpy数组
    rotation_array = np.array(values, dtype=np.float32)
    
    try:
        # 方法1: 使用Vector3 + scalar
        quat1 = mn.Quaternion(
            mn.Vector3(rotation_array[0], rotation_array[1], rotation_array[2]),
            rotation_array[3]
        )
        print(f"  方法1成功: {quat1}")
        
        # 测试transform_vector
        forward_vec = quat1.transform_vector(mn.Vector3(0, 0, -1))
        print(f"  transform_vector成功: {forward_vec}")
        
    except Exception as e:
        print(f"  方法1失败: {e}")
    
    try:
        # 方法2: 使用嵌套元组
        quat2 = mn.Quaternion(((float(rotation_array[0]), float(rotation_array[1]), float(rotation_array[2])), float(rotation_array[3])))
        print(f"  方法2成功: {quat2}")
        
    except Exception as e:
        print(f"  方法2失败: {e}")

# 测试 AgentState 中的四元数
print("\n测试 AgentState 中的四元数...")
try:
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0, 0, 0], dtype=np.float32)
    agent_state.rotation = np.array([0, 0, 0, 1], dtype=np.float32)
    print("AgentState设置成功")
    
    # 获取旋转并构造四元数
    rotation = agent_state.rotation
    print(f"获取的旋转: {rotation}, 类型: {type(rotation)}")
    
    quat = mn.Quaternion(
        mn.Vector3(rotation[0], rotation[1], rotation[2]),
        rotation[3]
    )
    print(f"从AgentState构造四元数成功: {quat}")
    
except Exception as e:
    print(f"AgentState测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")
