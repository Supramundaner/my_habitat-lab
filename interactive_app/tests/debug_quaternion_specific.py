#!/usr/bin/env python3
"""测试特定的四元数错误情况"""

import numpy as np
import magnum as mn

# 测试可能导致错误的参数组合
test_cases = [
    # 直接传入四个数值参数 - 这应该会失败
    lambda: mn.Quaternion(1, 0, 0, 0),
    lambda: mn.Quaternion(0, 1, 0, 0),
    lambda: mn.Quaternion(0, 0, 1, 0),
    lambda: mn.Quaternion(0, 0, 0, 1),
    
    # 传入numpy数组 - 这应该会失败
    lambda: mn.Quaternion(np.array([1, 0, 0, 0])),
    lambda: mn.Quaternion(np.array([0, 0, 0, 1])),
    
    # 传入列表 - 这应该会失败
    lambda: mn.Quaternion([1, 0, 0, 0]),
    lambda: mn.Quaternion([0, 0, 0, 1]),
]

print("测试可能导致四元数错误的情况...")

for i, test_func in enumerate(test_cases):
    try:
        result = test_func()
        print(f"测试 {i+1}: 成功 - {result}")
    except Exception as e:
        print(f"测试 {i+1}: 失败 - {e}")

# 测试特殊情况：可能某些地方在展开数组
print("\n测试数组展开情况...")
rotation_array = np.array([1, 0, 0, 0], dtype=np.float32)

try:
    # 使用*展开数组 - 这会导致错误
    quat = mn.Quaternion(*rotation_array)
    print("数组展开成功")
except Exception as e:
    print(f"数组展开失败: {e}")

# 这是正确的方式
try:
    quat = mn.Quaternion(
        mn.Vector3(rotation_array[0], rotation_array[1], rotation_array[2]),
        rotation_array[3]
    )
    print("正确方式成功")
except Exception as e:
    print(f"正确方式失败: {e}")
