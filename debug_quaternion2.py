#!/usr/bin/env python3
"""测试嵌套元组构造四元数"""

import numpy as np
import magnum as mn

print("测试嵌套元组构造...")

test_rotation = np.array([0, 0, 0, 1], dtype=np.float32)

try:
    # 方式3: 嵌套元组格式 ((x, y, z), w)
    quat = mn.Quaternion(((test_rotation[0], test_rotation[1], test_rotation[2]), test_rotation[3]))
    print("嵌套元组方式成功!")
    print(f"构造的四元数: {quat}")
except Exception as e:
    print(f"嵌套元组方式失败: {e}")

# 测试不同的值
test_rotation2 = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)

try:
    quat2 = mn.Quaternion(((float(test_rotation2[0]), float(test_rotation2[1]), float(test_rotation2[2])), float(test_rotation2[3])))
    print("非零值嵌套元组方式成功!")
    print(f"构造的四元数: {quat2}")
except Exception as e:
    print(f"非零值嵌套元组方式失败: {e}")
