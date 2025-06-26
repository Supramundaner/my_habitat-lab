#!/usr/bin/env python3
"""调试四元数构造问题"""

import numpy as np
import magnum as mn

print("测试四元数构造...")

# 测试不同的四元数构造方式
test_rotation = np.array([0, 0, 0, 1], dtype=np.float32)

print(f"测试数组: {test_rotation}")
print(f"数组形状: {test_rotation.shape}")
print(f"数组类型: {test_rotation.dtype}")

try:
    # 方式1: 直接传入数组
    quat1 = mn.Quaternion(test_rotation)
    print("方式1成功: 直接传入数组")
except Exception as e:
    print(f"方式1失败: {e}")

try:
    # 方式2: 分别传入向量和标量
    quat2 = mn.Quaternion(
        mn.Vector3(test_rotation[0], test_rotation[1], test_rotation[2]),
        test_rotation[3]
    )
    print("方式2成功: 分别传入向量和标量")
except Exception as e:
    print(f"方式2失败: {e}")

try:
    # 方式3: 使用四个单独参数
    quat3 = mn.Quaternion(test_rotation[0], test_rotation[1], test_rotation[2], test_rotation[3])
    print("方式3成功: 四个单独参数")
except Exception as e:
    print(f"方式3失败: {e}")

try:
    # 方式4: 从列表构造
    quat4 = mn.Quaternion([test_rotation[0], test_rotation[1], test_rotation[2], test_rotation[3]])
    print("方式4成功: 从列表构造")
except Exception as e:
    print(f"方式4失败: {e}")

print("测试完成")
