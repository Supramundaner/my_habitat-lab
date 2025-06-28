#!/usr/bin/env python3
"""
简单的朝向调试测试
逐步检查角度计算和四元数应用
"""

import sys
import os
import numpy as np
import math

# 添加项目路径
sys.path.append('/home/yaoaa/habitat-lab/interactive_app/src')
sys.path.append('/home/yaoaa/habitat-lab')

from habitat_navigator_app import HabitatSimulator
import magnum as mn

def debug_orientation():
    """调试朝向计算"""
    print("=== 朝向调试测试 ===\n")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    try:
        simulator = HabitatSimulator(scene_path, resolution=(256, 256))
        print("✓ 模拟器初始化成功")
        
        # 测试各个方向的朝向
        test_directions = [
            ("东", np.array([1, 0, 0])),   # 正X方向
            ("南", np.array([0, 0, 1])),   # 正Z方向  
            ("西", np.array([-1, 0, 0])),  # 负X方向
            ("北", np.array([0, 0, -1])),  # 负Z方向
        ]
        
        center = simulator.scene_center
        test_pos = np.array([center[0], center[1], center[2]], dtype=np.float32)
        
        print(f"测试位置: ({test_pos[0]:.2f}, {test_pos[2]:.2f})")
        print()
        
        for direction_name, direction_vec in test_directions:
            print(f"测试朝向: {direction_name} {direction_vec}")
            
            # 计算角度（当前方法）
            angle1 = math.atan2(direction_vec[0], direction_vec[2]) + math.pi
            print(f"  方法1 (当前): {math.degrees(angle1):.1f}度")
            
            # 计算角度（备选方法）
            angle2 = math.atan2(direction_vec[0], -direction_vec[2])
            print(f"  方法2 (备选): {math.degrees(angle2):.1f}度")
            
            # 创建四元数并应用
            quat1 = mn.Quaternion.rotation(mn.Rad(angle1), mn.Vector3.y_axis())
            rotation_array1 = np.array([quat1.vector.x, quat1.vector.y, quat1.vector.z, quat1.scalar], dtype=np.float32)
            
            # 移动智能体并设置朝向
            simulator.move_agent_to(test_pos, rotation_array1)
            
            # 验证实际朝向
            state = simulator.get_agent_state()
            actual_rotation = state.rotation
            
            if hasattr(actual_rotation, 'x'):
                rotation_data = np.array([actual_rotation.x, actual_rotation.y, actual_rotation.z, actual_rotation.w])
            else:
                rotation_data = actual_rotation
            
            actual_quat = mn.Quaternion(
                mn.Vector3(float(rotation_data[0]), float(rotation_data[1]), float(rotation_data[2])),
                float(rotation_data[3])
            )
            
            # 计算实际的前方向量
            actual_forward = actual_quat.transform_vector(mn.Vector3(0, 0, -1))  # -Z是前方
            actual_angle = math.atan2(actual_forward.x, -actual_forward.z)
            
            print(f"  实际前方向量: ({actual_forward.x:.3f}, {actual_forward.z:.3f})")
            print(f"  实际角度: {math.degrees(actual_angle):.1f}度")
            
            # 检查是否匹配期望方向
            expected_forward_x = direction_vec[0]
            expected_forward_z = direction_vec[2]
            
            error_x = abs(actual_forward.x - expected_forward_x)
            error_z = abs(actual_forward.z - expected_forward_z)
            
            if error_x < 0.1 and error_z < 0.1:
                print(f"  ✅ 朝向正确!")
            else:
                print(f"  ❌ 朝向错误 (误差: x={error_x:.3f}, z={error_z:.3f})")
            
            print()
        
        # 特殊测试：从A点朝向B点
        print("特殊测试：A点朝向B点")
        point_a = np.array([center[0] - 1, center[1], center[2]], dtype=np.float32)
        point_b = np.array([center[0] + 1, center[1], center[2]], dtype=np.float32)
        
        direction = point_b - point_a
        direction = direction / np.linalg.norm(direction)
        
        print(f"A点: ({point_a[0]:.2f}, {point_a[2]:.2f})")
        print(f"B点: ({point_b[0]:.2f}, {point_b[2]:.2f})")
        print(f"A->B方向: ({direction[0]:.3f}, {direction[2]:.3f})")
        
        # 测试不同的角度计算方法
        methods = [
            ("当前", lambda d: math.atan2(d[0], d[2]) + math.pi),
            ("备选1", lambda d: math.atan2(d[0], -d[2])),
            ("备选2", lambda d: math.atan2(d[0], d[2])),
            ("备选3", lambda d: math.atan2(-d[0], -d[2])),
        ]
        
        for method_name, calc_func in methods:
            angle = calc_func(direction)
            quat = mn.Quaternion.rotation(mn.Rad(angle), mn.Vector3.y_axis())
            rotation_array = np.array([quat.vector.x, quat.vector.y, quat.vector.z, quat.scalar], dtype=np.float32)
            
            simulator.move_agent_to(point_a, rotation_array)
            state = simulator.get_agent_state()
            
            if hasattr(state.rotation, 'x'):
                rot_data = np.array([state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w])
            else:
                rot_data = state.rotation
                
            actual_quat = mn.Quaternion(
                mn.Vector3(float(rot_data[0]), float(rot_data[1]), float(rot_data[2])),
                float(rot_data[3])
            )
            actual_forward = actual_quat.transform_vector(mn.Vector3(0, 0, -1))
            
            # 检查是否朝向B点
            dot_product = actual_forward.x * direction[0] + actual_forward.z * direction[2]
            
            print(f"  {method_name}: 角度={math.degrees(angle):.1f}°, "
                  f"前方向量=({actual_forward.x:.3f}, {actual_forward.z:.3f}), "
                  f"相似度={dot_product:.3f}")
        
        simulator.close()
        return True
        
    except Exception as e:
        print(f"✗ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_orientation()
