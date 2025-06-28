#!/usr/bin/env python3
"""
测试移动过程中人物朝向修复
验证从A点移动到B点时，人物是否正确朝向B点
"""

import sys
import os
import numpy as np
import math
from PIL import Image, ImageDraw

# 添加项目路径
sys.path.append('/home/yaoaa/habitat-lab/interactive_app/src')
sys.path.append('/home/yaoaa/habitat-lab')

from habitat_navigator_app import HabitatSimulator
import magnum as mn

def test_movement_orientation():
    """测试移动时的朝向"""
    print("=== 测试移动过程中人物朝向修复 ===\n")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    try:
        # 初始化模拟器
        print("1. 初始化模拟器...")
        simulator = HabitatSimulator(scene_path, resolution=(512, 512))
        print("✓ 模拟器初始化成功")
        
        # 设置测试点
        center = simulator.scene_center
        point_a = np.array([center[0] - 2, center[1], center[2]], dtype=np.float32)
        point_b = np.array([center[0] + 2, center[1], center[2]], dtype=np.float32)
        
        print(f"\n2. 测试从A点到B点的朝向:")
        print(f"   A点: ({point_a[0]:.2f}, {point_a[2]:.2f})")
        print(f"   B点: ({point_b[0]:.2f}, {point_b[2]:.2f})")
        
        # 计算期望的朝向（与主程序保持一致）
        direction = point_b - point_a
        direction_normalized = direction / np.linalg.norm(direction)
        expected_angle = math.atan2(direction_normalized[0], direction_normalized[2]) + math.pi  # 加180度修正
        
        print(f"   方向向量: ({direction_normalized[0]:.3f}, {direction_normalized[2]:.3f})")
        print(f"   期望朝向角度: {math.degrees(expected_angle):.1f}度")
        
        # 移动到A点
        print(f"\n3. 移动智能体到A点...")
        simulator.move_agent_to(point_a)
        state_a = simulator.get_agent_state()
        print(f"   实际到达位置: ({state_a.position[0]:.2f}, {state_a.position[2]:.2f})")
        
        # 创建朝向B点的旋转
        target_rotation = mn.Quaternion.rotation(mn.Rad(expected_angle), mn.Vector3.y_axis())
        target_rotation_array = np.array([
            target_rotation.vector.x, target_rotation.vector.y,
            target_rotation.vector.z, target_rotation.scalar
        ], dtype=np.float32)
        
        # 应用朝向并移动到A点
        simulator.move_agent_to(point_a, target_rotation_array)
        
        # 验证朝向
        state_oriented = simulator.get_agent_state()
        
        # 安全地处理rotation数据
        rotation_data = state_oriented.rotation
        if hasattr(rotation_data, 'x'):
            # quaternion类型
            rotation_array = np.array([rotation_data.x, rotation_data.y, rotation_data.z, rotation_data.w])
        elif isinstance(rotation_data, np.ndarray):
            rotation_array = rotation_data
        else:
            rotation_array = np.array(rotation_data)
        
        if len(rotation_array) != 4:
            print(f"警告: rotation数据格式异常: {rotation_array}")
            return False
            
        current_quat = mn.Quaternion(
            mn.Vector3(float(rotation_array[0]), float(rotation_array[1]), float(rotation_array[2])),
            float(rotation_array[3])
        )
        current_forward = current_quat.transform_vector(mn.Vector3(0, 0, -1))
        current_angle = math.atan2(current_forward.x, -current_forward.z)
        
        print(f"\n4. 验证朝向:")
        print(f"   当前朝向角度: {math.degrees(current_angle):.1f}度")
        print(f"   角度差异: {math.degrees(abs(current_angle - expected_angle)):.1f}度")
        
        # 生成测试图像
        print(f"\n5. 生成测试图像...")
        
        # 获取FPV图像
        fpv_image = simulator.sim.get_sensor_observations()["color_sensor"]
        fpv_pil = Image.fromarray(fpv_image[..., :3], "RGB")
        fpv_pil.save("/home/yaoaa/habitat-lab/interactive_app/images/test_orientation_fpv.png")
        print("   FPV图像已保存")
        
        # 生成带朝向标记的地图
        map_image = simulator.base_map_image.copy()
        draw = ImageDraw.Draw(map_image)
        
        # 转换坐标到地图像素
        def world_to_map_pixel(world_pos, image_size):
            world_min_x = simulator.scene_bounds[0][0]
            world_max_x = simulator.scene_bounds[1][0]
            world_min_z = simulator.scene_bounds[0][2]
            world_max_z = simulator.scene_bounds[1][2]
            
            # 考虑地图边距（参考原始实现）
            padding_left = 80
            padding_top = 40
            original_width = image_size[0] - padding_left - 40
            original_height = image_size[1] - padding_top - 60
            
            px = padding_left + int((world_pos[0] - world_min_x) / (world_max_x - world_min_x) * original_width)
            py = padding_top + int((world_pos[2] - world_min_z) / (world_max_z - world_min_z) * original_height)
            return (px, py)
        
        # 绘制A点和B点
        map_size = map_image.size
        px_a, py_a = world_to_map_pixel(point_a, map_size)
        px_b, py_b = world_to_map_pixel(point_b, map_size)
        
        # A点（蓝色）
        draw.ellipse([px_a-8, py_a-8, px_a+8, py_a+8], fill=(0, 0, 255), outline=(255, 255, 255), width=2)
        draw.text((px_a+12, py_a-8), "A", fill=(0, 0, 255))
        
        # B点（绿色）
        draw.ellipse([px_b-8, py_b-8, px_b+8, py_b+8], fill=(0, 255, 0), outline=(255, 255, 255), width=2)
        draw.text((px_b+12, py_b-8), "B", fill=(0, 255, 0))
        
        # 绘制智能体当前位置和朝向（红色）
        agent_px, agent_py = world_to_map_pixel(state_oriented.position, map_size)
        draw.ellipse([agent_px-6, agent_py-6, agent_px+6, agent_py+6], fill=(255, 0, 0))
        
        # 绘制朝向箭头
        arrow_length = 30
        arrow_end_x = agent_px + int(current_forward.x * arrow_length)
        arrow_end_y = agent_py + int(current_forward.z * arrow_length)
        
        draw.line([(agent_px, agent_py), (arrow_end_x, arrow_end_y)], fill=(255, 0, 0), width=3)
        
        # 箭头头部
        angle = math.atan2(current_forward.z, current_forward.x)
        head_length = 10
        head_angle1 = angle + math.pi * 0.8
        head_angle2 = angle - math.pi * 0.8
        
        head_x1 = arrow_end_x + int(math.cos(head_angle1) * head_length)
        head_y1 = arrow_end_y + int(math.sin(head_angle1) * head_length)
        head_x2 = arrow_end_x + int(math.cos(head_angle2) * head_length)
        head_y2 = arrow_end_y + int(math.sin(head_angle2) * head_length)
        
        draw.line([(arrow_end_x, arrow_end_y), (head_x1, head_y1)], fill=(255, 0, 0), width=2)
        draw.line([(arrow_end_x, arrow_end_y), (head_x2, head_y2)], fill=(255, 0, 0), width=2)
        
        # 绘制期望方向（虚线）
        expected_end_x = agent_px + int(direction_normalized[0] * arrow_length)
        expected_end_y = agent_py + int(direction_normalized[2] * arrow_length)
        
        # 虚线效果
        for i in range(0, arrow_length, 5):
            if i % 10 < 5:
                start_x = agent_px + int(direction_normalized[0] * i)
                start_y = agent_py + int(direction_normalized[2] * i)
                end_x = agent_px + int(direction_normalized[0] * min(i+3, arrow_length))
                end_y = agent_py + int(direction_normalized[2] * min(i+3, arrow_length))
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(255, 255, 0), width=2)
        
        # 添加说明文字
        draw.text((10, map_size[1]-80), f"红色实线: 当前朝向 ({math.degrees(current_angle):.1f}°)", fill=(255, 255, 255))
        draw.text((10, map_size[1]-60), f"黄色虚线: 期望朝向 ({math.degrees(expected_angle):.1f}°)", fill=(255, 255, 255))
        draw.text((10, map_size[1]-40), f"角度差异: {math.degrees(abs(current_angle - expected_angle)):.1f}°", fill=(255, 255, 255))
        draw.text((10, map_size[1]-20), "蓝色A → 红色智能体 → 绿色B", fill=(255, 255, 255))
        
        map_image.save("/home/yaoaa/habitat-lab/interactive_app/images/test_orientation_map.png")
        print("   地图图像已保存")
        
        # 判断测试结果
        angle_diff = math.degrees(abs(current_angle - expected_angle))
        if angle_diff < 10:  # 容忍10度误差
            print(f"\n✅ 测试通过！朝向正确 (误差: {angle_diff:.1f}°)")
            result = True
        else:
            print(f"\n❌ 测试失败！朝向错误 (误差: {angle_diff:.1f}°)")
            result = False
        
        # 清理
        simulator.close()
        return result
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_movement_orientation()
    if success:
        print("\n🎉 朝向修复测试通过！")
    else:
        print("\n🔧 朝向仍需进一步调整")
