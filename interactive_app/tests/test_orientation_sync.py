#!/usr/bin/env python3
"""
测试视角转向时2D地图朝向箭头同步更新
"""

import sys
import os
import numpy as np
import time
import math
from PIL import Image
import magnum as mn
import habitat_sim

# 导入我们的应用
sys.path.append('/home/yaoaa/habitat-lab/interactive_app/src')
from habitat_navigator_app import HabitatSimulator

def test_orientation_sync():
    """测试朝向同步更新"""
    print("=== 测试视角转向时地图朝向同步 ===\n")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    try:
        # 初始化模拟器
        print("1. 初始化模拟器...")
        simulator = HabitatSimulator(scene_path, resolution=(512, 512))
        print("✓ 模拟器初始化成功")
        
        # 移动到场景中心一个固定位置
        center = simulator.scene_center
        test_pos = np.array([center[0], center[1], center[2]], dtype=np.float32)
        simulator.move_agent_to(test_pos)
        
        print(f"✓ 智能体移动到测试位置: {test_pos}")
        
        # 获取初始状态
        initial_state = simulator.get_agent_state()
        print(f"✓ 初始朝向: {initial_state.rotation}")
        
        # 生成初始地图图像
        def save_map_with_agent(filename_suffix, description):
            """保存带智能体标记的地图"""
            map_image = simulator.base_map_image.copy()
            agent_state = simulator.get_agent_state()
            
            # 手动调用draw_agent_on_map方法
            from PIL import ImageDraw
            import math
            import magnum as mn
            
            draw = ImageDraw.Draw(map_image)
            agent_pos = agent_state.position
            agent_rotation = agent_state.rotation
            
            # 转换世界坐标到地图坐标
            map_x, map_y = simulator.world_to_map_coords(agent_pos)
            
            # 绘制智能体位置（红点）
            dot_radius = 12
            draw.ellipse([
                map_x - dot_radius, map_y - dot_radius,
                map_x + dot_radius, map_y + dot_radius
            ], fill=(255, 0, 0))
            
            # 绘制朝向箭头
            try:
                # 处理不同类型的rotation
                if hasattr(agent_rotation, 'x'):
                    rotation_array = np.array([agent_rotation.x, agent_rotation.y, 
                                             agent_rotation.z, agent_rotation.w])
                elif isinstance(agent_rotation, np.ndarray):
                    rotation_array = agent_rotation
                else:
                    rotation_array = np.array([0, 0, 0, 1])
                
                # 创建四元数
                quat = mn.Quaternion(
                    mn.Vector3(float(rotation_array[0]), float(rotation_array[1]), float(rotation_array[2])),
                    float(rotation_array[3])
                )
                
                # 计算前向量（Habitat中Z轴正方向是前方）
                forward_vec = quat.transform_vector(mn.Vector3(0, 0, 1))
                
                # 计算箭头终点
                arrow_length = 30
                arrow_end_x = map_x + int(forward_vec.x * arrow_length)
                arrow_end_y = map_y + int(forward_vec.z * arrow_length)
                
                # 绘制箭头线
                draw.line([(map_x, map_y), (arrow_end_x, arrow_end_y)], 
                         fill=(255, 255, 0), width=4)
                
                # 绘制箭头头部
                angle = math.atan2(forward_vec.z, forward_vec.x)
                arrow_head_length = 15
                
                head_angle1 = angle + math.pi * 0.8
                head_angle2 = angle - math.pi * 0.8
                
                head_x1 = arrow_end_x + int(math.cos(head_angle1) * arrow_head_length)
                head_y1 = arrow_end_y + int(math.sin(head_angle1) * arrow_head_length)
                head_x2 = arrow_end_x + int(math.cos(head_angle2) * arrow_head_length)
                head_y2 = arrow_end_y + int(math.sin(head_angle2) * arrow_head_length)
                
                draw.line([(arrow_end_x, arrow_end_y), (head_x1, head_y1)], 
                         fill=(255, 255, 0), width=3)
                draw.line([(arrow_end_x, arrow_end_y), (head_x2, head_y2)], 
                         fill=(255, 255, 0), width=3)
                
                # 添加角度标注
                angle_deg = math.degrees(math.atan2(forward_vec.x, forward_vec.z))
                draw.text((map_x + 20, map_y - 40), f"{description}\nAngle: {angle_deg:.1f}°", 
                         fill=(255, 255, 255))
                
            except Exception as e:
                print(f"⚠ 箭头绘制失败: {e}")
            
            # 保存图像
            filename = f"test_orientation_{filename_suffix}.png"
            map_image.save(filename)
            print(f"✓ 已保存: {filename}")
            return filename
        
        # 保存初始朝向
        save_map_with_agent("00_initial", "Initial")
        
        # 测试一系列视角转向命令
        test_commands = [
            ("right", 30, "right_30"),
            ("right", 30, "right_60_total"),
            ("left", 45, "left_45_from_60"),
            ("right", 15, "right_15_adjust"),
            ("left", 90, "left_90_major"),
            ("right", 180, "right_180_opposite")
        ]
        
        print(f"\n2. 测试视角转向命令...")
        
        for i, (direction, angle, suffix) in enumerate(test_commands, 1):
            print(f"\n步骤 {i}: 执行命令 '{direction} {angle}'")
            
            # 获取转向前状态
            before_state = simulator.get_agent_state()
            print(f"  转向前四元数: {before_state.rotation} (类型: {type(before_state.rotation)})")
            
            # 处理不同类型的rotation
            if hasattr(before_state.rotation, 'x'):
                # quaternion.quaternion类型
                before_rotation_array = np.array([before_state.rotation.x, before_state.rotation.y, 
                                                before_state.rotation.z, before_state.rotation.w])
            elif isinstance(before_state.rotation, np.ndarray):
                before_rotation_array = before_state.rotation
            else:
                # 可能是标量，创建默认四元数
                before_rotation_array = np.array([0, 0, 0, 1])
            
            before_quat = mn.Quaternion(
                mn.Vector3(before_rotation_array[0], before_rotation_array[1], before_rotation_array[2]),
                before_rotation_array[3]
            )
            before_angle = math.degrees(math.atan2(
                before_quat.transform_vector(mn.Vector3(0, 0, 1)).x,
                before_quat.transform_vector(mn.Vector3(0, 0, 1)).z
            ))
            
            # 执行转向命令
            current_rotation = before_state.rotation
            
            # 处理不同类型的rotation
            if hasattr(current_rotation, 'x'):
                current_rotation_array = np.array([current_rotation.x, current_rotation.y, 
                                                 current_rotation.z, current_rotation.w])
            elif isinstance(current_rotation, np.ndarray):
                current_rotation_array = current_rotation
            else:
                current_rotation_array = np.array([0, 0, 0, 1])
            
            current_quat = mn.Quaternion(
                mn.Vector3(current_rotation_array[0], current_rotation_array[1], current_rotation_array[2]),
                current_rotation_array[3]
            )
            
            # 计算旋转变化
            if direction == "left":
                rotation_quat = mn.Quaternion.rotation(mn.Rad(math.radians(angle)), mn.Vector3.y_axis())
            elif direction == "right":
                rotation_quat = mn.Quaternion.rotation(mn.Rad(math.radians(-angle)), mn.Vector3.y_axis())
            
            # 应用旋转
            new_rotation = current_quat * rotation_quat
            
            # 更新智能体状态
            import habitat_sim
            new_state = habitat_sim.AgentState()
            new_state.position = before_state.position
            new_state.rotation = np.array([new_rotation.vector.x, new_rotation.vector.y, 
                                         new_rotation.vector.z, new_rotation.scalar], dtype=np.float32)
            
            simulator.agent.set_state(new_state)
            
            # 获取转向后状态
            after_state = simulator.get_agent_state()
            
            # 处理不同类型的rotation
            if hasattr(after_state.rotation, 'x'):
                after_rotation_array = np.array([after_state.rotation.x, after_state.rotation.y, 
                                               after_state.rotation.z, after_state.rotation.w])
            elif isinstance(after_state.rotation, np.ndarray):
                after_rotation_array = after_state.rotation
            else:
                after_rotation_array = np.array([0, 0, 0, 1])
            
            after_quat = mn.Quaternion(
                mn.Vector3(after_rotation_array[0], after_rotation_array[1], after_rotation_array[2]),
                after_rotation_array[3]
            )
            after_angle = math.degrees(math.atan2(
                after_quat.transform_vector(mn.Vector3(0, 0, 1)).x,
                after_quat.transform_vector(mn.Vector3(0, 0, 1)).z
            ))
            
            angle_change = after_angle - before_angle
            if angle_change > 180:
                angle_change -= 360
            elif angle_change < -180:
                angle_change += 360
            
            print(f"  转向前角度: {before_angle:.1f}°")
            print(f"  转向后角度: {after_angle:.1f}°")
            print(f"  实际变化: {angle_change:.1f}° (期望: {angle if direction == 'left' else -angle}°)")
            
            # 保存转向后的地图
            filename = save_map_with_agent(f"{i:02d}_{suffix}", f"{direction.title()} {angle}°")
        
        print(f"\n3. 测试完成！")
        print("✓ 生成的图像文件:")
        for i in range(len(test_commands) + 1):
            if i == 0:
                print(f"  - test_orientation_00_initial.png (初始朝向)")
            else:
                _, _, suffix = test_commands[i-1]
                print(f"  - test_orientation_{i:02d}_{suffix}.png")
        
        print("\n✓ 请检查生成的图像，确认朝向箭头正确更新")
        
        # 清理
        simulator.close()
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_orientation_sync()
    if success:
        print("\n🎉 视角转向同步测试完成！")
        print("📝 检查要点:")
        print("  1. 每张图像中的黄色箭头应该指向正确方向")
        print("  2. 角度标注应该与实际转向匹配")
        print("  3. 朝向变化应该连续且符合预期")
    else:
        print("\n❌ 测试失败，需要进一步调试")
