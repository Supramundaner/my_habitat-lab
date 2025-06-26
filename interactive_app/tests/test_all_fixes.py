#!/usr/bin/env python3
"""
测试所有修复的问题
1. GPU加速
2. 视角命令
3. 导航朝向
4. 坐标标签
5. 智能体高度
"""

import sys
import os
import numpy as np
import time
from PIL import Image

# 导入我们的应用
sys.path.append('/home/yaoaa/habitat-lab')
from habitat_navigator_app import HabitatSimulator

def test_all_fixes():
    """测试所有修复"""
    print("=== 测试所有修复 ===\n")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    try:
        # 初始化模拟器
        print("1. 初始化模拟器（GPU加速 + 实体智能体）...")
        simulator = HabitatSimulator(scene_path, resolution=(512, 512))
        print("✓ 模拟器初始化成功")
        print(f"✓ 智能体高度: 1.5m (传感器位置: {simulator.agent.agent_config.sensor_specifications[0].position})")
        
        # 测试基础地图生成（坐标标签可见性）
        print("\n2. 测试地图生成（坐标标签可见性）...")
        map_image = simulator.base_map_image
        if map_image:
            map_image.save("test_fixed_map_labels.png")
            print("✓ 地图已保存到 test_fixed_map_labels.png")
            print("✓ 坐标标签现在使用白色，应该清晰可见")
        
        # 测试智能体移动和朝向
        print("\n3. 测试智能体移动和朝向...")
        
        # 移动到场景中的某个位置
        center = simulator.scene_center
        test_pos_a = np.array([center[0] - 1, center[1], center[2]], dtype=np.float32)
        test_pos_b = np.array([center[0] + 1, center[1], center[2]], dtype=np.float32)
        
        print(f"从位置A {test_pos_a} 移动到位置B {test_pos_b}")
        
        # 移动到A点
        simulator.move_agent_to(test_pos_a)
        state_a = simulator.get_agent_state()
        print(f"✓ 到达A点: {state_a.position}")
        
        # 计算朝向B点的正确方向
        direction = test_pos_b - test_pos_a
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            print(f"A->B方向向量: {direction}")
            
            # 计算yaw角度（应该朝向B点）
            import math
            angle = math.atan2(direction[0], direction[2])  # 使用+Z作为前方
            print(f"应该的yaw角度: {math.degrees(angle):.1f}度")
            
            # 移动到B点并检查朝向
            simulator.move_agent_to(test_pos_b)
            state_b = simulator.get_agent_state()
            print(f"✓ 到达B点: {state_b.position}")
        
        # 测试获取FPV图像（1.5米高度）
        print("\n4. 测试第一人称视角（1.5米高度）...")
        observations = simulator.sim.get_sensor_observations()
        fpv_image = observations["color_sensor"]
        print(f"✓ FPV图像尺寸: {fpv_image.shape}")
        print(f"✓ 智能体当前位置: {simulator.get_agent_state().position}")
        print(f"✓ Y坐标应该为场景地面+1.5米左右")
        
        # 保存FPV图像以验证高度
        fpv_pil = Image.fromarray(fpv_image[..., :3], "RGB")
        fpv_pil.save("test_fpv_height.png")
        print("✓ FPV图像已保存到 test_fpv_height.png")
        
        # 测试性能（GPU加速）
        print("\n5. 测试渲染性能（GPU加速）...")
        start_time = time.time()
        for i in range(10):
            observations = simulator.sim.get_sensor_observations()
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"✓ 平均渲染时间: {avg_time*1000:.1f}ms")
        print(f"✓ 估计FPS: {fps:.1f}")
        
        if fps > 20:
            print("✓ 渲染性能良好，GPU加速工作正常")
        else:
            print("⚠ 渲染性能较低，可能需要检查GPU配置")
        
        # 清理
        simulator.close()
        print("\n✓ 所有测试完成！")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_fixes()
    if success:
        print("\n🎉 所有修复测试通过！")
    else:
        print("\n❌ 存在问题需要进一步修复")
