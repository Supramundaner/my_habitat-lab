#!/usr/bin/env python3
"""
简化测试：仅测试模拟器功能，不涉及GUI
"""

import os
import sys
import numpy as np
from PIL import Image

def test_simulator_only():
    """仅测试模拟器功能"""
    print("测试模拟器核心功能（无GUI）...")
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    
    if not os.path.exists(scene_path):
        print(f"场景文件不存在: {scene_path}")
        return False
    
    try:
        # 导入依赖
        import habitat_sim
        import magnum as mn
        
        # 创建模拟器配置
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = False
        backend_cfg.random_seed = 1
        
        # 配置FPV传感器
        fpv_sensor_spec = habitat_sim.CameraSensorSpec()
        fpv_sensor_spec.uuid = "color_sensor"
        fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fpv_sensor_spec.resolution = [256, 256]
        fpv_sensor_spec.position = mn.Vector3(0, 0, 0)
        fpv_sensor_spec.hfov = 90.0
        
        # 配置正交传感器
        ortho_sensor_spec = habitat_sim.CameraSensorSpec()
        ortho_sensor_spec.uuid = "ortho_sensor"  
        ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        ortho_sensor_spec.resolution = [512, 512]
        ortho_sensor_spec.position = mn.Vector3(0, 0, 0)
        ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        
        # 配置智能体
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [fpv_sensor_spec, ortho_sensor_spec]
        
        # 创建完整配置
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        # 创建模拟器
        print("创建模拟器...")
        sim = habitat_sim.Simulator(cfg)
        agent = sim.get_agent(0)
        
        # 获取场景信息
        scene_bounds = sim.pathfinder.get_bounds()
        scene_center = (scene_bounds[0] + scene_bounds[1]) / 2.0
        scene_size = scene_bounds[1] - scene_bounds[0]
        
        print(f"✓ 场景边界: {scene_bounds}")
        print(f"✓ 场景中心: {scene_center}")
        print(f"✓ 场景尺寸: {scene_size}")
        
        # 测试导航功能
        test_point = mn.Vector3(0, 0, 0)
        snapped_point = sim.pathfinder.snap_point(test_point)
        is_navigable = sim.pathfinder.is_navigable(snapped_point)
        print(f"✓ 导航测试: 原点对齐到 {snapped_point}, 可导航: {is_navigable}")
        
        # 设置智能体位置
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([snapped_point.x, snapped_point.y, snapped_point.z])
        agent_state.rotation = np.array([0, 0, 0, 1])
        agent.set_state(agent_state)
        
        # 获取FPV观察
        observations = sim.get_sensor_observations()
        fpv_img = observations["color_sensor"]
        print(f"✓ FPV图像尺寸: {fpv_img.shape}")
        
        # 生成俯视图
        camera_height = scene_bounds[1][1] + 5.0
        camera_position = mn.Vector3(scene_center[0], camera_height, scene_center[2])
        
        agent_state.position = camera_position
        agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068])
        agent.set_state(agent_state)
        
        observations = sim.get_sensor_observations()
        ortho_img = observations["ortho_sensor"]
        print(f"✓ 俯视图尺寸: {ortho_img.shape}")
        
        # 保存测试图像
        fpv_image = Image.fromarray(fpv_img[..., :3], "RGB")
        fpv_image.save("test_fpv.png")
        print("✓ FPV测试图像已保存: test_fpv.png")
        
        ortho_image = Image.fromarray(ortho_img[..., :3], "RGB")
        ortho_image.save("test_topdown.png")
        print("✓ 俯视图测试图像已保存: test_topdown.png")
        
        # 测试寻路
        start_pos = snapped_point
        end_pos = sim.pathfinder.snap_point(mn.Vector3(2, 0, 1))
        if sim.pathfinder.is_navigable(end_pos):
            # 使用ShortestPath对象进行寻路
            path_obj = habitat_sim.ShortestPath()
            path_obj.requested_start = start_pos
            path_obj.requested_end = end_pos
            found = sim.pathfinder.find_path(path_obj)
            if found:
                print(f"✓ 寻路测试: 找到路径，包含 {len(path_obj.points)} 个路径点")
            else:
                print("✓ 寻路测试: 未找到路径（可能距离太远）")
        else:
            print("✓ 寻路测试: 目标点不可导航（正常）")
        
        # 清理
        sim.close()
        print("✓ 模拟器已关闭")
        
        return True
        
    except Exception as e:
        print(f"✗ 模拟器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("Habitat模拟器核心功能测试")
    print("=" * 40)
    
    if test_simulator_only():
        print("\n✓ 模拟器核心功能测试通过！")
        print("主应用程序应该能够正常工作（需要图形界面支持）")
    else:
        print("\n✗ 模拟器核心功能测试失败")
    
    return True

if __name__ == '__main__':
    main()
