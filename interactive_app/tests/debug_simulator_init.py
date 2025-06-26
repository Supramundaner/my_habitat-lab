#!/usr/bin/env python3
"""调试模拟器初始化问题"""

import sys
import os
import numpy as np

# 添加habitat-lab到路径
sys.path.append('/home/yaoaa/habitat-lab')

import habitat_sim
import magnum as mn

try:
    # 场景路径
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    
    # 配置后端
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    backend_cfg.random_seed = 1
    
    print("1. 创建临时模拟器获取场景边界...")
    temp_sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, []))
    temp_bounds = temp_sim.pathfinder.get_bounds()
    world_size_x = temp_bounds[1][0] - temp_bounds[0][0]
    world_size_z = temp_bounds[1][2] - temp_bounds[0][2]
    temp_sim.close()
    
    print(f"场景边界: {temp_bounds}")
    print(f"世界尺寸: {world_size_x:.2f} x {world_size_z:.2f}")
    
    # 计算保持纵横比的地图分辨率
    max_resolution = 1024
    aspect_ratio = world_size_x / world_size_z
    
    if aspect_ratio > 1:
        map_width = max_resolution
        map_height = int(max_resolution / aspect_ratio)
    else:
        map_height = max_resolution
        map_width = int(max_resolution * aspect_ratio)
    
    print(f"纵横比: {aspect_ratio:.2f}")
    print(f"计算的地图分辨率: {map_width} x {map_height}")
    
    # 配置FPV传感器
    fpv_sensor_spec = habitat_sim.CameraSensorSpec()
    fpv_sensor_spec.uuid = "color_sensor"
    fpv_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    fpv_sensor_spec.resolution = [512, 512]
    fpv_sensor_spec.position = mn.Vector3(0, 0, 0)
    fpv_sensor_spec.hfov = 90.0
    
    # 配置正交传感器
    ortho_sensor_spec = habitat_sim.CameraSensorSpec()
    ortho_sensor_spec.uuid = "ortho_sensor"
    ortho_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    ortho_sensor_spec.resolution = [map_width, map_height]
    ortho_sensor_spec.position = mn.Vector3(0, 0, 0)
    ortho_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    
    # 配置智能体
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [fpv_sensor_spec, ortho_sensor_spec]
    
    print("2. 创建完整模拟器...")
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    agent = sim.get_agent(0)
    
    print("3. 测试传感器...")
    # 移动到场景中心
    center = (temp_bounds[0] + temp_bounds[1]) / 2.0
    camera_height = temp_bounds[1][1] + 5.0
    camera_position = mn.Vector3(center[0], camera_height, center[2])
    
    agent_state = habitat_sim.AgentState()
    agent_state.position = camera_position
    agent_state.rotation = np.array([-0.7071068, 0, 0, 0.7071068], dtype=np.float32)  # 朝下看
    agent.set_state(agent_state)
    
    # 获取观察
    observations = sim.get_sensor_observations()
    ortho_img = observations["ortho_sensor"]
    
    print(f"正交图像形状: {ortho_img.shape}")
    print(f"正交图像类型: {ortho_img.dtype}")
    
    # 保存测试图像
    from PIL import Image
    if len(ortho_img.shape) == 3:
        if ortho_img.shape[2] == 4:
            test_image = Image.fromarray(ortho_img[..., :3], "RGB")
        else:
            test_image = Image.fromarray(ortho_img, "RGB")
        test_image.save('/home/yaoaa/habitat-lab/debug_ortho_image.png')
        print("保存调试正交图像: debug_ortho_image.png")
    
    sim.close()
    print("✅ 模拟器测试成功")
    
except Exception as e:
    print(f"❌ 模拟器测试失败: {e}")
    import traceback
    traceback.print_exc()

print("调试完成")
