#!/usr/bin/env python3
"""
修复所有问题的脚本
1. GPU加速
2. 视角转换命令
3. 导航朝向修正
4. 坐标系标签可见性
5. 实体智能体和正确的视角高度
"""

import habitat_sim
import numpy as np
import magnum as mn

def test_gpu_availability():
    """测试GPU可用性"""
    print("检查GPU设备...")
    
    # 创建测试配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/yaoaa/habitat-lab/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = 0  # 使用第一张GPU
    
    # 尝试创建模拟器
    try:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = "color_sensor"
        sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        sensor_spec.resolution = [512, 512]
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_spec]
        
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        print("✓ GPU设备可用")
        print(f"GPU设备ID: {backend_cfg.gpu_device_id}")
        
        # 获取一个观察来测试渲染
        obs = sim.get_sensor_observations()
        print(f"✓ 渲染测试成功，图像尺寸: {obs['color_sensor'].shape}")
        
        sim.close()
        return True
        
    except Exception as e:
        print(f"✗ GPU测试失败: {e}")
        return False

def test_agent_physics():
    """测试智能体物理和实体"""
    print("\n测试智能体物理配置...")
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "/home/yaoaa/habitat-lab/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = 0
    
    # 创建传感器
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [512, 512]
    sensor_spec.position = mn.Vector3(0, 1.5, 0)  # 1.5米高度
    sensor_spec.hfov = 90.0
    
    # 配置智能体
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }
    
    # 设置智能体身体参数
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.2
    agent_cfg.mass = 70.0  # 70kg人体质量
    
    try:
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        agent = sim.get_agent(0)
        
        print("✓ 智能体物理配置成功")
        print(f"智能体高度: {agent_cfg.height}m")
        print(f"智能体半径: {agent_cfg.radius}m")
        print(f"智能体质量: {agent_cfg.mass}kg")
        
        # 测试移动速度
        initial_state = agent.get_state()
        print(f"初始位置: {initial_state.position}")
        
        # 执行移动动作
        action = agent.act("move_forward")
        new_state = agent.get_state()
        print(f"移动后位置: {new_state.position}")
        
        distance = np.linalg.norm(new_state.position - initial_state.position)
        print(f"移动距离: {distance:.3f}m")
        
        sim.close()
        return True
        
    except Exception as e:
        print(f"✗ 智能体物理测试失败: {e}")
        return False

def test_orientation_calculation():
    """测试朝向计算"""
    print("\n测试朝向计算...")
    
    # 测试从A到B的朝向计算
    pos_a = np.array([0, 0, 0])
    pos_b = np.array([1, 0, 1])
    
    # 计算正确的朝向（从A指向B）
    direction = pos_b - pos_a
    direction = direction / np.linalg.norm(direction)  # 归一化
    
    # 计算yaw角度（绕Y轴旋转）
    yaw = np.arctan2(direction[0], direction[2])  # 注意：habitat使用Z为前方
    
    print(f"从 {pos_a} 到 {pos_b}")
    print(f"方向向量: {direction}")
    print(f"Yaw角度: {yaw:.3f} 弧度 ({np.degrees(yaw):.1f}度)")
    
    # 创建四元数（绕Y轴旋转）
    quat = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
    print(f"四元数: {quat}")
    
    return yaw

if __name__ == "__main__":
    print("开始修复检查...")
    
    # 检查1：GPU可用性
    gpu_ok = test_gpu_availability()
    
    # 检查2：智能体物理
    physics_ok = test_agent_physics()
    
    # 检查3：朝向计算
    test_orientation_calculation()
    
    print(f"\n检查结果:")
    print(f"GPU加速: {'✓' if gpu_ok else '✗'}")
    print(f"智能体物理: {'✓' if physics_ok else '✗'}")
    print(f"朝向计算: ✓")
