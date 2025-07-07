#!/usr/bin/env python3
"""
测试高级碰撞检测系统
演示动态碰撞检测和物理引擎集成
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multi_agent_navigation import MultiAgentSimulator


def create_collision_test_actions():
    """创建测试碰撞检测的动作序列"""
    # 创建会产生碰撞的动作序列
    collision_actions = {
        "agent_0": [
            {"action": "move_to", "target": [0.0, 0.0]},
            {"action": "move_to", "target": [2.0, 0.0]},  # 向右移动
            {"action": "move_to", "target": [2.0, 2.0]},  # 向上移动
            {"action": "move_to", "target": [0.0, 2.0]},  # 向左移动
        ],
        "agent_1": [
            {"action": "move_to", "target": [2.0, 0.0]},
            {"action": "move_to", "target": [0.0, 0.0]},  # 向左移动（与agent_0相向）
            {"action": "move_to", "target": [0.0, 2.0]},  # 向上移动
            {"action": "move_to", "target": [2.0, 2.0]},  # 向右移动
        ]
    }
    
    return collision_actions


def create_safe_test_actions():
    """创建安全的测试动作序列"""
    safe_actions = {
        "agent_0": [
            {"action": "move_to", "target": [0.0, 0.0]},
            {"action": "move_to", "target": [3.0, 0.0]},  # 向右移动
            {"action": "move_to", "target": [3.0, 3.0]},  # 向上移动
            {"action": "move_to", "target": [0.0, 3.0]},  # 向左移动
        ],
        "agent_1": [
            {"action": "move_to", "target": [0.0, -3.0]},
            {"action": "move_to", "target": [3.0, -3.0]},  # 向右移动（远离agent_0）
            {"action": "move_to", "target": [3.0, 0.0]},   # 向上移动
            {"action": "move_to", "target": [0.0, 0.0]},   # 向左移动
        ]
    }
    
    return safe_actions


def test_collision_detection():
    """测试碰撞检测功能"""
    print("=" * 60)
    print("高级碰撞检测系统测试")
    print("=" * 60)
    
    # 加载配置
    config_path = "config/advanced_collision_test_config.yaml"
    if not os.path.exists(config_path):
        print(f"测试配置文件不存在，尝试使用默认配置: {config_path}")
        config_path = "config/multi_agent_config.yaml"
        if not os.path.exists(config_path):
            print(f"默认配置文件也不存在: {config_path}")
            return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 启用详细日志
    config["logging"]["log_level"] = "INFO"  # 改为INFO避免过多日志
    config["logging"]["console_output"] = True
    
    # 确保启用物理引擎和碰撞检测
    config["simulator"]["enable_physics"] = True
    config["collision_detection"]["enabled"] = True
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # 初始化多智能体模拟器
        print("初始化多智能体模拟器...")
        simulator = MultiAgentSimulator(config)
        
        # 启用碰撞可视化
        simulator.enable_collision_visualization()
        
        # 打印智能体状态
        simulator.print_agent_status()
        
        # 生成初始碰撞报告
        print("\n生成初始碰撞检测报告...")
        initial_report = simulator.generate_collision_report()
        print(initial_report)
        
        # 性能基准测试
        print("\n执行碰撞检测性能基准测试...")
        benchmark_results = simulator.benchmark_collision_detection(num_tests=50)
        print("基准测试结果:")
        for key, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f} seconds" if "time" in key else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # 测试1: 碰撞动作序列
        print("\n测试1: 执行会产生碰撞的动作序列...")
        collision_actions = create_collision_test_actions()
        
        # 保存动作到文件
        collision_actions_file = "outputs/collision_test_actions.json"
        with open(collision_actions_file, 'w') as f:
            json.dump(collision_actions, f, indent=2)
        
        print(f"动作序列已保存到: {collision_actions_file}")
        
        # 执行碰撞测试
        print("执行碰撞测试动作...")
        result1 = simulator.execute_actions_sequence(collision_actions)
        print(f"碰撞测试结果: {'成功' if result1 else '失败'}")
        
        # 获取碰撞统计和可视化数据
        collision_stats = simulator.collision_detector.get_collision_statistics()
        print(f"碰撞统计: {collision_stats}")
        
        viz_data = simulator.get_collision_visualization_data()
        if viz_data.get("status") != "no_data":
            print(f"可视化数据: 接触点数量 = {len(viz_data.get('contact_points', []))}, "
                  f"碰撞对数量 = {len(viz_data.get('collision_pairs', []))}")
        
        # 保存调试数据
        debug_file = simulator.save_collision_debug_data("collision_test_debug.json")
        if debug_file:
            print(f"调试数据已保存到: {debug_file}")
        
        # 测试2: 安全动作序列
        print("\n测试2: 执行安全的动作序列...")
        safe_actions = create_safe_test_actions()
        
        # 保存动作到文件
        safe_actions_file = "outputs/safe_test_actions.json"
        with open(safe_actions_file, 'w') as f:
            json.dump(safe_actions, f, indent=2)
        
        print(f"动作序列已保存到: {safe_actions_file}")
        
        # 执行安全测试
        print("执行安全测试动作...")
        result2 = simulator.execute_actions_sequence(safe_actions)
        print(f"安全测试结果: {'成功' if result2 else '失败'}")
        
        # 获取最终统计
        final_stats = simulator.collision_detector.get_collision_statistics()
        print(f"最终统计: {final_stats}")
        
        # 生成最终报告
        print("\n生成最终碰撞检测报告...")
        final_report = simulator.generate_collision_report()
        print(final_report)
        
        # 保存最终调试数据
        final_debug_file = simulator.save_collision_debug_data("final_collision_debug.json")
        if final_debug_file:
            print(f"最终调试数据已保存到: {final_debug_file}")
        
        # 打印最终状态
        print("\n最终智能体状态:")
        simulator.print_agent_status()
        
        # 关闭模拟器
        simulator.close()
        
        print("\n测试完成！")
        print("查看输出文件:")
        print(f"  - 视频输出: outputs/")
        print(f"  - 日志文件: {config['logging']['log_file']}")
        print(f"  - 状态文件: {config['state_persistence']['state_file']}")
        print(f"  - 调试数据: {debug_file if debug_file else 'N/A'}")
        print(f"  - 最终调试数据: {final_debug_file if final_debug_file else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("Advanced Collision Detection Test")
    print("高级碰撞检测测试")
    print()
    
    success = test_collision_detection()
    
    if success:
        print("✓ 所有测试完成")
        return 0
    else:
        print("✗ 测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
