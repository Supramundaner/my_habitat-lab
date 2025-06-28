#!/usr/bin/env python3
"""
简化的公寓导航测试 - 专注于核心功能测试
"""

import sys
import os
import numpy as np
import time

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator


def test_apartment_basic():
    """基础公寓导航测试"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("=== 公寓基础导航测试 ===")
    print(f"场景: apartment_1.glb")
    
    output_dir = "./outputs/apartment_basic"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 创建视频生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=25,  # 降低帧率提高性能
            output_dir=output_dir
        )
        
        print(f"初始位置: {generator.get_agent_position()}")
        
        # 测试1: 简单环顾和前进
        print("\n--- 测试1: 环顾四周 ---")
        look_around_commands = [
            ["left", 90],   # 左转90度
            ["left", 90],   # 左转90度
            ["left", 90],   # 左转90度
            ["left", 90],   # 左转90度 (完成360度)
        ]
        
        video_path = generator.process_command_sequence(look_around_commands)
        if video_path:
            print(f"环顾视频已保存: {video_path}")
        
        # 测试2: 前进探索
        print("\n--- 测试2: 前进探索 ---")
        forward_exploration = [
            [1.0, 0.0],     # 向前1米
            ["left", 45],   # 左转45度观察
            ["right", 90],  # 右转90度观察
            ["left", 45],   # 回到前方
            [1.0, 0.0],     # 继续前进1米
        ]
        
        video_path = generator.process_command_sequence(forward_exploration)
        if video_path:
            print(f"前进探索视频已保存: {video_path}")
        
        # 测试3: 房间移动
        print("\n--- 测试3: 房间移动 ---")
        room_movement = [
            ["right", 90],  # 右转
            [0.0, 1.5],     # 向右移动1.5米
            ["left", 90],   # 左转面向前方
            [2.0, 0.0],     # 向前2米
            ["left", 90],   # 左转
            [0.0, 1.5],     # 向左移动1.5米
            ["left", 90],   # 左转回到起始方向
        ]
        
        video_path = generator.process_command_sequence(room_movement)
        if video_path:
            print(f"房间移动视频已保存: {video_path}")
        
        print(f"\n测试完成！最终位置: {generator.get_agent_position()}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'generator' in locals():
            generator.close()


def test_apartment_pathfinding():
    """公寓路径搜索测试"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("\n=== 公寓路径搜索测试 ===")
    
    output_dir = "./outputs/apartment_pathfinding"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=25,
            output_dir=output_dir
        )
        
        print(f"初始位置: {generator.get_agent_position()}")
        current_pos = generator.get_agent_position()
        
        # 测试不同距离的移动
        print("\n--- 路径搜索测试 ---")
        pathfinding_commands = [
            # 短距离移动
            [4.0,2.77],
            [0.0, 2.8],
            [0.0,0.0],
            [2.0, 0.5],
            ["left", 90],  # 左转
        ]
        
        video_path = generator.process_command_sequence(pathfinding_commands)
        if video_path:
            print(f"路径搜索视频已保存: {video_path}")
        
    except Exception as e:
        print(f"路径搜索测试中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'generator' in locals():
            generator.close()


def main():
    """主测试函数"""
    print("开始apartment_1.glb导航测试...")
    start_time = time.time()
    
    # 运行基础测试
    #test_apartment_basic()
    
    # 运行路径搜索测试
    test_apartment_pathfinding()
    
    end_time = time.time()
    print(f"\n所有测试完成！总耗时: {end_time - start_time:.2f}秒")
    print("请检查 ./outputs/ 目录中的视频文件")


if __name__ == "__main__":
    main()
