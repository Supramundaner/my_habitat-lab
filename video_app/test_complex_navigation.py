#!/usr/bin/env python3
"""
复杂导航测试 - 使用apartment_1.glb场景
测试各种复杂的导航模式，包括房间探索、路径规划、转向行为等
"""

import sys
import os
import numpy as np
import time

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator


def test_apartment_exploration():
    """公寓探索测试 - 模拟真实的房间探索行为"""
    
    # 场景文件路径
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    # 检查场景文件是否存在
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("=== 公寓复杂导航测试 ===")
    print(f"场景: {scene_path}")
    
    # 创建输出目录
    output_dir = "./outputs/apartment_exploration"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 初始化视频生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir=output_dir
        )
        
        print(f"初始位置: {generator.get_agent_position()}")
        print(f"初始旋转: {generator.get_agent_rotation()}")
        
        # 测试1: 房间系统性探索
        print("\n--- 测试1: 房间系统性探索 ---")
        room_exploration_commands = [
            # 初始环顾四周
            ["left", 90],   # 向左转90度
            ["left", 90],   # 继续左转90度
            ["left", 90],   # 再左转90度
            ["left", 90],   # 完成360度环顾
            
            # 探索前方区域
            [2.0, 0.0],     # 向前移动2米
            ["right", 45],  # 右转45度观察
            ["left", 90],   # 左转90度观察另一侧
            ["right", 45],  # 回到正前方
            
            # 探索左侧区域
            ["left", 90],   # 面向左侧
            [0.0, 2.0],     # 向左移动2米
            ["right", 45],  # 观察前方
            ["left", 90],   # 观察后方
            ["right", 45],  # 回到前方
            
            # 探索对角线移动
            ["right", 45],  # 转向对角线方向
            [1.5, 1.5],     # 对角线移动
            ["left", 45],   # 调整方向
            
            # 探索房间深处
            [0.0, 3.0],     # 向前深入3米
            ["left", 180],  # 转身看来路
            ["right", 180], # 再转回去
        ]
        
        video_path = generator.process_command_sequence(room_exploration_commands)
        if video_path:
            print(f"房间探索视频已保存: {video_path}")
        
        # 测试2: 复杂路径导航
        print("\n--- 测试2: 复杂路径导航 ---")
        complex_navigation_commands = [
            # Z字形路径
            [3.0, 0.0],     # 向前
            ["right", 90],  # 右转
            [0.0, 2.0],     # 向右
            ["left", 90],   # 左转
            [3.0, 0.0],     # 向前
            ["right", 90],  # 右转
            [0.0, 2.0],     # 向右
            
            # 螺旋式探索
            ["left", 90],   # 调整方向
            [1.0, 0.0],     # 前进1米
            ["left", 90],   # 左转
            [1.0, 0.0],     # 前进1米
            ["left", 90],   # 左转
            [2.0, 0.0],     # 前进2米
            ["left", 90],   # 左转
            [2.0, 0.0],     # 前进2米
            ["left", 90],   # 左转
            [3.0, 0.0],     # 前进3米
        ]
        
        video_path = generator.process_command_sequence(complex_navigation_commands)
        if video_path:
            print(f"复杂路径导航视频已保存: {video_path}")
        
        # 测试3: 精确转向测试
        print("\n--- 测试3: 精确转向测试 ---")
        precision_turning_commands = [
            # 小角度精确转向
            ["left", 15],   # 15度
            ["right", 30],  # -30度（相对于初始方向-15度）
            ["left", 45],   # +45度（相对于初始方向+30度）
            ["right", 60],  # -60度（相对于初始方向-30度）
            
            # 大角度转向
            ["left", 120],  # 大幅左转
            ["right", 240], # 大幅右转（超过180度）
            ["left", 120],  # 回到原始方向
            
            # 连续精确转向
            ["left", 10],
            ["left", 10],
            ["left", 10],
            ["left", 10],
            ["left", 10],   # 总共50度
            ["right", 50],  # 回到原位
        ]
        
        video_path = generator.process_command_sequence(precision_turning_commands)
        if video_path:
            print(f"精确转向测试视频已保存: {video_path}")
        
        # 测试4: 房间间移动模拟
        print("\n--- 测试4: 房间间移动模拟 ---")
        room_to_room_commands = [
            # 模拟从客厅到厨房
            ["right", 45],  # 转向可能的厨房方向
            [2.5, 1.0],     # 移动到厨房区域
            ["left", 90],   # 环顾厨房
            ["left", 90],
            ["left", 90],
            ["left", 90],
            
            # 模拟从厨房到卧室
            ["right", 135], # 转向卧室方向
            [1.0, 3.0],     # 移动到卧室
            ["left", 180],  # 转身关门
            ["right", 180], # 再转回来
            
            # 模拟从卧室到浴室
            ["left", 90],   # 转向浴室
            [1.5, 0.0],     # 移动到浴室
            ["right", 90],  # 进入浴室后转向
            
            # 返回起始区域
            ["left", 180],  # 转身
            [1.5, 0.0],     # 出浴室
            ["left", 90],   # 转向
            [1.0, -3.0],    # 回到中央
            ["right", 45],  # 调整朝向
            [2.5, -1.0],    # 回到起始附近
        ]
        
        video_path = generator.process_command_sequence(room_to_room_commands)
        if video_path:
            print(f"房间间移动视频已保存: {video_path}")
        
        # 测试5: 搜索行为模拟
        print("\n--- 测试5: 搜索行为模拟 ---")
        search_behavior_commands = [
            # 系统性搜索模式
            [1.0, 0.0],     # 前进
            ["right", 90],  # 右转检查
            ["left", 180],  # 左转检查另一侧
            ["right", 90],  # 回到前方
            
            [1.0, 0.0],     # 继续前进
            ["left", 90],   # 左转检查
            ["right", 180], # 右转检查另一侧
            ["left", 90],   # 回到前方
            
            # 角落搜索
            ["left", 45],   # 转向角落
            [0.7, 0.7],     # 移动到角落
            ["right", 90],  # 检查角落
            ["right", 90],
            ["right", 90],
            ["right", 90],  # 360度检查
            
            # 退出角落并搜索中央区域
            ["left", 135],  # 转向中央
            [1.0, 1.0],     # 移动到中央
            ["left", 60],   # 六边形搜索模式
            ["left", 60],
            ["left", 60],
            ["left", 60],
            ["left", 60],
            ["left", 60],   # 完成六边形
        ]
        
        video_path = generator.process_command_sequence(search_behavior_commands)
        if video_path:
            print(f"搜索行为模拟视频已保存: {video_path}")
        
        print(f"\n所有测试完成！输出目录: {output_dir}")
        print(f"最终位置: {generator.get_agent_position()}")
        print(f"最终旋转: {generator.get_agent_rotation()}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'generator' in locals():
            generator.close()


def test_apartment_systematic_coverage():
    """公寓系统性覆盖测试 - 尝试覆盖整个公寓空间"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("\n=== 公寓系统性覆盖测试 ===")
    
    output_dir = "./outputs/apartment_coverage"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir=output_dir
        )
        
        print(f"初始位置: {generator.get_agent_position()}")
        
        # 网格式系统覆盖
        grid_coverage_commands = [
            # 第一行扫描
            [4.0, 0.0],     # 向前4米
            ["right", 90],  # 右转
            [0.0, 1.0],     # 向右1米
            ["right", 90],  # 右转（现在面向后方）
            [4.0, 0.0],     # 向后4米（相对于新朝向是前进）
            ["left", 90],   # 左转
            [0.0, 1.0],     # 向右1米
            ["left", 90],   # 左转（现在又面向前方）
            
            # 第二行扫描
            [4.0, 0.0],     # 向前4米
            ["right", 90],  # 右转
            [0.0, 1.0],     # 向右1米
            ["right", 90],  # 右转
            [4.0, 0.0],     # 向后4米
            ["left", 90],   # 左转
            [0.0, 1.0],     # 向右1米
            ["left", 90],   # 左转
            
            # 对角线覆盖
            ["right", 45],  # 转向对角线
            [2.8, 2.8],     # 对角线移动
            ["left", 90],   # 调整方向
            [2.8, -2.8],    # 反向对角线
            ["right", 45],  # 回到标准方向
            
            # 螺旋覆盖
            [1.0, 0.0],
            ["right", 90],
            [0.0, 1.0],
            ["right", 90],
            [2.0, 0.0],
            ["right", 90],
            [0.0, 2.0],
            ["right", 90],
            [3.0, 0.0],
            ["right", 90],
            [0.0, 3.0],
        ]
        
        video_path = generator.process_command_sequence(grid_coverage_commands)
        if video_path:
            print(f"系统性覆盖视频已保存: {video_path}")
        
    except Exception as e:
        print(f"覆盖测试中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'generator' in locals():
            generator.close()


def main():
    """主测试函数"""
    print("开始公寓复杂导航测试...")
    start_time = time.time()
    
    # 运行主要的探索测试
    test_apartment_exploration()
    
    # 运行系统性覆盖测试
    test_apartment_systematic_coverage()
    
    end_time = time.time()
    print(f"\n所有测试完成！总耗时: {end_time - start_time:.2f}秒")
    print("请检查 ./outputs/ 目录中的视频文件")


if __name__ == "__main__":
    main()
