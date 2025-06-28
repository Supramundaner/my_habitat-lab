#!/usr/bin/env python3
"""
Apartment复杂导航测试 - 高级导航模式展示
包含多种现实导航行为模拟
"""

import sys
import os
import numpy as np
import time
import math

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator


def test_apartment_advanced_navigation():
    """高级导航模式测试"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("=== Apartment高级导航测试 ===")
    print(f"场景: apartment_1.glb")
    
    output_dir = "./outputs/apartment_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir=output_dir
        )
        
        print(f"初始位置: {generator.get_agent_position()}")
        print(f"场景边界: {generator.simulator.scene_bounds}")
        
        # 计算场景的关键位置
        bounds = generator.simulator.scene_bounds
        center_x = (bounds[0][0] + bounds[1][0]) / 2
        center_z = (bounds[0][2] + bounds[1][2]) / 2
        width = bounds[1][0] - bounds[0][0]
        depth = bounds[1][2] - bounds[0][2]
        
        print(f"场景中心: ({center_x:.2f}, {center_z:.2f})")
        print(f"场景尺寸: {width:.2f} x {depth:.2f} 米")
        
        # 测试1: 螺旋式房间探索
        print("\n--- 测试1: 螺旋式房间探索 ---")
        spiral_exploration = [
            # 开始螺旋探索
            [1.0, 0.0],     # 前进1米
            ["right", 90],  # 右转
            [0.0, 1.0],     # 右移1米
            ["right", 90],  # 右转
            [2.0, 0.0],     # 后退2米
            ["right", 90],  # 右转
            [0.0, 2.0],     # 左移2米
            ["right", 90],  # 右转
            [3.0, 0.0],     # 前进3米
            ["right", 90],  # 右转
            [0.0, 3.0],     # 右移3米
        ]
        
        video_path = generator.process_command_sequence(spiral_exploration)
        if video_path:
            print(f"螺旋探索视频已保存: {video_path}")
        
        # 测试2: 墙面巡查模式
        print("\n--- 测试2: 墙面巡查模式 ---")
        wall_patrol = [
            # 沿墙边移动
            ["left", 45],   # 朝向墙角
            [0.5, 0.5],     # 移动到墙角
            ["right", 45],  # 面向墙面
            [1.5, 0.0],     # 沿墙前进
            ["right", 90],  # 转弯
            [0.0, 1.5],     # 沿墙移动
            ["right", 90],  # 转弯
            [1.5, 0.0],     # 沿墙移动
            ["right", 90],  # 转弯
            [0.0, 1.5],     # 完成一圈
            ["right", 90],  # 面向内部
        ]
        
        video_path = generator.process_command_sequence(wall_patrol)
        if video_path:
            print(f"墙面巡查视频已保存: {video_path}")
        
        # 测试3: 房间对角线穿越
        print("\n--- 测试3: 对角线穿越 ---")
        diagonal_traversal = [
            # 记录当前位置作为起点
            ["left", 45],   # 朝向对角线
            [2.0, 2.0],     # 对角线移动
            ["left", 90],   # 转向观察
            ["right", 180], # 转向另一边
            ["left", 90],   # 回到对角线方向
            [1.5, 1.5],     # 继续对角线
            ["right", 135], # 转向反对角线
            [2.0, -2.0],    # 反对角线移动
            ["left", 45],   # 调整方向
        ]
        
        video_path = generator.process_command_sequence(diagonal_traversal)
        if video_path:
            print(f"对角线穿越视频已保存: {video_path}")
        
        # 测试4: 多点巡回路径
        print("\n--- 测试4: 多点巡回路径 ---")
        multi_point_patrol = [
            # 巡回多个兴趣点
            [2.0, 0.0],     # 到达点A
            ["left", 180],  # 检查周围
            ["right", 180], # 继续检查
            
            ["right", 90],  # 转向点B
            [0.0, 2.5],     # 到达点B
            ["left", 90],   # 检查周围
            ["right", 180], # 继续检查
            ["left", 90],   # 回到方向
            
            ["left", 90],   # 转向点C
            [2.5, 0.0],     # 到达点C
            ["right", 90],  # 检查周围
            ["left", 180],  # 继续检查
            ["right", 90],  # 回到方向
            
            ["right", 90],  # 转向点D
            [0.0, 2.0],     # 到达点D
            ["left", 180],  # 最终检查
            ["right", 180], # 完成巡回
        ]
        
        video_path = generator.process_command_sequence(multi_point_patrol)
        if video_path:
            print(f"多点巡回视频已保存: {video_path}")
        
        # 测试5: 精确定位和微调
        print("\n--- 测试5: 精确定位和微调 ---")
        precision_positioning = [
            # 精确角度调整
            ["left", 15],   # 小角度调整
            ["right", 30],  # 反向调整
            ["left", 20],   # 精确定位
            ["right", 5],   # 微调
            
            # 精确距离移动
            [0.5, 0.0],     # 短距离前进
            [0.3, 0.0],     # 更短距离
            [0.2, 0.0],     # 精确定位
            
            # 侧向精确移动
            ["right", 90],  # 转向侧方
            [0.0, 0.4],     # 精确侧移
            [0.0, 0.3],     # 更精确
            [0.0, 0.2],     # 最终定位
            
            # 复合精确移动
            ["left", 45],   # 对角线方向
            [0.4, 0.4],     # 对角线精确移动
            ["right", 45],  # 回到正向
        ]
        
        video_path = generator.process_command_sequence(precision_positioning)
        if video_path:
            print(f"精确定位视频已保存: {video_path}")
        
        # 测试6: 复杂路径规划模拟
        print("\n--- 测试6: 复杂路径规划 ---")
        complex_path_planning = [
            # 模拟避障路径
            [1.0, 0.0],     # 前进
            ["left", 30],   # 左转避障
            [0.8, 0.5],     # 斜向移动
            ["right", 60],  # 右转绕过障碍
            [0.5, -0.3],    # 调整位置
            ["left", 30],   # 回到主路径
            [1.2, 0.0],     # 继续前进
            
            # 模拟狭窄通道导航
            ["right", 90],  # 进入通道
            [0.0, 0.8],     # 小心移动
            ["left", 15],   # 微调方向
            [0.0, 0.6],     # 继续通过
            ["right", 15],  # 微调回来
            [0.0, 0.8],     # 通过通道
            ["left", 90],   # 出通道
            
            # 模拟开放空间快速移动
            [2.5, 0.0],     # 快速前进
            ["left", 90],   # 大转弯
            [0.0, 2.0],     # 侧向快移
            ["right", 90],  # 回到方向
        ]
        
        video_path = generator.process_command_sequence(complex_path_planning)
        if video_path:
            print(f"复杂路径规划视频已保存: {video_path}")
        
        print(f"\n高级导航测试完成！")
        print(f"最终位置: {generator.get_agent_position()}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'generator' in locals():
            generator.close()


def test_apartment_behavioral_patterns():
    """行为模式测试 - 模拟真实世界的导航行为"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene file not found: {scene_path}")
        return
    
    print("\n=== 行为模式测试 ===")
    
    output_dir = "./outputs/apartment_behavioral"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir=output_dir
        )
        
        # 模拟真实的探索行为
        print("\n--- 好奇心驱动的探索 ---")
        curiosity_exploration = [
            # 听到声音，转头查看
            ["left", 30],   # 听到左边声音
            ["right", 60],  # 听到右边声音
            ["left", 30],   # 回到前方
            
            # 发现兴趣点，靠近观察
            [1.5, 0.0],     # 靠近观察
            ["left", 45],   # 左侧观察
            ["right", 90],  # 右侧观察
            ["left", 45],   # 回到前方
            
            # 绕圈观察兴趣点
            ["right", 90],  # 开始绕圈
            [0.0, 1.0],     # 侧移
            ["right", 90],  # 继续绕
            [1.0, 0.0],     # 后移
            ["right", 90],  # 继续绕
            [0.0, 1.0],     # 侧移
            ["right", 90],  # 完成绕圈
            
            # 离开继续探索
            [1.0, 0.0],     # 离开兴趣点
        ]
        
        video_path = generator.process_command_sequence(curiosity_exploration)
        if video_path:
            print(f"好奇心探索视频已保存: {video_path}")
        
        # 模拟谨慎的导航行为
        print("\n--- 谨慎导航模式 ---")
        cautious_navigation = [
            # 小步前进，频繁观察
            [0.5, 0.0],     # 小步前进
            ["left", 45],   # 左侧警戒
            ["right", 90],  # 右侧警戒
            ["left", 45],   # 回到前方
            
            [0.5, 0.0],     # 继续小步前进
            ["right", 30],  # 右侧观察
            ["left", 60],   # 左侧观察
            ["right", 30],  # 回到前方
            
            # 发现"障碍"，小心绕行
            ["left", 15],   # 轻微左转
            [0.3, 0.2],     # 小心绕行
            ["right", 30],  # 调整方向
            [0.3, -0.1],    # 继续绕行
            ["left", 15],   # 回到主方向
            
            [0.8, 0.0],     # 安全距离后加速
        ]
        
        video_path = generator.process_command_sequence(cautious_navigation)
        if video_path:
            print(f"谨慎导航视频已保存: {video_path}")
        
        # 模拟目标导向的快速移动
        print("\n--- 目标导向快速移动 ---")
        goal_oriented_movement = [
            # 确定目标方向
            ["left", 90],   # 环顾寻找目标
            ["left", 90],
            ["left", 90],
            ["left", 90],   # 完成环顾
            
            # 快速直线移动到目标
            [2.5, 0.0],     # 快速前进
            
            # 到达目标区域后减速
            [0.5, 0.0],     # 减速靠近
            [0.3, 0.0],     # 精确定位
            
            # 目标达成，新目标搜索
            ["right", 45],  # 寻找新目标
            ["left", 90],   # 环顾
            ["right", 45],  # 确定方向
            
            # 移动到新目标
            ["right", 90],  # 转向新目标
            [0.0, 2.0],     # 侧向移动
            ["left", 90],   # 调整朝向
            [1.0, 0.0],     # 到达新目标
        ]
        
        video_path = generator.process_command_sequence(goal_oriented_movement)
        if video_path:
            print(f"目标导向移动视频已保存: {video_path}")
        
    except Exception as e:
        print(f"行为模式测试中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'generator' in locals():
            generator.close()


def main():
    """主测试函数"""
    print("开始Apartment复杂导航测试...")
    start_time = time.time()
    
    # 运行高级导航测试
    test_apartment_advanced_navigation()
    
    # 运行行为模式测试
    test_apartment_behavioral_patterns()
    
    end_time = time.time()
    print(f"\n所有复杂导航测试完成！总耗时: {end_time - start_time:.2f}秒")
    print("请检查 ./outputs/ 目录中的视频文件")


if __name__ == "__main__":
    main()
