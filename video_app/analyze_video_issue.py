#!/usr/bin/env python3
"""
分析为什么只生成一帧视频的原因
"""

import os
import json
import cv2

def analyze_video_generation():
    """分析视频生成问题"""
    print("=== 分析视频生成问题 ===")
    
    # 分析原始有问题的测试
    print("\n1. 原始测试 (two_agents_dynamic_test):")
    original_dir = "outputs/two_agents_dynamic_test"
    if os.path.exists(original_dir):
        for filename in os.listdir(original_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(original_dir, filename)
                file_size = os.path.getsize(file_path)
                
                # 使用OpenCV分析视频
                cap = cv2.VideoCapture(file_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                print(f"   {filename}: {file_size} bytes, {frame_count} 帧, {fps:.1f} FPS, {duration:.2f}s")
    
    # 分析改进后的测试
    print("\n2. 改进后测试 (safe_two_agents_test):")
    safe_dir = "outputs/safe_two_agents_test"
    if os.path.exists(safe_dir):
        for filename in os.listdir(safe_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(safe_dir, filename)
                file_size = os.path.getsize(file_path)
                
                # 使用OpenCV分析视频
                cap = cv2.VideoCapture(file_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                print(f"   {filename}: {file_size} bytes, {frame_count} 帧, {fps:.1f} FPS, {duration:.2f}s")
    
    # 分析日志文件
    print("\n3. 日志文件对比:")
    
    # 原始测试日志
    original_log = os.path.join(original_dir, "navigation.log")
    if os.path.exists(original_log):
        with open(original_log, 'r') as f:
            lines = f.readlines()
        print(f"   原始测试日志: {len(lines)} 行")
        
        # 查找关键信息
        steps_executed = 0
        collision_warnings = []
        for line in lines:
            if "Executing step" in line:
                steps_executed += 1
            if "Collision predicted" in line:
                collision_warnings.append(line.strip())
        
        print(f"   - 执行步骤数: {steps_executed}")
        print(f"   - 碰撞警告: {len(collision_warnings)}")
        for warning in collision_warnings:
            print(f"     {warning}")
    
    # 改进后测试日志
    safe_log = os.path.join(safe_dir, "navigation.log")
    if os.path.exists(safe_log):
        with open(safe_log, 'r') as f:
            lines = f.readlines()
        print(f"   改进后测试日志: {len(lines)} 行")
        
        # 查找关键信息
        steps_executed = 0
        collision_warnings = []
        for line in lines:
            if "Executing step" in line:
                steps_executed += 1
            if "Collision predicted" in line:
                collision_warnings.append(line.strip())
        
        print(f"   - 执行步骤数: {steps_executed}")
        print(f"   - 碰撞警告: {len(collision_warnings)}")
        for warning in collision_warnings:
            print(f"     {warning}")
    
    print("\n=== 总结 ===")
    print("造成只生成一帧视频的原因:")
    print("1. 初始位置过于接近 - 两个代理开始时就在碰撞检测范围内")
    print("2. 动作序列设计不当 - 代理移动路径会直接相撞")
    print("3. 碰撞检测过于敏感 - 在早期步骤就预测到未来碰撞")
    print("4. 执行步骤过少 - 只执行了1-2步就停止，无法生成足够的帧数")
    print("\n解决方案:")
    print("1. 增加初始位置间距离 - 确保代理开始时有足够安全距离")
    print("2. 优化动作序列 - 让代理朝不同方向移动，避免直接相撞")
    print("3. 调整碰撞检测参数 - 适当降低检测敏感度")
    print("4. 增加执行步骤 - 确保有足够的动作来生成多帧视频")

if __name__ == "__main__":
    analyze_video_generation()
