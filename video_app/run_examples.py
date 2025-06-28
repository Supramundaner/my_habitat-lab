#!/usr/bin/env python3
"""
使用示例脚本 - 展示video_app的各种功能
"""

import subprocess
import os
import time


def run_command(description, commands_json):
    """运行一个命令序列"""
    print(f"\n{'='*60}")
    print(f"示例: {description}")
    print(f"命令: {commands_json}")
    print('='*60)
    
    input_data = f"{commands_json}\nexit\n"
    
    try:
        result = subprocess.run(
            ['python', 'main.py'],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=120,
            cwd='/home/yaoaa/habitat-lab/video_app'
        )
        
        # 提取关键信息
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Processing' in line or 'Executing command' in line:
                print(line)
            elif 'Generated' in line and 'frames' in line:
                print(line)
            elif 'Video successfully saved' in line:
                print(line)
            elif 'ERROR:' in line:
                print(line)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("错误: 命令执行超时")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    """运行使用示例"""
    print("Habitat Video Generator - 使用示例")
    print("此脚本将演示各种功能，每个示例都会生成一个视频文件")
    
    examples = [
        ("基础移动", '[[2.6, 0.1]]'),
        ("旋转演示", '[["right", 45]]'),
        ("移动+旋转组合", '[["right", 30], [2.5, 0.4]]'),
        ("复杂导航序列", '[["left", 90], [3.0, 0.5], ["right", 180], [1.5, -0.5]]'),
        ("错误处理演示（不可导航区域）", '[["left", 45], [10.0, 10.0]]'),
    ]
    
    success_count = 0
    
    for description, commands in examples:
        success = run_command(description, commands)
        if success:
            success_count += 1
        
        # 等待一秒避免文件名冲突
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"示例完成: {success_count}/{len(examples)} 成功")
    print("所有生成的视频文件位于: ./outputs/")
    print('='*60)


if __name__ == "__main__":
    main()
