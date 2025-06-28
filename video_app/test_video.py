#!/usr/bin/env python3
"""
视频生成测试脚本
"""

import subprocess
import os
import time

def test_video_generation():
    """测试视频生成功能"""
    print("开始测试视频生成...")
    
    # 测试命令序列
    test_commands = [
        # 初始化到场景中心
        '[[2.6, 0.1]]',
        # 旋转和移动
        '[["right", 30], [2.5, 0.4]]',
        # 更复杂的序列
        '[["left", 90], [3.0, 0.8], ["right", 45]]'
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n测试 {i}: {cmd}")
        
        # 准备进程输入
        input_data = f"{cmd}\nexit\n"
        
        try:
            # 运行主程序
            result = subprocess.run(
                ['python', 'main.py'],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=60,  # 60秒超时
                cwd='/home/yaoaa/habitat-lab/video_app'
            )
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            if result.returncode != 0:
                print(f"程序退出码: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("测试超时")
        except Exception as e:
            print(f"测试失败: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_video_generation()
