#!/usr/bin/env python3
"""
Habitat Video Generator - 离线视频生成应用
接收JSON格式的导航指令序列，生成展示代理运动过程的MP4视频
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Habitat Video Generator')
    parser.add_argument('--scene', 
                       default="/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
                       help='Path to the .glb scene file')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='CUDA device ID (default: 0)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate (default: 30)')
    parser.add_argument('--output-dir', default='./outputs',
                       help='Output directory for videos (default: ./outputs)')
    return parser.parse_args()


def validate_command_sequence(commands):
    """验证指令序列格式"""
    if not isinstance(commands, list):
        return False, "Command sequence must be a list"
    
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, list):
            return False, f"Command {i} must be a list"
        
        if len(cmd) == 2:
            # 坐标指令 [x, z] 或旋转指令 ["direction", angle]
            if isinstance(cmd[0], str):
                # 旋转指令
                if cmd[0] not in ["left", "right"]:
                    return False, f"Command {i}: Invalid direction '{cmd[0]}', must be 'left' or 'right'"
                if not isinstance(cmd[1], (int, float)):
                    return False, f"Command {i}: Angle must be a number"
                if not (0 < cmd[1] <= 360):
                    return False, f"Command {i}: Angle must be between 0 and 360 degrees"
            elif isinstance(cmd[0], (int, float)):
                # 坐标指令
                if not isinstance(cmd[1], (int, float)):
                    return False, f"Command {i}: Both coordinates must be numbers"
            else:
                return False, f"Command {i}: Invalid format"
        else:
            return False, f"Command {i}: Must have exactly 2 elements"
    
    return True, "Valid"


def main():
    """主函数"""
    args = parse_args()
    
    # 检查场景文件是否存在
    if not os.path.exists(args.scene):
        print(f"ERROR: Scene file not found: {args.scene}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Habitat Video Generator Initialized.")
    print("Enter command sequence as a JSON string or 'exit'.")
    
    # 初始化视频生成器
    try:
        generator = HabitatVideoGenerator(
            scene_filepath=args.scene,
            gpu_device_id=args.gpu,
            fps=args.fps,
            output_dir=str(output_dir)
        )
        print(f"Scene loaded: {args.scene}")
        print(f"GPU device: {args.gpu}")
        print(f"Output directory: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize video generator: {e}")
        sys.exit(1)
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("> ").strip()
            
            # 检查退出命令
            if user_input.lower() == 'exit':
                print("Shutting down.")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 解析JSON指令
            try:
                commands = json.loads(user_input)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON format: {e}")
                continue
            
            # 验证指令格式
            is_valid, error_msg = validate_command_sequence(commands)
            if not is_valid:
                print(f"ERROR: {error_msg}")
                continue
            
            # 处理指令序列
            print(f"Processing {len(commands)} commands...")
            
            try:
                output_path = generator.process_command_sequence(commands)
                if output_path:
                    print(f"Video successfully saved to: {output_path}")
                else:
                    print("No video generated (empty command sequence or all commands failed)")
                    
            except KeyboardInterrupt:
                print("\nOperation interrupted by user.")
                continue
            except Exception as e:
                print(f"ERROR: Failed to process commands: {e}")
                continue
                
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except EOFError:
            print("\nShutting down.")
            break
    
    # 清理资源
    try:
        generator.close()
    except:
        pass


if __name__ == '__main__':
    main()
