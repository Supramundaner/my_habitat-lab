#!/usr/bin/env python3
"""
测试脚本 - 验证video_app的基本功能
"""

import sys
import os
import json

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator


def test_video_generator():
    """测试视频生成器"""
    
    # 场景文件路径
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    
    if not os.path.exists(scene_path):
        print(f"Scene file not found: {scene_path}")
        return False
    
    try:
        # 初始化生成器
        generator = HabitatVideoGenerator(
            scene_filepath=scene_path,
            gpu_device_id=0,
            fps=30,
            output_dir="./test_outputs"
        )
        
        print("Generator initialized successfully")
        
        # 测试简单的指令序列
        test_commands = [
            [0.0, 0.0],        # 移动到原点附近
            ["right", 45],     # 右转45度
            [1.0, 1.0],        # 移动到(1,1)
            ["left", 90],      # 左转90度
        ]
        
        print(f"Testing command sequence: {test_commands}")
        
        # 处理指令
        output_path = generator.process_command_sequence(test_commands)
        
        if output_path:
            print(f"Test video saved to: {output_path}")
            return True
        else:
            print("No video generated")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        try:
            generator.close()
        except:
            pass


if __name__ == '__main__':
    print("Testing Habitat Video Generator...")
    
    # 创建测试输出目录
    os.makedirs("./test_outputs", exist_ok=True)
    
    success = test_video_generator()
    
    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
        sys.exit(1)
