#!/usr/bin/env python3
"""
测试楼层过滤脚本的功能
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append('/home/yaoaa/habitat-lab')

from filter_cross_floor_episodes import FloorFilterProcessor


def test_single_scene():
    """测试单个场景的处理"""
    input_dir = Path("data/datasets/objectnav/hm3d/v1/val/content_preprocessed")
    output_dir = Path("data/datasets/objectnav/hm3d/v1/val/content_preprocessed_filtered_test")
    
    # 创建测试输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择一个较小的场景文件进行测试
    test_file = input_dir / "mL8ThkuaVTM.json"  # 这个文件相对较小
    
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return False
    
    print(f"测试文件: {test_file}")
    
    # 创建处理器
    processor = FloorFilterProcessor(str(input_dir), str(output_dir))
    
    # 处理单个文件
    success = processor.process_scene_file(test_file)
    
    if success:
        print("✅ 单场景测试成功!")
        
        # 检查输出文件
        output_file = output_dir / test_file.name
        if output_file.exists():
            with open(output_file, 'r') as f:
                filtered_data = json.load(f)
            
            print(f"输出文件已创建: {output_file}")
            print(f"过滤后episodes数量: {len(filtered_data.get('episodes', []))}")
            
            # 打印统计信息
            processor.print_final_statistics()
            
        return True
    else:
        print("❌ 单场景测试失败!")
        return False


def analyze_scene_structure():
    """分析场景文件的结构"""
    input_dir = Path("data/datasets/objectnav/hm3d/v1/val/content_preprocessed")
    test_file = input_dir / "mL8ThkuaVTM.json"
    
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return
    
    print(f"分析文件结构: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print("\n=== 文件结构分析 ===")
    print(f"主要键: {list(data.keys())}")
    
    if 'goals_by_category' in data:
        goals = data['goals_by_category']
        print(f"目标类别数量: {len(goals)}")
        print("目标类别:", list(goals.keys())[:5], "..." if len(goals) > 5 else "")
        
        # 分析第一个目标类别
        if goals:
            first_category = list(goals.keys())[0]
            first_goals = goals[first_category]
            print(f"第一个类别 '{first_category}' 有 {len(first_goals)} 个目标")
            if first_goals:
                print("第一个目标的位置:", first_goals[0]['position'])
    
    if 'episodes' in data:
        episodes = data['episodes']
        print(f"Episodes数量: {len(episodes)}")
        if episodes:
            print("第一个episode的起始位置:", episodes[0]['start_position'])
            print("第一个episode的目标类别:", episodes[0]['object_category'])
            print("第一个episode的场景ID:", episodes[0]['scene_id'])


if __name__ == "__main__":
    print("开始测试楼层过滤脚本...")
    
    # 首先分析文件结构
    analyze_scene_structure()
    
    print("\n" + "="*50)
    
    # 然后测试处理功能
    test_single_scene()

