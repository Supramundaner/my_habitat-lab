#!/usr/bin/env python3
"""
Split Controversial Episodes Script

This script splits a controversial episodes JSON file into smaller chunks
for parallel processing.

Usage:
    python split_episodes.py input_file.json num_chunks

Example:
    python split_episodes.py controversial_episodes.json 4
"""

import json
import sys
import argparse
from pathlib import Path


def split_episodes(input_file: str, num_chunks: int, output_prefix: str = "controversial_episodes_chunk"):
    """
    将 controversial episodes 文件分割成多个小文件
    
    Args:
        input_file: 输入的 controversial episodes JSON 文件路径
        num_chunks: 要分割成的块数
        output_prefix: 输出文件的前缀
    """
    
    # 读取输入文件
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败 - {e}")
        return False
    
    # 提取 episodes 列表
    if isinstance(data, list):
        episodes = data
        original_stats = {"total_episodes": len(episodes)}
    elif isinstance(data, dict) and 'controversial_episodes' in data:
        episodes = data['controversial_episodes']
        original_stats = data.get('statistics', {"total_episodes": len(episodes)})
    else:
        print("错误: 不支持的文件格式。期望包含 'controversial_episodes' 字段的对象或直接的 episodes 数组")
        return False
    
    if not episodes:
        print("错误: 没有找到任何 episodes")
        return False
    
    # 计算每个块的大小
    total_episodes = len(episodes)
    chunk_size = total_episodes // num_chunks
    remainder = total_episodes % num_chunks
    
    print(f"分割 {total_episodes} 个 episodes 到 {num_chunks} 个文件中...")
    print(f"基础块大小: {chunk_size}")
    if remainder > 0:
        print(f"前 {remainder} 个块将包含额外的 1 个 episode")
    
    created_files = []
    
    for i in range(num_chunks):
        # 计算当前块的起始和结束索引
        start_idx = i * chunk_size + min(i, remainder)
        if i < remainder:
            current_chunk_size = chunk_size + 1
        else:
            current_chunk_size = chunk_size
        end_idx = start_idx + current_chunk_size
        
        # 提取当前块的 episodes
        chunk_episodes = episodes[start_idx:end_idx]
        
        # 创建块数据结构
        chunk_data = {
            "controversial_episodes": chunk_episodes,
            "statistics": {
                "total_episodes": len(chunk_episodes),
                "chunk_id": i + 1,
                "total_chunks": num_chunks,
                "original_total": total_episodes,
                "chunk_range": f"{start_idx+1}-{end_idx}"
            }
        }
        
        # 如果原始文件有额外的统计信息，也复制过来
        if 'statistics' in data and isinstance(data['statistics'], dict):
            chunk_data['statistics'].update({
                "original_controversy_rate": data['statistics'].get('controversy_rate'),
                "original_total_episodes": data['statistics'].get('total_episodes')
            })
        
        # 保存块文件
        output_file = f"{output_prefix}_{i+1}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(chunk_data, f, indent=2)
            
            created_files.append(output_file)
            print(f"✓ 创建 {output_file}: {len(chunk_episodes)} episodes (范围: {start_idx+1}-{end_idx})")
            
        except Exception as e:
            print(f"错误: 无法创建文件 {output_file} - {e}")
            return False
    
    # 创建摘要信息
    summary = {
        "split_info": {
            "original_file": input_file,
            "total_episodes": total_episodes,
            "num_chunks": num_chunks,
            "created_files": created_files,
            "split_timestamp": None  # 可以添加时间戳
        },
        "chunks": []
    }
    
    # 验证分割结果
    total_split_episodes = 0
    for i, file_path in enumerate(created_files, 1):
        try:
            with open(file_path, 'r') as f:
                chunk_data = json.load(f)
            chunk_episodes = len(chunk_data['controversial_episodes'])
            total_split_episodes += chunk_episodes
            
            summary['chunks'].append({
                "chunk_id": i,
                "file": file_path,
                "episodes": chunk_episodes,
                "range": chunk_data['statistics']['chunk_range']
            })
        except Exception as e:
            print(f"警告: 无法验证文件 {file_path} - {e}")
    
    # 保存分割摘要
    summary_file = f"{output_prefix}_summary.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ 创建分割摘要: {summary_file}")
    except Exception as e:
        print(f"警告: 无法创建摘要文件 - {e}")
    
    # 验证总数
    if total_split_episodes == total_episodes:
        print(f"\n✅ 分割成功完成!")
        print(f"   原始 episodes: {total_episodes}")
        print(f"   分割后总计: {total_split_episodes}")
        print(f"   创建的文件: {len(created_files)}")
        return True
    else:
        print(f"\n❌ 分割验证失败!")
        print(f"   原始 episodes: {total_episodes}")
        print(f"   分割后总计: {total_split_episodes}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Split controversial episodes JSON file into chunks for parallel processing")
    parser.add_argument("input_file", help="Input controversial episodes JSON file")
    parser.add_argument("num_chunks", type=int, help="Number of chunks to create")
    parser.add_argument("--output_prefix", default="controversial_episodes_chunk", 
                       help="Prefix for output files (default: controversial_episodes_chunk)")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify chunks after creation")
    
    args = parser.parse_args()
    
    # 验证输入参数
    if args.num_chunks <= 0:
        print("错误: chunk 数量必须大于 0")
        return 1
    
    if not Path(args.input_file).exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        return 1
    
    # 执行分割
    success = split_episodes(args.input_file, args.num_chunks, args.output_prefix)
    
    if success:
        print(f"\n使用方法:")
        print(f"1. 为每个块配置单独的输出目录")
        print(f"2. 使用并行处理运行 discriminator:")
        for i in range(1, args.num_chunks + 1):
            chunk_file = f"{args.output_prefix}_{i}.json"
            print(f"   python discriminator_system.py --config_path config.json --controversial_episodes {chunk_file}")
        print(f"3. 处理完成后合并结果")
        return 0
    else:
        print("分割失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
