#!/usr/bin/env python3
"""
脚本用于从output文件夹中提取evaluation_results并计算平均SR和SPL
目录结构: output/{scene_name}/{episode_id}/output.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def extract_evaluation_results(output_dir: str, max_episodes: int = None) -> List[Dict]:
    """
    从output目录中提取所有的evaluation_results
    
    Args:
        output_dir: output文件夹的路径
        max_episodes: 最大提取的episode数量，None表示提取所有
        
    Returns:
        包含所有evaluation_results的列表
    """
    results = []
    episode_count = 0
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"错误: 目录 {output_dir} 不存在")
        return results
    
    # 遍历场景文件夹
    for scene_dir in output_path.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        print(f"处理场景: {scene_name}")
        
        # 遍历episode文件夹
        for episode_dir in scene_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            
            # 检查是否达到最大episode数量限制
            if max_episodes is not None and episode_count >= max_episodes:
                print(f"  已达到最大episode数量限制 ({max_episodes})，停止处理")
                return results
                
            episode_id = episode_dir.name
            output_json_path = episode_dir / "output.json"
            
            if not output_json_path.exists():
                print(f"  警告: {output_json_path} 不存在，跳过")
                continue
            
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'evaluation_results' in data:
                    eval_results = data['evaluation_results'].copy()
                    eval_results['scene_name'] = scene_name
                    eval_results['episode_id'] = episode_id
                    results.append(eval_results)
                    episode_count += 1
                    print(f"  提取episode {episode_id}: SR={eval_results.get('sr', 'N/A')}, SPL={eval_results.get('spl', 'N/A')}")
                else:
                    print(f"  警告: {output_json_path} 中没有evaluation_results字段")
                    
            except json.JSONDecodeError as e:
                print(f"  错误: 无法解析 {output_json_path}: {e}")
            except Exception as e:
                print(f"  错误: 处理 {output_json_path} 时发生错误: {e}")
    
    return results


def calculate_metrics(results: List[Dict]) -> Tuple[float, float, Dict]:
    """
    计算平均SR和SPL
    
    Args:
        results: evaluation_results列表
        
    Returns:
        (平均SR, 平均SPL, 统计信息)
    """
    if not results:
        return 0.0, 0.0, {}
    
    # 提取SR和SPL值
    sr_values = []
    spl_values = []
    success_count = 0
    
    for result in results:
        # SR (Success Rate) - 通常是布尔值，需要转换为数值
        sr = result.get('sr', False)
        if isinstance(sr, bool):
            sr_values.append(1.0 if sr else 0.0)
        else:
            sr_values.append(float(sr))
        
        # SPL (Success weighted by Path Length)
        spl = result.get('spl', 0.0)
        spl_values.append(float(spl))
        
        # 统计成功次数
        success = result.get('success', False)
        if success:
            success_count += 1
    
    # 计算平均值
    avg_sr = sum(sr_values) / len(sr_values) if sr_values else 0.0
    avg_spl = sum(spl_values) / len(spl_values) if spl_values else 0.0
    
    # 统计信息
    stats = {
        'total_episodes': len(results),
        'successful_episodes': success_count,
        'success_rate': success_count / len(results) if results else 0.0,
        'avg_sr': avg_sr,
        'avg_spl': avg_spl,
        'min_spl': min(spl_values) if spl_values else 0.0,
        'max_spl': max(spl_values) if spl_values else 0.0
    }
    
    return avg_sr, avg_spl, stats


def save_results(results: List[Dict], stats: Dict, output_file: str = None):
    """
    保存详细结果到JSON文件
    
    Args:
        results: evaluation_results列表
        stats: 统计信息
        output_file: 输出文件路径
    """
    if output_file:
        output_data = {
            'statistics': stats,
            'detailed_results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='提取evaluation_results并计算平均SR和SPL')
    parser.add_argument('--output_dir', '-d', 
                       default="/home/yaoaa/habitat-lab/habitat_video_project/onestage_eval/output",
                       help='output文件夹路径 (默认: %(default)s)')
    parser.add_argument('--save', '-s', 
                       help='保存详细结果到指定的JSON文件')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--max_episodes', '-k', type=int, default=None,
                       help='只处理前k个episode (默认处理所有)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("提取Evaluation Results并计算平均SR和SPL")
    print("=" * 60)
    
    # 提取evaluation results
    print(f"\n从目录提取数据: {args.output_dir}")
    if args.max_episodes:
        print(f"限制处理前 {args.max_episodes} 个episode")
    results = extract_evaluation_results(args.output_dir, args.max_episodes)
    
    if not results:
        print("没有找到任何evaluation_results数据")
        return
    
    # 计算指标
    avg_sr, avg_spl, stats = calculate_metrics(results)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总episode数量: {stats['total_episodes']}")
    print(f"成功episode数量: {stats['successful_episodes']}")
    print(f"成功率: {stats['success_rate']:.4f} ({stats['success_rate']*100:.2f}%)")
    print(f"平均SR (Success Rate): {avg_sr:.4f}")
    print(f"平均SPL (Success weighted by Path Length): {avg_spl:.4f}")
    print(f"SPL范围: {stats['min_spl']:.4f} - {stats['max_spl']:.4f}")
    
    if args.verbose:
        print("\n" + "=" * 60)
        print("详细结果")
        print("=" * 60)
        for result in results:
            scene = result['scene_name']
            episode = result['episode_id']
            sr = result.get('sr', 'N/A')
            spl = result.get('spl', 'N/A')
            success = result.get('success', 'N/A')
            print(f"{scene}/{episode}: Success={success}, SR={sr}, SPL={spl}")
    
    # 保存结果
    if args.save:
        save_results(results, stats, args.save)
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()

