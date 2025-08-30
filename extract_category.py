#!/usr/bin/env python3
"""
脚本用于从output文件夹中提取evaluation_results并计算平均SR和SPL
目录结构: output/{scene_name}/{episode_id}/output.json
支持按照数据集split（val_seen, val_seen_synonyms, val_unseen）分类统计
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set


def load_batch_episodes(batch_episodes_path: str) -> Dict[str, Dict[str, str]]:
    """
    从batch_episodes.json文件中加载episode到split的映射
    
    Args:
        batch_episodes_path: batch_episodes.json文件路径
        
    Returns:
        字典，key为"scene_name/episode_id"，value为包含split信息的字典
    """
    episode_to_split = {}
    
    if not os.path.exists(batch_episodes_path):
        print(f"警告: {batch_episodes_path} 不存在，将不进行split分类")
        return episode_to_split
    
    try:
        with open(batch_episodes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for task in data.get('evaluation_tasks', []):
            episode_json_path = task.get('episode_json_path', '')
            episode_ids = task.get('episode_ids', [])
            
            # 从路径中提取split信息
            split = None
            if '/val_seen_synonyms/' in episode_json_path:
                split = 'val_seen_synonyms'
            elif '/val_seen/' in episode_json_path:
                split = 'val_seen'
            elif '/val_unseen/' in episode_json_path:
                split = 'val_unseen'
            else:
                split = 'unknown'
            
            # 从路径中提取scene name
            scene_name = os.path.basename(episode_json_path).replace('.json', '')
            
            # 为每个episode_id创建映射
            for episode_id in episode_ids:
                key = f"{scene_name}/{episode_id}"
                episode_to_split[key] = {
                    'split': split,
                    'scene_name': scene_name,
                    'episode_id': str(episode_id)
                }
                
        print(f"成功加载 {len(episode_to_split)} 个episode的split信息")
        
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 {batch_episodes_path}: {e}")
    except Exception as e:
        print(f"错误: 处理 {batch_episodes_path} 时发生错误: {e}")
    
    return episode_to_split


def extract_evaluation_results(output_dir: str, episode_to_split: Dict[str, Dict[str, str]] = None) -> Tuple[List[Dict], List[str]]:
    """
    从output目录中提取所有的evaluation_results
    
    Args:
        output_dir: output文件夹的路径
        episode_to_split: episode到split的映射字典
        
    Returns:
        (包含所有evaluation_results的列表，缺失的episode列表)
    """
    results = []
    missing_episodes = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"错误: 目录 {output_dir} 不存在")
        return results, missing_episodes
    
    if episode_to_split is None:
        episode_to_split = {}
    
    # 创建一个集合来跟踪已找到的episodes
    found_episodes = set()
    
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
                
            episode_id = episode_dir.name
            episode_key = f"{scene_name}/{episode_id}"
            output_json_path = episode_dir / "output.json"
            
            if not output_json_path.exists():
                print(f"  警告: {output_json_path} 不存在，跳过")
                missing_episodes.append(f"{episode_key} (output.json不存在)")
                continue
            
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'evaluation_results' in data:
                    eval_results = data['evaluation_results'].copy()
                    eval_results['scene_name'] = scene_name
                    eval_results['episode_id'] = episode_id
                    
                    # 添加split信息
                    if episode_key in episode_to_split:
                        eval_results['split'] = episode_to_split[episode_key]['split']
                    else:
                        eval_results['split'] = 'unknown'
                        print(f"  警告: episode {episode_key} 未在batch_episodes.json中找到，标记为unknown")
                    
                    results.append(eval_results)
                    found_episodes.add(episode_key)
                    print(f"  提取episode {episode_id} [{eval_results['split']}]: SR={eval_results.get('sr', 'N/A')}, SPL={eval_results.get('spl', 'N/A')}")
                else:
                    print(f"  警告: {output_json_path} 中没有evaluation_results字段")
                    missing_episodes.append(f"{episode_key} (缺少evaluation_results字段)")
                    
            except json.JSONDecodeError as e:
                print(f"  错误: 无法解析 {output_json_path}: {e}")
                missing_episodes.append(f"{episode_key} (JSON解析错误)")
            except Exception as e:
                print(f"  错误: 处理 {output_json_path} 时发生错误: {e}")
                missing_episodes.append(f"{episode_key} (处理错误: {str(e)})")
    
    # 检查batch_episodes.json中定义但在output中没有找到的episodes
    if episode_to_split:
        for expected_episode in episode_to_split.keys():
            if expected_episode not in found_episodes:
                missing_episodes.append(f"{expected_episode} (在output目录中完全缺失)")
    
    return results, missing_episodes


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


def calculate_metrics_by_split(results: List[Dict]) -> Dict[str, Dict]:
    """
    按照split分类计算指标
    
    Args:
        results: evaluation_results列表，每个结果包含split信息
        
    Returns:
        字典，key为split名称，value为该split的统计信息
    """
    # 按split分组
    results_by_split = {}
    for result in results:
        split = result.get('split', 'unknown')
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)
    
    # 为每个split计算指标
    metrics_by_split = {}
    for split, split_results in results_by_split.items():
        avg_sr, avg_spl, stats = calculate_metrics(split_results)
        metrics_by_split[split] = {
            'avg_sr': avg_sr,
            'avg_spl': avg_spl,
            'stats': stats
        }
    
    return metrics_by_split


def save_results(results: List[Dict], stats: Dict, split_stats: Dict = None, output_file: str = None):
    """
    保存详细结果到JSON文件
    
    Args:
        results: evaluation_results列表
        stats: 总体统计信息
        split_stats: 按split分类的统计信息
        output_file: 输出文件路径
    """
    if output_file:
        output_data = {
            'overall_statistics': stats,
            'detailed_results': results
        }
        
        if split_stats:
            output_data['statistics_by_split'] = split_stats
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='提取evaluation_results并计算平均SR和SPL，支持按split分类统计')
    parser.add_argument('--output_dir', '-d', 
                       default='/home/yaoaa/habitat-lab/habitat_video_project/eval/output_cons6',
                       help='output文件夹路径 (默认: /home/yaoaa/habitat-lab/habitat_video_project/eval/output_cons6)')
    parser.add_argument('--batch_episodes', '-b',
                       default='/home/yaoaa/habitat-lab/habitat_video_project/eval/batch_episodes.json',
                       help='batch_episodes.json文件路径，用于split分类 (默认: /home/yaoaa/habitat-lab/habitat_video_project/eval/batch_episodes.json)')
    parser.add_argument('--save', '-s', 
                       help='保存详细结果到指定的JSON文件')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--split_only', action='store_true',
                       help='只显示按split分类的结果')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("提取Evaluation Results并计算平均SR和SPL (支持split分类)")
    print("=" * 80)
    
    # 加载batch episodes配置
    print(f"\n加载batch episodes配置: {args.batch_episodes}")
    episode_to_split = load_batch_episodes(args.batch_episodes)
    
    # 提取evaluation results
    print(f"\n从目录提取数据: {args.output_dir}")
    results, missing_episodes = extract_evaluation_results(args.output_dir, episode_to_split)
    
    if not results:
        print("没有找到任何evaluation_results数据")
        return
    
    # 报告缺失的episodes
    if missing_episodes:
        print(f"\n" + "⚠️ " * 30)
        print(f"发现 {len(missing_episodes)} 个缺失或有问题的episodes:")
        print("⚠️ " * 30)
        for missing in missing_episodes:
            print(f"  - {missing}")
        print(f"\n预期总数: {len(episode_to_split) if episode_to_split else 'N/A'}")
        print(f"实际提取: {len(results)}")
        print(f"缺失数量: {len(missing_episodes)}")
    else:
        print(f"\n✅ 所有episodes都已成功提取!")
    
    # 计算总体指标
    avg_sr, avg_spl, stats = calculate_metrics(results)
    
    # 计算按split分类的指标
    split_metrics = calculate_metrics_by_split(results)
    
    if not args.split_only:
        # 输出总体结果
        print("\n" + "=" * 80)
        print("总体统计结果")
        print("=" * 80)
        print(f"总episode数量: {stats['total_episodes']}")
        print(f"成功episode数量: {stats['successful_episodes']}")
        print(f"成功率: {stats['success_rate']:.4f} ({stats['success_rate']*100:.2f}%)")
        print(f"平均SR (Success Rate): {avg_sr:.4f}")
        print(f"平均SPL (Success weighted by Path Length): {avg_spl:.4f}")
        print(f"SPL范围: {stats['min_spl']:.4f} - {stats['max_spl']:.4f}")
    
    # 输出按split分类的结果
    print("\n" + "=" * 80)
    print("按Split分类的统计结果")
    print("=" * 80)
    
    for split, metrics in split_metrics.items():
        split_stats = metrics['stats']
        print(f"\n【{split.upper()}】")
        print(f"  Episode数量: {split_stats['total_episodes']}")
        print(f"  成功数量: {split_stats['successful_episodes']}")
        print(f"  成功率: {split_stats['success_rate']:.4f} ({split_stats['success_rate']*100:.2f}%)")
        print(f"  平均SR: {metrics['avg_sr']:.4f}")
        print(f"  平均SPL: {metrics['avg_spl']:.4f}")
        print(f"  SPL范围: {split_stats['min_spl']:.4f} - {split_stats['max_spl']:.4f}")
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("详细结果")
        print("=" * 80)
        for result in results:
            scene = result['scene_name']
            episode = result['episode_id']
            split = result.get('split', 'unknown')
            sr = result.get('sr', 'N/A')
            spl = result.get('spl', 'N/A')
            success = result.get('success', 'N/A')
            print(f"[{split}] {scene}/{episode}: Success={success}, SR={sr}, SPL={spl}")
    
    # 保存结果
    if args.save:
        # 在保存的数据中包含缺失episodes信息
        save_data = {
            'overall_statistics': stats,
            'detailed_results': results,
            'missing_episodes': missing_episodes,
            'summary': {
                'expected_total': len(episode_to_split) if episode_to_split else None,
                'actual_extracted': len(results),
                'missing_count': len(missing_episodes)
            }
        }
        
        if split_metrics:
            save_data['statistics_by_split'] = split_metrics
        
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {args.save}")
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()

