#!/usr/bin/env python3
"""
脚本用于比较两个不同文件夹中相同index episodes的evaluation results
计算两个文件夹中对应episodes的平均SR和SPL并进行对比
目录结构: output/{scene_name}/{episode_id}/output.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set


def extract_episode_indices(output_dir: str) -> Set[Tuple[str, str]]:
    """
    提取output目录中所有存在的(scene_name, episode_id)组合
    
    Args:
        output_dir: output文件夹的路径
        
    Returns:
        包含所有(scene_name, episode_id)组合的集合
    """
    indices = set()
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"错误: 目录 {output_dir} 不存在")
        return indices
    
    # 遍历场景文件夹
    for scene_dir in output_path.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        
        # 遍历episode文件夹
        for episode_dir in scene_dir.iterdir():
            if not episode_dir.is_dir():
                continue
                
            episode_id = episode_dir.name
            output_json_path = episode_dir / "output.json"
            
            # 只添加存在output.json文件的episode
            if output_json_path.exists():
                indices.add((scene_name, episode_id))
    
    return indices


def extract_specific_evaluation_results(output_dir: str, target_indices: Set[Tuple[str, str]]) -> List[Dict]:
    """
    从output目录中提取指定的evaluation_results
    
    Args:
        output_dir: output文件夹的路径
        target_indices: 目标(scene_name, episode_id)组合集合
        
    Returns:
        包含指定evaluation_results的列表
    """
    results = []
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"错误: 目录 {output_dir} 不存在")
        return results
    
    found_count = 0
    total_target = len(target_indices)
    
    # 遍历场景文件夹
    for scene_dir in output_path.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        
        # 遍历episode文件夹
        for episode_dir in scene_dir.iterdir():
            if not episode_dir.is_dir():
                continue
                
            episode_id = episode_dir.name
            
            # 检查是否是目标episode
            if (scene_name, episode_id) not in target_indices:
                continue
                
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
                    found_count += 1
                    print(f"  提取episode {scene_name}/{episode_id}: SR={eval_results.get('sr', 'N/A')}, SPL={eval_results.get('spl', 'N/A')}")
                else:
                    print(f"  警告: {output_json_path} 中没有evaluation_results字段")
                    
            except json.JSONDecodeError as e:
                print(f"  错误: 无法解析 {output_json_path}: {e}")
            except Exception as e:
                print(f"  错误: 处理 {output_json_path} 时发生错误: {e}")
    
    print(f"  找到 {found_count}/{total_target} 个目标episodes")
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


def analyze_cross_validation(folder1_results: List[Dict], folder2_results: List[Dict]) -> Dict:
    """
    交叉验证分析：找出在一个文件夹成功而另一个失败的episodes
    
    Args:
        folder1_results: 文件夹1的结果
        folder2_results: 文件夹2的结果
        
    Returns:
        包含交叉验证分析结果的字典
    """
    # 创建episode到结果的映射
    folder1_map = {(r['scene_name'], r['episode_id']): r for r in folder1_results}
    folder2_map = {(r['scene_name'], r['episode_id']): r for r in folder2_results}
    
    # 找出共同的episodes
    common_episodes = set(folder1_map.keys()) & set(folder2_map.keys())
    
    # 分类episodes
    f1_success_f2_fail = []  # 文件夹1成功，文件夹2失败
    f1_fail_f2_success = []  # 文件夹1失败，文件夹2成功
    both_success = []        # 两个都成功
    both_fail = []          # 两个都失败
    
    for episode_key in common_episodes:
        r1 = folder1_map[episode_key]
        r2 = folder2_map[episode_key]
        
        # 获取成功状态，优先使用success字段，如果没有则使用sr字段
        success1 = r1.get('success', r1.get('sr', False))
        success2 = r2.get('success', r2.get('sr', False))
        
        # 确保布尔值
        if isinstance(success1, (int, float)):
            success1 = success1 > 0
        if isinstance(success2, (int, float)):
            success2 = success2 > 0
            
        episode_info = {
            'scene_name': r1['scene_name'],
            'episode_id': r1['episode_id'],
            'f1_success': success1,
            'f2_success': success2,
            'f1_sr': r1.get('sr', 'N/A'),
            'f1_spl': r1.get('spl', 'N/A'),
            'f2_sr': r2.get('sr', 'N/A'),
            'f2_spl': r2.get('spl', 'N/A')
        }
        
        if success1 and not success2:
            f1_success_f2_fail.append(episode_info)
        elif not success1 and success2:
            f1_fail_f2_success.append(episode_info)
        elif success1 and success2:
            both_success.append(episode_info)
        else:
            both_fail.append(episode_info)
    
    cross_validation_stats = {
        'total_common_episodes': len(common_episodes),
        'f1_success_f2_fail_count': len(f1_success_f2_fail),
        'f1_fail_f2_success_count': len(f1_fail_f2_success),
        'both_success_count': len(both_success),
        'both_fail_count': len(both_fail),
        'f1_success_f2_fail_episodes': f1_success_f2_fail,
        'f1_fail_f2_success_episodes': f1_fail_f2_success,
        'both_success_episodes': both_success,
        'both_fail_episodes': both_fail
    }
    
    return cross_validation_stats


def save_comparison_results(folder1_results: List[Dict], folder1_stats: Dict,
                          folder2_results: List[Dict], folder2_stats: Dict,
                          comparison_stats: Dict, cross_validation_stats: Dict, output_file: str):
    """
    保存比较结果到JSON文件
    
    Args:
        folder1_results: 文件夹1的结果
        folder1_stats: 文件夹1的统计信息
        folder2_results: 文件夹2的结果
        folder2_stats: 文件夹2的统计信息
        comparison_stats: 比较统计信息
        cross_validation_stats: 交叉验证统计信息
        output_file: 输出文件路径
    """
    output_data = {
        'comparison_summary': comparison_stats,
        'cross_validation_analysis': cross_validation_stats,
        'folder1_statistics': folder1_stats,
        'folder2_statistics': folder2_stats,
        'folder1_detailed_results': folder1_results,
        'folder2_detailed_results': folder2_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细比较结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='比较两个文件夹中相同index episodes的evaluation results')
    parser.add_argument('--folder1', '-f1', default="/home/yaoaa/habitat-lab/habitat_video_project/eval_model/output",
                       help='文件夹1的路径 (episode数量较少的文件夹)')
    parser.add_argument('--folder2', '-f2', default="/home/yaoaa/habitat-lab/habitat_video_project/eval/output_challenge",
                       help='文件夹2的路径 (episode数量较多的文件夹)')
    parser.add_argument('--save', '-s', 
                       help='保存详细比较结果到指定的JSON文件')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--show_cross_details', '--cross', action='store_true',
                       help='显示详细的交叉验证分析，包括所有成功/失败差异的episodes')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("比较两个文件夹中相同index episodes的Evaluation Results")
    print("=" * 80)
    
    # 步骤1: 提取文件夹1中的所有episode indices
    print(f"\n步骤1: 从文件夹1提取episode indices")
    print(f"文件夹1: {args.folder1}")
    folder1_indices = extract_episode_indices(args.folder1)
    
    if not folder1_indices:
        print("文件夹1中没有找到任何valid episodes")
        return
    
    print(f"文件夹1中找到 {len(folder1_indices)} 个episodes")
    
    # 步骤2: 从文件夹1中提取evaluation results
    print(f"\n步骤2: 从文件夹1提取evaluation results")
    folder1_results = extract_specific_evaluation_results(args.folder1, folder1_indices)
    
    if not folder1_results:
        print("文件夹1中没有找到任何evaluation_results数据")
        return
    
    # 步骤3: 从文件夹2中提取相同indices的evaluation results
    print(f"\n步骤3: 从文件夹2提取相同indices的evaluation results")
    print(f"文件夹2: {args.folder2}")
    folder2_results = extract_specific_evaluation_results(args.folder2, folder1_indices)
    
    if not folder2_results:
        print("文件夹2中没有找到任何匹配的evaluation_results数据")
        return
    
    # 步骤4: 计算两个文件夹的指标
    print(f"\n步骤4: 计算指标")
    avg_sr1, avg_spl1, stats1 = calculate_metrics(folder1_results)
    avg_sr2, avg_spl2, stats2 = calculate_metrics(folder2_results)
    
    # 步骤5: 交叉验证分析
    print(f"\n步骤5: 交叉验证分析")
    cross_validation_stats = analyze_cross_validation(folder1_results, folder2_results)
    
    # 步骤6: 计算差异
    sr_diff = avg_sr2 - avg_sr1
    spl_diff = avg_spl2 - avg_spl1
    sr_improvement = (sr_diff / avg_sr1 * 100) if avg_sr1 > 0 else float('inf')
    spl_improvement = (spl_diff / avg_spl1 * 100) if avg_spl1 > 0 else float('inf')
    
    comparison_stats = {
        'common_episodes_count': len(folder1_indices),
        'folder1_found': len(folder1_results),
        'folder2_found': len(folder2_results),
        'sr_difference': sr_diff,
        'spl_difference': spl_diff,
        'sr_improvement_percent': sr_improvement,
        'spl_improvement_percent': spl_improvement
    }
    
    # 输出结果
    print("\n" + "=" * 80)
    print("比较结果")
    print("=" * 80)
    
    print(f"\n共同episodes数量: {len(folder1_indices)}")
    print(f"文件夹1成功提取: {len(folder1_results)} episodes")
    print(f"文件夹2成功提取: {len(folder2_results)} episodes")
    
    print(f"\nNew统计:")
    print(f"  总episodes: {stats1['total_episodes']}")
    print(f"  成功episodes: {stats1['successful_episodes']}")
    print(f"  成功率: {stats1['success_rate']:.4f} ({stats1['success_rate']*100:.2f}%)")
    print(f"  平均SR: {avg_sr1:.4f}")
    print(f"  平均SPL: {avg_spl1:.4f}")
    print(f"  SPL范围: {stats1['min_spl']:.4f} - {stats1['max_spl']:.4f}")
    
    print(f"\nBaseline统计 (相同episodes):")
    print(f"  总episodes: {stats2['total_episodes']}")
    print(f"  成功episodes: {stats2['successful_episodes']}")
    print(f"  成功率: {stats2['success_rate']:.4f} ({stats2['success_rate']*100:.2f}%)")
    print(f"  平均SR: {avg_sr2:.4f}")
    print(f"  平均SPL: {avg_spl2:.4f}")
    print(f"  SPL范围: {stats2['min_spl']:.4f} - {stats2['max_spl']:.4f}")
    
    print(f"\n性能差异:")
    print(f"  SR差异: {sr_diff:+.4f} ({sr_improvement:+.2f}%)")
    print(f"  SPL差异: {spl_diff:+.4f} ({spl_improvement:+.2f}%)")
    
    print(f"\n交叉验证分析:")
    print(f"  共同episodes总数: {cross_validation_stats['total_common_episodes']}")
    print(f"  两个都成功: {cross_validation_stats['both_success_count']} ({cross_validation_stats['both_success_count']/cross_validation_stats['total_common_episodes']*100:.1f}%)")
    print(f"  两个都失败: {cross_validation_stats['both_fail_count']} ({cross_validation_stats['both_fail_count']/cross_validation_stats['total_common_episodes']*100:.1f}%)")
    print(f"  文件夹1成功，文件夹2失败: {cross_validation_stats['f1_success_f2_fail_count']} ({cross_validation_stats['f1_success_f2_fail_count']/cross_validation_stats['total_common_episodes']*100:.1f}%)")
    print(f"  文件夹1失败，文件夹2成功: {cross_validation_stats['f1_fail_f2_success_count']} ({cross_validation_stats['f1_fail_f2_success_count']/cross_validation_stats['total_common_episodes']*100:.1f}%)")
    
    # 显示具体的差异episodes
    if cross_validation_stats['f1_success_f2_fail_count'] > 0:
        print(f"\n  文件夹1成功但文件夹2失败的episodes:")
        for episode in cross_validation_stats['f1_success_f2_fail_episodes'][:100]:  # 只显示前10个
            print(f"    {episode['scene_name']}/{episode['episode_id']} - F1_SPL: {episode['f1_spl']:.4f}, F2_SPL: {episode['f2_spl']:.4f}")
        if cross_validation_stats['f1_success_f2_fail_count'] > 100:
            print(f"    ... 还有 {cross_validation_stats['f1_success_f2_fail_count']-100} 个")
    
    if cross_validation_stats['f1_fail_f2_success_count'] > 0:
        print(f"\n  文件夹1失败但文件夹2成功的episodes:")
        for episode in cross_validation_stats['f1_fail_f2_success_episodes'][:100]:  # 只显示前10个
            print(f"    {episode['scene_name']}/{episode['episode_id']} - F1_SPL: {episode['f1_spl']:.4f}, F2_SPL: {episode['f2_spl']:.4f}")
        if cross_validation_stats['f1_fail_f2_success_count'] > 100:
            print(f"    ... 还有 {cross_validation_stats['f1_fail_f2_success_count']-100} 个")
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("详细结果对比")
        print("=" * 80)
        
        # 创建episode到结果的映射
        folder1_map = {(r['scene_name'], r['episode_id']): r for r in folder1_results}
        folder2_map = {(r['scene_name'], r['episode_id']): r for r in folder2_results}
        
        print(f"{'Scene/Episode':<30} {'F1_SR':<8} {'F1_SPL':<8} {'F2_SR':<8} {'F2_SPL':<8} {'SR_Diff':<8} {'SPL_Diff':<8}")
        print("-" * 80)
        
        for scene_episode in sorted(folder1_indices):
            scene, episode = scene_episode
            key = f"{scene}/{episode}"
            
            r1 = folder1_map.get(scene_episode, {})
            r2 = folder2_map.get(scene_episode, {})
            
            sr1 = r1.get('sr', 'N/A')
            spl1 = r1.get('spl', 'N/A')
            sr2 = r2.get('sr', 'N/A')
            spl2 = r2.get('spl', 'N/A')
            
            if isinstance(sr1, bool):
                sr1 = 1.0 if sr1 else 0.0
            if isinstance(sr2, bool):
                sr2 = 1.0 if sr2 else 0.0
                
            sr_diff_episode = sr2 - sr1 if (sr1 != 'N/A' and sr2 != 'N/A') else 'N/A'
            spl_diff_episode = spl2 - spl1 if (spl1 != 'N/A' and spl2 != 'N/A') else 'N/A'
            
            print(f"{key:<30} {sr1:<8} {spl1:<8.4f} {sr2:<8} {spl2:<8.4f} {sr_diff_episode:<8} {spl_diff_episode:<8.4f}")
    
    if args.show_cross_details:
        print("\n" + "=" * 80)
        print("详细交叉验证分析")
        print("=" * 80)
        
        if cross_validation_stats['f1_success_f2_fail_count'] > 0:
            print(f"\n文件夹1成功但文件夹2失败的所有episodes ({cross_validation_stats['f1_success_f2_fail_count']} 个):")
            print(f"{'Scene/Episode':<30} {'F1_SR':<8} {'F1_SPL':<8} {'F2_SR':<8} {'F2_SPL':<8}")
            print("-" * 70)
            for episode in cross_validation_stats['f1_success_f2_fail_episodes']:
                key = f"{episode['scene_name']}/{episode['episode_id']}"
                print(f"{key:<30} {episode['f1_sr']:<8} {episode['f1_spl']:<8.4f} {episode['f2_sr']:<8} {episode['f2_spl']:<8.4f}")
        
        if cross_validation_stats['f1_fail_f2_success_count'] > 0:
            print(f"\n文件夹1失败但文件夹2成功的所有episodes ({cross_validation_stats['f1_fail_f2_success_count']} 个):")
            print(f"{'Scene/Episode':<30} {'F1_SR':<8} {'F1_SPL':<8} {'F2_SR':<8} {'F2_SPL':<8}")
            print("-" * 70)
            for episode in cross_validation_stats['f1_fail_f2_success_episodes']:
                key = f"{episode['scene_name']}/{episode['episode_id']}"
                print(f"{key:<30} {episode['f1_sr']:<8} {episode['f1_spl']:<8.4f} {episode['f2_sr']:<8} {episode['f2_spl']:<8.4f}")
        
        if cross_validation_stats['both_success_count'] > 0:
            print(f"\n两个文件夹都成功的episodes ({cross_validation_stats['both_success_count']} 个):")
            print(f"{'Scene/Episode':<30} {'F1_SPL':<8} {'F2_SPL':<8} {'SPL_Diff':<8}")
            print("-" * 60)
            for episode in cross_validation_stats['both_success_episodes'][:20]:  # 只显示前20个
                key = f"{episode['scene_name']}/{episode['episode_id']}"
                spl_diff = episode['f2_spl'] - episode['f1_spl'] if isinstance(episode['f1_spl'], (int, float)) and isinstance(episode['f2_spl'], (int, float)) else 'N/A'
                print(f"{key:<30} {episode['f1_spl']:<8.4f} {episode['f2_spl']:<8.4f} {spl_diff:<8}")
            if cross_validation_stats['both_success_count'] > 20:
                print(f"    ... 还有 {cross_validation_stats['both_success_count']-20} 个")
        
        if cross_validation_stats['both_fail_count'] > 0:
            print(f"\n两个文件夹都失败的episodes ({cross_validation_stats['both_fail_count']} 个):")
            print(f"{'Scene/Episode':<30} {'F1_SPL':<8} {'F2_SPL':<8}")
            print("-" * 50)
            for episode in cross_validation_stats['both_fail_episodes'][:20]:  # 只显示前20个
                key = f"{episode['scene_name']}/{episode['episode_id']}"
                print(f"{key:<30} {episode['f1_spl']:<8.4f} {episode['f2_spl']:<8.4f}")
            if cross_validation_stats['both_fail_count'] > 20:
                print(f"    ... 还有 {cross_validation_stats['both_fail_count']-20} 个")
    
    # 保存结果
    if args.save:
        save_comparison_results(folder1_results, stats1, folder2_results, stats2, 
                              comparison_stats, cross_validation_stats, args.save)
    
    print("\n处理完成!")


if __name__ == "__main__":
    main()
