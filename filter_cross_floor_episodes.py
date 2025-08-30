#!/usr/bin/env python3
"""
楼层过滤脚本 - 过滤掉需要跨楼层的导航episode
基于楼层分离算法，保留同楼层的episode，过滤跨楼层的episode

使用方法:
python filter_cross_floor_episodes.py --input_dir data/datasets/objectnav/hm3d/v1/val/content_preprocessed --output_dir data/datasets/objectnav/hm3d/v1/val/content_preprocessed_filtered
"""

import os
import json
import argparse
import numpy as np
import habitat_sim
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import traceback
from tqdm import tqdm

# 导入楼层分离算法
from habitat_video_project.src.multi_floor_topdown import get_floor_navigable_extents


class FloorFilterProcessor:
    """楼层过滤处理器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_scenes': 0,
            'processed_scenes': 0,
            'failed_scenes': 0,
            'total_episodes_before': 0,
            'total_episodes_after': 0,
            'scene_details': {}
        }
    
    def create_simulator(self, scene_path: str) -> Optional[habitat_sim.Simulator]:
        """创建Habitat模拟器实例"""
        try:
            # 后端配置
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = scene_path
            backend_cfg.enable_physics = False
            backend_cfg.random_seed = 1
            
            # 简单的传感器配置（只需要用于导航网格）
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = "color_sensor"
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.resolution = [256, 256]
            sensor_spec.position = [0, 0, 0]
            sensor_spec.hfov = 90.0
            
            # 智能体配置
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [sensor_spec]
            
            # 创建模拟器
            cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
            sim = habitat_sim.Simulator(cfg)
            
            # 确保导航网格已加载
            if not sim.pathfinder.is_loaded:
                print(f"  重新计算导航网格...")
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
                if not success:
                    print(f"  警告: 导航网格计算失败")
                    sim.close()
                    return None
            
            return sim
            
        except Exception as e:
            print(f"  创建模拟器失败: {e}")
            return None
    
    def assign_position_to_floor(self, position: List[float], floor_extents: List[Dict], debug: bool = False) -> Optional[int]:
        """
        将3D位置分配到对应楼层
        逻辑：分配到Y坐标向下的第一个楼层（物体通常放在楼层表面上）
        """
        if not floor_extents:
            if debug:
                print(f"    警告: 没有楼层信息，无法分配位置 {position}")
            return None
            
        y_coord = position[1]
        
        # 按楼层高度从低到高排序，找到第一个楼层上边界在当前位置下方或接近的楼层
        sorted_floors = sorted(enumerate(floor_extents), key=lambda x: x[1]['max'])
        
        tolerance = -0.2  # 增加容差，因为物体可能稍微悬浮在楼层上方
        
        for floor_idx, floor in sorted_floors:
            # 如果当前位置在这个楼层的上边界上方或接近，则分配到这个楼层
            if y_coord <= floor['max'] + tolerance:
                if debug:
                    print(f"    位置Y={y_coord:.2f}分配到楼层{floor_idx} (范围[{floor['min']:.2f}, {floor['max']:.2f}]) - 第一个max在上方的楼层")
                return floor_idx
        
        # 如果没有找到合适的楼层，分配到最低的楼层
        lowest_floor_idx = min(enumerate(floor_extents), key=lambda x: x[1]['min'])[0]
        lowest_floor = floor_extents[lowest_floor_idx]
        if debug:
            print(f"    位置Y={y_coord:.2f}太低，强制分配到最低楼层{lowest_floor_idx} (范围[{lowest_floor['min']:.2f}, {lowest_floor['max']:.2f}])")
        return lowest_floor_idx
    
    def extract_scene_path_from_episode(self, episode: Dict) -> str:
        """从episode中提取场景文件路径"""
        scene_id = episode['scene_id']
        # scene_id格式: "hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
        # 需要转换为完整路径: /home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
        base_path = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2"
        return os.path.join(base_path, scene_id)
    
    def extract_scene_name_from_episode(self, episode: Dict) -> str:
        """从episode中提取场景名称"""
        scene_id = episode['scene_id']
        # 提取场景名称，如 "4ok3usBNeis"
        return scene_id.split('/')[-1].split('.')[0]
    
    def assign_agent_to_floor(self, position: List[float], floor_extents: List[Dict]) -> Optional[int]:
        """
        将agent位置分配到对应楼层
        逻辑：与assign_position_to_floor相同，分配到Y坐标向下的第一个楼层
        """
        return self.assign_position_to_floor(position, floor_extents)

    def should_keep_episode(self, episode: Dict, goals_by_category: Dict, 
                           floor_extents: List[Dict], debug: bool = False) -> Tuple[bool, str]:
        """判断是否应该保留episode"""
        try:
            episode_id = episode.get('episode_id', 'unknown')
            if debug:
                print(f"  分析episode {episode_id}:")
                
            # 获取起始位置楼层（agent使用不同的分配逻辑）
            start_position = episode['start_position']
            start_floor = self.assign_position_to_floor(start_position, floor_extents, debug)
            
            if start_floor is None:
                reason = f"起始位置Y={start_position[1]:.2f}无法分配到任何楼层"
                if debug:
                    print(f"    ❌ {reason}")
                return False, reason
            
            # 获取目标类别
            target_category = episode['object_category']
            scene_name = self.extract_scene_name_from_episode(episode)
            
            # 构造目标类别的完整键名
            category_key = f"{scene_name}.basis.glb_{target_category}"
            target_goals = goals_by_category.get(category_key, [])
            
            if debug:
                print(f"    目标类别: {target_category}, 键名: {category_key}")
                print(f"    起始楼层: {start_floor}")
            
            if not target_goals:
                reason = f"未找到目标类别 {target_category} 的goals"
                if debug:
                    print(f"    ❌ {reason}")
                    available_keys = list(goals_by_category.keys())[:5]
                    print(f"    可用键名示例: {available_keys}")
                return False, reason
            
            # 检查是否有同楼层的目标
            same_floor_goals = 0
            goal_floor_distribution = {}
            
            if debug:
                print(f"    检查 {len(target_goals)} 个目标的楼层分布:")
            
            for i, goal in enumerate(target_goals):
                goal_position = goal['position']
                goal_floor = self.assign_position_to_floor(goal_position, floor_extents, debug and i < 3)  # 只打印前3个
                
                if goal_floor is not None:
                    goal_floor_distribution[goal_floor] = goal_floor_distribution.get(goal_floor, 0) + 1
                    if goal_floor == start_floor:
                        same_floor_goals += 1
            
            if debug:
                print(f"    目标分布: {goal_floor_distribution}")
                print(f"    同楼层目标数: {same_floor_goals}")
            
            if same_floor_goals > 0:
                reason = f"起始楼层{start_floor}找到{same_floor_goals}个目标 (分布:{goal_floor_distribution})"
                if debug:
                    print(f"    ✅ {reason}")
                return True, reason
            else:
                reason = f"起始楼层{start_floor}无目标，{len(target_goals)}个目标分布:{goal_floor_distribution}"
                if debug:
                    print(f"    ❌ {reason}")
                return False, reason
                
        except Exception as e:
            return False, f"处理episode时出错: {e}"
    
    def process_scene_file(self, json_file_path: Path) -> bool:
        """处理单个场景文件"""
        scene_name = json_file_path.stem
        print(f"\n处理场景: {scene_name}")
        
        try:
            # 读取JSON数据
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            episodes = data.get('episodes', [])
            goals_by_category = data.get('goals_by_category', {})
            
            if not episodes:
                print(f"  警告: 场景 {scene_name} 没有episodes")
                return False
            
            print(f"  原始episodes数量: {len(episodes)}")
            
            # 获取场景路径（从第一个episode中提取）
            scene_path = self.extract_scene_path_from_episode(episodes[0])
            print(f"  场景路径: {scene_path}")
            
            # 创建模拟器
            sim = self.create_simulator(scene_path)
            if sim is None:
                print(f"  错误: 无法为场景 {scene_name} 创建模拟器")
                return False
            
            try:
                # 获取楼层信息
                print(f"  分析楼层结构...")
                floor_extents = get_floor_navigable_extents(sim)
                print(f"  检测到 {len(floor_extents)} 个楼层:")
                for i, floor in enumerate(floor_extents):
                    print(f"    楼层 {i+1}: Y范围 [{floor['min']:.2f}, {floor['max']:.2f}], 平均 {floor['mean']:.2f}")
                
                # 过滤episodes
                filtered_episodes = []
                filter_reasons = []
                
                print(f"  过滤episodes...")
                debug_sample_size = min(3, len(episodes))  # 只对前3个episode启用调试
                for i, episode in enumerate(tqdm(episodes, desc="  处理episodes")):
                    debug_this = i < debug_sample_size  # 只对前几个episode启用详细调试
                    should_keep, reason = self.should_keep_episode(episode, goals_by_category, floor_extents, debug_this)
                    if should_keep:
                        filtered_episodes.append(episode)
                    filter_reasons.append(reason)
                
                # 更新统计信息
                episodes_before = len(episodes)
                episodes_after = len(filtered_episodes)
                filter_rate = (episodes_before - episodes_after) / episodes_before * 100
                
                print(f"  过滤结果: {episodes_before} -> {episodes_after} ({filter_rate:.1f}% 被过滤)")
                
                # 保存统计详情
                self.stats['scene_details'][scene_name] = {
                    'episodes_before': episodes_before,
                    'episodes_after': episodes_after,
                    'filter_rate': filter_rate,
                    'floors_detected': len(floor_extents),
                    'floor_ranges': [f"[{f['min']:.2f}, {f['max']:.2f}]" for f in floor_extents]
                }
                
                # 更新全局统计
                self.stats['total_episodes_before'] += episodes_before
                self.stats['total_episodes_after'] += episodes_after
                
                # 创建过滤后的数据
                filtered_data = data.copy()
                filtered_data['episodes'] = filtered_episodes
                
                # 保存过滤后的文件
                output_file = self.output_dir / json_file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                
                print(f"  已保存到: {output_file}")
                return True
                
            finally:
                # 清理模拟器
                sim.close()
                
        except Exception as e:
            print(f"  处理场景 {scene_name} 时出错: {e}")
            traceback.print_exc()
            return False
    
    def process_all_scenes(self):
        """处理所有场景文件"""
        print(f"开始处理目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 获取所有JSON文件
        json_files = list(self.input_dir.glob("*.json"))
        self.stats['total_scenes'] = len(json_files)
        
        print(f"找到 {len(json_files)} 个场景文件")
        
        # 处理每个文件
        for json_file in json_files:
            success = self.process_scene_file(json_file)
            if success:
                self.stats['processed_scenes'] += 1
            else:
                self.stats['failed_scenes'] += 1
        
        # 保存统计信息
        self.save_statistics()
        
        # 打印最终统计
        self.print_final_statistics()
    
    def save_statistics(self):
        """保存统计信息到文件"""
        stats_file = self.output_dir / "filter_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计信息已保存到: {stats_file}")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        print("\n" + "="*60)
        print("最终统计结果")
        print("="*60)
        print(f"总场景数: {self.stats['total_scenes']}")
        print(f"成功处理: {self.stats['processed_scenes']}")
        print(f"处理失败: {self.stats['failed_scenes']}")
        print(f"处理成功率: {self.stats['processed_scenes']/self.stats['total_scenes']*100:.1f}%")
        print()
        print(f"总episodes数 (处理前): {self.stats['total_episodes_before']}")
        print(f"总episodes数 (处理后): {self.stats['total_episodes_after']}")
        
        if self.stats['total_episodes_before'] > 0:
            overall_filter_rate = (self.stats['total_episodes_before'] - self.stats['total_episodes_after']) / self.stats['total_episodes_before'] * 100
            print(f"总体过滤率: {overall_filter_rate:.1f}%")
        
        print("\n各场景详细信息:")
        for scene_name, details in self.stats['scene_details'].items():
            print(f"  {scene_name}: {details['episodes_before']} -> {details['episodes_after']} "
                  f"({details['filter_rate']:.1f}% 过滤, {details['floors_detected']} 楼层)")


def main():
    parser = argparse.ArgumentParser(description='过滤跨楼层的导航episodes')
    parser.add_argument('--input_dir', 
                       default='data/datasets/objectnav/hm3d/v1/val/content_preprocessed',
                       help='输入目录路径')
    parser.add_argument('--output_dir', 
                       default='data/datasets/objectnav/hm3d/v1/val/content_preprocessed_filtered',
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    # 创建处理器并运行
    processor = FloorFilterProcessor(args.input_dir, args.output_dir)
    processor.process_all_scenes()


if __name__ == "__main__":
    main()
