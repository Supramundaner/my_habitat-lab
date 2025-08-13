"""
Habitat视频生成器 - 主入口脚本
重构版本，采用MVC架构设计
"""

import argparse
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulator import HabitatSimulator
from src.video_composer import VideoComposer
from src.action_processor import ActionProcessor
from src.map_builder import OccupancyMapBuilder
from src.utils import (
    load_json_config, 
    write_json_report, 
    generate_output_paths,
    validate_config,
    initialize_gpu,
    clear_gpu_cache
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Habitat视频生成器 - 在3D场景中模拟智能体动作并生成视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.json',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--actions',
        type=str,
        default='configs/example_actions.json',
        help='动作序列文件路径'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置详细输出
    if args.verbose:
        print("详细模式已启用")
    
    simulator = None
    composer = None
    
    try:
        print("=" * 60)
        print("Habitat视频生成器 - 重构版本")
        print("=" * 60)
        
        # 1. 加载配置和动作序列
        print("1. 加载配置文件...")
        config = load_json_config(args.config)
        
        # 初始化GPU设置
        print("1.5. 初始化GPU设置...")
        initialize_gpu(config)
        
        # 验证配置文件
        if not validate_config(config):
            print("配置文件验证失败")
            return 1
        
        print("2. 加载动作序列...")
        actions = load_json_config(args.actions)
        
        # 覆盖输出目录设置
        if args.output_dir:
            config['output_dir'] = args.output_dir
        
        # 3. 生成输出路径
        print("3. 生成输出路径...")
        paths = generate_output_paths(config['output_dir'])
        print(f"   视频输出: {paths['video']}")
        print(f"   报告输出: {paths['report']}")
        
        # 4. 初始化核心组件
        print("4. 初始化模拟器...")
        simulator = HabitatSimulator(config)
        
        print("5. 设置场景和智能体...")
        agent_state = actions.get('agent_state', None)  # 获取3D agent_state，如果存在
        initial_state = actions.get('initial_state', None)  # 获取传统2D initial_state，如果存在
        
        if agent_state is None and initial_state is None:
            raise ValueError("必须提供 'agent_state' 或 'initial_state' 中的至少一个")
        
        # 如果没有initial_state但有agent_state，创建一个默认的initial_state以确保向后兼容
        if initial_state is None:
            initial_state = {
                "position": [0.0, 0.0],  # 默认2D位置
                "rotation": 0.0  # 默认朝向
            }
            print("警告: 没有提供initial_state，使用默认值以确保向后兼容")
        
        simulator.setup_scene_and_agent(initial_state, agent_state)
        
        print("6. 初始化视频合成器...")
        composer = VideoComposer(simulator, config, paths['video'])
        
        print("7. 初始化占用地图构建器...")
        use_gpu = config.get('gpu', {}).get('enabled', False)
        map_builder = OccupancyMapBuilder(use_gpu=use_gpu, config=config)
        composer.set_map_builder(map_builder, config)
        
        print("8. 初始化动作处理器...")
        processor = ActionProcessor(simulator, composer, config, map_builder)
        
        # 9. 添加初始帧
        print("9. 添加初始帧...")
        composer.add_frame()
        
        # 10. 执行动作序列
        print("10. 执行动作序列...")
        start_time = datetime.now()
        
        report_data = processor.execute_sequence(actions['sequence'])
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 11. 生成最终报告
        print("11. 生成执行报告...")
        final_state = simulator.get_robot_state()
        execution_stats = processor.get_execution_stats()
        
        full_report = {
            'execution_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time,
                'total_frames': execution_stats['total_frames'],
                'video_duration_seconds': execution_stats['total_duration']
            },
            'config': config,
            'final_agent_state': {
                'position': final_state['position'].tolist(),
                'rotation': final_state['rotation'].tolist()
            },
            'original_sequence': actions['sequence'],
            'completed_sequence': report_data['completed_actions'],
            'collision_at_action': report_data['collision_action'],
            'execution_stats': execution_stats
        }
        
        write_json_report(paths['report'], full_report)
        
        # 12. 输出执行总结
        print("\n" + "=" * 60)
        print("执行完成!")
        print("=" * 60)
        print(f"执行时间: {execution_time:.2f} 秒")
        print(f"生成帧数: {execution_stats['total_frames']}")
        print(f"视频时长: {execution_stats['total_duration']:.2f} 秒")
        print(f"成功动作: {len(report_data['completed_actions'])}/{len(actions['sequence'])}")
        
        if report_data['collision_action']:
            collision_info = report_data['collision_action']
            print(f"不可导航检测: 在第 {collision_info['index'] + 1} 个动作处检测到不可导航点")
            print(f"不可导航动作: {collision_info['action']}")
        else:
            print("不可导航检测: 正常")
        
        print(f"\n输出文件:")
        print(f"  视频: {paths['video']}")
        print(f"  报告: {paths['report']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
        return 1
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    finally:
        # 清理资源
        print("\n清理资源...")
        if composer:
            composer.save_and_close()
        if simulator:
            simulator.close()
        # 清理GPU缓存
        clear_gpu_cache()
        print("清理完成")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
