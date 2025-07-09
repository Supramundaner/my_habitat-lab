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
from src.utils import (
    load_json_config, 
    write_json_report, 
    generate_output_paths,
    validate_config
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
        simulator.setup_scene_and_agent(actions['initial_state'])
        
        print("6. 初始化视频合成器...")
        composer = VideoComposer(simulator, config, paths['video'])
        
        print("7. 初始化动作处理器...")
        processor = ActionProcessor(simulator, composer, config)
        
        # 8. 添加初始帧
        print("8. 添加初始帧...")
        composer.add_frame()
        
        # 9. 执行动作序列
        print("9. 执行动作序列...")
        start_time = datetime.now()
        
        report_data = processor.execute_sequence(actions['sequence'])
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 10. 生成最终报告
        print("10. 生成执行报告...")
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
        
        # 11. 输出执行总结
        print("\n" + "=" * 60)
        print("执行完成!")
        print("=" * 60)
        print(f"执行时间: {execution_time:.2f} 秒")
        print(f"生成帧数: {execution_stats['total_frames']}")
        print(f"视频时长: {execution_stats['total_duration']:.2f} 秒")
        print(f"成功动作: {len(report_data['completed_actions'])}/{len(actions['sequence'])}")
        
        if report_data['collision_action']:
            collision_info = report_data['collision_action']
            print(f"碰撞检测: 在第 {collision_info['index'] + 1} 个动作处检测到碰撞")
            print(f"碰撞动作: {collision_info['action']}")
        else:
            print("碰撞检测: 无碰撞")
        
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
        print("清理完成")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
