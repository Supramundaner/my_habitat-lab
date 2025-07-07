#!/usr/bin/env python3
"""
碰撞检测结果可视化工具
Collision Detection Visualization Tool
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import argparse


def load_debug_data(filepath):
    """加载调试数据"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load debug data: {e}")
        return None


def plot_agent_positions(debug_data, save_path=None):
    """绘制智能体位置图"""
    if not debug_data or 'agent_states' not in debug_data:
        print("No agent state data available")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    agent_states = debug_data['agent_states']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (agent_id, state) in enumerate(agent_states.items()):
        position = state['position']
        rotation = state['rotation']
        has_robot = state['has_robot']
        
        # 绘制智能体位置
        color = colors[i % len(colors)]
        marker = 'o' if has_robot else '^'
        label = f"{agent_id} ({'Robot' if has_robot else 'Virtual'})"
        
        ax.scatter(position[0], position[2], c=color, s=200, marker=marker, 
                  label=label, alpha=0.8, edgecolors='black', linewidth=2)
        
        # 绘制朝向箭头
        # 从四元数计算朝向 (简化版本)
        # 这里假设主要旋转在Y轴上
        yaw = 2 * np.arctan2(rotation[1], rotation[3])
        
        arrow_length = 0.5
        dx = arrow_length * np.cos(yaw)
        dz = arrow_length * np.sin(yaw)
        
        ax.arrow(position[0], position[2], dx, dz, 
                head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
        
        # 绘制安全半径
        if 'configuration' in debug_data:
            radius = debug_data['configuration'].get('min_agent_distance', 0.8) / 2
            circle = patches.Circle((position[0], position[2]), radius, 
                                  fill=False, edgecolor=color, linestyle='--', alpha=0.5)
            ax.add_patch(circle)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Z Position (m)')
    ax.set_title('Agent Positions and Orientations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Agent positions plot saved to: {save_path}")
    
    plt.show()


def plot_collision_statistics(debug_data, save_path=None):
    """绘制碰撞统计图表"""
    if not debug_data or 'collision_statistics' not in debug_data:
        print("No collision statistics available")
        return
    
    stats = debug_data['collision_statistics']
    recent_contacts = debug_data.get('recent_contacts', [])
    
    if not recent_contacts:
        print("No recent contact data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上图：接触点数量随时间变化
    timestamps = [entry['timestamp'] for entry in recent_contacts]
    contact_counts = [entry['contact_count'] for entry in recent_contacts]
    
    # 转换时间戳为相对时间
    if timestamps:
        base_time = min(timestamps)
        relative_times = [(t - base_time) for t in timestamps]
        
        ax1.plot(relative_times, contact_counts, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Contact Points')
        ax1.set_title('Contact Points Over Time')
        ax1.grid(True, alpha=0.3)
        
        if contact_counts:
            ax1.axhline(y=np.mean(contact_counts), color='r', linestyle='--', 
                       label=f'Average: {np.mean(contact_counts):.1f}')
            ax1.legend()
    
    # 下图：配置参数可视化
    config = debug_data.get('configuration', {})
    param_names = []
    param_values = []
    
    for key, value in config.items():
        if isinstance(value, (int, float)):
            param_names.append(key.replace('_', '\n'))
            param_values.append(value)
    
    if param_names:
        bars = ax2.bar(param_names, param_values, alpha=0.7, 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Collision Detection Configuration')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, param_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Collision statistics plot saved to: {save_path}")
    
    plt.show()


def create_summary_report(debug_data, output_file=None):
    """创建总结报告"""
    if not debug_data:
        return "No data available"
    
    report_lines = [
        "=" * 60,
        "COLLISION DETECTION ANALYSIS REPORT",
        "=" * 60,
        f"Generated at: {debug_data.get('timestamp', 'Unknown')}",
        ""
    ]
    
    # 碰撞统计
    stats = debug_data.get('collision_statistics', {})
    report_lines.extend([
        "COLLISION STATISTICS:",
        f"  Detection Enabled: {stats.get('detection_enabled', 'Unknown')}",
        f"  Physics Enabled: {stats.get('physics_enabled', 'Unknown')}",
        f"  Total Frames: {stats.get('total_frames', 0)}",
        f"  Recent Avg Contacts: {stats.get('recent_avg_contacts', 0):.2f}",
        f"  Last Contact Count: {stats.get('last_contact_count', 0)}",
        ""
    ])
    
    # 智能体状态
    agent_states = debug_data.get('agent_states', {})
    report_lines.append("AGENT STATES:")
    for agent_id, state in agent_states.items():
        pos = state['position']
        has_robot = state['has_robot']
        report_lines.extend([
            f"  {agent_id}:",
            f"    Type: {'Physical Robot' if has_robot else 'Virtual Agent'}",
            f"    Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        ])
    
    report_lines.append("")
    
    # 配置参数
    config = debug_data.get('configuration', {})
    if config:
        report_lines.append("CONFIGURATION:")
        for key, value in config.items():
            report_lines.append(f"  {key}: {value}")
    
    # 接触历史分析
    recent_contacts = debug_data.get('recent_contacts', [])
    if recent_contacts:
        contact_counts = [entry['contact_count'] for entry in recent_contacts]
        report_lines.extend([
            "",
            "CONTACT HISTORY ANALYSIS:",
            f"  Total Recorded Frames: {len(recent_contacts)}",
            f"  Max Contacts: {max(contact_counts)}",
            f"  Min Contacts: {min(contact_counts)}",
            f"  Average Contacts: {np.mean(contact_counts):.2f}",
            f"  Contact Variance: {np.var(contact_counts):.2f}"
        ])
    
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Summary report saved to: {output_file}")
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Collision Detection Visualization Tool')
    parser.add_argument('--input', '-i', default='outputs/collision_tests/collision_test_debug.json',
                       help='Input debug data file')
    parser.add_argument('--output-dir', '-o', default='outputs/collision_tests/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--show-plots', action='store_true', 
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading debug data from: {args.input}")
    debug_data = load_debug_data(args.input)
    
    if not debug_data:
        print("Failed to load debug data")
        return 1
    
    print("Data loaded successfully")
    
    # 生成可视化
    print("Generating visualizations...")
    
    # 智能体位置图
    pos_plot_path = output_dir / "agent_positions.png"
    plot_agent_positions(debug_data, pos_plot_path if not args.show_plots else None)
    
    # 碰撞统计图
    stats_plot_path = output_dir / "collision_statistics.png"
    plot_collision_statistics(debug_data, stats_plot_path if not args.show_plots else None)
    
    # 生成报告
    report_path = output_dir / "analysis_report.txt"
    report = create_summary_report(debug_data, report_path)
    print("\nSUMMARY REPORT:")
    print(report)
    
    print(f"\nVisualization completed. Files saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
