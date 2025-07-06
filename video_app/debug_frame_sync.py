#!/usr/bin/env python3
"""
调试脚本：检查每帧捕获时3D agent状态与2D地图标注的同步性
重点排查视频帧合成流程中的时序问题
"""

import sys
import os
import math
import numpy as np
from PIL import Image, ImageDraw
import json
from typing import List, Dict, Any

# 添加必要的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../interactive_app/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from habitat_video_generator import HabitatVideoGenerator
import habitat_sim
import magnum as mn


class FrameSyncDebugger:
    """帧同步调试器"""
    
    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.video_generator = HabitatVideoGenerator(scene_path)
        self.debug_data = []
        
    def test_single_frame_consistency(self, target_position: np.ndarray, target_rotation: np.ndarray = None):
        """测试单帧的3D状态与2D标注一致性"""
        print(f"\n=== 测试单帧一致性 ===")
        print(f"目标位置: {target_position}")
        
        # 初始化环境
        if not hasattr(self.video_generator, 'simulator') or self.video_generator.simulator is None:
            self.video_generator._initialize_simulator()
        
        # 如果没有指定旋转，使用默认朝向
        if target_rotation is None:
            target_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
        
        # 移动agent到目标位置
        print(f"移动agent到目标位置...")
        self.video_generator.simulator.move_agent_to(target_position, target_rotation)
        
        # 捕获并分析帧
        frame_data = self._capture_and_analyze_frame()
        
        return frame_data
    
    def test_movement_sequence_consistency(self, waypoints: List[np.ndarray]):
        """测试移动序列中每帧的一致性"""
        print(f"\n=== 测试移动序列一致性 ===")
        print(f"路径点数量: {len(waypoints)}")
        
        # 初始化环境
        self.video_generator.initialize_environment()
        
        sequence_data = []
        
        for i, waypoint in enumerate(waypoints):
            print(f"\n--- 处理路径点 {i+1}/{len(waypoints)} ---")
            print(f"目标位置: {waypoint}")
            
            # 移动到路径点
            self.video_generator.simulator.move_agent_to(waypoint)
            
            # 捕获并分析帧
            frame_data = self._capture_and_analyze_frame()
            frame_data['waypoint_index'] = i
            frame_data['target_position'] = waypoint.tolist()
            
            sequence_data.append(frame_data)
            
            # 检查是否有显著偏差
            if frame_data['position_error'] > 0.1:
                print(f"  ⚠️  WARNING: 位置偏差 {frame_data['position_error']:.3f}m")
        
        return sequence_data
    
    def _capture_and_analyze_frame(self) -> Dict[str, Any]:
        """捕获帧并分析3D/2D同步性"""
        # 获取当前agent状态
        agent_state = self.video_generator.simulator.get_agent_state()
        agent_pos = agent_state.position
        agent_rot = agent_state.rotation
        
        print(f"  Agent 3D位置: {agent_pos}")
        print(f"  Agent 3D旋转: {agent_rot}")
        
        # 转换为2D地图坐标
        map_x, map_y = self.video_generator.simulator.world_to_map_coords(agent_pos)
        print(f"  2D地图坐标: ({map_x:.1f}, {map_y:.1f})")
        
        # 反向转换验证
        recovered_pos = self.video_generator.simulator.map_coords_to_world(map_x, map_y)
        position_error = np.linalg.norm(recovered_pos - agent_pos)
        print(f"  坐标转换误差: {position_error:.6f}m")
        
        # 获取FPV图像
        fpv_image = self.video_generator.simulator.get_fpv_observation()
        
        # 获取地图图像并绘制agent
        map_image = self.video_generator.simulator.base_map_image.copy()
        self._draw_agent_debug(map_image, agent_pos, agent_rot, map_x, map_y)
        
        # 分析朝向
        forward_vector = self._extract_forward_vector(agent_rot)
        print(f"  Agent前向向量: {forward_vector}")
        
        # 检查navmesh snap的影响
        navmesh_pos = self.video_generator.simulator.get_position_with_navmesh_height(agent_pos)
        navmesh_diff = np.linalg.norm(navmesh_pos - agent_pos)
        print(f"  NavMesh调整差异: {navmesh_diff:.6f}m")
        
        # 检查是否在有效位置
        is_navigable = self.video_generator.simulator.sim.pathfinder.is_navigable(agent_pos)
        print(f"  位置可导航性: {is_navigable}")
        
        # 返回调试数据
        return {
            'agent_3d_position': agent_pos.tolist(),
            'agent_3d_rotation': agent_rot.tolist(),
            'map_2d_coords': [map_x, map_y],
            'recovered_position': recovered_pos.tolist(),
            'position_error': position_error,
            'forward_vector': forward_vector.tolist(),
            'navmesh_adjusted_pos': navmesh_pos.tolist(),
            'navmesh_difference': navmesh_diff,
            'is_navigable': is_navigable,
            'fpv_image_shape': fpv_image.shape,
            'map_image_size': map_image.size
        }
    
    def _draw_agent_debug(self, image: Image.Image, agent_pos: np.ndarray, agent_rot: np.ndarray, 
                         map_x: float, map_y: float):
        """在地图上绘制agent位置（调试版本）"""
        draw = ImageDraw.Draw(image)
        
        # 绘制红色圆点表示agent位置
        dot_radius = 12
        draw.ellipse([
            map_x - dot_radius, map_y - dot_radius,
            map_x + dot_radius, map_y + dot_radius
        ], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
        
        # 绘制朝向箭头
        forward_vec = self._extract_forward_vector(agent_rot)
        arrow_length = 30
        arrow_end_x = map_x + forward_vec[0] * arrow_length
        arrow_end_y = map_y + forward_vec[2] * arrow_length
        
        # 绘制箭头
        draw.line([(map_x, map_y), (arrow_end_x, arrow_end_y)], 
                 fill=(0, 255, 0), width=4)
        
        # 绘制坐标文字
        try:
            font_size = 20
            # 创建字体 - 使用默认字体
            text = f"({agent_pos[0]:.2f}, {agent_pos[2]:.2f})"
            draw.text((map_x + 15, map_y + 15), text, fill=(255, 255, 0))
        except Exception as e:
            print(f"    绘制文字失败: {e}")
    
    def _extract_forward_vector(self, rotation: np.ndarray) -> np.ndarray:
        """从四元数中提取前向向量"""
        try:
            if len(rotation) == 4:
                quat = mn.Quaternion(
                    mn.Vector3(rotation[0], rotation[1], rotation[2]),
                    rotation[3]
                )
                # 在Habitat中，-Z轴是前方
                forward_vec = quat.transform_vector(mn.Vector3(0, 0, -1))
                return np.array([forward_vec.x, forward_vec.y, forward_vec.z])
            else:
                return np.array([0, 0, -1])
        except Exception as e:
            print(f"    提取前向向量失败: {e}")
            return np.array([0, 0, -1])
    
    def test_frame_capture_timing(self, position: np.ndarray, num_captures: int = 5):
        """测试帧捕获时序 - 多次捕获同一位置检查一致性"""
        print(f"\n=== 测试帧捕获时序 ===")
        print(f"测试位置: {position}")
        print(f"捕获次数: {num_captures}")
        
        # 初始化环境
        self.video_generator.initialize_environment()
        
        # 移动到测试位置
        self.video_generator.simulator.move_agent_to(position)
        
        captures = []
        for i in range(num_captures):
            print(f"\n--- 第 {i+1} 次捕获 ---")
            
            # 重新获取agent状态（检查状态是否稳定）
            agent_state = self.video_generator.simulator.get_agent_state()
            
            # 捕获并分析
            frame_data = self._capture_and_analyze_frame()
            frame_data['capture_index'] = i
            
            captures.append(frame_data)
        
        # 分析时序一致性
        self._analyze_timing_consistency(captures)
        
        return captures
    
    def _analyze_timing_consistency(self, captures: List[Dict[str, Any]]):
        """分析时序一致性"""
        print(f"\n--- 时序一致性分析 ---")
        
        if len(captures) < 2:
            print("  捕获次数不足，无法分析")
            return
        
        # 检查位置一致性
        positions = [np.array(cap['agent_3d_position']) for cap in captures]
        position_vars = []
        
        for i in range(1, len(positions)):
            diff = np.linalg.norm(positions[i] - positions[0])
            position_vars.append(diff)
        
        max_position_var = max(position_vars) if position_vars else 0
        print(f"  最大位置变化: {max_position_var:.6f}m")
        
        # 检查2D坐标一致性
        map_coords = [cap['map_2d_coords'] for cap in captures]
        map_coord_vars = []
        
        for i in range(1, len(map_coords)):
            diff = np.linalg.norm(np.array(map_coords[i]) - np.array(map_coords[0]))
            map_coord_vars.append(diff)
        
        max_map_coord_var = max(map_coord_vars) if map_coord_vars else 0
        print(f"  最大2D坐标变化: {max_map_coord_var:.6f}像素")
        
        # 检查转换误差一致性
        errors = [cap['position_error'] for cap in captures]
        error_range = max(errors) - min(errors)
        print(f"  转换误差范围: {min(errors):.6f}m - {max(errors):.6f}m (差异: {error_range:.6f}m)")
        
        # 判断是否稳定
        if max_position_var < 0.001 and max_map_coord_var < 1.0 and error_range < 0.001:
            print("  ✅ 时序一致性良好")
        else:
            print("  ⚠️  时序一致性存在问题")
    
    def save_debug_data(self, data: Any, filename: str):
        """保存调试数据"""
        output_path = os.path.join(self.video_generator.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"调试数据已保存到: {output_path}")


def main():
    # 场景路径
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    # 创建调试器
    debugger = FrameSyncDebugger(scene_path)
    
    print("开始帧同步调试...")
    
    # 测试1：单帧一致性
    test_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    single_frame_data = debugger.test_single_frame_consistency(test_position)
    debugger.save_debug_data(single_frame_data, "single_frame_debug.json")
    
    # 测试2：移动序列一致性
    waypoints = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ]
    sequence_data = debugger.test_movement_sequence_consistency(waypoints)
    debugger.save_debug_data(sequence_data, "movement_sequence_debug.json")
    
    # 测试3：帧捕获时序
    timing_data = debugger.test_frame_capture_timing(test_position, 5)
    debugger.save_debug_data(timing_data, "frame_timing_debug.json")
    
    print("\n=== 调试完成 ===")


if __name__ == "__main__":
    main()
