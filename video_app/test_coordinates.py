#!/usr/bin/env python3
"""
坐标精度验证脚本
"""

import subprocess
import sys
import os

def test_coordinate_accuracy():
    """测试坐标精度"""
    print("Testing coordinate accuracy...")
    
    # 添加项目路径
    project_path = '/home/yaoaa/habitat-lab/video_app'
    sys.path.append(os.path.join(project_path, 'src'))
    
    # 导入模块
    from habitat_video_generator import HabitatVideoGenerator
    
    try:
        # 初始化生成器
        generator = HabitatVideoGenerator(
            scene_filepath="/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            gpu_device_id=0
        )
        
        # 测试几个已知坐标点
        test_points = [
            (2.6, 0.1),
            (3.0, 0.5),
            (1.5, -0.5),
            (0.0, 0.0)
        ]
        
        print("场景边界:", generator.simulator.scene_bounds)
        print("原始地图尺寸:", generator.simulator.base_map_image.size)
        print()
        
        for x, z in test_points:
            # 检查位置是否可导航
            navigable = generator.simulator.is_navigable(x, z)
            
            if navigable:
                # 移动到位置
                snapped_pos = generator.simulator.snap_to_navigable(x, z)
                if snapped_pos is not None:
                    generator.simulator.move_agent_to(snapped_pos)
                    
                    # 获取实际位置
                    actual_pos = generator.get_agent_position()
                    
                    # 计算地图坐标
                    map_coords = generator.simulator.world_to_map_coords(snapped_pos)
                    
                    print(f"目标: ({x:.2f}, {z:.2f})")
                    print(f"实际: ({actual_pos[0]:.3f}, {actual_pos[2]:.3f})")
                    print(f"地图坐标: {map_coords}")
                    print(f"距离差: {((actual_pos[0] - x)**2 + (actual_pos[2] - z)**2)**0.5:.3f}m")
                    print("-" * 40)
                else:
                    print(f"坐标 ({x}, {z}) 无法对齐到可导航位置")
            else:
                print(f"坐标 ({x}, {z}) 不可导航")
        
        generator.close()
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coordinate_accuracy()
