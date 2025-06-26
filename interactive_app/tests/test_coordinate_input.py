#!/usr/bin/env python3
"""
测试坐标输入的完整流程，模拟GUI操作
"""

def test_coordinate_input_flow():
    """测试坐标输入的完整流程"""
    print("测试坐标输入流程...")
    
    try:
        # 导入模拟器类
        from habitat_navigator_app import HabitatSimulator
        
        scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
        
        print("1. 创建模拟器...")
        sim = HabitatSimulator(scene_path)
        
        print(f"✓ 场景边界: {sim.scene_bounds}")
        print(f"✓ 场景中心: {sim.scene_center}")
        
        # 测试坐标输入 "2.6, 0.1"
        print("\n2. 测试坐标输入 '2.6, 0.1'...")
        
        # 模拟GUI的process_coordinate_command逻辑
        command = "2.6, 0.1"
        parts = command.split(',')
        x = float(parts[0].strip())
        z = float(parts[1].strip())
        
        print(f"   解析坐标: x={x}, z={z}")
        
        # 检查是否可导航
        print("3. 检查导航可达性...")
        is_navigable = sim.is_navigable(x, z)
        print(f"   可导航: {is_navigable}")
        
        if is_navigable:
            # 获取对齐的3D点
            print("4. 获取对齐的3D点...")
            target_pos = sim.snap_to_navigable(x, z)
            print(f"   对齐位置: {target_pos}")
            
            if target_pos is not None:
                # 获取当前智能体状态
                print("5. 获取当前智能体状态...")
                current_state = sim.get_agent_state()
                current_pos = current_state.position
                print(f"   当前位置: {current_pos}")
                
                # 检查是否是第一次设置位置
                scene_center = sim.scene_center
                is_first_placement = (
                    np.allclose(current_pos, [0, 0, 0], atol=0.1) or
                    np.allclose(current_pos, scene_center, atol=0.5) or
                    np.linalg.norm(current_pos) < 0.1
                )
                print(f"   首次放置: {is_first_placement}")
                
                # 移动智能体
                print("6. 移动智能体...")
                sim.move_agent_to(target_pos)
                
                # 获取FPV图像
                print("7. 获取FPV图像...")
                fpv_img = sim.get_fpv_observation()
                print(f"   FPV图像尺寸: {fpv_img.shape}")
                
                # 保存测试图像
                from PIL import Image
                fpv_image = Image.fromarray(fpv_img[..., :3], "RGB")
                fpv_image.save("test_coordinate_input_fpv.png")
                print("   ✓ FPV图像已保存: test_coordinate_input_fpv.png")
                
                # 测试地图更新（模拟update_map_display）
                print("8. 测试地图更新...")
                if sim.base_map_image:
                    # 复制基础地图
                    map_image = sim.base_map_image.copy()
                    
                    # 绘制智能体标记
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(map_image)
                    
                    # 转换世界坐标到地图坐标
                    map_x, map_y = sim.world_to_map_coords(target_pos)
                    print(f"   地图坐标: ({map_x}, {map_y})")
                    
                    # 绘制智能体位置（红点）
                    dot_radius = 8
                    draw.ellipse([
                        map_x - dot_radius, map_y - dot_radius,
                        map_x + dot_radius, map_y + dot_radius
                    ], fill=(255, 0, 0))
                    
                    # 保存带智能体标记的地图
                    map_image.save("test_coordinate_input_map.png")
                    print("   ✓ 带智能体标记的地图已保存: test_coordinate_input_map.png")
        
        # 清理
        sim.close()
        print("\n✓ 坐标输入流程测试完成")
        return True
        
    except Exception as e:
        print(f"\n✗ 坐标输入流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import numpy as np
    test_coordinate_input_flow()
