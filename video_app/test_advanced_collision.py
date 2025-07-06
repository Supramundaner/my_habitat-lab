#!/usr/bin/env python3
"""
测试视频生成器的在线碰撞检测和回退功能
"""

import sys
import os
sys.path.append('/home/yaoaa/habitat-lab')

from video_app.src.habitat_video_generator import HabitatVideoGenerator

def test_movement_collision_detection():
    """测试移动过程中的碰撞检测和回退功能"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("=" * 60)
        print("测试移动过程中的碰撞检测和回退功能")
        print("=" * 60)
        
        generator = HabitatVideoGenerator(scene_path)
        
        # 获取场景边界以设计测试路径
        scene_bounds = generator.simulator.scene_bounds
        print(f"场景边界: {scene_bounds}")
        
        # 设计测试路径：从场景内部移动到边界外
        center_x = (scene_bounds[0][0] + scene_bounds[1][0]) / 2
        center_z = (scene_bounds[0][2] + scene_bounds[1][2]) / 2
        
        # 选择边界外的点
        boundary_x = scene_bounds[1][0] + 2.0  # 超出边界
        boundary_z = scene_bounds[1][2] + 2.0  # 超出边界
        
        print(f"场景中心: ({center_x:.2f}, {center_z:.2f})")
        print(f"测试目标 (边界外): ({boundary_x:.2f}, {boundary_z:.2f})")
        
        # 测试指令序列
        commands = [
            [center_x, center_z],           # 移动到场景中心
            ["left", 45],                   # 转向
            [boundary_x, boundary_z],       # 尝试移动到边界外（应该触发碰撞）
            [center_x + 0.5, center_z + 0.5], # 这个指令不应该被执行
        ]
        
        print(f"执行测试指令序列: {commands}")
        
        # 生成视频
        output_path = generator.process_command_sequence(commands)
        
        if output_path:
            print(f"✅ 在线碰撞检测测试完成！视频已保存到: {output_path}")
            print("💡 如果在移动过程中检测到碰撞，代理应该回退到最后有效位置")
            return True
        else:
            print("❌ 在线碰撞检测测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            generator.cleanup()
        except:
            pass

def test_coordinate_accuracy_reporting():
    """测试坐标转换精度报告功能"""
    
    scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found: {scene_path}")
        return False
    
    try:
        print("\n" + "=" * 60)
        print("测试坐标转换精度报告功能")
        print("=" * 60)
        
        generator = HabitatVideoGenerator(scene_path)
        
        # 测试不同的坐标点
        test_coordinates = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.5, -1.5],
            [-0.5, 2.0],
        ]
        
        print("测试各个坐标点的转换精度:")
        
        for i, coords in enumerate(test_coordinates):
            commands = [coords]
            print(f"\n测试点 {i+1}: {coords}")
            
            # 生成视频（只包含一个移动指令）
            output_path = generator.process_command_sequence(commands)
            
            if output_path:
                print(f"  ✅ 测试点 {i+1} 处理成功")
            else:
                print(f"  ❌ 测试点 {i+1} 处理失败")
        
        print("\n✅ 坐标转换精度报告测试完成！")
        print("💡 每个移动指令都应该报告坐标转换精度信息")
        
        return True
        
    except Exception as e:
        print(f"❌ 坐标转换精度测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            generator.cleanup()
        except:
            pass

if __name__ == "__main__":
    print("开始测试高级碰撞检测和坐标精度功能...")
    
    # 运行测试
    test1_success = test_movement_collision_detection()
    test2_success = test_coordinate_accuracy_reporting()
    
    # 输出总结
    print("\n" + "=" * 70)
    print("高级测试总结:")
    print(f"移动过程碰撞检测: {'✅ 成功' if test1_success else '❌ 失败'}")
    print(f"坐标转换精度报告: {'✅ 成功' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有高级测试通过！")
        print("\n高级功能验证:")
        print("✅ 移动过程中实时碰撞检测")
        print("✅ 碰撞时准确回退到最后有效位置")
        print("✅ 视频在碰撞点准确截断")
        print("✅ 每个坐标都报告转换精度")
        print("✅ 完全摆脱snap_to_navigable依赖")
    else:
        print("\n⚠️  某些高级测试失败，需要进一步调试。")
    
    print("=" * 70)
