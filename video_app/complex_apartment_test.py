#!/usr/bin/env python3
"""
apartment_1.glb 复杂导航测试脚本
测试各种复杂的导航场景，包括房间探索、路径规划、视角控制等
"""

import subprocess
import json
import time
import os
from datetime import datetime

class ApartmentNavigationTester:
    """公寓导航测试器"""
    
    def __init__(self):
        self.scene_path = "/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/apartment_1.glb"
        self.test_results = []
        self.video_count = 0
        
    def run_command_sequence(self, description: str, commands: list) -> dict:
        """运行一个命令序列并记录结果"""
        print(f"\n{'='*60}")
        print(f"测试: {description}")
        print(f"命令序列: {json.dumps(commands, ensure_ascii=False)}")
        print('='*60)
        
        # 准备输入
        commands_json = json.dumps(commands, ensure_ascii=False)
        input_data = f"{commands_json}\nexit\n"
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['python', 'main.py', '--scene', self.scene_path],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=180,  # 3分钟超时
                cwd='/home/yaoaa/habitat-lab/video_app'
            )
            
            execution_time = time.time() - start_time
            
            # 解析输出
            success = result.returncode == 0
            frames_generated = 0
            video_path = None
            error_message = None
            
            for line in result.stdout.split('\n'):
                if 'Generated' in line and 'frames' in line:
                    try:
                        frames_generated = int(line.split()[1])
                    except:
                        pass
                elif 'Video successfully saved to:' in line:
                    video_path = line.split(':', 1)[1].strip()
                    self.video_count += 1
                elif 'ERROR:' in line:
                    error_message = line
            
            if result.stderr:
                error_message = result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr
            
            test_result = {
                'description': description,
                'commands': commands,
                'success': success,
                'execution_time': execution_time,
                'frames_generated': frames_generated,
                'video_path': video_path,
                'error': error_message
            }
            
            self.test_results.append(test_result)
            
            # 显示结果
            status = "✅ 成功" if success else "❌ 失败"
            print(f"状态: {status}")
            print(f"执行时间: {execution_time:.2f}秒")
            print(f"生成帧数: {frames_generated}")
            if video_path:
                print(f"视频文件: {video_path}")
            if error_message:
                print(f"错误信息: {error_message}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"测试超时 ({execution_time:.1f}秒)"
            print(f"❌ {error_msg}")
            
            test_result = {
                'description': description,
                'commands': commands,
                'success': False,
                'execution_time': execution_time,
                'frames_generated': 0,
                'video_path': None,
                'error': error_msg
            }
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"执行异常: {str(e)}"
            print(f"❌ {error_msg}")
            
            test_result = {
                'description': description,
                'commands': commands,
                'success': False,
                'execution_time': execution_time,
                'frames_generated': 0,
                'video_path': None,
                'error': error_msg
            }
            
            self.test_results.append(test_result)
            return test_result
    
    def run_all_tests(self):
        """运行所有复杂导航测试"""
        print("开始apartment_1.glb复杂导航测试")
        print(f"场景文件: {self.scene_path}")
        print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 测试1: 基础房间探索
        self.run_command_sequence(
            "基础房间探索 - 客厅到卧室",
            [
                [0.0, 0.0],  # 起始位置
                ["right", 45],  # 转向观察
                [2.0, 1.5],  # 移动到客厅中央
                ["left", 90],  # 环顾四周
                [4.0, 3.0],  # 移动到另一个房间
                ["right", 180]  # 回头看
            ]
        )
        
        # 测试2: 复杂路径规划
        self.run_command_sequence(
            "复杂路径规划 - 穿越多个房间",
            [
                [-1.0, -1.0],  # 起始点
                [1.0, 0.0],    # 第一个转折点
                ["right", 30], # 调整视角
                [3.0, 2.0],    # 第二个转折点
                ["left", 45],  # 再次调整视角
                [0.5, 4.0],    # 第三个转折点
                ["right", 90], # 最终调整视角
                [-0.5, 2.5]    # 终点
            ]
        )
        
        # 测试3: 精细导航控制
        self.run_command_sequence(
            "精细导航控制 - 小步移动和精确旋转",
            [
                [1.0, 1.0],    # 起始位置
                ["left", 15],  # 小角度调整
                [1.2, 1.1],    # 小步移动
                ["right", 22], # 精确角度
                [1.4, 1.3],    # 再次小步移动
                ["left", 33],  # 另一个精确角度
                [1.1, 1.5]     # 最终位置
            ]
        )
        
        # 测试4: 边界探索
        self.run_command_sequence(
            "边界探索 - 测试场景边界",
            [
                [0.0, 0.0],    # 中心起始
                ["right", 90], # 朝向边界
                [5.0, 0.0],    # 向边界移动
                ["left", 180], # 回头
                [-2.0, 0.0],   # 向另一边界移动
                ["right", 90], # 调整视角
                [0.0, 5.0],    # 向Z轴正方向边界
                ["left", 90],  # 再次调整
                [0.0, -3.0]    # 向Z轴负方向边界
            ]
        )
        
        # 测试5: 高频率旋转测试
        self.run_command_sequence(
            "高频率旋转测试 - 连续旋转观察",
            [
                [2.0, 2.0],    # 中心位置
                ["right", 60], # 第一次旋转
                ["left", 120], # 反向大角度旋转
                ["right", 30], # 小角度调整
                ["left", 45],  # 再次调整
                ["right", 75], # 最后调整
                [2.1, 2.1]     # 微小移动
            ]
        )
        
        # 测试6: 房间间穿越
        self.run_command_sequence(
            "房间间穿越 - 模拟真实导航",
            [
                [-1.0, -2.0],  # 起始房间
                ["right", 45], # 观察门口
                [0.0, -1.0],   # 移向门口
                ["left", 30],  # 调整进入角度
                [1.0, 0.0],    # 进入走廊
                ["right", 60], # 寻找下一个房间
                [2.0, 1.0],    # 进入下一房间
                ["left", 90],  # 环顾房间
                [3.0, 2.0]     # 房间深处
            ]
        )
        
        # 测试7: 障碍物导航
        self.run_command_sequence(
            "障碍物导航 - 绕行测试",
            [
                [1.0, 1.0],    # 起始位置
                [2.0, 1.0],    # 直线移动
                [2.0, 2.0],    # 垂直移动（可能遇到障碍）
                [1.5, 2.5],    # 对角移动
                ["right", 45], # 观察周围
                [1.0, 3.0],    # 继续绕行
                [0.5, 2.0],    # 回到起始区域附近
                ["left", 180]  # 回头观察路径
            ]
        )
        
        # 测试8: 极限位置测试
        self.run_command_sequence(
            "极限位置测试 - 测试snap_to_navigable",
            [
                [10.0, 10.0],  # 场景外位置（应该被修正）
                [-5.0, -5.0],  # 另一个场景外位置
                [0.0, 0.0],    # 回到安全位置
                ["right", 360], # 完整旋转
                [100.0, 0.0],  # 极端X坐标
                [0.0, 100.0]   # 极端Z坐标
            ]
        )
        
        # 测试9: 混合复杂导航
        self.run_command_sequence(
            "混合复杂导航 - 综合测试",
            [
                [0.0, 0.0],    # 起始
                ["right", 45], # 初始观察
                [1.5, 0.8],    # 移动到观察点
                ["left", 30],  # 调整视角
                [2.8, 1.6],    # 长距离移动
                ["right", 90], # 大角度旋转
                [1.2, 2.4],    # 回程移动
                ["left", 60],  # 再次调整
                [0.3, 1.1],    # 精确定位
                ["right", 15], # 微调视角
                [0.1, 0.9],    # 最终微调位置
                ["left", 5]    # 最终微调视角
            ]
        )
        
        # 测试10: 快速连续导航
        self.run_command_sequence(
            "快速连续导航 - 压力测试",
            [
                [0.5, 0.5], [1.0, 0.5], [1.5, 0.5], [2.0, 0.5],
                ["right", 90],
                [2.0, 1.0], [2.0, 1.5], [2.0, 2.0], [2.0, 2.5],
                ["left", 90],
                [1.5, 2.5], [1.0, 2.5], [0.5, 2.5], [0.0, 2.5],
                ["right", 90],
                [0.0, 2.0], [0.0, 1.5], [0.0, 1.0], [0.0, 0.5],
                ["left", 90]
            ]
        )
        
        # 等待一秒，然后生成报告
        time.sleep(1)
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        print(f"\n{'='*80}")
        print("apartment_1.glb 复杂导航测试报告")
        print('='*80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - successful_tests
        
        total_time = sum(result['execution_time'] for result in self.test_results)
        total_frames = sum(result['frames_generated'] for result in self.test_results)
        
        print(f"测试总数: {total_tests}")
        print(f"成功: {successful_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        print(f"总执行时间: {total_time:.2f}秒")
        print(f"总生成帧数: {total_frames}")
        print(f"生成视频数: {self.video_count}")
        
        if total_frames > 0:
            avg_fps = total_frames / total_time
            print(f"平均渲染速度: {avg_fps:.1f} FPS")
        
        print(f"\n详细结果:")
        print("-" * 80)
        
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"{i:2d}. {status} {result['description']}")
            print(f"    时间: {result['execution_time']:.2f}s | 帧数: {result['frames_generated']} | "
                  f"命令数: {len(result['commands'])}")
            
            if result['error']:
                print(f"    错误: {result['error'][:100]}...")
            
            if result['video_path']:
                print(f"    视频: {result['video_path']}")
            print()
        
        # 保存报告到文件
        report_file = f"/home/yaoaa/habitat-lab/video_app/apartment_1_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scene': self.scene_path,
                'test_time': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests,
                    'success_rate': successful_tests/total_tests*100,
                    'total_time': total_time,
                    'total_frames': total_frames,
                    'video_count': self.video_count
                },
                'results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"详细报告已保存到: {report_file}")
        print(f"视频文件位于: /home/yaoaa/habitat-lab/video_app/outputs/")

def main():
    """主函数"""
    print("启动apartment_1.glb复杂导航测试...")
    
    tester = ApartmentNavigationTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
