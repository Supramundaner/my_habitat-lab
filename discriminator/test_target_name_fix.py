#!/usr/bin/env python3
"""
测试目标对象名称获取修复

验证 extract_controversial.py 是否正确从 action.json 中获取目标对象名称
"""

import json
import tempfile
from pathlib import Path
import sys
import os

# Add discriminator directory to path
sys.path.append('/home/yaoaa/habitat-lab/discriminator')

def create_test_data():
    """创建测试数据结构"""
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    
    # Model 1 目录结构
    model1_dir = temp_dir / "model1"
    scene_dir = model1_dir / "test_scene"
    episode_dir = scene_dir / "123"
    preprocess_dir = episode_dir / "preprocess"
    
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 action.json 文件 (正确的目标对象名称)
    action_data = {
        "agent_state": {
            "position": [7.18678, 2.06447, 4.88622],
            "rotation": [0, 0.23418, 0, 0.97219]
        },
        "target_info": {
            "coordinate": [1.056197949185337, 0.4913067344680666],
            "name": "bed"  # 这是正确的目标对象名称
        },
        "wall_mask": "/path/to/wall_mask.png"
    }
    
    with open(preprocess_dir / "action.json", 'w') as f:
        json.dump(action_data, f, indent=2)
    
    # 创建 output.json 文件 (可能有错误的目标对象名称)
    output_data = {
        "object_category": "unknown",  # 这是错误的或者缺失的
        "evaluation_results": {
            "sr": True,
            "spl": 0.75,
            "success": True,
            "geodesic_distance_to_target": 2.5,
            "path_length": 3.0
        }
    }
    
    with open(episode_dir / "output.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return temp_dir, model1_dir

def test_target_name_extraction():
    """测试目标对象名称提取"""
    print("🧪 测试目标对象名称提取...")
    
    try:
        # 创建测试数据
        temp_dir, model1_dir = create_test_data()
        print(f"创建测试数据在: {temp_dir}")
        
        # 创建配置
        config = {
            'model_paths': {
                'model1_output': str(model1_dir),
                'model2_output': str(model1_dir)  # 使用相同目录进行测试
            },
            'output_config': {
                'discriminator_output': str(temp_dir / 'output')
            }
        }
        
        # 导入并测试
        from extract_controversial import ControversyExtractor
        
        extractor = ControversyExtractor(config)
        results = extractor.load_batch_results(model1_dir)
        
        # 验证结果
        episode_key = "test_scene/123"
        if episode_key in results:
            object_category = results[episode_key]['object_category']
            print(f"提取的目标对象名称: '{object_category}'")
            
            if object_category == "bed":
                print("✅ 成功: 正确从 action.json 中获取了目标对象名称 'bed'")
                success = True
            else:
                print(f"❌ 失败: 期望 'bed', 但得到 '{object_category}'")
                success = False
        else:
            print(f"❌ 失败: 没有找到 episode {episode_key}")
            success = False
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_fallback_mechanism():
    """测试回退机制 (当action.json不存在时)"""
    print("\n🧪 测试回退机制...")
    
    try:
        # 创建测试数据但没有action.json
        temp_dir = Path(tempfile.mkdtemp())
        model1_dir = temp_dir / "model1"
        scene_dir = model1_dir / "test_scene"
        episode_dir = scene_dir / "456"
        
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 只创建 output.json 文件
        output_data = {
            "object_category": "chair",  # 应该回退到这个值
            "evaluation_results": {
                "sr": False,
                "spl": 0.0,
                "success": False,
                "geodesic_distance_to_target": 10.0,
                "path_length": 15.0
            }
        }
        
        with open(episode_dir / "output.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # 创建配置
        config = {
            'model_paths': {
                'model1_output': str(model1_dir),
                'model2_output': str(model1_dir)
            },
            'output_config': {
                'discriminator_output': str(temp_dir / 'output')
            }
        }
        
        # 测试
        from extract_controversial import ControversyExtractor
        
        extractor = ControversyExtractor(config)
        results = extractor.load_batch_results(model1_dir)
        
        # 验证结果
        episode_key = "test_scene/456"
        if episode_key in results:
            object_category = results[episode_key]['object_category']
            print(f"回退机制提取的目标对象名称: '{object_category}'")
            
            if object_category == "chair":
                print("✅ 成功: 回退机制正常工作，从 output.json 中获取了目标对象名称")
                success = True
            else:
                print(f"❌ 失败: 期望 'chair', 但得到 '{object_category}'")
                success = False
        else:
            print(f"❌ 失败: 没有找到 episode {episode_key}")
            success = False
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("=" * 80)
    print("测试目标对象名称获取修复")
    print("=" * 80)
    
    # 运行测试
    test1_passed = test_target_name_extraction()
    test2_passed = test_fallback_mechanism()
    
    print("\n" + "=" * 80)
    print("测试结果总结:")
    print(f"✅ 从 action.json 获取目标名称: {'通过' if test1_passed else '失败'}")
    print(f"✅ 回退机制测试: {'通过' if test2_passed else '失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！修改已成功完成。")
        print("\n📝 主要改动：")
        print("1. extract_controversial.py 现在优先从 action.json 中获取目标对象名称")
        print("2. 当 action.json 不存在时，回退到 output.json 中的 object_category")
        print("3. 这确保了生成的 controversial_episodes.json 包含正确的目标对象名称")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查修改。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
