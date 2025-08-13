#!/usr/bin/env python3
"""
调试检测逻辑
"""

import os
import sys
import cv2
import numpy as np

# 添加src路径到sys.path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# 添加vlm路径到sys.path
vlm_path = os.path.join(os.path.dirname(__file__), 'vlm')
sys.path.insert(0, vlm_path)

from object_detector import ObjectDetector


def debug_detection():
    """调试检测逻辑"""
    print("🔍 调试检测逻辑")
    print("=" * 50)
    
    # 初始化ObjectDetector
    config = {
        'grounding_dino_port': 12181,
        'mobile_sam_port': 12184,
        'confidence_threshold': 0.35
    }
    
    object_detector = ObjectDetector(config)
    
    # 加载图像
    image_path = "/home/awangas/my_habitat-lab/habitat_video_project/data/cat_dog.jpeg"
    img = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"📐 图像尺寸: {rgb_image.shape}")
    
    # 测试不同的caption格式
    test_captions = [
        "cat .",
        "dog .", 
        "cat . dog .",
        "dog . cat ."
    ]
    
    for caption in test_captions:
        print(f"\n🎯 测试caption: '{caption}'")
        print("-" * 30)
        
        try:
            # 直接调用Grounding DINO
            detections = object_detector.grounding_dino.predict(rgb_image, caption)
            
            print(f"检测到 {detections.num_detections} 个物体")
            
            if detections.num_detections > 0:
                for i, (box, score, phrase) in enumerate(zip(detections.boxes, detections.logits, detections.phrases)):
                    print(f"  物体 {i+1}: {phrase} (置信度: {score:.3f})")
                    print(f"    边界框: {box.tolist()}")
                    
                    # 检查bbox是否归一化
                    if hasattr(box, 'cpu'):
                        box_np = box.cpu().numpy()
                    else:
                        box_np = box
                    
                    if box_np.max() <= 1:
                        print(f"    ✅ 已归一化")
                        height, width = rgb_image.shape[:2]
                        box_denorm = box_np * np.array([width, height, width, height])
                        print(f"    反归一化后: {box_denorm.astype(int)}")
                    else:
                        print(f"    ✅ 已经是像素坐标")
            else:
                print("  ❌ 未检测到任何物体")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    # 测试detect_object函数
    print(f"\n🔍 测试detect_object函数")
    print("=" * 50)
    
    target_objects = ["cat", "dog"]
    
    for target_object in target_objects:
        print(f"\n🎯 目标物体: {target_object}")
        
        try:
            # 先查看Grounding DINO的原始检测结果
            caption = f"{target_object} ."
            detections = object_detector.grounding_dino.predict(rgb_image, caption)
            print(f"  📊 Grounding DINO检测结果:")
            print(f"    检测到 {detections.num_detections} 个物体")
            for i, (box, score, phrase) in enumerate(zip(detections.boxes, detections.logits, detections.phrases)):
                print(f"    物体 {i+1}: {phrase} (置信度: {score:.3f})")
                if hasattr(box, 'cpu'):
                    box_np = box.cpu().numpy()
                else:
                    box_np = box
                if box_np.max() <= 1:
                    height, width = rgb_image.shape[:2]
                    box_denorm = box_np * np.array([width, height, width, height])
                    print(f"      边界框: {box_denorm.astype(int)}")
                else:
                    print(f"      边界框: {box_np.astype(int)}")
            
            # 然后测试detect_object函数
            result = object_detector.detect_object(rgb_image, target_object)
            
            if result is not None:
                object_mask, bbox = result
                print(f"  ✅ detect_object成功!")
                print(f"  📦 最终边界框: {bbox}")
                print(f"  🎭 Mask中True像素数量: {np.sum(object_mask)}")
                
                # 保存mask可视化
                colored_mask = np.zeros_like(rgb_image)
                colored_mask[object_mask > 0] = [0, 255, 0]
                alpha = 0.5
                result_img = cv2.addWeighted(rgb_image, 1-alpha, colored_mask, alpha, 0)
                
                output_path = f"debug_{target_object}_mask.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                print(f"  💾 保存为: {output_path}")
                
            else:
                print(f"  ❌ detect_object未检测到 {target_object}")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    debug_detection() 