#!/usr/bin/env python3
"""
测试bbox处理逻辑
"""

import os
import sys
import cv2
import numpy as np
import torch

# 添加vlm路径到sys.path
vlm_path = os.path.join(os.path.dirname(__file__), 'vlm')
sys.path.insert(0, vlm_path)

from vlm.grounding_dino import GroundingDINOClient


def test_bbox_format():
    """测试bbox格式和处理"""
    print("🔍 测试bbox格式和处理逻辑")
    print("=" * 50)
    
    # 初始化客户端
    object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
    print("✅ Grounding DINO客户端初始化成功")
    
    # 加载图像
    image_path = "/home/awangas/my_habitat-lab/habitat_video_project/data/cat_dog.jpeg"
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ 成功加载图像，尺寸: {img_rgb.shape}")
    
    # 进行检测
    caption = "cat . dog ."
    print(f"🎯 使用caption: {caption}")
    
    try:
        detections = object_detector.predict(img_rgb, caption=caption)
        print(f"✅ 检测完成，检测到 {len(detections.boxes)} 个物体")
        
        if len(detections.boxes) > 0:
            # 分析第一个检测结果的bbox
            bbox = detections.boxes[0]
            print(f"\n📊 Bbox分析:")
            print(f"  Bbox类型: {type(bbox)}")
            print(f"  Bbox形状: {bbox.shape}")
            print(f"  Bbox值: {bbox}")
            print(f"  Bbox最大值: {bbox.max()}")
            print(f"  Bbox最小值: {bbox.min()}")
            
            # 检查是否归一化
            if bbox.max() <= 1:
                print("  ✅ Bbox已归一化 (值在[0,1]范围内)")
                # 进行反归一化
                height, width = img_rgb.shape[:2]
                bbox_denorm = bbox * np.array([width, height, width, height])
                bbox_denorm = bbox_denorm.astype(int)
                print(f"  🔄 反归一化后: {bbox_denorm}")
            else:
                print("  ✅ Bbox已经是像素坐标")
                bbox_denorm = bbox.astype(int)
                print(f"  📏 像素坐标: {bbox_denorm}")
            
            # 验证bbox是否在图像范围内
            x1, y1, x2, y2 = bbox_denorm
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                print("  ✅ Bbox坐标在图像范围内")
            else:
                print("  ⚠️  Bbox坐标超出图像范围")
            
            # 测试可视化
            if hasattr(detections, 'annotated_frame') and detections.annotated_frame is not None:
                annotated_image = detections.annotated_frame
                output_path = "bbox_test_result.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                print(f"💾 可视化结果已保存为: {output_path}")
            else:
                print("⚠️  未找到标注图像")
                
        else:
            print("❌ 未检测到任何物体")
            
    except Exception as e:
        print(f"❌ 检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_bbox_format() 