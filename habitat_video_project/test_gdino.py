#!/usr/bin/env python3
"""
专门用于可视化的测试脚本
"""

import os
import sys
import cv2
import numpy as np

# 添加vlm路径到sys.path
vlm_path = os.path.join(os.path.dirname(__file__), 'vlm')
sys.path.insert(0, vlm_path)

# 导入相关模块
from vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlm.sam import MobileSAMClient


def load_image(image_path):
    """加载图像"""
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    
    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"✅ 成功加载图像: {image_path}")
    print(f"📐 图像尺寸: {img_rgb.shape}")
    
    return img_rgb


def main():
    """主函数"""
    print("🚀 开始可视化测试")
    print("=" * 50)
    
    # 初始化对象检测器
    object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
    print("✅ Grounding DINO客户端初始化成功")
    
    # 加载图像
    image_path = "/home/awangas/my_habitat-lab/habitat_video_project/data/cat_dog.jpeg"
    img = load_image(image_path)
    
    if img is None:
        print("❌ 图像加载失败，退出")
        return
    
    # 设置caption
    caption = "cat . dog ."
    print(f"🎯 使用caption: {caption}")
    
    try:
        # 进行检测
        print("🔍 开始对象检测...")
        detections = object_detector.predict(img, caption=caption)
        
        print(f"✅ 检测完成，检测到 {len(detections.boxes)} 个物体")
        
        # 显示检测结果
        for i, (box, score, phrase) in enumerate(zip(detections.boxes, detections.logits, detections.phrases)):
            print(f"  🎯 物体 {i+1}: {phrase} (置信度: {score:.3f})")
            print(f"     边界框: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        # 获取标注图像
        if hasattr(detections, 'annotated_frame') and detections.annotated_frame is not None:
            annotated_image = detections.annotated_frame
            
            # 保存标注图像
            output_path = "visualization_result.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"💾 标注图像已保存为: {output_path}")
            print(f"📐 输出图像尺寸: {annotated_image.shape}")
        else:
            print("⚠️  未找到标注图像，可能需要手动创建可视化")
            
    except Exception as e:
        print(f"❌ 检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
