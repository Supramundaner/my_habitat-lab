#!/usr/bin/env python3
"""
测试ObjectDetector的detect_object函数
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


def save_mask_visualization(mask, original_image, output_path):
    """保存mask可视化结果"""
    # 创建彩色mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask > 0] = [0, 255, 0]  # 绿色表示mask区域
    
    # 将mask叠加到原图上
    alpha = 0.5
    result = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
    
    # 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"💾 Mask可视化已保存为: {output_path}")


def test_object_detector():
    """测试ObjectDetector的detect_object函数"""
    print("🚀 测试ObjectDetector的detect_object函数")
    print("=" * 60)
    
    # 初始化ObjectDetector
    config = {
        'grounding_dino_port': 12181,
        'mobile_sam_port': 12184,
        'confidence_threshold': 0.35
    }
    
    object_detector = ObjectDetector(config)
    print("✅ ObjectDetector初始化成功")
    
    # 加载图像
    image_path = "/home/awangas/my_habitat-lab/habitat_video_project/data/window.png"
    rgb_image = load_image(image_path)
    
    if rgb_image is None:
        print("❌ 图像加载失败，退出")
        return
    
    # 设置目标物体
    target_object = "dog"
    print(f"🎯 目标物体: {target_object}")
    
    try:
        # 调用detect_object函数
        print("🔍 开始对象检测和分割...")
        result = object_detector.detect_object(rgb_image,target_object)
        
        if result is not None:
            object_mask, bbox = result
            print(f"✅ 检测成功!")
            print(f"📦 边界框: {bbox}")
            print(f"🎭 Mask形状: {object_mask.shape}")
            print(f"🎭 Mask中True像素数量: {np.sum(object_mask)}")
            
            # 保存bbox可视化（使用detections.py的逻辑）
            if hasattr(object_detector, 'grounding_dino'):
                # 重新获取detections用于可视化
                detections = object_detector.grounding_dino.predict(rgb_image, f"{target_object} .")
                
                if hasattr(detections, 'annotated_frame') and detections.annotated_frame is not None:
                    annotated_image = detections.annotated_frame
                    bbox_output_path = "object_detector_bbox_result.jpg"
                    cv2.imwrite(bbox_output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    print(f"💾 Bbox可视化已保存为: {bbox_output_path}")
                else:
                    print("⚠️  未找到bbox标注图像")
            
            # 保存mask可视化
            mask_output_path = "object_detector_mask_result.jpg"
            save_mask_visualization(object_mask, rgb_image, mask_output_path)
            
            # 创建组合可视化（bbox + mask）
            if hasattr(object_detector, 'grounding_dino'):
                detections = object_detector.grounding_dino.predict(rgb_image, f"{target_object} .")
                if hasattr(detections, 'annotated_frame') and detections.annotated_frame is not None:
                    combined_image = detections.annotated_frame.copy()
                    
                    # 在bbox可视化基础上叠加mask
                    colored_mask = np.zeros_like(combined_image)
                    colored_mask[object_mask > 0] = [0, 255, 0]  # 绿色
                    
                    alpha = 0.3
                    combined_result = cv2.addWeighted(combined_image, 1-alpha, colored_mask, alpha, 0)
                    
                    combined_output_path = "object_detector_combined_result.jpg"
                    cv2.imwrite(combined_output_path, cv2.cvtColor(combined_result, cv2.COLOR_RGB2BGR))
                    print(f"💾 组合可视化已保存为: {combined_output_path}")
            
        else:
            print("❌ 检测失败，未找到目标物体")
            
    except Exception as e:
        print(f"❌ 检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_objects():
    """测试多个物体的检测"""
    print("\n🔍 测试多个物体的检测")
    print("=" * 60)
    
    # 初始化ObjectDetector
    config = {
        'grounding_dino_port': 12181,
        'mobile_sam_port': 12184,
        'confidence_threshold': 0.1
    }
    
    object_detector = ObjectDetector(config)
    
    # 加载图像
    image_path = "/home/awangas/my_habitat-lab/habitat_video_project/data/window.png"
    rgb_image = load_image(image_path)
    
    if rgb_image is None:
        return
    
    # 测试多个目标物体
    target_objects = ["window"]
    
    for target_object in target_objects:
        print(f"\n🎯 检测目标: {target_object}")
        
        try:
            result = object_detector.detect_object(rgb_image, target_object)
            
            if result is not None:
                object_mask, bbox = result
                print(f"  ✅ 检测成功!")
                print(f"  📦 边界框: {bbox}")
                print(f"  🎭 Mask中True像素数量: {np.sum(object_mask)}")
                
                # 保存mask可视化
                mask_output_path = f"object_detector_{target_object}_mask.jpg"
                save_mask_visualization(object_mask, rgb_image, mask_output_path)
                
            else:
                print(f"  ❌ 未检测到 {target_object}")
                
        except Exception as e:
            print(f"  ❌ 检测 {target_object} 时出现错误: {e}")


def main():
    """主函数"""
    print("🚀 ObjectDetector测试开始")
    print("=" * 80)
    
    # 测试单个物体检测
    # test_object_detector()
    
    # 测试多个物体检测
    test_multiple_objects()
    
    print("\n" + "=" * 80)
    print("📋 测试完成!")
    print("💡 请查看生成的图像文件来验证检测和分割效果")


if __name__ == "__main__":
    main() 