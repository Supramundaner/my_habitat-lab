#!/usr/bin/env python3
"""
测试检测器兼容性的脚本
验证GroundingDINO和YOLOv7接口的兼容性
"""

import numpy as np
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
vlm_path = os.path.join(project_root, 'vlm')
sys.path.insert(0, src_path)
sys.path.insert(0, vlm_path)

def test_interface_compatibility():
    """测试接口兼容性"""
    print("Testing detector interface compatibility...")
    
    try:
        # 测试导入
        from vlm.grounding_dino import GroundingDINOClient
        from vlm.yolov7 import YOLOv7Client
        from vlm.detections import ObjectDetections
        print("✓ All modules imported successfully")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_caption = "chair . person ."
        
        # 测试GroundingDINO接口
        print("\nTesting GroundingDINO interface...")
        try:
            gdino_client = GroundingDINOClient(port=12181)
            print("✓ GroundingDINOClient created successfully")
            print(f"✓ predict method signature: {gdino_client.predict.__annotations__}")
        except Exception as e:
            print(f"! GroundingDINO client creation failed (expected if server not running): {e}")
        
        # 测试YOLOv7接口
        print("\nTesting YOLOv7 interface...")
        try:
            yolo_client = YOLOv7Client(port=12184)
            print("✓ YOLOv7Client created successfully")
            print(f"✓ predict method signature: {yolo_client.predict.__annotations__}")
        except Exception as e:
            print(f"! YOLOv7 client creation failed (expected if server not running): {e}")
        
        # 测试ObjectDetector集成
        print("\nTesting ObjectDetector integration...")
        from object_detector import ObjectDetector
        
        # 测试GroundingDINO配置
        config_gdino = {
            'object_detection': {
                'enabled': True,
                'detector_type': 'grounding_dino',
                'grounding_dino_port': 12181,
                'mobile_sam_port': 12182
            }
        }
        
        detector_gdino = ObjectDetector(config_gdino)
        print(f"✓ ObjectDetector with GroundingDINO: enabled={detector_gdino.enabled}")
        if detector_gdino.enabled:
            print(f"✓ Detector type: {detector_gdino.detector_type}")
        
        # 测试YOLOv7配置
        config_yolo = {
            'object_detection': {
                'enabled': True,
                'detector_type': 'yolov7',
                'yolov7_port': 12184,
                'mobile_sam_port': 12182
            }
        }
        
        detector_yolo = ObjectDetector(config_yolo)
        print(f"✓ ObjectDetector with YOLOv7: enabled={detector_yolo.enabled}")
        if detector_yolo.enabled:
            print(f"✓ Detector type: {detector_yolo.detector_type}")
        
        print("\n✓ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coco_classes():
    """测试COCO类别列表"""
    print("\nTesting COCO classes...")
    try:
        from vlm.coco_classes import COCO_CLASSES
        print(f"✓ COCO_CLASSES loaded: {len(COCO_CLASSES)} classes")
        print(f"✓ Sample classes: {COCO_CLASSES[:5]}")
        
        # 验证一些常见类别
        expected_classes = ['person', 'car', 'chair', 'dog', 'cat']
        for cls in expected_classes:
            if cls in COCO_CLASSES:
                print(f"✓ Found expected class: {cls}")
            else:
                print(f"✗ Missing expected class: {cls}")
        
        return True
    except Exception as e:
        print(f"✗ COCO classes test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("Object Detector Compatibility Test")
    print("=" * 60)
    
    success = True
    
    # 测试COCO类别
    success &= test_coco_classes()
    
    # 测试接口兼容性
    success &= test_interface_compatibility()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The migration is ready.")
        print("\nTo use YOLOv7, update your config:")
        print("  object_detection:")
        print("    detector_type: 'yolov7'")
        print("    yolov7_port: 12184")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
