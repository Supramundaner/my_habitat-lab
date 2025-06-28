#!/usr/bin/env python3
"""
测试指北针绘制是否正确
"""

from PIL import Image, ImageDraw, ImageFont
import os

def test_compass_drawing():
    """测试指北针绘制"""
    # 创建测试图像
    width, height = 400, 300
    image = Image.new('RGB', (width, height), (0, 0, 0))  # 黑色背景
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_small = ImageFont.load_default()
    
    # 颜色定义
    border_color = (255, 255, 255)   # 白色边框
    
    # 指北针位置
    compass_x = width - 60
    compass_y = 50
    
    # 指北针背景
    draw.rectangle([compass_x-25, compass_y-25, compass_x+25, compass_y+25], 
                  outline=border_color, width=1, fill=(50, 50, 50))
    
    # 绘制指北针箭头 - 修复后的版本
    # X轴（红色）：水平向右
    draw.line([(compass_x-15, compass_y), (compass_x+15, compass_y)], fill=(255, 0, 0), width=3)
    # 箭头头部
    draw.line([(compass_x+15, compass_y), (compass_x+10, compass_y-5)], fill=(255, 0, 0), width=2)
    draw.line([(compass_x+15, compass_y), (compass_x+10, compass_y+5)], fill=(255, 0, 0), width=2)
    
    # Z轴（绿色）：垂直向下
    draw.line([(compass_x, compass_y-15), (compass_x, compass_y+15)], fill=(0, 255, 0), width=3)
    # 箭头头部
    draw.line([(compass_x, compass_y+15), (compass_x-5, compass_y+10)], fill=(0, 255, 0), width=2)
    draw.line([(compass_x, compass_y+15), (compass_x+5, compass_y+10)], fill=(0, 255, 0), width=2)
    
    # 指北针标签
    draw.text((compass_x+18, compass_y-5), "+X", fill=(255, 0, 0), font=font_small)
    draw.text((compass_x-8, compass_y+18), "+Z", fill=(0, 255, 0), font=font_small)
    
    # 添加说明文字
    draw.text((10, 10), "修复后的指北针:", fill=(255, 255, 255), font=font_small)
    draw.text((10, 30), "红色箭头 (+X): 向右", fill=(255, 0, 0), font=font_small)
    draw.text((10, 50), "绿色箭头 (+Z): 向下", fill=(0, 255, 0), font=font_small)
    draw.text((10, 80), "符合Habitat坐标系:", fill=(255, 255, 255), font=font_small)
    draw.text((10, 100), "X向右, Y向上, Z向后(-Z向前)", fill=(255, 255, 255), font=font_small)
    
    # 绘制原始版本（错误的）用于对比
    compass_x_old = 100
    compass_y_old = 200
    
    # 原始指北针背景
    draw.rectangle([compass_x_old-25, compass_y_old-25, compass_x_old+25, compass_y_old+25], 
                  outline=border_color, width=1, fill=(50, 50, 50))
    
    # 原始版本（错误的）
    draw.line([(compass_x_old, compass_y_old-15), (compass_x_old, compass_y_old+15)], fill=(255, 0, 0), width=2)
    draw.line([(compass_x_old-15, compass_y_old), (compass_x_old+15, compass_y_old)], fill=(0, 255, 0), width=2)
    
    # 原始标签
    draw.text((compass_x_old+18, compass_y_old-15), "+X", fill=(255, 0, 0), font=font_small)
    draw.text((compass_x_old-8, compass_y_old+18), "+Z", fill=(0, 255, 0), font=font_small)
    
    # 说明
    draw.text((10, 150), "原始版本（错误）:", fill=(255, 255, 255), font=font_small)
    draw.text((10, 170), "红色是垂直的（应该是水平）", fill=(255, 255, 255), font=font_small)
    draw.text((10, 190), "绿色是水平的（应该是垂直）", fill=(255, 255, 255), font=font_small)
    
    return image

def main():
    """主函数"""
    print("生成指北针测试图像...")
    
    # 生成测试图像
    test_image = test_compass_drawing()
    
    # 保存图像
    output_path = "/home/yaoaa/habitat-lab/compass_test.png"
    test_image.save(output_path)
    
    print(f"测试图像已保存到: {output_path}")
    print("\n指北针修复说明:")
    print("1. 修复前: 红色垂直线标记+X（错误），绿色水平线标记+Z（错误）")
    print("2. 修复后: 红色水平箭头指向右侧标记+X（正确），绿色垂直箭头指向下方标记+Z（正确）")
    print("3. 符合Habitat坐标系: X向右, Y向上, Z向后(-Z向前)")
    print("4. 在2D地图中: X轴水平向右, Z轴垂直向下")

if __name__ == "__main__":
    main()
