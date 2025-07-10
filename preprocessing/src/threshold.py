import cv2
import numpy as np
import os

def threshold(image_path, output_mask_path, threshold_value=5):

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 根据图像通道数转换为灰度图
    if len(img.shape) == 3:
        if img.shape[2] == 4: # RGBA 图像 (4通道)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else: # BGR 图像 (3通道)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: # 已经是灰度图 (2维度)
        gray_img = img

    # 应用二值化阈值处理:
    # cv2.THRESH_BINARY:
    #   如果像素值 <= threshold_value，则设置为 0 (黑色)
    #   如果像素值 > threshold_value，则设置为 255 (白色)
    # 这样，原始图像中的黑色/暗部分在掩码中仍然是黑色。
    _, mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_mask_path, mask)
    
    return mask

# --- 示例用法 ---
if __name__ == "__main__":
    # 确保你有一个名为 'input_image.png' 的图片文件在脚本所在的目录下
    # 或者提供一个完整的图片路径
    input_image_name = "/home/awangas/my_habitat-lab/preprocessing/data/top_down/sample_2.png"
    output_mask_name = "/home/awangas/my_habitat-lab/preprocessing/data/processed/sample_2/threshold.png"


    # 调用函数生成掩码
    threshold(input_image_name, output_mask_name)
