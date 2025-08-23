# Batch Sample Generation - Visualization Features

## 新增功能概述

基于 `visualize_view_points.py` 的功能，我为 `batch_sample_generation.py` 添加了强大的可视化能力。

## 核心功能

### 1. Viewpoint可视化
- **Object位置标记**: 用金色圆圈标记object的确切位置
- **Viewpoint分布**: 用彩色圆点显示所有可能的观察位置
- **IoU颜色编码**: 根据IoU值显示不同颜色（蓝色=低，红色=高）
- **智能图例**: 显示object信息、viewpoint数量和IoU范围

### 2. 自动集成
- 在房间分割成功后自动创建可视化
- 使用metadata中的坐标变换信息
- 保存为 `viewpoint_visualization.png`

### 3. 错误处理
- 优雅处理缺失的依赖
- 在可视化失败时不影响主流程
- 详细的状态反馈

## 技术实现

### 关键函数

1. **`extract_viewpoint_positions(obj)`**
   - 从object数据中提取viewpoint位置和IoU
   - 支持多种数据格式
   - 返回 `(x, y, z, iou)` 元组列表

2. **`world_to_pixel(x, z, coords)`**
   - 将3D世界坐标投影到2D图像像素
   - 使用render_topdown_view返回的坐标信息

3. **`create_viewpoint_visualization(obj, topdown_path, coords, output_path)`**
   - 创建完整的可视化图像
   - 在俯视图上绘制object和viewpoints
   - 添加图例和颜色编码

### 依赖管理
```python
# 动态导入避免启动时的依赖错误
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
```

## 输出示例

### 文件结构
```
sample_01_bench_bench_1/
├── topdown_view.png              # 原始俯视图
├── room_annotation.png           # 房间标注
├── viewpoint_visualization.png   # 新增：可视化图像
└── ...
```

### 可视化内容
- **金色圆圈**: Object位置 (bench_1)
- **彩色圆点**: 38个viewpoints，颜色表示IoU值
- **图例**: Object类别、viewpoint数量、IoU范围

## 使用方式

### 自动生成
```bash
python batch_sample_generation.py data.json config.json output/ 10
```
可视化会在房间分割成功后自动创建。

### 手动调用
```python
from batch_sample_generation import create_viewpoint_visualization

success = create_viewpoint_visualization(
    obj_data, 
    "topdown_view.png", 
    coords_info, 
    "visualization.png"
)
```

## 优势

1. **一体化处理**: 无需单独运行可视化脚本
2. **批量处理**: 自动为所有采样object创建可视化
3. **信息丰富**: 同时显示空间分布和质量信息
4. **易于理解**: 直观的颜色编码和图例

## 兼容性

- 兼容原有的 `visualize_view_points.py` 功能
- 支持相同的坐标变换和投影方法
- 使用相同的颜色编码规则

## 错误处理

- 缺少依赖时优雅降级
- 坐标信息不可用时跳过可视化
- 详细的错误日志和状态反馈

这个增强功能让批量采样脚本不仅能生成房间分割图像，还能提供丰富的viewpoint分析可视化，极大提升了数据分析的效率。
