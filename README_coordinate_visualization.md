# Coordinate Visualization Tool

这个工具可以在3D场景的俯视图上可视化指定的二维坐标点（X,Z轴）。

## 功能特点

- 支持单个或多个坐标点的同时可视化
- 可通过楼层索引或Y坐标自动选择楼层
- 支持配置文件和命令行参数
- 自动生成不同颜色的标记点
- 显示坐标标签和索引
- 自动检测坐标点是否在图像边界内

## 使用方法

### 1. 命令行方式

#### 基本用法 - 单个坐标
```bash
python visualize_coordinates.py \
  --scene data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb \
  --floor 0 \
  --coordinates "1.5,2.3"
```

#### 多个坐标
```bash
python visualize_coordinates.py \
  --scene data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb \
  --floor 0 \
  --coordinates "1.5,2.3" "4.2,-1.8" "-0.5,3.1" "0.0,0.0"
```

#### 使用Y坐标自动选择楼层
```bash
python visualize_coordinates.py \
  --scene data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb \
  --floor_y -1.0 \
  --coordinates "1.5,2.3"
```

#### 自定义渲染参数
```bash
python visualize_coordinates.py \
  --scene data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb \
  --floor 0 \
  --coordinates "1.5,2.3" "4.2,-1.8" \
  --resolution 4096 \
  --marker_radius 12 \
  --draw_coordinates_grid \
  --output custom_output.png
```

### 2. 配置文件方式

```bash
python visualize_coordinates.py --config coordinate_config.json
```

配置文件示例（`coordinate_config.json`）：
```json
{
  "scene_config": {
    "scene_path": "data/versioned_data/hm3d-0.2/hm3d/val/00890-6s7QHgap2fW/6s7QHgap2fW.basis.glb",
    "target_floor": 0,
    "target_coordinate": null,
    "custom_ortho_scale": null,
    "target_coverage": 0.9,
    "draw_coordinates": false
  },
  "coordinates": [
    "1.5,2.3",
    "4.2,-1.8", 
    "-0.5,3.1",
    "0.0,0.0"
  ],
  "resolution": 2048,
  "visualization": {
    "show_indices": true,
    "marker_radius": 8
  },
  "output": {
    "output_path": null
  }
}
```

## 参数说明

### 必需参数
- `--scene`: 场景.glb文件路径
- `--floor` 或 `--floor_y`: 楼层选择方式
  - `--floor`: 楼层索引（从0开始）
  - `--floor_y`: Y坐标值，自动选择最近的楼层
- `--coordinates`: 坐标列表，格式为"x,z"

### 可选参数
- `--config`: 配置文件路径
- `--output`: 输出图像路径（默认：coordinates_visualization/<scene_id>_coordinates.png）
- `--resolution`: 渲染分辨率（默认：2048）
- `--custom_ortho_scale`: 自定义正交投影比例
- `--target_coverage`: 场景覆盖率（默认：0.9）
- `--draw_coordinates_grid`: 在图像上绘制坐标网格
- `--show_indices`: 显示坐标点索引（默认：true）
- `--marker_radius`: 标记点半径（默认：8像素）

## 输出说明

脚本会生成一个带有标记点的俯视图图像，其中：

1. **坐标点标记**：
   - 不同颜色的圆形标记表示不同的坐标点
   - 每个点旁边显示坐标值和索引
   
2. **信息面板**：
   - 显示总坐标数和可见坐标数
   - 显示图像的世界坐标边界范围

3. **文件输出**：
   - 默认保存到 `coordinates_visualization/` 目录
   - 文件名格式：`<scene_id>_coordinates.png`

## 依赖要求

- Python 3.7+
- PIL/Pillow
- NumPy
- habitat-sim
- 本项目的 `preprocess/current_topdown.py` 模块

## 注意事项

1. 确保场景文件路径正确且文件存在
2. 坐标格式必须是"x,z"，用逗号分隔
3. 如果坐标点超出图像边界，会显示警告信息
4. Y坐标选择楼层时，会自动选择底部Y值最接近的楼层

## 错误排查

- **"Scene file not found"**: 检查场景文件路径是否正确
- **"No valid coordinates found"**: 检查坐标格式，确保使用"x,z"格式
- **"Failed to render topdown view"**: 检查场景文件是否损坏或不兼容
- **坐标点不显示**: 检查坐标是否在场景范围内
