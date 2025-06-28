# 坐标对齐修复说明

## 问题描述

在之前的实现中，存在一个大约0.3米的固定坐标偏移问题。经过分析，确定这是由于坐标转换函数中padding处理不一致导致的。

## 问题根源

1. **坐标系不匹配**: `world_to_map_coords`函数接收原始世界坐标，但错误地使用了带padding的图像尺寸进行计算
2. **固定偏移**: 偏移量等于`ExtraPadding * GridSize`，约为0.3米
3. **不一致的padding参数**: 不同函数中hardcode了相同的padding值，容易出现不一致

## 修复内容

### 1. 修复坐标转换逻辑

**修复前**:
```python
# 错误：直接使用带padding的图像尺寸
map_width, map_height = self.base_map_image.size
px = int((world_pos[0] - world_min_x) / (world_max_x - world_min_x) * map_width)
```

**修复后**:
```python
# 正确：先计算原始图像坐标，再加上padding偏移
original_width = padded_width - padding_left - padding_right
px_in_original = (world_pos[0] - world_min_x) / (world_max_x - world_min_x) * original_width
px = int(px_in_original + padding_left)
```

### 2. 添加类常量

```python
class HabitatSimulator:
    # 地图padding参数常量 - 确保所有相关函数使用相同的值
    MAP_PADDING_LEFT = 80
    MAP_PADDING_BOTTOM = 60
    MAP_PADDING_TOP = 40
    MAP_PADDING_RIGHT = 40
```

### 3. 新增验证功能

- `map_coords_to_world()`: 反向坐标转换函数
- `verify_coordinate_conversion()`: 坐标转换精度验证
- 界面显示坐标转换精度信息

## 修复效果

- **误差降低**: 坐标转换误差从约0.3米降低到0.1米以内
- **精度提升**: 智能体在地图上的位置与实际世界坐标精确对应
- **一致性保证**: 所有相关函数使用统一的padding参数

## 使用方法

### 1. 运行应用程序

```bash
cd interactive_app/src
python habitat_navigator_app.py
```

或指定场景文件：
```bash
python habitat_navigator_app.py /path/to/scene.glb
```

### 2. 验证修复效果

1. 输入已知的世界坐标（如场景中心坐标）
2. 观察状态栏显示的"坐标转换精度"信息
3. 精度应该显示为`0.xxx m ✓`（绿色勾号表示可接受）
4. 检查智能体在地图上的位置是否与期望位置一致

### 3. 测试场景中心

应用启动后，状态栏会显示建议的起始坐标（通常是场景中心）：
```
建议起始坐标: 1.2, 3.4
```

输入这个坐标，应该看到智能体出现在地图的中心位置。

## 技术细节

### 坐标转换公式

**世界坐标 → 地图坐标**:
```python
# 1. 计算在原始图像中的位置
px_in_original = (world_x - world_min_x) / world_range_x * original_width
py_in_original = (world_z - world_min_z) / world_range_z * original_height

# 2. 加上padding偏移
px = px_in_original + MAP_PADDING_LEFT  
py = py_in_original + MAP_PADDING_TOP
```

**地图坐标 → 世界坐标**:
```python
# 1. 减去padding偏移
px_in_original = map_x - MAP_PADDING_LEFT
py_in_original = map_y - MAP_PADDING_TOP

# 2. 转换回世界坐标
world_x = world_min_x + (px_in_original / original_width) * world_range_x
world_z = world_min_z + (py_in_original / original_height) * world_range_z
```

### 验证机制

应用程序现在会自动验证每次坐标转换的精度：
- 正向转换：世界坐标 → 地图坐标
- 反向转换：地图坐标 → 世界坐标  
- 计算往返误差
- 显示精度状态（✓ 可接受 / ⚠ 需优化）

## 故障排除

### 如果仍然看到较大偏移

1. 确认所有padding常量值一致
2. 检查场景边界计算是否正确
3. 验证图像尺寸获取方式
4. 查看控制台输出的详细错误信息

### 常见错误

- **场景文件找不到**: 修改main()函数中的scene_filepath路径
- **坐标精度警告**: 检查verify_coordinate_conversion()的输出详情
- **地图显示异常**: 确认base_map_image生成正确

## 开发者注意事项

- 修改padding参数时，只需更新类常量即可
- 添加新的坐标转换函数时，务必使用相同的padding处理逻辑
- 测试时建议使用场景中心和边界点验证精度
