# Object Detection 功能实现文档

## 概述

本实现为habitat_video_project添加了基于物体检测的智能导航功能。当动作序列中包含`target`参数时，系统会使用Grounding DINO和Mobile SAM进行实时物体检测和分割，然后导航到检测到的物体位置。

## 实现方案

### 1. 架构设计

```
动作序列 → 检测target参数 → 物体检测 → 分割 → 3D投影 → 导航 → 终止序列
```

### 2. 核心组件

#### A. ObjectDetector类 (`src/object_detector.py`)
- 封装Grounding DINO客户端
- 封装Mobile SAM客户端
- 提供统一的物体检测和分割接口
- 处理深度投影和坐标计算

#### B. 独立VLM模块 (`vlm/`)
- `grounding_dino.py`: Grounding DINO客户端和服务器
- `sam.py`: Mobile SAM客户端和服务器
- `detections.py`: 检测结果数据结构
- `server_wrapper.py`: 服务器包装器工具
- `__init__.py`: 模块导出

#### B. 修改ActionProcessor类 (`src/action_processor.py`)
- 添加`_handle_object_detection_move`方法
- 修改`execute_sequence`方法支持智能终止
- 添加物体检测结果处理逻辑

#### C. 配置文件修改
- 添加object detection相关配置
- 添加模型端口配置

### 3. 工作流程

1. **启动后台服务**：使用tmux启动VLFM模型服务
2. **加载动作序列**：支持新的带target参数的动作格式
3. **执行检测**：对每个带target的move_to动作进行物体检测
4. **智能导航**：检测成功则导航到物体位置并终止序列
5. **回退机制**：检测失败则使用原始坐标继续执行

## 详细修改内容

### 1. 新增文件

#### `src/object_detector.py` ✅ 已完成
- `ObjectDetector`类：主要的物体检测接口
- `detect_object`方法：执行物体检测和分割
- `project_to_3d`方法：将2D分割结果投影到3D空间
- `calculate_target_position`方法：计算导航目标位置
- `detect_and_get_target_coords`方法：完整的检测流程

#### `start_object_detection_services.sh` ✅ 已完成
- 启动脚本：方便启动VLFM后台服务
- 错误检查：验证VLFM目录和脚本存在性
- 使用说明：提供tmux会话管理命令

#### `start_local_vlm_services.sh` ✅ 已完成
- 本地VLM服务启动脚本：使用独立的VLM模块
- 错误检查：验证本地VLM模块完整性
- 简化部署：不依赖外部VLFM项目

#### `OBJECT_DETECTION_USAGE.md` ✅ 已完成
- 详细的使用说明文档
- 安装配置指南
- 故障排除方法

### 2. 修改文件

#### `src/action_processor.py` ✅ 已完成
- 导入`ObjectDetector`类
- 在`__init__`中初始化物体检测器
- 修改`execute_sequence`：支持智能终止逻辑
- 新增`_handle_object_detection_move`：处理物体检测导航
- 新增`_detect_and_get_target_coords`：执行物体检测
- 新增`_camera_to_world_coords`：坐标系转换

#### `configs/default_config.json` ✅ 已完成
- 添加`object_detection`配置部分
- 配置模型端口（Grounding DINO: 12181, Mobile SAM: 12183）
- 配置检测参数（阈值、距离等）

#### `main.py` ✅ 已完成
- 更新动作序列加载逻辑：支持新旧两种格式
- 修改执行流程：支持多动作组和智能终止
- 更新报告生成：包含物体检测相关信息

### 3. 配置参数

```json
{
  "object_detection": {
    "enabled": true,
    "grounding_dino_port": 12181,
    "mobile_sam_port": 12183,
    "detection_threshold": 0.4,
    "max_detection_distance": 10.0,
    "target_reach_distance": 1.5
  }
}
```

## 使用方法

### 1. 启动后台服务
```bash
cd vlfm
bash scripts/launch_vlm_servers.sh
```

### 2. 配置动作序列
```json
{
  "action": [
    {
      "sequence": [
        {
          "type": "move_to",
          "params": {"x": 6.0, "z": 7.0}
        }
      ],
      "target": "Chair"
    }
  ]
}
```

### 3. 运行程序
```bash
python main.py --actions configs/example_actions_new.json
```

## 技术细节

### 1. 物体检测流程
1. 获取RGB图像和深度图像
2. 使用Grounding DINO检测目标物体
3. 使用Mobile SAM进行精确分割
4. 将分割结果投影到3D空间
5. 计算物体的中心点坐标

### 2. 坐标转换
- 使用深度信息和相机参数进行3D投影
- 考虑相机内参和变换矩阵
- 处理坐标系转换

#### 相机参数获取方式
相机参数通过以下方式动态计算：
1. **从配置获取HFOV**：`config['OCCUPANCY_MAP']['HFOV']` (默认90度)
2. **从图像尺寸计算**：基于实际RGB图像的width和height
3. **使用与map_builder相同的计算方式**：
   - `fx = width / (2.0 * tan(hfov_rad / 2.0))`
   - `fy = fx` (假设像素是正方形的)
   - `cx = width / 2.0` (主点在图像中心)
   - `cy = height / 2.0` (主点在图像中心)

这确保了与现有map_builder的坐标系转换逻辑完全一致。

### 3. 错误处理
- 检测失败时的回退机制
- 网络连接错误的处理
- 模型服务不可用的处理

## 依赖关系

### 1. 外部依赖
- VLFM模型服务（Grounding DINO, Mobile SAM）
- tmux（用于后台服务管理）

### 2. 本地依赖
- 独立VLM模块（`vlm/`目录）
- 本地服务器包装器
- 自定义检测结果数据结构

### 2. 内部依赖
- habitat_sim（模拟器）
- numpy（数值计算）
- cv2（图像处理）

## 性能考虑

### 1. 检测频率
- 每帧执行一次检测
- 可配置的检测间隔

### 2. 内存管理
- 及时释放检测结果
- 避免内存泄漏

### 3. 网络延迟
- 模型服务调用超时处理
- 重试机制

## 测试计划

### 1. 单元测试
- ObjectDetector类的各个方法
- 坐标转换的准确性
- 错误处理机制

### 2. 集成测试
- 完整的物体检测导航流程
- 与现有系统的兼容性
- 性能测试

### 3. 场景测试
- 不同物体的检测准确性
- 复杂环境下的表现
- 边界情况的处理

## 实现总结

### 完成的功能 ✅

1. **ObjectDetector类**：完整的物体检测和分割接口
   - Grounding DINO客户端集成
   - Mobile SAM客户端集成
   - 3D投影和坐标计算
   - 错误处理和回退机制

2. **智能导航逻辑**：基于物体检测的导航系统
   - 检测优先策略
   - 智能终止机制
   - 回退到原始坐标
   - 多动作组支持

3. **配置系统**：灵活的配置管理
   - 模型端口配置
   - 检测参数调整
   - 启用/禁用开关

4. **服务管理**：后台服务启动和管理
   - tmux会话管理
   - 自动启动脚本
   - 状态监控

5. **文档和说明**：完整的使用文档
   - 实现文档
   - 使用说明
   - 故障排除指南

### 技术特点

1. **模块化设计**：ObjectDetector独立封装，易于维护和扩展
2. **向后兼容**：支持新旧两种动作序列格式
3. **错误处理**：完善的异常处理和回退机制
4. **性能优化**：可配置的检测参数和阈值
5. **易于使用**：简单的启动脚本和配置

### 使用流程

1. 启动后台服务：`./start_object_detection_services.sh`
2. 配置动作序列：添加target参数
3. 运行程序：`python main.py --actions configs/example_actions_new.json`
4. 自动检测和导航：系统会自动检测目标物体并导航

## 未来扩展

### 1. 功能扩展
- 支持多目标检测
- 添加物体跟踪功能
- 支持动态目标

### 2. 性能优化
- 模型量化
- 并行处理
- 缓存机制

### 3. 用户体验
- 可视化检测结果
- 实时状态显示
- 配置界面 