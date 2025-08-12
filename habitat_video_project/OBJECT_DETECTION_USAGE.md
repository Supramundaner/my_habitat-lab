# Object Detection 功能使用说明

## 概述

本功能为habitat_video_project添加了基于物体检测的智能导航能力。当动作序列中包含`target`参数时，系统会使用Grounding DINO和Mobile SAM进行实时物体检测和分割，然后导航到检测到的物体位置。

## 安装和配置

### 1. 环境要求

- Python 3.8+
- CUDA支持（推荐）
- tmux
- VLFM模型权重文件

### 2. 模型文件准备

确保以下模型文件存在于正确位置：
- `vlfm/data/groundingdino_swint_ogc.pth` - Grounding DINO模型权重
- `vlfm/data/mobile_sam.pt` - Mobile SAM模型权重

### 3. 环境变量设置

在`.bashrc`中添加：
```bash
export VLFM_PYTHON=`which python`
export MOBILE_SAM_CHECKPOINT=vlfm/data/mobile_sam.pt
export GROUNDING_DINO_CONFIG=vlfm/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_WEIGHTS=vlfm/data/groundingdino_swint_ogc.pth
```

## 使用方法

### 1. 启动后台服务

#### 方式一：使用本地VLM模块（推荐）
```bash
# 在habitat_video_project目录下
./start_local_vlm_services.sh
```

#### 方式二：使用外部VLFM项目
```bash
# 在habitat_video_project目录下
./start_object_detection_services.sh
```

或者手动启动：
```bash
cd vlfm
bash scripts/launch_vlm_servers.sh
```

**重要**：等待90秒让模型完全加载。

### 2. 配置动作序列

创建包含target参数的动作序列文件，例如`configs/example_actions_new.json`：

```json
{
    "initial_state": {
        "position": [5.0, 5.0],
        "rotation": 0
    },
    "action": [
        {
            "sequence": [
                {
                    "type": "move_to",
                    "params": {"x": 6.0, "z": 7.0}
                },
                {
                    "type": "move_to",
                    "params": {"x": 8.0, "z": 7.0}
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

## 配置参数

在`configs/default_config.json`中可以调整以下参数：

```json
{
    "object_detection": {
        "enabled": true,                    // 是否启用物体检测
        "grounding_dino_port": 12181,       // Grounding DINO服务端口
        "mobile_sam_port": 12183,           // Mobile SAM服务端口
        "detection_threshold": 0.4,         // 检测置信度阈值
        "max_detection_distance": 10.0,     // 最大检测距离（米）
        "target_reach_distance": 1.5        // 目标到达距离（米）
    }
}
```

## 工作流程

1. **启动服务**：使用tmux启动VLFM模型服务
2. **加载配置**：读取动作序列和object detection配置
3. **执行检测**：对每个带target的move_to动作进行物体检测
4. **智能导航**：检测成功则导航到物体位置并终止序列
5. **回退机制**：检测失败则使用原始坐标继续执行

## 支持的物体类型

Grounding DINO支持任意文本描述的物体，例如：
- "Chair", "Table", "Lamp"
- "Red chair", "Wooden table"
- "Person", "Dog", "Cat"
- 等等

## 故障排除

### 1. 服务启动失败

```bash
# 检查tmux会话
tmux list-sessions

# 查看服务日志
tmux attach-session -t vlm_servers_XXXX
```

### 2. 检测失败

- 检查模型文件是否存在
- 确认服务端口是否正确
- 调整检测阈值参数
- 检查网络连接

### 3. 坐标转换错误

- 检查相机参数配置
- 确认深度图像质量
- 调整坐标系转换逻辑

## 性能优化

### 1. 检测频率

可以通过修改代码调整检测频率，避免每帧都进行检测。

### 2. 模型优化

- 使用量化模型减少内存占用
- 调整模型分辨率平衡速度和精度

### 3. 并行处理

可以并行处理多个检测请求以提高效率。

## 扩展功能

### 1. 多目标检测

可以扩展支持同时检测多个目标物体。

### 2. 物体跟踪

添加物体跟踪功能，提高检测稳定性。

### 3. 动态目标

支持移动物体的检测和跟踪。

## 注意事项

1. **内存使用**：VLFM模型需要大量GPU内存，确保有足够的显存
2. **网络延迟**：模型服务调用有网络延迟，考虑本地部署
3. **检测精度**：检测结果受光照、角度等因素影响
4. **坐标系**：确保相机坐标系到世界坐标系的转换正确

## 技术支持

如遇到问题，请检查：
1. 日志输出信息
2. 配置文件设置
3. 模型文件完整性
4. 服务状态 