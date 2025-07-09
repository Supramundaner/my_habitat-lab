# Habitat视频生成器 - 重构版本

## 项目概述

这是一个基于Habitat-Sim的智能体视频生成器，能够在3D场景中模拟智能体的动作序列并生成左右分屏视频。左侧显示智能体的第一人称视角，右侧显示带有智能体位置标注的俯视图。

## 主要特性

- **MVC架构**: 采用Model-View-Controller设计模式，代码结构清晰
- **物理机器人**: 支持加载URDF模型，具有真实的物理属性
- **碰撞检测**: 预先检测路径碰撞，避免不安全的动作
- **动态地图生成**: 自动适应场景大小，生成高质量俯视图
- **平滑动画**: 支持位置和旋转的平滑插值动画
- **坐标转换**: 支持2D指令坐标到3D世界坐标的自动转换

## 项目结构

```
habitat_video_project/
├── main.py                     # 主入口脚本
├── configs/
│   ├── default_config.json       # 默认配置文件
│   └── example_actions.json    # 示例动作序列
├── assets/
│   ├── scenes/                  # 场景文件目录
│   └── robots/                  # 机器人URDF文件目录
├── outputs/                     # 输出目录（自动创建）
└── src/
    ├── __init__.py
    ├── simulator.py              # 核心模拟器类 (Model)
    ├── video_composer.py         # 视频合成类 (View)
    ├── action_processor.py       # 动作处理类 (Controller)
    └── utils.py                  # 工具函数
```

## 安装依赖

确保已安装以下依赖：

```bash
# Habitat-Sim (根据你的系统安装)
conda install habitat-sim -c conda-forge -c aihabitat

# 其他依赖
pip install opencv-python pillow numpy magnum-python
```

## 配置文件

### 主配置文件 (default_config.json)

```json
{
    "video": {
        "fps": 30,                    # 视频帧率
        "resolution": {
            "width": 2048,            # 总视频宽度
            "height": 1024            # 视频高度
        },
        "fpv_width": 1024,           # 左侧FPV图像宽度
        "map_width": 1024            # 右侧地图宽度
    },
    "agent": {
        "linear_speed": 1.0,         # 线性移动速度 (m/s)
        "angular_speed": 45.0,       # 角速度 (deg/s)
        "sensor_height": 1.5         # 传感器高度 (m)
    },
    "scene": {
        "scene_file": "path/to/scene.glb",    # 场景文件路径
        "robot_urdf": "path/to/robot.urdf"    # 机器人URDF文件路径
    },
    "simulation": {
        "gpu_device_id": 0,          # GPU设备ID
        "enable_physics": true,      # 启用物理模拟
        "time_step": 0.033333       # 物理时间步长
    },
    "output_dir": "./outputs"        # 输出目录
}
```

### 动作序列文件 (example_actions.json)

```json
{
    "initial_state": {
        "position": [0.0, 0.0],      # 初始2D位置 [x, z]
        "rotation": 0.0              # 初始朝向角度 (度)
    },
    "sequence": [
        {
            "type": "move_to",
            "params": {
                "x": 2.0,            # 目标X坐标
                "z": 1.0             # 目标Z坐标
            }
        },
        {
            "type": "turn_left",
            "params": {
                "angle": 90.0        # 左转角度
            }
        },
        {
            "type": "turn_right",
            "params": {
                "angle": 45.0        # 右转角度
            }
        },
        {
            "type": "pause",
            "params": {
                "duration": 2.0      # 暂停时间 (秒)
            }
        }
    ]
}
```

## 使用方法

### 基本用法

```bash
cd habitat_video_project
python main.py --config configs/default_config.json --actions configs/example_actions.json
```

### 命令行参数

- `--config`: 配置文件路径 (默认: configs/default_config.json)
- `--actions`: 动作序列文件路径 (默认: configs/example_actions.json)
- `--output-dir`: 输出目录 (覆盖配置文件设置)
- `--verbose`: 启用详细输出
- `--help`: 显示帮助信息

### 示例

```bash
# 使用默认配置
python main.py

# 指定自定义配置和动作
python main.py --config /home/yaoaa/habitat-lab/habitat_video_project/configs/default_config.json --actions /home/yaoaa/habitat-lab/habitat_video_project/configs/example_actions.json

# 指定输出目录
python main.py --output-dir /path/to/output

# 详细输出
python main.py --verbose
```

## 支持的动作类型

### 1. move_to
直接移动到指定的2D坐标位置
```json
{
    "type": "move_to",
    "params": {
        "x": 3.0,     # 目标X坐标
        "z": 2.0      # 目标Z坐标
    }
}
```

### 2. turn_left
向左转指定角度
```json
{
    "type": "turn_left",
    "params": {
        "angle": 90.0   # 左转角度 (度)
    }
}
```

### 3. turn_right
向右转指定角度
```json
{
    "type": "turn_right",
    "params": {
        "angle": 45.0   # 右转角度 (度)
    }
}
```

### 4. pause
在原地暂停指定时间
```json
{
    "type": "pause",
    "params": {
        "duration": 3.0   # 暂停时间 (秒)
    }
}
```

## 输出文件

每次运行会在输出目录生成两个文件：

1. **视频文件**: `output_YYYYMMDD_HHMMSS.mp4`
   - 左侧: 智能体第一人称视角
   - 右侧: 带位置标注的俯视图
   
2. **报告文件**: `report_YYYYMMDD_HHMMSS.json`
   - 执行统计信息
   - 智能体最终状态
   - 完成的动作序列
   - 碰撞检测结果

## 核心算法

### 1. 坐标转换
- 2D坐标 (x, z) → 3D坐标 (x, y, z)
- 使用导航网格自动获取Y坐标

### 2. 碰撞检测
- 基于Habitat-Sim的`pathfinder.try_step()`
- 预先检测整条路径的安全性
- 逐步验证路径的每个小段

### 3. 动画插值
- 位置: 线性插值
- 旋转: 球面线性插值 (SLERP)
- 根据配置的速度计算帧数

### 4. 地图生成
- 参考TopViewGenerator.py算法
- 动态适应场景大小
- 正交投影生成高质量俯视图

## 故障排除

### 1. 场景文件未找到
确保配置文件中的`scene_file`路径正确，且文件存在。

### 2. URDF模型加载失败
- 检查`robot_urdf`路径是否正确
- 确保URDF文件格式正确
- 如果文件不存在，程序会跳过物理机器人加载

### 3. 导航网格问题
- 确保场景文件包含有效的导航网格
- 检查初始位置是否在可导航区域内

### 4. GPU内存不足
- 降低视频分辨率
- 减少地图分辨率
- 使用较小的场景文件

### 5. 碰撞检测过于敏感
- 调整`check_straight_path_collision`中的`step_size`参数
- 检查场景的导航网格质量

## 扩展开发

### 添加新的动作类型

1. 在`ActionProcessor`类中添加新的处理方法
2. 在`_execute_single_action`中添加对应的分支
3. 实现动画逻辑

### 自定义视频合成

修改`VideoComposer`类中的相关方法：
- `_compose_final_frame`: 修改布局
- `_draw_agent_marker`: 自定义标记样式
- `_process_fpv_image`: 添加图像处理效果

### 扩展配置选项

在配置文件中添加新字段，并在相应的类中读取和使用。

## 许可证

请参考Habitat-Sim的许可证要求。
