# Multi-Agent Habitat Navigation System

这是一个基于Habitat-AI的多智能体同步导航系统，支持预测性碰撞检测、状态持久化和第一人称视角视频生成。

## 🌟 核心特性

### 1. 配置文件驱动
- 使用 YAML 配置文件定义整个模拟环境
- 支持多个智能体的独立配置
- 灵活的传感器和动作空间配置

### 2. 真实物理智能体
- 支持加载物理智能体模型（如 Fetch 机器人）
- 启用物理引擎进行真实的碰撞检测
- 第一人称视角来自智能体模型的传感器

### 3. 预测性碰撞检测
- 执行前预测碰撞风险
- 检测与环境和其他智能体的碰撞
- 智能体碰撞时自动停止所有运动

### 4. 状态持久化
- 保存智能体的完整状态（位置、旋转等）
- 支持从保存的状态恢复并继续执行
- 可以无缝衔接新的动作序列

### 5. 独立视频生成
- 为每个智能体生成独立的视频文件
- 左侧：第一人称视角（FPV）
- 右侧：实时地图，显示当前智能体位置和朝向

### 6. 全面的日志记录
- 记录所有关键事件和动作执行结果
- 支持不同日志级别
- 便于调试和分析

## 📁 项目结构

```
video_app/
├── config/                          # 配置文件目录
│   ├── multi_agent_config.yaml     # 主配置文件
│   ├── actions_example.json        # 示例动作序列
│   └── sample_*.yaml               # 其他示例配置
├── src/                            # 源代码目录
│   ├── habitat_video_generator.py  # 基础Habitat模拟器
│   ├── multi_agent_navigation.py   # 多智能体导航主程序
│   └── utils/                      # 工具模块
├── outputs/                        # 输出目录
│   ├── agent_*_output.mp4         # 智能体视频
│   ├── multi_agent_nav.log        # 运行日志
│   └── agent_states.json          # 保存的状态
├── launcher.py                     # 启动器脚本
└── README.md                       # 本文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 Habitat-Sim 和相关依赖：

```bash
# 安装Habitat-Sim（如果尚未安装）
conda install habitat-sim -c conda-forge

# 安装其他依赖
pip install opencv-python pillow pyyaml numpy
```

### 2. 数据准备

确保有可用的场景数据集：

```bash
# 下载测试场景（如果需要）
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes

# 或者使用HM3D数据集
python -m habitat_sim.utils.datasets_download --uids hm3d_minival
```

### 3. 创建示例配置

```bash
# 创建示例配置和动作文件
python launcher.py --create-sample
```

### 4. 运行多智能体导航

```bash
# 使用默认配置运行
python launcher.py

# 使用自定义配置
python launcher.py -c my_config.yaml -a my_actions.json

# 从保存的状态恢复
python launcher.py -r ./outputs/agent_states.json
```

## 📝 配置文件详解

### 主配置文件 (multi_agent_config.yaml)

```yaml
# 场景配置
scene:
  scene_dataset_path: "data/scene_datasets/hm3d/hm3d-minival-habitat.scene_dataset_config.json"
  scene_id: null  # 自动选择第一个可用场景

# 模拟器配置
simulator:
  gpu_device_id: 0
  enable_physics: true
  random_seed: 1

# 智能体配置
agents:
  - id: "agent_0"
    agent_model_path: "data/robots/hab_fetch/hab_fetch.urdf"
    sensors:
      color_sensor:
        sensor_type: "COLOR"
        resolution: [512, 512]
        position: [0.0, 1.5, 0.0]
        hfov: 90.0
    # ... 更多配置选项
```

### 动作序列文件 (actions.json)

```json
{
  "agent_0": [
    { "action": "move_to", "target": [4.0, 3.0] },
    { "action": "turn_left", "angle": 30 },
    { "action": "move_forward", "distance": 2.0 }
  ],
  "agent_1": [
    { "action": "move_to", "target": [-1.5, 2.5] },
    { "action": "turn_right", "angle": 90 }
  ]
}
```

## 🎮 支持的动作类型

| 动作类型 | 参数 | 描述 |
|---------|------|------|
| `move_to` | `target: [x, z]` | 移动到指定坐标 |
| `move_forward` | `distance: float` | 向前移动指定距离 |
| `turn_left` | `angle: float` | 左转指定角度（度） |
| `turn_right` | `angle: float` | 右转指定角度（度） |

## 🛡️ 碰撞检测系统

### 检测类型
1. **环境碰撞**: 与墙壁、家具等静态物体
2. **智能体碰撞**: 智能体之间的碰撞

### 检测策略
- **预测性检测**: 在执行动作前预测是否会碰撞
- **实时检测**: 考虑智能体的半径和高度
- **安全停止**: 检测到碰撞风险时立即停止所有智能体

### 配置参数

```yaml
collision_detection:
  enabled: true
  agent_radius: 0.4        # 智能体半径（米）
  height_threshold: 0.3    # 高度阈值
  min_agent_distance: 0.8  # 智能体间最小距离
```

## 💾 状态持久化

### 自动保存
- 每个动作执行后保存状态（可配置）
- 模拟结束时保存最终状态
- 碰撞发生时保存当前状态

### 状态恢复
```bash
# 从保存的状态恢复
python launcher.py -r ./outputs/agent_states.json -a new_actions.json
```

### 状态文件格式
```json
{
  "agent_0": {
    "position": [1.0, 0.0, 2.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "velocity": 0.0,
    "angular_velocity": 0.0
  }
}
```

## 🎬 视频输出

### 视频特性
- **分辨率**: 1024x2048 (可配置)
- **帧率**: 30 FPS (可配置)
- **格式**: MP4
- **布局**: 左侧FPV + 右侧地图

### 地图特性
- 显示坐标网格
- 实时智能体位置标记
- 方向指示箭头
- 仅显示当前智能体（不显示其他智能体）

### 输出文件
```
outputs/
├── agent_0_output.mp4
├── agent_1_output.mp4
├── agent_2_output.mp4
└── multi_agent_nav.log
```

## 📊 日志系统

### 日志级别
- `DEBUG`: 详细调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息

### 日志内容
- 模拟器初始化信息
- 每个动作的执行结果
- 碰撞检测结果
- 模拟终止原因
- 最终智能体位置

### 示例日志
```
2025-07-06 10:30:15 - INFO - Multi-agent simulator initialized successfully
2025-07-06 10:30:16 - INFO - Agent agent_0 initialized at [4.0, 0.0, 3.0]
2025-07-06 10:30:17 - INFO - Executing step 1/5
2025-07-06 10:30:18 - INFO - Agent agent_0 moved to [4.0, 0.0, 3.0]
2025-07-06 10:30:19 - WARNING - Collision predicted at step 3: Agent collision predicted: agent_0-agent_1
2025-07-06 10:30:19 - INFO - Stopping all agents to avoid collision
```

## ⚙️ 高级配置

### 运动参数调整
```yaml
movement:
  linear_speed: 1.0      # 移动速度（米/秒）
  angular_speed: 30.0    # 旋转速度（度/秒）
  time_step: 0.033       # 时间步长（~30fps）
```

### 视频质量设置
```yaml
video_output:
  resolution: [1024, 2048]  # [高度, 宽度]
  fps: 30
  codec: "mp4v"
  save_frames: false        # 是否保存单独帧
```

### 地图外观自定义
```yaml
map_config:
  resolution: 1024
  show_grid: true
  grid_interval: 1.0
  agent_marker_size: 8
  agent_marker_color: [255, 0, 0]
```

## 🔧 故障排除

### 常见问题

1. **模拟器初始化失败**
   - 检查场景文件路径是否正确
   - 确保GPU内存充足
   - 验证Habitat-Sim安装

2. **智能体模型加载失败**
   - 确认URDF文件路径正确
   - 检查物理引擎是否启用
   - 回退到默认虚拟智能体

3. **碰撞检测过于敏感**
   - 调整 `agent_radius` 参数
   - 修改 `min_agent_distance` 设置
   - 调整 `height_threshold`

4. **视频生成失败**
   - 检查OpenCV安装
   - 确认输出目录权限
   - 验证视频编码器支持

### 调试技巧

1. **启用详细日志**
   ```yaml
   logging:
     log_level: "DEBUG"
     console_output: true
   ```

2. **保存中间帧**
   ```yaml
   video_output:
     save_frames: true
   ```

3. **禁用碰撞检测**
   ```yaml
   collision_detection:
     enabled: false
   ```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) 团队
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) 项目
- 所有贡献者和用户

---

如果您有任何问题或建议，请提出 Issue 或联系项目维护者。
