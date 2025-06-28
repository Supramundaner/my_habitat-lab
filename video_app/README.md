# Habitat Video Generator

离线视频生成应用，接收JSON格式的导航指令序列，生成展示代理运动过程的MP4视频。

## 功能特性

- **命令行交互**: 在终端中运行，支持持续的指令输入循环
- **JSON指令格式**: 支持绝对定位、相对旋转和移动指令
- **平滑动画**: 所有移动和旋转都通过插值生成平滑过渡
- **智能视角调整**: 移动时代理视角会平滑地指向目标方向
- **碰撞检测**: 实时检测代理与不可导航区域的碰撞
- **状态持久化**: 代理状态在指令序列间保持连续
- **GPU加速**: 利用NVIDIA GPU进行高效渲染
- **左右分屏**: 第一人称视角 + 俯视地图，地图保持正确纵横比

## 最新改进

### v2.0 更新 (2025-06-28)
- **修复视角连续性**: 移动时代理视角会平滑转向目标方向
- **改进地图显示**: 2D俯视地图现在保持正确的纵横比，避免拉伸
- **优化旋转插值**: 改进四元数插值算法，确保平滑旋转
- **参考interactive_app**: 基于interactive_app的成熟实现进行优化

## 安装依赖

```bash
pip install opencv-python pillow numpy
```

确保已正确安装Habitat-sim和相关依赖。

## 使用方法

### 基本运行

```bash
cd /home/yaoaa/habitat-lab/video_app
python main.py
```

### 命令行参数

```bash
python main.py --scene /path/to/scene.glb --gpu 0 --fps 30 --output-dir ./outputs
```

- `--scene`: .glb场景文件路径（默认: van-gogh-room.glb）
- `--gpu`: CUDA设备ID（默认: 0）
- `--fps`: 视频帧率（默认: 30）
- `--output-dir`: 输出目录（默认: ./outputs）

### 指令格式

指令序列为JSON格式的列表，每个指令可以是：

1. **绝对定位**: `[x, z]` - 将代理移动到指定坐标
   ```json
   [2.6, 0.1]
   ```

2. **相对旋转**: `["direction", angle]` - 旋转指定角度
   ```json
   ["right", 30]
   ["left", 45]
   ```

### 使用示例

```
Habitat Video Generator Initialized.
Enter command sequence as a JSON string or 'exit'.
> [[2.6, 0.1], ["right", 30], [2.5, 0.4]]
Processing 3 commands...
  Executing command 1/3: [2.6, 0.1]
  Executing command 2/3: ['right', 30]
  Executing command 3/3: [2.5, 0.4]
  Generated 156 frames in 2.34s
Video successfully saved to: /home/yaoaa/habitat-lab/video_app/outputs/output_20231027_153210.mp4
> [["left", 90], [3.0, 0.8]]
Processing 2 commands...
  Executing command 1/2: ['left', 90]
  Executing command 2/2: [3.0, 0.8]
    ERROR: Collision detected while moving to [3.0, 0.8]. Aborting.
  Generated 142 frames in 1.87s
Video successfully saved to: /home/yaoaa/habitat-lab/video_app/outputs/output_20231027_153345.mp4
> exit
Shutting down.
```

## 动画参数

- **旋转动画**: 每1度生成一帧
- **移动动画**: 每0.05米生成一帧
- **帧率**: 默认30 FPS

## 视频输出

- **格式**: MP4
- **分辨率**: 1024x512（左右各512x512）
- **编码**: H.264
- **文件名**: `output_YYYYMMDD_HHMMSS.mp4`

## 错误处理

- **碰撞检测**: 移动过程中检测到碰撞时停止执行并保存已生成的帧
- **位置验证**: 自动验证目标位置是否可导航
- **格式验证**: 检查JSON指令格式的正确性

## 技术细节

### 代码复用

- 高度复用 `interactive_app/src/habitat_navigator_app.py` 中的核心逻辑
- 继承 `HabitatSimulator` 类并扩展视频生成功能
- 保持与交互式应用相同的渲染质量和坐标系

### 性能优化

- 支持指定GPU设备进行硬件加速
- 使用OpenCV进行高效视频编码
- 优化帧捕获和图像处理流程

### 状态管理

- 代理状态在指令序列执行完毕后保持
- 支持连续的导航任务而无需重新初始化
- 自动处理位置对齐和朝向计算

## 故障排除

1. **场景文件未找到**: 确保场景文件路径正确且文件存在
2. **GPU设备错误**: 检查CUDA设备ID是否有效
3. **权限问题**: 确保对输出目录有写权限
4. **内存不足**: 减少动画步长或降低视频分辨率

## 目录结构

```
video_app/
├── main.py                 # 主程序入口
├── src/
│   └── habitat_video_generator.py  # 核心视频生成器
├── outputs/                # 视频输出目录
└── README.md              # 使用说明
```
