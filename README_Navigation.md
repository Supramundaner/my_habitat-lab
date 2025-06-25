# Habitat Coordinate Navigation System

基于Habitat Lab的坐标导航系统，支持在3D环境中进行坐标基础的智能体导航。

## 功能特点

1. **多数据集支持**: 支持Matterport3D、Habitat Test Scenes等多种数据集
2. **坐标导航**: 通过输入3D坐标直接控制智能体移动
3. **视角控制**: 支持调整智能体的观察视角
4. **多视角观察**: 提供第一人称、第三人称和俯视图
5. **全局地图**: 显示场景的拓扑结构图
6. **交互式界面**: 提供命令行交互界面

## 安装要求

```bash
# 安装Habitat Lab
pip install habitat-sim
pip install habitat-lab

# 安装其他依赖
pip install matplotlib opencv-python numpy
```

## 使用方法

### 1. 下载数据集

```bash
# 下载测试场景（推荐开始使用）
python download_matterport.py

# 或者手动下载Matterport3D数据集（需要学术许可）
python -m habitat_sim.utils.datasets_download --uids mp3d --data-path ./data
```

### 2. 运行导航系统

```bash
# 运行主界面
python navigation_demo.py

# 或者直接运行交互式导航
python examples/coordinate_navigation.py

# 或者运行测试程序
python test_navigation.py
```

## 命令说明

### 交互式导航命令

- `move x y z` - 移动到指定坐标 (x, y, z)
- `rotate yaw pitch` - 调整视角 (偏航角和俯仰角，单位为度)
- `view` - 显示当前观察结果
- `position` - 显示当前位置
- `scenes` - 列出可用场景
- `switch scene_id` - 切换到指定场景
- `help` - 显示帮助信息
- `quit` - 退出程序

### 示例命令

```bash
# 移动到坐标 (5, 1.5, 3)
move 5 1.5 3

# 向右转45度
rotate 45 0

# 向上看15度
rotate 0 15

# 切换到van-gogh-room场景
switch habitat-test-scenes/van-gogh-room
```

## 程序架构

### 主要文件

- `examples/coordinate_navigation.py` - 核心导航系统
- `navigation_demo.py` - 演示和交互界面
- `download_matterport.py` - 数据集下载工具
- `test_navigation.py` - 测试程序

### 核心类

#### CoordinateNavigationAgent

主要的导航智能体类，提供以下功能：

- **初始化**: `__init__(scene_id, data_path)`
- **移动控制**: `move_to_coordinate(target_position)`
- **视角控制**: `adjust_view_angle(yaw_delta, pitch_delta)`
- **观察获取**: `get_current_observations()`
- **地图生成**: `get_topdown_map()`
- **场景切换**: `switch_scene(scene_id)`

## 技术细节

### 坐标系统

- 使用右手坐标系
- X轴：东西方向（正方向为东）
- Y轴：上下方向（正方向为上）
- Z轴：南北方向（正方向为北）

### 视角系统

- 偏航角 (Yaw): 绕Y轴旋转，控制左右转向
- 俯仰角 (Pitch): 绕X轴旋转，控制上下观察
- 滚转角 (Roll): 绕Z轴旋转（当前未使用）

### 传感器配置

1. **RGB相机**: 第一人称视角，512x512分辨率
2. **深度相机**: 提供深度信息
3. **第三人称相机**: 从后上方观察智能体

## 数据集支持

### 当前支持的数据集

1. **Habitat Test Scenes**: 小型测试场景，适合开发和测试
   - apartment_1: 公寓场景
   - skokloster-castle: 城堡场景
   - van-gogh-room: 梵高房间场景

2. **Matterport3D**: 大型真实室内场景（需要学术许可）

3. **Replica CAD**: 高质量合成场景

### 添加新数据集

1. 将场景文件放置在 `data/scene_datasets/` 目录下
2. 修改 `_get_available_scenes()` 方法以支持新的数据集格式
3. 更新配置创建逻辑以处理新的文件路径

## 故障排除

### 常见问题

1. **Scene file not found**: 检查数据集是否正确下载
2. **Position not navigable**: 目标位置不可达，尝试其他坐标
3. **Simulator initialization failed**: 检查Habitat-Sim安装

### 调试技巧

1. 使用 `agent.list_available_scenes()` 查看可用场景
2. 使用 `agent.get_current_position()` 确认当前位置
3. 查看拓扑图了解可导航区域
4. 从较近的位置开始测试移动

## 扩展功能

### 可能的改进

1. **路径规划**: 添加A*或其他路径规划算法
2. **物体检测**: 集成语义分割和物体识别
3. **多智能体**: 支持多个智能体同时导航
4. **VR/AR支持**: 添加虚拟现实界面
5. **语音控制**: 通过语音命令控制导航

### API扩展

```python
# 扩展示例
class AdvancedNavigationAgent(CoordinateNavigationAgent):
    def plan_path(self, start, goal):
        """实现路径规划"""
        pass
    
    def avoid_obstacles(self):
        """实现障碍物避让"""
        pass
    
    def semantic_navigation(self, object_name):
        """基于语义的导航"""
        pass
```

## 许可和引用

如果您在研究中使用了本系统，请引用Habitat Lab：

```bibtex
@inproceedings{habitat19iccv,
  title     = {Habitat: A Platform for Embodied AI Research},
  author    = {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

## 联系和支持

- Habitat Lab官方文档: https://habitat-sim.readthedocs.io/
- GitHub仓库: https://github.com/facebookresearch/habitat-lab
- 问题反馈: 请在GitHub上提交Issue
