# 高级碰撞检测系统

基于Habitat官方文档的API，实现了一个完备的多智能体碰撞检测系统，支持动态碰撞检测和物理引擎集成。

## 主要功能

### 1. 多层次碰撞检测

- **物理引擎碰撞检测**: 使用 `sim.get_physics_contact_points()` 获取实时物理接触点
- **射线投射碰撞检测**: 使用 `sim.cast_ray()` 进行360度环境碰撞检测  
- **导航网格碰撞检测**: 使用 `sim.pathfinder` 作为回退方案
- **智能体间碰撞检测**: 检查智能体之间的最小安全距离

### 2. 动态路径碰撞检测

- **路径分段检测**: 沿运动路径检查多个中间点
- **方向性射线投射**: 根据运动方向进行精确碰撞检测
- **实时监控**: 在动画过程中持续监控物理接触点

### 3. 预测性碰撞检测

- **全面碰撞预测**: 结合多种检测方法进行综合预测
- **动作序列分析**: 分析整个动作序列的潜在碰撞风险
- **智能避障**: 检测到碰撞时停止相关智能体

## 新增的配置选项

```yaml
collision_detection:
  enabled: true
  agent_radius: 0.1
  height_threshold: 0.3
  prediction_steps: 10          # 新增：预测步数
  min_agent_distance: 0.8
  raycast_steps: 20             # 新增：射线检测步数  
  contact_threshold: 0.01       # 新增：物理接触阈值
```

## 核心API改进

### AdvancedCollisionDetector 类

取代了原始的 `CollisionDetector`，提供以下主要方法：

- `check_physics_contacts()`: 检查物理引擎接触点
- `check_environment_collision_with_raycast()`: 射线投射环境碰撞检测
- `check_dynamic_path_collision()`: 动态路径碰撞检测
- `predict_comprehensive_collision()`: 全面碰撞预测
- `step_physics_with_collision_monitoring()`: 物理步进时的碰撞监控

### 主要改进

1. **360度射线检测**: 
   - 水平方向8个射线
   - 斜向上8个射线  
   - 斜向下8个射线

2. **动态路径监控**:
   - 沿路径检查20个中间点
   - 每个点进行方向性碰撞检测
   - 检查与其他智能体的实时距离

3. **物理引擎集成**:
   - 实时获取接触点数据
   - 监控接触力和距离
   - 区分活跃和非活跃接触

4. **碰撞统计**:
   - 记录碰撞历史
   - 提供性能统计
   - 支持调试和优化

## 使用示例

### 基本使用

```python
# 初始化模拟器（自动启用高级碰撞检测）
simulator = MultiAgentSimulator(config)

# 查看碰撞检测状态
simulator.print_agent_status()

# 执行动作（自动进行碰撞预测和监控）
simulator.execute_actions_sequence(actions)

# 获取碰撞统计
stats = simulator.collision_detector.get_collision_statistics()
print(f"碰撞统计: {stats}")
```

### 高级功能

```python
# 启用碰撞可视化
simulator.enable_collision_visualization()

# 生成详细报告
report = simulator.generate_collision_report()
print(report)

# 性能基准测试
benchmark_results = simulator.benchmark_collision_detection(num_tests=100)

# 保存调试数据
debug_file = simulator.save_collision_debug_data("debug.json")

# 获取可视化数据
viz_data = simulator.get_collision_visualization_data()
```

## 测试和验证

### 运行基本测试

```bash
cd video_app
python test_advanced_collision.py
```

测试包括：
1. 碰撞动作序列测试（验证碰撞检测有效性）
2. 安全动作序列测试（验证正常操作不误报）
3. 性能基准测试
4. 可视化数据生成

### 可视化结果

```bash
# 生成可视化图表
python visualize_collision_results.py -i outputs/collision_tests/collision_test_debug.json

# 交互式显示
python visualize_collision_results.py --show-plots

# 自定义输出目录
python visualize_collision_results.py -o my_output_dir
```

### 配置文件

系统提供了专门的测试配置：

- `config/advanced_collision_test_config.yaml`: 优化的测试配置
- `config/multi_agent_config.yaml`: 标准配置

## 输出文件

运行测试后会生成以下文件：

```
outputs/collision_tests/
├── agent_states.json              # 智能体状态
├── advanced_collision_test.log    # 详细日志
├── collision_test_debug.json      # 碰撞调试数据
├── final_collision_debug.json     # 最终调试数据
├── agent_0_output.mp4             # 智能体0的视频
├── agent_1_output.mp4             # 智能体1的视频
└── visualizations/
    ├── agent_positions.png        # 智能体位置图
    ├── collision_statistics.png   # 碰撞统计图
    └── analysis_report.txt        # 分析报告
```

## 性能优化

- **分层检测**: 先快速检测，再精确验证
- **缓存机制**: 避免重复计算
- **异常处理**: 确保检测失败时的稳定性
- **可配置精度**: 根据需求调整检测精度

### 性能基准

典型的性能指标（在标准测试环境下）：

- 物理接触点检测: ~0.001 秒
- 射线投射检测: ~0.005 秒
- 综合碰撞预测: ~0.010 秒
- 平均接触点数量: 0-5个

## 兼容性

- 完全向后兼容原有的碰撞检测配置
- 新功能默认启用，可通过配置禁用
- 支持物理引擎启用/禁用的动态切换
- 支持虚拟智能体和物理机器人的混合使用

## 故障排除

### 常见问题

1. **物理引擎未启用**
   ```yaml
   simulator:
     enable_physics: true
   ```

2. **碰撞检测过于敏感**
   ```yaml
   collision_detection:
     agent_radius: 0.05      # 减小半径
     contact_threshold: 0.02 # 增大阈值
   ```

3. **性能问题**
   ```yaml
   collision_detection:
     raycast_steps: 10       # 减少射线数量
     prediction_steps: 5     # 减少预测步数
   ```

### 调试技巧

1. 启用详细日志:
   ```yaml
   logging:
     log_level: "DEBUG"
   ```

2. 生成调试数据:
   ```python
   debug_file = simulator.save_collision_debug_data()
   ```

3. 使用可视化工具分析结果

## 技术细节

### 射线投射算法

系统使用24条射线进行360度检测：
- 8条水平射线 (0°, 45°, 90°, ...)
- 8条向上倾斜射线 (35°仰角)
- 8条向下倾斜射线 (35°俯角)

### 物理接触点分析

接触点数据包括：
- 接触位置 (世界坐标)
- 接触法向量
- 接触力大小
- 渗透深度
- 活跃状态

### 预测算法

综合预测考虑：
1. 当前物理接触状态
2. 计划路径的分段检测  
3. 智能体间的最终位置冲突
4. 环境障碍物检测

这个高级碰撞检测系统提供了比原始实现更准确、更全面的碰撞检测能力，特别适用于复杂的多智能体导航场景。通过可视化工具和详细的调试信息，用户可以深入了解系统的运行状态并进行优化。
