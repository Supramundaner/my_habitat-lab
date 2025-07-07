# 高级碰撞检测系统 - 快速开始指南

## 🚀 快速开始

### 1. 运行测试

```bash
cd video_app
python test_advanced_collision.py
```

### 2. 查看结果

测试完成后，检查以下输出：

```bash
outputs/collision_tests/
├── 📊 collision_test_debug.json      # 调试数据
├── 📈 agent_positions.png            # 位置图 (需运行可视化)
├── 🎥 agent_0_output.mp4            # 智能体视频
└── 📝 advanced_collision_test.log    # 详细日志
```

### 3. 生成可视化

```bash
python visualize_collision_results.py
```

## 🔧 配置示例

### 基础配置

```yaml
# config/my_collision_test.yaml
collision_detection:
  enabled: true
  agent_radius: 0.15
  min_agent_distance: 1.0
  prediction_steps: 15
  raycast_steps: 25
```

### 性能优化配置

```yaml
# 高性能设置（减少计算量）
collision_detection:
  enabled: true
  prediction_steps: 5      # 更少预测步数
  raycast_steps: 10        # 更少射线
  contact_threshold: 0.02  # 更大阈值
```

### 高精度配置

```yaml
# 高精度设置（更准确检测）
collision_detection:
  enabled: true
  prediction_steps: 20     # 更多预测步数
  raycast_steps: 30        # 更多射线
  contact_threshold: 0.001 # 更小阈值
  agent_radius: 0.05       # 更小半径
```

## 📊 实时监控

### 在代码中检查状态

```python
# 初始化
simulator = MultiAgentSimulator(config)

# 实时监控
while simulation_running:
    # 获取统计信息
    stats = simulator.collision_detector.get_collision_statistics()
    print(f"当前接触点: {stats['last_contact_count']}")
    
    # 生成报告
    report = simulator.generate_collision_report()
    
    # 性能基准
    if need_benchmark:
        perf = simulator.benchmark_collision_detection(num_tests=10)
        print(f"检测时间: {perf['comprehensive_prediction_time']:.4f}秒")
```

## 🎯 常见使用场景

### 场景1：多机器人协作

```yaml
# 适合多个物理机器人
collision_detection:
  enabled: true
  min_agent_distance: 1.5  # 更大安全距离
  prediction_steps: 20     # 更多预测
```

### 场景2：密集环境导航

```yaml
# 适合复杂环境
collision_detection:
  enabled: true
  raycast_steps: 30        # 更多环境检测
  agent_radius: 0.2        # 考虑机器人尺寸
```

### 场景3：高速运动

```yaml
# 适合快速移动
collision_detection:
  enabled: true
  prediction_steps: 25     # 更远预测
  contact_threshold: 0.005 # 更敏感
movement:
  linear_speed: 2.0        # 更快速度
```

## 🐛 故障排除

### 问题1：检测不到碰撞
```yaml
# 解决方案：提高敏感度
collision_detection:
  agent_radius: 0.2        # 增大检测半径
  contact_threshold: 0.001 # 减小阈值
  min_agent_distance: 1.2  # 增大安全距离
```

### 问题2：误报过多
```yaml
# 解决方案：降低敏感度
collision_detection:
  agent_radius: 0.05       # 减小检测半径
  contact_threshold: 0.02  # 增大阈值
  raycast_steps: 10        # 减少射线
```

### 问题3：性能问题
```yaml
# 解决方案：优化性能
collision_detection:
  prediction_steps: 5      # 减少预测
  raycast_steps: 8         # 减少射线
video_output:
  fps: 15                  # 降低帧率
  resolution: [256, 512]   # 降低分辨率
```

## 📈 性能指标

### 典型基准测试结果

```
✓ 物理接触检测:    ~0.001秒
✓ 射线投射检测:    ~0.005秒  
✓ 综合碰撞预测:    ~0.010秒
✓ 平均接触点数:    0-5个
```

### 如何改进性能

1. **减少预测步数** (`prediction_steps: 5-10`)
2. **减少射线数量** (`raycast_steps: 8-15`)
3. **调整检测频率** (每N帧检测一次)
4. **使用虚拟智能体** (避免物理机器人开销)

## 🎮 交互式调试

### 启用实时可视化

```python
# 在代码中
simulator.enable_collision_visualization()

# 获取可视化数据
viz_data = simulator.get_collision_visualization_data()
if viz_data['contact_points']:
    print(f"发现 {len(viz_data['contact_points'])} 个接触点")
```

### 保存调试会话

```python
# 自动保存调试数据
debug_file = simulator.save_collision_debug_data("session_debug.json")

# 生成详细报告
report_file = "outputs/my_collision_report.txt"
with open(report_file, 'w') as f:
    f.write(simulator.generate_collision_report())
```

## 🔗 下一步

1. **运行基础测试** → 验证系统工作
2. **调整配置参数** → 适应你的场景
3. **集成到项目** → 使用API监控碰撞
4. **优化性能** → 根据需求调整
5. **分析结果** → 使用可视化工具

---

💡 **提示**: 从默认配置开始，然后根据实际需求逐步调整参数。使用可视化工具帮助理解系统行为。
