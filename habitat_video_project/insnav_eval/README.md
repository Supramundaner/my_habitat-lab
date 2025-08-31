# Image Instance Navigation Evaluation (InsnNav Eval)

This directory contains the evaluation pipeline for Image Instance Navigation tasks. Unlike Object Navigation where the goal can be any object of a specific category, Image Instance Navigation requires the agent to navigate to a specific object instance identified by a unique image goal.

## Overview

The evaluation pipeline consists of three main stages:
1. **Preprocessing** (`insnav_preprocess`): Generates navigation actions using LLM-guided room selection and graph-based path planning
2. **Video Generation**: Executes the navigation actions in Habitat simulator and records the agent's journey
3. **Evaluation**: Measures navigation success based on proximity to target viewpoints

## Files

- `run_eval.py`: Single episode evaluation script
- `batch_eval.py`: Batch evaluation script for multiple episodes
- `example_config.json`: Example configuration for single episode evaluation
- `batch_config_example.json`: Example configuration for batch evaluation
- `README.md`: This documentation file

## Key Differences from Object Navigation

1. **Goal Specificity**: Targets a unique object instance with specific image goals, not any object of a category
2. **Success Criteria**: Agent must reach viewpoints of the specific target object (identified by `goal_object_id` and `goal_image_id`)
3. **Preprocessing Pipeline**: Uses `insnav_preprocess` instead of `preprocess`, which generates goal images and performs image-guided navigation planning

## 使用方法 / Usage

### 单集评估 / Single Episode Evaluation

```bash
# Navigate to the insnav_eval directory
cd /home/yaoaa/habitat-lab/habitat_video_project/insnav_eval

# Run single episode evaluation
python run_eval.py example_config.json

# With verbose output for debugging
python run_eval.py example_config.json --verbose
```

### 批量评估 / Batch Evaluation

```bash
# Run batch evaluation on multiple episodes
python batch_eval.py batch_config_example.json
```

## 配置文件说明

### 单集配置 (`example_config.json`)

```json
{
    "preprocess": {
        "scene_config": {
            "custom_ortho_scale": null,
            "target_coverage": 0.9,
            "draw_coordinates": false
        },
        "resolution": [2048, 2048],
        "room_segmentation": { ... },
        "graph_generation": { ... },
        "llm_config": { ... },
        "prompts": { ... }
    },
    "video_generation": { ... },
    "evaluation": {
        "success_distance_threshold": 0.25  // 成功判定距离阈值（米）
    },
    "scene": {
        "robot_urdf": "/path/to/robot.urdf"
    },
    "episode": {
        "episode_json_path": "/path/to/episodes.json",
        "episode_id": "13"
    }
}
```

### 批量配置 (`batch_config_example.json`)

```json
{
    "preprocess": { ... },
    "video_generation": { ... },
    "evaluation": { ... },
    "scene": { ... },
    "evaluation_tasks": [
        {
            "scene_file": "/path/to/scene.glb",
            "episode_json_path": "/path/to/episodes.json",
            "episode_ids": ["13", "14", "15"]
        }
    ]
}
```

## 评估指标

系统使用以下指标评估导航性能：

1. **Success Rate (SR)**: 成功到达目标位置的比率
2. **SPL (Success weighted by Path Length)**: 考虑路径长度的成功率
3. **Distance to Target**: 到达目标视点的最小距离

### 成功判定标准

- 智能体最终位置与目标对象的任意视点（viewpoints）距离 ≤ 0.25m
- 注意：与物体导航不同，这里的目标是**特定的图像目标**，而不是该类别的任意对象

## 输出结果

### 单集结果 (`evaluation_results.json`)

```json
{
    "config": { ... },
    "episode_data": { ... },
    "preprocessing_success": true,
    "video_generation_success": true,
    "evaluation_results": {
        "success": true,
        "sr": 1.0,
        "spl": 0.85,
        "min_distance_to_target": 0.12,
        "success_threshold": 0.25,
        "final_position": [x, y, z],
        "total_viewpoints_checked": 15,
        "path_length": 25
    },
    "errors": []
}
```

### 批量结果 (`batch_output.json`)

```json
{
    "batch_summary": {
        "total_episodes_processed": 6,
        "succeeded": 4,
        "failed": 2,
        "overall_sr": 0.667,
        "overall_spl": 0.542
    },
    "successful_episodes": [...],
    "failed_episodes": [...],
    "episode_details": { ... }
}
```

## 依赖关系

本评估系统依赖以下组件：

1. **insnav_preprocess**: 预处理管道
   - 位置：`/home/yaoaa/habitat-lab/insnav_preprocess/`
   - 输出：`action.json` 文件
   
2. **habitat_video_project/src**: 视频生成
   - 模拟器、视频合成器等组件
   
3. **Habitat-Sim**: 3D 环境模拟和导航

## 关键区别：Object Navigation vs Image Instance Navigation

| 特性 | Object Navigation | Image Instance Navigation |
|------|-------------------|---------------------------|
| 目标定义 | 任意该类别的对象 | 特定的图像目标 |
| 成功判定 | 到达任意同类对象 | 到达特定对象的视点 |
| 输入数据 | 类别标签 | 目标图像 + 对象ID |
| 预处理 | main_workflow.py | main.py (insnav_preprocess) |
| 输出目录 | preprocess/output | insnav_preprocess/output_insnav |

## 故障排除

### 常见问题

1. **ImportError**: 确保所有路径正确设置
   ```bash
   export PYTHONPATH="/home/yaoaa/habitat-lab/insnav_preprocess:$PYTHONPATH"
   ```

2. **action.json 未找到**: 检查 insnav_preprocess 是否成功完成

3. **GPU 内存不足**: 在配置中设置：
   ```json
   "gpu": {
       "memory_efficient": true,
       "max_chunk_size": 10000
   }
   ```

### 调试选项

- 使用 `--verbose` 获取详细错误信息
- 检查各阶段的输出日志
- 验证配置文件格式

## 示例工作流

1. **准备数据**:
   ```bash
   # 确保 episodes.json 包含 image_goals 和 view_points
   # 确保场景文件可访问
   ```

2. **运行单集评估**:
   ```bash
   python run_eval.py example_config.json --verbose
   ```

3. **检查结果**:
   ```bash
   cat output/[scene_name]/[episode_id]/evaluation_results.json
   ```

4. **批量评估**:
   ```bash
   python batch_eval.py batch_config_example.json
   cat output/batch_output.json
   ```

## 示例工作流

1. **准备数据**:
   ```bash
   # 确保 episodes.json 包含 image_goals 和 view_points
   # 确保场景文件可访问
   ```

2. **运行单集评估**:
   ```bash
   python run_eval.py example_config.json --verbose
   ```

3. **检查结果**:
   ```bash
   cat output/[scene_name]/[episode_id]/evaluation_results.json
   ```

4. **批量评估**:
   ```bash
   python batch_eval.py batch_config_example.json
   cat output/batch_output.json
   ```

## 常见问题解决 / Troubleshooting

### Import 错误
```bash
# 如果遇到导入错误，检查路径设置
export PYTHONPATH="/home/yaoaa/habitat-lab/insnav_preprocess:$PYTHONPATH"
```

### 内存不足
```json
// 在配置中启用内存优化
"gpu": {
    "memory_efficient": true,
    "max_chunk_size": 10000
}
```

### 调试模式
```bash
# 使用详细输出查看错误详情
python run_eval.py example_config.json --verbose
```

## 输出文件结构 / Output Structure

```
insnav_eval/output/
├── [scene_name]/              # 按场景分组
│   └── [episode_id]/          # 按集数分组
│       ├── insnav_preprocess_output/  # 预处理输出
│       │   ├── action.json           # 导航动作序列
│       │   ├── topdown_view.png      # 顶视图
│       │   └── ...
│       ├── video_output/             # 视频生成输出
│       │   ├── navigation_video.mp4  # 导航视频
│       │   └── ...
│       ├── evaluation_results.json  # 详细评估指标
│       └── output.json              # 完整管道结果
└── batch_output.json                # 批量评估汇总
```

---

**注意**: 确保在运行评估前，`insnav_preprocess` 管道已正确设置并可用。
