# One-Stage Method Implementation Summary

本文档总结了为实现one-stage导航目标选择方法而进行的所有修改。

## 修改概述

我们成功地将原有的two-stage方法扩展为支持one-stage方法，主要修改包括：

### 1. onestage_preprocess/ 新增文件

#### 核心文件
- `step_3_onestage_node_selection.py` - 新的one-stage节点选择实现
- `main_workflow_onestage.py` - 专用的one-stage工作流程
- `input_config_onestage.json` - one-stage配置文件
- `README_onestage.md` - 使用说明文档

#### Prompt文件
- `prompts/choose_onestage_node.txt` - one-stage节点选择prompt

#### 修改的文件
- `main_workflow.py` - 添加了对one-stage模式的支持
- `input_config.json` - 明确标记为two-stage配置

### 2. onestage_eval/ 修改文件

#### 主要修改
- `run_eval.py` - 更新导入路径和配置，使用OneStageWorkflowOrchestrator
- `prompts/choose_onestage_node.txt` - 复制了one-stage prompt
- `test_config.json` - 更新为one-stage配置
- `eval_config_episode_427.json` - 更新为one-stage配置
- `batch_eval.json` - 更新为one-stage配置
- `README.md` - 新增one-stage使用说明

## 主要技术差异

### Two-Stage Method (原方法)
```
Step 0: 生成topdown视图
Step 1: 生成墙体掩码
Step 2: 房间分割和标注
Step 3: LLM选择目标房间
Step 4: 生成导航图
Step 5: LLM在选定房间内选择节点
Step 6: 路径规划
```

### One-Stage Method (新方法)
```
Step 0: 生成topdown视图
Step 1: 生成墙体掩码
Step 2: 生成导航图 (跳过房间分割)
Step 3: LLM直接从整个环境选择节点
Step 4: 路径规划
```

## 关键修改细节

### 1. 工作流程简化
- 跳过房间分割步骤（step_2_room_segmentation.py）
- 跳过房间选择步骤（step_3_llm_room_selection.py）
- 将图生成提前到step 2
- 使用新的step_3_onestage_node_selection.py直接选择节点

### 2. 配置更新
```json
{
    "workflow_type": "one_stage",  // 新增工作流类型标识
    "wall_mask": {                 // 替换room_segmentation配置
        "morph_closing_width_meters": 0.01
    },
    "prompts": {
        "choose_node_prompt": "prompts/choose_onestage_node.txt"  // 只需一个prompt
    }
}
```

### 3. LLM Prompt优化
one-stage prompt具有以下特点：
- 一次性分析整个环境
- 考虑房间上下文但不依赖房间分割
- 直接选择最优导航节点
- 包含详细的物体-房间关联知识

### 4. 导入路径修改
```python
# onestage_eval/run_eval.py 中的关键修改
onestage_preprocess_root = project_root / "onestage_preprocess"
from main_workflow_onestage import OneStageWorkflowOrchestrator
```

## 使用方法

### onestage_preprocess/
```bash
# Two-stage方法
python main_workflow.py input_config.json

# One-stage方法（方式1）
python main_workflow.py input_config_onestage.json

# One-stage方法（方式2）
python main_workflow_onestage.py input_config_onestage.json
```

### onestage_eval/
```bash
# 单个episode评估
python run_eval.py test_config.json

# 批量评估
python batch_eval.py batch_eval.json
```

## 优势对比

### One-Stage优势
1. **处理速度快**: 减少2个步骤，只需1次LLM调用
2. **内存占用少**: 不需要房间分割的中间结果
3. **适用性强**: 适合开放式布局或房间边界不清晰的环境
4. **实现简单**: 减少了复杂的房间分割逻辑

### Two-Stage优势
1. **精度更高**: 先选房间再选节点，决策更精准
2. **可解释性强**: 两步决策过程更容易理解和调试
3. **适合复杂环境**: 对于有明确房间边界的环境表现更好
4. **容错性好**: 如果房间选择错误，节点选择仍可能成功

## 注意事项

1. **导入依赖**: onestage_eval依赖onestage_preprocess中的文件
2. **配置兼容性**: 确保使用对应的配置文件格式
3. **视频生成**: 视频生成和导航部分保持不变
4. **评估指标**: SR/SPL等评估指标计算方式不变

## 文件结构对比

### 修改前
```
preprocess/
├── main_workflow.py (two-stage only)
├── step_3_llm_room_selection.py
├── step_5_node_selection.py
└── prompts/
    ├── choose_a_room.txt
    └── choose_a_node.txt
```

### 修改后
```
onestage_preprocess/
├── main_workflow.py (支持both)
├── main_workflow_onestage.py (one-stage专用)
├── step_3_llm_room_selection.py (保留)
├── step_3_onestage_node_selection.py (新增)
├── step_5_node_selection.py (保留)
└── prompts/
    ├── choose_a_room.txt
    ├── choose_a_node.txt
    └── choose_onestage_node.txt (新增)
```

这些修改实现了完整的one-stage导航目标选择系统，同时保持了与原有two-stage系统的兼容性。
