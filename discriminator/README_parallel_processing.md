# 并行处理指南：使用 Controversial Episodes JSON 文件

本指南介绍如何使用预先生成的 `controversial_episodes.json` 文件来进行并行处理，从而大大加速 discriminator 分析过程。

## 概述

通过使用预先生成的 controversial episodes 文件，您可以：
1. 跳过重复的 controversy extraction 步骤
2. 使用 tmux 进行并行处理
3. 将大型数据集分割成小批次并行处理
4. 在多台机器上分布式处理

## 第一步：生成 Controversial Episodes 文件

首先，运行完整的 pipeline 一次来生成 controversial episodes 文件：

```bash
# 生成 controversial episodes 文件
python run_discriminator_pipeline.py \
    --config_path discriminator_config.json \
    --skip_discrimination \
    --skip_final
```

这将生成 `controversial_episodes.json` 文件，通常位于：
- `./discriminator/controversial_episodes.json`
- 或者在配置文件中指定的路径

## 第二步：检查 Controversial Episodes 文件格式

生成的文件格式如下：

```json
{
  "controversial_episodes": [
    {
      "scene_id": "p53SfW6mjZe",
      "episode_id": "1376", 
      "model1_success": true,
      "model2_success": false,
      "model1_target": [10.5, 20.3],
      "model2_target": [12.1, 18.7],
      "object_category": "chair"
    },
    ...
  ],
  "statistics": {
    "total_episodes": 1997,
    "controversial_episodes": 382,
    "controversy_rate": 0.191
  }
}
```

## 第三步：使用现有文件进行 Discrimination

### 方法 1：直接使用现有文件

```bash
# 使用现有的 controversial episodes 文件
python run_discriminator_pipeline.py \
    --config_path discriminator_config.json \
    --controversial_episodes /home/yaoaa/habitat-lab/discriminator/controversial_episodes.json \
    --skip_extraction
```

### 方法 2：仅运行 Discrimination 步骤

```bash
# 只运行 discrimination，跳过其他步骤
python discriminator_system.py \
    --config_path discriminator_config.json \
    --controversial_episodes controversial_episodes.json
```

## 第四步：并行处理设置

### 方法 1：分割文件进行并行处理

首先创建一个脚本来分割 controversial episodes 文件：

```python
# split_episodes.py
import json
import sys
from pathlib import Path

def split_episodes(input_file, num_chunks):
    """将 controversial episodes 文件分割成多个小文件"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    episodes = data['controversial_episodes']
    chunk_size = len(episodes) // num_chunks
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        if i == num_chunks - 1:
            # 最后一个 chunk 包含剩余的所有 episodes
            end_idx = len(episodes)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk_episodes = episodes[start_idx:end_idx]
        
        chunk_data = {
            "controversial_episodes": chunk_episodes,
            "statistics": {
                "total_episodes": len(chunk_episodes),
                "chunk_id": i + 1,
                "total_chunks": num_chunks
            }
        }
        
        output_file = f"controversial_episodes_chunk_{i+1}.json"
        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        print(f"创建 chunk {i+1}: {len(chunk_episodes)} episodes -> {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    num_chunks = int(sys.argv[2])
    split_episodes(input_file, num_chunks)
```

使用方法：
```bash
# 将 382 个 episodes 分割成 4 个文件
python split_episodes.py controversial_episodes.json 6
```

### 方法 2：使用 tmux 进行并行处理

创建 tmux 会话并在每个窗口中运行不同的 chunk：

```bash
#!/bin/bash
# parallel_discrimination.sh

# 创建新的 tmux 会话
tmux new-session -d -s discrimination_parallel

# 分割 episodes 文件
python split_episodes.py controversial_episodes.json 4

# 为每个 chunk 创建新的 tmux 窗口
for i in {1..4}; do
    # 创建新窗口
    tmux new-window -t discrimination_parallel -n "chunk_$i"
    
    # 在窗口中运行 discrimination
    tmux send-keys -t discrimination_parallel:chunk_$i \
        "cd /home/yaoaa/habitat-lab/discriminator" Enter
    tmux send-keys -t discrimination_parallel:chunk_$i \
        "python discriminator_system.py --config_path discriminator_config.json --controversial_episodes controversial_episodes_chunk_$i.json" Enter
done

# 显示所有窗口
tmux list-windows -t discrimination_parallel

echo "并行处理已启动!"
echo "使用 'tmux attach -t discrimination_parallel' 查看进度"
echo "使用 'tmux list-windows -t discrimination_parallel' 查看所有窗口状态"
```

### 方法 3：配置不同的输出目录

为了避免文件冲突，为每个并行进程配置不同的输出目录：

```bash
# 为每个 chunk 创建单独的配置文件
for i in {1..4}; do
    # 复制基础配置
    cp discriminator_config.json discriminator_config_chunk_$i.json
    
    # 修改输出目录
    sed -i "s|\"discriminator_output\": \"./discriminator/output\"|\"discriminator_output\": \"./discriminator/output_chunk_$i\"|g" discriminator_config_chunk_$i.json
done

# 使用不同的配置文件运行
python discriminator_system.py \
    --config_path discriminator_config_chunk_1.json \
    --controversial_episodes controversial_episodes_chunk_1.json
```

## 第五步：合并结果

处理完成后，您需要合并所有 chunk 的结果：

```python
# merge_results.py
import json
import glob
from pathlib import Path

def merge_discrimination_results():
    """合并所有 chunk 的 discrimination 结果"""
    all_results = []
    output_dirs = glob.glob("./discriminator/output_chunk_*")
    
    for output_dir in output_dirs:
        result_file = Path(output_dir) / "discrimination_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                chunk_data = json.load(f)
                all_results.extend(chunk_data.get('discriminated_results', []))
    
    # 合并统计信息
    total_episodes = len(all_results)
    model1_wins = sum(1 for r in all_results if r.get('decision') == 'model1')
    model2_wins = sum(1 for r in all_results if r.get('decision') == 'model2')
    
    merged_results = {
        "discriminated_results": all_results,
        "summary": {
            "total_episodes": total_episodes,
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "model1_win_rate": model1_wins / total_episodes if total_episodes > 0 else 0,
            "model2_win_rate": model2_wins / total_episodes if total_episodes > 0 else 0
        }
    }
    
    # 保存合并后的结果
    with open("./discriminator/output/discrimination_results_merged.json", 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"合并完成: {total_episodes} episodes")
    print(f"Model 1 wins: {model1_wins} ({model1_wins/total_episodes:.1%})")
    print(f"Model 2 wins: {model2_wins} ({model2_wins/total_episodes:.1%})")

if __name__ == "__main__":
    merge_discrimination_results()
```

## 第六步：监控进度

### 查看 tmux 会话状态
```bash
# 列出所有 tmux 会话
tmux list-sessions

# 连接到并行处理会话
tmux attach -t discrimination_parallel

# 查看特定窗口
tmux select-window -t discrimination_parallel:chunk_1
```

### 监控各个 chunk 的进度
```bash
# 检查日志文件
tail -f ./discriminator/output_chunk_*/discriminator.log

# 检查已完成的 episodes 数量
for i in {1..4}; do
    echo "Chunk $i:"
    find ./discriminator/output_chunk_$i -name "discrimination_result.json" | wc -l
done
```

## 性能优化建议

### 1. 根据硬件调整并行数量
```bash
# 基于 CPU 核心数确定 chunk 数量
NUM_CORES=$(nproc)
NUM_CHUNKS=$((NUM_CORES / 2))  # 保守估计，避免过载
```

### 2. 内存使用优化
```json
{
  "discrimination_config": {
    "max_episodes_per_batch": 5,  // 减少批次大小以节省内存
    "save_intermediate_results": false  // 跳过中间结果以节省磁盘空间
  }
}
```

### 3. 网络优化（对于 LLM 调用）
```json
{
  "llm_config": {
    "max_tokens": 35000,
    "timeout": 30,  // 设置超时
    "retry_attempts": 3  // 设置重试次数
  }
}
```

## 完整的并行处理工作流程

```bash
#!/bin/bash
# complete_parallel_workflow.sh

echo "=== 步骤 1: 生成 Controversial Episodes 文件 ==="
python run_discriminator_pipeline.py \
    --config_path discriminator_config.json \
    --skip_discrimination --skip_final

echo "=== 步骤 2: 分割文件 ==="
python split_episodes.py controversial_episodes.json 4

echo "=== 步骤 3: 设置并行配置 ==="
for i in {1..4}; do
    cp discriminator_config.json discriminator_config_chunk_$i.json
    sed -i "s|\"discriminator_output\": \"./discriminator/output\"|\"discriminator_output\": \"./discriminator/output_chunk_$i\"|g" discriminator_config_chunk_$i.json
done

echo "=== 步骤 4: 启动并行处理 ==="
tmux new-session -d -s discrimination_parallel
for i in {1..4}; do
    tmux new-window -t discrimination_parallel -n "chunk_$i"
    tmux send-keys -t discrimination_parallel:chunk_$i \
        "cd /home/yaoaa/habitat-lab/discriminator && python discriminator_system.py --config_path discriminator_config_chunk_$i.json --controversial_episodes controversial_episodes_chunk_$i.json" Enter
done

echo "=== 并行处理已启动! ==="
echo "使用 'tmux attach -t discrimination_parallel' 查看进度"
echo "处理完成后运行 'python merge_results.py' 合并结果"
```

## 故障排除

### 1. 内存不足
- 减少 `max_episodes_per_batch`
- 减少并行 chunk 数量
- 增加系统 swap 空间

### 2. LLM API 限制
- 增加请求间隔时间
- 使用不同的 API keys 分布请求
- 设置合理的超时和重试机制

### 3. 磁盘空间不足
- 设置 `save_intermediate_results: false`
- 定期清理临时文件
- 使用符号链接到更大的存储设备

通过这种方式，您可以显著加速 discriminator 分析过程，特别是对于大型数据集。并行处理可以将处理时间从几小时缩短到几十分钟。
