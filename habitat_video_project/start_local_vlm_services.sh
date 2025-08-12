#!/bin/bash

# Local VLM 服务启动脚本
# 使用独立的VLM模块，不依赖外部VLFM项目

echo "启动本地VLM服务..."

# 检查VLM目录是否存在
if [ ! -d "vlm" ]; then
    echo "错误: vlm目录不存在"
    exit 1
fi

# 检查必要的文件是否存在
if [ ! -f "vlm/grounding_dino.py" ] || [ ! -f "vlm/sam.py" ]; then
    echo "错误: VLM模块文件不完整"
    exit 1
fi

# 设置环境变量
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export SAM_PORT=${SAM_PORT:-12183}

session_name=local_vlm_servers_${RANDOM}

# 创建tmux会话
tmux new-session -d -s ${session_name}

# 分割窗口
tmux split-window -h -t ${session_name}:0

# 在第一个pane启动Grounding DINO
tmux send-keys -t ${session_name}:0.0 "cd $(pwd) && python -m vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m

# 在第二个pane启动Mobile SAM
tmux send-keys -t ${session_name}:0.1 "cd $(pwd) && python -m vlm.sam --port ${SAM_PORT}" C-m

echo "创建了tmux会话 '${session_name}'"
echo "请等待90秒让模型完全加载..."
echo "可以使用以下命令查看服务状态："
echo "tmux list-sessions"
echo "tmux attach-session -t ${session_name}" 