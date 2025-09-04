#!/bin/bash

# Local VLM 服务启动脚本
# 使用独立的VLM模块，不依赖外部VLFM项目

echo "启动本地VLM服务..."

# 设置conda环境名称
CONDA_ENV_NAME=${CONDA_ENV_NAME:-vlfm}

# 激活conda环境
echo "激活conda环境: ${CONDA_ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

# 检查conda环境是否激活成功
if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 ${CONDA_ENV_NAME}"
    echo "请确保环境存在，或设置正确的CONDA_ENV_NAME环境变量"
    exit 1
fi

echo "成功激活conda环境: ${CONDA_ENV_NAME}"
echo "Python路径: $(which python)"

# 检查VLM目录是否存在
if [ ! -d "vlm" ]; then
    echo "错误: vlm目录不存在"
    exit 1
fi

# 检查必要的文件是否存在
if [ ! -f "vlm/grounding_dino.py" ] || [ ! -f "vlm/sam.py" ] || [ ! -f "vlm/yolov7.py" ]; then
    echo "错误: VLM模块文件不完整"
    exit 1
fi

# 设置环境变量
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export SAM_PORT=${SAM_PORT:-12184}
export YOLOV7_PORT=${YOLOV7_PORT:-12185}

session_name=local_vlm_servers_${RANDOM}

# 创建tmux会话
tmux new-session -d -s ${session_name}

# 分割窗口创建3个pane
tmux split-window -h -t ${session_name}:0
tmux split-window -v -t ${session_name}:0.1

# 在第一个pane启动Grounding DINO（在激活的环境中）
tmux send-keys -t ${session_name}:0.0 "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && python -m vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m

# 在第二个pane启动Mobile SAM（在激活的环境中）
tmux send-keys -t ${session_name}:0.1 "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && python -m vlm.sam --port ${SAM_PORT}" C-m

# 在第三个pane启动YOLOv7（在激活的环境中）
tmux send-keys -t ${session_name}:0.2 "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && python -m vlm.yolov7 --port ${YOLOV7_PORT}" C-m

echo "创建了tmux会话 '${session_name}'"
echo "请等待90秒让模型完全加载..."
echo "可以使用以下命令查看服务状态："
echo "  tmux list-sessions"
echo "  tmux attach-session -t ${session_name}"
echo ""
echo "要停止所有服务："
echo "  tmux kill-session -t ${session_name}" 