#!/bin/bash

# Object Detection 服务启动脚本
# 用于启动Grounding DINO和Mobile SAM后台服务

echo "启动Object Detection后台服务..."

# 检查VLFM目录是否存在
if [ ! -d "vlfm" ]; then
    echo "错误: vlfm目录不存在，请确保VLFM已正确安装"
    exit 1
fi

# 进入VLFM目录
cd vlfm

# 检查启动脚本是否存在
if [ ! -f "scripts/launch_vlm_servers.sh" ]; then
    echo "错误: launch_vlm_servers.sh脚本不存在"
    exit 1
fi

# 启动服务
echo "正在启动VLFM模型服务..."
bash scripts/launch_vlm_servers.sh

echo "服务启动完成！"
echo "请等待90秒让模型完全加载..."
echo "可以使用以下命令查看服务状态："
echo "tmux list-sessions"
echo "tmux attach-session -t vlm_servers_XXXX" 