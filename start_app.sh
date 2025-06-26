#!/bin/bash

# Habitat导航应用程序启动脚本

echo "启动Habitat导航应用程序..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 检查PyQt5是否安装
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "警告: PyQt5 未安装，正在尝试安装..."
    pip3 install PyQt5
fi

# 检查其他依赖
echo "检查依赖..."
pip3 install -r requirements.txt

# 设置环境变量
export QT_QPA_PLATFORM_PLUGIN_PATH=""

# 运行应用程序
echo "启动应用程序..."
python3 habitat_navigator_app.py

echo "应用程序已退出"
