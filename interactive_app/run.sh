#!/bin/bash
# 启动Habitat Interactive Navigator

echo "🚀 启动Habitat Interactive Navigator..."
echo "================================="

# 检查是否在正确的目录
if [ ! -f "src/main.py" ]; then
    echo "❌ 错误: 请在interactive_app目录下运行此脚本"
    echo "   cd /home/yaoaa/habitat-lab/interactive_app"
    echo "   ./run.sh"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查是否有GUI环境
if [ -z "$DISPLAY" ]; then
    echo "⚠️  警告: 没有检测到GUI环境"
    echo "   运行功能验证测试..."
    cd tests
    python3 final_verification.py
else
    echo "✅ 检测到GUI环境"
    echo "   启动完整应用..."
    cd src
    python3 main.py
fi
