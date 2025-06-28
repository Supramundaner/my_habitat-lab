#!/bin/bash

# Habitat Video Generator 启动脚本

# 设置默认参数
SCENE_PATH="/home/yaoaa/habitat-lab/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
GPU_ID=0
FPS=30
OUTPUT_DIR="./outputs"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE_PATH="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test)
            echo "Running test..."
            python test.py
            exit $?
            ;;
        --help|-h)
            echo "Habitat Video Generator"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scene PATH      Path to .glb scene file (default: van-gogh-room.glb)"
            echo "  --gpu ID          CUDA device ID (default: 0)"
            echo "  --fps FPS         Video frame rate (default: 30)"
            echo "  --output-dir DIR  Output directory (default: ./outputs)"
            echo "  --test            Run test mode"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --scene /path/to/scene.glb --gpu 0 --fps 30"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 检查场景文件是否存在
if [ ! -f "$SCENE_PATH" ]; then
    echo "Error: Scene file not found: $SCENE_PATH"
    echo "Available test scenes:"
    find /home/yaoaa/habitat-lab/data -name "*.glb" 2>/dev/null | head -5
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 显示启动信息
echo "================================"
echo "Habitat Video Generator"
echo "================================"
echo "Scene: $SCENE_PATH"
echo "GPU Device: $GPU_ID"
echo "FPS: $FPS"
echo "Output Directory: $OUTPUT_DIR"
echo "================================"
echo ""

# 启动程序
python main.py \
    --scene "$SCENE_PATH" \
    --gpu "$GPU_ID" \
    --fps "$FPS" \
    --output-dir "$OUTPUT_DIR"
