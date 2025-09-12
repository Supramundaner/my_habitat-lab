#!/bin/bash
# parallel_discrimination.sh
# 自动化并行处理脚本

set -e  # 遇到错误时退出

# 默认配置
CONFIG_FILE="discriminator_config.json"
CONTROVERSIAL_EPISODES_FILE="controversial_episodes.json"
NUM_CHUNKS=4
OUTPUT_PREFIX="controversial_episodes_chunk"
USE_TMUX=true
SESSION_NAME="discrimination_parallel"

# 帮助信息
show_help() {
    cat << EOF
并行 Discrimination 处理脚本

用法: $0 [选项]

选项:
    -c, --config FILE           配置文件路径 (默认: discriminator_config.json)
    -e, --episodes FILE         controversial episodes 文件路径 (默认: controversial_episodes.json)
    -n, --chunks NUM           分割块数 (默认: 4)
    -p, --prefix PREFIX        输出文件前缀 (默认: controversial_episodes_chunk)
    -s, --session NAME         tmux 会话名称 (默认: discrimination_parallel)
    --no-tmux                  不使用 tmux，直接在后台运行
    --dry-run                  仅显示将要执行的命令，不实际执行
    -h, --help                 显示此帮助信息

示例:
    $0                                              # 使用默认设置
    $0 -n 8 -c my_config.json                     # 使用8个块和自定义配置
    $0 --no-tmux -n 6                             # 不使用tmux，使用6个块
    $0 --dry-run -n 4                             # 预览模式

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -e|--episodes)
            CONTROVERSIAL_EPISODES_FILE="$2"
            shift 2
            ;;
        -n|--chunks)
            NUM_CHUNKS="$2"
            shift 2
            ;;
        -p|--prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -s|--session)
            SESSION_NAME="$2"
            shift 2
            ;;
        --no-tmux)
            USE_TMUX=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 执行命令函数
execute_command() {
    local cmd="$1"
    local description="$2"
    
    log_info "$description"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] $cmd"
    else
        echo "  执行: $cmd"
        if eval "$cmd"; then
            log_success "$description 完成"
        else
            log_error "$description 失败"
            return 1
        fi
    fi
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python 未安装"
        exit 1
    fi
    
    # 检查必要文件
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    if [[ ! -f "$CONTROVERSIAL_EPISODES_FILE" ]]; then
        log_error "Controversial episodes 文件不存在: $CONTROVERSIAL_EPISODES_FILE"
        exit 1
    fi
    
    # 检查 tmux (如果需要)
    if [[ "$USE_TMUX" == "true" ]] && ! command -v tmux &> /dev/null; then
        log_warning "tmux 未安装，切换到后台运行模式"
        USE_TMUX=false
    fi
    
    # 检查必要的Python脚本
    for script in "split_episodes.py" "discriminator_system.py" "merge_results.py"; do
        if [[ ! -f "$script" ]]; then
            log_error "必要脚本不存在: $script"
            exit 1
        fi
    done
    
    log_success "依赖检查完成"
}

# 分割episodes文件
split_episodes() {
    log_info "分割 controversial episodes 文件..."
    
    local cmd="python split_episodes.py \"$CONTROVERSIAL_EPISODES_FILE\" $NUM_CHUNKS --output_prefix \"$OUTPUT_PREFIX\""
    execute_command "$cmd" "分割 episodes 文件"
}

# 创建配置文件
create_chunk_configs() {
    log_info "为每个块创建单独的配置文件..."
    
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        local chunk_config="${CONFIG_FILE%.json}_chunk_$i.json"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] 创建配置文件: $chunk_config"
        else
            # 复制原始配置文件
            cp "$CONFIG_FILE" "$chunk_config"
            
            # 修改输出目录
            if command -v sed &> /dev/null; then
                sed -i "s|\"discriminator_output\": \"./discriminator/output\"|\"discriminator_output\": \"./discriminator/output_chunk_$i\"|g" "$chunk_config"
            else
                # 如果没有sed，使用Python
                python -c "
import json
with open('$chunk_config', 'r') as f:
    config = json.load(f)
config['output_config']['discriminator_output'] = './discriminator/output_chunk_$i'
with open('$chunk_config', 'w') as f:
    json.dump(config, f, indent=2)
"
            fi
            echo "  创建配置文件: $chunk_config"
        fi
    done
    
    log_success "配置文件创建完成"
}

# 使用tmux启动并行处理
start_tmux_processing() {
    log_info "使用 tmux 启动并行处理..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] tmux new-session -d -s $SESSION_NAME"
        for ((i=1; i<=NUM_CHUNKS; i++)); do
            echo "  [DRY RUN] tmux new-window -t $SESSION_NAME -n chunk_$i"
            echo "  [DRY RUN] 在窗口中运行: python discriminator_system.py --config_path ${CONFIG_FILE%.json}_chunk_$i.json --controversial_episodes ${OUTPUT_PREFIX}_$i.json"
        done
        return
    fi
    
    # 创建新的tmux会话
    tmux new-session -d -s "$SESSION_NAME" || {
        log_warning "tmux 会话 $SESSION_NAME 已存在，先终止旧会话"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        tmux new-session -d -s "$SESSION_NAME"
    }
    
    # 为每个块创建新窗口并启动处理
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        local chunk_config="${CONFIG_FILE%.json}_chunk_$i.json"
        local chunk_episodes="${OUTPUT_PREFIX}_$i.json"
        
        # 创建新窗口
        tmux new-window -t "$SESSION_NAME" -n "chunk_$i"
        
        # 在窗口中运行discrimination
        local cmd="cd $(pwd) && python discriminator_system.py --config_path \"$chunk_config\" --controversial_episodes \"$chunk_episodes\""
        tmux send-keys -t "$SESSION_NAME:chunk_$i" "$cmd" Enter
        
        log_info "启动 chunk $i 处理"
    done
    
    log_success "所有 chunks 已在 tmux 中启动"
    echo ""
    echo "监控命令:"
    echo "  查看所有窗口: tmux list-windows -t $SESSION_NAME"
    echo "  连接到会话: tmux attach -t $SESSION_NAME"
    echo "  查看特定窗口: tmux select-window -t $SESSION_NAME:chunk_1"
}

# 使用后台进程启动并行处理
start_background_processing() {
    log_info "使用后台进程启动并行处理..."
    
    local pids=()
    
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        local chunk_config="${CONFIG_FILE%.json}_chunk_$i.json"
        local chunk_episodes="${OUTPUT_PREFIX}_$i.json"
        local log_file="chunk_$i.log"
        
        local cmd="python discriminator_system.py --config_path \"$chunk_config\" --controversial_episodes \"$chunk_episodes\""
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] 后台运行: $cmd > $log_file 2>&1 &"
        else
            echo "  启动 chunk $i: $cmd"
            eval "$cmd" > "$log_file" 2>&1 &
            local pid=$!
            pids+=($pid)
            echo "  Chunk $i PID: $pid, 日志: $log_file"
        fi
    done
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log_success "所有 chunks 已在后台启动"
        echo ""
        echo "监控命令:"
        echo "  查看进程: ps aux | grep discriminator_system"
        echo "  查看日志: tail -f chunk_*.log"
        echo "  等待完成: wait ${pids[*]}"
        
        # 保存PIDs到文件以便后续使用
        printf "%s\n" "${pids[@]}" > parallel_pids.txt
        echo "  进程IDs已保存到: parallel_pids.txt"
    fi
}

# 等待处理完成
wait_for_completion() {
    if [[ "$USE_TMUX" == "true" ]]; then
        log_info "检查 tmux 会话中的处理进度..."
        echo "使用以下命令监控进度:"
        echo "  tmux attach -t $SESSION_NAME"
        echo "  tmux list-windows -t $SESSION_NAME"
    else
        log_info "等待后台进程完成..."
        if [[ -f "parallel_pids.txt" ]]; then
            local pids=($(cat parallel_pids.txt))
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    log_info "等待进程 $pid 完成..."
                    wait "$pid"
                fi
            done
            log_success "所有后台进程已完成"
        fi
    fi
}

# 合并结果
merge_results() {
    log_info "合并处理结果..."
    
    local output_pattern="./discriminator/output_chunk_*"
    local merged_file="./discriminator/output/discrimination_results_merged.json"
    
    # 确保输出目录存在
    mkdir -p "./discriminator/output"
    
    local cmd="python merge_results.py --output_pattern \"$output_pattern\" --output_file \"$merged_file\""
    execute_command "$cmd" "合并结果"
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] 清理配置文件和分割文件"
        return
    fi
    
    # 清理chunk配置文件
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        local chunk_config="${CONFIG_FILE%.json}_chunk_$i.json"
        if [[ -f "$chunk_config" ]]; then
            rm "$chunk_config"
            echo "  删除: $chunk_config"
        fi
    done
    
    # 清理分割的episodes文件
    for ((i=1; i<=NUM_CHUNKS; i++)); do
        local chunk_episodes="${OUTPUT_PREFIX}_$i.json"
        if [[ -f "$chunk_episodes" ]]; then
            rm "$chunk_episodes"
            echo "  删除: $chunk_episodes"
        fi
    done
    
    # 清理摘要文件
    local summary_file="${OUTPUT_PREFIX}_summary.json"
    if [[ -f "$summary_file" ]]; then
        rm "$summary_file"
        echo "  删除: $summary_file"
    fi
    
    # 清理PID文件
    if [[ -f "parallel_pids.txt" ]]; then
        rm "parallel_pids.txt"
        echo "  删除: parallel_pids.txt"
    fi
    
    log_success "清理完成"
}

# 显示使用说明
show_usage_info() {
    echo ""
    echo "==============================================="
    echo "并行处理已启动!"
    echo "==============================================="
    echo ""
    echo "监控进度:"
    if [[ "$USE_TMUX" == "true" ]]; then
        echo "  tmux attach -t $SESSION_NAME          # 连接到会话"
        echo "  tmux list-windows -t $SESSION_NAME    # 查看所有窗口"
        echo "  tmux select-window -t $SESSION_NAME:chunk_1  # 切换到特定窗口"
    else
        echo "  tail -f chunk_*.log                   # 查看所有日志"
        echo "  ps aux | grep discriminator_system    # 查看运行进程"
    fi
    echo ""
    echo "完成后合并结果:"
    echo "  python merge_results.py --output_pattern './discriminator/output_chunk_*'"
    echo ""
    echo "或者运行清理脚本:"
    echo "  $0 --merge-only  # (如果添加此选项)"
    echo ""
}

# 主函数
main() {
    echo "==============================================="
    echo "并行 Discrimination 处理脚本"
    echo "==============================================="
    echo "配置文件: $CONFIG_FILE"
    echo "Episodes文件: $CONTROVERSIAL_EPISODES_FILE"
    echo "分割块数: $NUM_CHUNKS"
    echo "使用 tmux: $USE_TMUX"
    echo "==============================================="
    
    # 检查依赖
    check_dependencies
    
    # 分割episodes文件
    split_episodes
    
    # 创建chunk配置文件
    create_chunk_configs
    
    # 启动并行处理
    if [[ "$USE_TMUX" == "true" ]]; then
        start_tmux_processing
    else
        start_background_processing
    fi
    
    # 显示使用说明
    if [[ "$DRY_RUN" != "true" ]]; then
        show_usage_info
    fi
    
    log_success "并行处理设置完成!"
}

# 运行主函数
main "$@"
