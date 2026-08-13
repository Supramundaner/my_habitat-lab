#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlfm}"
GROUNDING_DINO_PORT="${GROUNDING_DINO_PORT:-12181}"
SAM_PORT="${SAM_PORT:-12184}"
YOLOV7_PORT="${YOLOV7_PORT:-12185}"

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command '$1' was not found in PATH"
}

require_absolute_file() {
    local variable_name="$1"
    local value="${!variable_name:-}"
    [[ -n "${value}" ]] || fail "${variable_name} is required and must point to an absolute file path"
    [[ "${value}" = /* ]] || fail "${variable_name} must be an absolute path (got: ${value})"
    [[ -f "${value}" ]] || fail "${variable_name} does not point to a file: ${value}"
}

require_absolute_directory() {
    local variable_name="$1"
    local value="${!variable_name:-}"
    [[ -n "${value}" ]] || fail "${variable_name} is required and must point to an absolute directory path"
    [[ "${value}" = /* ]] || fail "${variable_name} must be an absolute path (got: ${value})"
    [[ -d "${value}" ]] || fail "${variable_name} does not point to a directory: ${value}"
}

validate_port() {
    local name="$1"
    local value="${!name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be an integer (got: ${value})"
    (( value >= 1 && value <= 65535 )) || fail "${name} must be between 1 and 65535"
}

require_command conda
require_command tmux

require_absolute_file GROUNDING_DINO_CONFIG
require_absolute_file GROUNDING_DINO_WEIGHTS
require_absolute_directory YOLOV7_ROOT
require_absolute_file YOLOV7_WEIGHTS
require_absolute_file MOBILE_SAM_CHECKPOINT

validate_port GROUNDING_DINO_PORT
validate_port SAM_PORT
validate_port YOLOV7_PORT

for module_file in grounding_dino.py sam.py yolov7.py; do
    [[ -f "${SCRIPT_DIR}/vlm/${module_file}" ]] || fail "missing VLM module: ${SCRIPT_DIR}/vlm/${module_file}"
done

echo "Validating Python environment '${CONDA_ENV_NAME}'..."
conda run -n "${CONDA_ENV_NAME}" python -c \
    'import cv2, flask, numpy, requests, torch, torchvision' \
    || fail "base VLM Python dependencies are missing from conda environment '${CONDA_ENV_NAME}'"
conda run -n "${CONDA_ENV_NAME}" python -c \
    'import groundingdino, mobile_sam' \
    || fail "groundingdino or mobile_sam is missing from conda environment '${CONDA_ENV_NAME}'"
PYTHONPATH="${YOLOV7_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    conda run -n "${CONDA_ENV_NAME}" python -c \
    'from models.experimental import attempt_load; from utils.datasets import letterbox' \
    || fail "YOLOv7 could not be imported from YOLOV7_ROOT=${YOLOV7_ROOT}"

session_name="local_vlm_servers_${RANDOM}"
session_created=false
cleanup_failed_launch() {
    if [[ "${session_created}" == true ]]; then
        tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_failed_launch ERR

printf -v gdino_command \
    'exec conda run --no-capture-output -n %q python -m reason_navi.vlm.grounding_dino --port %q --config %q --weights %q' \
    "${CONDA_ENV_NAME}" "${GROUNDING_DINO_PORT}" "${GROUNDING_DINO_CONFIG}" "${GROUNDING_DINO_WEIGHTS}"
printf -v sam_command \
    'exec conda run --no-capture-output -n %q python -m reason_navi.vlm.sam --port %q --checkpoint %q' \
    "${CONDA_ENV_NAME}" "${SAM_PORT}" "${MOBILE_SAM_CHECKPOINT}"
printf -v yolov7_command \
    'exec conda run --no-capture-output -n %q python -m reason_navi.vlm.yolov7 --port %q --root %q --weights %q' \
    "${CONDA_ENV_NAME}" "${YOLOV7_PORT}" "${YOLOV7_ROOT}" "${YOLOV7_WEIGHTS}"

tmux new-session -d -s "${session_name}" -n gdino -c "${REPO_ROOT}" "${gdino_command}"
session_created=true
tmux new-window -d -t "${session_name}" -n sam -c "${REPO_ROOT}" "${sam_command}"
tmux new-window -d -t "${session_name}" -n yolov7 -c "${REPO_ROOT}" "${yolov7_command}"
trap - ERR

echo "Started loopback-only VLM services in tmux session '${session_name}':"
echo "  Grounding DINO: 127.0.0.1:${GROUNDING_DINO_PORT}"
echo "  Mobile SAM:      127.0.0.1:${SAM_PORT}"
echo "  YOLOv7:          127.0.0.1:${YOLOV7_PORT}"
echo "Inspect logs with: tmux attach-session -t ${session_name}"
echo "Stop services with: tmux kill-session -t ${session_name}"
