#!/bin/bash
# common.sh — lingbot-deploy 共享配置与工具函数（被其他脚本 source，不直接执行）

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${RESET} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${RESET} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${RESET} $*"; exit 1; }

# ─── 路径约定（RunPod: /workspace 是持久卷，pod 重启不丢，venv/权重都放这里） ───
WORK_DIR="${WORK_DIR:-/workspace/lingbot}"
LINGBOT_REPO="${LINGBOT_REPO:-${WORK_DIR}/lingbot-video}"
VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
MODELS_DIR="${MODELS_DIR:-${WORK_DIR}/models}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${WORK_DIR}/outputs}"

DENSE_MODEL_DIR="${DENSE_MODEL_DIR:-${MODELS_DIR}/lingbot-video-dense-1.3b}"
MOE_MODEL_DIR="${MOE_MODEL_DIR:-${MODELS_DIR}/lingbot-video-moe-30b-a3b}"

PY="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
HF_BIN="${VENV_DIR}/bin/hf"

activate_venv() {
    [ -f "${VENV_DIR}/bin/activate" ] || err "venv 不存在: ${VENV_DIR}，先运行 bash setup_env.sh"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
}

# ─── GPU 显存监控（后台 nvidia-smi 采样；峰值用于校准显存预算表） ───
GPU_MON_PID=""
GPU_MON_LOG=""

start_gpu_monitor() {
    GPU_MON_LOG="$1/gpu_mem.csv"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -lms 500 -i "${CUDA_MON_DEVICE:-0}" > "$GPU_MON_LOG" &
    GPU_MON_PID=$!
    trap stop_gpu_monitor EXIT
}

stop_gpu_monitor() {
    if [ -n "$GPU_MON_PID" ]; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        GPU_MON_PID=""
    fi
}

report_gpu_peak() {
    [ -f "$GPU_MON_LOG" ] || return 0
    local peak
    peak=$(awk '{if ($1+0 > m) m=$1+0} END {print m+0}' "$GPU_MON_LOG")
    log "GPU 显存峰值: ${BOLD}${peak} MiB ($(awk "BEGIN{printf \"%.1f\", ${peak}/1024}") GiB)${RESET}  (采样明细: $GPU_MON_LOG)"
}
