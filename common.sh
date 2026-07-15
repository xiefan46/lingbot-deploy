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

# ─── 路径约定（同 verl-deploy：重 I/O 放本地盘，不放 /workspace 网络卷） ───
# RunPod 的 /workspace 是网络卷：venv 的海量小文件 import 慢、60GB 权重加载慢数倍。
# 因此 venv/上游仓库/权重默认放 /root（本地 NVMe 容器盘）。代价：pod stop 会清空
# 容器盘，重跑 setup_env.sh + download_models.sh 即恢复（hf_transfer 重下权重约
# 5–15 分钟）。想全持久化（慢）：WORK_DIR=/workspace/lingbot bash setup_env.sh
WORK_DIR="${WORK_DIR:-/root/lingbot}"
LINGBOT_REPO="${LINGBOT_REPO:-${WORK_DIR}/lingbot-video}"
VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
MODELS_DIR="${MODELS_DIR:-${WORK_DIR}/models}"
# 实验产物（mp4/log/显存采样，都很小）优先放 /workspace 网络卷，pod stop 后不丢
if [ -z "${OUTPUTS_DIR:-}" ]; then
    if [ -d /workspace ]; then
        OUTPUTS_DIR="/workspace/lingbot-outputs"
    else
        OUTPUTS_DIR="${WORK_DIR}/outputs"
    fi
fi

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
