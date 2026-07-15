#!/bin/bash
# setup_env.sh — 在 RunPod GPU pod（推荐 H200 SXM 141GB）上搭建 LingBot-Video 推理环境
#
# 流程：
#   1. 系统工具 (git/tmux)
#   2. clone lingbot-video → ${WORK_DIR}/lingbot-video
#   3. python venv → ${WORK_DIR}/.venv
#   4. 安装 torch：优先上游 pin 的 cu130 nightly；驱动不支持或 nightly 已下架则回退最新稳定版
#   5. 其余依赖 + lingbot_video (editable) + HF CLI
#   6. 源码编译 FlashAttention-3（Hopper；DiT 的 batch_cfg/packed 注意力硬依赖，~20-60 分钟）
#   7. 验证（CUDA / GPU / torch._grouped_mm / FA3 / 关键 import）
#
# 磁盘布局：默认全放 /root/lingbot（本地 NVMe，快）。pod stop 后容器盘清空，重跑本
# 脚本即恢复（幂等，~5 分钟）。/workspace 网络卷 I/O 慢，只用来存实验产物（见 common.sh）。
#
# 用法:
#   bash setup_env.sh
#   WORK_DIR=/workspace/lingbot bash setup_env.sh   # 改放持久卷（全持久化但 I/O 慢）
#   FORCE_TORCH_FALLBACK=1 bash setup_env.sh        # 跳过 nightly 直接装稳定版 torch
#   SKIP_FA3=1 bash setup_env.sh                    # 跳过 FA3 编译（届时运行必须 BATCH_CFG=0）
#   FA3_MAX_JOBS=8 bash setup_env.sh                # FA3 编译并行度（默认 16，内存紧张时调小）

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# 上游 requirements.txt 的 pin（Robbyant/lingbot-video @ 2026-07）
TORCH_PIN="torch==2.12.0.dev20260220+cu130"
TORCHVISION_PIN="torchvision==0.26.0.dev20260220+cu130"
NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cu130"
# 统一用实验 fork（xiefan46/lingbot-video，公开仓库，pod 上 HTTPS 匿名可 clone）；
# 需要原版时: LINGBOT_GIT=https://github.com/Robbyant/lingbot-video.git bash setup_env.sh
LINGBOT_GIT="${LINGBOT_GIT:-https://github.com/xiefan46/lingbot-video.git}"

command -v nvidia-smi &>/dev/null || err "未检测到 nvidia-smi，请在 GPU pod 上运行"
python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null \
    || err "需要 python3 >= 3.10（当前: $(python3 -V 2>&1)）"

# ─── 1. 系统工具 ───
export DEBIAN_FRONTEND=noninteractive
NEED_PKGS=""
command -v git  &>/dev/null || NEED_PKGS="${NEED_PKGS} git"
command -v tmux &>/dev/null || NEED_PKGS="${NEED_PKGS} tmux"
if [ -n "$NEED_PKGS" ]; then
    log "安装系统工具:${NEED_PKGS}..."
    apt-get install -y ${NEED_PKGS} 2>/dev/null || { apt-get update && apt-get install -y ${NEED_PKGS}; }
fi

# ─── 2. clone 上游仓库 ───
mkdir -p "$WORK_DIR"
if [ -d "${LINGBOT_REPO}/.git" ]; then
    log "lingbot-video 已存在: $LINGBOT_REPO"
else
    log "clone lingbot-video → $LINGBOT_REPO"
    git clone "$LINGBOT_GIT" "$LINGBOT_REPO"
fi

# ─── 3. venv ───
if [ -f "${VENV_DIR}/bin/activate" ]; then
    log "venv 已存在: $VENV_DIR"
else
    log "创建 venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
"$PIP" install -qU pip

# ─── 4. torch ───
driver_cuda=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo "0")
driver_cuda_major=${driver_cuda%%.*}
log "驱动支持的 CUDA: ${driver_cuda} | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

install_torch_fallback() {
    warn "回退安装最新稳定版 torch/torchvision"
    warn "（上游只在 pin 的 nightly 上测过；MoE 默认后端依赖 torch._grouped_mm，装完看验证输出确认）"
    "$PIP" install -U torch torchvision
}

if [ "${FORCE_TORCH_FALLBACK:-}" = "1" ]; then
    install_torch_fallback
elif "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "torch 已安装: $("$PY" -c 'import torch; print(torch.__version__)')，跳过（强制重装: rm -rf $VENV_DIR 后重跑）"
elif [ "$driver_cuda_major" -ge 13 ]; then
    log "安装上游 pin: ${TORCH_PIN}（cu130 nightly，约 3GB）..."
    "$PIP" install --extra-index-url "$NIGHTLY_INDEX" "$TORCH_PIN" "$TORCHVISION_PIN" || install_torch_fallback
else
    warn "驱动 CUDA ${driver_cuda} < 13，无法用 cu130 nightly"
    install_torch_fallback
fi

# ─── 5. 其余依赖（滤掉 requirements.txt 里的 torch pin，避免覆盖上面装好的版本） ───
log "安装其余依赖..."
grep -vE '^(--extra-index-url|torch==|torchvision==)' "${LINGBOT_REPO}/requirements.txt" > /tmp/lingbot_reqs.txt
"$PIP" install -q -r /tmp/lingbot_reqs.txt
"$PIP" install -q -e "$LINGBOT_REPO"
"$PIP" install -qU "huggingface_hub[cli]" hf_transfer

# ─── 6. FlashAttention-3（Hopper 专用；DiT 在 batch_cfg/packed 注意力路径硬依赖
#        flash_attn_interface，但上游 requirements 没带它，只能源码编译） ───
if [ "${SKIP_FA3:-}" = "1" ]; then
    warn "SKIP_FA3=1: 跳过 FA3 编译。运行脚本时必须 BATCH_CFG=0（B=1 走 SDPA，不需要 FA3）"
elif "$PY" -c "import flash_attn_interface" 2>/dev/null; then
    log "flash_attn_interface (FA3) 已安装，跳过编译"
else
    command -v nvcc &>/dev/null || err "缺少 nvcc（FA3 必须源码编译，需要 devel 镜像）。换带 CUDA toolkit 的模板，或 SKIP_FA3=1 重跑并在运行时用 BATCH_CFG=0"
    nvcc_ver=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    torch_cuda=$("$PY" -c "import torch; print(torch.version.cuda or '?')")
    [ "${nvcc_ver%%.*}" = "${torch_cuda%%.*}" ] || warn "nvcc(${nvcc_ver}) 与 torch CUDA(${torch_cuda}) 大版本不一致，编译可能失败——失败就换 CUDA 版本匹配的镜像，或 FORCE_TORCH_FALLBACK=1 重装 torch 后再试"
    log "源码编译 FlashAttention-3 (hopper/)，nvcc ${nvcc_ver}，MAX_JOBS=${FA3_MAX_JOBS:-16}，约 20-60 分钟..."
    "$PIP" install -q ninja packaging
    FA_DIR="${WORK_DIR}/flash-attention"
    [ -d "${FA_DIR}/.git" ] || git clone https://github.com/Dao-AILab/flash-attention.git "$FA_DIR"
    SECONDS=0
    (cd "${FA_DIR}/hopper" && MAX_JOBS="${FA3_MAX_JOBS:-16}" "$PY" setup.py install)
    "$PY" -c "import flash_attn_interface" 2>/dev/null || err "FA3 编译安装失败（看上方编译输出）。临时绕过: SKIP_FA3=1 重跑本脚本，运行时 BATCH_CFG=0"
    log "FA3 编译完成 (${SECONDS}s)"
fi

# ─── 7. 验证 ───
log "验证环境..."
"$PY" - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
assert torch.cuda.is_available(), "CUDA not available!"
print(f"CUDA runtime: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"显存: {total:.0f} GiB")
if hasattr(torch, "_grouped_mm"):
    print("torch._grouped_mm: OK（MoE 默认 grouped_mm 后端可用）")
else:
    print("WARN: torch._grouped_mm 缺失 — 跑 MoE 需换 torch 版本，或安装 requirements-sglang.txt 后用 LINGBOT_MOE_EXPERT_BACKEND=sglang_triton")
try:
    import flash_attn_interface  # noqa: F401
    print("flash_attn_interface (FA3): OK（batch_cfg/packed 注意力可用）")
except Exception:
    print("WARN: FA3 缺失 — 默认 batch_cfg 路径会报 flash_attn_varlen_func required，运行时加 BATCH_CFG=0")
import diffusers, transformers
print(f"diffusers: {diffusers.__version__}, transformers: {transformers.__version__}")
import lingbot_video
print("lingbot_video: OK")
print("\n=== All checks passed! ===")
EOF

echo -e "\n${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}${GREEN}  环境就绪${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "${YELLOW}下一步: bash download_models.sh dense   # 先 1.3B 冒烟${RESET}"
echo -e "${YELLOW}       bash download_models.sh moe     # 30B, ~121GB${RESET}\n"
