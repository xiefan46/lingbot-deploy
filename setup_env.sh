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
#   HF_CACHE_REPO=user/repo bash setup_env.sh       # FA3 wheel 缓存仓（默认 xiefan46/lingbot-env-cache）
#   FA3_REBUILD=1 bash setup_env.sh                 # 忽略 HF 缓存，强制重新编译
#
# FA3 缓存（仿 verl-deploy）：首次源码编译后 wheel 自动上传 HF 缓存仓（上传需要 write token，
# hf auth login 一次即可；仓库默认公开，下载免 token），之后的新 pod 直接下载 wheel 秒装。

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

# 配置 NVIDIA cuda apt 源（幂等；处理镜像自带源与 cuda-keyring 的 Signed-By 冲突）
ensure_nvidia_apt_repo() {
    . /etc/os-release
    local distro="ubuntu${VERSION_ID//./}"
    if ! grep -rqs "developer.download.nvidia.com/compute/cuda/repos" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
            && dpkg -i /tmp/cuda-keyring.deb >/dev/null || return 1
    fi
    if ! apt-get update -qq 2>/tmp/apt_update.err; then
        if grep -q "Signed-By" /tmp/apt_update.err; then
            warn "apt cuda 源 Signed-By 冲突，移除 cuda-keyring 的重复条目后重试"
            rm -f "/etc/apt/sources.list.d/cuda-${distro}-x86_64.list"
            apt-get update -qq || return 1
        else
            cat /tmp/apt_update.err >&2
            return 1
        fi
    fi
}

# 驱动 <13 时尝试 CUDA forward-compat：数据中心卡（H200/H100/A100）可在老驱动上
# 用 cuda-compat-13-0 提供的用户态 libcuda 跑 CUDA 13 应用，无需动宿主机驱动。
CUDA_COMPAT_ACTIVE=0
try_cuda_forward_compat() {
    local d
    for d in /usr/local/cuda-13.0/compat /usr/local/cuda/compat; do
        if [ -e "${d}/libcuda.so.1" ]; then
            export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
            CUDA_COMPAT_ACTIVE=1
            log "forward-compat 已就绪（${d}），老驱动可跑 cu13"
            return 0
        fi
    done
    log "驱动 CUDA ${driver_cuda} < 13，尝试安装 forward-compat 包 cuda-compat-13-0..."
    ensure_nvidia_apt_repo || { warn "NVIDIA apt 源不可用，装不了 compat"; return 1; }
    apt-get install -y -qq cuda-compat-13-0 || { warn "cuda-compat-13-0 安装失败"; return 1; }
    for d in /usr/local/cuda-13.0/compat /usr/local/cuda/compat; do
        if [ -e "${d}/libcuda.so.1" ]; then
            export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
            CUDA_COMPAT_ACTIVE=1
            log "forward-compat 就绪: ${d}"
            return 0
        fi
    done
    return 1
}
if [ "$driver_cuda_major" -lt 13 ]; then
    try_cuda_forward_compat || warn "forward-compat 不可用——将走 cu12x torch 路线（FA3 缓存不可用，运行需 BATCH_CFG=0）"
fi

install_torch_fallback() {
    warn "回退安装稳定版 torch/torchvision（上游只在 pin 的 nightly 上测过；装完看验证输出的 torch._grouped_mm 确认）"
    if [ -n "${TORCH_CUDA_VARIANT:-}" ]; then
        "$PIP" install -U torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VARIANT}"
        return
    fi
    if [ "$driver_cuda_major" -ge 13 ] || [ "$CUDA_COMPAT_ACTIVE" = "1" ]; then
        # PyPI 默认 wheel 目前就是 cu13 构建
        "$PIP" install -U torch torchvision
    else
        # 驱动只到 CUDA 12.x：PyPI 默认的 cu13 wheel 会 CUDA not available，必须装 cu12x 构建。
        # 注意：FA3 缓存 wheel（fa3/torch-cu13）在此环境不可用——要么 SKIP_FA3=1 + 运行时
        # BATCH_CFG=0，要么换驱动 CUDA>=13 的宿主机（RunPod 创建页可按 CUDA Version 过滤）。
        warn "宿主机驱动仅支持 CUDA ${driver_cuda} —— 安装 cu12x 构建的 torch"
        warn "此环境下 FA3 的 cu13 缓存 wheel 不可用：请 SKIP_FA3=1 重跑本脚本，并全程 BATCH_CFG=0 运行；官方 batch_cfg 路径请换 CUDA>=13 驱动的 pod"
        "$PIP" install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
            || "$PIP" install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
    fi
}

if [ "${FORCE_TORCH_FALLBACK:-}" = "1" ]; then
    install_torch_fallback
elif "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "torch 已安装: $("$PY" -c 'import torch; print(torch.__version__)')，跳过（强制重装: rm -rf $VENV_DIR 后重跑）"
elif [ "$driver_cuda_major" -ge 13 ] || [ "$CUDA_COMPAT_ACTIVE" = "1" ]; then
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
"$PIP" install -qU huggingface_hub hf_transfer

# ─── 6. FlashAttention-3（Hopper 专用；DiT 在 batch_cfg/packed 注意力路径硬依赖
#        flash_attn_interface，但上游 requirements 没带它。仿 verl-deploy 的缓存策略：
#        源码编译一次 → wheel 上传 HF 缓存仓（stable ABI + abi3，可跨 torch/python 小版本
#        复用）→ 之后的 pod 直接下载 wheel 秒装，不再重编。 ───
HF_CACHE_REPO="${HF_CACHE_REPO:-xiefan46/lingbot-env-cache}"
FA_DIR="${WORK_DIR}/flash-attention"

_fa3_cache_path() {
    local cu
    cu=$("$PY" -c "import torch; print((torch.version.cuda or '0').split('.')[0])")
    echo "fa3/torch-cu${cu}"
}

# torch 的 cpp_extension 要求 nvcc 与 torch 的 CUDA 大版本一致（如 torch cu13.0 配 nvcc 13.x），
# 否则直接 RuntimeError。先在本机找匹配的 nvcc，找不到就从 NVIDIA apt 源装一个（~4GB，几分钟）。
_fa3_pick_cuda_home() {
    local torch_cuda torch_major d v
    torch_cuda=$("$PY" -c "import torch; print(torch.version.cuda or '')")
    torch_major=${torch_cuda%%.*}
    for d in "/usr/local/cuda-${torch_cuda}" /usr/local/cuda-* /usr/local/cuda; do
        [ -x "${d}/bin/nvcc" ] || continue
        v=$("${d}/bin/nvcc" --version | grep -oE 'release [0-9]+' | grep -oE '[0-9]+' | head -1)
        [ "$v" = "$torch_major" ] && { echo "$d"; return 0; }
    done
    return 1
}

_fa3_ensure_toolkit() {
    CUDA_HOME_MATCHED=$(_fa3_pick_cuda_home || true)
    if [ -n "$CUDA_HOME_MATCHED" ]; then
        log "使用 nvcc: ${CUDA_HOME_MATCHED}/bin/nvcc"
        return 0
    fi
    local torch_cuda distro toolkit_pkg
    torch_cuda=$("$PY" -c "import torch; print(torch.version.cuda or '')")
    . /etc/os-release
    distro="ubuntu${VERSION_ID//./}"
    toolkit_pkg="cuda-toolkit-${torch_cuda%%.*}-${torch_cuda#*.}"
    log "镜像没有与 torch CUDA ${torch_cuda} 匹配的 nvcc，从 NVIDIA apt 源安装 ${toolkit_pkg}（~4GB）..."
    ensure_nvidia_apt_repo \
        || err "NVIDIA apt 源配置失败。备选: 换 CUDA ${torch_cuda} 的 devel 镜像；或 SKIP_FA3=1 重跑并在运行时用 BATCH_CFG=0"
    apt-get install -y -qq "$toolkit_pkg" \
        || err "${toolkit_pkg} 安装失败。备选: 换 CUDA ${torch_cuda} 的 devel 镜像；或 SKIP_FA3=1 重跑并在运行时用 BATCH_CFG=0"
    CUDA_HOME_MATCHED=$(_fa3_pick_cuda_home) || err "toolkit 装完仍找不到匹配的 nvcc"
    log "使用 nvcc: ${CUDA_HOME_MATCHED}/bin/nvcc"
}

# 编译/打包 wheel，路径写入 FA3_WHL。ninja 有编译缓存：全新 20-40 分钟，增量打包 1-2 分钟。
_fa3_build_wheel() {
    _fa3_ensure_toolkit
    "$PIP" install -q ninja packaging
    [ -d "${FA_DIR}/.git" ] || git clone https://github.com/Dao-AILab/flash-attention.git "$FA_DIR"
    log "编译/打包 FlashAttention-3 wheel（MAX_JOBS=${FA3_MAX_JOBS:-16}）..."
    SECONDS=0
    # PATH 里带上 venv/bin：让 torch 找到 pip 装的 ninja（否则退化成单线程 distutils，慢一个数量级）
    (cd "${FA_DIR}/hopper" \
        && rm -rf dist \
        && CUDA_HOME="$CUDA_HOME_MATCHED" \
           PATH="${CUDA_HOME_MATCHED}/bin:${VENV_DIR}/bin:${PATH}" \
           MAX_JOBS="${FA3_MAX_JOBS:-16}" \
           "$PY" setup.py bdist_wheel)
    FA3_WHL=$(ls "${FA_DIR}/hopper/dist/"*.whl 2>/dev/null | head -1)
    [ -n "$FA3_WHL" ] || err "FA3 wheel 打包失败（看上方编译输出）。临时绕过: SKIP_FA3=1 重跑本脚本，运行时 BATCH_CFG=0"
    log "wheel 就绪 (${SECONDS}s): ${FA3_WHL}"
}

_fa3_upload_wheel() {
    local whl="$1" sub
    sub=$(_fa3_cache_path)
    if "$PY" - "$whl" "$HF_CACHE_REPO" "$sub" <<'UPLOADEOF'
import os, sys
from huggingface_hub import HfApi
whl, repo, sub = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
private = os.environ.get("HF_CACHE_PRIVATE", "0") == "1"
api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
api.upload_file(path_or_fileobj=whl, path_in_repo=f"{sub}/{os.path.basename(whl)}",
                repo_id=repo, repo_type="dataset")
print(f"uploaded: {repo}/{sub}/{os.path.basename(whl)}")
UPLOADEOF
    then
        log "FA3 wheel 已上传缓存仓 ${HF_CACHE_REPO}，之后的 pod 会直接秒装"
    else
        warn "上传 HF 缓存失败（需要 write 权限 token：先 hf auth login 或 export HF_TOKEN=hf_xxx，再重跑本脚本补传）"
        warn "手动补传: hf upload --repo-type dataset ${HF_CACHE_REPO} ${whl} ${sub}/$(basename "$whl")"
    fi
}

if [ "${SKIP_FA3:-}" = "1" ]; then
    warn "SKIP_FA3=1: 跳过 FA3。运行脚本时必须 BATCH_CFG=0（B=1 走 SDPA，不需要 FA3）"
elif "$PY" -c "import flash_attn_interface" 2>/dev/null; then
    log "flash_attn_interface (FA3) 已安装"
    # 已装但缓存 wheel 还没打包/上传过（比如老版脚本 setup.py install 装的）→ 增量打包并上传
    if [ "${FA3_NO_CACHE:-}" != "1" ] && [ -d "${FA_DIR}/hopper/build" ] && ! ls "${FA_DIR}/hopper/dist/"*.whl >/dev/null 2>&1; then
        log "检测到本地编译产物但缓存 wheel 缺失，增量打包上传（ninja 缓存，~1-2 分钟）..."
        _fa3_build_wheel
        _fa3_upload_wheel "$FA3_WHL"
    fi
else
    # 1) 先试 HF 缓存的 wheel
    if [ "${FA3_REBUILD:-}" != "1" ]; then
        FA3_SUB=$(_fa3_cache_path)
        FA3_CACHE_DIR="${WORK_DIR}/fa3-wheel-cache"
        if "$HF_BIN" download --repo-type dataset "$HF_CACHE_REPO" --include "${FA3_SUB}/*.whl" --local-dir "$FA3_CACHE_DIR" >/dev/null 2>&1 \
            && ls "${FA3_CACHE_DIR}/${FA3_SUB}/"*.whl >/dev/null 2>&1; then
            log "命中 HF 缓存（${HF_CACHE_REPO}/${FA3_SUB}），直接安装 wheel"
            "$PIP" install -q "${FA3_CACHE_DIR}/${FA3_SUB}/"*.whl
        else
            log "HF 缓存未命中，走源码编译（编完会自动上传缓存）"
        fi
    fi
    # 2) 缓存没救到 → 源码编译 wheel + 安装 + 上传
    if ! "$PY" -c "import flash_attn_interface" 2>/dev/null; then
        _fa3_build_wheel
        "$PIP" install -q "$FA3_WHL"
        "$PY" -c "import flash_attn_interface" 2>/dev/null \
            || err "FA3 安装后 import 仍失败（看上方输出）。临时绕过: SKIP_FA3=1 重跑本脚本，运行时 BATCH_CFG=0"
        [ "${FA3_NO_CACHE:-}" != "1" ] && _fa3_upload_wheel "$FA3_WHL"
    fi
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
