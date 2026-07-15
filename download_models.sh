#!/bin/bash
# download_models.sh — 下载 LingBot-Video 权重（HF 公开仓库，无需登录/token）
#
# 用法:
#   bash download_models.sh dense   # 1.3B 冒烟测试用, ~12GB
#   bash download_models.sh moe     # 30B-A3B, ~121GB（开 pod 时容器盘要给够，建议 ≥250GB）
#   bash download_models.sh all
#
# 权重落盘: ${MODELS_DIR}（默认 /root/lingbot/models，本地 NVMe；pod stop 后清空，重跑即重下）

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

TARGET="${1:-dense}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$MODELS_DIR"
[ -x "$HF_BIN" ] || err "hf CLI 不存在（$HF_BIN），先运行 bash setup_env.sh"

free_gb=$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
log "磁盘可用: ${free_gb}GB @ $MODELS_DIR"

download_one() {
    local repo="$1" dst="$2" need_gb="$3"
    if [ -f "${dst}/model_index.json" ]; then
        log "已存在，跳过: $dst（重新下载: 删除该目录后重跑）"
        return
    fi
    [ "$free_gb" -ge "$need_gb" ] || warn "磁盘可用 ${free_gb}GB < 建议 ${need_gb}GB，可能中途写满"
    log "下载 ${repo} → ${dst}"
    SECONDS=0
    "$HF_BIN" download "$repo" --local-dir "$dst"
    log "完成 (${SECONDS}s): $(du -sh "$dst" | cut -f1)"
}

case "$TARGET" in
    dense)
        download_one robbyant/lingbot-video-dense-1.3b "$DENSE_MODEL_DIR" 15
        ;;
    moe)
        download_one robbyant/lingbot-video-moe-30b-a3b "$MOE_MODEL_DIR" 130
        ;;
    all)
        download_one robbyant/lingbot-video-dense-1.3b "$DENSE_MODEL_DIR" 15
        download_one robbyant/lingbot-video-moe-30b-a3b "$MOE_MODEL_DIR" 145
        ;;
    *)
        err "用法: bash download_models.sh [dense|moe|all]"
        ;;
esac

echo -e "\n${BOLD}${GREEN}  权重就绪${RESET}"
echo -e "${YELLOW}冒烟: bash run_dense_t2v.sh${RESET}"
echo -e "${YELLOW}MoE:  bash run_moe_t2v.sh${RESET}\n"
