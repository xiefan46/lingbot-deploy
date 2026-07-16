#!/bin/bash
# run_dense_t2v.sh — Dense 1.3B 单卡 T2V 冒烟测试（验证环境全链路，峰值 ~15GiB）
#
# 用法:
#   bash run_dense_t2v.sh                                # 480p/40步, 自带 example prompt
#   HEIGHT=192 WIDTH=320 STEPS=20 bash run_dense_t2v.sh  # 更快的小烟测
#   PROMPT_JSON=/path/prompt.json bash run_dense_t2v.sh
#
# 结束时打印总耗时与显存峰值（gpu_mem.csv / run.log 留档在输出目录）

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
activate_venv

MODEL_DIR="${MODEL_DIR:-$DENSE_MODEL_DIR}"
[ -f "${MODEL_DIR}/model_index.json" ] || err "模型不存在: $MODEL_DIR（先 bash download_models.sh dense）"

export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"
# 环境里没装 flash-attn 3（上游默认 FA3 是 Hopper 专属且需单独编译），text encoder 用 sdpa
export LINGBOT_QWEN_ATTN_IMPLEMENTATION="${LINGBOT_QWEN_ATTN_IMPLEMENTATION:-sdpa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROMPT_JSON="${PROMPT_JSON:-${LINGBOT_REPO}/assets/cases/t2v/example_1/prompt.json}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
STEPS="${STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"
SHIFT="${SHIFT:-3}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"
BATCH_CFG="${BATCH_CFG:-1}"

OUT_DIR="${OUT_DIR:-${OUTPUTS_DIR}/dense_t2v_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
[ "$BATCH_CFG" = "1" ] && EXTRA_ARGS+=(--batch_cfg)
[ -n "${NUM_FRAMES:-}" ] && EXTRA_ARGS+=(--num_frames "$NUM_FRAMES")
# 注意: 自带 example prompt 含 duration 字段，runner 会用它覆盖 --num_frames——
# 想真正改帧数用 DURATION（秒），如 DURATION=2.5 → 61 帧
[ -n "${DURATION:-}" ] && EXTRA_ARGS+=(--duration "$DURATION")

log "模型: $MODEL_DIR"
log "输出: $OUT_DIR | prompt: $PROMPT_JSON"
log "参数: ${HEIGHT}x${WIDTH} steps=${STEPS} batch_cfg=${BATCH_CFG}"

start_gpu_monitor "$OUT_DIR"
SECONDS=0
(cd "$LINGBOT_REPO" && "$PY" scripts/inference.py \
    --backend diffusers \
    --model_dir "$MODEL_DIR" \
    --mode t2v \
    --prompt_json "$PROMPT_JSON" \
    --output "$OUT_DIR/t2v.mp4" \
    --height "$HEIGHT" --width "$WIDTH" \
    --steps "$STEPS" --guidance_scale "$GUIDANCE_SCALE" --shift "$SHIFT" \
    --seed "$SEED" --fps "$FPS" \
    --transformer_dtype bf16 --text_encoder_dtype bf16 --vae_dtype fp32 \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}) 2>&1 | tee "$OUT_DIR/run.log"
elapsed=$SECONDS
stop_gpu_monitor

log "总耗时: ${elapsed}s"
report_gpu_peak
log "产物: ${BOLD}${OUT_DIR}/t2v.mp4${RESET}"
