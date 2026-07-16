#!/bin/bash
# run_ablations.sh — MoE 30B 显存归因消融实验（验证 101.2 GiB 峰值的拆解是否正确）
#
# 背景：H200 实测 MoE 峰值 101.2 GiB = 权重 ~70GB + 非权重 ~31GB。
# 拆解归因（batched-CFG 序列长度、MoE dispatch/restore 缓冲、fp32 还原副本）来自
# 代码阅读 + 手算，本脚本用对照实验逐项证伪/证实——每组都有事前预测值。
#
# 用法:
#   bash run_ablations.sh              # 6 组基础实验，共 ~25 分钟
#   RUN_FP8=1 bash run_ablations.sh    # 追加第 7 组：runtime FP8 显存不省反增的验证
#   ABL_STEPS=40 bash run_ablations.sh # 每组跑满 40 步（默认 12 步——峰值在前几步就定型）
#
# 结果: 逐组打印 + 汇总表存 ${OUTPUTS_DIR}/ablation_summary_<ts>.md

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
activate_venv

ABL_STEPS="${ABL_STEPS:-12}"
[ -f "${MOE_MODEL_DIR}/model_index.json" ] || err "MoE 模型不存在（先 bash download_models.sh moe）"

TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${OUTPUTS_DIR}/ablation_summary_${TS}.md"
mkdir -p "$OUTPUTS_DIR"
{
    echo "# MoE 30B-A3B 显存消融实验（H200，480p，steps=${ABL_STEPS}，${TS}）"
    echo
    echo "| 实验 | 变量 | 预测峰值 | 实测峰值 | 每步耗时 |"
    echo "| --- | --- | --- | --- | --- |"
} > "$SUMMARY_FILE"

run_one() {
    local name="$1" pred="$2"; shift 2
    local out="${OUTPUTS_DIR}/abl_${TS}_${name}"
    log "════ 实验 [${name}]  变量: ${*:-baseline}  预测: ${pred} ════"
    if env "$@" STEPS="$ABL_STEPS" OUT_DIR="$out" bash "${SCRIPT_DIR}/run_moe_t2v.sh"; then
        local peak_mib peak_gib step
        peak_mib=$(awk '{if ($1+0 > m) m=$1+0} END{print m+0}' "$out/gpu_mem.csv")
        peak_gib=$("$PY" -c "print(f'{${peak_mib}/1024:.1f}')")
        step=$(grep -oE '[0-9.]+s/it' "$out/run.log" | tail -1 || true)
        echo "| ${name} | ${*:-baseline} | ${pred} | **${peak_gib} GiB** (${peak_mib} MiB) | ${step:-—} |" >> "$SUMMARY_FILE"
        log "[${name}] 实测峰值 ${peak_gib} GiB（预测 ${pred}）"
    else
        echo "| ${name} | ${*:-baseline} | ${pred} | ❌ 失败（${out}/run.log） | — |" >> "$SUMMARY_FILE"
        warn "[${name}] 运行失败，继续下一组（日志: ${out}/run.log）"
    fi
}

# 1) 基线复测（同 pod 同口径锚点；此前全量 40 步实测 101.2 GiB）
run_one baseline "~101 GiB"

# 2) 关 batched CFG：packed 序列 97k→48.4k。若"工作区随序列长度近似线性"成立 → 明显下降
run_one no_batch_cfg "87–90 GiB" BATCH_CFG=0

# 3) chunked 还原：若"默认 scatter 路径的 fp32 还原副本 ≈6GB"成立 → 恰好降这么多
run_one chunked_scatter "95–96 GiB" LINGBOT_MOE_RESTORE_BACKEND=chunked_scatter

# 4) 组合降压：H20 96GB 可行性的直接答案
run_one combo "84–86 GiB" BATCH_CFG=0 LINGBOT_MOE_RESTORE_BACKEND=chunked_scatter

# 5) 一致性检验：时长减半（DURATION=2.5 → 61 帧）+batch_cfg 开 → packed 序列 ≈ 实验2。
#    注意必须用 DURATION：--num_frames 会被 prompt.json 的 duration 覆盖（首轮实验实测踩坑，
#    峰值与 baseline 逐 MiB 相同）。
run_one half_duration "≈实验2（~84 GiB）" DURATION=2.5

# 6) 192p：token 降到基线的 ~15% → 逼近"权重+固定开销"下限；也回答 80GB 卡 192p 可行性
run_one res_192p "75–77 GiB" HEIGHT=192 WIDTH=320

# 7)（可选）runtime FP8：验证"开源 FP8 路径缓存副本、显存不省反增 ~29GB"的代码分析
if [ "${RUN_FP8:-0}" = "1" ]; then
    log "安装 SGLang 可选依赖（--no-deps，不动 torch 栈）..."
    "$PIP" install -q --no-deps -r "${LINGBOT_REPO}/requirements-sglang.txt" \
        || warn "SGLang 依赖安装失败，跳过 FP8 实验"
    # sglang 的 python 侧还要 orjson；pin 的 sglang-kernel 0.4.4 预编译 so 与新 torch ABI
    # 可能不匹配（undefined symbol），升级到最新（--no-deps 保护 torch 栈）
    "$PIP" install -q orjson
    "$PIP" install -qU --no-deps sglang-kernel || warn "sglang-kernel 升级失败，FP8 实验可能因 ABI 不匹配失败"
    run_one fp8_runtime "128–132 GiB（不省反增）" LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8
fi

echo
log "全部完成，汇总表: ${SUMMARY_FILE}"
echo "──────────────────────────────────────────"
cat "$SUMMARY_FILE"
echo "──────────────────────────────────────────"
log "别忘了归档: hf upload --repo-type dataset xiefan46/lingbot-env-cache ${OUTPUTS_DIR} outputs"
