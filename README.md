# lingbot-deploy

在 **RunPod H200 单卡**上部署 [LingBot-Video](https://github.com/Robbyant/lingbot-video)（30B-A3B MoE 视频生成模型，arXiv 2607.07675）做推理与显存实测。

背景调研见 serving 仓库的 `research/2026-07-15-lingbot-video-serving-analysis.md`（架构/显存预算分析）。本仓库的目标：**验证 30B MoE bf16 单卡部署，并实测显存峰值/耗时来校准预算表**。

## RunPod 开机配置

| 项 | 建议 |
| --- | --- |
| GPU | H200 SXM（141GB）×1；H20 96GB 也可跑 base-only |
| Volume | **≥ 200GB**，挂载 `/workspace`（MoE 权重 121GB + dense 12GB + venv） |
| 模板 | 任意带 NVIDIA 驱动的模板即可（如官方 PyTorch 模板）；脚本自建 venv，不依赖镜像里的 torch |
| 注意 | 权重/venv 全落 `/workspace` 持久卷，pod 停开机后重跑 `setup_env.sh` 秒级恢复 |

## 快速开始

```bash
cd /workspace && git clone git@github.com:xiefan46/lingbot-deploy.git && cd lingbot-deploy

bash setup_env.sh              # 环境（clone 上游 + venv + torch + 依赖 + 验证）
bash download_models.sh dense  # 1.3B, ~12GB
bash run_dense_t2v.sh          # 冒烟：验证全链路，峰值 ~15GiB

bash download_models.sh moe    # 30B, ~121GB（下载较久，建议 tmux 里跑）
bash run_moe_t2v.sh            # 正菜：30B MoE 单卡 480p T2V
```

产物在 `/workspace/lingbot/outputs/<run>/t2v.mp4`，同目录的 `run.log` / `gpu_mem.csv` 是日志和显存采样（500ms 粒度），脚本结束时会打印**显存峰值和总耗时**。

所有脚本支持环境变量覆盖（`HEIGHT/WIDTH/STEPS/NUM_FRAMES/PROMPT_JSON/SEED/OUT_DIR/...`），路径约定见 `common.sh`。

## 预期数字（来自权重实测大小 + 推算，待本仓库实测校准）

| 项 | 数值 |
| --- | --- |
| MoE DiT 权重 (bf16) | 60.3 GB |
| 文本编码器 Qwen3-VL-4B (bf16) | 8.9 GB |
| VAE (fp32) | ~1 GB |
| 480p×121帧 激活/工作区（batch_cfg 开） | ~10–15 GB |
| **预期峰值 (H200)** | **~80–90 GiB / 141 GiB** |
| 耗时（5s@480p，40 步，H200，估） | ~2–4 min/条 + 首次加载数分钟 |

## 显存不够时的旋钮（80GB 卡参考）

按收益排序，可叠加：

```bash
BATCH_CFG=0 bash run_moe_t2v.sh                                   # CFG 改串行，工作区约减半
LINGBOT_MOE_RESTORE_BACKEND=chunked_scatter bash run_moe_t2v.sh   # 压 MoE 还原缓冲的 fp32 峰值
HEIGHT=192 WIDTH=320 bash run_moe_t2v.sh                          # 降分辨率桶
NUM_FRAMES=49 bash run_moe_t2v.sh                                 # 5s→2s（帧数须为 4n+1）
```

## 已知坑

1. **torch 版本**：上游 pin 了 `torch==2.12.0.dev20260220+cu130`（nightly）。`setup_env.sh` 会检测驱动 CUDA 版本，装不上时自动回退最新稳定版；回退后注意验证输出里的 `torch._grouped_mm` 检查（MoE 默认后端依赖它，缺失时装 `requirements-sglang.txt` 后改用 `LINGBOT_MOE_EXPERT_BACKEND=sglang_triton`）。
2. **text encoder 注意力**：上游默认 `flash_attention_3`，但 FA3 需要单独编译且 requirements 里没有——运行脚本已默认改为 `sdpa`（文本编码只占总时长很小一部分，无感）。
3. **不要开 refiner**（`--run_refiner`）：refiner 是**同尺寸的另一个 30B DiT**，双模型 120GB 权重 + 1080p 工作区，单卡（含 H200）放不下。上游 refiner 方案是 8 卡 CP8+FSDP。
4. **不要用 `LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8` 来省显存**：开源实现是运行时量化+额外缓存，bf16 权重不释放，显存反增 ~29GB（它是多卡 FSDP 下的提速方案）。单卡省显存只能用默认 `grouped_mm`。
5. **rewriter 不用部署**：冒烟用上游自带的结构化 prompt（`assets/cases/`）。要用自己的 prompt 时才需要 Qwen3.6-27B rewriter（bf16 ~54GB，届时单独规划）。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `common.sh` | 路径约定 + log/显存监控函数（被其他脚本 source） |
| `setup_env.sh` | 环境搭建（幂等，pod 重启后重跑即恢复） |
| `download_models.sh` | 下载权重 `dense\|moe\|all` |
| `run_dense_t2v.sh` | 1.3B 冒烟测试 |
| `run_moe_t2v.sh` | 30B MoE 单卡 T2V + 显存/耗时实测 |
