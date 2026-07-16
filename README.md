# lingbot-deploy

在 **RunPod H200 单卡**上部署 [LingBot-Video](https://github.com/Robbyant/lingbot-video)（30B-A3B MoE 视频生成模型，arXiv 2607.07675）做推理与显存实测。

代码统一用我们的实验 fork **[xiefan46/lingbot-video](https://github.com/xiefan46/lingbot-video)**（`setup_env.sh` 默认从 fork clone，实验改动都提交到 fork；上游是 Robbyant/lingbot-video）。

背景调研见 serving 仓库的 `research/2026-07-15-lingbot-video-serving-analysis.md`（架构/显存预算分析）。本仓库的目标：**验证 30B MoE bf16 单卡部署，并实测显存峰值/耗时来校准预算表**。

## RunPod 开机配置

| 项 | 建议 |
| --- | --- |
| GPU | H200 SXM（141GB）×1；H20 96GB 也可跑 base-only |
| **Container Disk** | **≥ 250GB**（本地 NVMe：venv + 两个模型 ~140GB + 余量；开 pod 时就要给够，后改麻烦） |
| Volume | 可选，~20GB 挂 `/workspace`（只存实验产物 mp4/log/显存采样；没有也行，产物落本地记得拷走） |
| 模板 | 任意带 NVIDIA 驱动的模板即可（如官方 PyTorch 模板）；脚本自建 venv，不依赖镜像里的 torch |

**磁盘布局（同 verl-deploy 的思路）**：`/workspace` 是网络卷，I/O 慢（venv 小文件 import、60GB 权重加载都吃亏），所以 venv/上游仓库/权重默认全放 `/root/lingbot`（本地 NVMe）。代价是 pod **stop 会清空容器盘**——重跑 `setup_env.sh`（~5 分钟）+ `download_models.sh`（hf_transfer 重下权重 ~5–15 分钟）即恢复。实验产物默认写到 `/workspace/lingbot-outputs/`（体积小、断电不丢）。想全持久化（慢）：`WORK_DIR=/workspace/lingbot bash setup_env.sh`。

## 快速开始

```bash
# pod 上没有 GitHub SSH key，统一用 HTTPS clone（本仓库是公开的，无需 token）
cd /root && git clone https://github.com/xiefan46/lingbot-deploy.git && cd lingbot-deploy

bash setup_env.sh              # 环境 + FA3 源码编译（首次 ~30-60 分钟，建议 tmux 里跑）
bash download_models.sh dense  # 1.3B, ~12GB
bash run_dense_t2v.sh          # 冒烟：验证全链路，峰值 ~15GiB

bash download_models.sh moe    # 30B, ~121GB（下载较久，建议 tmux 里跑）
bash run_moe_t2v.sh            # 正菜：30B MoE 单卡 480p T2V
```

产物在 `/workspace/lingbot-outputs/<run>/t2v.mp4`（无 `/workspace` 卷时落在 `/root/lingbot/outputs/`），同目录的 `run.log` / `gpu_mem.csv` 是日志和显存采样（500ms 粒度），脚本结束时会打印**显存峰值和总耗时**。

所有脚本支持环境变量覆盖（`HEIGHT/WIDTH/STEPS/NUM_FRAMES/PROMPT_JSON/SEED/OUT_DIR/...`），路径约定见 `common.sh`。

## 预期数字（来自权重实测大小 + 推算，待本仓库实测校准）

| 项 | 数值 |
| --- | --- |
| MoE DiT 权重 (bf16) | 60.3 GB |
| 文本编码器 Qwen3-VL-4B (bf16) | 8.9 GB |
| VAE (fp32) | ~1 GB |
| 480p×121帧 激活/工作区（batch_cfg 开） | dense 实测 ~17 GB / **MoE 实测 ~31 GB** |
| **峰值 (H200，实测)** | **101.2 GiB / 141 GiB** |
| 耗时（5s@480p，40 步，H200，实测） | **6.86 s/步，全程 316s ≈ 5.3 min/条** |

**实测（2026-07-15/16，H200）**：Dense 1.3B：峰值 **29.2 GiB**（两台 pod 复测一致），2.53 s/步，全程 139s。**MoE 30B-A3B：峰值 101.2 GiB**，6.86 s/步，全程 316s。结论：H200 ✅（余量 ~40GB）；H20 96GB ⚠️ 需 BATCH_CFG=0 降压；**80GB 卡 ❌ bf16 官方配置不可行（须量化或大幅降配）**。原始数据在 HF `xiefan46/lingbot-env-cache` 的 `outputs/`。

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
2. **FlashAttention-3 是隐性硬依赖**：DiT 在 `--batch_cfg`（batch>1 走 packed 注意力）或 context parallel 下强制要求 FA3（`flash_attn_interface`），但上游 requirements 没带它，干净环境下上游单卡脚本默认参数必报 `flash_attn_interface.flash_attn_varlen_func is required`。`setup_env.sh` 会在 Hopper 上源码编译安装（~20–40 分钟，并行度 `FA3_MAX_JOBS` 默认 16）；若镜像的 nvcc 与 torch 的 CUDA 大版本不匹配（torch cpp_extension 会硬报错），脚本自动从 NVIDIA apt 源安装匹配的 cuda-toolkit（~4GB）。编译失败或 `SKIP_FA3=1` 跳过时的绕过：运行加 **`BATCH_CFG=0`**——B=1 走 SDPA 路径，完全不需要 FA3，代价是 CFG 两分支串行、稍慢。约定：**不改上游模型代码**，依赖问题一律在部署层解决。**编译只需一次**（仿 verl-deploy）：wheel 会自动上传 HF 缓存仓（`xiefan46/lingbot-env-cache`，上传需要一次 `hf auth login` 的 write token；仓库公开、下载免 token），之后的新 pod `setup_env.sh` 直接下载 wheel 秒装。
3. **text encoder 注意力**：上游默认 `flash_attention_3`；运行脚本保守默认 `sdpa`（文本编码只占总时长很小一部分）。FA3 装好后想对齐上游：`LINGBOT_QWEN_ATTN_IMPLEMENTATION=flash_attention_3`。
4. **不要开 refiner**（`--run_refiner`）：refiner 是**同尺寸的另一个 30B DiT**，双模型 120GB 权重 + 1080p 工作区，单卡（含 H200）放不下。上游 refiner 方案是 8 卡 CP8+FSDP。
5. **不要用 `LINGBOT_MOE_EXPERT_BACKEND=sglang_triton_fp8` 来省显存**：开源实现是运行时量化+额外缓存，bf16 权重不释放，显存反增 ~29GB（它是多卡 FSDP 下的提速方案）。单卡省显存只能用默认 `grouped_mm`。
6. **rewriter 不用部署**：冒烟用上游自带的结构化 prompt（`assets/cases/`）。要用自己的 prompt 时才需要 Qwen3.6-27B rewriter（bf16 ~54GB，届时单独规划）。
7. **宿主机驱动 <13 时靠 forward-compat 现场解决**：RunPod 不同宿主机驱动版本不一（实测踩过 12.4 的），驱动 12.x 时 cu13 torch 会 `CUDA not available`。`setup_env.sh` 会自动安装 **`cuda-compat-13-0`**（NVIDIA forward-compatibility，数据中心卡专用的用户态 libcuda），`common.sh` 自动把 `/usr/local/cuda-13.0/compat` 加进 `LD_LIBRARY_PATH`——之后 cu13 torch 和 FA3 缓存 wheel 照常可用。若该驱动分支不在 compat 支持矩阵内（验证仍报 CUDA not available）：要么开 pod 时筛 CUDA Version ≥13.0，要么就地降级 `rm -rf /root/lingbot/.venv && TORCH_CUDA_VARIANT=cu128 SKIP_FA3=1 bash setup_env.sh` + 运行全程 `BATCH_CFG=0`。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `common.sh` | 路径约定 + log/显存监控函数（被其他脚本 source） |
| `setup_env.sh` | 环境搭建（幂等，pod 重启后重跑即恢复） |
| `download_models.sh` | 下载权重 `dense\|moe\|all` |
| `run_dense_t2v.sh` | 1.3B 冒烟测试 |
| `run_moe_t2v.sh` | 30B MoE 单卡 T2V + 显存/耗时实测 |
