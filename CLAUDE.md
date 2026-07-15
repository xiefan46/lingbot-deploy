# lingbot-deploy

RunPod H200 单卡部署 LingBot-Video（30B-A3B MoE，arXiv 2607.07675）的脚本仓库。

## 背景

- 母项目是本地 `serving` 调研仓库（研究 30–50B 具身 MoE 世界模型的单卡部署）；LingBot-Video 是第一个参照系统。
- 架构与显存分析见 serving 仓库 `research/2026-07-15-lingbot-video-serving-analysis.md`；本仓库负责把分析落地成 RunPod 上可一键执行的部署与实测脚本。
- 目标：跑通 30B MoE bf16 单卡推理，实测显存峰值/耗时，回填校准调研文档的显存预算表。

## 约定

- **lingbot-video 代码统一用实验 fork [xiefan46/lingbot-video](https://github.com/xiefan46/lingbot-video)**（`setup_env.sh` 默认 clone 它，实验改动提交到 fork）；上游 Robbyant/lingbot-video 只作跟进合并来源。
- 脚本风格参照 [xiefan46/verl-deploy](https://github.com/xiefan46/verl-deploy)：中文注释、幂等、彩色 log（`common.sh`）、末尾验证块。
- 磁盘布局同 verl-deploy：**重 I/O（venv/上游仓库/权重）放本地 NVMe `/root/lingbot`，不放 `/workspace` 网络卷**（网络卷 I/O 慢）。pod stop 清空容器盘，重跑 `setup_env.sh` + `download_models.sh` 恢复。实验产物（小文件）默认写 `/workspace/lingbot-outputs/` 防丢，无卷时回落本地。
- 所有运行参数用环境变量覆盖，不改脚本本体。
- **pod 上 clone 一律用 HTTPS**（`https://github.com/xiefan46/lingbot-deploy.git`）：RunPod pod 里没有用户的 GitHub SSH key，SSH 方式会 `Permission denied (publickey)`。本仓库公开，HTTPS 匿名可读；上游 lingbot-video 的 clone（setup_env.sh）本来就是 HTTPS。
- 新增实验脚本时同步更新 README 的文件说明表。

## 关键技术事实（改脚本前必读）

- **不改上游/fork 的模型代码——用户拍板的决定**。依赖缺口一律在部署层（setup_env.sh）解决，不给 lingbot-video 打补丁。
- DiT 的 packed 注意力路径（`--batch_cfg` 即 batch>1，或 context parallel）硬依赖 FA3 的 `flash_attn_interface`，上游 requirements 未带 → setup_env.sh 源码编译 flash-attention 的 `hopper/` 子目录安装；无 FA3 时运行加 `BATCH_CFG=0` 绕过（B=1 走 SDPA 路径，数值等价、稍慢）。
- **FA3 wheel 缓存（仿 verl-deploy 的 env-cache）**：首次编译后 wheel（stable ABI + abi3）自动上传 HF dataset 仓 `HF_CACHE_REPO`（默认 `xiefan46/lingbot-env-cache`，公开）；setup_env.sh 优先下载缓存，未命中才编译。上传需 write token（`hf auth login`）；`FA3_REBUILD=1` 强制重编。
- 上游 pin `torch==2.12.0.dev20260220+cu130`；MoE 默认后端 `grouped_mm` 依赖 `torch._grouped_mm`。
- 单卡禁开 refiner（同尺寸 30B ×2 放不下）；禁用 `sglang_triton_fp8` 省显存（运行时量化会额外缓存，显存反增）。
- text encoder（Qwen3-VL-4B）默认 FA3 → 脚本保守默认 `LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa`，FA3 装好后可改回 `flash_attention_3`。
