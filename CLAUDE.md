# lingbot-deploy

RunPod H200 单卡部署 LingBot-Video（30B-A3B MoE，arXiv 2607.07675）的脚本仓库。

## 背景

- 母项目是本地 `serving` 调研仓库（研究 30–50B 具身 MoE 世界模型的单卡部署）；LingBot-Video 是第一个参照系统。
- 架构与显存分析见 serving 仓库 `research/2026-07-15-lingbot-video-serving-analysis.md`；本仓库负责把分析落地成 RunPod 上可一键执行的部署与实测脚本。
- 目标：跑通 30B MoE bf16 单卡推理，实测显存峰值/耗时，回填校准调研文档的显存预算表。

## 约定

- 脚本风格参照 [xiefan46/verl-deploy](https://github.com/xiefan46/verl-deploy)：中文注释、幂等、彩色 log（`common.sh`）、末尾验证块。
- 磁盘布局同 verl-deploy：**重 I/O（venv/上游仓库/权重）放本地 NVMe `/root/lingbot`，不放 `/workspace` 网络卷**（网络卷 I/O 慢）。pod stop 清空容器盘，重跑 `setup_env.sh` + `download_models.sh` 恢复。实验产物（小文件）默认写 `/workspace/lingbot-outputs/` 防丢，无卷时回落本地。
- 所有运行参数用环境变量覆盖，不改脚本本体。
- **pod 上 clone 一律用 HTTPS**（`https://github.com/xiefan46/lingbot-deploy.git`）：RunPod pod 里没有用户的 GitHub SSH key，SSH 方式会 `Permission denied (publickey)`。本仓库公开，HTTPS 匿名可读；上游 lingbot-video 的 clone（setup_env.sh）本来就是 HTTPS。
- 新增实验脚本时同步更新 README 的文件说明表。

## 关键技术事实（改脚本前必读）

- 上游 pin `torch==2.12.0.dev20260220+cu130`；MoE 默认后端 `grouped_mm` 依赖 `torch._grouped_mm`。
- 单卡禁开 refiner（同尺寸 30B ×2 放不下）；禁用 `sglang_triton_fp8` 省显存（运行时量化会额外缓存，显存反增）。
- text encoder（Qwen3-VL-4B）默认 FA3 但环境未装 → 脚本统一 `LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa`。
