# ms-swift Agent Lab

面向 **RTX 5090 Laptop 24GB** 的纯文本 Agent 训练环境：使用 **Qwen3-1.7B**，先做 LoRA SFT，再接入多轮 GRPO。提供 Docker 配置、依赖修复、模型下载和环境验收脚本。

已包含商品查询 Agent 的工具环境、数据生成、轨迹判定、原始模型/LoRA 评估和正式 SFT 启动脚本。先阅读 [商品查询 Agent 操作指南](docs/shopping-agent.md)。多轮 GRPO 训练器尚未接入。环境验证见 [验收记录](VALIDATION.md)。

| 组件 | 版本 |
| --- | --- |
| ms-swift | 4.5.3 |
| PyTorch / CUDA | 2.13.0 / 13.0 |
| Transformers | 5.16.1 |
| PEFT / TRL | 0.20.0 / 0.29.1 |
| vLLM | 0.28.0 |

## 准备

需要 Linux、兼容 CUDA 13.0 的 NVIDIA 驱动、Docker Engine、Docker Compose 和 NVIDIA Container Toolkit。宿主机模型下载与校验脚本使用 Python 3.11+。

```bash
git clone https://github.com/Ajay-Wong/ms-swift-agent-lab.git
cd ms-swift-agent-lab
cp .env.example .env
```

根据需要修改 `.env`：`MODELS_DIR` 默认为宿主机 `$HOME/Models`，`LOCAL_UID` / `LOCAL_GID` 默认为 1000，可用 `id -u` / `id -g` 查看。本机继续使用 `/home/ajay/Models`。

仓库只保存配置、脚本和文件校验清单，不包含模型权重、Python 安装包、缓存、训练结果或本机日志。

## 构建镜像

```bash
bash scripts/restore-image.sh
```

脚本依次拉取官方基础镜像、下载固定版本的 Python wheels，再离线构建修复层。首次官方镜像下载约 10.7 GiB，需要为展开后的镜像预留充足磁盘空间。

- 基础镜像：`ajay/ms-swift-official:4.5.3-cu130`
- 最终镜像：`ajay/ms-swift:4.5.3-cu130`
- 运行容器：`ms-swift-agent`

已有基础镜像和 wheels 时，可直接使用 `docker compose build` 重建。镜像本身未发布到镜像仓库，需要本地构建。

## 下载模型

如果已有完整的 `Qwen3-1.7B` 目录，可跳过下载。

```bash
python3 -m venv .venv-download
.venv-download/bin/pip install modelscope==1.40.1
.venv-download/bin/python scripts/download_model.py
python3 scripts/verify_model_files.py
```

默认保存到 `$HOME/Models/Qwen3-1.7B`。使用自定义目录时，下载脚本和校验脚本需显式指定路径，并使 `.env` 中的 `MODELS_DIR` 与其一致：

```bash
.venv-download/bin/python scripts/download_model.py --models-dir /path/to/Models
python3 scripts/verify_model_files.py --model-dir /path/to/Models/Qwen3-1.7B
```

模型从 Qwen 在 ModelScope 的官方仓库下载，遵循模型仓库自己的许可。`model-files.json` 保存本次验收使用的文件大小与 SHA-256；上游文件更新后，校验会报告差异。

## 启动与进入

```bash
bash scripts/start.sh
docker exec -it ms-swift-agent bash
```

模型以只读方式挂载到 `/models/Qwen3-1.7B`。容器内 `/workspace` 对应仓库下的 `workspace/`，用于保存训练数据、输出与缓存。容器使用配置的普通用户，不开放网络端口。

停止容器：

```bash
bash scripts/stop.sh
```

再次启动可运行 `bash scripts/start.sh`；`docker compose down` 只删除容器和网络，模型及工作目录仍保留。

## 环境验收

建议按顺序执行，避免同时占用显存：

```bash
docker compose exec trainer python -m pip check
docker compose exec trainer python /opt/agent-scripts/check_env.py
docker compose exec trainer python /opt/agent-scripts/check_model.py
docker compose exec trainer bash /opt/agent-scripts/smoke_sft.sh
docker compose exec trainer python /opt/agent-scripts/check_vllm.py
```

- `check_env.py`：依赖版本、GPU BF16 前向和反向运算。
- `check_model.py`：在内存中添加 LoRA，验证梯度与短文本生成，不修改原始模型。
- `smoke_sft.sh`：用四条小样本完成一次固定非零学习率的 LoRA 更新，不保存模型权重；日志位于 `workspace/smoke-sft/`。
- `check_vllm.py`：短上下文、低并发推理验证，不等同于完整多轮 Agent GRPO 验证。

## 依赖修复

官方基础镜像预装依赖存在冲突。Dockerfile 将 NumPy 固定为 2.3.5，并调整 Pydantic、Starlette、Setuptools 等版本；具体见 [requirements-fixes.txt](requirements-fixes.txt)。构建最后运行 `pip check`，检查失败就会中止构建。

移除了存在冲突或缺失依赖的可选组件 `ms-agent`、`ms-opencompass`、`decord`、旧版 `gradio`。本环境不包含 SWIFT WebUI，面向纯文本命令行 SFT/GRPO；上述组件不是工具调用轨迹 SFT 或自定义工具环境 GRPO 的必要依赖。

## 后续训练起点

商品查询示例已提供完整工具调用轨迹生成与 SFT 脚本，默认 BF16 LoRA、rank 16、micro batch 1、长度 2048、gradient checkpointing。具体生成、评估和训练命令见 [操作指南](docs/shopping-agent.md)。

GRPO 建议从短轨迹、低并发的 colocate rollout 开始。验收主机内存约 16GB，不宜依赖大量 CPU offload；正式训练的显存需求需实际测量。

## 官方资料

- [ms-swift 安装与镜像](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/GetStarted/SWIFT-installation.md)
- [Qwen3-1.7B：ModelScope](https://modelscope.cn/models/Qwen/Qwen3-1.7B)
- [Qwen3-1.7B：模型说明](https://huggingface.co/Qwen/Qwen3-1.7B)
