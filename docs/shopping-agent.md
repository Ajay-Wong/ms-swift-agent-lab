# 商品查询 Agent：从工具到 SFT

这个示例先实现一个可以自动判定结果的小场景：按品类和预算找到最便宜的现货商品。模型为 Qwen3-1.7B，关闭 thinking。所有价格是整数元，同价按商品 ID 字符串升序选择。

目前包含真实工具环境、独立答案判定、可复现的数据生成、模型在线工具评估和 SFT 脚本。**GRPO 优化器与 ms-swift 多轮 rollout 插件尚未接入**；`judge_episode()` 已提供可复用的 0/1 结果奖励。

## 目录与边界

| 文件 | 功能 |
| --- | --- |
| `agent_lab/shopping/environment.py` | 两个工具、每轮独立状态、答案 oracle、轨迹重放评分 |
| `agent_lab/shopping/tasks.py` | 六种任务的确定性生成 |
| `agent_lab/shopping/trajectory.py` | 正确示范轨迹、严格输出解析、有限步数执行循环 |
| `agent_lab/shopping/generate.py` | 导出任务世界和 ms-swift SFT JSONL |
| `agent_lab/shopping/check_template.py` | 实际模板编码、长度、监督目标与损失掩码检查 |
| `agent_lab/shopping/evaluate.py` | 脚本策略、原始模型、LoRA 的真实工具评估 |
| `scripts/train_shopping_sft.sh` | 正式 LoRA SFT，保存 checkpoint |
| `tests/test_shopping.py` | 边界、状态隔离、无效调用、伪造结果、数据隔离测试 |

生成器、环境、判定器和单元测试只用 Python 标准库，不需要 GPU 或外部模型 API。模型评估与 SFT 在已有的训练容器内执行。

## 1. 在宿主机生成数据

以下宿主机命令均在仓库根目录执行。

```bash
python3 -m unittest discover -s tests -v
python3 -m agent_lab.shopping.demo --index 4
python3 -m agent_lab.shopping.generate \
  --output workspace/shopping/data \
  --train 1000 --val 200 --test 200 --seed 42
```

如果同一路径已经生成过数据，命令会拒绝覆盖。复用已有数据即可；确实要重建时，换目录或显式使用 `--overwrite`。

场景包括正常选品、无候选商品、全部缺货、价格刚好等于预算、同价排序和最便宜商品缺货。不同 split 使用独立的随机状态、商品 ID 与库存；种子与文件 SHA-256 记录在 `manifest.json`。

| 生成文件 | 用途 |
| --- | --- |
| `train.sft.jsonl` / `val.sft.jsonl` | 训练和验证 loss；只包含可见交互与工具定义 |
| `test.sft.jsonl` | 保留的正确示范，仅用于离线检查，不传给训练命令 |
| `*.tasks.jsonl` | 环境中的完整商品世界，供工具执行与判定使用 |
| `manifest.json` | 数据规模、种子、各场景数量、文件哈希 |

模型只收到用户问题、工具定义和已执行工具的返回内容，**不会收到完整商品世界或正确答案**。

## 2. 更新容器挂载并检查真实模板

```bash
docker compose up -d
docker exec ms-swift-agent python -m agent_lab.shopping.check_template \
  --data /workspace/shopping/data/train.sft.jsonl
docker exec ms-swift-agent python -m agent_lab.shopping.check_template \
  --data /workspace/shopping/data/val.sft.jsonl
```

新代码以只读方式挂载到 `/opt/agent-lab/agent_lab`，无需重建镜像。SFT 和评估均使用 ms-swift 的 Qwen3 + Hermes 工具模板，并关闭 thinking。

检查会确认搜索和库存调用都进入监督目标、工具返回没有进入监督目标、最终答案进入监督目标，以及轨迹没有超过 2048 tokens。训练另设 `strict=true`，拒绝有问题的数据行。

## 3. 评估原始模型

先用少量验证任务确认流程，再扩大到完整验证集。下面命令不会更新模型权重：

```bash
docker exec ms-swift-agent python -m agent_lab.shopping.evaluate \
  --backend model \
  --tasks /workspace/shopping/data/val.tasks.jsonl \
  --output /workspace/shopping/eval-base \
  --limit 12
```

移除 `--limit` 会评估整个文件。完整轨迹保存在 `episodes.jsonl`，汇总在 `summary.json`；已有同名结果默认不会被覆盖。

`--backend scripted` 执行确定性的示范策略，可在宿主机无 GPU 检查环境；它的成功率不是模型成功率。

## 4. 启动正式 SFT

```bash
docker exec -it ms-swift-agent bash /opt/agent-scripts/train_shopping_sft.sh
```

默认配置：

- 训练 1000 条轨迹，独立验证集 200 条，2 epochs。
- BF16 LoRA，rank 16，作用于 Q/K/V/O projection。
- micro batch 1，梯度累积 8，上下文 2048，开启 gradient checkpointing。
- 学习率 1e-4，warmup 比例 0.05；每 50 步评估。`save_steps=0.5` 按总训练步数的 50% 间隔保存（向上取整），默认 250 步时在第 125、250 步保存，保留两个 checkpoint。
- 修改保存配置只对新启动的训练生效；已经运行的训练和已有 checkpoint 不会自动改变。保留数量限制针对单次训练的版本目录，不会清理其他训练版本。
- 结果写入 `/workspace/shopping/sft/` 下的版本目录，原始模型只读。

训练结束后，记录日志中的 `last_model_checkpoint` 路径。验收脚本 `smoke_sft.sh` 与这里不同：它不会保存模型，不能代替正式训练。

更换数据或输出位置可通过环境变量：

```bash
docker exec -it \
  -e DATA_DIR=/workspace/shopping/data \
  -e OUTPUT_DIR=/workspace/shopping/sft \
  ms-swift-agent bash /opt/agent-scripts/train_shopping_sft.sh
```

只验证训练流程时，可额外传入 `--max_steps 2 --gradient_accumulation_steps 1 --save_steps 2 --eval_steps 2 --warmup_ratio 0 --lr_scheduler_type constant`。这只适合验收，不能据此判断训练效果。

## 5. 加载 LoRA 并比较

将下面的 `CHECKPOINT` 改为训练日志中实际给出的容器路径：

```bash
CHECKPOINT='/workspace/shopping/sft/实际版本目录/checkpoint-实际步数'
docker exec ms-swift-agent python -m agent_lab.shopping.evaluate \
  --backend model --adapter "$CHECKPOINT" \
  --tasks /workspace/shopping/data/val.tasks.jsonl \
  --output /workspace/shopping/eval-sft \
  --limit 12
```

比较前后必须使用相同任务文件和相同 `--limit`。汇总同时记录任务文件哈希与实际评估的任务 ID，可以检查可比性。选定配置之后，再在保留的 `test.tasks.jsonl` 上评估最终结果。

关键指标：任务成功率、答案准确率、有效工具调用率、平均工具调用次数。验证 loss 只能作为训练信号，不能替代实际任务成功率。

## 判定与后续 RL

工具接口：

```text
search_products(category, max_price) -> products（不含库存）
get_stock(product_ids) -> stock（只接受本轮已查询出的 ID）
```

每个任务最多三次工具调用。最终输出必须是：

```json
{"product_id":"P示例-00","price":199}
```

没有符合条件的商品时两个字段均为 `null`。

成功必须同时满足：

1. 最终商品和价格正确，输出字段和类型正确。
2. 按用户的品类与预算执行过查询。
3. 查询过全部候选商品的库存；无候选商品时无需库存调用。
4. 工具调用有效、没有超出次数限制，记录的工具返回与环境重放一致。

因此，“猜对答案但没有执行必要查询”也记为失败。这是本示例明确的流程约束；将来若改为允许提前停止或只查询较便宜的候选商品，应同步修改系统提示、示范策略、判定和测试。

`judge_episode()` 返回 `reward=1.0` 或 `0.0`。后续 GRPO 需要让训练器按任务创建 `ShoppingEnv`，执行多轮 rollout，再把真实事件记录和最终答案交给这个判定器；**当前没有宣称或启动完整 GRPO 训练**。
