# 计划 09：RewardBench Chat Hard 内部分割与真实模型试运行

## 状态

```text
状态：已完成（12 条内部试运行；不等于官方榜单评测）
前置计划：08 — 真实 LLM Backend
后续方向：独立训练数据与完整 held-out evaluation
```

## 目标与边界

从官方 [`allenai/reward-bench`](https://huggingface.co/datasets/allenai/reward-bench) 的 **`filtered`** split 中按官方 [Chat Hard 分类](https://github.com/allenai/reward-bench/blob/main/rewardbench/constants.py)选取六个子集：`mt-bench-hard`、`llmbar-natural`、`llmbar-adver-neighbor`、`llmbar-adver-GPTInst`、`llmbar-adver-GPTOut`、`llmbar-adver-manual`。原始 Chat Hard 包含 456 条；先每个子集采 2 条，1 条 train、1 条 test，共 6+6。训练从空经验池开始，测试阶段经验池/Beta 状态只读。

**重要：** 本次训练使用 Chat Hard 的一部分，只报告项目内部剩余样本的初始 rubric 偏好判断率，不能称为官方 RewardBench 榜单分数，也不能据此评估对完整 Chat Hard 的泛化能力。要做官方意义的 held-out 评测，必须用独立于 RewardBench 的训练数据形成经验池，再在未参与训练的完整 benchmark 上评估。

## 数据与复现

- 按上游 40 位提交 SHA 固定源版本，只读 `filtered`（2985 条），过滤六个 Chat Hard 子集；使用 Hugging Face Hub 下载 Parquet 并以 PyArrow 读取，避免全局环境的 `datasets`/SciPy 二进制依赖故障。本地导出 JSONL，沿用已有 `JsonlDataset` 与 Pipeline，不改训练/测试规则。
- 固定 `seed=42`；按子集进行稳定哈希排序，同一 prompt（归一化空白）不得进入不同 split；导出 `sample_id`、`task`、`chosen`、`rejected`、`metadata`（原始 id、subset、源版本）。数据集不提交到 Git。
- `provenance.json` 保存源 SHA、样本数、随机种子及两个文件的 SHA-256；run manifest 额外记录输入 JSONL 的文件 SHA-256，统计前核对 test 文件未变化并确认六个子集均有等量样本。新输出目录避免覆盖。上游许可为 ODC-BY，组成子集还须遵守 MT-Bench（Apache-2.0）及 LLMBar（MIT）许可。

## 操作

```sh
python3 -m pip install -r requirements.txt
# 仅使用实验配置：新数据目录不存在时 run 自动准备，已存在时校验复用。
# 原 6+6 数据已删除，此处为历史试验的执行命令；运行前须指定全新输出目录。
# DASHSCOPE_API_KEY_2 从环境读取，不写入配置或输出目录。
python3 -m verge.cli run --config configs/rewardbench-chat-hard-qwen.yaml
python3 -m verge.cli summarize-rewardbench --run-dir runs/rewardbench-chat-hard-qwen
```

实际请求会产生模型费用；`runs/` 已被 Git 忽略，重跑需更换准备目录与实验输出目录。运行失败后不在同目录恢复：Memory 可能已经写入，须新建目录重跑。

## 验收与完成记录

- 源 SHA：`168d848cdbbea9764fae4a544dc9ca1e6cca4931`；官方 `filtered` 2985 条，Chat Hard 456 条（六子集为 37/100/134/92/47/46）；按 seed=42 各取 1 train + 1 test，共 6+6，训练与测试题面无交叉。
- Qwen `qwen3.5-flash` 完整跑完 6 条 train 和 6 条 test；48 次模型请求全部 HTTP 200；完整轨迹 12 条，输入哈希与 run manifest 一致。训练初始判断 4/6，1 条修订成功、1 条最终失败，经验池生成 2 条经验；测试无修订、无经验池写入。
- **内部测试初始偏好判断 5/6（83.3%）**，唯一未判对的是 `llmbar-adver-GPTOut`。每子集只有 1 条 test，方差极高；训练来自同一官方 benchmark 的另 6 条，因此结果既不具统计代表性，也**不能**作为 RewardBench 官方榜单分数。
- 当时 `python3 -m pytest -q`：119 passed。早期两次真实试运行分别因 Judge 把非待判断标准返回、Revision Generator 提供了执行器专属 `before_criterion` 而中止；相应改为 Judge 仅接收待判断标准、执行器独立补全旧标准。原始 6+6 smoke 及失败试运行的本地数据/运行目录已在计划 11 新划分核验后依用户要求删除；5/6 仅为历史记录，不能再从本地旧轨迹复算。
- 下一步：使用独立于 RewardBench 的训练集训练经验池，再对未参与学习的完整 Chat Hard 运行冻结经验池的评测，并报告每个子集的样本数、准确率及不确定性。
