# 计划 11：Chat Hard 扩大样本内部试验

## 范围

- 同一 RewardBench `filtered`/Chat Hard 数据源内进行新实验，不宣称官方榜单成绩或跨数据源泛化。
- 六个子集各 24 条训练、10 条冻结测试，总计 144 train + 60 test；从空经验池重新运行 `qwen3.5-flash` + DashScope 向量召回。
- 保持此前 smoke 的 6 条 train 和 6 条 test 在原 split，其他样本用固定 seed=42 稳定采样；本次 train/test 不得有重复样本 ID 或归一化后相同的题面。以前的 12 条运行结果不并入本次统计，也不在本次测试报告里设置额外的“重复评测”类别。固定分组的原始 source_id（按六子集 `CHAT_HARD_SUBSETS` 顺序）：train `[2496, 2621, 2645, 2821, 2893, 2919]`，test `[2489, 2539, 2669, 2799, 2901, 2929]`；旧输入 SHA-256 为 train `fe28de81d6e40cca464e7b2b743eac72c9182975051dc48a423c0c06c4704c67`、test `c1a2697b47fc2edfdf6ce2dbe822ae7631bf0bc03818f3329481b8efc10f271a`。
- 数据源固定 SHA `168d848cdbbea9764fae4a544dc9ca1e6cca4931`；输出 JSONL 和 provenance 记录源版本、seed、样本数量及输入哈希。`runs/` 中的样本和运行输出均不提交。

## 操作与验收

数据已通过如下命令制备（**历史命令**：旧 smoke 数据目录随后按用户要求删除，不要原样重跑）：

```sh
python3 -m verge.cli prepare-rewardbench \
  --output-dir runs/rewardbench-chat-hard-data-144x60 \
  --revision 168d848cdbbea9764fae4a544dc9ca1e6cca4931 \
  --seed 42 --per-subset 34 --train-per-subset 24 \
  --preserve-splits-from runs/rewardbench-chat-hard-data
```

如需另外生成同成员划分，可将 `--preserve-splits-from` 指向现存的 `runs/rewardbench-chat-hard-data-144x60`，并使用全新输出目录；原数据文件及其中的 provenance 已保留（成员相同，但行顺序/文件哈希可能不同）。实验命令：

```sh
python3 -m verge.cli run --config configs/rewardbench-chat-hard-qwen-vector-144x60.yaml
python3 -m verge.cli summarize-rewardbench \
  --run-dir runs/rewardbench-chat-hard-qwen-vector-144x60
```

- 输入校验已通过：144/60、六子集 24/10、旧 6+6 split 归属、全体 204 条样本 ID 与归一化题面无重复、上游 revision 与 JSONL 哈希一致；新 split 已保存于被 Git 忽略的 `runs/` 中。
- 统计仅取本轮冻结 test 的 initial judgment；汇总时拒绝缺失或重复的轨迹 ID，报告总体正确数/60 与六子集各正确数/10，并记录运行完整性、成本和失败情况。独立的上一轮 12 条 smoke 不与本轮合并。
- 训练阶段写入 Memory 和 Beta；测试阶段不增加经验、不回写 credit。若实验中途失败，按 V1 约束不在原目录续跑，先保留报错及已完成进度再决定是否重跑。
- 已在新输入校验后删除旧 6+6 smoke 的数据、成功运行目录和两次失败试运行目录；计划 09 中 5/6 仅作为历史记录，不与本轮统计合并。

## 注意

- 仅凭这组实验不能断言向量召回优于 Fake 或模型对未见来源泛化；如果训练后经验数量不超过 Top 20，所有经验仍会被召回。
- 历史 6+6 smoke 共 48 次 Qwen 请求（训练 30、测试 18）；按相同请求密度粗估本轮约 144×5 + 60×3 = 900 次 Qwen 请求，经验池变大后提示长度/实际费用可能增加。正式成本与时长须按运行日志核算，启动之前先验证输入和 key 环境；不要把估算视为费用上限。
