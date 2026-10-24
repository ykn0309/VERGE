# 计划 11：Chat Hard 扩大样本内部试验

## 范围

- 同一 RewardBench `filtered`/Chat Hard 数据源内进行新实验，不宣称官方榜单成绩或跨数据源泛化。
- 六个子集各 24 条训练、10 条冻结测试，总计 144 train + 60 test；从空经验池重新运行 `qwen3.5-flash` + DashScope 向量召回。
- 保持此前 smoke 的 6 条 train 和 6 条 test 在原 split，其他样本用固定 seed=42 稳定采样；本次 train/test 不得有重复样本 ID 或归一化后相同的题面。以前的 12 条运行结果不并入本次统计，也不在本次测试报告里设置额外的“重复评测”类别。固定分组的原始 source_id（按六子集 `CHAT_HARD_SUBSETS` 顺序）：train `[2496, 2621, 2645, 2821, 2893, 2919]`，test `[2489, 2539, 2669, 2799, 2901, 2929]`；旧输入 SHA-256 为 train `fe28de81d6e40cca464e7b2b743eac72c9182975051dc48a423c0c06c4704c67`、test `c1a2697b47fc2edfdf6ce2dbe822ae7631bf0bc03818f3329481b8efc10f271a`。
- 数据源固定 SHA `168d848cdbbea9764fae4a544dc9ca1e6cca4931`；输出 JSONL 和 provenance 记录源版本、seed、样本数量及输入哈希。`runs/` 中的样本和运行输出均不提交。

## 操作与验收

历史制备参数：六子集各取 34 条（24 train / 10 test）、seed=42、固定上游 revision，并保留现已删除的旧 6+6 smoke 的 split 归属。目前这些参数记录在 `configs/rewardbench-chat-hard-qwen-vector-144x60-stage14-parallel.yaml` 的 `dataset.config.prepare`；新实验只执行 `run --config <实验配置>`，数据目录完全不存在时自动准备，已存在时校验复用。若需要另外生成同成员划分，在新配置中将 `preserve_splits_from` 指向现存的 `runs/rewardbench-chat-hard-data-144x60`，同时为 `output_dir`、train/test 路径和运行目录指定新的路径；现有数据不重制。历史续跑配置及旧运行目录已按要求清理，仅保留固定的 144+60 数据和 provenance。当前阶段 14 使用全新目录运行：

```sh
python3 -m verge.cli run --config configs/rewardbench-chat-hard-qwen-vector-144x60-stage14-parallel.yaml
```

- 输入校验已通过：144/60、六子集 24/10、旧 6+6 split 归属、全体 204 条样本 ID 与归一化题面无重复、上游 revision 与 JSONL 哈希一致；新 split 已保存于被 Git 忽略的 `runs/` 中。
- 统计仅取本轮冻结 test 的 initial judgment；汇总时拒绝缺失或重复的轨迹 ID，报告总体正确数/60 与六子集各正确数/10，并记录运行完整性、成本和失败情况。独立的上一轮 12 条 smoke 不与本轮合并。
- 训练阶段写入 Memory 和 Beta；测试阶段不增加经验、不回写 credit。中断后不在原目录续跑；仅当完整轨迹前缀、Memory 证据和失败样本阶段经校验，才复制到新目录并从失败样本继续。其他中断点仍须重跑或人工排查。
- 首次正式运行目录 `runs/rewardbench-chat-hard-qwen-vector-144x60` 已在第 28 条训练样本的 `revision_1.executor` 因修订后不足 4 条标准而停止；保留其 27 条完整训练轨迹、116 次 Qwen 请求、22 次 embedding 请求和 104430 个已记录 token（尚未开始测试），不参与新实验统计。按照 [原子操作修订](../历史设计/2026-10-24-V1/设计方案/07-原子操作修订.md) 的结构错误策略，修订输出最多纠错重试 2 次，不占用两次有效 revision attempts，成功轨迹中的 `validation_failures` 保留校验错误用于审计；随后已在全新目录 `runs/rewardbench-chat-hard-qwen-vector-144x60-restart1` 从空经验池重新运行。
- 第二次正式运行 `runs/rewardbench-chat-hard-qwen-vector-144x60-restart1` 在第 96 条训练样本修订结果经两次纠错仍不足 4 条时中断：95 条完整训练轨迹、506 次 Qwen 请求、125 次 embedding 请求，合计 525220 个记录 token；测试尚未开始。现将初始和修订 rubric 范围统一为 2–8；如修订后仍超范围，记录该次无效修订、继续剩余尝试，不向 Judge 传入无效 rubric。重新从空经验池在 `runs/rewardbench-chat-hard-qwen-vector-144x60-restart2` 独立运行，汇总命令的 `invalid_revision_attempts` 单独报告无效修订次数；前两次部分训练结果均不并入正式评估。
- 第三次正式运行 `runs/rewardbench-chat-hard-qwen-vector-144x60-restart2` 在第 126 条训练样本 `initial_generator` 停止：125 条完整训练轨迹、617 次 Qwen 请求、155 次 embedding 请求，合计 540147 个记录 token；测试 0/60，已完成样本中的无效修订数为 0。模型请求返回 HTTP 200，但生成的初始 rubric 未通过本地约束（base ID、2–8 条标准或 source_experiences 引用/去重之一）；日志未保存原始输出，不能确定是哪一项。已核验前 125 条轨迹严格匹配数据集前缀、输入哈希未变、Memory 中不存在失败样本的来源或证据；旧目录保留供审计，通过显式 `--resume-from` 在新目录复制此前 125 条完整轨迹及经验池，剩余 19 条训练和 60 条测试才进行新的请求。续跑启用最多 2 次初始生成校验纠错并在新 manifest 保留旧/新代码指纹和文件哈希；旧部分训练仅作为续跑后整体训练前缀，不单独算正式测试成绩。
- **完成结果**（`runs/rewardbench-chat-hard-qwen-vector-144x60-resume1`）：144/144 训练与 60/60 冻结测试均有完整、无重复的轨迹；初始判断正确 **43/60 = 71.67%**。六子集各 10 条，正确数依次为 `mt-bench-hard 9`、`llmbar-natural 9`、`llmbar-adver-neighbor 8`、`llmbar-adver-GPTInst 6`、`llmbar-adver-GPTOut 6`、`llmbar-adver-manual 5`。总计 898 次 Qwen 与 259 次 embedding 请求（旧检查点已产生的成功及失败请求也计入），记录 token 分别为 942868、12962，合计 955830；续跑段新请求为 281 次 Qwen、104 次 embedding，其中另有 1 条训练样本的初始生成经历了 1 次额外校验纠错重试。成功轨迹的无效修订尝试为 0；经验池最终 20 条，测试样本未写入经验来源。旧目录三份检查点文件的 SHA-256 与续跑 manifest 一致；测试结果仅取新目录中 60 条完整测试轨迹。金额未由请求日志直接记录，勿将 token 当作实际费用。
- 已在新输入校验后删除旧 6+6 smoke 的数据、成功运行目录和两次失败试运行目录；计划 09 中 5/6 仅作为历史记录，不与本轮统计合并。

## 注意

- 仅凭这组实验不能断言向量召回优于 Fake 或模型对未见来源泛化；如果训练后经验数量不超过 Top 20，所有经验仍会被召回。
- 历史 6+6 smoke 共 48 次 Qwen 请求（训练 30、测试 18）；按相同请求密度粗估本轮约 144×5 + 60×3 = 900 次 Qwen 请求，向量检索的 embedding 请求另计，经验池变大后提示长度/实际费用可能增加。正式成本与时长须按运行日志核算，启动之前先验证输入和 key 环境；不要把估算视为费用上限。
