# VERGE 项目计划

## 1. 文件职责

本文件是项目计划的纲领与状态索引，只记录：

- 未来的大方向；
- 子计划文档引用；
- 高层 TODO；
- 已完成计划；
- 当前状态与下一项计划。

具体任务、测试和验收标准统一写在 [`docs/plans/`](docs/plans/) 中，每个阶段一个文件。

方法规则以 [`docs/设计方案.md`](docs/设计方案.md) 为准，工程约束以 [`docs/工程设计.md`](docs/工程设计.md) 为准。

## 2. 当前状态

```text
当前阶段：14 已完成固定 144+60 数据的真实模型同源回归实验（训练修复 26/43，测试 45/60）；13 的人工复核与 12 的 D 仍待完成
下一个方向：复核本次 Critic analysis/operations、执行与第二轮历史及错误案例；再安排独立 held-out 评估
当前阻塞：无代码实施前置消融；缺少人工盲标与未参与提示词调整的新数据，已分析的 Chat Hard 测试不能作无偏终测
```

已完成：

- [x] 方法设计文档
- [x] 工程设计文档
- [x] 最小 Python 项目骨架
- [x] 项目级 Agent 约束
- [x] 分阶段实现计划拆分

## 3. V1 实现计划

按顺序执行以下计划：

- [x] [01 — 领域模型](docs/plans/01-领域模型.md)
- [x] [02 — 确定性计算](docs/plans/02-确定性计算.md)
- [x] [03 — 配置与数据输入](docs/plans/03-配置与数据输入.md)
- [x] [04 — LLM 接口与 Fake Modules](docs/plans/04-LLM接口与Fake模块.md)
- [x] [05 — Experience Memory 与 Trajectory](docs/plans/05-经验记忆与轨迹.md)
- [x] [06 — Pipeline Runner](docs/plans/06-Pipeline-Runner.md)
- [x] [07 — CLI](docs/plans/07-CLI.md)
- [x] [08 — 真实 LLM Backend](docs/plans/08-真实LLM后端.md)
- [x] [09 — RewardBench Chat Hard](docs/plans/09-RewardBench-Chat-Hard.md)
- [x] [10 — DashScope 向量召回](docs/plans/10-DashScope-向量召回.md)
- [x] [11 — Chat Hard 扩大样本内部试验](docs/plans/11-Chat-Hard-扩大样本内部试验.md)
- [ ] [12 — 经验适用性与来源归因改进](docs/plans/12-经验适用性与来源归因改进.md)（A/B/C 已实现，D 待评估）
- [ ] [13 — Chat Hard 修订失败诊断与定向改进](docs/plans/13-Chat-Hard-修订失败诊断与定向改进.md)（25 条初步诊断完成，人工复核待做；后续方法改动另见计划 14）
- [x] [14 — Critic 诊断与原子修订合一](docs/plans/14-Critic-诊断与原子修订合一.md)（新基线已实现；含测试并行与进度的离线测试 203 项通过，真实模型与效果尚未评估）

### 下一项方向

> Chat Hard 同源 144 训练 + 60 测试已有 v6 与后续 v8 回归；测试集已用于诊断，非无偏终测。根据计划 13 的失败分析及后续讨论，计划 14 已直接把 Critic 诊断与原子操作合并，不要求预先做旧/新消融；修订操作是否含候选内容只通过提示词约束，不新增内容审查。阶段 13 的 Judge/标签人工复核和阶段 12 的独立开发/终测仍须另行完成，不能以重复测试集结果替代泛化验证。

## 4. V1 目标

V1 应实现：

```text
读取 preference samples
→ 生成并判断 initial rubric
→ 立即写入已有经验的下游信用
→ initial failure 时最多执行两次同基线修订
→ successful revision 后生成候选经验
→ 分配 revision credit 并执行 Admission
→ MERGE / ADD 到 Experience Memory
→ 保存包含主要阶段结果的 trajectory
```

总体要求：

- 单进程运行；训练顺序执行，冻结经验池的测试可在进程内限流并发，轨迹仍按数据集顺序写入；
- 可测试、可复算；
- 使用 fake modules 完成离线端到端测试；
- 可替换为真实 LLM backend；
- Memory 在下游信用计算和 Fusion 阶段分别写入，trajectory 保存阶段结果；V1 不处理通用跨阶段回滚、崩溃恢复、重复运行去重或跨文件原子性。实验 11 仅允许对已核验未写入 Memory 的初始生成断点，将完整前缀复制到新目录续跑；详见工程设计。

## 5. 后续方向

V1 完成后再评估：

- [ ] 支持更多数据集适配器
- [ ] 增加更多 OpenAI-compatible 模型配置
- [ ] 建立 held-out evaluation 与指标汇总
- [ ] 实现方法消融实验
- [ ] 比较 retrieval、credit 和 reliability 策略
- [ ] 增加批量实验与结果分析工具
- [ ] 根据实际代码规模决定是否拆分扁平模块

这些方向尚未拆分为执行计划，不应提前实现。

## 6. 全局非目标

当前不计划实现：

- 面向运行与修改实验状态的 Web UI（本地只读日志诊断页面不在此限制内）；
- 微服务；
- 分布式队列或多机并行；
- 外部插件发现；
- 数据库或远程向量库；
- 自动 prompt 优化；
- 内部内容去重流程；
- 独立 Operation Audit 阶段。

## 7. 状态维护规则

每完成一个子计划：

1. 运行该计划规定的测试；
2. 在对应计划文件中记录完成结果；
3. 在本文件勾选对应 TODO；
4. 更新“当前阶段”和“下一项计划”；
5. 如方向发生变化，先更新设计文档，再调整计划。
