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
当前阶段：11 — Chat Hard 144/60 同源内部试验已完成（冻结测试 43/60，71.67%，非官方成绩）
下一个方向：独立训练数据与完整 held-out evaluation
当前阻塞：待选择独立数据集并规划跨源评估；同源内部试验不代表泛化能力
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

### 下一项方向

> Chat Hard 同源内部 144 训练 + 60 冻结测试已完成（初始判别 43/60，非官方成绩）；真实向量召回已实现。下一步规划独立训练数据及跨源 held-out evaluation。

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

- 单进程运行；
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
