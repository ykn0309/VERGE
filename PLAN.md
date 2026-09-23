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
当前阶段：基础工程准备完成，尚未开始功能实现
下一个计划：01 — 领域模型
当前阻塞：无
```

已完成：

- [x] 方法设计文档
- [x] 工程设计文档
- [x] 最小 Python 项目骨架
- [x] 项目级 Agent 约束
- [x] 分阶段实现计划拆分

## 3. V1 实现计划

按顺序执行以下计划：

- [ ] [01 — 领域模型](docs/plans/01-领域模型.md)
- [ ] [02 — 确定性计算](docs/plans/02-确定性计算.md)
- [ ] [03 — 配置与数据输入](docs/plans/03-配置与数据输入.md)
- [ ] [04 — LLM 接口与 Fake Modules](docs/plans/04-LLM接口与Fake模块.md)
- [ ] [05 — Experience Memory 与 Trajectory](docs/plans/05-经验记忆与轨迹.md)
- [ ] [06 — Pipeline Runner](docs/plans/06-Pipeline-Runner.md)
- [ ] [07 — CLI](docs/plans/07-CLI.md)
- [ ] [08 — 真实 LLM Backend](docs/plans/08-真实LLM后端.md)

### 下一项计划

> [01 — 领域模型](docs/plans/01-领域模型.md)

完成后应更新本文件：勾选计划、修改“当前阶段”，并明确下一项计划。

## 4. V1 目标

V1 应实现：

```text
读取 preference samples
→ 生成并判断 initial rubric
→ initial failure 时最多执行两次同基线修订
→ successful revision 后生成候选经验
→ 分配 revision credit 并执行 Admission
→ MERGE / ADD 到 Experience Memory
→ 保存可恢复的完整 trajectory
```

总体要求：

- 单进程运行；
- 可测试、可复算、可恢复；
- 使用 fake modules 完成离线端到端测试；
- 可替换为真实 LLM backend；
- sample-level memory update 保持原子性和幂等性。

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

- Web UI；
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
