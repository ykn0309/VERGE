# Generator 多轮对话修订

## 决策

参照 RF2 `experiments/critic-refine/run_critic_refine_v5_1.py:798–819`：同一个 Generator 在单样本上下文中保留初始 system/user、实际 assistant 回复，并追加 Critic 的公开 improvement_advice 作为 user 消息。Critic 的私有 analysis 仅供日志和 Reflector，候选回答及 Judge 结果不直接进入 Generator。公开建议必须候选无关，但提示词隔离不能保证不存在语义泄露，不新增内容审核。

Critic 返回 analysis + improvement_advice，不输出操作；Generator 输出完整 rubric，无独立 Refiner 角色。第二轮修订最近一次有效 rubric，历史会话保留，而非从 R0 重做；无效修订有限纠错，不能交给 Judge。未取得有效新 rubric 时以最近有效 rubric 继续，保存错误证据，不伪造分析或判断。Judge 复用最近有效 rubric 中相同问题的判断，既有下游信用仍只由 R0 计算。

Reflector 读取初始和成功标准、两侧判断及成功 Critic 分析与公开建议，生成 0–1 条候选。无原子操作、Executor 运行路径、操作绑定或修订信用。新经验 Beta(1,1)，MERGE 不计信用。Web 展示反馈、完整修订和 Generator 实际多轮消息，不展示操作。

跨样本/训练测试上下文隔离；新提示词和配置名阻止旧检查点跨方法续跑。旧日志保留，不启动真实请求。

## 验收

- `python3 -m pytest -q`：317 passed；JavaScript 语法与 `git diff --check` 通过。
- 模拟真实 HTTP 验证实际 assistant 原文、后端格式纠错消息和同会话追加；Fake 验证两轮最新基线、无效完整输出有限纠错/耗尽、网络错误中止、跨样本和线程上下文隔离。
- Web 浏览器验证 Generator 三次调用、两轮完整结果与 Critic 分析/建议；无控制台错误。初始分析摘要不被修订分析覆盖。
- 仅在 `/tmp/verge-generator-dialogue-smoke` 运行 Fake smoke，未调用真实模型、未覆盖旧运行或重启实验。改动尚未提交。
