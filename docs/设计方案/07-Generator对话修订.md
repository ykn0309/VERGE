# Generator 对话修订

> 上位文档：[VERGE 设计总览](../设计方案.md) ｜ 调用方：[失败修订](./06-失败修订.md)

同一 Generator 的单样本会话保留初始 system/user 及模型实际 assistant 原文；追加 Critic 公开建议为新 user，再输出完整 rubric。结构为：

```text
system: Generator 职责与专属输出要求（_ROLE_INSTRUCTIONS）
user: task 和所选经验
assistant: analysis + criteria（初始输出）
user: Critic improvement_advice 与完整修订请求
assistant: analysis + criteria（第一轮完整修订）
user: 第二轮公开建议
assistant: analysis + criteria（第二轮完整修订）
```

不将 chosen/rejected、Judge 结果或 Critic analysis 直接加入会话。初始 Generator 始终 preference-blind，修订获得的建议可能存在提示词未能阻止的间接泄露。

完整输出仍是 `{analysis, criteria}`；每条含 question、importance_score、source_experiences。修订的来源固定空数组，不对训练中修订结果计算经验的下游信用。程序分配新的标准/rubric ID，base 指向最近有效 rubric。没有操作编号、ADD/DELETE/MODIFY 或 Executor。

每个样本开始重置会话，ContextVar 隔离并发测试线程。初始成功、只读测试和 baseline 不追加 Critic 消息。保存每次真实请求中的完整消息及原始输出，可通过 Web Generator 页签审查。

无效但结构可解析的回复仍保留在对话中，后续 user 告知校验原因，有限次数要求重输完整 rubric；不可解析输出的后端格式纠错和实际消息照实落日志，未获得有效回复时不能假装有修订。第二轮以最近有效标准为基线，而不是以无效输出构造 Judge 结果。

参考 RF2：`experiments/critic-refine/run_critic_refine_v5_1.py:798–819` 中 `messages=[*state["active_messages"], feedback]`，模型与角色仍为 Generator。
