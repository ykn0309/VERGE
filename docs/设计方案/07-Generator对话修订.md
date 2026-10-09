# Generator 对话修订

> 上位文档：[VERGE 设计总览](../设计方案.md) ｜ 调用方：[失败修订](./06-失败修订.md)

同一 Generator 的单样本会话保留初始 system/user 及模型实际 assistant 原文；追加 Critic 公开建议为新 user，再输出完整 rubric。结构为：

```text
system: Generator 职责与专属输出要求（_ROLE_INSTRUCTIONS）
user: task 和所选经验
assistant: analysis + criteria（初始输出）
user: Judge失败状态 + Critic improvement_advice + 重新生成请求
assistant: analysis + criteria（第一轮完整修订）
user: 最近有效rubric的Judge失败状态 + 第二轮公开建议
assistant: analysis + criteria（第二轮完整修订）
```

仅告知Judge失败状态，不将 chosen/rejected、Judge逐项判断/得分或 Critic analysis 直接加入会话。初始 Generator 始终 preference-blind，修订获得的建议可能存在提示词未能阻止的间接泄露。

Generator结合task判断公开建议是否适用，任务目标与明确约束优先于经验和修订建议；保留合理标准，仅调整有依据的问题，并在原analysis中简述关键调整依据。不为提高判别分差降低明确约束的重要性或增加无依据要求。反馈仍为简短user消息，职责写入角色system，不恢复操作协议或新增审核阶段。

完整输出仍是 `{analysis, criteria}`；每条含 question、importance_score、source_experiences。修订来源由程序接管：基本结构/类型校验后，在编号反解前于返回值副本中统一清空，不要求模型清空、不向Generator说明内部处理规则。R0仍严格核验所选经验引用，其他字段及角色的校验不放宽。不对训练中修订结果计算已有经验下游信用。程序分配新的标准/rubric ID，base 指向最近有效 rubric。没有操作编号、ADD/DELETE/MODIFY 或 Executor。

每个样本开始重置会话，ContextVar 隔离并发测试线程。初始成功、只读测试和 baseline 不追加 Critic 消息。保存每次真实请求中的完整消息及原始输出，可通过 Web Generator 页签审查。

无效但结构可解析的回复仍保留在对话中，后续 user 告知校验原因，有限次数要求重输完整 rubric；不可解析输出的后端格式纠错和实际消息照实落日志，未获得有效回复时不能假装有修订。第二轮以最近有效标准为基线，而不是以无效输出构造 Judge 结果。

参考 RF2：`experiments/critic-refine/run_critic_refine_v5_1.py:798–819` 中 `messages=[*state["active_messages"], feedback]`，模型与角色仍为 Generator。

正常修订的user消息只说明“你生成的 rubric 未通过 Judge 评判”，附Critic公开建议并请求重新生成，不重复analysis/criteria、数量、重要性、来源或ID格式。输出要求沿用初始system；修订来源清空只在程序内执行，system/user均不声明该内部规则。原始回复和backend解析输出原样留在role_io，assistant历史保留原文，轨迹Rubric保存程序规范化结果；不修改旧日志。输出校验纠错仍只说明实际校验问题，不把未经过Judge的无效回复描述成Judge失败。
