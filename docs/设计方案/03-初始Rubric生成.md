# 初始 Rubric 生成

> 上位文档：[VERGE 设计总览](../设计方案.md)
>
> 输入：[经验检索与筛选](./02-经验检索与筛选.md) ｜ 输出：[Rubric Judge](./04-Rubric-Judge.md)

## 1. 模块职责

根据 task 和 selected experiences 生成 initial rubric $R_0$。这是 preference-blind 阶段，不允许查看 chosen/rejected。

## 2. 输入

```text
Task
Selected Experiences（可以为空）
```

禁止输入：

```text
chosen/rejected
preference label
judge results
critic feedback
```

## 3. 输出

Generator 先返回非空 `analysis`，分析 task 的目标、实际输入与完整性、核心要求和当前条件下合理的回答行为；再依据分析返回 `criteria` 列表，每项只包含 `question`、`importance_score` 和 `source_experiences`。系统按列表顺序自动分配 criterion ID，并创建 rubric ID 和空 base；后续 Judge/Critic 使用系统分配的编号。来源经验引用仍使用输入提供的局部数字 ID。分析只依据 task 和所选经验，记录在既有 role_io 中；领域 Rubric 和信用计算不使用分析。标准数量仍为 2–8 条，由覆盖任务核心要求决定，不强行增加数量。

## 4. Importance Score

模型为每条 criterion 输出 1–10 的整数重要性分数：

$$
1\le s_i\le10
$$

不要求这些分数之和等于 1。分数只表示 criterion 对 task 的重要程度，不能表示其预计区分当前 response pair 的能力。

Generator 不计算 normalized weight。归一化由 [Rubric Judge](./04-Rubric-Judge.md) 中的 Calculator 完成。

## 5. Provenance

先根据 task 的显式要求形成标准，再考虑所选经验能否真正补充或改变标准。`source_experiences` 只记录对 criterion 形成有实质作用的经验；来源 ID 合法不等于语义归因正确。

- 仅根据 task 就足以提出该 criterion 的问题，即便某条经验也提出类似原则，仍标 `[]`；
- 经验真正新增或实质改变该 criterion 的判断问题时，才记录对应 experience ID；多来源时逐个检查；
- 一条经验可能因实质帮助另一条标准而被选中，不能因此把它附在本来仅凭 task 就能写出的标准上。已选但未实际使用的经验不得挂名；不得因为经验出现在上下文里就全部引用，也不得为避免误标而把真正有用的来源一律删除。

Generator 不能看到 chosen/rejected。是否实质贡献属于语义判断，确定性校验只保证来源 ID 已选、无重复，不声称能证明该判断正确。

## 6. Criterion 约束

每条 criterion 必须：

- 只判断一个问题；
- 能回答 Yes / No；
- 能直接评价 response；
- 不包含候选回答中特有信息；
- 不引用 chosen/rejected；
- 与其他 criteria 尽量不重复。

## 7. 初始输出校验与有限重试

生成结果须为无 base 的 R0、包含 2–8 条标准；每条 `source_experiences` 不得重复，且只能引用已选经验。普通运行与续跑默认均允许 2 次额外语义纠错，仅反馈不含响应文本的校验原因。Backend 的格式重试耗尽后不叠加语义重试。仍无合法输出时记录 `sample_error`（stage/type/message），不构造 R0/Judge，不计为正确或错误判别，继续下一样本。Generator 不获得 chosen/rejected 或 Judge 结果。

## Generator 会话边界

初始生成只接收 task 和经验；每个样本重置会话。训练初始失败后，原有 Generator 会话按[对话修订](./07-Generator对话修订.md)追加 Critic 公开建议，使用同一角色/后端输出完整 rubric。只读测试和 baseline 不执行修订。模型原始 assistant 回复保留；初始分析的 Web 摘要不被后续修订分析覆盖。
