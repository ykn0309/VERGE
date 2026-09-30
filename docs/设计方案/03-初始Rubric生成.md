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

Generator 返回 `criteria` 列表，每项只包含 `question`、`importance_score` 和 `source_experiences`。系统按列表顺序自动分配 criterion ID，并创建 rubric ID 和空 base；后续 Judge/Critic 使用系统分配的编号。来源经验引用仍使用输入提供的局部数字 ID。

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

生成结果须为无 base 的 R0、包含 2–8 条标准；每条 `source_experiences` 不得重复，且只能引用已选经验。输出未通过这些校验时，可有限次数重新生成完整 R0，仅反馈不含响应文本的校验原因，不把 chosen/rejected、Judge 结果或原始无效输出传给 Generator；重试仍失败则停止当前运行，不将无效 rubric 交给 Judge。实验 11 的严格检查点续跑默认允许 2 次额外尝试，记录在续跑 manifest 中；普通 `run` 的默认行为不变。
