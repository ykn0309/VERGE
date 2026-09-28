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

Generator 输出 2–8 条原子 Boolean criteria：

```text
Criterion {
    criterion_id
    question
    importance_score
    source_experiences
}
```

示例：

```json
{
  "criterion_id": "c_1",
  "question": "回答是否分别满足任务中的每个显式要求？",
  "importance_score": 8,
  "source_experiences": ["exp_001"]
}
```

## 4. Importance Score

模型为每条 criterion 输出 1–10 的整数重要性分数：

$$
1\le s_i\le10
$$

不要求这些分数之和等于 1。分数只表示 criterion 对 task 的重要程度，不能表示其预计区分当前 response pair 的能力。

Generator 不计算 normalized weight。归一化由 [Rubric Judge](./04-Rubric-Judge.md) 中的 Calculator 完成。

## 5. Provenance

`source_experiences` 只记录对 criterion 形成有实质作用的经验。

- task 自身足以产生的 criterion：`[]`；
- 经验确实改变或补充 criterion：记录相应 experience ID；
- 不允许因为经验出现在上下文里就全部引用。

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
