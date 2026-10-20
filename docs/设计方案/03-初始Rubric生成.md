# 初始 Rubric 生成

> **当前实现：计划68信用继承与稳定性惩罚，Manifest/Trajectory10、Memory8。** 演化修订继承历史信用，拆分均分，惩罚为最近5例累计(P+N)的20%；详见[计划68](../plans/68-演化信用继承与稳定性惩罚.md)。其余方法承接[方案14](14-原子经验版本演化与局部混合检索.md)；计划67局部结果保留其原协议解释，见[报告67](../实验分析/67-版本演化与混合检索局部验证.md)。最近完整训练仍为计划66。

> 上位文档：[VERGE 设计总览](../设计方案.md)
>
> 输入：[经验检索与筛选](./02-经验检索与筛选.md) ｜ 输出：[Rubric Judge](./04-Rubric-Judge.md)

## 1. 模块职责

根据 task 和 selected experiences 生成 initial rubric $R_0$。这是 preference-blind 阶段，不允许查看 chosen/rejected。

先区分指令要求执行的任务操作与输入材料讲述的主题，以任务目标与明确约束决定标准及重要性，再按条件吸收经验。材料中描述的行为与回答应执行的行为分别处理；经验不能把任务目标带偏，与任务冲突时以任务为准。条件性指导只在当前条件支撑的范围内落地，维度与权重强度由 task 支撑。

## 2. 输入

```text
Task
Selected Atomic Experience Versions（活跃版本experience_id/content/trigger，可以为空；不含类别或任何信用数值）
```

禁止输入：

```text
chosen/rejected
preference label
judge results
critic feedback
```

已选版本的confidence/evidence_amount不进入Generator，仍保留在Selector输入、选中快照和审计对象中。初始R0没有本轮候选；应用阶段另附经过Atomizer的当前候选集合并标记尚未验证。Generator设计标准依据task及方法，不把历史可靠性数值转成标准权重。

## 3. 输出

Generator 先返回非空 `analysis`，分析 task 的目标、实际输入与完整性、核心要求和当前条件下合理的回答行为；再依据分析返回 `criteria` 列表，每项只包含 `question`、`weight` 和 `source_experiences`。系统按列表顺序自动分配 criterion ID，并创建独立 rubric ID；后续 Judge/Critic 使用系统分配的编号。来源经验引用仍使用输入提供的局部数字 ID。分析只依据 task 和所选经验，记录在既有 role_io 中；领域 Rubric 和信用计算不使用分析。标准数量仍为 2–8 条，由覆盖任务核心要求决定，不强行增加数量。

## 4. 相对权重预算

模型针对整套标准共同分配总预算1，每条`weight`必须是有限正数，允许不同标准之间超过10:1的比例。实际输出之和可以不为1，Calculator统一归一化，原值保留在rubric中。权重仅反映task目标与明确约束的重要程度，不表示预计区分偏好对的能力。

生成时检查标准之间是否相容、是否重复计权，以及是否实际落实所采用经验的设计动作。一条经验可以只影响一项，也可以改变多个问题表达及权重关系；不强制重构全部标准，不增加输入状态或适用维度专门字段。

## 5. Provenance

先根据 task 的显式要求形成标准，再考虑所选经验能否真正补充或改变标准。`source_experiences` 只记录对 criterion 形成有实质作用的经验；来源 ID 合法不等于语义归因正确。

- 仅根据 task 就足以提出该 criterion 的问题，即便某条经验也提出类似原则，仍标 `[]`；
- 经验真正新增或实质改变该 criterion 的判断问题或权重分配时，才记录对应 experience ID；多来源时逐个检查；
- 一条经验可能因实质帮助另一条标准而被选中，不能因此把它附在本来仅凭 task 就能写出的标准上。已选但未实际使用的经验不得挂名；不得因为经验出现在上下文里就全部引用，也不得为避免误标而把真正有用的来源一律删除。

Generator 不能看到 chosen/rejected。是否实质贡献属于语义判断，确定性校验只保证来源 ID 已选、无重复，不声称能证明该判断正确。

## 6. Criterion 约束

每条 criterion 必须：

- 只判断一个问题；
- 能回答Yes/No，并以“是＝满足要求”的正向语义表达，不用“是否存在错误/遗漏”等缺陷问题搭配正向加分；
- 能直接评价 response；
- 不包含候选回答中特有信息；
- 不引用 chosen/rejected；
- 与其他 criteria 尽量不重复。

## 7. 初始输出校验与有限重试

生成结果须为独立 R0、包含 2–8 条标准；每条 `source_experiences` 不得重复，且只能引用已选经验。普通运行与续跑默认均允许 2 次额外语义纠错，仅反馈不含响应文本的校验原因。Backend 的格式重试耗尽后不叠加语义重试。仍无合法输出时记录 `sample_error`（stage/type/message），不构造 R0/Judge，不计为正确或错误判别，继续下一样本。Generator 不获得 chosen/rejected 或 Judge 结果。

## Generator 会话边界

R0只接收task与所选活跃版本，每个样本重置会话，R0来源严格限定E。训练R0有效失败后进入[原子候选集合新会话应用](./07-Generator对话修订.md)：Critic草稿经Atomizer处理，再以task＋固定E＋本轮原子候选集合重新开始Generator会话，不继承R0、旧分析、assistant或普通建议。E的ID、文本和顺序在本题固定；演化子版本只参与未来样本。E和C均由Generator自主判断是否适用，明确约束优先。真实输入输出分别保存，初始分析不能被后续覆盖。只读测试和baseline仅初始生成。

新会话和正向Boolean输出沿用既有协议，Atomizer、多候选集合及不可变版本衔接已实现并通过离线验证。正向问题表达不额外增加语义审核门禁。
