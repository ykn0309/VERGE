# 计划 06：Pipeline Runner

## 状态

```text
状态：已完成
前置计划：05 — Experience Memory 与 Trajectory
后续计划：07 — CLI
```

## 目标

在 `verge/pipeline.py` 中串联 VERGE 全流程，并通过 fake modules 覆盖所有主要分支。

## 修改范围

- `verge/pipeline.py`
- `tests/test_pipeline.py`

## 执行顺序

```text
Retrieval
→ Selection
→ Initial Rubric
→ Judge
→ Existing Experience Credit → 立即更新已有经验
→ 必要时 Revision Attempt 1 / 2
→ Reflection
→ Operation Contribution 与 Binding
→ Provisional Revision Credit
→ Admission
→ Fusion → 写入 ADD / MERGE
→ 保存 Trajectory
```

## 必须覆盖的分支

1. Initial success；
2. Attempt 1 success；
3. Attempt 2 success；
4. Final failure；
5. Reflection 返回 0 个 candidates；
6. Candidate 无法绑定 operation；
7. Admission FAIL；
8. Fusion MERGE；
9. Fusion ADD。

## 约束

- `$R_1$` 和 `$R_2$` 都以 `$R_0$` 为 base；
- failed revision 不生成经验；
- Admission FAIL 不改变其他 candidate credit；
- Runner 在初始信用计算后和 Fusion 时分别更新 Memory；后续阶段失败不撤销此前的下游信用；
- Pipeline 不依赖具体 LLM 提供方。

## 完成标准

- 使用 fake modules 跑通全部分支；
- 每条 trajectory 包含完整阶段结果；每次修订尝试只保存一份补全后的操作列表；
- 测试全部通过。

## 完成记录

- `verge/pipeline.py` 实现 `RevisionExecutor` 与 `PipelineRunner`：R1/R2 均基于 R0 执行，补齐原标准，只允许真正改变问题或 importance score 的 MODIFY；修订后 rubric 不携带旧经验来源。
- 初始 Judge 后立即写入下游信用；修订成功后依次反思、绑定、计算候选信用、准入与 ADD/MERGE，并逐条立即写入 Memory。失败的修订不反思；评估模式只执行初始生成和 Judge，不写 Memory。
- 对相同问题（含换 criterion ID）检查 Boolean judgment 复用，防止重新判断产生虚假修订收益。
- `tests/test_pipeline.py` 覆盖全部九条指定分支、空操作后继续第二次修订、跨样本即时更新、Admission FAIL 后不重新分配信用、单条候选顺序 Fusion、无 preference 泄漏和只读评估。
- `python3 -m pytest -q`：94 passed；`python3 -m compileall -q verge` 与 `git diff --check` 通过。
