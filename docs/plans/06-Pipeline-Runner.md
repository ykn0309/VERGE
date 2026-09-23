# 计划 06：Pipeline Runner

## 状态

```text
状态：待开始
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

待完成后填写。
