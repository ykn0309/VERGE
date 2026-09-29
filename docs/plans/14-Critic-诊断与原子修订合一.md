# 计划 14：Critic 诊断与原子修订合一

> 状态：**实现完成，离线测试通过；真实模型效果未验证**。前一基线为 `01eb456`；本次实现尚未提交，也未用新流程重跑历史实验。与[计划 13](13-Chat-Hard-修订失败诊断与定向改进.md)的诊断工作并行；[计划 12](12-经验适用性与来源归因改进.md)的独立终测仍未完成。
>
> 方法变更前先同步[设计总览](../设计方案.md)、[失败修订](../设计方案/06-失败修订.md)、[原子操作修订](../设计方案/07-原子操作修订.md)、[训练流程](../设计方案/12-训练流程.md)与[工程设计](../工程设计.md)；本文不是这些权威文档的替代。

## 1. 目标与明确决策

当前流程是 `Critic → 自然语言反馈 → Revision Generator → 原子操作 → Executor → Judge`。下一版**直接替换基线**为 `Critic → {analysis, operations} → Executor → Judge`，删除独立的 Revision Generator 模型角色；不要求先做旧/新方案消融，也不因本计划改变 Selector、初始 rubric、Judge、经验记忆和信用计算规则。

- `analysis`：Critic 的私有失败诊断。以给定偏好标签“chosen 应优于 rejected”为目标，允许明确引用两份候选回答的具体内容、逐项 Judge 判断、权重和 margin，说明为什么现有 rubric 没有让 chosen 得分更高。第二轮还要分析第一次修订、Judge 结果及未奏效之处。若初始逐项 Judge 或标签本身可疑，可以在分析中表达疑点；不要凭空捏造 chosen 优势。
- `operations`：Critic **同次调用**直接输出 `ADD` / `DELETE` / `MODIFY` 操作，修订后的标准应是仅依据 task 可提出、能评价未来回答的通用问题，不写入当前 chosen/rejected 特有内容。此处仅通过提示词表达**期望**，不作模型必定遵循的保证。
- **不增加任何针对 operations 是否含候选回答内容的审查**：不增加词面匹配、语义检查、独立模型审核、内容拦截或针对这一性质的自动重试。Executor 仍执行现有 **结构合法性**检查（编号、引用、冲突、条数、权重等）；经验的既有 Admission 不新增专门的候选内容审查步骤。分析与操作可能发生内容串漏，是新基线明确接受的风险，不能把结构合法称作内容安全。
- 两轮上限不变；每轮都对 **R0** 生成可独立执行的完整操作集，第二轮不会自动继承第一轮操作，不能从 R1 接续修订；仅修订成功才反思并分配修订信用。

## 2. 输入输出契约与调用边界

单次 Critic 调用的结构化输出示意：

```json
{
  "analysis": "解释当前 R0 或上轮修订为什么没把 chosen 排在 rejected 前面；可引用候选回答",
  "operations": [
    {
      "operation_id": "1",
      "operation_type": "MODIFY",
      "target_criterion_id": "2",
      "new_or_updated_criterion": {
        "criterion_id": "2",
        "question": "仅从任务要求推导的、适用于未来回答的可判定问题",
        "importance_score": 8,
        "source_experiences": []
      }
    }
  ]
}
```

这里的编号是当前模型请求中的临时数字 ID；内部 ID 按现有映射严格还原。`before_criterion` 仍由 Executor 从 R0 补全，LLM 不计算归一化权重、credit 或成功标签。`analysis` 是可审计的私有文本，**不是**可迁移经验，也不发送给未来 Selector / 初始生成器；不要在 Reflector 中把它直接当作已验证原则。

| 轮次 | Critic 可见输入 | 输出与执行 |
| --- | --- | --- |
| 第一次 | task、chosen/rejected、R0、R0 逐项 Judge 判断与 margin | 同一响应输出 `analysis + operations`；Executor 在 R0 执行成 R1，再由 Judge 独立评价。 |
| 第二次（有效 R1 仍失败） | 上述输入，以及第一次的 analysis、已执行 operations、R1、逐项 Judge 结果与校验记录 | 先对照第一次失败结果解释不足，再给出对 R0 一次性执行的完整 O2；不复用 R1 的新增标准 ID。 |
| 第二次（第一次仅有结构无效操作） | 上述基础输入，以及第一次 analysis 和结构校验失败摘要；**没有**虚构的 R1 或 Judge 判断 | 仍以 R0 为起点；不传未保存/未执行的无效操作。 |

提示词要求私有 `analysis` 可以讨论 chosen/rejected 的细节，但 `operations` 中的标准只能由 task 的要求支撑；不得把“chosen 答案包含 X”转写成通用标准“必须包含 X”（除非 task 自身明确要求 X）。对可疑 Judge/偏好证据，可在 `analysis` 中标明不确定性；它不触发新增操作内容审查，也不改变给定偏好标签这一修订目标。输出分析要具体关联标准及 Judge 结果，不能只有泛泛的“权重不合理”；操作可能为空时仍需明确由现有 Executor/流程如何判为无效修订，不得产生虚假的 Rk。

## 3. 代码与数据迁移步骤

1. **先改权威方法/工程文档**：从“Critic 仅给反馈＋Revision Generator 产出操作”改为“Critic 单次诊断并产出操作”；明确 analysis 的私有性质、候选信息的提示词边界、零内容审查、两次 R0 修订和原有 Executor/Judge/Admission 责任；审阅旧示例和角色列表是否仍描述已删除角色。
2. **领域模型与日志**：定义能同时承载非空 `analysis`、`operations` 的 Critic 返回结构。有效 `RevisionAttempt`、无效 `InvalidRevisionAttempt` 都要保存对应的分析；第二轮能取到首次 analysis 与已执行修订或无效原因。旧 `feedback` 字段与既有 JSONL/UI/inspect-sample 的读取需明确兼容策略，不能静默把旧记录当作新 `analysis`，也不能改写旧运行产物。新增本地 role_io 要如实记录新的 Critic 输入输出，保护 chosen/rejected，且不会为旧日志补造数据。
3. **接口与流水线**：更新 `verge/modules.py` 的 Critic 协议及 LLM 实现，移除 RevisionGenerator 协议/实现与角色注册；更新 `verge/pipeline.py` 的两轮调度，使 Critic 的 `operations` 直接进入原有 Executor、Judge、贡献计算与反思流程。第一次无效时仍可第二轮；只对原有操作结构错误有限重试，第二次能看到第一次实际保存的失败摘要。其它 LLM 后端/格式错误继续遵循既有失败策略，不用伪造可执行操作。角色专属配置、假后端及默认配置须同步检查，不让过期 `revision_generator` 角色配置被静默忽略。
4. **提示词与数字 ID**：修改 `verge/prompts.py` 的 Critic system/user 模板、Critic 输出解析和数字编号反解，覆盖 `operations` 嵌套的操作 ID、标准 ID、新增 ID，以及第二轮历史中的操作和 rubric 引用；去掉不再使用的 Revision Generator 提示词/版本。候选响应只在训练中的 Judge/Critic 可见，Selector 与初始生成器仍 task-only。提示词版本更新；旧运行配置、manifest、resume 指纹要保持来源可辨，新流程不应在不相容的旧检查点上续跑。
5. **只读诊断视图与文档**：轨迹/UI 清晰区分“私有分析”和“执行的操作／修订结果”，旧记录仍可读；更新 README、计划索引与实验说明中的阶段图、调用次数和版本说明。历史 v5/v6/v8 结果不回填为新方案结果。

## 4. 测试与验收

- 单元测试：初始失败时 Critic 一次返回 `analysis + operations`；Executor 完整执行 ADD/DELETE/MODIFY 后补全旧标准；Judge 仅对有效 Rk 评估。数字 ID 的嵌套映射和第二轮独立编号空间正确，错漏引用按原有结构错误路径处理。
- 两次修订测试：第一次有效但失败时，第二次收到真实的 analysis/operations/R1/Judge 并仍从 R0 重提完整操作；第一次仅操作无效时，第二次只收到 analysis/校验摘要、不看到虚构结果；第二次有效成功仅由该次已执行操作触发 Reflection 与信用。
- 错误与兼容：Critic 结构不合法、Executor 校验耗尽、Judge 编号错漏重试耗尽、后端故障的处理保持可区分；旧轨迹/Web/CLI 可读，旧检查点不得越版本误恢复；日志不泄露凭据或为未发生的角色调用捏造记录。
- **不写针对候选内容的操作审查测试**；只检查提示词确实表达 `analysis` 可引用候选、`operations` 应 task-grounded 的要求，不把“测试通过”描述为模型实际不会泄漏候选内容。
- `python3 -m pytest -q` 全部通过。必要时以独立新目录运行少量真实模型 smoke，核对发送给 Critic 的真实输入、操作执行、第二轮历史和落盘轨迹；再决定是否开启完整训练/测试。真实请求费用与调用次数应先估算，不能自动覆盖旧运行。

## 5. 效果记录与边界

这是**直接替换基线**，不设置“必须先做 A/B 消融”门槛。新运行应记录初始失败数、第一次/第二次修复率、无效操作率、Judge 错误、成本与最终测试指标；对疑似 Judge/标签问题保留可复核证据。与旧 v8 等运行的数值只能作描述性参照：初始 rubric、模型随机输出、经验池和 Judge 表现可能变化，不能把差值归因于删除 Revision Generator。

实现验收：`python3 -m pytest -q` **190 passed**（Fake 全流程、模拟真实 HTTP、数字 ID、两次修订、Critic 错误编号/结构重试、重复问题判断复用、Admission 数字 ID、旧轨迹和跨提示词版本续跑拒绝）；`git diff --check` 无格式错误。修订结构错误默认额外重试 2 次；未发起任何真实模型请求；历史实验配置及运行产物未改写，需另开目录做 smoke 和效果评估。

已反复分析的 Chat Hard 60 条测试不是未污染终测；计划 12 的独立 held-out 任务仍须另行评估。Critic 的 analysis 和操作同次生成虽然减少一次模型调用，却可能使私有候选信息渗入通用标准；本计划依用户决定**不增加内容审查**，后续若要研究泛化风险应另提研究任务，不能暗中加回内容门禁。
