# 计划 04：LLM 接口与 Fake Modules

## 状态

```text
状态：已完成
前置计划：03 — 配置与数据输入
后续计划：05 — Experience Memory 与 Trajectory
```

## 目标

定义最小 LLM 和方法模块接口，并用 deterministic fake modules 支持无网络测试。

## 修改范围

- `verge/llm.py`
- `verge/prompts.py`
- `verge/modules.py`
- `tests/test_modules.py`

## 实现内容

统一 LLM 接口：

```text
generate(role, payload) -> structured result
```

定义模块：

- Retriever
- Selector
- Initial Rubric Generator
- Judge
- Critic
- Revision Generator
- Reflector
- Revision Credit Assigner
- Admission Checker
- Fusion Judge

为各模块提供 deterministic fake 实现。

## 约束

- preference-blind 阶段不得接收 chosen/rejected；
- LLM 不直接计算 normalized weight 或最终 credit；
- Fusion Judge 只比较 `content + trigger`；
- fake module 输出必须通过领域模型校验；
- 暂不实现自动插件发现和复杂 Registry。

## 测试

- initial success 固定场景；
- revision success 固定场景；
- final failure 固定场景；
- preference 数据隔离；
- structured result 类型检查。

## 完成标准

- 全部模块可以在无网络环境运行；
- fake modules 能驱动后续 Pipeline 测试；
- 接口足以替换真实 LLM backend；
- 测试全部通过。

## 完成记录

`LLMBackend.generate(role, payload)` 定义统一 structured-output 边界；`FakeLLMBackend` 固定支持 initial success、attempt 1/2 success 和 final failure，并可配置空反思、无绑定或 Admission FAIL。`verge/modules.py` 提供十种模块 Protocol、离线 FakeRetriever 和九个复用同一 Backend 的角色模块；Judge 的 Boolean 结果由 Backend 给出，权重/margin 与 revision credit 只由 Calculator 计算。`ExperienceView` 隔离经验来源和证据；preference-blind 角色只接受 task 与经验投影；所有修订角色的 payload 不包含 criterion 来源，Revision Generator 的来源输出会被忽略；Fusion 只接收 content/trigger 文本（匹配位置由调用方映射成 ID）。Executor 和 Runner 留待计划 06，测试中仅对固定 MODIFY 使用轻量 fixture。`tests/test_modules.py` 覆盖分支、隔离、malformed structured output、修订尝试 ID 一致性及 Admission credit 边界。`python3 -m pytest -q`：67 passed。
