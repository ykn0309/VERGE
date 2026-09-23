# 计划 04：LLM 接口与 Fake Modules

## 状态

```text
状态：待开始
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

待完成后填写。
