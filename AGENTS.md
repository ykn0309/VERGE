# AGENTS.md

本文件定义 AI coding agent 在 VERGE 项目中的工作约束。

## 1. 项目目标

VERGE（Verified Experience-guided Rubric Generation）从偏好轨迹中提炼可复用的评分标准设计经验，指导Generator设计评分标准，并通过实际应用与后续任务反馈维护经验及其可信度。

## 2. 权威文档

修改代码前按任务范围阅读：

1. [`docs/设计方案.md`](docs/设计方案.md)：方法总览与全局原则；
2. [`docs/设计方案/`](docs/设计方案/)：各模块公式、输入输出和边界；
3. [`docs/工程设计.md`](docs/工程设计.md)：工程约束；
4. [`PLAN.md`](PLAN.md)：总体方向、TODO 和当前状态；
5. [`docs/plans/`](docs/plans/)：各阶段的具体执行计划。

“项目目标”和“权威文档”两节保持固定，不随阶段进展更新，以保持提示词前缀稳定、利于缓存命中。当前协议、实验结果和进度统一维护在上述设计文档、PLAN.md及阶段计划中；仅在用户明确要求调整项目目标或文档入口时修改这两节。

## 3. 当前目录约定

保持源码结构扁平：

```text
verge/
├── __init__.py
├── models.py
├── config.py
├── dataset.py
├── llm.py
├── prompts.py
├── calculator.py
├── modules.py
├── memory.py
├── pipeline.py
└── cli.py
```

其他目录：

```text
configs/       实验配置
tests/         测试
docs/          方法、工程文档和阶段计划
```

## 4. 项目约定

- WebUI 启动或重启必须在沙盒外执行（`sandbox_permissions="require_escalated"`），仅监听 `127.0.0.1`，并验证苹果翻译正常。仅修改 JS/CSS 时刷新即可，无需重启。
- 改进模块或提示词时，每个模块固定至少20条按任务去重、有代表性的历史真实输入，覆盖问题、正常及边界情况，独立复测，不跑完整流水线；以Qwen3.5-Flash为基准，可按用户要求比较其他模型，并明确记录思考设置。结果另存且不改原产物，只作为模块局部效果证据。
- 模块复测必须审查实际内容质量，结构有效与语义评价分别报告；未审查内容时不能简称“复测通过”。Critic需检查诊断依据及候选的合理性、原子性和通用性，不能只看字段是否合法。
