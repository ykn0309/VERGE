# AGENTS.md

本文件定义 AI coding agent 在 VERGE 项目中的工作约束。

## 1. 项目目标

VERGE（Verified Experience-guided Rubric Generation）从preference trajectory提出rubric-generation experience，直接交给新Generator会话自主应用，再用实际结果及后续任务维护可信度。当前协议已删除Admission，见计划42；计划44将标准改为相对权重预算，Judge输出逐项Boolean及依据供Critic核对：每轮一次Critic提案、最多两轮，只有声明采用且有效应用成功的候选才可Fusion。已有经验R0与候选统一按来源标准累计无折扣正向P/负向N，alpha加P、beta加N，不先抵消、不使用lambda或margin增益；双通道均0才不增加数值证据。融合保留文本版本，应用证据不解释为因果贡献。真实实验固定所有LLM角色Qwen3.5-Flash、关闭思考；历史配置与产物保留。保持有限纠错与样本/无效轮隔离，基础设施和程序错误fail-fast，不做旧协议兼容或跨方法resume。

## 2. 权威文档

修改代码前按任务范围阅读：

1. [`docs/设计方案.md`](docs/设计方案.md)：方法总览与全局原则；
2. [`docs/设计方案/`](docs/设计方案/)：各模块公式、输入输出和边界；
3. [`docs/工程设计.md`](docs/工程设计.md)：工程约束；
4. [`PLAN.md`](PLAN.md)：总体方向、TODO 和当前状态；
5. [`docs/plans/`](docs/plans/)：各阶段的具体执行计划。

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
