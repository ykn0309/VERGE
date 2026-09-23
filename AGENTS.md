# AGENTS.md

本文件定义 AI coding agent 在 VERGE 项目中的工作约束。

## 1. 项目目标

VERGE（Verified Experience-guided Rubric Generation）研究如何从 preference trajectory 中提炼 rubric-generation experience，并根据修订贡献和后续任务表现维护经验可信度。

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
