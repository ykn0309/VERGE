# AGENTS.md

本文件定义 AI coding agent 在 VERGE 项目中的工作约束。

## 1. 项目目标

VERGE（Verified Experience-guided Rubric Generation）研究如何从 preference trajectory 中提炼 rubric-generation experience，并根据修订贡献和后续任务表现维护经验可信度。

当前目标是实现一个简单、单进程、可测试、可恢复的 Python V1。

## 2. 权威文档

修改代码前按任务范围阅读：

1. [`docs/设计方案.md`](docs/设计方案.md)：方法总览与全局原则；
2. [`docs/设计方案/`](docs/设计方案/)：各模块公式、输入输出和边界；
3. [`docs/工程设计.md`](docs/工程设计.md)：工程约束；
4. [`PLAN.md`](PLAN.md)：总体方向、TODO 和当前状态；
5. [`docs/plans/`](docs/plans/)：各阶段的具体执行计划。

发生冲突时，优先级为：

```text
用户当前要求
> docs/设计方案及其模块文档
> docs/工程设计.md
> PLAN.md
> 现有代码
```

不要仅根据 `PLAN.md` 推断方法公式。

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

除非现有文件已经明显过大且用户同意，否则不要创建多层 package、插件系统或额外框架。

其他目录：

```text
configs/       实验配置
tests/         测试
docs/          方法、工程文档和阶段计划
```
