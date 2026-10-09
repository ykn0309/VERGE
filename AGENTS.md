# AGENTS.md

本文件定义 AI coding agent 在 VERGE 项目中的工作约束。

## 1. 项目目标

VERGE（Verified Experience-guided Rubric Generation）研究如何从preference trajectory提出rubric-generation experience，先准入，再通过新Generator会话自主应用并观察结果，结合后续任务维护可信度；融合保留文本版本，成功应用只提供弱证据，不解释为因果贡献。新分支目标见计划31，代码尚未迁移；已有经验R0与新候选统一用实际引用标准的无折扣正向P/负向N证据，alpha加P、beta加N，不使用lambda或margin增益；不先抵消，双通道均0才不更新，净方向仅展示，入池资格不因此改变。已确定删除无经验诊断和旧方案兼容，保留既有格式/契约错误的有限纠错与样本/无效轮隔离；不兼容不授权删除历史数据或产物。

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
