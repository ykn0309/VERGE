# 计划 07：CLI

## 状态

```text
状态：待开始
前置计划：06 — Pipeline Runner
后续计划：08 — 真实 LLM Backend
```

## 目标

提供运行、恢复和检查 VERGE 实验的最小命令行入口。

## 修改范围

- `verge/cli.py`
- `tests/test_cli.py`

## 命令

```text
python -m verge.cli run --config configs/default.yaml
python -m verge.cli resume --run-dir <path>
python -m verge.cli inspect-sample --run-dir <path> --sample-id <id>
```

## 实现内容

- 参数解析；
- 配置加载；
- Pipeline 启动；
- 未完成运行恢复；
- sample trajectory 查看；
- 明确退出码和错误消息。

## 测试

- 正常 run；
- 非法配置；
- resume；
- 不存在的 sample；
- inspect 输出主要阶段；
- 退出码。

## 完成标准

- 三个命令均可执行；
- 错误包含 sample ID 和失败阶段；
- `inspect-sample` 显示 rubric、judgment、operations、credits、Admission 和 Fusion；
- 测试全部通过。

## 完成记录

待完成后填写。
