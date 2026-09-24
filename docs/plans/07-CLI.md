# 计划 07：CLI

## 状态

```text
状态：已完成
前置计划：06 — Pipeline Runner
后续计划：08 — 真实 LLM Backend
```

## 目标

提供运行和检查 VERGE 实验的最小命令行入口。

## 修改范围

- `verge/cli.py`
- `verge/pipeline.py`（失败阶段标记）
- `verge/dataset.py`（可识别时报告损坏行的 sample ID）
- `configs/default.yaml` 与 `configs/data/`（可运行的离线 Fake 示例）
- `tests/test_cli.py`

## 命令

```text
python -m verge.cli run --config configs/default.yaml
python -m verge.cli inspect-sample --run-dir <path> --sample-id <id>
```

## 实现内容

- 参数解析；
- 配置加载；
- Pipeline 启动；
- sample trajectory 查看；
- 明确退出码和错误消息。

## 测试

- 正常 run；
- 非法配置；
- 不存在的 sample；
- inspect 输出主要阶段；
- 退出码。

## 完成标准

- 两个命令均可执行；
- 错误包含 sample ID 和失败阶段；
- `inspect-sample` 显示 rubric、judgment、operations、credits、Admission 和 Fusion；
- 测试全部通过。

## 完成记录

实现 `python -m verge.cli run --config configs/default.yaml` 与 `inspect-sample`：配置和组件装配校验（本阶段仅支持 Fake backend、Fake 检索和本地存储，未实现的组合明确报错），训练后用只读 Memory 运行可选 test，严格拒绝本阶段未支持的配置字段，保存包含有效配置、Git 提交与 Python 源码 SHA-256 指纹的 `run_manifest.json`，从持久化 trajectory 输出完整样本记录。非法配置退出码 2，样本运行失败退出码 1 并报告 sample ID 与阶段；已有输出目录不覆盖；train/test 的 sample ID 全局唯一，避免按 ID 检查时产生歧义。`configs/default.yaml` 从阶段 03 的骨架 placeholder 更新为含独立 train/test 合成样本的可运行 Fake 示例。此 CLI 没有中途失败回滚/崩溃恢复能力，遵循 V1 工程约束。`python3 -m pytest -q`：104 passed；`python3 -m compileall -q verge` 与 `git diff --check` 通过。
