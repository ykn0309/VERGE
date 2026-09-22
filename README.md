# VERGE

Verified Experience-guided Rubric Generation.

方法设计见 [`docs/设计方案.md`](docs/设计方案.md)，工程设计见 [`docs/工程设计.md`](docs/工程设计.md)。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 目录

```text
configs/default.yaml   默认实验配置
verge/
  models.py            领域数据结构
  config.py            配置加载
  dataset.py           数据集读取
  llm.py               LLM 调用接口
  prompts.py           Prompt 定义
  calculator.py        确定性计算
  modules.py           方法模块
  memory.py            Memory 与 trajectory 存储
  pipeline.py          流程编排
  cli.py               命令入口
tests/                 测试
docs/                  设计文档
```

当前版本仅包含最小项目骨架与配置。
