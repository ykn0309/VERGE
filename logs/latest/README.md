# 最新实验：v3.1小样本完整流程124

20题在线训练＋5题冻结推理。代码7b9f119，Qwen3.5-Flash关闭思考，从空经验池开始。这里只验证运行与内容问题，不是泛化效果结论。

- [总体报告124](../../docs/实验分析/124-v3.1小样本真实流程验证.md)
- [逐题轨迹](samples/README.md)：全部25题完整任务、回答、初稿、各路修订、Judge及Critic。
- [经验索引](experiences/README.md)与[全部经验](experience_pool.json)。
- [原始模型请求及回复](raw/runs/v31-smoke124/role_io.jsonl.parts/)：按记录分段，均为UTF-8文本。
- [输入、配置、协议、旁路校验与人工审查](raw/artifacts/v31-smoke124/)。
- [原文件及分段摘要](manifest.json)：顺序拼接parts可恢复原字节，全部校验通过。没有压缩包。

上一轮v3的120／121／122证据已无损移入忽略的logs/archive/2026-11-13-before124，映射见[archive-index.json](archive-index.json)。Git中仍可通过[7b9f119时的旧日志](https://github.com/ykn0309/VERGE/tree/7b9f119/logs/latest)直接读取。数据集输入目录不属于实验日志，不归档。
