# 原始日志文本

按原仓库路径组织，包含全部669个原始文件；只有超过512 KiB的文件被拆分，未省略内容。

- [500题运行](runs/history-learning-general500-subset120/)：原始请求／回复`role_io.jsonl.parts/`、请求计时和usage的`llm_requests.jsonl.parts/`、完整检查点及配置。
- [训练阶段材料120](artifacts/general500-subset120/)：输入、协议、代码快照、中间检查点、统计与人工监控记录。
- [聚类审查121](artifacts/general500-cluster-audit121/)：任务组与经验组、成员正文、审查记录。
- [内容审查122](artifacts/general500-case-audit122/)：114题主审、匿名辅助请求／回复、失败首版、最终复核和方法统计。

`.parts/`中的分段按文件名升序拼接，可精确恢复原文件。JSONL分段包含完整记录，可分别读取；普通JSON的TXT分段可能在对象内部断开。[文件映射与校验清单](../manifest.json)记录每段路径及SHA256。

如需逐条阅读经验或逐题分析，优先使用[经验索引](../experiences/README.md)和[样本索引](../samples/README.md)，无需拼接大型检查点。
