# 计划 10：DashScope 向量召回

## 状态

```text
状态：已完成（真实 Embedding 连通性 + 125 项离线测试通过）
前置计划：09 — RewardBench Chat Hard
```

## 目标与边界

将 `FakeRetriever` 替换为可配置的真实语义广召回；保留 Fake 实现供离线测试及消融。根据 `docs/设计方案/02-经验检索与筛选.md`，以 task 查询经验 `content + trigger`，按余弦相似度取 Top 20，不将 confidence 混入召回分数；后续 LLM Selector 最多选 5 条。同一个检索器用于 Admission 之后的 Fusion 候选查找，是否 MERGE 仍只由 Fusion Judge 判定。

第一版只使用进程内文档向量缓存和线性扫描，不引入数据库、远程向量服务或额外文件。经验 ADD 后下一次检索应立即可见；MERGE 若文本未变可复用向量。经验池为空时不得请求 Embedding。训练与只读评估共用缓存，但向量缓存不改变 Memory/Beta 状态。

## 模型与配置

- 使用 DashScope `text-embedding-v4` 的 OpenAI-compatible Embeddings 接口，默认 1024 维；复用默认 Qwen Backend 的 `base_url` 与 `api_key_env`，不保存 API key。官方接口：https://help.aliyun.com/en/model-studio/embedding-interfaces-compatible-with-openai
- 配置 `modules.retrieval.implementation: vector`、`config.embedding_model`、`config.dimensions`，沿用 `top_l: 20`；仅接受 DashScope 的 `compatible-mode/v1` base_url，不支持的组合在创建 run 目录之前报错，不静默退回 Fake。限流时按 `Retry-After` 有界退避。
- 请求仅传 task 或经验 `content + trigger`，不传 preference pair、来源轨迹或置信度。仅记录模型、状态码和 token 用量等安全元数据；网络/格式异常抛出明确错误，有限重试。

## 验收

- 离线注入可控向量验证 Top 20 排序、并列稳定、空池、缓存及 ADD/MERGE 后及时刷新、同一实现的 Fusion 检索、训练/评估信息隔离；MockTransport 验证同一环境变量鉴权和输出校验。
- 用现有 `DASHSCOPE_API_KEY_2` 实际调用 `text-embedding-v4`，得到 1024 维向量；示例 query「回答问题并注明引用来源」把「回答问题时给出可靠来源」排在「保持语言简洁」之前，证明 API 与排序链路可用。示例仅用于连通性验证，不是检索效果评测。最终 `python3 -m pytest -q` 为 125 passed；旧 RewardBench 5/6 来自 Fake 检索，不能冒充真实向量检索结果。
- 当前训练后仅有两条经验，全部均会进入 Top 20；真实排序质量须在经验池大于 20 时另行评估，本阶段不宣称效果提升。
