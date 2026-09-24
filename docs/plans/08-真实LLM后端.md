# 计划 08：真实 LLM Backend

## 状态

```text
状态：已完成（无凭据环境下以 MockTransport 验证；真实服务待运行验证）
前置计划：07 — CLI
后续计划：V1 held-out evaluation
```

## 目标

增加 OpenAI-compatible backend，在不修改 Pipeline 的情况下用真实模型替换 fake modules。

## 修改范围

- `verge/llm.py`
- `verge/prompts.py`
- `verge/modules.py`
- `configs/default.yaml`
- `requirements.txt`
- 对应测试文件

## 实现内容

- OpenAI-compatible API 调用；
- 按角色选择模型；
- structured output；
- timeout、rate limit 和格式错误重试；
- prompt 版本记录；
- 模型和请求元数据记录；
- API key 环境变量读取。

## 约束

- 不在配置、日志或 trajectory 中保存 API key；
- 格式错误可以有限重试；
- Binding 无结果和 Admission FAIL 不重试；
- LLM 不直接计算最终数值；
- 网络测试和普通单元测试分离。

## 测试

- mocked API success；
- structured output 解析；
- timeout 和 rate limit；
- 格式错误重试上限；
- 缺少 API key；
- fake 与 real backend 可替换性。

## 完成标准

- 不修改 Pipeline 即可切换 backend；
- 一个真实 sample 可以产生完整 trajectory；
- 日志不包含凭据；
- 非网络测试全部通过。

## 完成记录

- `OpenAICompatibleBackend` 使用 Chat Completions JSON 模式；角色共享 `generate(role, payload)` 接口，角色 prompt 独立且记录 `v1` 版本。请求按配置切换模型/后端，HTTP 超时、限流、5xx 和缺字段/非法 JSON 均有有限重试；绑定为空及 Admission FAIL 不重试。
- API key 仅从 `api_key_env` 指定的环境变量读取；拒绝配置明文 `api_key` 与带凭据的 URL。Manifest 记录 backend/model 和 prompt/module 版本；`artifacts/llm_requests.jsonl` 仅保存 sample ID、角色、模型、请求 ID、状态、usage 等白名单元数据。
- `configs/default.yaml` 保留无需网络的 Fake 演示；真实 API 示例见 `configs/openai.example.yaml`。使用前先设 `OPENAI_API_KEY`，再执行 `python -m verge.cli run --config configs/openai.example.yaml`；按服务商修改模型和 HTTPS `base_url`。若服务商不支持 Chat Completions 的 `response_format: json_object`，需适配其协议，不能假定所有兼容接口完全相同。
- 不改 Pipeline：MockTransport 已覆盖完整修订成功 trajectory 以及按角色选用真实 Backend。未在本机使用外部凭据发送真实网络请求；真实服务验收需持有凭据另行执行。现有 `fake` retrieval 仍非向量检索，本阶段只替换 LLM Backend。
