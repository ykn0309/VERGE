# 计划 08：真实 LLM Backend

## 状态

```text
状态：待开始
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

待完成后填写。
