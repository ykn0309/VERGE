# 42 — 删除Admission，直接应用候选经验

## 1. 授权与目标

用户要求：“我想删除 admission 这个环节。”本次实现删除环节并同步协议、界面、配置、测试和活跃文档，不启动新的真实实验，不提交或push。所有真实LLM角色继续固定Qwen3.5-Flash、关闭思考；历史配置、报告、运行产物保留。

## 2. 新流程

```text
R0有效失败
→ 每轮一次Critic提案（0–1条候选）
→ 独立Generator(task+原始固定E+当前C)
→ Judge
→ 成功且声明采用C：Fusion
→ 成功未采用C：结束，不入池
→ 失败：有剩余轮次则重新提出下一条候选
最多两轮，最多两次业务提案；空候选正常结束。
```

删除AdmissionChecker/LLMAdmissionChecker、角色prompt/schema/后端构造、decision/reason、拒绝候选反馈、同轮重生成及admission_exhausted。技术纠错仍有限且单独计数，输出耗尽消耗当前轮，不伪造Judge或增加机会。Critic的ID新颖性和未融合状态仍作结构契约检查，不新增语义门禁。

Generator保持全新会话、固定E和当前C来源白名单、自主采用及任务优先。候选实际消息改为“当前候选、尚未验证，无历史置信度”，不再宣称准入通过。第二轮只替换C，不累积旧候选或筛选E。

成功且采用同一候选才Fusion；P/N双通道、R0信用一次性写入、负/零信用合法Fusion、ADD原文/MERGE文本版本、held-out冻结和全局故障边界保持。

## 3. 协议与历史边界

| 项目 | 当前值 |
|---|---|
| application实现 | `critic_candidate_application` |
| Prompt标识 | `critic-candidate-direct-fresh-application` |
| Manifest format_version | 3 |
| Trajectory schema_version | 3 |
| Memory schema_version | 2（字段未变） |
| ProposalAttempt | round_number、analysis、candidate、error |
| 候选字段 | 删除admission/admission_reason；保留Fusion关联字段 |
| 活跃LLM角色 | Selector、Generator、Judge、Critic、Fusion |

Web删除准入节点、PASS/FAIL徽标、拒绝原因和统计，展示直接应用候选。当前CLI/Web拒绝旧manifest和轨迹；不迁移旧运行，不改写历史配置或产物。

更新默认Fake/OpenAI/Qwen示例配置，准备新的 [Flash30条配置](../../configs/rewardbench-chat-hard-qwen-direct-experience-flash-30.yaml)：roles为空继承Flash非思考、timeout30/max_retries2，新目录 `runs/rewardbench-chat-hard-qwen-direct-experience-flash-30`。**尚未运行**。

## 4. 离线验证

`python3 -m pytest -q`：**209 passed**。移除只适用于Admission调用与拒绝重生成的测试，保留历史语义观察fixture和报告。新增/同步验证直接两轮HTTP、没有准入调用、统一经验展示/候选证据状态和旧轨迹版本拒绝。

覆盖两次提案对应两次新Generator、第二轮真实失败证据、固定E/替换C、无旧历史、严格数字ID来源、未采用成功不入池、双通道信用、负/零信用Fusion、错误隔离/fail-fast、冻结、续跑及实际前端函数渲染。语法、差异空白和文档链接检查通过。

## 5. 解释边界

删除Admission消除了前置语义拒绝，但尚无新的真实效果数据。候选会直接得到应用，可能增加试用机会及模型请求；不能据此认定泛化、忠实采用或任务成功改善。Critic抽象、Generator落实经验、Judge事实/方向错误仍需后续实证。历史同任务结果不作为当前流程有效性结论。
