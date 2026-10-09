# 100样本Critic校准版本复测

## 授权与设置

计划28实现后，用户授权“再跑100条样本，看看是否还有这些问题”。不要求提交/push，不修改模型、数据、Judge或额外提示词；从新目录开始，不resume旧运行。

- 配置：`configs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100.yaml`。
- 新目录：`runs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100`。
- 对照：`runs/rewardbench-chat-hard-qwen-revision-only-100`（计划27）。
- 配置语义与对照仅output_dir不同：相同有序前100条、seed42、qwen3.5-flash、thinking=false、经验链关闭、两轮修订上限、超时与重试设置，无test split。temperature及模型采样seed仍不发送，不同时增加采样参数这一变量。
- 提示词版本：`generator-dialogue-evidence-grounded-critic`。源码为HEAD a4d9d11加计划28未提交修改，启动前保存实际代码指纹，运行中不改源码/提示词。
- 启动前完整离线测试 **473 passed**；新输出目录不存在、数据及凭据存在、无其他CLI实验进程。旧运行manifest/artifacts哈希保存，完成后检查未改变。

## 复核目标

先核对100条完整前缀、实际三角色system、经验/embedding零调用、空引用/候选/池、有效与无效修订、sample_error/judge_error及请求重试。统计初始成功、R1/R2救回与最终成功，复算judgment。

语义复核覆盖所有有效Critic反馈及全部最终失败，不只抽查十个旧失败：

1. 对实际前后标准的误读或虚构修改历史。
2. 新增无任务依据要求、继续提高门槛、降低明确约束优先级。
3. 建议部分分、一票否决、不适用项自动排除等程序不支持机制。
4. 公开建议泄漏候选特有内容、偏好身份、评分或比较评价。
5. Generator是否执行有依据建议、保留任务合理标准，成功是否依赖额外限制。

旧问题样本若本轮初始成功，没有触发Critic，必须标为“未复测修订行为”，不算缺陷已解决。比较原始数量及实际Critic调用/修订样本分母；报告输入组织变更的确定性结果与模型语义行为分开，不把字面扫描当完整审计。

本次是同序开发诊断样本的一次重复，非独立held-out或单因素对照（Generator与Critic提示词及Critic输入共同变化）；默认采样及初始失败子集变化影响可比性。成功率与问题减少只作为观察，不证明因果/泛化改善。

## 执行状态

已启动：`python3 -m verge.cli run --config configs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100.yaml`。后台任务 `b4f8612a2`，PID90693；自动完成通知开启，不为等待轮询。

启动源码指纹：`a4d9d1180d5e2d10fcc004230025df602d5b9e7a+sha256:ad6d14431660fcd67dcd27d50bf371ce3ada35bdf4182cdcad30567a394af684`。启动审计保存在 `/tmp/verge_evidence_grounded_100_launch_audit.json`，含实际数据SHA256及三个历史运行的文件哈希，不含凭据。

运行至100条正常暂停；完成后补充实际结果与逐例分析，不自动继续全量、不新增门禁或下一轮付费实验。

### 超时中断与授权续跑

原任务b4f8612a2在完成98条后，第99条 `rewardbench:llmbar-adver-neighbor:2761` 的Critic连续三次ReadTimeout中止（exit1）。未提交该样本轨迹；已完成前缀为初始成功65、R1救回19、R2救回5、最终失败9。这里只记录进度，不作为100条结果。

用户明确要求“继续”。只读核验源码指纹/提示词/数据/config保持一致、98条为精确有序前缀、经验池为空、请求及role_io的未完成样本只有第99条；现有 `_resume_checkpoint` 通过。原目录完整保留，不修补或删除日志。

- 续跑配置：`configs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100-resumed.yaml`，仅output_dir不同。
- 新目录：`runs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100-resumed`。
- 命令：`python3 -m verge.cli run --config configs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100-resumed.yaml --resume-from runs/rewardbench-chat-hard-qwen-revision-only-evidence-grounded-100`。
- 任务bb9aa3911，PID95416，自动完成通知开启。
- 原98条直接复制并跳过；第99条从初始生成重做（不是仅重试Critic），第100条继续原数据顺序，绝对限制仍100，test=0。旧超时及未提交请求保留在继承日志中，后续调用统计须区分重做同一样本的请求，不将其当作额外样本或有效修订。
- 父目录manifest/artifacts哈希与lineage核验记录保存于 `/tmp/verge_evidence_grounded_100_resume_audit.json`，完成后核对复制的98条与父目录不变。

不更改超时、模型、数据、源码、评分机制；本次续跑仍验证计划28，不实现用户新提出的“Critic先生成候选经验、新会话应用”方法。

### 完成与复核

bb9aa3911正常完成（exit0）：最终目录100/456、test=0，原98条前缀逐字节一致、父目录及历史运行哈希未变、源码/system一致、经验零调用。

初始成功67，33条失败救回24（R1=19、R2=5），最终91/100；47轮全部有效，无无效修订/sample_error/judge_error。与旧轮90相比仅+1，共同24条初始失败两轮都救回15条。公开身份泄漏35/47、评分机制越界及无依据要求未解决，并发现负向问题奖励缺陷的假成功。

详细统计、全部47条公开建议审阅及九个最终失败见[30 — 100样本Critic校准版本结果复核](30-100样本Critic校准版本结果复核.md)。无新付费请求、代码修改或方法切换。
