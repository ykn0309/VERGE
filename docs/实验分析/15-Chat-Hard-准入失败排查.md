# 15 — Chat Hard 准入失败排查

> 只读排查，未修改提示词或准入规则，未发起额外模型请求。依据主运行 `runs/rewardbench-chat-hard-qwen-vector-full-train-resume1/` 的轨迹、角色真实 IO，以及 `verge/prompts.py`、`verge/modules.py` 和方法文档。

## 1. 漏斗与拒绝理由

156 条反思候选，151 条进入 Admission，25 PASS、126 FAIL，准入通过率 **16.56%**。绑定阶段只损失 5 条，不是主要瓶颈。

126 条 FAIL 理由均提到 trigger/触发条件；58 条还提及“生成方法/方法论”，55 条提及“泄漏/偏好”。这些是按理由关键词统计的重叠标签，不是独立原因分类，更不是人工金标准错误率。

FAIL 中 108 条 credit 为正、15 条为零、3 条为负；PASS 中 21 条为正、2 条为零、2 条为负。代码没有正信用最低阈值，低准入不是确定性 credit 截断导致。正 credit 也不意味着候选一定合格。

## 2. 已确认的判定问题

### A. 把 task 可观察条件误当成事后条件

规则明确允许凭未来 task 的题型、显式要求和任务上下文判断。实际模型却把阅读输入、识别题型或语义分析本身当成违规。

- `rewardbench:llmbar-adver-neighbor:2634 / candidate_1`：输入为摄氏转华氏，候选要求建立精确结果检查，trigger 为数值输入和转换规则。credit=1.0，绑定操作包含实际结果正确性标准。拒绝理由把“识别数值转换题型”称为事后验证，与规则不符。
- `rewardbench:llmbar-adver-neighbor:2747 / candidate_1`：用户只说“展开这些要点”却未提供要点；trigger 是任务要求展开但缺少输入。是否缺少要点可从 task 判断，拒绝理由却认为要结合未来回答。
- `rewardbench:llmbar-adver-manual:2926 / candidate_1`：trigger 是统计指定引文单词数。模型把识别 task 中的引文也判为依赖执行结果。

这些证实拒绝理由存在错误，不代表每个候选在其他条件上都必然应 PASS。

### B. 混淆 content 的证据来源和 trigger 的未来可见性

Admission 本应查看成功操作来验证 content 支撑，只有未来 trigger 不能依赖这些操作。实际模型有时因为候选源自成功修订就否定迁移性。

- `rewardbench:llmbar-adver-GPTInst:2768 / candidate_2`：数量约束经验有对应 ADD 支撑，trigger 为任务的明确数量词。拒绝理由承认数量条件可见，却又因 content 来自修订而拒绝。
- 对照 `rewardbench:llmbar-adver-neighbor:2681 / candidate_1`：近似的“只提供一种方法”原则与 task 数量词 trigger 被 PASS。两者不是完全同一输入，但理由呈现明显不一致。

### C. 将轨迹细节错误归给候选

- `rewardbench:llmbar-adver-neighbor:2716` 的两条候选均未包含“1到50”，但拒绝理由称候选包含该范围。“1到50”实际在 task 和成功 rubric 中。
- `rewardbench:llmbar-adver-GPTInst:2782 / candidate_2` 绑定事实忠实性 ADD，却被要求支持另一操作的“有机融合”要求，表现出跨操作证据混淆。

实际 Admission 请求包含完整成功 rubric、全部操作和逐项判断；真实 IO 核验没有发现候选被数字 ID 映射成另一条文本。全量轨迹上下文可能促成混淆，但尚不能据此认定因果来源。

## 3. 确有不合格候选，也存在误放行

不能把 126 条拒绝都当误拒。

- `rewardbench:llmbar-adver-GPTInst:2779 / candidate_1` 的 trigger 明确依赖“初始标准中缺乏定义”，拒绝符合 task-only 边界。
- 同任务 `candidate_2` 的 trigger 也依赖“现有标准模糊”，却被 PASS，并发生 MERGE。
- 另外 `rewardbench:llmbar-natural:2688`、`rewardbench:llmbar-adver-neighbor:2733`、`rewardbench:llmbar-adver-neighbor:2704`、`rewardbench:llmbar-adver-GPTInst:2773` 均有 PASS 候选的 trigger 明确依赖初始标准、初始回答或评价标准。至少这五条 PASS 有明显的事后 trigger 问题。
- 反思候选还存在回答依赖、当前案例专有细节、把响应要求直接写成经验等问题；有些“回答应……”可转述为检查标准，但不能仅因此就认定其满足 rubric-generation 方法要求。

因此同时存在 **Reflector 候选不合格**和 **Admission 误拒/误放行**，不是统一提高通过率就能解决。

## 4. 建议的下一步

1. 先建立固定复核样例：数值计算、数量限制、缺失输入等合法 task 条件；与回答/rubric 条件、专有细节泄漏等负例配对。逐条评估，不用整体通过率作唯一目标。
2. 澄清 Admission 提示词：读取和语义理解 task 是允许的；content 可由成功操作支持；trigger 才受未来可见性约束；泄漏理由必须引用候选真实片段。分别判断这些条件，避免把问题混为一个 FAIL。
3. 同步 Reflector：经验明确写成“生成/检查什么 rubric 标准”，trigger 只描述 task 的适用范围，禁止把现有标准缺陷或回答表现写入条件。
4. 必要时精简 Admission 证据为候选及已绑定操作的 before/after，避免无关操作细节干扰；是否保留更多成功轨迹信息需按方法约束设计，不能先假定移除一定有效。
5. 使用保存的候选做离线重判对照，统计有依据的纠错和错误放行；额外模型费用需确认。不要改写原实验候选、判断或经验池，也不要直接把所有 FAIL 重新灌入池。

结论：准入低不仅来自候选差；已有多例证明检查器把合法 task 条件当违规，并混淆候选与轨迹细节。应优先修正规则执行准确性，再观察经验积累，而非机械放宽阈值或取消准入。当前没有完整人工金标准，不能给出精确误拒比例或修复后池规模。
