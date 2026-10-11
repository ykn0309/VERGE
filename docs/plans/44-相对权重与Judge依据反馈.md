# 44 — 相对权重与Judge依据反馈

## 1. 目标与范围

根据用户讨论确认的方案，将1–10整数重要性改为整套标准的相对权重预算；Judge逐项提供简短可核对依据；整体重写Generator/Critic处理过程，使经验表达可执行的标准设计动作，并检查采用后的标准是否相容。

不增加输入状态/适用维度专用字段、Admission、审核角色、业务轮次、外部核验工具或探索名额；不开展专项跨实例验证或随机性对照。本轮完成实现与离线验收，真实实验仍固定Qwen3.5-Flash、关闭思考，另行启动新目录实验。

## 2. 数据与评分

- Criterion以`weight`替代`importance_score`，值为有限正数。Generator按总预算1共同分配；允许实际总和不为1，由Calculator统一归一化。保留模型分配原值，Judge结果保存`normalized_weights`。
- 2–8条正向Boolean标准。权重反映task主次，不按偏好身份或预期区分能力分配；避免重复维度通过拆分重复占据预算。
- 用输入数字的十进制表示计算权重比例与margin符号，避免0.1+0.2与0.3的浮点差异形成伪成功；仍严格margin>0成功，无新阈值、否决、跳过或部分分。
- P/N、多来源均分、R0一次性信用、候选采用成功后Fusion以及Memory字段保持。数值证据仍与来源自报/语义有效性区分。

## 3. Judge依据与角色边界

- 每侧每项输出`criterion_id`、`satisfied`、`reason`；reason必须为非空字符串，用简短事实依据指出对应内容、缺失信息或数量，不要求长篇推理。
- 合并后的CriterionJudgment新增必填`chosen_reason`和`rejected_reason`。理由随两侧Boolean、编号保存到轨迹、实际IO、Critic当前/前轮证据与Web中。
- 两侧请求独立，不展示偏好身份；Judge理由不进入Generator/Selector/Fusion输入。重试仍有限，格式耗尽和基础设施错误边界保持。
- 复用判断时连同两侧原理由一起复用；同文本出现矛盾Boolean时沿用既有规则，不凭理由选择有利判断。
- 理由是待核对的模型判断依据，不是事实认证。Critic用task、原回答与依据区分设计缺陷、表达歧义、Judge执行错误与证据不足。

## 4. 提示词

Generator仍以全新会话读取task及固定E/当前C。依据task形成整套正向标准，自主采用经验；采用可能只影响一项，也可能影响问题表达、维度关系及权重。检查各项与task和已采用经验是否相容，不强制全套重构，不只以来源引用当作落实。

Critic先核对事实及Judge依据，定位可支持的设计动作，再将本题常量转为任务参数、主体转为必要语义角色。content描述如何设计标准，trigger仅描述事前可识别条件，允许必要领域边界。第二轮区分前轮诊断错误、方法不足、未落实和Judge问题，依据新证据调整，避免重复建议。仅有Judge误判且没有可信设计动作时可空候选，不迁就错误判断。

## 5. 版本与历史

application标识改为`critic_candidate_weighted_application`，Prompt版本为`critic-candidate-weighted-judge-reasons`。Manifest和Trajectory升级到4，Memory仍为2。CLI、baseline、Web、resume拒绝旧协议；不迁移历史产物、不跨方法resume。默认与示例配置更新，已运行实验配置和产物保留。

## 6. 离线验收

- 权重合法性、归一化、比例不变性、小数精确平分/微小正负margin及P/N。
- Judge必填依据、两侧对应、数字ID、有限纠错、复用；真实HTTP模拟验证依据进入Critic且不泄入Generator。
- 两轮直接应用、采用分流、Fusion、冻结、存储、版本拒绝和resume。
- Web小数权重与两侧理由展示、转义、JavaScript语法；全套pytest及差异空白检查。

## 7. 完成记录

已实现模型weight契约、统一归一化及十进制精确比例评分；Judge两侧reason输出校验、持久化、复用与Critic当前/前轮实际消息；Generator/Critic/Judge提示词；Web原始/归一化权重及默认折叠、转义的两侧依据。

新增HTTP测试发现Judge后端格式重试耗尽后原适配器会再叠加结构重试，已改为直接转为JudgeOutputError；保持样本隔离和网络/鉴权错误优先级，另一侧有效结果不重试。

协议升级为Manifest/Trajectory 4，Memory 2；旧importance字段、Boolean-only判断、旧manifest/trajectory/config与跨协议resume明确拒绝。默认/示例配置更新，历史实验配置、原始运行、快照及分析未改写。检索文档中的探索名额描述已校准为当前未实现、CLI只接受0；选择机制未改变。

验收：`/opt/anaconda3/bin/python -m pytest -q` **238 passed**；`node --check verge/web_assets/app.js`、`git diff --check`、活跃文档本地链接检查通过。实际Node前端函数渲染验证权重、两侧依据、HTML转义及默认折叠；本轮未进行完整浏览器交互验收。环境使用`/opt/anaconda3/bin/python`（Python 3.12.7）；系统`python3`为3.9，不能运行项目已有PEP 604类型解码，最初该环境的测试失败不作为协议结果。

状态：实现与离线验收完成。当前版本离线实现已完成；后续提交与真实验证见计划45。离线契约通过不代表提示词语义效果、Judge事实正确性或经验迁移改善。
