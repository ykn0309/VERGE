# 73 — General在线采集30题预试验

已按用户要求完成30道General任务的真实完整链路采集。没有扩展到200题，没有离线标注、参数估计或中途提示词调优；本批用于检查链路和Checker，不能当作正式校准结果。

## 配置与采样

数据来自已有2712条General/英语/单轮/强偏好/标注一致且任务去重的训练集，不接触validation。按偏好强度绝对值2/3及task字符长度三分位（70、375字符）共6层比例抽样，种子42；30题按最大余数法分配后各层恰好5题，不是先强行均衡。抽样后随机打乱执行顺序。代表的是该筛选后General总体，不是完整HelpSteer3或真实用户分布。

所有在线LLM角色Qwen3.5-Flash，enable_thinking=false；混合检索Top20、Selector最多5条；空池、最多2轮Critic，候选Generator仅task+C。run.mode=collection剔除Selector输入中的Q/证据/使用次数，配置拒绝概率参数、自动修订和held-out路径。Q始终未估计，经验文本在入池后保持不变。三个Checker仍按实际引用版本各自调用，不为准入调用。

## 链路结果

| 指标 | 结果 |
|---|---:|
| 完成任务 | 30/30 |
| 首次判断正确 | 14/30（46.7%） |
| 候选纠错后正确 | 21/30（70.0%） |
| 新经验入池 | 9条 |
| 候选业务轮次 | 22轮 |
| 选择了已有经验的任务 | 10题 |
| 实际引用已有经验的任务 | 6题 |
| 下游引用 | 6次，来自2条经验 |
| 下游反馈 | positive 0、reverse 2、tie 4 |
| 选中但未引用 | 4次，unused |
| 样本级程序/请求异常 | 0 |
| 格式纠错耗尽 | 1题（第29题，Critic连续两轮结构无效） |

状态分布：initial_success 14、application_success 7、validation_exhausted 4、no_candidate 4、invalid_application_exhausted 1。第29题已记录错误并继续第30题，无需重启。14→21是同批样本的候选纠错结果，不是后续经验提高首次正确率的因果证据。

最终状态training_completed，evaluation_mode=training_only、run_mode=collection；无test评价，不称为冻结实验完成。

## Checker实际覆盖

| Checker | correct | error | unknown | failed |
|---|---:|---:|---:|---:|
| Selector Checker | 1 | 5 | 0 | 0 |
| Generator Checker | 3 | 3 | 0 | 0 |
| Judge Checker | 5 | 1 | 0 | 0 |

6次引用都有三份最终结构有效报告，共18份；Judge Checker出现2次格式纠错，故实际Judge Checker请求为8次。六次综合O均为error，但它仅代表有Checker报错，不能解释为六次执行都真的错误。运行中曾报告5次/15份，是最后一题入账前的统计，最终以6次/18份为准。

实际复用覆盖只有6题、2条经验；没有正向下游反馈。本轮足以检查接口和找到具体问题，不足以估计四组Q/C概率表。

### 优先查看的案例

- **第12题：Checker把自己的格式纠错当成Judge错误。**任务以“UPTOWN SPRINGDALE / Natsuki…”开始。Judge Checker最初返回未提供的criterion_id=1b，纠错时又返回unknown；最后格式有效，却把这些编号问题归到Judge身上。实际编号错误来自Checker自身输出，不能把最终error作为准确执行标签。完整三次请求均保留。
- **第28题：适用性与应用核对的边界。**任务询问某年份美国州的谋杀率，经验是“具体实体准确性”，Trigger括号中列出剧名、人名、角色名。Selector/Generator Checker都以“州名不属于上述类别”为错误依据；这可能涉及示例是否穷尽、Trigger边界，以及Generator Checker是否越界重复做适用性检查，应对照实际经验和标准讨论，不能仅用两个error就认定经验错误。
- **第29题：Critic格式失败隔离。**两轮均在有限纠错后报`Method: expected an object`，未准入，程序继续下一题。这是模型输出结构失败，不是整次运行崩溃。
- **第30题：短输入不等于应短答。**任务明确要求50种逻辑谬误及例子，却引用“简短非正式输入应匹配响应长度”的经验；可检查Selector是否忽略了任务的具体输出要求及Checker如何判断。

本轮先展示问题，不擅自修改提示词。若后续改提示词，应单独运行固定案例局部复测，正式参数采集重新固定版本，避免把调试批次和新版本采集混成一个概率总体。

## 验证与审计

- 174项离线测试通过，包括采集模式剔除质量统计、禁止参数/修订/held-out、无Q和修订、分层配额、任务去重及可复现顺序。
- Selector修改输入契约另以20条固定真实历史任务独立复测：20/20有效，实际请求均只含任务、方法四字段、短编号与选择上限，无Q或使用次数；输出不反馈本次在线运行。不是离线多模型标注。
- 检查采样源/输出哈希不变、执行代码指纹与manifest一致、Memory/Trajectory最终认证哈希一致；Q全为null、无修订事件、6次引用无缺失Checker组。
- 在线请求元数据250条，其中36次embedding、214次LLM；含有限格式纠错。记录prompt_tokens=320831、completion_tokens=64441、total_tokens=385272；未配置单价，不报告推测金额。Selector独立复测另20次请求，不混入在线数字。
- Web服务127.0.0.1:8765，本次运行API返回200，可查看完整轨迹和按需加载原始角色请求。未做真实浏览器视觉验收。
- 启动Web后实际核验苹果翻译，系统返回缺少英语/简体中文语言包；不是本轮模型或沙盒限制，未自动修改系统语言包。

## 入口

- 配置：`configs/helpsteer3-general-v2-collection-qwen-flash-30.yaml`。
- 运行：`runs/helpsteer3-general-v2-collection-qwen-flash-30/`。
- 抽样：`runs/helpsteer3-general-v2-collection-30-data/sampling.json`。
- 统计与逐题审阅：`artifacts/v2-collection-73/audit/audit.json`、`sample_review.json`、`checker_review.md`。
- Selector独立复测：`artifacts/v2-collection-73/selector-replay/`，含固定输入、源码快照、请求与摘要。
- Web：http://127.0.0.1:8765/?run=helpsteer3-general-v2-collection-qwen-flash-30 。

本轮未自动提交或推送代码。
