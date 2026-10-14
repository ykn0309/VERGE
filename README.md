# VERGE

Verified Experience-guided Rubric Generation.

## 当前状态

[计划61](docs/plans/61-延迟归纳与正负案例经验更新.md)已实现：保留Beta正负累计，反向完整负信用、同判25%轻负信用；候选自身P>N，正常结束最多新增一条。已有经验正负案例每3个不同下游任务归纳一次，第10个案例最后归纳，之后冻结文本/案例、继续信用。取消Fusion/MERGE及拆分，Critic样本内多轮和Generator固定E/当前C保持。

最近完整链路实验是计划49的HelpSteer3前100条（55初始/17救回/72最终），属于旧拆分协议。本次301项程序测试通过，20个历史目标任务的归纳模块新旧独立复测完成：40输出/42请求，新20有效、旧19有效。结构简化，但负向案例修正仍不稳定；见[简报](docs/实验分析/历史运行归档/delayed-experience-signed-replay-20/实验简报.md)。没有跑新完整链路，历史原产物保留，不跨协议续跑。
- [方法设计](docs/设计方案.md)
- [工程设计](docs/工程设计.md)
- [实施计划48](docs/plans/48-局部正向准入与有限样本经验更新.md)
- [状态索引](PLAN.md)

## 当前机制

```text
task → 检索/Selector固定E → Generator R0 → Judge → 训练R0来源P/N写入
R0失败：最多两轮Critic → 独立Generator(task+固定E+当前C) → Judge
  候选采用且自身P>N：暂存，与整体成功独立
正常结束：优先合格C2，否则合格C1，每题最多新增一条
样本收尾：R0有信用的已有经验收录正负案例 → 每3个不同下游任务归纳 → 第10个案例最后归纳
```

Critic接续本样本消息/真实回复，Generator仅替换C、不积累旧候选。R0已有经验及被选候选分别累计P/N，alpha加P、beta加N，无折扣、不先抵消；P>N只控制候选入池，同判给25%轻负信用。案例包含初始来源，原题不计3任务窗口；每3个不同下游任务归纳，第10条强制最后一次，之后冻结文本/案例；有变化才增加文本版本，旧信用不认证新文本。

## 已确认的工程边界

- 删除无经验诊断模式及experience_enabled开关，不保留旧对话修订分支。空池/Selector不选E仍可执行完整新经验学习。
- 沿用有限纠错与样本/无效轮隔离：格式、类型、引用等错误耗尽不能全局终止；不伪造Judge失败、不额外增加业务机会。网络/鉴权、存储/数据和程序异常仍按既有全局故障策略。
- 不兼容旧配置、协议、提示词版本、检查点或Web展示。新代码只处理新方案，不为历史增加解析或迁移层。
- main、准备数据、截图和历史分析报告保留。随后按用户清理指示删除了5个旧实验运行目录；截图及生成图留在本地并加入Git忽略规则。
- held-out冻结经验池/文本/Beta，仅初始生成评价，不提出候选、不修订、不回写信用。
- 新实验需新配置/输出目录和明确授权，不自动重复100条或全456条。

Generator按总预算1分配有限正数`weight`，程序归一化；Judge逐项输出`satisfied`和非空`reason`，两侧依据保存到轨迹、传给Critic并在Web默认折叠展示。理由不进入Generator，且不等于事实认证。

## 环境与目录

当前数据切换：[计划46](docs/plans/46-HelpSteer3数据切换与Flash30验证.md)使用HelpSteer3-Preference的General英文单轮强一致偏好任务。RewardBench配置和原产物保留，现有提示词、模型和评分流程保持。

```bash
# 仅准备固定版本官方训练集；不会读取validation或覆盖已有输出。
python -m scripts.prepare_helpsteer3
# 独立空池实验，最多30条；官方validation不参与本次调试。
python -m verge.cli run --config configs/helpsteer3-general-bounded-experience-qwen-flash-30.yaml
```

筛选规则与源文件/JSONL哈希保存在`runs/helpsteer3-general-consensus-data/provenance.json`。人工分数、reasoning及附带feedback仅保存在样本metadata用于复核，不进入LLM角色输入。准备脚本与配置独立，不扩充业务协议或为RewardBench增加兼容处理。

需要Python 3.10或更高版本；本机验收使用`/opt/anaconda3/bin/python`（3.12）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

源码保持扁平，主要文件：

```text
verge/models.py       数据结构
verge/config.py       配置
verge/dataset.py      数据输入
verge/llm.py          LLM后端
verge/prompts.py      角色职责与输出协议
verge/calculator.py   确定性计算
verge/modules.py      方法模块
verge/memory.py       记忆、索引与轨迹
verge/pipeline.py     流程编排
verge/cli.py          实验入口
verge/web.py          loopback只读诊断
verge/web_assets/     页面资源
configs/             实验配置
tests/               离线测试
docs/                方法、工程与计划
```

新版配置使用 `modules.application.implementation: local_credit_bounded_experience`，`max_rounds: 2`；每轮一次Critic业务提案固定在流程中，`invalid_output_retries`只控制有限结构纠错，不增加业务机会。默认Fake配置及OpenAI/Qwen示例已迁移；历史实验配置保留原样，新CLI拒绝它们。

```bash
# 离线Fake示例；输出目录必须不存在。
python3 -m verge.cli run --config configs/default.yaml
python3 -m pytest -q
python3 -m verge.web --runs-dir runs --port 8765
```

manifest、trajectory使用版本7，Memory使用版本5；提示词标识为 `local-credit-delayed-experience-signed-cases`。Web只列当前协议，展示候选、应用/采用、Judge、局部资格/最终入池、案例数/文本版本及P/N；不解析旧版本，不跨方法resume，历史产物保留。新版Web仅列新协议运行，当前没有新完整运行可展示；本轮未重启Web。

历史计划43配置 [rewardbench-chat-hard-qwen-direct-experience-flash-30.yaml](configs/rewardbench-chat-hard-qwen-direct-experience-flash-30.yaml) 已按用户授权完成，使用独立输出目录；结果见[计划43](docs/plans/43-无Admission固定Flash30样本实验.md)，不自动继续剩余样本。

离线验证覆盖两轮预算、实际HTTP消息、固定E/替换C、严格来源、未采用成功、候选零/净负不入池、延迟C2/C1选择、10/11条案例和文本冻结、错误边界、冻结/续跑以及前端渲染。Mock和Fake仅验证工程契约。历史100条准入流程结果见[计划32](docs/plans/32-新机制100样本真实验证.md)，最近Flash30条结果见[计划41](docs/plans/41-固定Flash非思考30样本实验.md)。

## 历史结果

两次100条仅生成/修订诊断：[计划27](docs/plans/27-100样本仅生成与修订复核.md)、[计划30](docs/plans/30-100样本Critic校准版本结果复核.md)。分别最终90/100与91/100；后者公开偏好身份泄漏35/47，并含负向问题按是加分的假成功。结果不证明经验机制有效，也不能视为新方案评价。历史方法留在main与Git记录，新分支不承担旧运行加载兼容。
