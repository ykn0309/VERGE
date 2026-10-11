# VERGE

Verified Experience-guided Rubric Generation.

## 当前状态

当前主链已按用户要求删除Admission：**Critic候选 → 独立Generator应用 → Judge → 合格Fusion**。每轮一次提案、最多两轮。当前已按[计划44](docs/plans/44-相对权重与Judge依据反馈.md)升级相对权重与Judge依据反馈，238项离线测试通过；该版本尚未真实实验。所有真实实验LLM角色保持Qwen3.5-Flash且关闭思考；[新30条实验](docs/plans/43-无Admission固定Flash30样本实验.md)已完成：17初始成功、4救回、21最终成功，4条入池，零请求错误；存在事后条件及实际落实问题，未证明迁移或因果效果。历史原始生成产物已按用户指令删除，报告保留。

- [方法设计](docs/设计方案.md)
- [工程设计](docs/工程设计.md)
- [实施计划42](docs/plans/42-删除Admission直接应用候选经验.md)
- [状态索引](PLAN.md)

## 当前机制

```text
task → 检索 / Selector得到固定E
→ Generator首次生成R0 → Judge
→ 按R0实际来源的P/N更新已有经验
R0失败：
  Critic提出0–1条C → 全新Generator(task+E+C)，自主采用 → Judge
    成功且采用C：Fusion → Memory
    成功未采用C：任务成功，C未验证，不入池
    失败：有剩余轮次则依据实际失败结果重新提出并独立应用
最多2次业务提案、2轮应用；没有准入或拒绝反馈重生成。
```

实际Generator消息使用统一“本次可用经验”列表，固定E在前、当前C在后；每轮只替换C，不积累历史。C标注“当前候选、尚未验证”，没有历史可信度，任务约束优先。来源是采用自报，不证明真实实施或因果贡献。

R0已有经验与应用候选统一按来源标准的P/N分别累计：alpha加P、beta加N，多来源均分，无折扣、不先抵消、不用margin增益；双通道均0才不增加数值证据。整体成功可以有负信用，零信用也可合法入池。Fusion默认ADD原文，仅兼容的同一设计问题才MERGE并保留文本版本；继承证据不认证扩展。

## 已确认的工程边界

- 删除无经验诊断模式及experience_enabled开关，不保留旧对话修订分支。空池/Selector不选E仍可执行完整新经验学习。
- 沿用有限纠错与样本/无效轮隔离：格式、类型、引用等错误耗尽不能全局终止；不伪造Judge失败、不额外增加业务机会。网络/鉴权、存储/数据和程序异常仍按既有全局故障策略。
- 不兼容旧配置、协议、提示词版本、检查点或Web展示。新代码只处理新方案，不为历史增加解析或迁移层。
- main、准备数据、截图和历史分析报告保留。随后按用户清理指示删除了5个旧实验运行目录；截图及生成图留在本地并加入Git忽略规则。
- held-out冻结经验池/文本/Beta，仅初始生成评价，不提出候选、不修订、不回写信用。
- 新实验需新配置/输出目录和明确授权，不自动重复100条或全456条。

Generator按总预算1分配有限正数`weight`，程序归一化；Judge逐项输出`satisfied`和非空`reason`，两侧依据保存到轨迹、传给Critic并在Web默认折叠展示。理由不进入Generator，且不等于事实认证。

## 环境与目录

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

新版配置使用 `modules.application.implementation: critic_candidate_weighted_application`，`max_rounds: 2`；每轮一次Critic业务提案固定在流程中，`invalid_output_retries`只控制有限结构纠错，不增加业务机会。默认Fake配置及OpenAI/Qwen示例已迁移；历史实验配置保留原样，新CLI拒绝它们。

```bash
# 离线Fake示例；输出目录必须不存在。
python3 -m verge.cli run --config configs/default.yaml
python3 -m pytest -q
python3 -m verge.web --runs-dir runs --port 8765
```

manifest、trajectory使用版本4，Memory保持版本2；提示词标识为 `critic-candidate-weighted-judge-reasons`。Web只列当前协议，展示候选、应用/采用、Judge、Fusion及P/N；不解析旧版本，不跨方法resume，历史原始目录已按用户指令清理，研究报告保留。

历史计划43配置 [rewardbench-chat-hard-qwen-direct-experience-flash-30.yaml](configs/rewardbench-chat-hard-qwen-direct-experience-flash-30.yaml) 已按用户授权完成，使用独立输出目录；结果见[计划43](docs/plans/43-无Admission固定Flash30样本实验.md)，不自动继续剩余样本。

离线验证覆盖两轮预算、实际HTTP消息、固定E/替换C、严格来源、未采用成功、负/零信用Fusion、错误边界、冻结/续跑以及前端渲染。Mock和Fake仅验证工程契约。历史100条准入流程结果见[计划32](docs/plans/32-新机制100样本真实验证.md)，最近Flash30条结果见[计划41](docs/plans/41-固定Flash非思考30样本实验.md)。

## 历史结果

两次100条仅生成/修订诊断：[计划27](docs/plans/27-100样本仅生成与修订复核.md)、[计划30](docs/plans/30-100样本Critic校准版本结果复核.md)。分别最终90/100与91/100；后者公开偏好身份泄漏35/47，并含负向问题按是加分的假成功。结果不证明经验机制有效，也不能视为新方案评价。历史方法留在main与Git记录，新分支不承担旧运行加载兼容。
