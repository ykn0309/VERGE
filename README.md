# VERGE

Verified Experience-guided Rubric Generation.

## 当前状态

分支 `feat/critic-experience-validation` 从基线 `7ab9bdc` 建立。**计划31已实现，230项离线测试通过**：主链已切换为Critic候选→Admission→独立Generator应用→Judge→合格Fusion。473项测试属于历史基线，真实模型的新机制效果尚未验证。

- [方法设计](docs/设计方案.md)
- [工程设计](docs/工程设计.md)
- [实施计划31](docs/plans/31-Critic候选经验准入与新会话验证.md)
- [状态索引](PLAN.md)

本轮已按授权实现新机制、清理旧实验日志并提交代码；未push、未启动付费模型或实验。

## 新机制

```text
task → 检索 / Selector得到固定E
→ Generator首次生成R0 → Judge
→ 按R0实际来源信用更新已有经验置信度
R0失败：
  Critic提出候选C → Admission
    首次FAIL：拒绝原因反馈同一个Critic，仅重生成一次
    再次FAIL：结束本任务，不进入Generator
    PASS：全新Generator(task + E + C)，自主采用 → Judge
      成功且采用C：Fusion → Memory
      成功未采用C：任务成功，C未验证、不入池
      失败：有额度则重新提出候选并重复同一流程
最多2轮应用，每轮一次准入重生成；Critic最多4次业务生成。
```

E与C都是可选的rubric设计指导，task明确约束优先；C没有历史可信度，不强制引用。来源标注是模型自报，不证明因果性。信用已确定：R0已有经验与本轮候选统一按来源标准的P/N分别无折扣累计，alpha加P、beta加N，不先抵消、不用lambda或margin增益。双通道均0才不更新，净方向仅展示；任务成功不意味着只有正证据，入池资格不额外改变，单次应用不能证明因果性或泛化。

Reflector退出新主链。候选先准入，再固定原文应用；成功后不重新提炼经验，不重复Admission或增加审核角色。Fusion默认ADD，仅同一设计问题的等价指导/必要细化且条件兼容才MERGE，保存文本版本；继承证据不认证扩展。

## 已确认的工程边界

- 删除无经验诊断模式及experience_enabled开关，不保留旧对话修订分支。空池/Selector不选E仍可执行完整新经验学习。
- 沿用有限纠错与样本/无效轮隔离：格式、类型、引用等错误耗尽不能全局终止；不伪造Judge失败、不额外增加业务机会。网络/鉴权、存储/数据和程序异常仍按既有全局故障策略。
- 不兼容旧配置、协议、提示词版本、检查点或Web展示。新代码只处理新方案，不为历史增加解析或迁移层。
- main、准备数据、截图和历史分析报告保留。随后按用户清理指示删除了5个旧实验运行目录；截图及生成图留在本地并加入Git忽略规则。
- held-out冻结经验池/文本/Beta，仅初始生成评价，不提出候选、不修订、不回写信用。
- 新实验需新配置/输出目录和明确授权，不自动重复100条或全456条。

## 环境与目录

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

新版配置使用 `modules.application.implementation: critic_candidate_validation`，`max_rounds: 2`；每轮一次Admission拒绝重生成固定在流程中，`invalid_output_retries`只控制有限结构纠错，不增加业务机会。默认Fake配置及OpenAI/Qwen示例已迁移；历史实验配置保留原样，新CLI拒绝它们。

```bash
# 离线Fake示例；输出目录必须不存在。
python3 -m verge.cli run --config configs/default.yaml
python3 -m pytest -q
python3 -m verge.web --runs-dir runs --port 8765
```

manifest、trajectory、Memory使用版本2，提示词标识为 `critic-candidate-admission-fresh-application`。Web仅列出当前协议运行，分别展示候选生成/准入、提供/声明采用、Judge结果、Fusion和P/N证据。旧实验日志已按用户指示清理；历史方法仍在main和Git记录中，新版不支持跨方法续跑。

离线验收包括4/2业务预算、严格来源、实际HTTP消息、错误隔离、双通道信用、冻结及同协议续跑；浏览器核验无JavaScript错误或横向溢出。Mock和Fake仅验证工程契约，真实验证另行授权。

## 历史结果

两次100条仅生成/修订诊断：[计划27](docs/plans/27-100样本仅生成与修订复核.md)、[计划30](docs/plans/30-100样本Critic校准版本结果复核.md)。分别最终90/100与91/100；后者公开偏好身份泄漏35/47，并含负向问题按是加分的假成功。结果不证明经验机制有效，也不能视为新方案评价。历史方法留在main与Git记录，新分支不承担旧运行加载兼容。
