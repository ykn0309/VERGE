# VERGE

Verified Experience-guided Rubric Generation.

从偏好轨迹中提炼可复用的评分标准设计经验，指导Generator设计rubric，并通过后续应用反馈维护经验与可信度。

## 当前状态

当前代码为[计划68](docs/plans/68-演化信用继承与稳定性惩罚.md)：独立原子化、固定文本版本、混合检索、局部归类、持续演化及信用继承/稳定性惩罚。Memory8、Manifest/Trajectory10，不兼容旧协议或跨方法resume。

最新完整实验为[计划69](docs/plans/69-信用继承版本200样本完整链路实验.md)：200条处理完成，113条首轮正确、148条最终正确，5条技术错误隔离；54个经验版本（42活跃/12退役）、8个类别。结果只描述该次训练运行，不证明泛化、因果收益或好坏经验已经可靠分化。详见[实验报告](docs/实验分析/69-信用继承版本200样本实验结果.md)。

## 文档入口

- [完整设计方案 V2](docs/设计方案.md)：半结构化经验、并行核对、贝叶斯质量更新与修订回放；目标设计，尚未实现。
- [旧版实现设计归档](docs/历史设计/2026-10-24-V1/设计方案.md)：解释当前计划68代码及历史实验。
- [工程设计](docs/工程设计.md)、[状态与计划](PLAN.md)
- [组会材料](docs/组会汇报/2026-10-17/方法流程-分层详细版.html)

## 当前实现流程

```text
task → 向量＋BM25/RRF检索类别/未分类经验 → Selector固定E
     → Generator(task＋E)生成R0 → 两侧独立Judge → 程序评分及R0来源P/N
失败 → 最多两轮Critic草稿 → Atomizer候选集合C
     → 新Generator(task＋固定E＋当前C) → Judge → 逐候选引用及P>N资格
正常收尾 → 最新合格轮的合格集合入池
         → 每5个新下游有区分任务，用最近5例调用Updater
         → keep，或经Atomizer修订/拆分为新版本、退役父版本
         → 局部归类及类别名称/描述更新
```

P/N按实际引用标准的归一化权重分配，多来源均分；同判只计次，不加减信用、不进归纳。修订继承父信用，拆分均分；演化总惩罚为最近5例累计信用量的20%。继承不等于新版本已经验证，实际反馈、继承及惩罚分别记账。

类别与成员分别建索引，任务检索Top20单位、归类Top10类别，Selector最多选5条小经验。类别无信用。Generator不见偏好对或信用，Judge分别评价单份回答且不见偏好标签。held-out冻结所有业务状态，只运行检索、选择及R0评价。

## 环境与运行

需要Python 3.10或更高版本；本次离线验证使用Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

# 离线Fake示例；配置中的输出目录必须不存在。
python -m verge.cli run --config configs/default.yaml
python -m pytest -q

# 本地只读WebUI，仅监听127.0.0.1。
python -m verge.web --runs-dir runs --port 8765
```

真实实验默认Qwen3.5-Flash、关闭思考；参见[配置示例](configs/qwen3.5-flash.yaml)。密钥通过环境变量提供，不写入代码或日志。当前数据为HelpSteer3 General英文单轮强一致偏好任务；准备官方固定版本训练集使用：

```bash
python -m scripts.prepare_helpsteer3
```

原始数据、运行日志、经验池及缓存不随仓库发布。数据准备的筛选条件与哈希在运行产物中保存；历史配置与报告保留各自原协议解释，不代表都能由当前CLI直接运行。

## 工程约束

源码保持扁平，主要模块位于`verge/`，配置在`configs/`，离线测试在`tests/`，方法及阶段记录在`docs/`。

格式/结构和可隔离请求错误采用有限纠错，耗尽后隔离样本或模块并继续；鉴权/配置、存储/数据和程序错误fail-fast。合法R0反馈不因后续错误撤销，不静默截断输入或重复记账。

改进模块时，每模块至少20条代表性真实历史输入独立复测，不自动重跑完整流水线。Mock/Fake测试仅验证工程契约，不代表方法有效。研究设计、未解决问题及效果边界以对应文档为准。
