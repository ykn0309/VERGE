# VERGE

Verified Experience-guided Rubric Generation.

方法设计见 [`docs/设计方案.md`](docs/设计方案.md)，工程设计见 [`docs/工程设计.md`](docs/工程设计.md)。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 目录

```text
configs/default.yaml   默认实验配置
verge/
  models.py            领域数据结构
  config.py            配置加载
  dataset.py           数据集读取
  llm.py               LLM 调用接口
  prompts.py           Prompt 定义
  calculator.py        确定性计算
  modules.py           方法模块
  memory.py            Memory 与 trajectory 存储
  pipeline.py          流程编排
  cli.py               实验运行入口
  web.py               本地只读日志可视化服务
  web_assets/           页面资源
tests/                 测试
docs/                  设计文档
```

## 查看运行日志

在运行完成后（也支持运行中查看已有记录），启动本地只读页面：

```bash
python -m verge.web --runs-dir runs --port 8765
# 或指定启动时默认选中的运行目录（同级目录仍可在页面中切换）：
python -m verge.web --run-dir runs/<运行目录> --port 8765
```

在浏览器打开 `http://127.0.0.1:8765/`，通过顶部下拉框切换指定目录下的运行日志（仅扫描含 `run_manifest.json` 的直接子目录，无须重启服务）；切换时样本和经验池同时更新。顶部「收起侧栏 / 显示侧栏」可切换导航宽度，选择会保存在本地浏览器。样本索引使用独立的训练/测试分页，各页单独搜索和筛选结果；无可靠 split 信息的轨迹单列为「未标注」。较长的召回列表、Chosen/Rejected 回答和初始/修订 Rubric 可独立滚动；样本页与经验池页直接显示内容，不再重复顶部页面标题；样本详情不显示流水线状态条或模型请求卡片。经验名称是从内容生成的摘要，悬浮或聚焦可预览完整内容与适用条件，点击查看证据演化。可查看概览（包括已保存轨迹中的候选经验总数、新增/合并/未入池数量）、各样本召回→Rubric→判断→修订→反思→入池流水线，以及经验池置信度/证据演化；初始先验关联入池样本，但先验本身不计入样本信用。页面每次刷新读取最新 JSONL 和快照；未写入轨迹的中断请求会单列显示。经验时间线依据证据顺序重建，并非真实时间戳；内存仅保存当前规范经验和可靠性证据，不保存历次完整内容版本。训练/测试及任务正文只在原始数据文件与 manifest 哈希匹配时展示。样本详情顶部有「流水线」及每个 LLM 角色各自的 Tab，切换后查看该角色每次实际调用的输入、模型原始输出及失败重试；展示的是当时的真实角色投影和消息，而非从最终轨迹推测的提示词。新运行会把原始 I/O 写入 `artifacts/role_io.jsonl`（仅本机文件权限 0600，包含完整任务与偏好对，不含鉴权头/API key）；网页按需加载选中样本，不把全量原始消息放进概览响应；单样本页面最多展示前 256 次角色调用，超过 1 GiB 的原始日志不通过网页读取。Web 仅支持当前提示词版本；当前运行尚未写入角色记录时显示未记录。续跑会复制原文件及哈希，缺失记录不会补造。请勿共享这份原始日志。Task、Chosen、Rejected 默认展开，各有「翻译成中文 / 显示原文」按钮。翻译仅在点击时使用 Apple 设备端 Translation 框架，不调用实验模型或上传文本；需 macOS 26+、Swift 命令行工具，并预先在「系统设置 → 通用 → 语言与地区 → 翻译语言」安装原文语言和简体中文语言包。首次点击会编译本地桥接程序，稍慢；未安装语言包或系统不支持时页面会显示错误，原文仍可查看。只监听本机回环地址，不提供修改接口；日志可能包含敏感信息，请勿将端口暴露至公网。

## 独立无经验 baseline

主实验启动后可以用同一固定训练数据及 Generator/Judge 模型设置运行 baseline。
仅生成无经验初始 rubric 并 Judge，不修订、不检索、不更新主经验池。来源数据哈希和提示词版本必须匹配，输出目录必须全新且在来源目录之外。

**以下 `run` 命令会产生模型费用；须另行确认运行，默认不与主实验同时抢占请求额度。**

```bash
python -m verge.baseline run \
  --source-run runs/rewardbench-chat-hard-qwen-vector-full-train \
  --output-dir runs/rewardbench-chat-hard-qwen-vector-full-baseline

# 只读配对比较，不发起模型请求；双方未完成时明确报告 provisional。
python -m verge.baseline compare \
  --source-run runs/rewardbench-chat-hard-qwen-vector-full-train \
  --baseline-run runs/rewardbench-chat-hard-qwen-vector-full-baseline
```

baseline 保存独立 manifest、轨迹、请求元数据、0600 原始角色 IO 和最终汇总；空 memory 文件仅供现有只读页面显示，不实例化经验更新流程。Judge 输出纠错耗尽记为未知并继续；后端故障或初始生成失败仍中止，不伪造结果，也不支持任意断点恢复。比较只用双方有效的**初始**判断，输出配对正确/错误、净收益、50 条分段和子集统计，修订成功不替代初始结果。

开发 baseline 不重启主实验或改写来源产物；新增 Python 源码会改变后续续跑指纹，不能篡改旧 manifest 绕过校验。该对照不消除模型随机性或重复使用 Chat Hard 做诊断的污染。

### 阶段性观察

`python3 -m verge.cli run --config <配置> --stop-after-train 100` 在第 100 条训练任务完整写入后正常退出，保留全量数据与顺序，不进入测试；`run_manifest.json` 记录该前缀上限，返回状态 `paused_after_train_prefix`。这不是完整实验结束。随后以新输出目录使用 `--resume-from <原目录>` 继续；恢复仍核验数据、配置、提示词版本、经验池及完整轨迹前缀，上限包括已复制的样本。

候选经验生成与操作绑定由 Reflector 在一次模型调用中完成：每条候选返回 content、trigger 和 linked_operation_ids，系统自动分配候选编号并校验成功操作引用。修订信用分配是纯程序计算，不再设置独立的绑定模型角色或 Web 角色页签；贡献、预算、共享分摊和空关联处理规则不变。
