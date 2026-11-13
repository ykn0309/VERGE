# VERGE

从偏好轨迹提出评分标准设计经验，以同题单经验对照计算任务条件下的信用，供后续任务选择经验。基础模型参数固定。

当前代码已实现v3.1并完成离线验证，真实模型效果尚未验证。v3的500题实测未显示经验净收益；见[当前设计](docs/设计方案.md)、[工程设计](docs/工程设计.md)、[v3归档](docs/历史设计/2026-11-13-v3/README.md)、[当前计划](PLAN.md)和[实测报告](docs/实验分析/120-General500题四路流程实测.md)。

## 运行

需要Python≥3.10（本机验证使用`/opt/anaconda3/bin/python`3.12）。安装`requirements.txt`中的依赖后，先设置配置中的`run.output_dir`为不存在的目录。

```bash
python -m verge.cli run configs/default.yaml
python -m verge.cli run configs/qwen-flash.yaml --stop-after-train 20
python -m verge.cli run configs/qwen-flash.yaml --resume
python -m verge.cli inspect runs/运行名/checkpoint.json
python -m verge.cli evaluate configs/qwen-flash.yaml runs/运行名/checkpoint.json 测试.jsonl 新评价目录
python -m verge.web --port 8765
python -m pytest -q tests
```

`default.yaml`使用离线夹具，只验证程序；`qwen-flash.yaml`使用真实API，须自行准备General数据和`DASHSCOPE_API_KEY_2`环境变量，默认关闭思考。配置文件本身不启动实验。数据准备入口是`scripts/prepare_helpsteer3.py`与`prepare_general_collection.py`。

## 保存与边界

每个完成案例把经验、聚类、下游信用证据和共享初稿及各单经验轨迹一起原子保存到`checkpoint.json`；调用输入、输出和错误存入`role_io.jsonl`，usage存入`llm_requests.jsonl`。Selector训练最多选择3条，分别与无经验路平等试用；推理最多选择1条。每题只生成一个任务初稿，每路均修订一次（包括无经验）。逐条信用为有／无经验判对值之差，未知不补0；只有全部初始路径明确判错才触发Critic，每轮最多一条、最多两轮。来源验证不记收益、不建关联。

Web仅监听127.0.0.1，显示共享初稿、各路修订和独立Judge、逐轮候选验证，以及原始请求回复；翻译调用本机苹果服务。训练与Web加载器只接受`verge-v3.1`检查点，不兼容旧状态。旧实验及源码通过Git历史保留，日志入口仍为`logs/latest`；不要把v3结果当成v3.1验证。

聚类采用类别分布与文本混合相似度、成员平均低限和模块度多起点搜索；任务低限0.35、经验低限0.50。任务组检索不设门槛，按文本相似度取前3组，再汇集最多6条历史收益经验和全池语义候选，共最多10条。每半轮保存所用对侧划分、矩阵摘要、修复及预算状态；不保证全局最优。

离线验收见[实现报告123](docs/实验分析/123-v3.1实现与离线验证.md)。旧版专用验证脚本和组合测试已删除；复现旧实验应切换对应历史提交。
