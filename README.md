# VERGE

从偏好轨迹提出评分标准设计经验，以同题经验子集对照计算任务条件下的单条信用，供后续任务选择经验。基础模型参数固定。

当前代码为v3，500题实测未显示经验净收益；v3.1设计已整理，尚未实现。见[当前设计](docs/设计方案.md)、[工程设计](docs/工程设计.md)、[v3归档](docs/历史设计/2026-11-13-v3/README.md)、[当前计划](PLAN.md)和[实测报告](docs/实验分析/120-General500题四路流程实测.md)。

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

每个完成案例把经验、聚类、下游信用证据和各子集轨迹一起原子保存到`checkpoint.json`；调用输入、输出和错误存入`role_io.jsonl`，usage存入`llm_requests.jsonl`。Selector最多选择两条经验；训练运行无经验、单条和组合，初始实际路径固定为全部选中经验。冻结推理只运行这一条路径。Generator不输出经验引用；来源验证只决定准入，不记收益、不建来源关联。

Web仅监听127.0.0.1，显示新四路协议及旧两路实验的只读结果，可查看完整提示词和失败原因；翻译调用本机苹果服务。训练加载器仅接受verge-subset-learning-v2，不混用旧两路收益。旧实验文件仍保留，旧源码／配置／专用测试可以从Git快照`9faede2`取得；不要把旧实验结果视作新方法验证。
