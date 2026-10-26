# 78 — General30题提示词改进完整复跑

用户授权重新运行30题完整链路，观察计划75至77提示词修正后的输出和应用效果。沿用计划73相同30个General训练任务、顺序、seed42、模型Qwen3.5-Flash关闭思考、混合检索Top20、Selector最多5条、Critic两轮每轮0/1候选；从空经验池开始，不使用旧池或旧输出。collection模式，不加载概率参数、不更新Q、不启用修订，不读取测试集。

配置：`configs/helpsteer3-general-v2-prompt-v9-qwen-flash-30.yaml`；新运行：`runs/helpsteer3-general-v2-prompt-v9-qwen-flash-30/`；审计：`artifacts/v2-prompt-v9-full-78/`。运行前保存旧manifest、memory、trajectory及输入数据的SHA256，运行后核验未改。仅运行本次30题，不扩大实验，不中途改提示词，不自动提交推送。

完成后比较首次/最终正确率、候选结构失败、经验模板/参数覆盖、Generator原始权重及Checker覆盖；逐任务对照并抽查问题案例。多个提示词同时变化、模型随机性和经验池路径分化存在，不将前后差异直接当作任一修改的因果效果，也不声称泛化收益。

状态：完成。首次18/30、最终19/30，入池1条、无下游引用；5题候选应用耗尽但继续。详见[报告78](../实验分析/78-General30题提示词改进完整复跑.md)。
