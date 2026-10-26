# 80 — General30题语义模板应用完整实验

用户要求先不修复新发现的问题，再运行30题完整实验。使用当前计划79代码与v10提示词：模板应用由Generator Checker语义核对，不再以程序检查模板逐字渲染、参数覆盖与原文子串；其他已发现的标准设计、Critic证据误读等问题不改。

沿用计划73/78同一批30个General任务、顺序、seed42；Qwen3.5-Flash关闭思考、混合检索Top20、Selector最多5条、最多两轮Critic且每轮0/1候选。空池启动，collection模式，无概率参数、Q更新、修订或测试集。有限纠错，样本失败隔离。不在中途调提示词，不扩大样本。

配置`configs/helpsteer3-general-v2-prompt-v10-qwen-flash-30.yaml`；运行`runs/helpsteer3-general-v2-prompt-v10-qwen-flash-30/`；审计`artifacts/v2-prompt-v10-full-80/`。运行前保存上轮manifest、Memory、Trajectory及数据SHA256，结束后检查未改变。

对比计划78首次/最终正确、无效耗尽、候选/入池模板、实际引用与Checker覆盖。上轮v9、本轮v10，差异包括机制变化、模型随机性与经验池路径，不把单轮差异当作严格因果证据。实验完成只报告，不自动修复。

状态：完成。首次16/30、最终17/30，0条入池，6题候选阶段无效耗尽但未中断；详见[报告80](../实验分析/80-General30题语义模板应用完整实验.md)。
