# 79 — 模板应用交由Checker语义核对

用户要求Generator使用经验时不以程序检查是否遵循模板，交给Generator Checker结合语义与上下文判断。

删除应用阶段：绑定键与方法参数键完全相等、binding_evidence逐字task子串、question与template.format完全相等，以及bindings与binding_evidence键集合一致的硬门槛。保留JSON对象/字符串类型、非空文本、经验来源ID、数量/权重等接口约束；保留Method定义自身的占位符/参数一致性，不把该定义结构检查误称为Generator应用语义检查。

Generator仍应合理绑定、忠实保留方法，可自然表述，依据可为原文或简短推导说明。Generator Checker明确检查参数遗漏、额外绑定、无依据要求及语义偏离，不因等价改写、翻译或合理推导机械判错。Checker可能误判；候选准入仍按现行实际引用/净正向，不增候选Checker或语义准入门槛。

验证：离线测试覆盖模板改写、缺失/额外参数及非逐字依据被保留，类型错误仍拒绝。独立复测Generator与Generator Checker各20个不同历史任务，Qwen3.5-Flash关闭思考，新输出不反馈其他角色或池；Generator使用本次30题中真实请求，优先带模板失败候选，Checker沿用计划71固定真实应用（不含新模板，明确覆盖局限）。不自动重跑完整链路。

产物`artifacts/v2-semantic-templates-79/`；脚本`scripts/replay_semantic_templates.py`。状态：完成。203项离线测试通过，Generator独立20题18有效、Generator Checker20题均结构有效；后者历史输入缺乏新模板，语义覆盖仍有限，见[报告79](../实验分析/79-模板应用交由Checker语义核对.md)。
