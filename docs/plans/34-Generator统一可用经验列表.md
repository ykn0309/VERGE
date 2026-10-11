# Generator统一可用经验列表

用户指出实际Generator消息将“本次选中的经验：暂无”和“本轮可选候选经验”分开，而期望当前候选附在所选经验中。

## 调整

`verge/prompts.py`在实际Generator消息渲染时，将固定E与当前已准入C合并为一份“本次可用经验”列表。E保持原顺序，C附在末尾；无E有C时只显示候选，局部编号仍从1开始。R0也使用同一标题，E/C均为空时显示暂无。

已有E保留confidence/evidence_amount；C保留content/trigger及“已准入、尚未验证，无历史置信度”，不补造Beta、confidence或evidence_amount。角色指导、自主采用、两层预算、来源白名单、第二轮替换C及资格信用均不变。

审计payload的selected_experiences/candidate_experience继续分别保存，避免渲染修改固定E快照或失去证据状态。合并的是实际system/user中的user经验列表，不把候选提前写入Memory，也不累计旧候选。

## 验证与历史

- 无E+C、E+C两种回归验证实际消息只有一份列表、C出现一次、C位于E后、编号映射有效且payload不被修改。
- 实际HTTP最坏4次候选/2次应用回归检查统一列表及固定E/C对应编号，合法来源、替换与Fusion保持。
- 全套232项测试通过（原230加2个参数化案例），`git diff --check`通过；无真实模型请求。
- 计划32已完成的100条是在旧分区消息下运行，5个manifest/轨迹/Memory/角色/请求产物哈希保持不变，不回写旧消息、不重复运行。此改动会改变Python源码指纹，旧检查点不能作为同源码续跑。
- 文档已同步到工程设计和Generator模块；未提交或push。

## 后续角色术语统一

用户进一步要求将各角色提示词中的rubric改为“评分标准”。已替换Critic、Admission、Fusion职责文本中的全部6处自然语言rubric；其余角色原本已使用中文标准名称。JSON字段、内部类型/变量与ID命名不改，也不回写任务原文、历史候选或已完成实验日志。已有角色提示词测试现在检查所有角色的实际消息没有英文rubric，完整232项测试通过；未发送模型请求、提交或push。
