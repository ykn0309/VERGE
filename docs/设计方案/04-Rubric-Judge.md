# Rubric Judge

> **当前实现：计划68信用继承与稳定性惩罚，Manifest/Trajectory10、Memory8。** 演化修订继承历史信用，拆分均分，惩罚为最近5例累计(P+N)的20%；详见[计划68](../plans/68-演化信用继承与稳定性惩罚.md)。其余方法承接[方案14](14-原子经验版本演化与局部混合检索.md)；计划67局部结果保留其原协议解释，见[报告67](../实验分析/67-版本演化与混合检索局部验证.md)。最近完整训练仍为计划66。

> 上位文档：[VERGE 设计总览](../设计方案.md)
>
> 输入：[Initial Rubric Generation](./03-初始Rubric生成.md)

## 1. 模块职责

归一化 rubric 的 相对权重，并分别评价 chosen 和 rejected response，输出 criterion-level Boolean judgments与依据、加权分数和 margin。

这是流程中第一次处理 preference pair 的阶段。工程上对两份回答分别发起独立的 Judge 调用，并行执行；每次模型接收原始 `task`（指令、用户提供的输入与上下文）、待判断标准和单份 `response`，返回每条标准的 `satisfied` Boolean和必填非空`reason`，简短说明回答中的依据、缺失信息或计数结果。原始task是必需输入，适用于初始、原子候选集合的新会话应用、测试和独立baseline判断；外部链接内容不会自动抓取或补造。chosen/rejected 的对应关系仅由系统维护，不进入 Judge 模型输入。两侧都校验成功后，程序合并判断并计算分数；一侧重试时保留另一侧成功结果。原有判断复用规则不变。

## 2. Weight Normalization

设 criterion 的 原始相对权重 为 $s_i$。对于完整 rubric $R$：

$$
\omega_i(R)=\frac{s_i}{\sum_{j\in R}s_j}
$$

因此：

$$
\omega_i(R)>0,
\qquad
\sum_{i\in R}\omega_i(R)=1
$$

`weight` 保存在 rubric 中；`normalized_weight` 是 Calculator 针对当前完整 rubric 计算的派生值。

## 3. Criterion-level Judgment

$$
J(y,c_i)\in\{0,1\}
$$

$$
d_i=J(y^+,c_i)-J(y^-,c_i)
\in\{-1,0,1\}
$$

- $d_i=1$：criterion 支持 gold preference；
- $d_i=0$：没有区分力；
- $d_i=-1$：方向与 gold preference 相反。

Boolean decomposition 便于解释和细粒度归因；相关思路可参见 [CheckEval](https://arxiv.org/abs/2403.18771)。

## 4. 加权分数与 Margin

$$
S_R(y)=\sum_{i\in R}\omega_i(R)J(y,c_i)
$$

$$
M(R)=S_R(y^+)-S_R(y^-)
=\sum_{i\in R}\omega_i(R)d_i
$$

判断规则：

```text
M(R) > 0  → success
M(R) ≤ 0  → failure
```

Tie视为failure。程序以原始weight数字的十进制表示计算精确比例和margin符号，最终序列化为浮点数；不通过epsilon引入成功阈值，0.1+0.2对0.3保持精确平分。

## 5. 输出

```text
RubricJudgment {
    rubric_id
    normalized_weights
    criterion_judgments
    chosen_score
    rejected_score
    margin
    success
}
```

每条合并判断保存`criterion_id/chosen/rejected/chosen_reason/rejected_reason`，两侧依据与Boolean严格对应，传给Critic并保存轨迹。依据仅供核对，不视为事实认证。Generator仍不可见判断与理由。

新方案沿用加权Boolean评分、独立两侧请求和可用的相同问题判断复用；复用时原理由一同保留。跨本样本有效rubric比较时，相同问题可沿用已知判断，即使权重或ID变化；它只影响评价复用，不使Generator获得旧rubric或判断。若已有相同问题却有互相矛盾的判断，不凭文本猜测答案：只按未变ID和问题复用，其余重新Judge。新调用接口的复用基准由执行层记录，见[计划44](../plans/44-相对权重与Judge依据反馈.md)。模型只看到本次待判断标准的 `"1"`、`"2"`……短编号；内部 ID 由适配层还原。模型漏答、重复编号、返回未知编号、无效布尔值或缺失/空理由时有限重试；仍失败则写入显式 Judge 错误轨迹、跳过该样本后续信用与经验生成，不把未知结果计为失败或成功。网络请求重试耗尽后按[计划63](../plans/63-请求失败样本隔离与经验排序.md)记录样本错误并继续，有效R0证据保留；鉴权/配置等全局错误仍中止运行。
