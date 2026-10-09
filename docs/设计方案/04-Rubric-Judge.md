# Rubric Judge

> 上位文档：[VERGE 设计总览](../设计方案.md)
>
> 输入：[Initial Rubric Generation](./03-初始Rubric生成.md)

## 1. 模块职责

归一化 rubric 的 importance scores，并分别评价 chosen 和 rejected response，输出 criterion-level Boolean judgments、加权分数和 margin。

这是流程中第一次处理 preference pair 的阶段。工程上对两份回答分别发起独立的 Judge 调用，并行执行；每次模型接收原始 `task`（指令、用户提供的输入与上下文）、待判断标准和单份 `response`，返回每条标准的 `satisfied` Boolean。原始task是必需输入，适用于初始、新会话应用、测试和独立baseline判断（历史对话修订同样要求）；外部链接内容不会自动抓取或补造。chosen/rejected 的对应关系仅由系统维护，不进入 Judge 模型输入。两侧都校验成功后，程序合并判断并计算分数；一侧重试时保留另一侧成功结果。原有判断复用规则不变。

## 2. Weight Normalization

设 criterion 的 importance score 为 $s_i$。对于完整 rubric $R$：

$$
\omega_i(R)=\frac{s_i}{\sum_{j\in R}s_j}
$$

因此：

$$
\omega_i(R)>0,
\qquad
\sum_{i\in R}\omega_i(R)=1
$$

`importance_score` 保存在 rubric 中；`normalized_weight` 是 Calculator 针对当前完整 rubric 计算的派生值。

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

Tie 在 V1 中视为 failure。

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

新方案沿用既有Judge评分、独立两侧请求和可用的相同问题判断复用，不改变Judge提示词。跨本样本有效rubric比较时，相同问题可沿用已知判断，即使重要性或ID变化；它只影响评价复用，不使Generator获得旧rubric或判断。若已有相同问题却有互相矛盾的判断，不凭文本猜测答案：只按未变ID和问题复用，其余重新Judge。新调用接口的复用基准由执行层记录，见[计划31](../plans/31-Critic候选经验准入与新会话验证.md)。模型只看到本次待判断标准的 `"1"`、`"2"`……短编号；内部 ID 由适配层还原。模型漏答、重复编号、返回未知编号或无效布尔值时有限重试；仍失败则写入显式 Judge 错误轨迹、跳过该样本后续信用与经验生成，不把未知结果计为失败或成功。网络/鉴权等后端错误仍中止运行。
