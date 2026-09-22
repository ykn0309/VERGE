# Rubric Judge

> 上位文档：[VERGE 设计总览](../设计方案.md)
>
> 输入：[Initial Rubric Generation](./03-初始Rubric生成.md)

## 1. 模块职责

归一化 rubric 的 importance scores，并分别评价 chosen 和 rejected response，输出 criterion-level Boolean judgments、加权分数和 margin。

这是流程中第一次允许查看 preference pair 的阶段。

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

同一 revision comparison 中，内容未变化的 criterion 应复用已有 Boolean judgment，避免 Judge 随机性制造虚假 operation contribution。
