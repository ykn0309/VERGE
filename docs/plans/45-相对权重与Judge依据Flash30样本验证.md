# 45 — 相对权重与Judge依据 Flash 30样本验证

## 1. 授权与目的

用户要求提交当前代码并运行一些样本查看效果。使用新协议在计划43的有序训练前30条上进行一次空池运行，结果用于检查实际输出、请求错误、评分与经验质量。相同前缀便于逐样本定位变化，但模型生成随机性与池演化不受控，汇总差异不能解释为改动的因果效应。

## 2. 冻结配置

配置：[rewardbench-chat-hard-qwen-weighted-judge-reasons-flash-30.yaml](../../configs/rewardbench-chat-hard-qwen-weighted-judge-reasons-flash-30.yaml)。独立运行目录`runs/rewardbench-chat-hard-qwen-weighted-judge-reasons-flash-30`；`stop_after_train: 30`、seed42、单个测试worker。所有角色Qwen3.5-Flash、关闭思考；timeout30秒、最多2次网络重试；Top20检索、Selector最多5条、exploration0；最多2轮应用、格式/结构纠错2次。数据为既有完整训练准备集，有序前缀与计划43相同；不执行test或后续426条。

数据SHA256：`da4a65ac5dcc93fb3b613fc6179875fc9eeb1b89ed9603a771286a7363c9d8d8`。运行前提交代码及配置，以提交版本作为源码边界；保留当前原始运行与计划43原始产物。

## 3. 复核口径

完成后记录运行终态、30条前缀、初始/最终成功、候选/应用/Fusion、已有经验选择与下游证据、请求/格式错误和tokens。核对相对权重及Judge两侧理由在真实请求与轨迹中一致；抽看2857与计数/事实错误样本，以及入池经验的trigger、设计动作与实际来源信用。对照计划43时标注非因果限制。

状态：待运行。
