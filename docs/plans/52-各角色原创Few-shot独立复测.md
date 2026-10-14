# 计划52：各角色原创Few-shot独立复测

2026-10-13，用户授权将已确定的任务分工给子代理：各角色从过去日志挑选表现不佳输入，独立小规模复测，不运行完整流水线。教学示例必须独立构造，不从数据集提取或改写。

## 范围

- Updater、Generator各4组；Selector、Judge、Critic各3组，总17组。每组当前baseline和追加few-shot各一次，正常34次调用，沿用有限格式/API和结构纠错，不扩大样本或重试预算。
- 所有真实调用固定Qwen3.5-Flash、关闭思考；每例互相独立，不把新输出反馈到其他案例。不新增analysis、支持更新字段、验收角色、经验信用或容量机制；旧文本锚定对照不在本轮范围。
- 3个子代理分别负责Updater、Generator、Selector/Judge/Critic；主代理负责工具、契约检查、语义复核和汇总。各角色产物独立存入`runs/module-few-shot-replay/<role>/`。

## 输入与产物

`scripts/replay_role_few_shot.py`按selection.json的一基trace_line提取100条源运行的成功实际请求。保留完整task、回答、经验和真实Critic历史，除system替换为当前提示词及few-shot外，其余实际消息保持不变。旧Updater请求含长sample_id，按计划51投影为短编号并明确记录转换；两变体转换完全相同，不做截断。

每角色的few_shot.json保存2–3个独立构造input/output示例，保持现有输出结构。示例只在独立实验system追加，尚未写入生产提示词；结构通过不替代对示例语义的检查。

复测保存源输入、实际消息、历史及新输出、原始回复、模型配置、提示词/脚本指纹、耗时、请求记录和错误。源manifest、Memory、role_io、trajectories以及selection/examples哈希核验不变。基础设施或程序错误fail-fast，角色格式/结构耗尽记录无效，不回写源池。

## 结果解释

检查针对原问题的变化和正常行为退化，不以少选、少拆分或整体偏好分差代替模块质量。历史旧审查仅作线索，不能当金标；每变体一次，只说明这些固定输入的局部行为，不说明链路准确率、泛化或因果效果。具体结果与上线取舍见[汇总报告](../实验分析/历史运行归档/module-few-shot-replay/实验简报.md)。
