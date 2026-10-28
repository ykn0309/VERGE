# VERGE

Verified Experience-guided Rubric Generation：从偏好轨迹提炼评分标准设计经验，通过后续应用维护与修订。

## 当前状态

V2核心链路已实现：适用条件＋核验方法的双字段原子经验、直接混合检索、经验核对准入、来源验证与独立执行Checker、Q/C贝叶斯网络和Keep/Revise配对回放。正式参数估计尚未开展；无可用参数时Q显示未估计。中期预实验曾经用户授权启用明确标注的人工假设参数，不代表参数已校准。批次并行训练及冻结对照已运行，见[PLAN](PLAN.md)。经验利用改进候选仅在独立复测中运行，未替换生产提示词或经验池。

## 文档

- [完整设计方案](docs/设计方案.md)
- [工程设计](docs/工程设计.md)
- [当前问题与代码审查提示词](docs/研究讨论/经验表示与利用问题-外部模型咨询提示词.md)
- [经验利用独立复测结果](docs/实验分析/93-经验利用问题审查与独立复测.md)
- [计划70：实现与独立模块复测](docs/plans/70-V2核心链路与独立模块验证.md)
- [V1历史设计](docs/历史设计/2026-10-24-V1/设计方案.md)

## 运行

需要Python ≥3.10，推荐3.12（本机可用`/opt/anaconda3/bin/python`）；系统Python 3.9不支持。

```bash
python -m pip install -r requirements.txt
python -m verge.cli run --config configs/default.yaml
python -m verge.cli run --config configs/helpsteer3-v2-qwen-flash.yaml
python -m verge.cli inspect --run-dir runs/v2-fake --sample-id demo-train-1
python -m verge.web --runs-dir runs --port 8765
```

仅含train的运行明确标记training_only/训练完成，不声称完成held-out评价；配置test时训练完成后才进行冻结评价。CLI的`--stop-after-train`须为正数。

真实调用使用配置指定的环境变量，不在文件中填写密钥。默认Fake无需API；示例输出目录必须不存在。完整实验需明确授权，不因实现完成自动启动。

```bash
python -m pytest -q tests
python scripts/replay_trigger_principle.py --output artifacts/独立复测新目录
```

当前模块复测脚本读取保留的计划83真实运行，输出目录必须不存在；旧复测脚本依赖的日志已按用户要求清理，作为历史实现参考。输入转换与覆盖不足记录在各产物中；新输出不进入其他模块或经验池，不是完整流水线实验。旧Memory/Trajectory不兼容V2；运行产物、数据和密钥不纳入Git。

参数内容或状态变更必须在新目录显式重算，不复用旧后验：

```bash
python -m verge.cli reestimate --run-dir runs/已认证运行 --parameter-file 新参数.json --output-dir runs/重估副本
```

重估只重建当前Memory的Q，不调用LLM、不改原产物或历史评价；原Trajectory与修订快照保留当时参数下的Q。审查修复与验证见[计划71](docs/plans/71-V2审查修复.md)。
