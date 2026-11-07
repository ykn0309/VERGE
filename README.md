# VERGE

研究状态与探索路径：[研究进度看板](docs/研究进度.html) · [当前计划](PLAN.md)。

Verified Experience-guided Rubric Generation：从偏好轨迹提炼评分标准设计经验，通过后续应用维护与修订。

## 当前状态

2026-11-06：正在重设计。已将确认的有／无经验对照、历史效果选择、Critic直接提出经验与版本验证机制整理到[新设计方案](docs/设计方案.md)，尚未实现。现有代码仍是[旧实现](docs/历史设计/2026-11-06-旧实现/README.md)，下面命令用于旧协议。此前200题对照无经验50.5%、有经验46.0%，不能作为新设计的验证结果。

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
python -m verge.cli run --config configs/helpsteer3-general-role-guides-qwen-flash-20.yaml
python -m verge.cli inspect --run-dir runs/v2-fake --sample-id demo-train-1
python -m verge.web --runs-dir runs --port 8765
```

仅含train的运行明确标记training_only/训练完成，不声称完成held-out评价；配置test时训练完成后才进行冻结评价。CLI的`--stop-after-train`须为正数。

真实调用使用配置指定的环境变量，不在文件中填写密钥。默认Fake无需API；示例输出目录必须不存在。完整实验需明确授权，不因实现完成自动启动。

```bash
python -m pytest -q tests
python scripts/validate_protocol100.py prepare --output artifacts/独立复测新目录
python scripts/validate_protocol100.py run --output artifacts/独立复测新目录
```

当前模块脚本依赖本机保留的历史真实轨迹与固定输入，输出目录必须为新目录；历史脚本仅作对应版本参考。输入转换与覆盖不足记录在各产物中；新输出不进入其他模块或经验池，不是完整流水线实验。旧Memory/Trajectory不兼容V2；运行产物、数据和密钥不纳入Git。

参数内容或状态变更必须在新目录显式重算，不复用旧后验：

```bash
python -m verge.cli reestimate --run-dir runs/已认证运行 --parameter-file 新参数.json --output-dir runs/重估副本
```

重估只重建当前Memory的Q，不调用LLM、不改原产物或历史评价；原Trajectory与修订快照保留当时参数下的Q。审查修复与验证见[计划71](docs/plans/71-V2审查修复.md)。

样本持续并行通过run.train_workers配置，本轮试用5；revision_workers默认1、max_inflight_requests默认16。Ctrl+C停止补题并排空在途任务后认证暂停。意外中断时，只有完整结果已持久化且可认证才可恢复：

```bash
python -m verge.cli recover --run-dir runs/中断运行 --output-dir runs/恢复副本
```

该命令不重跑模型调用；缺少完整结果时安全拒绝，保留原目录。Web目前支持Memory15、Trajectory/Manifest17，旧格式仅只读展示。

新的20题示例默认5样本持续补位，不附虚构可用参数表；无校准表时Q未估计。本轮带假设表的实际配置保存在各运行目录，不能与默认配置混称。
