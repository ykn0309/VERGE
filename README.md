# VERGE

Verified Experience-guided Rubric Generation：从偏好轨迹提炼评分标准设计经验，通过后续应用维护与修订。

## 当前状态

V2核心链路已实现：半结构化原子经验、直接混合检索、经验核对准入、来源验证与独立执行Checker、Q/C贝叶斯网络和Keep/Revise配对回放。参数估计实验尚未开展；无可用参数时Q显示未估计，不使用假设概率决策。实现与验证进度见[PLAN](PLAN.md)。

## 文档

- [完整设计方案](docs/设计方案.md)
- [工程设计](docs/工程设计.md)
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
python -m pytest -q
python scripts/replay_v2_modules.py --prepare
python scripts/replay_v2_modules.py --workers 4
```

模块复测准备依赖隔离保留的历史来源，转换记录在各输入中；新输出不进入其他模块或经验池，不是完整流水线实验。旧Memory/Trajectory不兼容V2；运行产物、数据和密钥不纳入Git。

参数内容或状态变更必须在新目录显式重算，不复用旧后验：

```bash
python -m verge.cli reestimate --run-dir runs/已认证运行 --parameter-file 新参数.json --output-dir runs/重估副本
```

重估只重建当前Memory的Q，不调用LLM、不改原产物或历史评价；原Trajectory与修订快照保留当时参数下的Q。审查修复与验证见[计划71](docs/plans/71-V2审查修复.md)。
