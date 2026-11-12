# 最新实验日志

本目录随Git保存最近一次完整实验：500题四路训练120，以及基于同一检查点的聚类审查121、114题内容审查122。包括原始请求／回复、全部失败、检查点、输入、代码快照、阶段统计和审查记录；未修改原始产物。

- `experiment-120-122.tar.gz.part-*`：完整日志的5个分卷，保留原仓库相对路径。
- `manifest.json`：压缩包及669个原始文件的大小、SHA256与完整性校验结果。
- `archive-index.json`：旧日志原路径、归档路径及逐文件SHA256；归档内容本身不上传。

在仓库根目录恢复日志：

```sh
cat logs/latest/experiment-120-122.tar.gz.part-* > /tmp/verge-experiment-120-122.tar.gz
shasum -a 256 /tmp/verge-experiment-120-122.tar.gz
# 将结果与manifest.json中的bundle.sha256核对后解压。
tar -xzf /tmp/verge-experiment-120-122.tar.gz
```

恢复后的目录：

```text
runs/history-learning-general500-subset120/
artifacts/general500-subset120/
artifacts/general500-cluster-audit121/
artifacts/general500-case-audit122/
```

本地仍保留上述未压缩文件，便于WebUI读取。`runs/`下的复用输入数据集保留原路径，继续忽略。旧实验及模块诊断移到`logs/archive/2026-11-12/{runs,artifacts}/`，未删除；历史报告中的旧路径可按`archive-index.json`定位。需要复查旧实验时，从归档将对应目录恢复到原位置。

实验代码固定为`6ff0e57`；当前代码不代表当时运行快照。压缩只改变存储形式，不重新运行或修正实验。结论见[总体报告120](../../docs/实验分析/120-General500题四路流程实测.md)、[聚类审查121](../../docs/实验分析/121-500题多成员聚类内容审查.md)、[内容分析122](../../docs/实验分析/122-改善与退化114题内容分析.md)。

下次实验完成后，将本目录的上一轮文件移入忽略的归档区，再保存新一轮日志和校验清单。
