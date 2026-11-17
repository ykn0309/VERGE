"""Archive prior evidence and publish experiments 127 and 128 as plain text."""

import hashlib
import json
from pathlib import Path
import shutil

from verge.memory import ExperienceMemory
from verge.models import TrajectoryRecord


LATEST = Path("logs/latest")
ARCHIVE = Path("logs/archive/2026-11-17-before127-128")
SMALL_RUN = Path("runs/reference127")
SMALL_ARTIFACTS = Path("artifacts/reference127")
FULL_RUN = Path("runs/reference500-128")
FULL_ARTIFACTS = Path("artifacts/reference500-128")
PART_BYTES = 512 * 1024


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def archive_previous():
    if not LATEST.exists():
        return []
    if ARCHIVE.exists():
        index = json.loads((LATEST / "archive-index.json").read_text())
        assert index["previous_git_commit"] == "aaa01c6"
        return index["moves"]
    files = [
        (str(path.relative_to(LATEST)), sha256(path), path.stat().st_size)
        for path in sorted(LATEST.rglob("*"))
        if path.is_file()
    ]
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(LATEST), str(ARCHIVE / "logs" / "latest"))
    archived = ARCHIVE / "logs" / "latest"
    assert all(
        sha256(archived / relative) == digest for relative, digest, _ in files
    )
    return [
        {
            "original": str(LATEST),
            "archive": str(archived),
            "files": len(files),
            "bytes": sum(size for _, _, size in files),
            "all_hashes_verified": True,
        }
    ]


def text_chunks(data, path):
    if len(data) <= PART_BYTES:
        return [data]
    if path.suffix == ".jsonl":
        chunks = []
        chunk = b""
        for line in data.splitlines(keepends=True):
            if chunk and len(chunk) + len(line) > PART_BYTES:
                chunks.append(chunk)
                chunk = b""
            chunk += line
        if chunk:
            chunks.append(chunk)
        return chunks
    chunks = []
    remaining = data
    while remaining:
        cut = min(PART_BYTES, len(remaining))
        while cut and cut < len(remaining) and remaining[cut] & 0xC0 == 0x80:
            cut -= 1
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    return chunks


def publish_raw_files():
    manifest = []
    sources = (SMALL_RUN, SMALL_ARTIFACTS, FULL_RUN, FULL_ARTIFACTS)
    for source_root in sources:
        for source in sorted(source_root.rglob("*")):
            if (
                not source.is_file()
                or "__pycache__" in source.parts
                or source.suffix == ".pyc"
                or source.name == ".DS_Store"
            ):
                continue
            data = source.read_bytes()
            data.decode("utf-8")
            destination = LATEST / "raw" / source
            chunks = text_chunks(data, source)
            if len(chunks) == 1:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(chunks[0])
                part_paths = [destination]
            else:
                parts_dir = Path(f"{destination}.parts")
                parts_dir.mkdir(parents=True, exist_ok=True)
                part_paths = []
                for number, chunk in enumerate(chunks, 1):
                    part_path = parts_dir / f"{number:04d}.txt"
                    part_path.write_bytes(chunk)
                    part_paths.append(part_path)
            assert b"".join(part.read_bytes() for part in part_paths) == data
            manifest.append(
                {
                    "source": str(source),
                    "bytes": len(data),
                    "sha256": sha256(source),
                    "parts": [
                        {
                            "path": str(part.relative_to(LATEST)),
                            "sha256": sha256(part),
                        }
                        for part in part_paths
                    ],
                }
            )
    write_json(
        LATEST / "manifest.json",
        {
            "files": manifest,
            "all_reconstructed_hashes_verified": True,
        },
    )
    return manifest


def publish_experiences(state):
    pool = list(state.experiences.values())
    write_json(LATEST / "experience_pool.json", [item.to_dict() for item in pool])
    sample_id_by_task = {
        record.sample.task_id: record.sample.sample_id
        for record in state.trajectories.values()
    }
    lines = [
        "# 入池经验索引",
        "",
        "来源验证只允许候选进入经验池，不代表经验可靠。",
        "",
        "| 编号 | 经验ID与正文 | 来源样本 | 后续引用任务数 |",
        "|---|---|---|---:|",
    ]
    for number, experience in enumerate(pool, 1):
        name = f"E{number:02d}.json"
        write_json(LATEST / "experiences" / name, experience.to_dict())
        source_sample = sample_id_by_task[experience.source_task_id]
        reference_count = len(
            state.clusters.references("experience", experience.experience_id)
        )
        lines.append(
            f"| E{number:02d} | [{experience.experience_id}]({name}) "
            f"| {source_sample} | {reference_count} |"
        )
    (LATEST / "experiences" / "README.md").write_text("\n".join(lines) + "\n")


def publish_samples(state, run_root, artifacts_root, folder):
    training = {
        record.sample.sample_id: record for record in state.trajectories.values()
    }
    ordering = [
        json.loads(line)["sample_id"]
        for line in (artifacts_root / "train.jsonl").read_text().splitlines()
    ]
    tests = [
        TrajectoryRecord.from_dict(
            {
                key: value
                for key, value in json.loads(line).items()
                if key != "sample_id"
            }
        )
        for line in (run_root / "test.jsonl").read_text().splitlines()
    ]
    records = [training[sample_id] for sample_id in ordering] + tests
    lines = [
        "# 逐题轨迹索引",
        "",
        "训练经验路独立试用；冻结测试按推理协议只运行选定路径。",
        "",
        "| 编号 | 阶段 | 样本ID | 完整轨迹 | 初始路径结果 | Critic轮数 |",
        "|---|---|---|---|---|---:|",
    ]

    def result(path):
        if path.error or not path.judgment:
            return "未知"
        return "对" if path.judgment.success else "错"

    for number, record in enumerate(records, 1):
        phase = "train" if number <= len(ordering) else "test"
        phase_number = number if phase == "train" else number - len(ordering)
        name = f"{phase}-{phase_number:03d}.json"
        write_json(LATEST / "samples" / folder / name, record.to_dict())
        paths = "；".join(
            (
                "无经验"
                if not path.provided_experience_ids
                else path.provided_experience_ids[0]
            )
            + ":"
            + result(path)
            for path in record.paths
        )
        lines.append(
            f"| {number} | {phase} | {record.sample.sample_id} "
            f"| [{name}]({name}) | {paths or '任务未知'} "
            f"| {len(record.critic_rounds)} |"
        )
    (LATEST / "samples" / folder / "README.md").write_text(
        "\n".join(lines) + "\n"
    )
    return len(records)


def publish_cluster_audit():
    source = FULL_ARTIFACTS / "cluster-audit"
    target = LATEST / "cluster-audit128"
    for path in source.rglob("*"):
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    shutil.copy2(
        FULL_ARTIFACTS / "audit_clusters.py",
        target / "audit_clusters.py",
    )


def publish():
    full_manifest = json.loads((FULL_RUN / "manifest.json").read_text())
    small_manifest = json.loads((SMALL_RUN / "manifest.json").read_text())
    assert full_manifest["status"] == small_manifest["status"] == "completed"
    moves = archive_previous()
    LATEST.mkdir(parents=True, exist_ok=True)
    write_json(
        LATEST / "archive-index.json",
        {
            "previous_git_commit": "aaa01c6",
            "moves": moves,
            "older_archive": "logs/archive/2026-11-16-before125",
        },
    )
    raw_files = publish_raw_files()
    state = ExperienceMemory.load(FULL_RUN / "checkpoint.json")
    publish_experiences(state)
    sample_count = publish_samples(
        ExperienceMemory.load(SMALL_RUN / "checkpoint.json"),
        SMALL_RUN,
        SMALL_ARTIFACTS,
        "reference127",
    )
    sample_count += publish_samples(
        state,
        FULL_RUN,
        FULL_ARTIFACTS,
        "reference128",
    )
    publish_cluster_audit()
    (LATEST / "README.md").write_text(
        """# 最新实验127／128：双侧引用聚类验证

- [实验128报告](../../docs/实验分析/128-双侧引用聚类五百题冻结验证.md)
- [实验127报告](../../docs/实验分析/127-双侧引用聚类实现与少样本验证.md)
- [实验127逐题轨迹](samples/reference127/README.md)
- [实验128逐题轨迹](samples/reference128/README.md)
- [69条经验索引](experiences/README.md)及[完整经验池](experience_pool.json)
- [实验128分组与引用审计](cluster-audit128/statistics.json)
- [原始请求回复与测试对照](raw/runs/)；[输入、摘要及分析](raw/artifacts/)
- [原文件与分段摘要](manifest.json)：全部UTF-8文本，按parts顺序拼接可恢复原字节，没有压缩包。

实验127为20题训练＋5题冻结测试，实验128为500题训练＋200题冻结测试及同题无经验对照。全部失败与有限重试记录保留。旧125日志已无损归档至忽略的`logs/archive/2026-11-17-before127-128`；[归档索引](archive-index.json)。Git历史`aaa01c6`仍可读取原125文本日志。
"""
    )
    print(
        json.dumps(
            {
                "archived": moves,
                "published_raw_files": len(raw_files),
                "experiences": len(state.experiences),
                "samples": sample_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    publish()
