"""Rolling snapshot workers; only the coordinator writes the shared ledger."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextvars import copy_context
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib, json, time
from uuid import uuid4
from verge.memory import ExperienceMemory, TrajectoryLog, atomic_json
from verge.models import Experience, RevisionRecord, TrajectoryRecord


def sha(data):
    return hashlib.sha256(data).hexdigest()


def recover_sample_commit(memory, log):
    """Finish only a hash-certified pending sample commit; never rerun its LLM calls."""
    path = memory.root_dir / "pending_sample_commit.json"
    if not path.exists():
        return
    pending = json.loads(path.read_text())
    current = sha(memory.path.read_bytes())
    # Canonical JSON hash is independent of atomic_json's whitespace.
    post = pending["memory"]
    current_data = json.loads(memory.path.read_text())
    if current != pending["before_sha256"] and current_data != post:
        raise ValueError(
            "pending sample commit has an unrelated ledger; manual recovery required"
        )
    if current_data != post:
        versions = {
            e["experience_id"]: Experience.from_dict(e) for e in post["experiences"]
        }
        revisions = [RevisionRecord.from_dict(e) for e in post["revisions"]]
        memory._commit(versions, revisions)
    rec = TrajectoryRecord.from_dict(pending["record"])
    before_size = pending["trajectory_size"]
    before_hash = pending["trajectory_sha256"]
    raw = log.path.read_bytes() if log.path.exists() else b""
    prefix = raw[:before_size]
    if len(prefix) != before_size or sha(prefix) != before_hash:
        raise ValueError("pending sample trajectory prefix does not match")
    tail = raw[before_size:]
    expected = (rec.to_json() + "\n").encode()
    if tail and not expected.startswith(tail):
        raise ValueError("pending sample has unrelated trajectory tail")
    if tail != expected:
        import os, tempfile

        log.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=log.path.parent, delete=False) as f:
            name = f.name
            f.write(prefix + expected)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(name, log.path)
        finally:
            if Path(name).exists():
                Path(name).unlink()
    path.unlink()


def commit_sample(memory, log, record, added):
    """Stage every admission and quality update, then publish one sample transaction."""
    staged_at = time.monotonic()
    with TemporaryDirectory(prefix="verge-commit-") as root:
        Path(root, "experience_memory.json").write_bytes(memory.path.read_bytes())
        staged = ExperienceMemory(root, parameters=memory.parameters)
        for e in added:
            staged.add(e, e.source_sample, e.admission_review)
        for app in record.applications:
            staged.record(app)
        for attempt in record.proposal_attempts:
            for app in attempt.applications:
                staged.record(app)
        post = json.loads(staged.path.read_text())
        record.timings["coordinator_stage_seconds"] = time.monotonic() - staged_at
        atomic_json(
            memory.root_dir / "pending_sample_commit.json",
            {
                "before_sha256": sha(memory.path.read_bytes()),
                "memory": post,
                "record": record.to_dict(),
                "trajectory_size": log.path.stat().st_size if log.path.exists() else 0,
                "trajectory_sha256": sha(
                    log.path.read_bytes() if log.path.exists() else b""
                ),
            },
        )
    recover_sample_commit(memory, log)


def run_training_stream(
    samples,
    modules_factory,
    memory,
    log,
    *,
    workers=5,
    revision_workers=1,
    runner_options=None,
    on_complete=None,
    stop_event=None,
):
    from verge.pipeline import PipelineRunner, revision_reasons

    if (
        type(workers) is not int
        or workers < 1
        or type(revision_workers) is not int
        or revision_workers < 1
    ):
        raise ValueError("positive worker counts required")
    options = dict(runner_options or {})
    recover_sample_commit(memory, log)
    completed = {r.sample_id for r in log.all()}
    input_samples = list(samples)
    if len({s.sample_id for s in input_samples}) != len(input_samples):
        raise ValueError("duplicate sample inputs")
    samples = iter(s for s in input_samples if s.sample_id not in completed)
    results = []
    running = {}
    revisions = {}
    queued = set()
    jobs = []
    sample_jobs_path = memory.root_dir / "sample_jobs.json"
    sample_jobs = []
    queue_path = memory.root_dir / "revision_jobs.json"
    if queue_path.exists():
        previous = json.loads(queue_path.read_text())
        for job in previous:
            if job["state"] != "running":
                continue
            result_path = (
                memory.root_dir / "revision_results" / (job["job_id"] + ".json")
            )
            if not result_path.exists():
                raise ValueError(
                    "unfinished revision has no durable result; refusing duplicate calls"
                )
            event = RevisionRecord.from_dict(json.loads(result_path.read_text()))
            if event.event_id not in {r.event_id for r in memory.revisions}:
                if memory.get(event.parent_id).status == "active":
                    memory.finish_revision(event)
                else:
                    job["state"] = "stale_parent"
            if job["state"] == "running":
                job["state"] = "completed"
                job["event_id"] = event.event_id
        jobs = previous
        atomic_json(queue_path, jobs)
    sequence = sum(r.split == "train" for r in log.all())

    def work(sample, snapshot):
        with TemporaryDirectory(prefix="verge-stream-") as root:
            Path(root, "experience_memory.json").write_bytes(snapshot)
            local = ExperienceMemory(root, parameters=memory.parameters)
            before = {e.experience_id for e in local.all()}
            modules, cleanup = modules_factory(sample)
            try:
                runner = PipelineRunner(
                    modules,
                    local,
                    TrajectoryLog(root),
                    **{**options, "revision_enabled": False},
                )
                rec = runner.run_sample(sample)
                rec.snapshot_digest = sha(snapshot)
                added = [e for e in local.all() if e.experience_id not in before]
                atomic_json(
                    memory.root_dir
                    / "sample_results"
                    / (sha(sample.sample_id.encode()) + ".json"),
                    {"record": rec.to_dict(), "added": [e.to_dict() for e in added]},
                )
                return rec, added
            finally:
                cleanup()

    def revision_work(e, snapshot, trigger, reasons, job_id):
        with TemporaryDirectory(prefix="verge-revision-") as root:
            Path(root, "experience_memory.json").write_bytes(snapshot)
            local = ExperienceMemory(root, parameters=memory.parameters)
            modules, cleanup = modules_factory(e.source_sample)
            try:
                runner = PipelineRunner(modules, local, TrajectoryLog(root), **options)
                runner.replay_factory = modules_factory
                event = runner._revise(e, trigger, reasons, event_id=job_id)
                atomic_json(
                    memory.root_dir / "revision_results" / (job_id + ".json"),
                    event.to_dict(),
                )
                return event
            finally:
                cleanup()

    def fill(pool):
        while len(running) < workers and not (stop_event and stop_event.is_set()):
            try:
                sample = next(samples)
            except StopIteration:
                return
            snap = memory.path.read_bytes()
            sample_jobs.append(
                {
                    "sample_id": sample.sample_id,
                    "snapshot_digest": sha(snap),
                    "state": "running",
                }
            )
            atomic_json(sample_jobs_path, sample_jobs)
            running[pool.submit(copy_context().run, work, sample, snap)] = (
                sample.sample_id
            )

    def queue_revisions(pool, trigger):
        if not options.get("revision_enabled", True) or (
            stop_event and stop_event.is_set()
        ):
            return
        # A bounded revision pool; excess due versions are reconsidered after a completion.
        for e in memory.all(active_only=True):
            if len(revisions) >= revision_workers:
                break
            if e.experience_id in queued:
                continue
            reasons = revision_reasons(
                e,
                interval=options.get("interval", 3),
                window=options.get("window", 6),
                errors=options.get("error_threshold", 3),
                q_threshold=options.get("q_threshold", 0.5),
            )
            if reasons is None:
                continue
            memory.mark_checked(e.experience_id, reasons, options.get("window", 6))
            if not reasons:
                continue
            queued.add(e.experience_id)
            snap = memory.path.read_bytes()  # e retains the unconsumed trigger evidence
            job = {
                "job_id": "revision_" + uuid4().hex,
                "parent_id": e.experience_id,
                "state": "running",
                "snapshot_digest": sha(snap),
                "trigger_sample_id": trigger,
                "reasons": reasons,
            }
            jobs.append(job)
            atomic_json(queue_path, jobs)
            f = pool.submit(
                copy_context().run,
                revision_work,
                e,
                snap,
                trigger,
                reasons,
                job["job_id"],
            )
            revisions[f] = job

    with (
        ThreadPoolExecutor(max_workers=workers) as sample_pool,
        ThreadPoolExecutor(max_workers=revision_workers) as revision_pool,
    ):
        fill(sample_pool)
        while running or revisions:
            done, _ = wait([*running, *revisions], return_when=FIRST_COMPLETED)
            for f in done:
                if f in running:
                    running.pop(f)
                    rec, added = f.result()
                    sequence += 1
                    rec.completion_sequence = sequence
                    commit_sample(memory, log, rec, added)
                    next(j for j in sample_jobs if j["sample_id"] == rec.sample_id)[
                        "state"
                    ] = "completed"
                    atomic_json(sample_jobs_path, sample_jobs)
                    results.append(rec)
                    if on_complete:
                        on_complete(rec, sequence)
                    queue_revisions(revision_pool, rec.sample_id)
                    fill(sample_pool)
                else:
                    job = revisions.pop(f)
                    try:
                        event = f.result()
                        if memory.get(event.parent_id).status == "active":
                            memory.finish_revision(event)
                            job["state"] = "completed"
                            job["event_id"] = event.event_id
                        else:
                            job["state"] = "stale_parent"
                    except Exception as exc:
                        job["state"] = "failed"
                        job["error_type"] = type(exc).__name__
                        atomic_json(queue_path, jobs)
                        raise
                    finally:
                        queued.discard(job["parent_id"])
                    atomic_json(queue_path, jobs)
                    queue_revisions(revision_pool, job["trigger_sample_id"])
    return results
