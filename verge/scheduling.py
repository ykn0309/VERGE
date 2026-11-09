"""Rolling sample concurrency; commit completed records before starting replacements."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextvars import copy_context


def run_training_stream(
    samples,
    state,
    runner_factory,
    *,
    workers=5,
    on_commit=None,
):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive integer")
    seen = set(state.trajectories)
    pending_samples = []
    for sample in samples:
        if sample.task_id not in seen:
            seen.add(sample.task_id)
            pending_samples.append(sample)
    iterator = iter(enumerate(pending_samples))

    def run(sample, snapshot):
        return runner_factory(sample).run_sample(sample, snapshot, training=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def fill():
            while len(pending) < workers:
                try:
                    index, sample = next(iterator)
                except StopIteration:
                    break
                future = executor.submit(
                    copy_context().run, run, sample, state.snapshot()
                )
                pending[future] = index

        fill()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: pending[f]):
                    del pending[future]
                    result = future.result()
                    state.commit(result)
                    if on_commit:
                        on_commit(result.record)
                fill()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return state
