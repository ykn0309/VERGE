"""Deterministic constrained modularity search on one fixed dense matrix.

The floor constrains every member average. Weak positive edges remain in the
objective. Search is local, multi-start, and explicitly bounded, not global.
"""

from dataclasses import dataclass
import numpy as np

EPS = 1e-12


def canonical(groups):
    return tuple(sorted(tuple(sorted(c)) for c in groups if c))


@dataclass
class PartitionResult:
    groups: tuple
    quality: float
    repairs: list
    moves: int
    budget: bool


class ModularitySearch:
    def __init__(self, matrix, threshold, budget=100):
        self.a = np.asarray(matrix, dtype=float)
        self.n = len(self.a)
        self.threshold, self.budget = threshold, budget
        degree = self.a.sum(axis=1)
        self.total = float(degree.sum())
        self.b = (
            self.a - np.outer(degree, degree) / self.total
            if self.total
            else self.a.copy()
        )

    def feasible(self, members):
        if len(members) < 2:
            return True
        sums = self.a[np.ix_(members, members)].sum(axis=1)
        return bool(np.all(sums / (len(members) - 1) >= self.threshold - EPS))

    def quality(self, groups):
        return (
            sum(float(self.b[np.ix_(c, c)].sum()) for c in groups) / self.total
            if self.total
            else 0.0
        )

    def repair(self, groups):
        groups = [list(c) for c in canonical(groups)]
        repairs = []
        for c in list(groups):
            while not self.feasible(c):
                averages = self.a[np.ix_(c, c)].sum(axis=1) / (len(c) - 1)
                lowest = float(averages.min())
                x = min(x for x, score in zip(c, averages) if score <= lowest + EPS)
                c.remove(x)
                groups.append([x])
                repairs.append({"object_index": x, "average": lowest})
        return canonical(groups), repairs

    def _search(self, start, movable=None):
        groups = [list(c) for c in canonical(start)]
        moves = 0
        allowed = range(self.n) if movable is None else sorted(movable)
        for _ in range(self.budget):
            moved = False
            for x in allowed:
                source = next(c for c in groups if x in c)
                rest = [v for v in source if v != x]
                if not self.feasible(rest):
                    continue
                loss = float(self.b[x, rest].sum())
                best, best_gain = None, EPS
                targets = [c for c in groups if c is not source] + [[]]
                for target in sorted(targets, key=lambda c: tuple(c)):
                    if not target and not rest:
                        continue
                    gain = 2 * (float(self.b[x, target].sum()) - loss) / self.total
                    if gain <= best_gain + EPS:
                        continue
                    if not self.feasible(sorted([*target, x])):
                        continue
                    best, best_gain = target, gain
                if best is not None:
                    source.remove(x)
                    if not source:
                        groups.remove(source)
                    if not best:
                        groups.append(best)
                    best.append(x)
                    best.sort()
                    moves += 1
                    moved = True
            if movable is not None:
                # Online text placement only moves new objects, never merges old groups.
                if not moved:
                    return canonical(groups), moves, False
                continue
            best_pair, best_gain = None, EPS
            groups.sort(key=lambda c: tuple(c))
            for i, left in enumerate(groups):
                for right in groups[i + 1 :]:
                    gain = 2 * float(self.b[np.ix_(left, right)].sum()) / self.total
                    if gain <= best_gain + EPS:
                        continue
                    cross = float(self.a[np.ix_(left, right)].mean())
                    if cross < self.threshold - EPS or not self.feasible(
                        sorted(left + right)
                    ):
                        continue
                    best_pair, best_gain = (left, right), gain
            if best_pair is not None:
                left, right = best_pair
                groups.remove(right)
                left.extend(right)
                left.sort()
                moves += 1
            elif not moved:
                return canonical(groups), moves, False
        return canonical(groups), moves, True

    def run(self, previous, *, movable=None):
        if self.n <= 1 or not self.total:
            groups = tuple((i,) for i in range(self.n))
            return PartitionResult(groups, 0, [], 0, False)
        repaired, repairs = (
            self.repair(previous) if movable is None else (canonical(previous), [])
        )
        if movable is not None:
            groups, moves, budget = self._search(repaired, movable)
            return PartitionResult(groups, self.quality(groups), repairs, moves, budget)
        singles = tuple((i,) for i in range(self.n))
        seeds = {repaired, singles}
        if self.feasible(list(range(self.n))):
            seeds.add((tuple(range(self.n)),))
        old_pairs = {(x, y) for c in previous for x in c for y in c if x < y}

        def changes(groups):
            return len(
                old_pairs.symmetric_difference(
                    {(x, y) for c in groups for x in c for y in c if x < y}
                )
            )

        best = None
        budget_seen = False
        total_moves = 0
        for seed in sorted(seeds):
            groups, moves, budget = self._search(seed)
            quality = self.quality(groups)
            budget_seen |= budget
            total_moves += moves
            if (
                best is None
                or quality > best[0] + EPS
                or (
                    abs(quality - best[0]) <= EPS
                    and (changes(groups), groups) < (changes(best[1]), best[1])
                )
            ):
                best = quality, groups
        return PartitionResult(best[1], best[0], repairs, total_moves, budget_seen)
