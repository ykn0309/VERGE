"""Audit final clustering and provision-edge coverage without mutating state."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from verge.memory import ExperienceMemory


ROOT = Path("runs/reference500-128")
OUT = Path("artifacts/reference500-128/cluster-audit")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def pair_statistics(matrix, positions):
    if len(positions) < 2:
        return {}
    upper = np.triu_indices(len(positions), 1)
    values = matrix[np.ix_(positions, positions)][upper]
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "minimum_member_average": float(
            (matrix[np.ix_(positions, positions)].sum(axis=1) / (len(positions) - 1)).min()
        ),
    }


def main():
    state = ExperienceMemory.load(ROOT / "checkpoint.json")
    clustering = state.clusters
    task_ids = sorted(clustering.assignments("task"))
    experience_ids = sorted(clustering.assignments("experience"))

    task_degree = {
        task_id: len(clustering.references("task", task_id)) for task_id in task_ids
    }
    experience_degree = {
        experience_id: len(clustering.references("experience", experience_id))
        for experience_id in experience_ids
    }
    actual_provision_ids = {
        record.sample.task_id: {
            experience_id
            for path in record.paths
            for experience_id in path.provided_experience_ids
        }
        for record in state.trajectories.values()
    }
    assert all(
        actual_provision_ids[task_id] == clustering.references("task", task_id)
        for task_id in task_ids
    )
    assert sum(task_degree.values()) == sum(experience_degree.values()) == len(
        state.ledger.records
    )

    selected_without_draft = []
    for record in state.trajectories.values():
        task_id = record.sample.task_id
        if record.selection_reasons and not actual_provision_ids[task_id]:
            selected_without_draft.append(
                {
                    "sample_id": record.sample.sample_id,
                    "task_id": task_id,
                    "selector_experience_ids": sorted(record.selection_reasons),
                    "error": record.error,
                }
            )

    task_group_profiles = defaultdict(Counter)
    experience_group_profiles = defaultdict(Counter)
    for task_id in task_ids:
        task_group = clustering.assignments("task")[task_id]
        for experience_id in clustering.references("task", task_id):
            experience_group = clustering.assignments("experience")[experience_id]
            task_group_profiles[task_group][experience_group] += 1
            experience_group_profiles[experience_group][task_group] += 1
    task_groups_without_edge = sum(
        not any(clustering.references("task", task_id) for task_id in members)
        for _, members in clustering.groups("task").items()
    )

    task_matrix_ids, mixed_task_matrix = clustering._matrix("task")
    text_matrix_ids, text_task_matrix = clustering._matrix("task", text_only=True)
    assert task_matrix_ids == text_matrix_ids
    task_positions = {object_id: i for i, object_id in enumerate(task_matrix_ids)}
    experience_matrix_ids, experience_matrix = clustering._matrix("experience")
    experience_positions = {
        object_id: i for i, object_id in enumerate(experience_matrix_ids)
    }

    task_groups = []
    for group_id, members in sorted(
        clustering.groups("task").items(), key=lambda item: (-len(item[1]), item[0])
    ):
        positions = [task_positions[member] for member in members]
        linked_experiences = sorted(
            {
                experience_id
                for task_id in members
                for experience_id in clustering.references("task", task_id)
            }
        )
        linked_experience_groups = sorted(
            {
                clustering.assignments("experience")[experience_id]
                for experience_id in linked_experiences
            }
        )
        task_groups.append(
            {
                "group_id": group_id,
                "size": len(members),
                "linked_experience_count": len(linked_experiences),
                "linked_experience_group_count": len(linked_experience_groups),
                "linked_experience_groups": task_group_profiles.get(group_id, {}),
                "tasks_without_reference": sum(not task_degree[task_id] for task_id in members),
                "mixed_similarity": pair_statistics(mixed_task_matrix, positions),
                "text_similarity": pair_statistics(text_task_matrix, positions),
                "members": [
                    {
                        "task_id": task_id,
                        "sample_id": state.trajectories[task_id].sample.sample_id,
                        "task": state.trajectories[task_id].sample.task,
                        "experience_ids": sorted(clustering.references("task", task_id)),
                    }
                    for task_id in members
                ],
            }
        )

    experience_groups = []
    for group_id, members in sorted(
        clustering.groups("experience").items(), key=lambda item: (-len(item[1]), item[0])
    ):
        positions = [experience_positions[member] for member in members]
        member_details = []
        for experience_id in members:
            task_group_profile = Counter(
                clustering.assignments("task")[task_id]
                for task_id in clustering.references("experience", experience_id)
            )
            member_details.append(
                {
                    "experience_id": experience_id,
                    "reference_count": experience_degree[experience_id],
                    "reference_task_group_count": len(task_group_profile),
                    "reference_task_groups": task_group_profile,
                    "method": state.experiences[experience_id].method.to_dict(),
                }
            )
        experience_groups.append(
            {
                "group_id": group_id,
                "size": len(members),
                "link_similarity": pair_statistics(experience_matrix, positions),
                "reference_task_group_count": len(
                    experience_group_profiles.get(group_id, {})
                ),
                "members": member_details,
            }
        )

    actual_provision_counts = Counter(
        len(clustering.references("task", task_id)) for task_id in task_ids
    )
    selector_request_counts = Counter(
        len(record.selection_reasons) for record in state.trajectories.values()
    )
    task_group_size_counts = Counter(len(members) for _, members in clustering.groups("task").items())
    experience_group_size_counts = Counter(
        len(members) for _, members in clustering.groups("experience").items()
    )
    statistics = {
        "source": str(ROOT / "checkpoint.json"),
        "objects": {
            "tasks": len(task_ids),
            "experiences": len(experience_ids),
            "provision_edges": sum(task_degree.values()),
        },
        "task_provision_coverage": {
            "actual_provision_count_histogram": dict(sorted(actual_provision_counts.items())),
            "selector_request_count_histogram": dict(
                sorted(selector_request_counts.items())
            ),
            "tasks_with_at_least_one_edge": sum(bool(value) for value in task_degree.values()),
            "tasks_without_edge": sum(not value for value in task_degree.values()),
            "selector_selected_then_draft_failed": selected_without_draft,
        },
        "experience_reference_coverage": {
            "degree_zero": sum(not value for value in experience_degree.values()),
            "degree_one": sum(value == 1 for value in experience_degree.values()),
            "degree_at_least_two": sum(value >= 2 for value in experience_degree.values()),
            "degree_histogram": dict(
                sorted(Counter(experience_degree.values()).items())
            ),
        },
        "clusters": {
            "task_group_count": len(task_groups),
            "task_group_size_histogram": dict(sorted(task_group_size_counts.items())),
            "task_groups_without_edge": task_groups_without_edge,
            "experience_group_count": len(experience_groups),
            "experience_group_size_histogram": dict(
                sorted(experience_group_size_counts.items())
            ),
            "task_group_to_experience_group_relations": sum(
                len(profile) for profile in task_group_profiles.values()
            ),
        },
        "task_groups": task_groups,
        "experience_groups": experience_groups,
    }
    write_json(OUT / "statistics.json", statistics)
    print(
        json.dumps(
            {
                key: statistics[key]
                for key in (
                    "objects",
                    "task_provision_coverage",
                    "experience_reference_coverage",
                    "clusters",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
