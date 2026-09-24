"""实验配置形状及数值边界。"""

from copy import deepcopy
from pathlib import Path

import pytest

from verge.config import load_config


DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads_without_creating_output_dir():
    config = load_config(DEFAULT)
    assert config["run"] == {"seed": 42, "output_dir": "runs/default"}
    assert config["dataset"]["implementation"] == "jsonl"
    assert config["models"]["default"]["backend"] == "fake"
    assert config["modules"]["revision"]["config"]["max_attempts"] == 2
    assert not Path("runs/default").exists()


def test_required_fields_and_types_report_paths(tmp_path):
    import yaml

    valid = load_config(DEFAULT)
    invalid_cases = [
        (lambda c: c["run"].pop("seed"), "run.seed"),
        (lambda c: c["run"].update(seed=True), "run.seed"),
        (lambda c: c["run"].update(output_dir="  "), "run.output_dir"),
        (lambda c: c["dataset"].pop("implementation"), "dataset.implementation"),
        (lambda c: c["dataset"].update(config=[]), "dataset.config"),
        (lambda c: c["models"]["default"].pop("backend"), "models.default.backend"),
        (lambda c: c["models"].update(roles={"judge": {"model": "m"}}),
         "models.roles.judge.backend"),
        (lambda c: c["modules"]["revision"].update(implementation=""),
         "modules.revision.implementation"),
        (lambda c: c["store"].update(config={"root_dir": ""}), "store.config.root_dir"),
        (lambda c: c.pop("modules"), "config.modules"),
    ]
    for mutate, path in invalid_cases:
        data = deepcopy(valid)
        mutate(data)
        file = tmp_path / "invalid.yaml"
        file.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=path):
            load_config(file)


def test_numeric_ranges_and_non_finite_values(tmp_path):
    import yaml

    valid = load_config(DEFAULT)
    cases = [
        ("run", "seed", -1),
        ("modules.retrieval", "top_l", 0),
        ("modules.retrieval", "top_l", 1),
        ("modules.retrieval", "top_l", 21),
        ("modules.selection", "max_selected", 6),
        ("modules.selection", "exploration_slots", 2),
        ("modules.revision", "max_attempts", 3),
        ("modules.revision_credit", "max_budget", -0.1),
        ("modules.revision_credit", "max_budget", float("nan")),
        ("modules.revision_credit", "max_budget", True),
    ]
    for section, name, bad in cases:
        data = deepcopy(valid)
        if section == "run":
            data["run"][name] = bad
            expected = f"run.{name}"
        else:
            module = section.split(".")[1]
            data["modules"][module]["config"][name] = bad
            expected = f"modules.{module}.config.{name}"
        file = tmp_path / "invalid.yaml"
        file.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            load_config(file)


def test_exploration_slots_cannot_exceed_selection_limit(tmp_path):
    import yaml

    data = load_config(DEFAULT)
    data["modules"]["selection"]["config"]["max_selected"] = 0
    data["modules"]["selection"]["config"]["exploration_slots"] = 1
    file = tmp_path / "invalid.yaml"
    file.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exploration_slots: cannot exceed max_selected"):
        load_config(file)


def test_invalid_yaml_and_non_mapping_root(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text("run: [broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_config(file)
    file.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config.yaml: expected a mapping"):
        load_config(file)
    file.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_config(file)
