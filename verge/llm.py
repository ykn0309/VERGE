"""角色无关的 structured-output LLM 边界与离线固定场景 backend。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping, Protocol


class LLMBackend(Protocol):
    """Backend 只返回结构化字段；权重与 credit 由 Calculator 计算。"""

    def generate(self, role: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


FakeScenario = Literal["initial_success", "attempt_1_success", "attempt_2_success", "final_failure"]


class FakeLLMBackend:
    """用于无网络 contract/pipeline 测试；不是实际语义评价器。"""

    def __init__(self, scenario: FakeScenario = "attempt_1_success", *,
                 reflect_count: int = 1, unbound: bool = False,
                 admission_pass: bool = True) -> None:
        if scenario not in ("initial_success", "attempt_1_success", "attempt_2_success", "final_failure"):
            raise ValueError("unknown fake scenario")
        if type(reflect_count) is not int or not 0 <= reflect_count <= 3:
            raise ValueError("reflect_count must be between 0 and 3")
        self.scenario = scenario
        self.reflect_count = reflect_count
        self.unbound = unbound
        self.admission_pass = admission_pass
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def generate(self, role: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = deepcopy(dict(payload))
        self.requests.append((role, request))
        if role == "selector":
            task = request["task"].casefold()
            matching = [item["experience_id"] for item in request["candidates"]
                        if item["trigger"] == "*" or item["trigger"].casefold() in task]
            result: dict[str, Any] = {"selected_experience_ids": matching[:request["max_selected"]]}
        elif role == "initial_generator":
            sources = ([request["selected_experiences"][0]["experience_id"]]
                       if request["selected_experiences"] else [])
            result = {"rubric": {"rubric_id": "r0", "base_rubric_id": None,
                                 "criteria": [
                                     {"criterion_id": "c1", "question": "回答是否完整？",
                                      "importance_score": 2, "source_experiences": sources},
                                     {"criterion_id": "c2", "question": "回答是否符合要求？",
                                      "importance_score": 8, "source_experiences": []},
                                     {"criterion_id": "c3", "question": "回答是否清楚？",
                                      "importance_score": 1, "source_experiences": []},
                                     {"criterion_id": "c4", "question": "回答是否准确？",
                                      "importance_score": 1, "source_experiences": []},
                                 ]}}
        elif role == "judge":
            initial_success = self.scenario == "initial_success"
            result = {"criterion_judgments": [
                {"criterion_id": item["criterion_id"],
                 "chosen": (True if item["criterion_id"] == "c1" and initial_success
                            else item["question"] == "是否逐项满足关键要求？"),
                 "rejected": item["criterion_id"] == "c1" and not initial_success}
                for item in request["criteria"]]}
        elif role == "critic":
            result = {"feedback": "需要将关键任务要求写成可独立判定的标准。"}
        elif role == "revision_generator":
            attempt = request["attempt_number"]
            success = (self.scenario == "attempt_1_success" and attempt == 1
                       or self.scenario == "attempt_2_success" and attempt == 2)
            question = ("是否逐项满足关键要求？" if success
                        else f"是否概述任务目标（第{attempt}次）？")
            result = {"operations": [{
                "operation_id": f"op{attempt}", "operation_type": "MODIFY",
                "target_criterion_id": "c2",
                "new_or_updated_criterion": {"criterion_id": "c2", "question": question,
                                             "importance_score": 8, "source_experiences": []},
                "before_criterion": None,
            }]}
        elif role == "reflector":
            result = {"candidates": [
                {"candidate_id": f"candidate_{i}",
                 "content": "为关键任务要求设置单独的可判定标准。",
                 "trigger": "任务包含明确的关键要求。"}
                for i in range(self.reflect_count)]}
        elif role == "revision_credit_assigner":
            first_op = request["operations"][0]["operation_id"] if request["operations"] else None
            result = {"bindings": {item["candidate_id"]: ([first_op] if first_op and not self.unbound else [])
                                   for item in request["candidates"]}}
        elif role == "admission_checker":
            result = {"decision": "PASS" if self.admission_pass else "FAIL"}
        elif role == "fusion_judge":
            candidate = request["candidate"]
            match = next((index for index, item in enumerate(request["existing"])
                          if (item["content"].strip().casefold() == candidate["content"].strip().casefold()
                              and item["trigger"].strip().casefold() == candidate["trigger"].strip().casefold())),
                         None)
            result = {"match_index": match}
        else:
            raise ValueError(f"unknown LLM role: {role}")
        return deepcopy(result)
