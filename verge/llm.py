"""角色无关的 structured-output LLM 边界与离线固定场景 backend。"""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
import json
import math
import os
import re
import time
from typing import Any, Callable, Literal, Mapping, Protocol
from urllib.parse import urlsplit

import httpx

from verge.prompts import PROMPT_VERSION, llm_messages


JUDGE_RESPONSE_SIDE: ContextVar[str | None] = ContextVar("judge_response_side", default=None)


class LLMBackend(Protocol):
    """Backend 只返回结构化字段；权重与 credit 由 Calculator 计算。"""

    def generate(self, role: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


FakeScenario = Literal["initial_success", "attempt_1_success", "attempt_2_success", "final_failure"]


class LLMRequestError(RuntimeError):
    """网络/API 调用失败；信息不包含 URL、请求正文或凭据。"""


class LLMFormatError(ValueError):
    """模型输出不是所需结构，已用尽格式重试。"""


_OUTPUT_FIELDS: dict[str, tuple[str, type | tuple[type, ...]]] = {
    "selector": ("selected_experience_ids", list),
    "initial_generator": ("criteria", list),
    "judge": ("criterion_judgments", list),
    "critic": ("operations", list),
    "reflector": ("candidates", list),
    "revision_credit_assigner": ("bindings", dict),
    "admission_checker": ("decision", str),  # 另需非空 reason；在 _valid_shape 中联合校验。
    "fusion_judge": ("match_index", (int, type(None))),
}


def _reject_constant(_value: str) -> None:
    raise ValueError("invalid JSON constant")


_NUMERIC_ID_FIELDS = frozenset({"rubric_id", "base_rubric_id", "criterion_id",
                                "target_criterion_id", "operation_id", "candidate_id"})
_NUMERIC_ID_LIST_FIELDS = frozenset({"selected_experience_ids", "source_experiences"})


def _normalize_numeric_ids(role: str, value: Any) -> None:
    """只在模型 ID 字段把正整数 1 规范化为 "1"；原始响应仍留在 trace 中。"""
    if role not in {"selector", "initial_generator", "judge", "critic", "reflector",
                    "revision_credit_assigner"}:
        return

    def numeric_id(item: Any) -> Any:
        return str(item) if type(item) is int and item > 0 else item

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                if key in _NUMERIC_ID_FIELDS:
                    item[key] = numeric_id(child)
                elif key in _NUMERIC_ID_LIST_FIELDS and isinstance(child, list):
                    item[key] = [numeric_id(value) for value in child]
                elif key == "bindings" and isinstance(child, dict):
                    for binding_key, ids in child.items():
                        if isinstance(ids, list):
                            child[binding_key] = [numeric_id(value) for value in ids]
                else:
                    visit(child)

    visit(value)


def _valid_shape(role: str, value: Any) -> bool:
    """缺字段/字段类型不对才重试；ID引用、绑定为空、准入FAIL等语义决策不重试。"""
    if role == "admission_checker":
        return (isinstance(value, dict) and set(value) == {"decision", "reason"}
                and value["decision"] in ("PASS", "FAIL")
                and isinstance(value["reason"], str) and bool(value["reason"].strip()))
    if role == "critic":
        if (not isinstance(value, dict) or set(value) != {"analysis", "operations"}
                or not isinstance(value["analysis"], str) or not value["analysis"].strip()
                or not isinstance(value["operations"], list)):
            return False
        data = value["operations"]
    else:
        field, expected = _OUTPUT_FIELDS[role]
        if not isinstance(value, dict) or set(value) != {field} or not isinstance(value[field], expected):
            return False
        data = value[field]
    if role == "critic":
        return all(isinstance(item, dict)
                   and set(item) == {"operation_type", "target_criterion_id", "new_or_updated_criterion"}
                   and item["operation_type"] in ("ADD", "DELETE", "MODIFY")
                   and (item["target_criterion_id"] is None or isinstance(item["target_criterion_id"], str))
                   and (item["new_or_updated_criterion"] is None
                        or (isinstance(item["new_or_updated_criterion"], dict)
                            and set(item["new_or_updated_criterion"]) == {"question", "importance_score"}
                            and isinstance(item["new_or_updated_criterion"]["question"], str)
                            and type(item["new_or_updated_criterion"]["importance_score"]) is int
                            and 1 <= item["new_or_updated_criterion"]["importance_score"] <= 10))
                   for item in data)
    if role == "selector":
        return all(isinstance(key, str) for key in data)
    if role == "initial_generator":
        return all(isinstance(item, dict)
                   and set(item) == {"question", "importance_score", "source_experiences"}
                   and isinstance(item["question"], str)
                   and type(item["importance_score"]) is int
                   and 1 <= item["importance_score"] <= 10
                   and isinstance(item["source_experiences"], list)
                   and all(isinstance(key, str) for key in item["source_experiences"])
                   for item in data)
    if role == "judge":
        return all(isinstance(item, dict) and set(item) == {"criterion_id", "satisfied"}
                   and isinstance(item["criterion_id"], str)
                   and type(item["satisfied"]) is bool for item in data)
    if role == "reflector":
        return all(isinstance(item, dict) and set(item) == {"content", "trigger"}
                   and all(isinstance(item[key], str) for key in ("content", "trigger"))
                   for item in data)
    if role == "revision_credit_assigner":
        return all(isinstance(key, str) and isinstance(ids, list)
                   and all(isinstance(item, str) for item in ids) for key, ids in data.items())
    if role == "fusion_judge":
        return data is None or type(data) is int
    return True


class OpenAICompatibleBackend:
    """同步 Chat Completions JSON backend；只返回角色结构字段，绝不返回数值 credit。"""

    def __init__(self, model: str, *, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout_seconds: float = 30,
                 max_retries: int = 2, enable_thinking: bool | None = None,
                 client: httpx.Client | None = None,
                 metadata_hook: Callable[[dict[str, Any]], None] | None = None,
                 trace_hook: Callable[[dict[str, Any]], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        parsed = urlsplit(base_url)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.scheme not in ("https", "http")
                or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))):
            raise ValueError("base_url requires HTTPS (HTTP allowed only on localhost), without credentials or query")
        if client is not None and (str(client.base_url).rstrip("/") != base_url.rstrip("/")
                                   or client.follow_redirects):
            raise ValueError("client must use the validated base_url without redirects")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("api_key_env must name an environment variable")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"api_key_env: environment variable {api_key_env} is missing")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be in (0, 300]")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer in [0, 5]")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be Boolean when specified")
        self.enable_thinking = enable_thinking
        self.model = model
        self._key = key
        self.max_retries = max_retries
        self.metadata_hook = metadata_hook
        self.trace_hook = trace_hook
        self._sleep = sleep
        self._client = client if client is not None else httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout_seconds, follow_redirects=False,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _metadata(self, role: str, attempt: int, *, status: int | None = None,
                  response: httpx.Response | None = None, body: Any = None) -> None:
        if self.metadata_hook is None:
            return
        event: dict[str, Any] = {"role": role, "model": self.model, "prompt_version": PROMPT_VERSION,
                                 "attempt": attempt, "status": status}
        if role == "judge":
            event["response_side"] = JUDGE_RESPONSE_SIDE.get()
        if response is not None:
            request_id = response.headers.get("x-request-id", "")
            if (request_id and self._key not in request_id and len(request_id) <= 128
                    and request_id.isascii() and request_id.isprintable()):
                event["request_id"] = request_id
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                reason = choices[0].get("finish_reason")
                if isinstance(reason, str) and self._key not in reason and len(reason) <= 32:
                    event["finish_reason"] = reason
            usage = body.get("usage")
            if isinstance(usage, dict):
                event["usage"] = {key: usage[key] for key in
                                  ("prompt_tokens", "completion_tokens", "total_tokens")
                                  if type(usage.get(key)) is int and usage[key] >= 0}
        self.metadata_hook(event)

    def _trace(self, role: str, payload: Mapping[str, Any], messages: list[dict[str, str]],
               attempt: int, *, status: int | None = None, raw_output: str | None = None,
               output: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        if self.trace_hook is not None:
            self.trace_hook({"role": role, "backend": "openai_compatible", "model": self.model,
                             "prompt_version": PROMPT_VERSION, "attempt": attempt,
                             **({"response_side": JUDGE_RESPONSE_SIDE.get()} if role == "judge" else {}),
                             "status": status, "input": dict(payload), "messages": messages,
                             "raw_output": raw_output, "output": output, "error": error})

    def generate(self, role: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if role not in _OUTPUT_FIELDS:
            raise ValueError(f"unknown LLM role: {role}")
        messages = llm_messages(role, payload)
        for attempt in range(1, self.max_retries + 2):
            try:
                request_body: dict[str, Any] = {"model": self.model, "messages": messages,
                                                "response_format": {"type": "json_object"}}
                if self.enable_thinking is not None:
                    request_body["enable_thinking"] = self.enable_thinking
                response = self._client.post(
                    "chat/completions",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=request_body,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self._trace(role, payload, messages, attempt, error=type(exc).__name__)
                self._metadata(role, attempt)
                if attempt <= self.max_retries:
                    self._sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise LLMRequestError(f"{role}: request failed after {attempt} attempts ({type(exc).__name__})") from None
            status = response.status_code
            if status == 429 or 500 <= status < 600:
                self._trace(role, payload, messages, attempt, status=status, error=f"HTTP {status}")
                self._metadata(role, attempt, status=status, response=response)
                if attempt <= self.max_retries:
                    retry_after = response.headers.get("retry-after", "")
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2 ** (attempt - 1)
                    if not math.isfinite(delay):
                        delay = 2 ** (attempt - 1)
                    self._sleep(min(max(delay, 0), 8))
                    continue
                raise LLMRequestError(f"{role}: HTTP {status} after {attempt} attempts")
            if status < 200 or status >= 300:
                self._trace(role, payload, messages, attempt, status=status, error=f"HTTP {status}")
                self._metadata(role, attempt, status=status, response=response)
                raise LLMRequestError(f"{role}: HTTP {status}")
            raw_output: str | None = None
            try:
                body = json.loads(response.text, parse_constant=_reject_constant)
                choices = body["choices"]
                choice = choices[0]
                raw_output = choice["message"].get("content")
                if not isinstance(raw_output, str):
                    raw_output = response.text
                if choice.get("finish_reason") == "content_filter" or choice["message"].get("refusal"):
                    self._trace(role, payload, messages, attempt, status=status,
                                raw_output=raw_output, error="model refused the request")
                    self._metadata(role, attempt, status=status, response=response, body=body)
                    raise LLMRequestError(f"{role}: model refused the request")
                if choice.get("finish_reason") == "length":
                    raise ValueError("truncated output")
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("missing message content")
                result = json.loads(content, parse_constant=_reject_constant)
                _normalize_numeric_ids(role, result)
                if not _valid_shape(role, result):
                    raise ValueError("unexpected response fields")
            except LLMRequestError:
                raise
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                self._trace(role, payload, messages, attempt, status=status,
                            raw_output=raw_output if raw_output is not None else response.text,
                            error="invalid structured output")
                self._metadata(role, attempt, status=status, response=response)
                if attempt <= self.max_retries:
                    messages = [*messages, {"role": "user", "content": "上次输出格式无效；请严格只返回指定字段的 JSON 对象。"}]
                    continue
                raise LLMFormatError(f"{role}: invalid structured output after {attempt} attempts") from None
            self._trace(role, payload, messages, attempt, status=status,
                        raw_output=raw_output, output=result)
            self._metadata(role, attempt, status=status, response=response, body=body)
            return result
        raise AssertionError("unreachable retry state")


class OpenAICompatibleEmbedding:
    """同一 API key 的密集文本向量接口；错误消息不包含正文或凭据。"""

    def __init__(self, model: str, *, base_url: str, api_key_env: str,
                 dimensions: int = 1024, timeout_seconds: float = 30,
                 max_retries: int = 2, client: httpx.Client | None = None,
                 metadata_hook: Callable[[dict[str, Any]], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        parsed = urlsplit(base_url)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.scheme not in ("https", "http")
                or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))):
            raise ValueError("embedding base_url requires HTTPS without credentials or query")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("embedding model must be non-empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("embedding api_key_env must name an environment variable")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"embedding api_key_env: environment variable {api_key_env} is missing")
        if type(dimensions) is not int or dimensions not in (64, 128, 256, 512, 768, 1024, 1536, 2048):
            raise ValueError("embedding dimensions must be a supported text-embedding-v4 dimension")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 300:
            raise ValueError("embedding timeout_seconds must be in (0, 300]")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("embedding max_retries must be in [0, 5]")
        if client is not None and (str(client.base_url).rstrip("/") != base_url.rstrip("/")
                                   or client.follow_redirects):
            raise ValueError("embedding client must use the validated base_url without redirects")
        self.model, self.dimensions, self._key = model, dimensions, key
        self.max_retries, self.metadata_hook, self._sleep = max_retries, metadata_hook, sleep
        self._client = client if client is not None else httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout_seconds, follow_redirects=False,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding input must be non-empty text")
        for attempt in range(1, self.max_retries + 2):
            status = None
            usage: dict[str, int] = {}
            try:
                response = self._client.post(
                    "embeddings", headers={"Authorization": f"Bearer {self._key}"},
                    json={"model": self.model, "input": text, "dimensions": self.dimensions,
                          "encoding_format": "float"},
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt <= self.max_retries:
                    self._sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise LLMRequestError(f"embedding: request failed after {attempt} attempts ({type(exc).__name__})") from None
            status = response.status_code
            if status == 429 or 500 <= status < 600:
                if attempt <= self.max_retries:
                    retry_after = response.headers.get("retry-after", "")
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2 ** (attempt - 1)
                    if not math.isfinite(delay):
                        delay = 2 ** (attempt - 1)
                    self._sleep(min(max(delay, 0), 8))
                    continue
                raise LLMRequestError(f"embedding: HTTP {status} after {attempt} attempts")
            if not 200 <= status < 300:
                raise LLMRequestError(f"embedding: HTTP {status}")
            try:
                body = json.loads(response.text, parse_constant=_reject_constant)
                data = body["data"]
                if not isinstance(data, list) or len(data) != 1 or data[0]["index"] != 0:
                    raise ValueError("invalid embedding data")
                vector = data[0]["embedding"]
                if (not isinstance(vector, list) or len(vector) != self.dimensions
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               for value in vector)
                        or not any(value != 0 for value in vector)):
                    raise ValueError("invalid embedding vector")
                if isinstance(body.get("usage"), dict):
                    usage = {key: body["usage"][key] for key in ("prompt_tokens", "total_tokens")
                             if type(body["usage"].get(key)) is int and body["usage"][key] >= 0}
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                raise LLMFormatError("embedding: invalid vector response") from None
            if self.metadata_hook is not None:
                event: dict[str, Any] = {"role": "embedding", "model": self.model,
                                         "attempt": attempt, "status": status, "usage": usage}
                request_id = response.headers.get("x-request-id", "")
                if (request_id and self._key not in request_id and len(request_id) <= 128
                        and request_id.isascii() and request_id.isprintable()):
                    event["request_id"] = request_id
                self.metadata_hook(event)
            return tuple(vector)
        raise AssertionError("unreachable embedding retry state")


class FakeLLMBackend:
    """用于无网络 contract/pipeline 测试；不是实际语义评价器。"""

    def __init__(self, scenario: FakeScenario = "attempt_1_success", *,
                 reflect_count: int = 1, unbound: bool = False,
                 admission_pass: bool = True,
                 trace_hook: Callable[[dict[str, Any]], None] | None = None) -> None:
        if scenario not in ("initial_success", "attempt_1_success", "attempt_2_success", "final_failure"):
            raise ValueError("unknown fake scenario")
        if type(reflect_count) is not int or not 0 <= reflect_count <= 3:
            raise ValueError("reflect_count must be between 0 and 3")
        self.scenario = scenario
        self.reflect_count = reflect_count
        self.unbound = unbound
        self.admission_pass = admission_pass
        self.trace_hook = trace_hook
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
            preferred = JUDGE_RESPONSE_SIDE.get() == "chosen"
            result = {"criterion_judgments": [
                {"criterion_id": item["criterion_id"],
                 "satisfied": ((True if item["criterion_id"] == "c1" and initial_success
                                else item["question"] == "是否逐项满足关键要求？")
                               if preferred else item["criterion_id"] == "c1" and not initial_success)}
                for item in request["criteria"]]}
        elif role == "critic":
            attempt = request["attempt_number"]
            success = (self.scenario == "attempt_1_success" and attempt == 1
                       or self.scenario == "attempt_2_success" and attempt == 2)
            question = ("是否逐项满足关键要求？" if success
                        else f"是否概述任务目标（第{attempt}次）？")
            result = {"analysis": "R0 对关键要求的判别不足，需要针对任务要求调整标准。", "operations": [{
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
            result = {"decision": "PASS" if self.admission_pass else "FAIL",
                      "reason": "候选满足准入条件。" if self.admission_pass else "候选不满足准入条件。"}
        elif role == "fusion_judge":
            candidate = request["candidate"]
            match = next((index for index, item in enumerate(request["existing"])
                          if (item["content"].strip().casefold() == candidate["content"].strip().casefold()
                              and item["trigger"].strip().casefold() == candidate["trigger"].strip().casefold())),
                         None)
            result = {"match_index": match}
        else:
            raise ValueError(f"unknown LLM role: {role}")
        if self.trace_hook is not None:
            self.trace_hook({"role": role, "backend": "fake", "model": self.scenario,
                             "attempt": 1, "status": None, "input": request,
                             **({"response_side": JUDGE_RESPONSE_SIDE.get()} if role == "judge" else {}),
                             "messages": None, "raw_output": None, "output": result, "error": None})
        return deepcopy(result)
