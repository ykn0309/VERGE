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


JUDGE_RESPONSE_SIDE: ContextVar[str | None] = ContextVar(
    "judge_response_side", default=None
)
ROLE_CALL_ID = ContextVar("role_call_id", default=None)
APPLICATION_CONTEXT = ContextVar("application_context", default={})
ROLE_ID_MAPPING: ContextVar[dict | None] = ContextVar("role_id_mapping", default=None)


class LLMBackend(Protocol):
    """Backend返回分配权重、Boolean与依据等结构化字段；归一化、反馈及质量推断由Calculator计算。"""

    def generate(
        self,
        role: str,
        payload: Mapping[str, Any],
        *,
        conversation: list[dict[str, str]] | None = None,
    ) -> Mapping[str, Any]: ...


FakeScenario = Literal[
    "initial_success", "attempt_1_success", "attempt_2_success", "final_failure"
]


class LLMRequestError(RuntimeError):
    """网络/API 调用失败；信息不包含 URL、请求正文或凭据。"""


class LLMSampleError(LLMRequestError):
    """有限请求重试耗尽或输入被拒绝；记录当前样本并继续。"""


class LLMRefusalError(LLMSampleError):
    """当前样本遭模型拒答；可记录并跳过，区别于鉴权错误。"""


class LLMFormatError(ValueError):
    """模型输出不是所需结构，已用尽格式重试。"""


_OUTPUT_FIELDS = {
    role: None
    for role in (
        "selector",
        "initial_generator",
        "judge",
        "critic",
        "experience_checker",
        "experience_updater",
        "selector_checker",
        "generator_checker",
        "judge_checker",
    )
}


def _reject_constant(value):
    raise ValueError("invalid JSON constant")


def _normalize_numeric_ids(role, value):
    return None  # The V2 codec normalizes IDs only in typed ID fields.


def _valid_shape(role, value):
    return isinstance(
        value, dict
    )  # Full contracts are validated once by the role adapter.


class OpenAICompatibleBackend:
    """同步 Chat Completions JSON backend；只返回角色结构字段，绝不返回数值 credit。"""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 30,
        max_retries: int = 2,
        enable_thinking: bool | None = None,
        client: httpx.Client | None = None,
        metadata_hook: Callable[[dict[str, Any]], None] | None = None,
        trace_hook: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.scheme not in ("https", "http")
            or (
                parsed.scheme == "http"
                and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            )
        ):
            raise ValueError(
                "base_url requires HTTPS (HTTP allowed only on localhost), without credentials or query"
            )
        if client is not None and (
            str(client.base_url).rstrip("/") != base_url.rstrip("/")
            or client.follow_redirects
        ):
            raise ValueError("client must use the validated base_url without redirects")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("api_key_env must name an environment variable")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"api_key_env: environment variable {api_key_env} is missing"
            )
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
        self._client = (
            client
            if client is not None
            else httpx.Client(
                base_url=base_url.rstrip("/") + "/",
                timeout=timeout_seconds,
                follow_redirects=False,
            )
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _metadata(
        self,
        role: str,
        attempt: int,
        *,
        status: int | None = None,
        response: httpx.Response | None = None,
        body: Any = None,
    ) -> None:
        if self.metadata_hook is None:
            return
        event: dict[str, Any] = {
            "role": role,
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "call_id": ROLE_CALL_ID.get(),
            "attempt": attempt,
            "status": status,
        }
        if role == "judge":
            event["response_side"] = JUDGE_RESPONSE_SIDE.get()
        if response is not None:
            request_id = response.headers.get("x-request-id", "")
            if (
                request_id
                and self._key not in request_id
                and len(request_id) <= 128
                and request_id.isascii()
                and request_id.isprintable()
            ):
                event["request_id"] = request_id
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                reason = choices[0].get("finish_reason")
                if (
                    isinstance(reason, str)
                    and self._key not in reason
                    and len(reason) <= 32
                ):
                    event["finish_reason"] = reason
            usage = body.get("usage")
            if isinstance(usage, dict):
                event["usage"] = {
                    key: usage[key]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int and usage[key] >= 0
                }
        self.metadata_hook(event)

    def _trace(
        self,
        role: str,
        payload: Mapping[str, Any],
        messages: list[dict[str, str]],
        attempt: int,
        *,
        status: int | None = None,
        raw_output: str | None = None,
        output: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self.trace_hook is not None:
            self.trace_hook(
                {
                    "role": role,
                    "backend": "openai_compatible",
                    "model": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "call_id": ROLE_CALL_ID.get(),
                    "attempt": attempt,
                    **(
                        {"response_side": JUDGE_RESPONSE_SIDE.get()}
                        if role == "judge"
                        else {}
                    ),
                    **(
                        {"id_mapping": ROLE_ID_MAPPING.get()}
                        if ROLE_ID_MAPPING.get() is not None
                        else {}
                    ),
                    "status": status,
                    "input": dict(payload),
                    "messages": messages,
                    "raw_output": raw_output,
                    "output": output,
                    "error": error,
                }
            )

    def generate(
        self,
        role: str,
        payload: Mapping[str, Any],
        *,
        conversation: list[dict[str, str]] | None = None,
    ) -> Mapping[str, Any]:
        if role not in _OUTPUT_FIELDS:
            raise ValueError(f"unknown LLM role: {role}")
        if conversation is not None and role != "critic":
            raise ValueError("only critic accepts conversation history")
        messages = (
            llm_messages(role, payload, history=conversation)
            if conversation is not None
            else llm_messages(role, payload)
        )
        for attempt in range(1, self.max_retries + 2):
            if sum(len(m["content"].encode("utf-8")) for m in messages) > payload.get(
                "_message_budget", 100_000
            ):
                raise LLMSampleError(
                    f"{role}: retry context budget exceeded; no truncation"
                )
            try:
                request_body: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
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
                raise LLMSampleError(
                    f"{role}: request failed after {attempt} attempts ({type(exc).__name__})"
                ) from None
            status = response.status_code
            if status in (408, 429) or 500 <= status < 600:
                self._trace(
                    role,
                    payload,
                    messages,
                    attempt,
                    status=status,
                    error=f"HTTP {status}",
                )
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
                raise LLMSampleError(f"{role}: HTTP {status} after {attempt} attempts")
            if status < 200 or status >= 300:
                self._trace(
                    role,
                    payload,
                    messages,
                    attempt,
                    status=status,
                    error=f"HTTP {status}",
                )
                self._metadata(role, attempt, status=status, response=response)
                error_type = (
                    LLMSampleError if status in (400, 413, 422) else LLMRequestError
                )
                raise error_type(f"{role}: HTTP {status}")
            raw_output: str | None = None
            assistant_output: str | None = None
            try:
                body = json.loads(response.text, parse_constant=_reject_constant)
                choices = body["choices"]
                choice = choices[0]
                raw_output = choice["message"].get("content")
                if isinstance(raw_output, str):
                    assistant_output = raw_output
                if not isinstance(raw_output, str):
                    raw_output = response.text
                if choice.get("finish_reason") == "content_filter" or choice[
                    "message"
                ].get("refusal"):
                    self._trace(
                        role,
                        payload,
                        messages,
                        attempt,
                        status=status,
                        raw_output=raw_output,
                        error="model refused the request",
                    )
                    self._metadata(
                        role, attempt, status=status, response=response, body=body
                    )
                    raise LLMRefusalError(f"{role}: model refused the request")
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
                self._trace(
                    role,
                    payload,
                    messages,
                    attempt,
                    status=status,
                    raw_output=raw_output if raw_output is not None else response.text,
                    error="invalid structured output",
                )
                self._metadata(role, attempt, status=status, response=response)
                if conversation is not None:
                    if assistant_output is not None:
                        messages = [
                            *messages,
                            {"role": "assistant", "content": assistant_output},
                        ]
                    conversation[:] = [dict(message) for message in messages]
                if attempt <= self.max_retries:
                    messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": "上次输出格式无效；请严格只返回指定字段的 JSON 对象。",
                        },
                    ]
                    continue
                raise LLMFormatError(
                    f"{role}: invalid structured output after {attempt} attempts"
                ) from None
            self._trace(
                role,
                payload,
                messages,
                attempt,
                status=status,
                raw_output=raw_output,
                output=result,
            )
            self._metadata(role, attempt, status=status, response=response, body=body)
            if conversation is not None:
                conversation[:] = [
                    *[dict(message) for message in messages],
                    {"role": "assistant", "content": content},
                ]
            return result
        raise AssertionError("unreachable retry state")


class OpenAICompatibleEmbedding:
    """同一 API key 的密集文本向量接口；错误消息不包含正文或凭据。"""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str,
        dimensions: int = 1024,
        timeout_seconds: float = 30,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        metadata_hook: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.scheme not in ("https", "http")
            or (
                parsed.scheme == "http"
                and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            )
        ):
            raise ValueError(
                "embedding base_url requires HTTPS without credentials or query"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("embedding model must be non-empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("embedding api_key_env must name an environment variable")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(
                f"embedding api_key_env: environment variable {api_key_env} is missing"
            )
        if type(dimensions) is not int or dimensions not in (
            64,
            128,
            256,
            512,
            768,
            1024,
            1536,
            2048,
        ):
            raise ValueError(
                "embedding dimensions must be a supported text-embedding-v4 dimension"
            )
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 300:
            raise ValueError("embedding timeout_seconds must be in (0, 300]")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("embedding max_retries must be in [0, 5]")
        if client is not None and (
            str(client.base_url).rstrip("/") != base_url.rstrip("/")
            or client.follow_redirects
        ):
            raise ValueError(
                "embedding client must use the validated base_url without redirects"
            )
        self.trace_hook = None
        self.model, self.dimensions, self._key = model, dimensions, key
        self.max_retries, self.metadata_hook, self._sleep = (
            max_retries,
            metadata_hook,
            sleep,
        )
        self._client = (
            client
            if client is not None
            else httpx.Client(
                base_url=base_url.rstrip("/") + "/",
                timeout=timeout_seconds,
                follow_redirects=False,
            )
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, text: str) -> tuple[float, ...]:
        try:
            return self._embed(text)
        except (LLMSampleError, LLMFormatError) as exc:
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "role": "embedding",
                        "model": self.model,
                        "input": {"text": text, "dimensions": self.dimensions},
                        "error": str(exc),
                    }
                )
            if self.metadata_hook is not None:
                self.metadata_hook(
                    {
                        "role": "embedding",
                        "model": self.model,
                        "error": str(exc),
                        "usage": {},
                    }
                )
            raise

    def _embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding input must be non-empty text")
        for attempt in range(1, self.max_retries + 2):
            status = None
            usage: dict[str, int] = {}
            try:
                response = self._client.post(
                    "embeddings",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={
                        "model": self.model,
                        "input": text,
                        "dimensions": self.dimensions,
                        "encoding_format": "float",
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt <= self.max_retries:
                    self._sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise LLMSampleError(
                    f"embedding: request failed after {attempt} attempts ({type(exc).__name__})"
                ) from None
            status = response.status_code
            if status in (408, 429) or 500 <= status < 600:
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
                raise LLMSampleError(
                    f"embedding: HTTP {status} after {attempt} attempts"
                )
            if not 200 <= status < 300:
                error_type = (
                    LLMSampleError if status in (400, 413, 422) else LLMRequestError
                )
                raise error_type(f"embedding: HTTP {status}")
            try:
                body = json.loads(response.text, parse_constant=_reject_constant)
                data = body["data"]
                if (
                    not isinstance(data, list)
                    or len(data) != 1
                    or data[0]["index"] != 0
                ):
                    raise ValueError("invalid embedding data")
                vector = data[0]["embedding"]
                if (
                    not isinstance(vector, list)
                    or len(vector) != self.dimensions
                    or any(
                        type(value) not in (int, float) or not math.isfinite(value)
                        for value in vector
                    )
                    or not any(value != 0 for value in vector)
                ):
                    raise ValueError("invalid embedding vector")
                if isinstance(body.get("usage"), dict):
                    usage = {
                        key: body["usage"][key]
                        for key in ("prompt_tokens", "total_tokens")
                        if type(body["usage"].get(key)) is int
                        and body["usage"][key] >= 0
                    }
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                raise LLMFormatError("embedding: invalid vector response") from None
            if self.metadata_hook is not None:
                event: dict[str, Any] = {
                    "role": "embedding",
                    "model": self.model,
                    "attempt": attempt,
                    "status": status,
                    "usage": usage,
                }
                request_id = response.headers.get("x-request-id", "")
                if (
                    request_id
                    and self._key not in request_id
                    and len(request_id) <= 128
                    and request_id.isascii()
                    and request_id.isprintable()
                ):
                    event["request_id"] = request_id
                self.metadata_hook(event)
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "role": "embedding",
                        "model": self.model,
                        "input": {"text": text, "dimensions": self.dimensions},
                        "output": vector,
                        "usage": usage,
                        "attempt": attempt,
                    }
                )
            return tuple(vector)
        raise AssertionError("unreachable embedding retry state")


class FakeLLMBackend:
    """Deterministic transport fixture; never claims empirical quality."""

    def __init__(self, scenario="attempt_1_success"):
        if scenario not in (
            "initial_success",
            "attempt_1_success",
            "attempt_2_success",
            "final_failure",
        ):
            raise ValueError("unknown fake scenario")
        self.scenario = scenario
        self.trace_hook = None
        self.metadata_hook = None
        self.calls = []

    def generate(self, role, payload, *, conversation=None):
        self.calls.append((role, payload))
        if role == "selector":
            result = {
                "selected_experience_ids": [
                    e["experience_id"]
                    for e in payload["candidates"][: payload["max_selected"]]
                ]
            }
        elif role == "initial_generator":
            experiences = payload["experiences"]
            n = payload.get("_round", 0)
            success = self.scenario == "initial_success" or n >= (
                2 if self.scenario == "attempt_2_success" else 1
            )
            if self.scenario == "final_failure":
                success = False
            criteria = []
            for i in range(2):
                e = experiences[i % len(experiences)] if experiences else None
                question = "验证通过" if success else "基础核验"
                criteria.append(
                    {
                        "question": question,
                        "weight": 0.5,
                        "source_experiences": [e["experience_id"]] if e else [],
                    }
                )
            result = {"analysis": "离线接口示例", "criteria": criteria}
        elif role == "judge":
            side = JUDGE_RESPONSE_SIDE.get()
            result = {
                "criterion_judgments": [
                    {
                        "criterion_id": c["criterion_id"],
                        "satisfied": side == "chosen" and c["question"] == "验证通过",
                        "reason": "离线固定判断",
                    }
                    for c in payload["criteria"]
                ]
            }
        elif role == "critic":
            result = {
                "analysis": "离线诊断",
                "candidates": [
                    {
                        "trigger": "任务有明确要求",
                        "principle": "核对回答是否落实任务要求",
                    }
                ],
            }
        elif role == "experience_checker":
            result = {
                "approved": not (
                    self.scenario == "attempt_2_success"
                    and sum(r == "experience_checker" for r, _ in self.calls) == 1
                ),
                "reason": "离线候选核对报告",
            }
        elif role.endswith("_checker"):
            result = {"status": "correct", "evidence": "离线核对报告", "details": []}
        elif role == "experience_updater":
            result = {
                "action": "keep",
                "method": None,
                "target_fields": [],
                "reason": "没有已确认的内容缺陷",
            }
        else:
            raise ValueError("unknown fake role")
        messages = llm_messages(role, payload, history=conversation)
        if conversation is not None:
            conversation[:] = [
                *messages,
                {
                    "role": "assistant",
                    "content": json.dumps(result, ensure_ascii=False),
                },
            ]
        if self.trace_hook:
            self.trace_hook(
                {
                    "role": role,
                    "input": dict(payload),
                    "messages": messages,
                    "output": result,
                    "raw_output": json.dumps(result, ensure_ascii=False),
                    "model": "fake",
                    "prompt_version": PROMPT_VERSION,
                    "call_id": ROLE_CALL_ID.get(),
                    "id_mapping": ROLE_ID_MAPPING.get(),
                }
            )
        return result
