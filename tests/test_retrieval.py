"""向量广召回与 DashScope 兼容 Embeddings 的无网络测试。"""

import json

import httpx
import pytest

from verge.llm import LLMFormatError, LLMRequestError, LLMSampleError, OpenAICompatibleEmbedding
from verge.memory import ExperienceMemory, TrajectoryLog
from verge.models import Experience, PreferenceSample
from verge.modules import VectorRetriever, make_fake_modules
from verge.pipeline import PipelineRunner


class StubEmbedding:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if text == "task":
            return (1.0, 0.0)
        number = int(text.split("\n", 1)[0].removeprefix("doc"))
        return (number + 1.0, 1.0)


def pool(count=21):
    return [Experience(f"exp_{number:02d}", f"doc{number}", "applicable", 2, 1, [], [])
            for number in range(count)]


def test_vector_top20_cache_add_and_text_change():
    backend = StubEmbedding()
    retriever = VectorRetriever(backend)
    items = pool()
    assert [item.experience_id for item in retriever.retrieve("task", items)] == [
        f"exp_{number:02d}" for number in range(20, 0, -1)
    ]
    assert len(backend.calls) == 22  # 1 个 task，21 条经验
    assert backend.calls[1] == "doc0\napplicable"
    retriever.retrieve("task", items)
    assert len(backend.calls) == 23  # task 不缓存；经验向量命中缓存
    items.append(pool(22)[-1])
    retriever.retrieve("task", items)
    assert len(backend.calls) == 25  # ADD 实时可见，只编码一条新经验
    items[0] = Experience("exp_00", "doc42", "applicable", 2, 1, [], [])
    assert retriever.retrieve("task", items)[0].experience_id == "exp_00"
    assert len(backend.calls) == 27  # 经验文本变化后缓存失效
    assert not any("exp_00" in text or "confidence" in text for text in backend.calls)


def test_vector_fork_copies_cache_without_sharing_mutations():
    backend = StubEmbedding()
    retriever = VectorRetriever(backend)
    items = pool(2)
    retriever.retrieve("task", items)
    fork = retriever.fork()
    assert fork is not retriever and fork.backend is backend
    fork.retrieve("task", items)
    assert backend.calls == ["task", "doc0\napplicable", "doc1\napplicable", "task"]
    modified = Experience("exp_00", "doc42", "applicable", 2, 1, [], [])
    fork.retrieve("task", [modified, items[1]])
    retriever.retrieve("task", items)
    assert backend.calls[-1] == "task"  # fork 的文本更新不影响原缓存。


def test_vector_empty_pool_ties_and_invalid_vectors():
    class FixedEmbedding:
        def __init__(self, vector):
            self.vector, self.calls = vector, 0

        def embed(self, text):
            self.calls += 1
            return self.vector

    backend = FixedEmbedding((1.0, 1.0))
    retriever = VectorRetriever(backend)
    assert retriever.retrieve("task", []) == [] and backend.calls == 0
    assert [item.experience_id for item in retriever.retrieve("task", pool(21))] == [
        f"exp_{number:02d}" for number in range(20)
    ]
    with pytest.raises(ValueError, match="nonzero"):
        VectorRetriever(FixedEmbedding((0.0, 0.0))).retrieve("task", pool(1))
    with pytest.raises(ValueError, match="unique"):
        retriever.retrieve("task", [pool(1)[0], pool(1)[0]])


def test_pipeline_vector_retrieval_and_read_only_evaluation(
    tmp_path,
):
    class CountingEmbedding:
        def __init__(self):
            self.calls = []

        def embed(self, text):
            self.calls.append(text)
            return (1.0, 0.5)

    backend = CountingEmbedding()
    modules = make_fake_modules("attempt_1_success")
    modules.retriever = VectorRetriever(backend)
    memory, log = ExperienceMemory(tmp_path), TrajectoryLog(tmp_path)
    train = PipelineRunner(modules, memory, log)

    def sample(sample_id):
        return PreferenceSample(sample_id, "明确回答问题", "合理答复", "不合理答复", {})

    assert train.run_sample(sample("train-1")).admitted_experience_id
    assert backend.calls == []
    assert train.run_sample(sample("train-2")).admitted_experience_id
    assert len(memory.all()) == 2
    assert len(backend.calls) == 2  # task and one existing experience; no Fusion retrieval
    original = (tmp_path / "experience_memory.json").read_bytes()
    evaluation = PipelineRunner(
        modules, ExperienceMemory(tmp_path, read_only=True), log, training=False
    )
    assert evaluation.run_sample(sample("test-1")).application_attempts == []
    assert (tmp_path / "experience_memory.json").read_bytes() == original
    assert len(backend.calls) == 4
    assert all(
        "合理答复" not in text and "不合理答复" not in text for text in backend.calls
    )


def test_embedding_reuses_env_key_and_parses_response(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY_2", "private-test-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * 64}],
                                        "usage": {"prompt_tokens": 2, "total_tokens": 2}},
                              headers={"x-request-id": "safe-request-id"})

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = httpx.Client(base_url=base_url + "/", transport=httpx.MockTransport(respond))
    events = []
    backend = OpenAICompatibleEmbedding("text-embedding-v4", base_url=base_url,
                                        api_key_env="DASHSCOPE_API_KEY_2", dimensions=64,
                                        client=client, metadata_hook=events.append)
    assert backend.embed("task text") == (1.0,) * 64
    assert requests[0].url.path.endswith("/compatible-mode/v1/embeddings")
    assert requests[0].headers["Authorization"] == "Bearer private-test-key"
    assert json.loads(requests[0].content) == {
        "model": "text-embedding-v4", "input": "task text", "dimensions": 64,
        "encoding_format": "float",
    }
    assert events == [{"role": "embedding", "model": "text-embedding-v4", "attempt": 1,
                       "status": 200, "usage": {"prompt_tokens": 2, "total_tokens": 2},
                       "request_id": "safe-request-id"}]
    assert "private-test-key" not in json.dumps(events)
    client.close()


def test_embedding_retry_and_fail_closed_on_invalid_response(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY_2", "private-test-key")
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 64}]})

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = httpx.Client(base_url=base_url + "/", transport=httpx.MockTransport(respond))
    delays = []
    backend = OpenAICompatibleEmbedding("text-embedding-v4", base_url=base_url,
                                        api_key_env="DASHSCOPE_API_KEY_2", dimensions=64,
                                        client=client, sleep=delays.append)
    with pytest.raises(LLMFormatError, match="invalid vector"):
        backend.embed("task")
    assert len(calls) == 2 and delays == [0.25]
    client.close()
    forbidden = httpx.Client(base_url=base_url + "/", transport=httpx.MockTransport(
        lambda _: httpx.Response(401, text="private-test-key")))
    backend = OpenAICompatibleEmbedding("text-embedding-v4", base_url=base_url,
                                        api_key_env="DASHSCOPE_API_KEY_2", dimensions=64,
                                        client=forbidden)
    with pytest.raises(LLMRequestError, match="embedding: HTTP 401") as error:
        backend.embed("task")
    assert "private-test-key" not in str(error.value)
    assert not isinstance(error.value, LLMSampleError)
    forbidden.close()


@pytest.mark.parametrize('failure', ['timeout', 408, 429, 503])
def test_embedding_exhausted_requests_are_sample_scoped(monkeypatch, failure):
    monkeypatch.setenv('DASHSCOPE_API_KEY_2', 'private-test-key')
    calls = []

    def respond(request):
        calls.append(request)
        if failure == 'timeout':
            raise httpx.ReadTimeout('private endpoint details')
        return httpx.Response(failure)

    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    with httpx.Client(base_url=base_url+'/', transport=httpx.MockTransport(respond)) as client:
        backend = OpenAICompatibleEmbedding('text-embedding-v4', base_url=base_url,
            api_key_env='DASHSCOPE_API_KEY_2', dimensions=64, client=client,
            max_retries=2, sleep=lambda _: None)
        with pytest.raises(LLMSampleError, match='after 3 attempts') as error:
            backend.embed('task')
    assert len(calls) == 3 and 'private endpoint' not in str(error.value)
