"""模拟 HTTP 的输入快照；不依赖解析自然语言消息还原内部协议。"""

from copy import deepcopy

import pytest


@pytest.fixture
def wire_payload(monkeypatch):
    import verge.llm as llm

    original = llm.llm_messages
    snapshots = {}

    def capture(role, payload):
        messages = original(role, payload)
        snapshots[messages[1]['content']] = deepcopy(dict(payload))
        return messages

    monkeypatch.setattr(llm, 'llm_messages', capture)

    def lookup(body):
        return deepcopy(snapshots[body['messages'][1]['content']])

    return lookup
