"""模拟 HTTP 的输入快照；不依赖解析自然语言消息还原内部协议。"""

from copy import deepcopy
import json

import pytest


@pytest.fixture
def wire_payload(monkeypatch):
    import verge.llm as llm

    original = llm.llm_messages
    snapshots = {}

    def capture(role, payload, **kwargs):
        messages = original(role, payload, **kwargs)
        snapshots[json.dumps(messages, ensure_ascii=False)] = deepcopy(dict(payload))
        return messages

    monkeypatch.setattr(llm, 'llm_messages', capture)

    def lookup(body):
        messages = body['messages']
        while json.dumps(messages, ensure_ascii=False) not in snapshots and messages:
            messages = messages[:-1]  # backend 格式纠错追加的 user 消息
        return deepcopy(snapshots[json.dumps(messages, ensure_ascii=False)])

    return lookup
