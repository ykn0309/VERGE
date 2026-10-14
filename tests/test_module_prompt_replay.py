from copy import deepcopy
import pytest
from scripts.prepare_module_prompt_replay import GUIDANCE, EXAMPLES, variant_messages


@pytest.mark.parametrize('role',list(GUIDANCE))
def test_prompt_variants_change_only_system_and_keep_original_history(role):
    row={'role':role,'messages':[{'role':'system','content':'original JSON protocol'},
         {'role':'user','content':'完整原文，包括1\n2'},
         {'role':'assistant','content':'真实历史分析与候选'},
         {'role':'user','content':'真实反馈'}]}
    original=deepcopy(row)
    assert variant_messages(row,'baseline')==row['messages']
    for name in ('concise','few_shot'):
        messages=variant_messages(row,name)
        assert messages[1:]==row['messages'][1:]
        assert messages[0]!=row['messages'][0]
        assert 'JSON' in messages[0]['content']
    assert row==original
    assert EXAMPLES[role]


def test_prompt_adapter_rejects_context_overflow_without_truncating():
    row={'role':'critic','messages':[{'role':'system','content':'JSON'},
         {'role':'user','content':'x'*100_001}]}
    original=deepcopy(row)
    with pytest.raises(ValueError,match='never truncate'):variant_messages(row,'few_shot')
    assert row==original
