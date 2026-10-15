"""Current role registry excludes removed Fusion calls."""

from verge.prompts import PROMPT_VERSION, _ROLE_GUIDANCE


def test_fusion_role_removed():
    assert 'fusion_judge' not in _ROLE_GUIDANCE
    assert 'experience_updater' in _ROLE_GUIDANCE
    assert PROMPT_VERSION == 'atomic-experience-categories-neutral-observations'
