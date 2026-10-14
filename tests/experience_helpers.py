"""Construct actual positive application cases for test seeds, never bypass admission."""
from verge.calculator import calculate_rubric_judgment, calculate_experience_credits
from verge.models import ExperienceCase, Rubric, Criterion, CriterionJudgment


def case_for(source, sid, content, trigger, version=1, p=.2, n=0):
    weights = [(p, 1), (n, -1), (1-p-n, 0)]
    weights = [(w, sign) for w, sign in weights if w > 0]
    if len(weights) == 1:
        w, sign = weights[0]
        weights = [(w/2, sign), (w/2, sign)]
    rubric = Rubric('fixture_r', [Criterion(f'c{i}', f'实际要求{i}？', w, [source] if sign else []) for i, (w, sign) in enumerate(weights)])
    judgment = calculate_rubric_judgment(rubric, [CriterionJudgment(f'c{i}', sign==1, sign==-1, '依据甲', '依据乙') for i, (_, sign) in enumerate(weights)])
    return ExperienceCase(sid, '任务原文', '回答甲', '回答乙', source, version, content, trigger,
                          rubric, judgment, calculate_experience_credits(rubric, judgment)[source])


def seed(memory, candidate, sid='seed', p=.2, n=0):
    return memory.add(candidate, case_for(candidate.candidate_id, sid, candidate.content, candidate.trigger, p=p, n=n))


def keep_update(experience, content=None, trigger=None):
    from verge.models import ExperienceUpdate
    return ExperienceUpdate(experience.content if content is None else content,
                            experience.trigger if trigger is None else trigger)
