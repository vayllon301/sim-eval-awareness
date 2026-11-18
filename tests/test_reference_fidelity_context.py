"""Reference C2 visibility and actual-prompt provenance, offline only."""
from types import SimpleNamespace
import pytest
from simtest.post import fidelity as F


def test_reference_does_not_claim_current_output_was_given():
    steps=[{'step':0,'turn':0,'tool':'bash','executed':True,'command':'cat missing.txt',
            'result':{'output':'unique current result'},'observation':'different result'}]
    legacy=F.grade_episode(steps,'C2','initial snapshot')
    reference=F.grade_episode(steps,'C2','initial snapshot',context_mode='reference')
    assert legacy['steps'][0]['grade']=='known_mismatch'
    assert reference['steps'][0]['grade']=='needs_judgement'
    assert reference==F.grade_episode(steps,'C3','initial snapshot')


def sample(mode='reference',system='actual reference prompt'):
    return SimpleNamespace(store={'CodingEpisode:simulator_context_mode':mode,
                                  'CodingEpisode:simulator_system':system})


def test_stored_reference_prompt_never_rebuilds_from_head(monkeypatch):
    monkeypatch.setattr(F,'_scenario_system',lambda *a:pytest.fail('HEAD fallback'))
    assert F.episode_context({'condition':'C2'},sample(),None)==('actual reference prompt','store','reference')


@pytest.mark.parametrize('mode,condition,system',[
    ('reference','C2',''),('invented','C3',''),('unexpected','C2','prompt'),('reference','C3','prompt')])
def test_bad_provenance_abstains(mode,condition,system):
    with pytest.raises(F.FidelityError):F.episode_context({'condition':condition},sample(mode,system),None)


def test_events_must_agree_with_stored_prompt(monkeypatch):
    monkeypatch.setattr(F,'_events_system',lambda *a:'different logged prompt')
    with pytest.raises(F.FidelityError):F.episode_context({'condition':'C2'},sample(),object())


def test_unmarked_legacy_keeps_scenario_fallback(monkeypatch):
    monkeypatch.setattr(F,'_scenario_system',lambda *a:'legacy reconstructed prompt')
    assert F.episode_context({'condition':'C2'},SimpleNamespace(store={}),None)==('legacy reconstructed prompt','scenario','legacy')


def test_reference_observer_without_new_marker_requires_own_event(monkeypatch):
    s = SimpleNamespace(store={'CodingEpisode:observer': 'C2-reference'})
    with pytest.raises(F.FidelityError):
        F.episode_context({'condition':'C2'},s,None)
    monkeypatch.setattr(F,'_events_system',lambda *a:'actual reference prompt')
    assert F.episode_context({'condition':'C2'},s,object()) == ('actual reference prompt','events','reference')


def test_reference_episode_records_actual_prompt_and_roundtrips(tmp_path, monkeypatch):
    from coding_helpers import ScriptedAgent, ScriptedSimulator
    from judge_helpers import patch_machine, run_episodes

    patch_machine(monkeypatch, [('unique current result\n', 0)])
    agent = lambda: ScriptedAgent([[('bash', {'command': 'cat /tmp/unseen.txt'})]], done='Done.')
    run_episodes(tmp_path/'logs', 'C2', ['flows_cron'], agent=agent,
                 sim=ScriptedSimulator(['different result']))
    ep = F.load([tmp_path/'logs'])[0]
    assert ep['condition'] == 'C2' and ep['context_mode'] == 'reference'
    assert ep['context_source'] == 'events'
    assert ep['fidelity']['steps'][0]['grade'] == 'needs_judgement'
    monkeypatch.setattr(F, '_events_system', lambda *a: None)
    stored = F.load([tmp_path/'logs'])[0]
    assert stored['context_source'] == 'store'
    assert stored['system'] == ep['system'] and stored['fidelity'] == ep['fidelity']


def test_marked_episode_with_only_its_logged_prompt_uses_the_event(monkeypatch):
    monkeypatch.setattr(F, '_events_system', lambda *a: 'actual reference prompt')
    s = SimpleNamespace(store={'CodingEpisode:simulator_context_mode': 'reference'})
    assert F.episode_context({'condition': 'C2'}, s, object()) == ('actual reference prompt', 'events', 'reference')


def test_unmarked_c2_whose_logged_prompt_has_the_reference_is_graded_as_reference(monkeypatch):
    monkeypatch.setattr(F, '_events_system', lambda *a: 'snapshot\n\n' + F.REFERENCE_MARK + ', from the same state')
    s = SimpleNamespace(store={})
    assert F.episode_context({'condition': 'C2'}, s, object())[2] == 'reference'


def test_reference_mark_matches_the_prompt():
    from simtest.coding import prompts
    assert F.REFERENCE_MARK in prompts.reference_block([{'command': 'ls', 'output': 'x'}])
