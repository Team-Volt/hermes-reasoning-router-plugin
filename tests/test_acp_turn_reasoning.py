"""ACP sentinel policy is ephemeral and independent of the messaging adapter."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from test_reasoning_router import load_plugin


@pytest.fixture
def router():
    plugin = load_plugin()
    plugin._RUNTIME_CONFIG_OVERRIDE = {'acp_none_means_auto': True, 'log_decisions': False}
    return plugin


def turn(router, **changes):
    payload = dict(platform='acp', profile_home='/isolated/profile', session_id='s', turn_id='t',
                   user_message='hello', conversation_history=[], model='gpt-6.1-sol',
                   reasoning={'source': 'session', 'pinned': True, 'raw_effort': 'none', 'config': {'enabled': False}})
    payload.update(changes)
    return router.select_turn_reasoning(**payload)


@pytest.mark.parametrize('raw,pinned,source', [
    ('none', False, 'default'), (False, True, 'session'), ('disabled', True, 'session'),
    ('false', True, 'session'), ('default', False, 'default'), (None, False, 'default'),
    ({'enabled': False}, True, 'session'), ('low', True, 'session'), ('medium', True, 'session'),
    ('high', True, 'session'), ('xhigh', True, 'session'), ('max', True, 'session'),
    ('none', True, 'request'), ('None', True, 'session'), (' none ', True, 'session'),
])
def test_only_original_literal_session_none_is_consent(router, raw, pinned, source, monkeypatch):
    monkeypatch.setattr(router, 'classify_message', lambda *_a: pytest.fail('manual/default input was classified'))
    assert turn(router, reasoning={'source': source, 'pinned': pinned, 'raw_effort': raw}) is None


@pytest.mark.parametrize('setting', [False, 'true', 1, None])
def test_option_requires_an_actual_boolean_opt_in(router, setting):
    router._RUNTIME_CONFIG_OVERRIDE['acp_none_means_auto'] = setting
    assert turn(router) is None


def test_easy_hard_easy_and_no_policy_feedback(router):
    policy = {'source': 'session', 'pinned': True, 'raw_effort': 'none', 'config': {'enabled': False}}
    efforts = [turn(router, turn_id=str(i), user_message=text, reasoning=policy) for i, text in enumerate(
        ('hello', 'Implement the gateway fix and run the regression tests', 'hello'))]
    assert efforts[0] == efforts[2] == 'low'
    assert efforts[1] in {'high', 'xhigh'}
    assert policy == {'source': 'session', 'pinned': True, 'raw_effort': 'none', 'config': {'enabled': False}}


def test_pending_intent_is_profile_scoped_and_consumed_once(router, monkeypatch):
    monkeypatch.setattr(router, '_hermes_home', lambda: Path('/profile/a'))
    router.post_llm_call(session_id='s', platform='acp', user_message='Implement the code',
                         assistant_response='I can implement the gateway fix. Should I proceed?', conversation_history=[])
    assert turn(router, profile_home='/profile/b', user_message='yes') == 'low'
    assert turn(router, profile_home='/profile/a', user_message='yes') in {'high', 'xhigh'}
    assert turn(router, profile_home='/profile/a', user_message='yes', turn_id='t2') == 'low'


@pytest.mark.parametrize('changes', [{'shadow_mode': True}, {'enabled': False}])
def test_inactive_modes_preserve_native_reasoning(router, changes):
    router._RUNTIME_CONFIG_OVERRIDE.update(changes)
    assert turn(router) is None


@pytest.mark.parametrize('platform', ['discord', 'telegram', 'buzz', 'cli', 'local'])
def test_turn_hook_never_reclassifies_other_ingress(router, platform, monkeypatch):
    monkeypatch.setattr(router, 'classify_message', lambda *_a: pytest.fail('double route'))
    assert turn(router, platform=platform) is None


def test_old_host_retains_messaging_hooks(router, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'agent.turn_reasoning', None)
    hooks = {}
    ctx = SimpleNamespace(register_hook=lambda name, fn: hooks.update({name: fn}), register_command=lambda *_a, **_k: None)
    router.register(ctx)
    assert hooks['pre_gateway_dispatch'] is router.pre_gateway_dispatch
    assert hooks['post_llm_call'] is router.post_llm_call
    assert 'select_turn_reasoning' not in hooks
