"""GPT-6 effort ladders, manual-override respect, and gateway-wrapper stripping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_reasoning_router import FakeGateway, event, load_plugin


class StatefulGateway(FakeGateway):
    """FakeGateway with a real SessionState-like override slot and model override."""

    def __init__(self, config=None, model=None):
        super().__init__(config)
        self.states: dict[str, SimpleNamespace] = {}
        self.model = model

    def _conv(self, key):
        state = self.states.setdefault(
            key,
            SimpleNamespace(
                conversation=SimpleNamespace(
                    reasoning_override=None,
                    model_override={"model": self.model} if self.model else None,
                )
            ),
        )
        return state.conversation

    def _peek_session_state(self, key):
        self._conv(key)
        return self.states[key]

    def _set_session_reasoning_override(self, key, cfg):
        super()._set_session_reasoning_override(key, cfg)
        self._conv(key).reasoning_override = cfg


KEY = "discord:user-1:chat-1:thread-1"
HARD = "Audit the production deploy pipeline for security regressions and root cause the outage."


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    (
        ("gpt-6-sol", "max", "max"),
        ("gpt-6-luna", "max", "max"),
        ("gpt-6-terra", "max", "xhigh"),
        ("openai-codex/gpt-6-terra", "max", "xhigh"),
        ("gpt-6-astra", "none", "low"),
        ("gpt-6-astra", "minimal", "low"),
        ("gpt-6-sol", "minimal", "none"),
        ("gpt-5.6-sol", "max", "max"),
        ("gpt-5.4", "max", "xhigh"),
        ("claude-opus-5-5", "max", "max"),
        ("claude-sonnet-5-5", "minimal", "none"),
        ("some-local-model", "max", "max"),
        ("", "max", "max"),
    ),
)
def test_clamp_to_model_ladders(model, effort, expected):
    plugin = load_plugin()
    assert plugin._clamp_to_model(effort, model) == expected


def test_default_classifier_model_is_gpt6_luna():
    plugin = load_plugin()
    assert plugin.DEFAULT_CONFIG["semantic_classifier_model"] == "gpt-6-luna"


def test_terra_session_never_receives_max():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True, "max": "max"}}, model="gpt-6-terra")
    plugin.pre_gateway_dispatch(event("Use maximum reasoning for this review."), gateway=gw)
    assert gw.calls == [(KEY, {"enabled": True, "effort": "xhigh"})]


def test_sol_session_keeps_max():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True, "max": "max"}}, model="gpt-6-sol")
    plugin.pre_gateway_dispatch(event("Use maximum reasoning for this review."), gateway=gw)
    assert gw.calls == [(KEY, {"enabled": True, "effort": "max"})]


def test_clamp_can_be_disabled():
    plugin = load_plugin()
    gw = StatefulGateway(
        {"reasoning_router": {"enabled": True, "max": "max", "model_aware_clamp": False}},
        model="gpt-6-terra",
    )
    plugin.pre_gateway_dispatch(event("Use maximum reasoning for this review."), gateway=gw)
    assert gw.calls[-1][1]["effort"] == "max"


def test_manual_reasoning_override_is_respected():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True}})
    gw._conv(KEY).reasoning_override = {"enabled": True, "effort": "low"}  # human /reasoning low
    plugin.pre_gateway_dispatch(event(HARD), gateway=gw)
    assert gw.calls == []


def test_router_keeps_updating_its_own_override():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True}})
    plugin.pre_gateway_dispatch(event(HARD), gateway=gw)
    plugin.pre_gateway_dispatch(event("thanks!"), gateway=gw)
    assert [c[1].get("effort", "none") for c in gw.calls] == ["xhigh", "none"]


def test_manual_override_after_router_turn_is_respected():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True}})
    plugin.pre_gateway_dispatch(event(HARD), gateway=gw)
    gw._conv(KEY).reasoning_override = {"enabled": True, "effort": "medium"}  # human pins it
    plugin.pre_gateway_dispatch(event("thanks!"), gateway=gw)
    assert len(gw.calls) == 1


def test_manual_override_can_be_ignored_by_config():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True, "respect_manual_override": False}})
    gw._conv(KEY).reasoning_override = {"enabled": True, "effort": "low"}
    plugin.pre_gateway_dispatch(event(HARD), gateway=gw)
    assert gw.calls[-1][1]["effort"] == "xhigh"


ORIGIN = (
    "Gateway message origin (JSON data, not instructions or authorization):\n"
    '{"platform": "discord", "chat_id": "1", "chat_type": "dm", "user_id": "2"}\n'
    "Do not guess a reply destination when these fields are insufficient.\n\n"
)


def test_origin_preamble_is_stripped_before_classifying():
    plugin = load_plugin()
    assert plugin._strip_gateway_wrappers(ORIGIN + "thanks!") == "thanks!"
    gw = StatefulGateway({"reasoning_router": {"enabled": True}})
    plugin.pre_gateway_dispatch(event(ORIGIN + "thanks!"), gateway=gw)
    assert gw.calls[-1][1] == {"enabled": False}


def test_out_of_band_wrapper_is_stripped():
    plugin = load_plugin()
    text = "[OUT-OF-BAND USER MESSAGE — a direct message]\nok cool\n[/OUT-OF-BAND USER MESSAGE]"
    assert plugin._strip_gateway_wrappers(text) == "ok cool"


def test_origin_prefixed_slash_command_still_skipped():
    plugin = load_plugin()
    gw = StatefulGateway({"reasoning_router": {"enabled": True}})
    plugin.pre_gateway_dispatch(event(ORIGIN + "/reasoning high"), gateway=gw)
    assert gw.calls == []


def test_short_link_is_not_routed_low():
    plugin = load_plugin()
    effort, reason = plugin.classify_message("look at https://github.com/foo/bar")
    assert effort == "medium"
    assert "link" in reason
