"""Regression tests for lifecycle and config defects found in code review."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from test_gpt6_overrides import KEY, StatefulGateway
from test_reasoning_router import FakeSessionStore, event, load_plugin

SID = "session-1"
HARD = "Audit the production deploy pipeline for security regressions and root cause the outage."


def _history(tool_calls: int):
    history = [{"role": "user", "content": "do the thing"}]
    for i in range(tool_calls):
        history.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    history.append({"role": "assistant", "content": "Done."})
    return history


def _cfg_file(plugin):
    path = plugin._config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_reasoning_display_toggle_does_not_disown_router_override():
    plugin = load_plugin()
    gateway, store = StatefulGateway(), FakeSessionStore(KEY, SID)
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=store)
    for toggle in ("/reasoning show", "/reasoning hide", "/reasoning"):
        plugin.pre_gateway_dispatch(event(toggle), gateway=gateway, session_store=store)
    plugin.pre_gateway_dispatch(event(HARD), gateway=gateway, session_store=store)
    assert gateway.calls[-1][1].get("effort") in {"high", "xhigh"}


def test_reasoning_level_command_still_counts_as_manual():
    plugin = load_plugin()
    assert plugin._reasoning_command_changes_override("/reasoning high")
    assert plugin._reasoning_command_changes_override("/reasoning reset")
    assert plugin._reasoning_command_changes_override("/reasoning --global xhigh")
    assert not plugin._reasoning_command_changes_override("/reasoning off")


def test_session_key_uses_host_source_normalization():
    plugin = load_plugin()

    class Gateway:
        def _normalize_source_for_session_key(self, source):
            return SimpleNamespace(**{**vars(source), "thread_id": "topic-42"})

        def _session_key_for_source(self, source):
            return f"telegram:{source.user_id}:{source.chat_id}:{source.thread_id or ''}"

    key = plugin._session_key_for(event("hi", platform="telegram", thread_id=None), Gateway())
    assert key.endswith(":topic-42")


def test_broken_config_releases_router_override():
    plugin = load_plugin()
    gateway, store = StatefulGateway(), FakeSessionStore(KEY, SID)
    path = _cfg_file(plugin)
    path.write_text("enabled_platforms: [discord]\n")
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=store)
    assert gateway._conv(KEY).reasoning_override is not None
    path.write_text("foo: [unclosed\n")
    plugin.pre_gateway_dispatch(event("thanks"), gateway=gateway, session_store=store)
    assert gateway._conv(KEY).reasoning_override is None


def test_router_command_refuses_to_overwrite_broken_config():
    plugin = load_plugin()
    path = _cfg_file(plugin)
    original = "min: high\nenabled_platforms: [discord]\nfoo: [unclosed\n"
    path.write_text(original)
    reply = plugin.reasoning_router_command("on")
    assert "unreadable" in reply
    assert path.read_text() == original
    assert "NOT ROUTING" in plugin.reasoning_router_command("status")


def test_config_cache_notices_same_size_edit():
    plugin = load_plugin()
    path = _cfg_file(plugin)
    path.write_text("default: high\n")
    assert plugin._read_router_config_from_disk()["default"] == "high"
    stat = path.stat()
    path.write_text("default: none\n")
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert plugin._read_router_config_from_disk()["default"] == "none"


def test_config_error_is_per_thread():
    plugin = load_plugin()
    path = _cfg_file(plugin)
    path.write_text("foo: [unclosed\n")
    plugin._read_router_config_from_disk()
    assert plugin._config_error()
    seen = []
    thread = threading.Thread(target=lambda: seen.append(plugin._config_error()))
    thread.start()
    thread.join()
    assert seen == [None]


def test_bad_typed_config_values_do_not_crash_routing():
    plugin = load_plugin()
    gateway, store = StatefulGateway(), FakeSessionStore(KEY, SID)
    path = _cfg_file(plugin)
    for body in ("live_system_terms: 5\n", "live_system_terms: true\n", "momentum_ttl_minutes: 1e30\n"):
        path.write_text("enabled_platforms: [discord]\n" + body)
        plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                             conversation_history=_history(20), platform="discord")
        plugin.pre_gateway_dispatch(event("check the server"), gateway=gateway, session_store=store)
        assert gateway.calls[-1][1].get("enabled") is True


def test_momentum_does_not_lift_unrelated_new_topics():
    plugin = load_plugin()
    for text in ("what is the capital of france", "good morning", "how tall is mount everest"):
        plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                             conversation_history=_history(20), platform="discord")
        effort, _ = plugin._apply_momentum(text, "low", "quick", SID, {})
        assert effort == "low", text
    plugin.post_llm_call(session_id=SID, user_message="x", assistant_response="Done.",
                         conversation_history=_history(20), platform="discord")
    assert plugin._apply_momentum("yes", "low", "quick", SID, {})[0] == "high"


def test_oob_unwrap_is_linear_on_malformed_wrappers():
    import time

    plugin = load_plugin()
    wrapped = "[OUT-OF-BAND USER MESSAGE — x]\nhello there\n[/OUT-OF-BAND USER MESSAGE]"
    assert plugin._strip_gateway_wrappers(wrapped) == "hello there"
    start = time.perf_counter()
    plugin._strip_gateway_wrappers("[OUT-OF-BAND USER MESSAGE]\n" + " \n" * 50_000)
    assert time.perf_counter() - start < 0.1
