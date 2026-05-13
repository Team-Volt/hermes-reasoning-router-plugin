from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    """Keep tests from reading the live ~/.hermes reasoning-router config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def load_plugin():
    spec = importlib.util.spec_from_file_location("reasoning_router_plugin", PLUGIN_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeGateway:
    def __init__(self, config=None):
        self.config_data = config or {}
        self.calls = []

    def _session_key_for_source(self, source):
        return f"{source.platform.value}:{source.user_id}:{source.chat_id}:{source.thread_id or ''}"

    def _set_session_reasoning_override(self, session_key, reasoning_config):
        self.calls.append((session_key, reasoning_config))


class FakeSessionStore:
    def __init__(self, session_key: str, session_id: str):
        self._entries = {session_key: SimpleNamespace(session_id=session_id)}
        self.loaded = False

    def _ensure_loaded(self):
        self.loaded = True


def event(text: str):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="discord"),
        user_id="user-1",
        chat_id="chat-1",
        thread_id="thread-1",
    )
    return SimpleNamespace(text=text, source=source, internal=False)


def test_quick_question_routes_low():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})

    result = plugin.pre_gateway_dispatch(event("what time is it?"), gateway=gateway)

    assert result == {"action": "allow"}
    assert gateway.calls == [
        (
            "discord:user-1:chat-1:thread-1",
            {"enabled": True, "effort": "low"},
        )
    ]


def test_simple_code_change_routes_high():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})

    result = plugin.pre_gateway_dispatch(
        event("Patch the plugin status text and run the focused tests"),
        gateway=gateway,
    )

    assert result == {"action": "allow"}
    assert gateway.calls == [
        (
            "discord:user-1:chat-1:thread-1",
            {"enabled": True, "effort": "high"},
        )
    ]


def test_complex_multi_system_work_routes_xhigh():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})

    result = plugin.pre_gateway_dispatch(
        event(
            "Flesh out the Hermes reasoning-router plugin, add persistent logs, "
            "update config, restart the gateway, and be thorough"
        ),
        gateway=gateway,
    )

    assert result == {"action": "allow"}
    assert gateway.calls == [
        (
            "discord:user-1:chat-1:thread-1",
            {"enabled": True, "effort": "xhigh"},
        )
    ]


def test_slash_commands_are_left_alone():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})

    result = plugin.pre_gateway_dispatch(event("/reasoning high"), gateway=gateway)

    assert result == {"action": "allow"}
    assert gateway.calls == []


def test_config_caps_effort():
    plugin = load_plugin()
    gateway = FakeGateway(
        {"reasoning_router": {"enabled": True, "min": "low", "max": "medium"}}
    )

    result = plugin.pre_gateway_dispatch(
        event("Migrate the database schema and patch the provider transport"),
        gateway=gateway,
    )

    assert result == {"action": "allow"}
    assert gateway.calls == [
        (
            "discord:user-1:chat-1:thread-1",
            {"enabled": True, "effort": "medium"},
        )
    ]


def test_default_config_allows_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Investigate the auth migration failure, patch the gateway transport, and verify rollback safety"
    )

    assert effort == "xhigh"
    assert "xhigh" in reason or "multiple" in reason



def test_short_technical_feasibility_followup_routes_medium():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Does doing that require modifying hermes source?"
    )

    assert effort == "medium"
    assert "technical feasibility" in reason


def test_honest_opinion_request_routes_medium_not_high_or_low():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "I want your honest opinion on gbrain. Is there actually value there?"
    )

    assert effort == "medium"
    assert "opinion" in reason


def test_memory_system_content_migration_routes_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Copy the contents of gbrain in to hindsight if we don’t already have the fact/info in hindsight"
    )

    assert effort == "xhigh"
    assert "xhigh" in reason or "memory" in reason


def test_apply_tweak_approval_routes_high():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Yeah that’s what I meant. Go ahead and apply whatever tweak you recommend to prevent under routing again"
    )

    assert effort == "high"
    assert "implementation approval" in reason


def test_go_ahead_fork_and_start_working_routes_high():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Go ahead and fork the repo and start working on it"
    )

    assert effort == "high"
    assert "approval" in reason


def test_delete_fork_and_update_ha_rollout_routes_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "You can delete the personal fork. You can also go ahead and update HA to use the new fork and we can see how it works out"
    )

    assert effort == "xhigh"
    assert "xhigh" in reason or "rollout" in reason


def test_merge_fork_and_update_ha_from_main_routes_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "If you're happy with it and it works, you can make PR's and merge them into our fork, and then update HA to properly use the fork and not a branch"
    )

    assert effort == "xhigh"
    assert "xhigh" in reason or "rollout" in reason


def test_careful_lcm_db_lifecycle_review_routes_high():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Carefully review the LCM DB to see how to deal with the lifecycle fragmentation"
    )

    assert effort == "high"
    assert "diagnostic review" in reason


def test_readme_wording_with_restart_terms_routes_medium():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "The prompt in the readme is a bit overly descriptive, and keep in mind that not everyone uses Linux and systemctl. Some people are on macOS\n\n"
        "I think you could be more generic and say something like ask the user to restart the gateway etc\n\n"
        "Take a look at the Hermes-lcm readme and be more like that - standardized"
    )

    assert effort == "medium"
    assert "documentation wording" in reason


def test_ha_fork_docs_wording_routes_medium_not_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Update the HA docs to mention the new fork"
    )

    assert effort == "medium"
    assert "documentation wording" in reason


def test_security_docs_wording_can_still_route_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Patch the README security wording for OAuth token handling and permission boundaries"
    )

    assert effort == "xhigh"
    assert "xhigh" in reason


def test_short_ordinary_question_still_routes_low():
    plugin = load_plugin()

    effort, reason = plugin.classify_message("what time is it?")

    assert effort == "low"
    assert "quick" in reason


def test_explicit_set_snippet_intro_routes_medium_not_low():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Set this one please:\n"
        "```display:\n"
        "  background_process_notifications: off```"
    )

    assert effort == "medium"
    assert "config snippet" in reason


def test_service_shutdown_request_routes_xhigh_not_low():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "I think at this point we can shut down the gbrain mcp"
    )

    assert effort == "xhigh"
    assert "service-control" in reason or "xhigh" in reason


def test_docs_restart_wording_stays_medium_after_service_control_tweak():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "In the README, say the user may need to restart the gateway after installation"
    )

    assert effort == "medium"
    assert "documentation wording" in reason


def test_pr_and_merge_request_routes_high_not_low():
    plugin = load_plugin()

    effort, reason = plugin.classify_message("You should make a PR and merge it")

    assert effort == "high"
    assert "github workflow" in reason


def test_backup_all_skills_remove_any_mentions_routes_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "Back up our current skills. Look through all of them and remove any mention of the retired archive. We're done with that thing"
    )

    assert effort == "xhigh"
    assert "xhigh" in reason


def test_single_file_remove_mention_does_not_route_xhigh():
    plugin = load_plugin()

    effort, reason = plugin.classify_message(
        "In the README, remove any mention of the old option"
    )

    assert effort == "medium"
    assert "documentation wording" in reason


def test_semantic_classifier_is_disabled_by_default(monkeypatch):
    plugin = load_plugin()
    calls = []

    def fake_semantic_classifier(*args, **kwargs):
        calls.append((args, kwargs))
        return {"effort": "medium", "confidence": 0.99, "reason": "fake"}

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    effort, reason = plugin.classify_message("Set this one please")

    assert calls == []
    assert effort == "low"
    assert "quick" in reason


def test_semantic_classifier_can_raise_ambiguous_short_request(monkeypatch):
    plugin = load_plugin()

    def fake_semantic_classifier(text, config):
        assert text == "Set this one please"
        assert config["semantic_classifier_model"] == "gpt-5.4-mini"
        return {
            "effort": "medium",
            "confidence": 0.91,
            "risk_categories": ["config_change"],
            "reason": "short imperative likely asks to change config",
        }

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    effort, reason = plugin.classify_message(
        "Set this one please",
        {"semantic_classifier_enabled": True},
    )

    assert effort == "medium"
    assert "semantic classifier" in reason
    assert "config_change" in reason


def test_semantic_classifier_low_confidence_falls_back(monkeypatch):
    plugin = load_plugin()

    def fake_semantic_classifier(text, config):
        return {
            "effort": "high",
            "confidence": 0.42,
            "risk_categories": ["uncertain"],
            "reason": "not sure",
        }

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    effort, reason = plugin.classify_message(
        "Set this one please",
        {"semantic_classifier_enabled": True, "semantic_classifier_min_confidence": 0.75},
    )

    assert effort == "low"
    assert "quick" in reason


def test_semantic_classifier_does_not_lower_or_call_for_obvious_xhigh(monkeypatch):
    plugin = load_plugin()
    calls = []

    def fake_semantic_classifier(*args, **kwargs):
        calls.append((args, kwargs))
        return {"effort": "low", "confidence": 0.99, "reason": "fake"}

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    effort, reason = plugin.classify_message(
        "Please restart the gateway",
        {"semantic_classifier_enabled": True},
    )

    assert calls == []
    assert effort == "xhigh"
    assert "xhigh" in reason


def test_semantic_classifier_can_lower_question_false_positive(monkeypatch):
    plugin = load_plugin()
    calls = []

    def fake_semantic_classifier(text, config):
        calls.append((text, config))
        return {
            "effort": "medium",
            "confidence": 0.93,
            "risk_categories": ["service_restart_question"],
            "reason": "asking whether restart is needed, not requesting restart",
        }

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    effort, reason = plugin.classify_message(
        "Do you need me to restart the gateway?",
        {"semantic_classifier_enabled": True, "semantic_classifier_min_confidence": 0.75},
    )

    assert len(calls) == 1
    assert effort == "medium"
    assert "lowered" in reason
    assert "service_restart_question" in reason


def test_semantic_classifier_does_not_lower_action_request(monkeypatch):
    plugin = load_plugin()
    calls = []

    def fake_semantic_classifier(text, config):
        calls.append((text, config))
        return {"effort": "medium", "confidence": 0.99, "reason": "too low"}

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    effort, reason = plugin.classify_message(
        "Please restart the gateway",
        {"semantic_classifier_enabled": True},
    )

    assert calls == []
    assert effort == "xhigh"
    assert "xhigh" in reason


def test_semantic_classifier_messages_include_compact_recent_context():
    plugin = load_plugin()

    messages = plugin._semantic_classifier_messages(
        "Go ahead and set up the automation",
        {
            "last_assistant_intent": "Asked whether to create cron automation",
            "recent_messages": [
                {"role": "tool", "content": "secret tool output should be ignored"},
                {"role": "user", "content": "Can old sessions be pruned automatically?"},
                {"role": "assistant", "content": "We can create a cron job."},
                {"role": "user", "content": "Go ahead"},
                {"role": "assistant", "content": "I will create the cron job."},
            ],
            "pending_action": "create_cron_job",
        },
    )
    payload = json.loads(messages[1]["content"])

    assert payload["current_user_message"] == "Go ahead and set up the automation"
    assert payload["last_assistant_intent"] == "Asked whether to create cron automation"
    assert payload["pending_action"] == "create_cron_job"
    assert [item["role"] for item in payload["recent_messages"]] == ["assistant", "user", "assistant"]
    assert "tool output" not in messages[1]["content"]


def test_semantic_classifier_today_snippet_expectations(monkeypatch):
    plugin = load_plugin()

    fake_outputs = {
        "Do you need me to restart the gateway?": {"effort": "medium", "confidence": 0.93, "risk_categories": ["service_restart_question"], "reason": "question only"},
        "So you're saying we feed all prompts through something like gpt-5.4-mini first, to determine what reasoning level the task should actually complete as?": {"effort": "medium", "confidence": 0.92, "risk_categories": ["design_clarification"], "reason": "clarification"},
    }

    def fake_semantic_classifier(text, config):
        return fake_outputs[text]

    monkeypatch.setattr(plugin, "_semantic_classify_with_codex_proxy", fake_semantic_classifier)

    cases = [
        ("Do you need me to restart the gateway?", "medium"),
        ("So you're saying we feed all prompts through something like gpt-5.4-mini first, to determine what reasoning level the task should actually complete as?", "medium"),
        ("You should make a PR and merge it", "high"),
        ("I think at this point we can shut down the gbrain mcp", "xhigh"),
    ]

    for text, expected in cases:
        effort, _reason = plugin.classify_message(
            text,
            {"semantic_classifier_enabled": True, "semantic_classifier_min_confidence": 0.75},
        )
        assert effort == expected


def test_semantic_classifier_prompt_is_minimal_and_json_only():
    plugin = load_plugin()

    messages = plugin._semantic_classifier_messages(
        "Set this one please",
        {"last_assistant_intent": "apply a config change"},
    )
    serialized = json.dumps(messages)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Return JSON only" in messages[0]["content"]
    assert "Hermes Agent" not in serialized
    assert "SOUL.md" not in serialized
    assert "tool" not in serialized.lower()
    assert "apply a config change" in serialized


def test_disabled_router_does_nothing():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": False}})

    result = plugin.pre_gateway_dispatch(event("Migrate the database schema"), gateway=gateway)

    assert result == {"action": "allow"}
    assert gateway.calls == []


def test_reasoning_router_command_status_reports_state(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "reasoning-router" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "min": "low",
                "default": "medium",
                "max": "high",
                "log_decisions": True,
            }
        )
    )

    output = plugin.reasoning_router_command("status")

    assert "Reasoning router: on" in output
    assert "min=low" in output
    assert "default=medium" in output
    assert "max=high" in output


def test_reasoning_router_command_status_defaults_to_xhigh(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({}))

    output = plugin.reasoning_router_command("status")

    assert "max=xhigh" in output
    assert "decision_log=" in output


def test_config_path_is_outside_plugin_directory(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert plugin._config_path() == tmp_path / "reasoning-router" / "config.yaml"
    assert plugin._config_path() != tmp_path / "plugins" / "reasoning-router" / "config.yaml"


def test_standalone_config_wins_over_legacy_plugin_config(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    standalone = tmp_path / "reasoning-router" / "config.yaml"
    legacy = tmp_path / "plugins" / "reasoning-router" / "config.yaml"
    standalone.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    standalone.write_text(yaml.safe_dump({"max": "xhigh", "decision_log": True}))
    legacy.write_text(yaml.safe_dump({"max": "medium", "decision_log": False}))

    cfg = plugin._read_router_config_from_disk()

    assert cfg["max"] == "xhigh"
    assert cfg["decision_log"] is True


def test_legacy_plugin_config_is_fallback_when_standalone_missing(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    legacy = tmp_path / "plugins" / "reasoning-router" / "config.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(yaml.safe_dump({"max": "medium", "decision_log": True}))

    cfg = plugin._read_router_config_from_disk()

    assert cfg["max"] == "medium"
    assert cfg["decision_log"] is True


def test_main_config_is_fallback_after_standalone_and_legacy_missing(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"reasoning_router": {"max": "high", "decision_log": True}})
    )

    cfg = plugin._read_router_config_from_disk()

    assert cfg["max"] == "high"
    assert cfg["decision_log"] is True


def test_reasoning_router_command_toggles_config(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["reasoning-router"]}}))

    off_output = plugin.reasoning_router_command("off")
    plugin_cfg = yaml.safe_load((tmp_path / "reasoning-router" / "config.yaml").read_text())
    assert off_output == "Reasoning router disabled. Use `/reasoning-router on` to re-enable."
    assert plugin_cfg["enabled"] is False

    on_output = plugin.reasoning_router_command("on")
    plugin_cfg = yaml.safe_load((tmp_path / "reasoning-router" / "config.yaml").read_text())
    assert on_output == "Reasoning router enabled."
    assert plugin_cfg["enabled"] is True


def test_reasoning_router_command_updates_max_and_test_classifies(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"reasoning_router": {"enabled": True, "min": "low", "max": "high"}})
    )

    max_output = plugin.reasoning_router_command("max medium")
    plugin_cfg = yaml.safe_load((tmp_path / "reasoning-router" / "config.yaml").read_text())
    assert max_output == "Reasoning router max effort set to medium."
    assert plugin_cfg["max"] == "medium"

    test_output = plugin.reasoning_router_command("test Migrate the database schema")
    assert "would route to medium" in test_output
    assert "high" in test_output


def test_persistent_decision_log_jsonl(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gateway = FakeGateway(
        {
            "reasoning_router": {
                "enabled": True,
                "decision_log": True,
                "decision_log_path": "logs/reasoning-router.jsonl",
            }
        }
    )

    result = plugin.pre_gateway_dispatch(
        event("Design a rollback-safe migration plan for the auth gateway"), gateway=gateway
    )

    assert result == {"action": "allow"}
    log_path = tmp_path / "logs" / "reasoning-router.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows[-1]["session_key"] == "discord:user-1:chat-1:thread-1"
    assert rows[-1]["effort"] == "xhigh"
    assert rows[-1]["platform"] == "discord"
    assert "message_preview" in rows[-1]


def test_pending_affirmation_inherits_prior_xhigh_intent():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})
    session_key = "discord:user-1:chat-1:thread-1"
    store = FakeSessionStore(session_key, "session-1")

    plugin.post_llm_call(
        session_id="session-1",
        user_message="Plan a rollback-safe architecture migration for the Hermes gateway plugin.",
        assistant_response=(
            "Here is the full implementation plan with rollback safety and tests. "
            "Want me to proceed with implementing the project end to end?"
        ),
        platform="discord",
    )

    result = plugin.pre_gateway_dispatch(event("yes"), gateway=gateway, session_store=store)

    assert result == {"action": "allow"}
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "xhigh"})
    assert gateway._reasoning_router_decisions[session_key]["message_preview"] == "yes"
    assert gateway._reasoning_router_decisions[session_key]["pending_task_preview"]
    assert "affirmed pending" in gateway._reasoning_router_decisions[session_key]["reason"]

    gateway.calls.clear()
    plugin.pre_gateway_dispatch(event("yes"), gateway=gateway, session_store=store)
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "low"})


def test_next_step_approval_inherits_prior_xhigh_recommendation():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})
    session_key = "discord:user-1:chat-1:thread-1"
    store = FakeSessionStore(session_key, "session-1")

    plugin.post_llm_call(
        session_id="session-1",
        user_message="Carefully review the LCM DB to deal with lifecycle fragmentation using xhigh reasoning.",
        assistant_response=(
            "Next step: phase 2 should be read-only classification of the remaining lifecycle rows, "
            "split cron-owned Discord payload rows from orphan payload rows, and produce repair candidates."
        ),
        platform="discord",
    )

    result = plugin.pre_gateway_dispatch(
        event("Go ahead and do the next step"), gateway=gateway, session_store=store
    )

    assert result == {"action": "allow"}
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "xhigh"})
    decision = gateway._reasoning_router_decisions[session_key]
    assert decision["message_preview"] == "Go ahead and do the next step"
    assert decision["pending_task_preview"]
    assert "affirmed pending" in decision["reason"]


def test_pending_rejection_clears_without_inheritance():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})
    session_key = "discord:user-1:chat-1:thread-1"
    store = FakeSessionStore(session_key, "session-1")

    plugin.post_llm_call(
        session_id="session-1",
        user_message="Design a multi-system auth migration with rollback safety.",
        assistant_response="Want me to proceed with implementing the migration now?",
        platform="discord",
    )

    plugin.pre_gateway_dispatch(event("no"), gateway=gateway, session_store=store)
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "low"})

    gateway.calls.clear()
    plugin.pre_gateway_dispatch(event("yes"), gateway=gateway, session_store=store)
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "low"})


def test_substantive_new_request_clears_pending_intent():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})
    session_key = "discord:user-1:chat-1:thread-1"
    store = FakeSessionStore(session_key, "session-1")

    plugin.post_llm_call(
        session_id="session-1",
        user_message="Plan a production deployment and rollback-safe config migration.",
        assistant_response="Want me to proceed with deploying the changes?",
        platform="discord",
    )

    plugin.pre_gateway_dispatch(event("what time is it?"), gateway=gateway, session_store=store)
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "low"})

    gateway.calls.clear()
    plugin.pre_gateway_dispatch(event("yes"), gateway=gateway, session_store=store)
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "low"})


def test_plain_answer_does_not_arm_pending_intent():
    plugin = load_plugin()
    gateway = FakeGateway({"reasoning_router": {"enabled": True}})
    session_key = "discord:user-1:chat-1:thread-1"
    store = FakeSessionStore(session_key, "session-1")

    plugin.post_llm_call(
        session_id="session-1",
        user_message="What is the status?",
        assistant_response="The service is running normally.",
        platform="discord",
    )

    plugin.pre_gateway_dispatch(event("yes"), gateway=gateway, session_store=store)
    assert gateway.calls[-1] == (session_key, {"enabled": True, "effort": "low"})


def test_register_adds_hook_and_discord_slash_command():
    plugin = load_plugin()
    calls = []

    class Ctx:
        def register_hook(self, name, handler):
            calls.append(("hook", name, handler.__name__))

        def register_command(self, name, handler, description="", args_hint=""):
            calls.append(("command", name, handler.__name__, description, args_hint))

    plugin.register(Ctx())

    assert ("hook", "pre_gateway_dispatch", "pre_gateway_dispatch") in calls
    assert ("hook", "post_llm_call", "post_llm_call") in calls
    assert any(call[:3] == ("command", "reasoning-router", "reasoning_router_command") for call in calls)
    assert any("threshold" in call[-1] and "recent" in call[-1] for call in calls if call[0] == "command")


def test_reasoning_router_command_threshold_and_recent(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "reasoning_router": {
                    "enabled": True,
                    "decision_log": True,
                    "decision_log_path": "logs/reasoning-router.jsonl",
                }
            }
        )
    )

    threshold_output = plugin.reasoning_router_command("threshold 2")
    plugin_cfg = yaml.safe_load((tmp_path / "reasoning-router" / "config.yaml").read_text())
    assert threshold_output == "Reasoning router xhigh threshold set to 2 high-complexity categories."
    assert plugin_cfg["xhigh_high_match_threshold"] == 2

    gateway = FakeGateway({"reasoning_router": plugin._read_router_config_from_disk()})
    plugin.pre_gateway_dispatch(event("Patch the plugin and run tests"), gateway=gateway)

    recent = plugin.reasoning_router_command("recent 1")
    assert "Recent reasoning-router decisions:" in recent
    assert "xhigh" in recent
    assert "Patch the plugin" in recent
