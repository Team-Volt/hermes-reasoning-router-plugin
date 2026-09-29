"""Reproducible routing accuracy for the synthetic eval sets.

usage: python eval/evaluate.py [--show] [--config eval/config.yaml]

Sets (all synthetic or paraphrased; no real chat text):
  eval/tuning.json       labeled messages used while tuning the rules
  eval/holdout.json      labeled messages held out from tuning
  eval/adversarial.json  misroutes found in an adversarial pass, with the accepted range

Each labeled item is routed through ``pre_gateway_dispatch`` on a fresh fake
gateway, so pending intents and session momentum behave as in Hermes. Items with
``context: mid_task`` first simulate a tool-heavy turn; ``idle`` items follow a
plain "thanks!". Adversarial items go through ``classify_message`` and pass when
the effort falls inside the accepted range (for example ``medium-xhigh``).
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
MID_TASK_ASK = "can you fix the backup job on the nas and verify it runs"
MID_TASK_REPLY = "I patched the job and it ran once. It still fails on the second share."


class _Gateway:
    def __init__(self, model: str):
        self.config_data: dict = {}
        self.states: dict = {}
        self.model = model

    def _conv(self, key):
        state = self.states.setdefault(
            key,
            SimpleNamespace(conversation=SimpleNamespace(reasoning_override=None, model_override={"model": self.model})),
        )
        return state.conversation

    def _peek_session_state(self, key):
        self._conv(key)
        return self.states[key]

    def _session_key_for_source(self, source):
        return f"agent:main:discord:dm:{source.chat_id}"

    def _set_session_reasoning_override(self, key, cfg):
        self._conv(key).reasoning_override = None if cfg is None else dict(cfg)

    def _is_user_authorized(self, _source):
        return True


class _Store:
    def __init__(self):
        self._entries: dict = {}


def _load_plugin(home: Path, config_text: str):
    os.environ["HERMES_HOME"] = str(home)
    (home / "reasoning-router").mkdir(parents=True, exist_ok=True)
    (home / "reasoning-router" / "config.yaml").write_text(config_text)
    spec = importlib.util.spec_from_file_location("reasoning_router_eval", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _route(plugin, gateway, store, chat: str, text: str):
    source = SimpleNamespace(
        platform=SimpleNamespace(value="discord"), user_id="u", chat_id=chat, thread_id=None, chat_type="dm"
    )
    key = f"agent:main:discord:dm:{chat}"
    store._entries[key] = SimpleNamespace(session_id=f"sid-{chat}")
    event = SimpleNamespace(text=text, source=source, internal=False, message_id="m")
    plugin.pre_gateway_dispatch(event, gateway=gateway, session_store=store)
    override = gateway._conv(key).reasoning_override
    if override is None:
        return None
    return "none" if override.get("enabled") is False else override.get("effort")


def _after_turn(plugin, chat: str, user_text: str, reply: str, tool_calls: int, model: str):
    history = [{"role": "user", "content": user_text}]
    for i in range(tool_calls):
        history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    history.append({"role": "assistant", "content": reply})
    plugin.post_llm_call(session_id=f"sid-{chat}", user_message=user_text, assistant_response=reply,
                         conversation_history=history, model=model, platform="discord")


def _in_range(effort: str, accept: str) -> bool:
    low, _, high = accept.partition("-")
    high = high or low
    return ORDER.index(low) <= ORDER.index(effort) <= ORDER.index(high)


def evaluate(config_path: Path | None = None, model: str = "claude-opus-5-5") -> dict:
    config_text = (config_path or HERE / "config.yaml").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        plugin = _load_plugin(Path(tmp), config_text)
        cfg = plugin._router_config(None)
        results: dict = {}
        for name in ("tuning", "holdout"):
            items = json.loads((HERE / f"{name}.json").read_text())
            misses = []
            for i, item in enumerate(items):
                gateway, store, chat = _Gateway(model), _Store(), f"{name}{i}"
                if item.get("context") == "mid_task":
                    _route(plugin, gateway, store, chat, MID_TASK_ASK)
                    _after_turn(plugin, chat, MID_TASK_ASK, MID_TASK_REPLY, 15, model)
                elif item.get("context") == "idle":
                    _route(plugin, gateway, store, chat, "thanks!")
                    _after_turn(plugin, chat, "thanks!", "Anytime.", 0, model)
                effort = _route(plugin, gateway, store, chat, item["text"]) or "unrouted"
                if effort not in item["accept"]:
                    misses.append({"text": item["text"], "accept": item["accept"], "got": effort,
                                   "category": item.get("category")})
            results[name] = {"total": len(items), "passed": len(items) - len(misses), "misses": misses}
        items = json.loads((HERE / "adversarial.json").read_text())
        misses = []
        for item in items:
            effort, reason = plugin.classify_message(item["text"], cfg)
            if not _in_range(effort, item["accept"]):
                misses.append({"text": item["text"], "accept": item["accept"], "got": effort, "reason": reason})
        results["adversarial"] = {"total": len(items), "passed": len(items) - len(misses), "misses": misses}
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="print every miss")
    args = parser.parse_args(argv)
    results = evaluate(args.config)
    for name, res in results.items():
        pct = res["passed"] / max(1, res["total"])
        cats = collections.Counter(m.get("category") or m.get("accept") for m in res["misses"])
        print(f"{name:12} {res['passed']:3}/{res['total']:<3} = {pct:6.1%}   misses: {dict(cats.most_common())}")
        if args.show:
            for miss in res["misses"]:
                print(f"    want {miss['accept']} got {miss['got']:8} | {miss['text'][:90]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
