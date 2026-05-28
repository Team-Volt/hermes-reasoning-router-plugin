# reasoning-router

`reasoning-router` is a Hermes gateway plugin that automatically chooses the reasoning effort for each incoming Discord message.

It is useful when one chat contains both tiny messages (`thanks`, `what time is it?`) and deeper work (`debug this gateway issue`, `patch the plugin and verify it`). Instead of running every turn at the same reasoning level, the plugin classifies the incoming request and sets Hermes' real per-session reasoning override before the model request is made.

> Current chat-surface support: **Discord only**. The plugin is written against Hermes gateway session behavior used by Discord. Cron jobs and other chat surfaces do not currently use this router.

## What it does

- Routes each Discord gateway message to one of:
  - `none`
  - `minimal`
  - `low`
  - `medium`
  - `high`
  - `xhigh`
- Uses Hermes' session reasoning override, so the selected effort reaches the provider as request configuration rather than prompt text.
- Keeps simple messages cheap.
- Sends implementation/config/debugging/system work to higher effort.
- Handles the annoying but common approval case:
  - Assistant: “Want me to proceed with implementing this?”
  - User: “yes”
  - Router: inherits the planned task effort instead of treating `yes` as `low`.
- Can log routing decisions to the gateway journal and/or a JSONL file for audits.
- Can optionally call an OpenAI-compatible semantic classifier for ambiguous routes; this is disabled by default and bounded by confidence, length, min/max clamps, and deterministic guardrails.

## How it works

The plugin registers two Hermes hooks:

- `pre_gateway_dispatch`
  - Runs before Hermes dispatches the incoming Discord message to the agent.
  - Classifies the message with deterministic heuristics.
  - Sets the session reasoning override through Hermes' gateway session mechanism.
  - Fails open: if anything is missing or unsupported, the message is still allowed through.

- `post_llm_call`
  - Runs after an assistant response.
  - Detects when the assistant asked for approval to proceed with a substantive task.
  - Stores a one-shot pending intent for that session.
  - If the next user message is a short approval like `yes`, `go ahead`, or `do it`, the router consumes that pending intent and inherits the planned effort.

Routing is intentionally conservative and explainable. It uses pattern groups for architecture, security, configuration, implementation, debugging, operations, destructive changes, verification, logging/audit, technical follow-ups, and simple acknowledgements.

If `semantic_classifier_enabled` is true, deterministic routing still runs first. The semantic classifier is only asked to adjudicate ambiguous cases: short/default routes that might deserve escalation, and question/clarification forms that otherwise matched high-risk keywords. It can raise ambiguous routes, and it can lower question/clarification false positives only to `medium` or higher. It cannot bypass `min` / `max` clamps, does not handle slash commands, and is skipped for obvious low chatter or messages longer than `semantic_classifier_max_chars`.

## Routing guide

| Effort | Typical use |
|---|---|
| `low` | Thanks, acknowledgements, tiny factual questions, very short simple messages |
| `medium` | Normal inspection, research, status checks, file/log questions, short technical feasibility follow-ups |
| `high` | Implementation, config changes, debugging, Hermes internals, ops, tests, verification, audit/log work |
| `xhigh` | Architecture, security/auth, rollback-sensitive work, multi-system changes, gateway restarts, explicit “think hard” requests |

High-complexity categories are counted. If a message hits at least `xhigh_high_match_threshold` categories, it routes to `xhigh` even without an explicit “xhigh” phrase.

## Repository contents

```text
.
├── __init__.py                     # plugin implementation
├── plugin.yaml                     # Hermes plugin metadata
├── examples/config.yaml            # example standalone config
├── scripts/minilm_router_eval.py   # offline MiniLM/nearest-neighbor evaluator POC
├── tests/test_reasoning_router.py  # focused regression tests
├── tests/test_minilm_router_eval.py
└── README.md
```

## Requirements

- Hermes Agent with gateway plugins available
- A Discord gateway/chat surface
- Python 3.11+

## Install

Canonical install path: clone `reasoning-router` as a general user plugin.

```bash
git clone https://github.com/Team-Volt/hermes-reasoning-router-plugin \
  ~/.hermes/plugins/reasoning-router
```

For a profile-specific install:

```bash
git clone https://github.com/Team-Volt/hermes-reasoning-router-plugin \
  ~/.hermes/profiles/myprofile/plugins/reasoning-router
```

## Configure

The plugin reads standalone config from:

```text
~/.hermes/reasoning-router/config.yaml
```

For profile-specific installs, use the matching profile home, for example `~/.hermes/profiles/myprofile/reasoning-router/config.yaml`.

Start from the example config:

```bash
mkdir -p ~/.hermes/reasoning-router
cp ~/.hermes/plugins/reasoning-router/examples/config.yaml \
  ~/.hermes/reasoning-router/config.yaml
```

Legacy installs with `~/.hermes/plugins/reasoning-router/config.yaml` still read that file if the standalone config is missing, but slash-command writes now go to the standalone path.

Example config with every supported persistent option:

```yaml
enabled: true
default: medium
min: none
max: xhigh

log_decisions: true
decision_log: true
decision_log_path: logs/reasoning-router.jsonl

low_char_limit: 80
xhigh_high_match_threshold: 4

pending_intent_enabled: true
pending_intent_ttl_minutes: 30

# Optional live semantic classifier. Disabled by default.
semantic_classifier_enabled: false
semantic_classifier_url: http://127.0.0.1:8080/v1/chat/completions
semantic_classifier_model: gpt-5.4-mini
# Prefer CODEX_PROXY_API_KEY or OPENAI_API_KEY in the environment instead of
# storing a key here. Leave empty to use env vars, or omit the field entirely.
semantic_classifier_api_key: ""
semantic_classifier_timeout_seconds: 8
semantic_classifier_min_confidence: 0.75
semantic_classifier_max_chars: 1200
```

Config fields:

| Field | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the router on or off. When off, messages pass through without changing reasoning. |
| `default` | `medium` | Fallback effort when no deterministic rule or accepted semantic result strongly matches. Valid efforts: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. |
| `min` | `none` | Minimum allowed effort after routing. Invalid values are ignored. |
| `max` | `xhigh` | Maximum allowed effort after routing. Invalid values are ignored. If `min` is higher than `max`, the plugin swaps the clamp bounds. |
| `log_decisions` | `true` | Log concise routing decisions to the Hermes gateway logger / journal. |
| `decision_log` | `false` | Write persistent JSONL routing decisions for later review. |
| `decision_log_path` | `logs/reasoning-router.jsonl` | JSONL path. Relative paths resolve under `~/.hermes`; absolute paths are used as-is. |
| `low_char_limit` | `80` | Short-message cutoff used by the low-effort fallback after stronger rules and semantic adjudication have had a chance to run. |
| `xhigh_high_match_threshold` | `4` | Number of high-complexity categories needed to escalate to `xhigh` without an explicit xhigh/risk phrase. |
| `pending_intent_enabled` | `true` | Enable one-shot effort inheritance for short approvals after the assistant asks to proceed. |
| `pending_intent_ttl_minutes` | `30` | Expiration window for pending approval intent. Values below 1 are clamped to 1 minute. |
| `semantic_classifier_enabled` | `false` | Enables the optional OpenAI-compatible classifier for ambiguous cases only. Deterministic guardrails still win. |
| `semantic_classifier_url` | `http://127.0.0.1:8080/v1/chat/completions` | Chat-completions endpoint used by the semantic classifier. Intended for a local codex-proxy/OpenAI-compatible service. |
| `semantic_classifier_model` | `gpt-5.4-mini` | Model name sent to the classifier endpoint. |
| `semantic_classifier_api_key` | `""` | Bearer token for the classifier endpoint. If omitted or empty, the plugin checks `CODEX_PROXY_API_KEY`, then `OPENAI_API_KEY`; if no key is available, it sends no `Authorization` header. Do not commit real keys. |
| `semantic_classifier_timeout_seconds` | `8` | HTTP timeout for the classifier call. Values below 1 are clamped to 1 second. |
| `semantic_classifier_min_confidence` | `0.75` | Minimum classifier confidence required before a semantic result is accepted. Lower-confidence results fall back to deterministic routing. |
| `semantic_classifier_max_chars` | `1200` | Maximum normalized message length eligible for semantic classification. Longer messages skip the classifier. |

### Configuration examples

Minimal deterministic router:

```yaml
enabled: true
default: medium
min: none
max: high
log_decisions: true
decision_log: false
low_char_limit: 80
xhigh_high_match_threshold: 4
pending_intent_enabled: true
```

Persistent audit trail for tuning:

```yaml
decision_log: true
decision_log_path: logs/reasoning-router.jsonl
log_decisions: true
```

Conservative semantic classifier via local OpenAI-compatible proxy:

```yaml
semantic_classifier_enabled: true
semantic_classifier_url: http://127.0.0.1:8080/v1/chat/completions
semantic_classifier_model: gpt-5.4-mini
# Export CODEX_PROXY_API_KEY instead of storing the key here:
#   export CODEX_PROXY_API_KEY='...'
semantic_classifier_timeout_seconds: 3
semantic_classifier_min_confidence: 0.85
semantic_classifier_max_chars: 600
```

More permissive semantic classifier for shadow testing on ambiguous short commands:

```yaml
semantic_classifier_enabled: true
semantic_classifier_timeout_seconds: 8
semantic_classifier_min_confidence: 0.70
semantic_classifier_max_chars: 1200
```

Hard cap the router so it never requests `xhigh`:

```yaml
min: none
max: high
```

## Activate

Enable the plugin in `~/.hermes/config.yaml` if your Hermes setup requires explicit plugin enablement:

```yaml
plugins:
  enabled:
    - reasoning-router
```

Restart the Hermes gateway after installing or changing plugin config. Use the restart method for your environment; this may be a desktop app restart, process-manager restart, launchd service restart, systemd service restart, Docker/container restart, or asking the operator of the Hermes host to restart it.

## Verify

Verify that Hermes loaded the plugin:

```bash
python - <<'PY'
from hermes_cli.plugins import PluginManager, get_plugin_command_handler

m = PluginManager()
m.discover_and_load()
p = m._plugins.get("reasoning-router")
print("enabled:", bool(p and p.enabled))
print("error:", getattr(p, "error", None) if p else None)
print("hooks:", getattr(p, "hooks_registered", None) if p else None)
print("commands:", getattr(p, "commands_registered", None) if p else None)

handler = get_plugin_command_handler("reasoning-router")
print(handler("test Does this require modifying Hermes source?") if handler else "no command handler")
PY
```

## Slash command

The plugin registers `/reasoning-router`.

Useful commands:

```text
/reasoning-router status
/reasoning-router on
/reasoning-router off
/reasoning-router min <effort>
/reasoning-router max <effort>
/reasoning-router default <effort>
/reasoning-router threshold <N>
/reasoning-router pending on|off
/reasoning-router log on|off
/reasoning-router recent [N]
/reasoning-router test <message>
```

Example:

```text
/reasoning-router test Patch the plugin and run the focused tests
```

## Logging and audits

When `log_decisions: true`, the Hermes gateway journal includes entries like:

```text
reasoning-router: session=<session_key> effort=<level> reason=<why>
```

Gateway log access depends on how Hermes is running. On systemd-based Linux hosts, for example:

```bash
journalctl --user -u hermes-gateway.service | grep reasoning-router
```

On macOS or other setups, use the log path or process manager for your Hermes gateway.

When `decision_log: true`, the plugin writes JSONL rows to:

```text
~/.hermes/logs/reasoning-router.jsonl
```

Each row includes timestamp, platform, user/chat/thread IDs, session key, selected effort, routing reason, and a short message preview. Pending-intent approvals include the inherited pending effort and previews of the pending task.

## Live semantic classifier

The live semantic classifier is optional and disabled by default. It posts a chat-completions request to `semantic_classifier_url` with:

```json
{
  "model": "gpt-5.4-mini",
  "messages": [
    {"role": "system", "content": "...routing instructions..."},
    {"role": "user", "content": "{...current_user_message/context JSON...}"}
  ],
  "max_tokens": 220,
  "temperature": 0
}
```

The classifier should return JSON in the assistant message content:

```json
{
  "effort": "medium",
  "confidence": 0.92,
  "risk_categories": ["config_change"],
  "reason": "Terse request likely refers to applying a setting from recent context."
}
```

Accepted `effort` values are `none`, `low`, `medium`, `high`, and `xhigh`; `minimal` is accepted but normalized to `low`. Results with invalid effort, invalid JSON, missing content, HTTP errors, timeouts, or confidence below `semantic_classifier_min_confidence` are ignored and the router falls back to deterministic classification.

The classifier receives only compact routing context: the current user message, the last assistant intent when available, an optional pending action string, and up to three recent user/assistant messages read from Hermes `state.db`. That context is for resolving terse approvals and deictic references, not for solving the request.

### Offline MiniLM evaluator POC

`scripts/minilm_router_eval.py` is a read-only evaluator for the persistent JSONL log. It embeds historical message previews, runs leave-one-out nearest-neighbor effort prediction, and reports accuracy, confusion, and conflicted examples. It is a shadow-analysis tool only; it does not affect live routing.

Run a dependency-free smoke with the built-in lexical hash embedder:

```bash
python scripts/minilm_router_eval.py --embedder lexical --limit 100
```

Run the actual MiniLM-style pass if `sentence-transformers` is installed and the model is available locally / downloadable:

```bash
python scripts/minilm_router_eval.py \
  --embedder minilm \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --input ~/.hermes/logs/reasoning-router.jsonl
```

Use the report to decide whether a local semantic-neighbor layer is useful before adding any live-routing integration.

## Test

From this repository:

```bash
python -m py_compile __init__.py scripts/minilm_router_eval.py
python -m pytest -q tests -o 'addopts='
```

## LLM install prompt

Give this to an agent with access to your Hermes install:

```text
Install the Hermes reasoning-router plugin from https://github.com/Team-Volt/hermes-reasoning-router-plugin.

Use the README as the source of truth. Install it as a Hermes user plugin, copy the example config, enable the plugin if this Hermes setup requires explicit plugin enablement, and verify that it loads.

Important constraints:
- It currently supports Discord only.
- Do not edit Hermes source code.
- Preserve existing Hermes config and enabled plugins.
- Use profile-specific paths if this install uses a Hermes profile.
- Restart the Hermes gateway using the normal method for this machine, or ask me to restart it if you are not sure.
- After restart, verify plugin discovery and run a routing smoke test with the reasoning-router command or command handler.
- If any step fails, stop and explain the failure instead of guessing.
```

## Notes and limitations

- Discord is the only currently supported chat surface.
- Cron jobs do not pass through the Discord gateway dispatch hook and therefore do not use this router.
- The plugin depends on Hermes gateway internals for session reasoning overrides. It guards those calls and fails open, but future Hermes changes could require a small compatibility update.
- The default classifier is deterministic and intentionally inspectable; it is not an LLM judge.
- The optional semantic classifier is a bounded ambiguity resolver, not the primary router. Keep it behind a local/protected endpoint and avoid storing real API keys in `config.yaml`.
