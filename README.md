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
├── examples/config.yaml            # example plugin-local config
├── tests/test_reasoning_router.py  # focused regression tests
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

The plugin reads plugin-local config from:

```text
~/.hermes/plugins/reasoning-router/config.yaml
```

For profile-specific installs, use the matching profile plugin directory.

Start from the example config:

```bash
cp ~/.hermes/plugins/reasoning-router/examples/config.yaml \
  ~/.hermes/plugins/reasoning-router/config.yaml
```

Example config:

```yaml
enabled: true
default: medium
min: low
max: xhigh
log_decisions: true
decision_log: true
decision_log_path: logs/reasoning-router.jsonl
low_char_limit: 80
xhigh_high_match_threshold: 4
pending_intent_enabled: true
pending_intent_ttl_minutes: 30
```

Config fields:

| Field | Meaning |
|---|---|
| `enabled` | Turns the router on or off |
| `default` | Fallback effort when no heuristic strongly matches |
| `min` / `max` | Clamp all routing decisions into this range |
| `log_decisions` | Log concise decisions to the gateway journal |
| `decision_log` | Write persistent JSONL decisions for later review |
| `decision_log_path` | JSONL path relative to `~/.hermes` unless absolute |
| `low_char_limit` | Short-message cutoff used by the low-effort fallback |
| `xhigh_high_match_threshold` | Number of high-complexity categories needed to escalate to `xhigh` |
| `pending_intent_enabled` | Enable effort inheritance for short approvals |
| `pending_intent_ttl_minutes` | Expiration window for pending approval intent |

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

## Test

From this repository:

```bash
python -m py_compile __init__.py
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
- The classifier is deterministic and intentionally inspectable; it is not an LLM judge.
