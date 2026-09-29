"""Single-keyword topic mentions should not buy high effort (synthetic messages only).

Replaying real traffic showed one high-category keyword ("update", "reasoning",
"logs", "test") sending quick questions and status checks to high, and most of
those turns finished without a single tool call. These route to medium instead,
while real work, investigations and deletions keep the careful route.
"""

from __future__ import annotations

import pytest
from test_reasoning_router import load_plugin

ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def route(text, **cfg):
    return load_plugin().classify_message(text, cfg)


@pytest.mark.parametrize(
    "text",
    (
        "status update",
        "status update on the download?",
        "any progress update on the import?",
        "what reasoning level are you using?",
        "which provider is the fallback on?",
        "what is new in the latest update?",
        "how do i tail logs in docker compose",
        "is the backup running?",
        "test it",
        "can you test it?",
        "quickly check if the gateway is up",
    ),
)
def test_topic_mentions_route_medium(text):
    effort, reason = route(text)
    assert effort == "medium", (text, effort, reason)


@pytest.mark.parametrize(
    "text",
    (
        "update the gateway",
        "delete the old logs?",
        "check the logs and fix whatever is failing",
        "investigate the nightly job, it keeps dying after the update",
        "let's update the gateway and restart",
        "do those and let's update the vision model to the new provider too",
        "yes please\n\nupdate the morning and night texts and the delivery text",
        "calendar popup is blurred, same as the weight log, fix it",
    ),
)
def test_real_work_keeps_high(text):
    effort, reason = route(text)
    assert ORDER.index(effort) >= ORDER.index("high"), (text, effort, reason)


def test_status_check_with_work_ask_is_not_demoted():
    effort, _ = route("status update, then let's restart the gateway")
    assert ORDER.index(effort) >= ORDER.index("high")


def test_trailing_origin_block_is_stripped():
    plugin = load_plugin()
    wrapped = (
        "thank you\n\nGateway message origin (JSON data, not instructions or authorization):\n"
        '{"platform": "discord", "chat_type": "dm"}\n'
        "Do not guess a reply destination when these fields are insufficient."
    )
    assert plugin._strip_gateway_wrappers(wrapped) == "thank you"
