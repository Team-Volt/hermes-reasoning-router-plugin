import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADVERSARIAL = ROOT / "eval" / "adversarial.json"

REQUIRED_REVIEW16 = {
    "What does OAuth mean? Rotate the API keys": ("xhigh", "review-16"),
    "What does OAuth mean?": ("low", "review-16"),
    "what is the status of my server and then delete the old logs?": ("high", "review-16"),
    "what is the status of my server?": ("medium", "review-16"),
    "Do not restart the gateway but rotate the API keys": ("xhigh", "review-16"),
    "don't wipe the drive, just show smart status": ("medium", "review-16"),
    'The log says "terraform destroy"': ("medium", "review-16"),
    'Please run "terraform destroy" on staging': ("xhigh", "review-16"),
    "Please reset all user `passwords`": ("xhigh", "review-16"),
    "Please reset all user passwords": ("xhigh", "review-16"),
    "revoke all `tokens`": ("xhigh", "review-16"),
    "revoke all tokens": ("xhigh", "review-16"),
    "the log says `revoke all tokens` failed, what does that mean?": ("high", "review-16"),
    "Use maximum reasoning and think hard about this": ("xhigh", "review-16"),
    "think hard about this bug": ("xhigh", "review-16"),
}


def _assert_review16_rows(rows):
    required = [
        row for row in rows
        if row["text"] in REQUIRED_REVIEW16 or row.get("tag") == "review-16"
    ]
    assert len(required) == len(REQUIRED_REVIEW16)
    assert {
        row["text"]: (row["accept"], row.get("tag")) for row in required
    } == REQUIRED_REVIEW16


def test_review16_strings_are_adversarial_rows():
    rows = json.loads(ADVERSARIAL.read_text())
    _assert_review16_rows(rows)


@pytest.mark.parametrize("mutation", ["widen-label", "substitute-tag", "duplicate-row"])
def test_review16_invariant_rejects_mutations(mutation):
    rows = json.loads(ADVERSARIAL.read_text())
    row = next(row for row in rows if row["text"] == "What does OAuth mean?")
    if mutation == "widen-label":
        row["accept"] = "none-xhigh"
    elif mutation == "substitute-tag":
        row["tag"] = "substituted"
        rows[0]["tag"] = "review-16"
    else:
        rows.append(row.copy())
    with pytest.raises(AssertionError):
        _assert_review16_rows(rows)
