"""#16 review strings live in eval/adversarial.json, not only in pytest."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADVERSARIAL = ROOT / "eval" / "adversarial.json"

REQUIRED_REVIEW16 = (
    "What does OAuth mean? Rotate the API keys",
    "What does OAuth mean?",
    "what is the status of my server and then delete the old logs?",
    "what is the status of my server?",
    "Do not restart the gateway but rotate the API keys",
    "don't wipe the drive, just show smart status",
    'The log says "terraform destroy"',
    'Please run "terraform destroy" on staging',
    "Please reset all user `passwords`",
    "Please reset all user passwords",
    "revoke all `tokens`",
    "revoke all tokens",
    "the log says `revoke all tokens` failed, what does that mean?",
    "Use maximum reasoning and think hard about this",
    "think hard about this bug",
)


def test_review16_strings_are_adversarial_rows():
    rows = json.loads(ADVERSARIAL.read_text())
    texts = {row["text"] for row in rows}
    missing = [text for text in REQUIRED_REVIEW16 if text not in texts]
    assert missing == []
    tagged = [row for row in rows if row.get("tag") == "review-16"]
    assert len(tagged) == len(REQUIRED_REVIEW16)
