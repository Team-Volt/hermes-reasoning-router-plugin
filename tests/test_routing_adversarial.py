"""Regression cases from an adversarial pass over the classifier (synthetic messages only)."""

from __future__ import annotations

import time

import pytest
from test_reasoning_router import load_plugin

ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def route(text, **cfg):
    return load_plugin().classify_message(text, cfg)[0]


def at_least(effort, floor):
    return ORDER.index(effort) >= ORDER.index(floor)


def at_most(effort, ceiling):
    return ORDER.index(effort) <= ORDER.index(ceiling)


@pytest.mark.parametrize(
    "text",
    (
        "git push --force origin main",
        "terraform destroy",
        "zpool destroy tank",
        "dd if=/dev/zero of=/dev/sda",
        "docker system prune -a --volumes",
        "chmod 777 the whole /srv tree",
        "open port 22 to the world",
        "allow all traffic from 0.0.0.0/0 to the db",
        "make the repo public",
        "put the api key in the repo",
        "reset all user passwords",
        "decommission the old nas",
        "swap the dns records to the new host",
        "ship it to production",
        "i think someone hacked my server, what now",
        "would you mind exposing the dashboard to the public internet",
    ),
)
def test_risky_commands_get_at_least_high(text):
    assert at_least(route(text), "high"), text


@pytest.mark.parametrize(
    "text",
    (
        "cant ssh into the box",
        "tunnel is flapping",
        "wifi drops every hour",
        "the site is down",
        "still down",
        "my pi is dead",
        "grafana dashboard shows no data",
        "ping works but ssh times out",
        'ok now it says "connection reset by peer"',
        "what is eating my ram",
        "smart says pending sectors on sdb, worried",
    ),
)
def test_problem_reports_are_not_low(text):
    assert at_least(route(text), "medium"), text


@pytest.mark.parametrize(
    "text",
    (
        "dont wipe the drive, i just want the smart status",
        "i am NOT asking you to drop the table, just show me the schema",
        "no need to restart the gateway, just check its status",
        "please dont expose anything to the internet, just show the listening ports",
    ),
)
def test_negated_risk_is_not_xhigh(text):
    assert at_most(route(text), "high"), text


@pytest.mark.parametrize(
    "text",
    (
        "whats a good secret santa gift under 20 bucks",
        "whats the security deposit for renting a garage",
        "the incident at the zoo lol",
        "the extra high setting on my washer is broken lol",
        "thanks that migration went fine",
        "thanks. dont delete anything btw",
        "add milk to my grocery list",
        'sure, "rotate all keys across the fleet" lol as if',
        "oh great another outage, love that for me",
    ),
)
def test_everyday_phrases_do_not_trip_risk(text):
    assert at_most(route(text), "medium"), text


@pytest.mark.parametrize(
    "text",
    (
        "ok do all of it",
        "yes do all three",
        "walk me through setting up wireguard on a fresh vps",
        "can you write a github action that builds and pushes my docker image",
        "quick q: is 16gb ram enough for proxmox with 6 vms?",
        "what is the best nas for plex",
        "thx. next: compare zfs vs btrfs for me",
    ),
)
def test_multi_step_and_research_asks_are_not_low(text):
    assert at_least(route(text), "medium"), text


def test_clearing_build_output_is_not_xhigh():
    assert at_most(route("rm -rf node_modules"), "high")


def test_questions_about_risky_topics_stop_at_high():
    for text in (
        "how do people usually do credential rotation",
        "explain what a production migration involves",
        "what does rm -rf do",
    ):
        assert at_most(route(text), "high"), text


def test_explicit_effort_words_still_win():
    assert route("think hard about this one") == "xhigh"
    assert route("use extra high reasoning for the plan") == "xhigh"


def test_long_pastes_classify_quickly():
    plugin = load_plugin()
    for text in ("set it\n" + " \n" * 25000, "pump turn on " * 4000, "e\u0301" * 50000):
        start = time.perf_counter()
        plugin.classify_message(text, {})
        assert time.perf_counter() - start < 0.5
