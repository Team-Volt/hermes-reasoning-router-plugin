from test_reasoning_router import PLUGIN_ROOT, load_plugin


def test_file_exec_load_uses_sibling_effort_module():
    plugin = load_plugin()
    assert (PLUGIN_ROOT / "effort.py").is_file()
    assert plugin._reasoning_config_for_effort("none") == {"enabled": False}
    assert plugin._reasoning_config_for_effort("low") == {"enabled": True, "effort": "low"}
    assert plugin._reasoning_config_for_effort("ultra") == {"enabled": True, "effort": "max"}
    ladder = ("none", "low", "medium", "high", "xhigh", "max")
    assert plugin._fallback_clamp_effort("minimal", ladder) == "low"
    assert plugin.DEFAULT_CONFIG["default"] == plugin._effort.DEFAULT_EFFORT


def test_enabled_effort_never_builds_disabled_override():
    plugin = load_plugin()
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
        wire = plugin._reasoning_config_for_effort(effort)
        assert wire.get("enabled") is not False, effort
        assert "ultra" not in wire.values()
