"""Optional integration: actual Hermes Buzz adapter and ingress replay guard, with all I/O stubbed."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip('gateway.run_inbound_hooks')
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run_inbound_hooks import GatewayInboundHooksMixin
from gateway.session_ingress import executing_admission
from plugins.platforms.buzz.adapter import BuzzAdapter


def test_actual_buzz_dispatch_routes_once_and_turn_adapter_is_inert(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('buzz_router_fixture', Path(__file__).resolve().parents[1] / '__init__.py')
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    router._RUNTIME_CONFIG_OVERRIDE = {'enabled': True, 'enabled_platforms': ['buzz'],
                                       'acp_none_means_auto': True, 'log_decisions': False}
    calls = []
    overrides = {}
    def setter(key, value):
        calls.append((key, value))
        overrides[key] = value
    runner = SimpleNamespace(config_data={}, _session_reasoning_overrides=overrides,
                             _set_session_reasoning_override=setter)
    from gateway.session import SessionStore
    from gateway.config import GatewayConfig
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    runner.session_store = store
    runner._session_key_for_source = store._generate_session_key
    from hermes_cli import lifecycle
    async def hooks(name, **payload):
        assert name == 'pre_gateway_dispatch'
        return [router.pre_gateway_dispatch(**payload)]
    monkeypatch.setattr(lifecycle, 'ainvoke_hook', hooks)
    async def run():
        delivered = asyncio.Event()
        async def ingress(event):
            source = event.source
            assert source.platform.value == 'buzz' and source.chat_id == 'channel'
            assert await GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook(runner, event, source) is event
            token = executing_admission.set(True)
            try:
                assert await GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook(runner, event, source) is event
                assert router.select_turn_reasoning(platform='buzz', user_message=event.text) is None
            finally:
                executing_admission.reset(token)
            delivered.set()
            return None
        adapter = object.__new__(BuzzAdapter)
        BasePlatformAdapter.__init__(adapter, PlatformConfig(enabled=True), Platform('buzz'))
        adapter._channel_names = {}
        adapter._localize_inbound_media = AsyncMock(return_value=('Implement the gateway fix', [], [], None))
        adapter.send_reaction = AsyncMock()
        adapter.send_typing = AsyncMock()
        adapter.stop_typing = AsyncMock()
        adapter.set_message_handler(ingress)
        await adapter._dispatch_message('Implement the gateway fix', 'channel', 'dm', 'user', 'User', 'msg', 0)
        await asyncio.wait_for(delivered.wait(), 5)
        assert len(calls) == 1
        assert calls[0][1]['effort'] in {'high', 'xhigh'}
    asyncio.run(run())
