# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.v1.engine.async_llm as async_llm_mod
import vllm.v1.engine.core_client as core_client_mod
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError


def test_mp_client_monitor_ignores_expected_shutdown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    sentinel = object()
    proc = SimpleNamespace(sentinel=sentinel, name="EngineCore_DP0")

    class ImmediateThread:
        def __init__(self, target, daemon, name):
            self._target = target

        def start(self):
            self._target()

    class DummyClient:
        start_engine_core_monitor = core_client_mod.MPClient.start_engine_core_monitor
        begin_shutdown = core_client_mod.MPClient.begin_shutdown

        def __init__(self):
            self.resources = SimpleNamespace(
                engine_manager=SimpleNamespace(processes=[proc]),
                engine_dead=False,
                shutting_down=False,
            )
            self._finalizer = SimpleNamespace(alive=True)
            self.shutdown = MagicMock()

    monkeypatch.setattr(
        core_client_mod.multiprocessing.connection,
        "wait",
        lambda sentinels: [sentinel],
    )
    monkeypatch.setattr(core_client_mod, "Thread", ImmediateThread)

    client = DummyClient()
    client.begin_shutdown()

    with caplog.at_level("ERROR"):
        client.start_engine_core_monitor()

    assert not client.resources.engine_dead
    client.shutdown.assert_not_called()
    assert "died unexpectedly" not in caplog.text


def test_output_handler_ignores_engine_dead_during_shutdown():
    async def run_test():
        async def get_output_async():
            raise EngineDeadError()

        engine = object.__new__(AsyncLLM)
        engine.engine_core = SimpleNamespace(
            begin_shutdown=MagicMock(),
            get_output_async=get_output_async,
            shutdown=lambda timeout=None: None,
        )
        engine.output_processor = SimpleNamespace(
            process_outputs=MagicMock(),
            update_scheduler_stats=MagicMock(),
            propagate_error=MagicMock(),
            needs_iteration_stats=False,
        )
        engine.log_stats = False
        engine.logger_manager = None
        engine.renderer = SimpleNamespace(
            stat_mm_cache=lambda: {},
            shutdown=lambda: None,
        )
        engine.output_handler = None
        engine._shutting_down = False

        engine.begin_shutdown()
        engine._run_output_handler()

        await asyncio.wait_for(engine.output_handler, timeout=1.0)

        engine.engine_core.begin_shutdown.assert_called_once()
        engine.output_processor.propagate_error.assert_not_called()

    asyncio.run(run_test())


def test_async_llm_shutdown_logs_phases(
    monkeypatch: pytest.MonkeyPatch,
):
    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        begin_shutdown=MagicMock(),
        shutdown=MagicMock(),
    )
    engine.renderer = SimpleNamespace(shutdown=MagicMock())
    engine.output_handler = None
    engine._shutting_down = False
    messages: list[str] = []
    monkeypatch.setattr(
        async_llm_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    engine.shutdown(timeout=7.0)

    engine.engine_core.begin_shutdown.assert_called_once()
    engine.engine_core.shutdown.assert_called_once_with(timeout=7.0)
    joined = "\n".join(messages)
    assert "AsyncLLM shutdown start timeout=7.0" in joined
    assert "AsyncLLM shutdown: renderer.shutdown start" in joined
    assert "AsyncLLM shutdown: engine_core.shutdown done" in joined


def test_core_client_shutdown_logs_phases(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Resources(SimpleNamespace):
        def __call__(self):
            self.released = True

    class DummyClient:
        shutdown = core_client_mod.MPClient.shutdown
        begin_shutdown = core_client_mod.MPClient.begin_shutdown

        def __init__(self):
            self.resources = _Resources(
                engine_manager=SimpleNamespace(shutdown=MagicMock()),
                shutting_down=False,
                released=False,
            )
            self._finalizer = SimpleNamespace(detach=MagicMock(return_value=object()))

    client = DummyClient()
    messages: list[str] = []
    monkeypatch.setattr(
        core_client_mod.logger,
        "info",
        lambda message, *args, **_kwargs: messages.append(
            message % args if args else message),
    )

    client.shutdown(timeout=9.0)

    client.resources.engine_manager.shutdown.assert_called_once_with(timeout=9.0)
    assert client.resources.released is True
    joined = "\n".join(messages)
    assert "EngineCore client shutdown start timeout=9.0" in joined
    assert "EngineCore client shutdown: engine_manager.shutdown done" in joined
    assert "EngineCore client shutdown done" in joined
