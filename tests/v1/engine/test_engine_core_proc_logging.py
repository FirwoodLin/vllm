# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import signal
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm.v1.engine import core as core_mod
from vllm.v1.engine.core import EngineCoreProc


def test_run_engine_core_redirects_logs(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, object]] = []

    def fake_set_process_title(name: str):
        events.append(("set_process_title", name))

    def fake_build_process_log_path(log_dir: str, process_name: str, pid: int):
        events.append(("build_process_log_path", (log_dir, process_name, pid)))
        return f"{log_dir}/{process_name}.pid{pid}.log"

    def fake_redirect_stdio_to_file(log_path: str):
        events.append(("redirect_stdio_to_file", log_path))
        return log_path

    def fake_decorate_logs(process_name: str | None = None):
        events.append(("decorate_logs", process_name))

    def fake_init_worker_tracer(
        service_name: str, span_name: str, process_name: str
    ):
        events.append(
            ("maybe_init_worker_tracer", (service_name, span_name, process_name))
        )

    def fake_signal(_signum, _handler):
        events.append(("signal", _signum))
        return None

    class DummySignalCallback:
        def __init__(self, _callback):
            events.append(("signal_callback_init", None))

        def trigger(self):
            events.append(("signal_callback_trigger", None))

        def stop(self):
            events.append(("signal_callback_stop", None))

    def fake_init(self, *args, **kwargs):
        events.append(("engine_init", kwargs["engine_index"]))
        self._shutdown_waiters = []
        self.output_queue = queue.Queue()
        self.output_thread = SimpleNamespace(is_alive=lambda: False)

    def fake_run_busy_loop(self):
        events.append(("run_busy_loop", None))

    def fake_shutdown(self):
        events.append(("shutdown", None))

    def fake_wait_for_output_queue_idle(self, timeout: float = 5.0):
        events.append(("wait_for_output_queue_idle", timeout))

    monkeypatch.setattr(core_mod, "set_process_title", fake_set_process_title)
    monkeypatch.setattr(core_mod, "build_process_log_path", fake_build_process_log_path)
    monkeypatch.setattr(core_mod, "redirect_stdio_to_file", fake_redirect_stdio_to_file)
    monkeypatch.setattr(core_mod, "decorate_logs", fake_decorate_logs)
    monkeypatch.setattr(
        core_mod, "maybe_init_worker_tracer", fake_init_worker_tracer
    )
    monkeypatch.setattr(core_mod, "SignalCallback", DummySignalCallback)
    monkeypatch.setattr(
        core_mod, "maybe_register_config_serialize_by_value", lambda: None
    )
    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr(EngineCoreProc, "__init__", fake_init)
    monkeypatch.setattr(EngineCoreProc, "run_busy_loop", fake_run_busy_loop)
    monkeypatch.setattr(EngineCoreProc, "shutdown", fake_shutdown)
    monkeypatch.setattr(
        EngineCoreProc, "wait_for_output_queue_idle", fake_wait_for_output_queue_idle
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank_local=None,
            data_parallel_index=None,
            data_parallel_size_local=2,
            data_parallel_rank=None,
        ),
        kv_transfer_config=None,
        model_config=SimpleNamespace(is_moe=False),
        observability_config=SimpleNamespace(
            engine_core_log_dir="/tmp/engine-core-logs"
        ),
    )

    EngineCoreProc.run_engine_core(
        vllm_config=vllm_config,
        local_client=True,
        handshake_address="tcp://127.0.0.1:12345",
        executor_class=object,
        log_stats=False,
        dp_rank=3,
        local_dp_rank=1,
    )

    assert events[:6] == [
        ("set_process_title", "EngineCore_DP3"),
        (
            "build_process_log_path",
            ("/tmp/engine-core-logs", "EngineCore_DP3", events[1][1][2]),
        ),
        (
            "redirect_stdio_to_file",
            f"/tmp/engine-core-logs/EngineCore_DP3.pid{events[1][1][2]}.log",
        ),
        ("decorate_logs", "EngineCore_DP3"),
        (
            "maybe_init_worker_tracer",
            ("vllm.engine_core", "engine_core", "EngineCore_DP3"),
        ),
        ("engine_init", 3),
    ]
    assert ("run_busy_loop", None) in events
    assert ("signal_callback_stop", None) in events
    assert events[-2:] == [("shutdown", None), ("wait_for_output_queue_idle", 5.0)]


def test_run_engine_core_waits_for_teardown_before_resolving_shutdown_waiters(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[tuple[str, object]] = []
    waiter: Future[None] = Future()

    class DummySignalCallback:
        def __init__(self, _callback):
            pass

        def stop(self):
            pass

    def fake_init(self, *args, **kwargs):
        self._shutdown_waiters = [waiter]
        self.output_queue = queue.Queue()
        self.output_thread = SimpleNamespace(is_alive=lambda: False)

    def fake_run_busy_loop(self):
        events.append(("run_busy_loop", waiter.done()))

    def fake_shutdown(self):
        events.append(("shutdown", waiter.done()))

    def fake_wait_for_output_queue_idle(self, timeout: float = 5.0):
        events.append(("wait_for_output_queue_idle", (timeout, waiter.done())))

    monkeypatch.setattr(core_mod, "set_process_title", lambda *_args: None)
    monkeypatch.setattr(core_mod, "decorate_logs", lambda *_args: None)
    monkeypatch.setattr(core_mod, "maybe_init_worker_tracer", lambda *_args: None)
    monkeypatch.setattr(core_mod, "SignalCallback", DummySignalCallback)
    monkeypatch.setattr(
        core_mod, "maybe_register_config_serialize_by_value", lambda: None
    )
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.setattr(EngineCoreProc, "__init__", fake_init)
    monkeypatch.setattr(EngineCoreProc, "run_busy_loop", fake_run_busy_loop)
    monkeypatch.setattr(EngineCoreProc, "shutdown", fake_shutdown)
    monkeypatch.setattr(
        EngineCoreProc, "wait_for_output_queue_idle", fake_wait_for_output_queue_idle
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=None,
            data_parallel_index=None,
            data_parallel_size_local=1,
            data_parallel_rank=None,
        ),
        kv_transfer_config=None,
        model_config=SimpleNamespace(is_moe=False),
        observability_config=SimpleNamespace(engine_core_log_dir=None),
    )

    EngineCoreProc.run_engine_core(
        vllm_config=vllm_config,
        local_client=True,
        handshake_address="tcp://127.0.0.1:12345",
        executor_class=object,
        log_stats=False,
    )

    assert events == [
        ("run_busy_loop", False),
        ("shutdown", False),
        ("wait_for_output_queue_idle", (5.0, True)),
    ]
    assert waiter.done()
    assert waiter.result() is None


def test_run_engine_core_propagates_shutdown_failures_to_waiters(
    monkeypatch: pytest.MonkeyPatch,
):
    waiter: Future[None] = Future()

    class DummySignalCallback:
        def __init__(self, _callback):
            pass

        def stop(self):
            pass

    def fake_init(self, *args, **kwargs):
        self._shutdown_waiters = [waiter]
        self.output_queue = queue.Queue()
        self.output_thread = SimpleNamespace(is_alive=lambda: False)

    monkeypatch.setattr(core_mod, "set_process_title", lambda *_args: None)
    monkeypatch.setattr(core_mod, "decorate_logs", lambda *_args: None)
    monkeypatch.setattr(core_mod, "maybe_init_worker_tracer", lambda *_args: None)
    monkeypatch.setattr(core_mod, "SignalCallback", DummySignalCallback)
    monkeypatch.setattr(
        core_mod, "maybe_register_config_serialize_by_value", lambda: None
    )
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.setattr(EngineCoreProc, "__init__", fake_init)
    monkeypatch.setattr(EngineCoreProc, "run_busy_loop", lambda self: None)
    monkeypatch.setattr(
        EngineCoreProc,
        "shutdown",
        lambda self: (_ for _ in ()).throw(RuntimeError("shutdown failed")),
    )
    monkeypatch.setattr(
        EngineCoreProc, "wait_for_output_queue_idle", lambda self, timeout=5.0: None
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=None,
            data_parallel_index=None,
            data_parallel_size_local=1,
            data_parallel_rank=None,
        ),
        kv_transfer_config=None,
        model_config=SimpleNamespace(is_moe=False),
        observability_config=SimpleNamespace(engine_core_log_dir=None),
    )

    with pytest.raises(RuntimeError, match="shutdown failed"):
        EngineCoreProc.run_engine_core(
            vllm_config=vllm_config,
            local_client=True,
            handshake_address="tcp://127.0.0.1:12345",
            executor_class=object,
            log_stats=False,
        )

    assert waiter.done()
    assert isinstance(waiter.exception(), RuntimeError)
    assert str(waiter.exception()) == "shutdown failed"
