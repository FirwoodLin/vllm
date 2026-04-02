# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import torch

from vllm.distributed import parallel_state


def test_destroy_distributed_environment_barriers_before_teardown(
    monkeypatch,
) -> None:
    events: list[object] = []
    cpu_group = object()

    class _FakeWorld:
        world_size = 2
        cpu_group = cpu_group

        def destroy(self) -> None:
            events.append("world_destroy")

    def fake_monitored_barrier(*, group, timeout, **kwargs) -> None:
        events.append(("barrier", group, timeout))

    monkeypatch.setattr(parallel_state, "_WORLD", _FakeWorld())
    monkeypatch.setattr(parallel_state, "_NODE_COUNT", 2)
    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        fake_monitored_barrier,
        raising=False,
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: events.append("default_pg_destroy"),
    )

    parallel_state.destroy_distributed_environment()

    assert events == [
        ("barrier", cpu_group, timedelta(seconds=5)),
        "world_destroy",
        "default_pg_destroy",
    ]


def test_best_effort_shutdown_barrier_falls_back_to_group_barrier() -> None:
    events: list[str] = []

    class _FakeStatelessGroup:
        world_size = 2

        def barrier(self) -> None:
            events.append("barrier")

    parallel_state._best_effort_shutdown_barrier("stateless", _FakeStatelessGroup())

    assert events == ["barrier"]
