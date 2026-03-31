# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class GraphReplayTimingStats:
    reply_global_rank: int
    dp_rank: int
    tp_rank: int
    dcp_rank: int
    node_rank: int
    replay_count: int
    replay_gpu_ms: float
    replay_wall_ms: float
    runtime_mode: str
    graph_impl: str
    seqlen: int | None = None
    total_seqlen: int | None = None
    async_output_copy_wait_ms: float = 0.0


@dataclass
class _PendingReplay:
    graph_impl: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    wall_time_ns: int


@dataclass
class GraphTimingContext:
    reply_global_rank: int
    dp_rank: int
    tp_rank: int
    dcp_rank: int
    node_rank: int
    runtime_mode: str
    event_pool_size: int = 4

    replay_count: int = 0
    seqlen: int | None = None
    total_seqlen: int | None = None
    graph_impls: set[str] = field(default_factory=set)
    _event_pool: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list, init=False, repr=False
    )
    _pending_replays: list[_PendingReplay] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for _ in range(self.event_pool_size):
            self._event_pool.append(self._new_event_pair())

    def _new_event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def _acquire_event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        event_index = self.replay_count
        if event_index >= len(self._event_pool):
            self._event_pool.append(self._new_event_pair())
        return self._event_pool[event_index]

    def record_replay(self, graph_impl: str, replay_fn: Callable[[], None]) -> None:
        start_event, end_event = self._acquire_event_pair()
        wall_start_ns = time.perf_counter_ns()
        start_event.record()
        replay_fn()
        end_event.record()

        self.replay_count += 1
        self.graph_impls.add(graph_impl)
        self._pending_replays.append(
            _PendingReplay(
                graph_impl=graph_impl,
                start_event=start_event,
                end_event=end_event,
                wall_time_ns=time.perf_counter_ns() - wall_start_ns,
            )
        )

    def update_seqlen(
        self,
        seqlen: int | None,
        total_seqlen: int | None = None,
    ) -> None:
        if seqlen is None:
            return
        self.seqlen = seqlen if self.seqlen is None else max(self.seqlen, seqlen)
        if total_seqlen is not None:
            self.total_seqlen = (
                total_seqlen
                if self.total_seqlen is None
                else max(self.total_seqlen, total_seqlen)
            )

    def resolve(self, async_output_copy_wait_ms: float = 0.0) -> GraphReplayTimingStats:
        replay_gpu_ms = 0.0
        replay_wall_ms = 0.0
        for replay in self._pending_replays:
            replay_gpu_ms += replay.start_event.elapsed_time(replay.end_event)
            replay_wall_ms += replay.wall_time_ns / 1e6

        self._pending_replays.clear()
        graph_impl = ",".join(sorted(self.graph_impls)) if self.graph_impls else "none"
        return GraphReplayTimingStats(
            reply_global_rank=self.reply_global_rank,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            dcp_rank=self.dcp_rank,
            node_rank=self.node_rank,
            replay_count=self.replay_count,
            replay_gpu_ms=replay_gpu_ms,
            replay_wall_ms=replay_wall_ms,
            runtime_mode=self.runtime_mode,
            graph_impl=graph_impl,
            seqlen=self.seqlen,
            total_seqlen=self.total_seqlen,
            async_output_copy_wait_ms=async_output_copy_wait_ms,
        )
