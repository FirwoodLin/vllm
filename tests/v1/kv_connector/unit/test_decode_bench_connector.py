# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DecodeBenchConnector."""

import pytest
import torch

from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
    DecodeBenchConnectorWorkerMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


class DecodeBenchTestRunner:
    """Test harness for scheduler/worker DecodeBenchConnector interactions."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        *,
        dummy_prefill: bool = False,
        dummy_output_token_id: int = 0,
    ):
        self.req_id = -1

        vllm_config = create_vllm_config(
            block_size=block_size,
            max_num_batched_tokens=1000,
            kv_connector_extra_config={
                "dummy_prefill": dummy_prefill,
                "dummy_output_token_id": dummy_output_token_id,
            },
        )
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="DecodeBenchConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "dummy_prefill": dummy_prefill,
                "dummy_output_token_id": dummy_output_token_id,
            },
        )

        self.vllm_config = vllm_config
        self.scheduler: Scheduler = create_scheduler(
            vllm_config,
            num_blocks=num_gpu_blocks,
        )
        self.worker_connector = DecodeBenchConnector(
            vllm_config,
            KVConnectorRole.WORKER,
        )

        num_heads = 4
        head_dim = 64
        kv_caches = {
            f"layer_{i}": torch.zeros(
                num_gpu_blocks,
                2,
                num_heads,
                block_size,
                head_dim,
            )
            for i in range(2)
        }
        self.worker_connector.register_kv_caches(kv_caches)

        scheduler_connector = self.scheduler.connector
        assert isinstance(scheduler_connector, DecodeBenchConnector)
        self.scheduler_connector = scheduler_connector

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(block_size, sha256)
        self._dummy_ctx = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            virtual_engine=0,
            slot_mapping={},
        )

    def new_request(self, token_ids: list[int]) -> Request:
        self.req_id += 1

        sampling_params = SamplingParams(max_tokens=100)
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
        req = Request(
            request_id=str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=self._block_hasher,
        )
        self.scheduler.add_request(req)
        return req

    def run_connector(
        self, scheduler_output
    ) -> tuple[DecodeBenchConnectorMetadata, KVConnectorOutput | None]:
        metadata = scheduler_output.kv_connector_metadata
        assert isinstance(metadata, DecodeBenchConnectorMetadata)

        self.worker_connector.bind_connector_metadata(metadata)
        self.worker_connector.start_load_kv(self._dummy_ctx)
        finished_sending, finished_recving = self.worker_connector.get_finished(
            scheduler_output.finished_req_ids
        )
        worker_meta = self.worker_connector.build_connector_worker_meta()
        self.worker_connector.clear_connector_metadata()

        if (
            finished_sending is None
            and finished_recving is None
            and worker_meta is None
        ):
            return metadata, None

        return metadata, KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            kv_connector_worker_meta=worker_meta,
        )


def test_decode_bench_connector_default_mode_remains_sync():
    block_size = 16
    runner = DecodeBenchTestRunner(
        block_size=block_size,
        num_gpu_blocks=100,
        dummy_prefill=False,
    )
    prompt_len = block_size * 2
    req = runner.new_request([1] * prompt_len)

    scheduler_output = runner.scheduler.schedule()
    metadata, kv_output = runner.run_connector(scheduler_output)

    assert req.status == RequestStatus.RUNNING
    assert req.num_output_tokens == 0
    assert req.num_computed_tokens == prompt_len - 1
    assert scheduler_output.num_scheduled_tokens[req.request_id] == 1
    assert req.request_id in metadata.reqs_to_fill
    assert metadata.reqs_to_fill[req.request_id][1] == prompt_len - 1
    assert kv_output is None


def test_decode_bench_connector_dummy_prefill_uses_async_full_hit():
    block_size = 16
    synthetic_token_id = 7
    runner = DecodeBenchTestRunner(
        block_size=block_size,
        num_gpu_blocks=100,
        dummy_prefill=True,
        dummy_output_token_id=synthetic_token_id,
    )
    runner.vllm_config.observability_config.enable_logging_ttft_timing_details = True
    prompt_len = block_size * 2
    req = runner.new_request([1] * prompt_len)

    first_output = runner.scheduler.schedule()
    metadata, kv_output = runner.run_connector(first_output)

    assert first_output.total_num_scheduled_tokens == 0
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req.num_tokens == prompt_len + 1
    assert req.num_output_tokens == 1
    assert list(req.output_token_ids) == [synthetic_token_id]
    assert req.request_id in metadata.reqs_to_fill
    assert metadata.reqs_to_fill[req.request_id][1] == prompt_len + 1

    assert kv_output is not None
    assert kv_output.finished_recving == {req.request_id}
    assert isinstance(
        kv_output.kv_connector_worker_meta, DecodeBenchConnectorWorkerMetadata
    )
    assert req.request_id in kv_output.kv_connector_worker_meta.req_batch_load_kv_ns

    model_runner_output = create_model_runner_output(reqs=[])
    model_runner_output.kv_connector_output = kv_output
    runner.scheduler.update_from_output(first_output, model_runner_output)
    assert req.request_id in runner.scheduler.finished_recving_kv_req_ids

    second_output = runner.scheduler.schedule()
    scheduled_req = second_output.scheduled_new_reqs[0]

    assert req.status == RequestStatus.RUNNING
    assert req.num_computed_tokens == prompt_len
    assert req.num_cached_tokens == prompt_len
    assert second_output.num_scheduled_tokens[req.request_id] == 1
    assert scheduled_req.req_id == req.request_id
    assert scheduled_req.num_computed_tokens == prompt_len
    assert scheduled_req.prompt_token_ids == [1] * prompt_len
    assert scheduled_req.output_token_ids == [synthetic_token_id]
