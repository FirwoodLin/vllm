# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            -dp=2 \
            -tp=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    -dp=2 \
                    -tp=2 \
                    --dp-num-nodes=2 \
                    --dp-node-rank=0 \
                    --dp-master-addr=10.99.48.128 \
                    --dp-master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    -dp=2 \
                    -tp=2 \
                    --dp-num-nodes=2 \
                    --dp-node-rank=1 \
                    --dp-master-addr=10.99.48.128 \
                    --dp-master-port=13345

Per-rank request config:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            -dp=2 \
            -tp=2 \
            --request-config=/tmp/dp_requests.json

    Example /tmp/dp_requests.json:
    {
      "0": [
        {
          "prompt": "Hello, my name is",
          "sampling_params": {"temperature": 0.0, "max_tokens": 16}
        }
      ],
      "1": [
        {
          "prompt_token_count": 4096,
          "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 16,
            "min_tokens": 16,
            "ignore_eos": true
          }
        }
      ]
    }

DeepSeek-V3 style multi-node profiling:
    python examples/offline_inference/data_parallel.py \
            --start-sh-defaults \
            --profile \
            --warmup-iters=5 \
            --request-config=/tmp/dp_requests.json \
            -dp=4 \
            --dp-num-nodes=4 \
            --dp-node-rank=0 \
            --dp-master-addr=10.0.0.1 \
            --dp-master-port=29501 \
            --profiler-config.profiler=cuda
"""

import argparse
import dataclasses
import json
import os
import random
from time import sleep
from typing import Any

from vllm import LLM, EngineArgs, SamplingParams
from vllm.distributed.utils import StatelessProcessGroup
from vllm.inputs import PromptType
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port

DEFAULT_REQUEST_TOKEN_VOCAB_SIZE = 10_000
CONTROL_PORT_OFFSET = 200
DEFAULT_START_SH_MODEL = "/mnt/nvme1n1/ml_research/models/deepseek-v3"
DEFAULT_START_SH_KV_TRANSFER_CONFIG = {
    "kv_connector": "DecodeBenchConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "fill_mean": 0.015,
        "fill_std": 0.0,
    },
}
DEFAULT_START_SH_COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
}


def create_parser():
    parser = FlexibleArgumentParser(description="Data Parallel Inference")

    # Add all engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="ibm-research/PowerMoE-3b",
        enable_expert_parallel=True,
    )

    # Add DP-specific args (separate from engine args to avoid conflicts)
    parser.add_argument(
        "--dp-num-nodes",
        type=int,
        default=1,
        help="Total number of nodes for data parallel.",
    )
    parser.add_argument(
        "--dp-node-rank",
        type=int,
        default=0,
        help="Rank of the current node for data parallel.",
    )
    parser.add_argument(
        "--dp-master-addr",
        type=str,
        default="",
        help="Master node IP address for DP coordination.",
    )
    parser.add_argument(
        "--dp-master-port",
        type=int,
        default=0,
        help="Master node port for DP coordination.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Number of seconds before unresponsive process is killed.",
    )
    parser.add_argument(
        "--request-config",
        type=str,
        default="",
        help=(
            "Optional path to a JSON file keyed by global_dp_rank. Each value "
            "must be a list of requests with `prompt`, "
            "`prompt_prefix`+`prompt_size`, `prompt_token_ids`, or "
            "`prompt_token_count`, plus optional `sampling_params`."
        ),
    )
    parser.add_argument(
        "--request-token-seed",
        type=int,
        default=20260325,
        help=(
            "Base RNG seed used when request-config entries specify "
            "`prompt_token_count` instead of explicit token ids."
        ),
    )
    parser.add_argument(
        "--request-token-vocab-size",
        type=int,
        default=DEFAULT_REQUEST_TOKEN_VOCAB_SIZE,
        help="Vocabulary size for synthetic prompt_token_ids generation.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Call start_profile/stop_profile around the profiled generate call.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Number of warmup generate iterations before profiling/output.",
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default="",
        help="Optional trace prefix passed to llm.start_profile().",
    )
    parser.add_argument(
        "--post-profile-sleep",
        type=int,
        default=1,
        help="Seconds to sleep after the profiled run for profiler flush.",
    )
    parser.add_argument(
        "--start-sh-defaults",
        action="store_true",
        help=(
            "Fill engine args with DeepSeek-V3 decode-benchmark defaults "
            "matching nano-test/start.sh when those fields are still unset."
        ),
    )

    return parser


def _build_prompt(prefix: str, prompt_size: int) -> str:
    if prompt_size <= 0:
        return ""
    if not prefix:
        prefix = " "
    if len(prefix) >= prompt_size:
        return prefix[:prompt_size]
    repeat_count = (prompt_size + len(prefix) - 1) // len(prefix)
    return (prefix * repeat_count)[:prompt_size]


def _build_prompt_token_ids(
    prompt_token_count: int,
    *,
    seed: int,
    vocab_size: int,
) -> list[int]:
    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be non-negative.")
    if vocab_size <= 0:
        raise ValueError("token_id_vocab_size must be positive.")
    rng = random.Random(seed)
    return [rng.randrange(vocab_size) for _ in range(prompt_token_count)]


def _load_requests_for_rank(
    request_config_path: str,
    global_dp_rank: int,
    request_token_seed: int,
    request_token_vocab_size: int,
) -> tuple[list[PromptType], list[SamplingParams]]:
    with open(request_config_path, encoding="utf-8") as request_config_file:
        request_table = json.load(request_config_file)

    if not isinstance(request_table, dict):
        raise ValueError(
            "--request-config must point to a JSON object keyed by "
            "global_dp_rank."
        )

    rank_key = str(global_dp_rank)
    if rank_key not in request_table:
        available_ranks = ", ".join(sorted(request_table)) or "<none>"
        raise ValueError(
            f"Missing request list for global_dp_rank={global_dp_rank}. "
            f"Available ranks: {available_ranks}"
        )

    raw_requests = request_table[rank_key]
    if not isinstance(raw_requests, list):
        raise ValueError(
            f"Request list for global_dp_rank={global_dp_rank} must be a list."
        )

    prompts: list[PromptType] = []
    sampling_params_list: list[SamplingParams] = []

    for request_idx, request in enumerate(raw_requests):
        if not isinstance(request, dict):
            raise ValueError(
                f"Request #{request_idx} for global_dp_rank={global_dp_rank} "
                "must be a JSON object."
            )

        prompt = request.get("prompt")
        prompt_prefix = request.get("prompt_prefix", "")
        prompt_size = request.get("prompt_size")
        prompt_token_ids = request.get("prompt_token_ids")
        prompt_token_count = request.get("prompt_token_count")

        prompt_spec_count = sum(
            (
                prompt is not None,
                prompt_size is not None,
                prompt_token_ids is not None,
                prompt_token_count is not None,
            )
        )
        if prompt_spec_count != 1:
            raise ValueError(
                f"Request #{request_idx} for global_dp_rank={global_dp_rank} "
                "must define exactly one of `prompt`, `prompt_size`, "
                "`prompt_token_ids`, or `prompt_token_count`."
            )

        if prompt is not None:
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank={global_dp_rank} "
                    "has a non-string prompt."
                )
            request_prompt: PromptType = prompt
        elif prompt_size is not None:
            if not isinstance(prompt_prefix, str):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank="
                    f"{global_dp_rank} has a non-string prompt_prefix."
                )
            if not isinstance(prompt_size, int):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank="
                    f"{global_dp_rank} must define an integer `prompt_size`."
                )
            request_prompt = _build_prompt(prompt_prefix, prompt_size)
        elif prompt_token_ids is not None:
            if (
                not isinstance(prompt_token_ids, list)
                or not prompt_token_ids
                or not all(isinstance(token_id, int) for token_id in prompt_token_ids)
            ):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank="
                    f"{global_dp_rank} has invalid `prompt_token_ids`."
                )
            request_prompt = {"prompt_token_ids": prompt_token_ids}
        else:
            if not isinstance(prompt_token_count, int):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank="
                    f"{global_dp_rank} has non-integer `prompt_token_count`."
                )

            token_id_seed = request.get(
                "token_id_seed",
                request_token_seed + global_dp_rank * 100_000 + request_idx,
            )
            token_id_vocab_size = request.get(
                "token_id_vocab_size",
                request_token_vocab_size,
            )
            if not isinstance(token_id_seed, int):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank="
                    f"{global_dp_rank} has non-integer `token_id_seed`."
                )
            if not isinstance(token_id_vocab_size, int):
                raise ValueError(
                    f"Request #{request_idx} for global_dp_rank="
                    f"{global_dp_rank} has non-integer `token_id_vocab_size`."
                )

            request_prompt = {
                "prompt_token_ids": _build_prompt_token_ids(
                    prompt_token_count,
                    seed=token_id_seed,
                    vocab_size=token_id_vocab_size,
                )
            }

        raw_sampling_params = request.get("sampling_params", {})
        if not isinstance(raw_sampling_params, dict):
            raise ValueError(
                f"Request #{request_idx} for global_dp_rank={global_dp_rank} "
                "has non-object `sampling_params`."
            )

        try:
            sampling_params = SamplingParams(**raw_sampling_params)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid sampling_params for request #{request_idx} of "
                f"global_dp_rank={global_dp_rank}: {exc}"
            ) from exc

        prompts.append(request_prompt)
        sampling_params_list.append(sampling_params)

    return prompts, sampling_params_list


def _profiler_enabled(engine_args: dict[str, Any]) -> bool:
    profiler_config = engine_args.get("profiler_config")
    if isinstance(profiler_config, dict):
        return profiler_config.get("profiler") is not None
    return getattr(profiler_config, "profiler", None) is not None


def _apply_start_sh_defaults(args: dict[str, Any], profile: bool) -> None:
    if args.get("model") in (None, "", "ibm-research/PowerMoE-3b"):
        args["model"] = DEFAULT_START_SH_MODEL
    if args.get("tensor_parallel_size", 1) == 1:
        args["tensor_parallel_size"] = 8
    if args.get("decode_context_parallel_size", 1) == 1:
        args["decode_context_parallel_size"] = 8
    if args.get("dcp_comm_backend") == "ag_rs":
        args["dcp_comm_backend"] = "a2a"
    if args.get("attention_backend") is None:
        args["attention_backend"] = "FLASHMLA"
    if args.get("all2all_backend") == "allgather_reducescatter":
        args["all2all_backend"] = "deepep_low_latency"
    if args.get("load_format") == "auto":
        args["load_format"] = "dummy"
    if args.get("gpu_memory_utilization") == 0.9:
        args["gpu_memory_utilization"] = 0.85
    if args.get("compilation_config") in (None, {}):
        args["compilation_config"] = dict(DEFAULT_START_SH_COMPILATION_CONFIG)
    if args.get("kv_transfer_config") in (None, {}):
        args["kv_transfer_config"] = dict(DEFAULT_START_SH_KV_TRANSFER_CONFIG)
    if not _profiler_enabled(args) and profile:
        args["profiler_config"] = {"profiler": "cuda"}

    args["enable_expert_parallel"] = True
    args["enable_prefix_caching"] = False
    os.environ.setdefault("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "uniform_random")


def _init_control_group(
    *,
    dp_size: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    timeout: int,
) -> StatelessProcessGroup | None:
    if dp_size <= 1:
        return None
    control_port = dp_master_port + CONTROL_PORT_OFFSET
    print(
        f"DP rank {global_dp_rank}: init control group on "
        f"{dp_master_ip}:{control_port}"
    )
    return StatelessProcessGroup.create(
        host=dp_master_ip,
        port=control_port,
        rank=global_dp_rank,
        world_size=dp_size,
        store_timeout=timeout,
    )


def main(
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    engine_args,
    request_config,
    request_token_seed,
    request_token_vocab_size,
    profile,
    warmup_iters,
    profile_prefix,
    post_profile_sleep,
    timeout,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.
    control_group = None
    if profile:
        control_group = _init_control_group(
            dp_size=dp_size,
            global_dp_rank=global_dp_rank,
            dp_master_ip=dp_master_ip,
            dp_master_port=dp_master_port,
            timeout=timeout,
        )

    if request_config:
        prompts, sampling_params = _load_requests_for_rank(
            request_config,
            global_dp_rank,
            request_token_seed,
            request_token_vocab_size,
        )
        print(
            f"DP rank {global_dp_rank} loaded {len(prompts)} requests from "
            f"{request_config}"
        )
    else:
        # Sample prompts.
        prompts: list[PromptType] = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ] * 100

        # with DP, each rank should process different prompts.
        # usually all the DP ranks process a full dataset,
        # and each rank processes a different part of the dataset.
        floor = len(prompts) // dp_size
        remainder = len(prompts) % dp_size

        # Distribute prompts into even groups.
        def start(rank):
            return rank * floor + min(rank, remainder)

        prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
        # since we are doing data parallel, every rank can have different
        # sampling params. here we set different max_tokens for different
        # ranks for demonstration.
        sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=[16, 20][global_dp_rank % 2]
        )

    if len(prompts) == 0:
        # Keep every DP rank participating even if its configured request list
        # is empty.
        prompts = [{"prompt_token_ids": [0]}]
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
        )
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create an LLM.
    llm = LLM(**engine_args)
    for warmup_idx in range(warmup_iters):
        print(
            f"DP rank {global_dp_rank}: warmup "
            f"{warmup_idx + 1}/{warmup_iters}"
        )
        llm.generate(prompts, sampling_params, use_tqdm=False)

    if profile:
        if control_group is not None:
            control_group.barrier(timeout=timeout)
        print(f"DP rank {global_dp_rank}: start_profile({profile_prefix})")
        llm.start_profile(profile_prefix=profile_prefix or None)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    if profile:
        llm.stop_profile()
        print(f"DP rank {global_dp_rank}: stop_profile()")
        if control_group is not None:
            control_group.barrier(timeout=timeout)

    # Print the outputs.
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        if prompt is None:
            prompt = f"<{len(output.prompt_token_ids)} prompt tokens>"
        generated_text = output.outputs[0].text
        print(
            f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}"
        )

    # Give engines time to pause their processing loops before exiting.
    sleep(post_profile_sleep if profile else 1)


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())

    # Extract DP-specific args (pop to remove from engine_args)
    dp_size = args.pop("data_parallel_size")
    dp_num_nodes = args.pop("dp_num_nodes")
    dp_node_rank = args.pop("dp_node_rank")
    dp_master_addr = args.pop("dp_master_addr")
    dp_master_port = args.pop("dp_master_port")
    timeout = args.pop("timeout")
    request_config = args.pop("request_config")
    request_token_seed = args.pop("request_token_seed")
    request_token_vocab_size = args.pop("request_token_vocab_size")
    profile = args.pop("profile")
    warmup_iters = args.pop("warmup_iters")
    profile_prefix = args.pop("profile_prefix")
    post_profile_sleep = args.pop("post_profile_sleep")
    start_sh_defaults = args.pop("start_sh_defaults")

    if dp_num_nodes == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port_val = get_open_port()
    else:
        dp_master_ip = dp_master_addr
        dp_master_port_val = dp_master_port

    assert dp_size % dp_num_nodes == 0, "dp_size should be divisible by dp_num_nodes"
    dp_per_node = dp_size // dp_num_nodes
    if dp_num_nodes > 1 and not dp_master_ip:
        raise ValueError("--dp-master-addr must be set when --dp-num-nodes > 1.")
    if dp_num_nodes > 1 and dp_master_port_val <= 0:
        raise ValueError("--dp-master-port must be set when --dp-num-nodes > 1.")

    if args.get("data_parallel_size_local") is None:
        args["data_parallel_size_local"] = dp_per_node
    if start_sh_defaults:
        _apply_start_sh_defaults(args, profile)

    engine_args_obj = EngineArgs.from_cli_args(argparse.Namespace(**args))
    engine_args = dataclasses.asdict(engine_args_obj)
    if profile and not _profiler_enabled(engine_args):
        raise ValueError(
            "--profile requires profiler_config to be enabled. "
            "Example: --profiler-config.profiler=cuda"
        )

    from multiprocessing import Process

    if current_platform.is_rocm():
        from multiprocessing import set_start_method

        set_start_method("spawn", force=True)

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(dp_node_rank * dp_per_node, (dp_node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port_val,
                engine_args,
                request_config,
                request_token_seed,
                request_token_vocab_size,
                profile,
                warmup_iters,
                profile_prefix,
                post_profile_sleep,
                timeout,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
