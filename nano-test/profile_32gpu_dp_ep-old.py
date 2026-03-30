# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

"""
Profile 32-GPU multi-node DP + EP inference with fixed synthetic workloads.

Default topology:
    - 4 data-parallel ranks across 4 nodes
    - 8 GPUs per DP rank (tensor-parallel / expert-parallel world)
    - 32 GPUs total

Scenarios:
    1. uniform_short:
       64 short requests per GPU card
       input_len=4K, output_len=32
    2. mixed_long_tail:
       63 short + 1 long request per GPU card
       short: input_len=4K, output_len=32
       long:  input_len=256K, output_len=32

Example on node 0:
    python nano-test/profile_32gpu_dp_ep.py \
        --model /path/to/model \
        --scenario uniform_short \
        -dp 4 -tp 8 -dcp 8 \
        --node-size 4 \
        --node-rank 0 \
        --master-addr 10.0.0.1 \
        --master-port 26199 \
        --profile-dir /path/to/profile

Then run the same command on the other three nodes with --node-rank 1/2/3.
"""

import json
import multiprocessing
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import sleep
from typing import Any

from vllm import EngineArgs, LLM, SamplingParams
from vllm.config import ProfilerConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.utils import StatelessProcessGroup
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port

SHORT_INPUT_LEN = 4 * 1024
LONG_INPUT_LEN = 256 * 1024
OUTPUT_LEN = 32
UNIFORM_REQS_PER_CARD = 64
MIXED_SHORT_REQS_PER_CARD = 63
MIXED_LONG_REQS_PER_CARD = 1
WARMUP_UNIFORM_REQS_PER_CARD = 8
WARMUP_MIXED_SHORT_REQS_PER_CARD = 7
WARMUP_MIXED_LONG_REQS_PER_CARD = 1
TOKEN_ID_VOCAB_SIZE = 10_000
CONTROL_PORT_OFFSET = 200
DEFAULT_KV_TRANSFER_CONFIG = {
    "kv_connector": "DecodeBenchConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "fill_mean": 0.015,
        "fill_std": 0.0,
    },
}
DEFAULT_DEEP_GEMM_WARMUP = os.getenv("VLLM_DEEP_GEMM_WARMUP", "skip")
os.environ.setdefault(
        "VLLM_RPC_TIMEOUT", "1800000"
    )


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    short_requests: int
    long_requests: int
    short_input_len: int
    long_input_len: int
    output_len: int

    @property
    def total_requests(self) -> int:
        return self.short_requests + self.long_requests

    @property
    def total_input_tokens(self) -> int:
        return (
            self.short_requests * self.short_input_len
            + self.long_requests * self.long_input_len
        )

    @property
    def required_max_model_len(self) -> int:
        max_prompt_len = (
            self.long_input_len if self.long_requests > 0 else self.short_input_len
        )
        return max_prompt_len + self.output_len


def create_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="Profile multi-node DP + EP inference with fixed workloads."
    )

    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        data_parallel_size=4,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        attention_backend="FLASHMLA",
        enable_expert_parallel=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        load_format="dummy",
        kv_transfer_config=DEFAULT_KV_TRANSFER_CONFIG,
        all2all_backend="deepep_low_latency",
        nnodes=4,
        master_port=26199,
    )

    parser.add_argument(
        "--node-size",
        dest="nnodes",
        type=int,
        default=4,
        help="Alias for --nnodes. Total number of nodes participating in the DP job.",
    )
    parser.add_argument(
        "--scenario",
        choices=["uniform_short", "mixed_long_tail"],
        default="uniform_short",
        help="Workload scenario to profile.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./nano-test-profiles",
        help="Directory to save torch profiler traces and scenario summaries.",
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default="",
        help="Optional prefix for profile trace names.",
    )
    parser.add_argument(
        "--token-seed",
        type=int,
        default=20260323,
        help="Base seed for synthetic prompt token ids.",
    )
    parser.add_argument(
        "--requests-per-card",
        type=int,
        default=UNIFORM_REQS_PER_CARD,
        help="Requests per GPU card for the uniform_short scenario.",
    )
    parser.add_argument(
        "--warmup-requests-per-card",
        type=int,
        default=WARMUP_UNIFORM_REQS_PER_CARD,
        help="Warmup requests per GPU card for the uniform_short scenario.",
    )
    parser.add_argument(
        "--short-requests-per-card",
        type=int,
        default=MIXED_SHORT_REQS_PER_CARD,
        help="Short requests per GPU card for the mixed_long_tail scenario.",
    )
    parser.add_argument(
        "--long-requests-per-card",
        type=int,
        default=MIXED_LONG_REQS_PER_CARD,
        help="Long requests per GPU card for the mixed_long_tail scenario.",
    )
    parser.add_argument(
        "--warmup-short-requests-per-card",
        type=int,
        default=WARMUP_MIXED_SHORT_REQS_PER_CARD,
        help="Warmup short requests per GPU card for the mixed_long_tail scenario.",
    )
    parser.add_argument(
        "--warmup-long-requests-per-card",
        type=int,
        default=WARMUP_MIXED_LONG_REQS_PER_CARD,
        help="Warmup long requests per GPU card for the mixed_long_tail scenario.",
    )
    parser.add_argument(
        "--short-input-len",
        type=int,
        default=SHORT_INPUT_LEN,
        help="Prompt length for short requests.",
    )
    parser.add_argument(
        "--long-input-len",
        type=int,
        default=LONG_INPUT_LEN,
        help="Prompt length for long requests.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=OUTPUT_LEN,
        help="Generated length for every request.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="How many full-batch warmup iterations to run before profiling.",
    )
    parser.add_argument(
        "--deep-gemm-warmup",
        choices=["skip", "relax", "full"],
        default=DEFAULT_DEEP_GEMM_WARMUP,
        help=(
            "DeepGEMM kernel warmup mode for engine startup. "
            'Default: %(default)s. "skip" is usually preferred for this '
            "decode-focused benchmark because the script's own warmup requests "
            "already JIT the shapes used before profiling."
        ),
    )
    parser.add_argument(
        "--post-profile-sleep",
        type=int,
        default=30,
        help="Seconds to sleep after stop_profile for profiler flush.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds before killing an unresponsive local worker process.",
    )
    return parser

def make_prompt_batch(
    *,
    short_requests: int,
    long_requests: int,
    short_input_len: int,
    long_input_len: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    rng = random.Random(seed)
    prompts: list[list[int]] = []

    for _ in range(short_requests):
        prompts.append(
            [rng.randrange(TOKEN_ID_VOCAB_SIZE) for _ in range(short_input_len)]
        )

    for _ in range(long_requests):
        prompts.append(
            [rng.randrange(TOKEN_ID_VOCAB_SIZE) for _ in range(long_input_len)]
        )

    rng.shuffle(prompts)
    return [{"prompt_token_ids": prompt_token_ids} for prompt_token_ids in prompts]


def build_scenario(
    scenario_name: str,
    *,
    cards_per_dp_rank: int,
    short_input_len: int,
    long_input_len: int,
    output_len: int,
    requests_per_card: int,
    short_requests_per_card: int,
    long_requests_per_card: int,
) -> ScenarioSpec:
    if scenario_name == "uniform_short":
        return ScenarioSpec(
            name=scenario_name,
            description=(
                f"{requests_per_card} short requests/card, "
                f"input_len={short_input_len}, output_len={output_len}"
            ),
            short_requests=requests_per_card * cards_per_dp_rank,
            long_requests=0,
            short_input_len=short_input_len,
            long_input_len=long_input_len,
            output_len=output_len,
        )
    if scenario_name == "mixed_long_tail":
        return ScenarioSpec(
            name=scenario_name,
            description=(
                f"{short_requests_per_card} short + "
                f"{long_requests_per_card} long requests/card, "
                f"short_input_len={short_input_len}, "
                f"long_input_len={long_input_len}, "
                f"output_len={output_len}"
            ),
            short_requests=short_requests_per_card * cards_per_dp_rank,
            long_requests=long_requests_per_card * cards_per_dp_rank,
            short_input_len=short_input_len,
            long_input_len=long_input_len,
            output_len=output_len,
        )
    raise ValueError(f"Unsupported scenario: {scenario_name}")


def build_profiler_config(profile_dir: str) -> ProfilerConfig:
    profile_path = Path(profile_dir).expanduser().resolve()
    profile_path.mkdir(parents=True, exist_ok=True)
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(profile_path),
        ignore_frontend=True,
    )


def build_sampling_params(output_len: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_len,
        min_tokens=output_len,
        ignore_eos=True,
    )


def build_legacy_compatible_compilation_config(
    compilation_config: Any | None,
) -> dict[str, Any]:
    if isinstance(compilation_config, dict):
        config = dict(compilation_config)
    elif compilation_config is None:
        config = {}
    else:
        config = {
            field.name: getattr(compilation_config, field.name)
            for field in fields(compilation_config)
            if field.init
        }

    # Stay closer to the old script: prefer FULL_DECODE_ONLY unless the user
    # supplied something else, but do not force capture sizes here.
    config.setdefault("cudagraph_mode", CUDAGraphMode.FULL_DECODE_ONLY)
    return config


def init_control_group(
    *,
    dp_size: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    control_timeout: int,
) -> StatelessProcessGroup:
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
        store_timeout=control_timeout,
    )


def save_summary_if_needed(
    *,
    gathered_stats: list[dict[str, Any]],
    profile_dir: str,
    scenario: ScenarioSpec,
    global_gpu_count: int,
) -> None:
    rank_stats = sorted(gathered_stats, key=lambda item: item["global_dp_rank"])
    total_input_tokens = sum(item["total_input_tokens"] for item in rank_stats)
    total_output_tokens = sum(item["total_output_tokens"] for item in rank_stats)
    total_requests = sum(item["total_requests"] for item in rank_stats)

    summary = {
        "scenario": asdict(scenario),
        "global_gpu_count": global_gpu_count,
        "total_requests": total_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "per_rank": rank_stats,
    }

    summary_path = (
        Path(profile_dir).expanduser().resolve() / f"{scenario.name}_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote aggregated summary to {summary_path}")


def run_profile(
    *,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    warmup_scenario: ScenarioSpec,
    scenario: ScenarioSpec,
    profile_prefix: str,
    engine_args: dict[str, Any],
    token_seed: int,
    warmup_iters: int,
    post_profile_sleep: int,
    control_timeout: int,
    cards_per_dp_rank: int,
    global_gpu_count: int,
) -> None:
    os.environ.setdefault(
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "uniform_random"
    )
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    control_group = init_control_group(
        dp_size=dp_size,
        global_dp_rank=global_dp_rank,
        dp_master_ip=dp_master_ip,
        dp_master_port=dp_master_port,
        control_timeout=control_timeout,
    )

    rank_seed = token_seed + global_dp_rank * 10_000
    warmup_prompts = make_prompt_batch(
        short_requests=warmup_scenario.short_requests,
        long_requests=warmup_scenario.long_requests,
        short_input_len=warmup_scenario.short_input_len,
        long_input_len=warmup_scenario.long_input_len,
        seed=rank_seed,
    )
    profile_prompts = make_prompt_batch(
        short_requests=scenario.short_requests,
        long_requests=scenario.long_requests,
        short_input_len=scenario.short_input_len,
        long_input_len=scenario.long_input_len,
        seed=rank_seed + 1_000,
    )

    print(
        f"DP rank {global_dp_rank}: scenario={scenario.name}, "
        f"warmup_requests={warmup_scenario.total_requests}, "
        f"profile_requests={scenario.total_requests}, "
        f"warmup_short={warmup_scenario.short_requests}, "
        f"warmup_long={warmup_scenario.long_requests}, "
        f"profile_short={scenario.short_requests}, "
        f"profile_long={scenario.long_requests}, "
        f"cards_per_dp_rank={cards_per_dp_rank}, "
        f"global_gpu_count={global_gpu_count}"
    )

    llm = LLM(**engine_args)
    sampling_params = build_sampling_params(scenario.output_len)

    for warmup_idx in range(warmup_iters):
        print(
            f"DP rank {global_dp_rank}: warmup "
            f"{warmup_idx + 1}/{warmup_iters} with {len(warmup_prompts)} requests"
        )
        llm.generate(warmup_prompts, sampling_params)

    control_group.barrier(timeout=control_timeout)
    print(f"DP rank {global_dp_rank}: start_profile({profile_prefix})")
    llm.start_profile(profile_prefix=profile_prefix)
    outputs = llm.generate(profile_prompts, sampling_params)
    llm.stop_profile()
    print(f"DP rank {global_dp_rank}: stop_profile()")
    control_group.barrier(timeout=control_timeout)

    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    rank_stats = {
        "global_dp_rank": global_dp_rank,
        "local_dp_rank": local_dp_rank,
        "scenario": scenario.name,
        "total_requests": len(outputs),
        "short_requests": scenario.short_requests,
        "long_requests": scenario.long_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "max_prompt_len": max(len(output.prompt_token_ids) for output in outputs),
        "min_prompt_len": min(len(output.prompt_token_ids) for output in outputs),
    }
    print(f"DP rank {global_dp_rank} stats: {json.dumps(rank_stats, sort_keys=True)}")

    gathered_stats = control_group.all_gather_obj(rank_stats)
    if global_dp_rank == 0:
        save_summary_if_needed(
            gathered_stats=gathered_stats,
            profile_dir=engine_args["profiler_config"].torch_profiler_dir,
            scenario=scenario,
            global_gpu_count=global_gpu_count,
        )

    sleep(post_profile_sleep)


def main() -> None:
    parser = create_parser()
    args = vars(parser.parse_args())

    dp_size = args.pop("data_parallel_size")
    node_size = args.pop("nnodes")
    node_rank = args.pop("node_rank")
    master_addr = args.pop("master_addr")
    master_port = args.pop("master_port")
    scenario_name = args.pop("scenario")
    profile_dir = args.pop("profile_dir")
    profile_prefix = args.pop("profile_prefix")
    token_seed = args.pop("token_seed")
    requests_per_card = args.pop("requests_per_card")
    warmup_requests_per_card = args.pop("warmup_requests_per_card")
    short_requests_per_card = args.pop("short_requests_per_card")
    long_requests_per_card = args.pop("long_requests_per_card")
    warmup_short_requests_per_card = args.pop("warmup_short_requests_per_card")
    warmup_long_requests_per_card = args.pop("warmup_long_requests_per_card")
    short_input_len = args.pop("short_input_len")
    long_input_len = args.pop("long_input_len")
    output_len = args.pop("output_len")
    warmup_iters = args.pop("warmup_iters")
    deep_gemm_warmup = args.pop("deep_gemm_warmup")
    post_profile_sleep = args.pop("post_profile_sleep")
    timeout = args.pop("timeout")

    os.environ["VLLM_DEEP_GEMM_WARMUP"] = deep_gemm_warmup

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = master_addr
        dp_master_port = master_port

    if dp_size % node_size != 0:
        raise ValueError(
            f"data_parallel_size ({dp_size}) must be divisible by node_size "
            f"({node_size})."
        )
    if output_len <= 0:
        raise ValueError("output_len must be positive.")
    if not args["enable_expert_parallel"]:
        raise ValueError(
            "This script is intended for DP + EP profiling. "
            "Please keep enable_expert_parallel=True."
        )
    if requests_per_card <= 0:
        raise ValueError("requests_per_card must be positive.")
    if warmup_requests_per_card <= 0:
        raise ValueError("warmup_requests_per_card must be positive.")
    if short_requests_per_card < 0 or long_requests_per_card < 0:
        raise ValueError("Mixed-scenario request counts must be non-negative.")
    if (
        warmup_short_requests_per_card < 0
        or warmup_long_requests_per_card < 0
    ):
        raise ValueError("Warmup mixed-scenario request counts must be non-negative.")
    if short_requests_per_card + long_requests_per_card <= 0:
        raise ValueError("Mixed-scenario total requests per card must be positive.")
    if warmup_short_requests_per_card + warmup_long_requests_per_card <= 0:
        raise ValueError(
            "Warmup mixed-scenario total requests per card must be positive."
        )
    if short_input_len <= 0 or long_input_len <= 0:
        raise ValueError("Input lengths must be positive.")

    cards_per_dp_rank = (
        args["tensor_parallel_size"] * args.get("pipeline_parallel_size", 1)
    )
    global_gpu_count = dp_size * cards_per_dp_rank
    warmup_scenario = build_scenario(
        scenario_name,
        cards_per_dp_rank=cards_per_dp_rank,
        short_input_len=short_input_len,
        long_input_len=long_input_len,
        output_len=output_len,
        requests_per_card=warmup_requests_per_card,
        short_requests_per_card=warmup_short_requests_per_card,
        long_requests_per_card=warmup_long_requests_per_card,
    )
    scenario = build_scenario(
        scenario_name,
        cards_per_dp_rank=cards_per_dp_rank,
        short_input_len=short_input_len,
        long_input_len=long_input_len,
        output_len=output_len,
        requests_per_card=requests_per_card,
        short_requests_per_card=short_requests_per_card,
        long_requests_per_card=long_requests_per_card,
    )

    profile_name = (
        profile_prefix
        if profile_prefix
        else f"{scenario.name}_dp{dp_size}_tp{args['tensor_parallel_size']}"
    )

    args["profiler_config"] = build_profiler_config(profile_dir)

    # Requests are constructed per DP instance as:
    #   requests/card * cards_per_dp_rank
    # DCP reuses the TP ranks, so it does not change cards_per_dp_rank.
    requests_per_dp_instance = scenario.total_requests
    current_max_num_seqs = args.get("max_num_seqs") or 0
    args["max_num_seqs"] = max(
        current_max_num_seqs, requests_per_dp_instance
    )
    args["compilation_config"] = build_legacy_compatible_compilation_config(
        args.get("compilation_config"),
    )

    current_max_model_len = args.get("max_model_len") or 0
    args["max_model_len"] = max(
        current_max_model_len,
        scenario.required_max_model_len,
    )

    dp_per_node = dp_size // node_size
    print(
        "Launching profile job with "
        f"scenario={scenario.name}, description={scenario.description}, "
        f"warmup_description={warmup_scenario.description}, "
        f"dp_size={dp_size}, node_size={node_size}, node_rank={node_rank}, "
        f"cards_per_dp_rank={cards_per_dp_rank}, global_gpu_count={global_gpu_count}, "
        f"warmup_requests={warmup_scenario.total_requests}, "
        f"profile_requests={scenario.total_requests}, "
        f"deep_gemm_warmup={deep_gemm_warmup}, "
        f"enable_prefix_caching={args['enable_prefix_caching']}, "
        f"max_num_seqs={args['max_num_seqs']}, "
        f"max_num_batched_tokens={args['max_num_batched_tokens']}, "
        f"max_model_len={args['max_model_len']}, "
        f"cudagraph_capture_sizes="
        f"{args['compilation_config'].get('cudagraph_capture_sizes')}, "
        f"profile_dir={args['profiler_config'].torch_profiler_dir}"
    )
        # f"cudagraph_mode={args['compilation_config']['cudagraph_mode'].name}, "

    if current_platform.is_rocm():
        multiprocessing.set_start_method("spawn", force=True)

    procs: list[multiprocessing.Process] = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = multiprocessing.get_context("spawn").Process(
            target=run_profile,
            kwargs=dict(
                dp_size=dp_size,
                local_dp_rank=local_dp_rank,
                global_dp_rank=global_dp_rank,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                warmup_scenario=warmup_scenario,
                scenario=scenario,
                profile_prefix=profile_name,
                engine_args=dict(args),
                token_seed=token_seed,
                warmup_iters=warmup_iters,
                post_profile_sleep=post_profile_sleep,
                control_timeout=timeout,
                cards_per_dp_rank=cards_per_dp_rank,
                global_gpu_count=global_gpu_count,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} after timeout={timeout}s.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
