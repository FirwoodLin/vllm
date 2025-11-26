# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            --dp-size=2 \
            --tp-size=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import os
import json
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port
# import torch.distributed as dist # 导入 PyTorch 分布式库
# import datetime # 用于设置 timeout

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./"
os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="/models/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/ba82a1060073fa0ecdc70d7b1922ec071f60cf3e",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="127.0.0.1", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=26390, help="Master node port")
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode execution."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help=("Maximum number of sequences to be processed in a single iteration."),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help=("Maximum number of tokens to be processed in a single iteration."),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=("Number of seconds before unresponsive process is killed."),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=("Fraction of GPU memory vLLM is allowed to allocate (0.0, 1.0]."),
    )
    parser.add_argument(
        "--enable-dbo",
        action="store_true",
        help=("Enable microbatched execution"),
    )
    parser.add_argument(
        "--compilation-config",
        type=int,
        help=("Compilation optimization (O) mode 0-3."),
    )
    parser.add_argument(
        "--quantization",
        type=str,
    )
    parser.add_argument(
        "--kv-transfer-config",
        type=str,
        default='{ "kv_connector": "DecodeBenchConnector", "kv_role": "kv_both", "kv_connector_extra_config": { "fill_mean": 0.015, "fill_std": 0.0 } }', # decode only
        help="KV transfer configuration JSON string.",
    )
    parser.add_argument(
        "--disable-expert-parallel",
        dest="enable_expert_parallel",
        action="store_false",
        help="Disable expert parallel (default: enabled).",
    )
    parser.set_defaults(enable_expert_parallel=True)
    return parser.parse_args()


def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    enforce_eager,
    enable_expert_parallel,
    trust_remote_code,
    max_num_seqs,
    max_model_len,
    compilation_config,
    gpu_memory_utilization,
    enable_dbo,
    quantization,
    kv_transfer_config,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 2

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    # Distribute prompts into even groups.
    def start(rank):
        return rank * floor + min(rank, remainder)

    prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different [10, 16][global_rank %2]
    # ranks for demonstration.
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=10
    )

    # Create an LLM.
    if kv_transfer_config and isinstance(kv_transfer_config, str):
        kv_transfer_config = json.loads(kv_transfer_config)

    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
        decode_context_parallel_size=2,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_dbo=enable_dbo,
        quantization=quantization,
        compilation_config=compilation_config,
        load_format="dummy",
        kv_transfer_config=kv_transfer_config,
    )
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    # 建议：先做一个 Warmup，避免把模型编译/初始化的时间算进 profile 里
    # 如果你想看完整的启动过程，可以注释掉下面这两行
    print(f"DP rank {global_dp_rank} warming up...")
    llm.generate(["Warmup prompt"], sampling_params)
    
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    
    if dp_size > 1 :
        # 默认的进程组 (Default Process Group) 此时应该是我们刚刚初始化的 DP 组
        from vllm.distributed.parallel_state import get_world_group

        print(f"DP rank {global_dp_rank} reached global barrier, waiting for all ranks to stop profiling...")
        get_world_group().cpu_group.barrier()
        print(f"DP rank {global_dp_rank} passed global barrier.")
    
    
    # Print the outputs.
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}"
        )

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    import multiprocessing

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = multiprocessing.get_context("spawn").Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.enforce_eager,
                args.enable_expert_parallel,
                args.trust_remote_code,
                args.max_num_seqs,
                args.max_model_len,
                args.compilation_config,
                args.gpu_memory_utilization,
                args.enable_dbo,
                args.quantization,
                args.kv_transfer_config,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
