#!/usr/bin/env python3
"""Generate CSV plans for benchmark tests."""

import csv
from pathlib import Path

# Constants
CASE_CSV_FIELDNAMES = (
    "enabled", "name", "cluster", "model", "dataset", "strategy",
    "dispatch_policy", "request_rate", "rate_phase", "max_num_seqs",
    "gpu_memory_utilization", "max_requests", "warmup_requests",
    "max_model_len", "data_parallel_rpc_port", "reason", "historical_reference"
)

# Configuration mappings
STRATEGY_MAX_NUM_SEQS = {
    "dp16cp2": 384,
    "dp32": 256,
}

STRATEGY_GPU_MEMORY_UTILIZATION = {
    "dp16cp2": 0.85,
    "dp32": 0.87,
}

MODELS = {
    "dpsk": "deepseek_v3_1024k",
    "kimi": "kimi_k2_instruct_0905",
}

DATASETS = {
    "issue01random": "issue01_random",
    "issue03random": "issue03_random",
    "issue05random": "issue05_random",
    "long_full": "long_full",
}

def build_rate_range(start: float, stop: float, step: float) -> list[float]:
    """Build a range of request rates."""
    count = int(round((stop - start) / step))
    rates = [round(start + step * index, 10) for index in range(count + 1)]
    return rates

def calculate_max_requests(rate: float, duration: float = 600.0) -> int:
    """Calculate max_requests based on rate and duration."""
    return int(rate * duration)

def create_case_row(
    model: str,
    dataset: str,
    strategy: str,
    rate: float,
    dispatch_policy: str = "waiting_x4_plus_running",
    rate_phase: str = "custom",
    reason: str = "manual_plan",
) -> dict:
    """Create a single case row."""
    model_short = "DPSK" if "deepseek" in model else "KIMI"
    dataset_short = dataset.replace("_", "")
    
    name = f"{model_short}__{dataset}__{strategy}__rate{rate}__bs{STRATEGY_MAX_NUM_SEQS[strategy]}"
    
    return {
        "enabled": "1",
        "name": name,
        "cluster": "4node_h200",
        "model": model,
        "dataset": dataset,
        "strategy": strategy,
        "dispatch_policy": dispatch_policy,
        "request_rate": str(rate),
        "rate_phase": rate_phase,
        "max_num_seqs": str(STRATEGY_MAX_NUM_SEQS[strategy]),
        "gpu_memory_utilization": f"{STRATEGY_GPU_MEMORY_UTILIZATION[strategy]:g}",
        "max_requests": str(calculate_max_requests(rate)),
        "warmup_requests": "32",
        "max_model_len": "1000000",
        "data_parallel_rpc_port": "29550",
        "reason": reason,
        "historical_reference": "",
    }

def write_csv(filepath: Path, rows: list[dict]):
    """Write rows to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Created: {filepath} ({len(rows)} rows)")

def main():
    output_dir = Path("/mnt/nvme1n1/ml_research/linbinbin1/vllm-v0180/benchmarks/plans-unfinished/0407")
    
    # 1. DPSK: issue03random DP32 rate=21, DP16cp2 rate=26
    rows = []
    rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue03random"], "dp32", 21.0, 
                                reason="DPSK issue03random DP32 rate=21"))
    rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue03random"], "dp16cp2", 26.0,
                                reason="DPSK issue03random DP16cp2 rate=26"))
    write_csv(output_dir / "dpsk_issue03random_dp32_dp16cp2_specific.csv", rows)
    
    # 2. DPSK: issue05random DP32 rate=16, DP16cp2 rate=26
    rows = []
    rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue05random"], "dp32", 16.0,
                                reason="DPSK issue05random DP32 rate=16"))
    rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue05random"], "dp16cp2", 26.0,
                                reason="DPSK issue05random DP16cp2 rate=26"))
    write_csv(output_dir / "dpsk_issue05random_dp32_dp16cp2_specific.csv", rows)
    
    # 3. DPSK: Long_full DP32 and DP16 complete runs (full2p5 range: 2.5 to 90.0, step 2.5)
    full2p5_rates = build_rate_range(2.5, 90.0, 2.5)
    
    # DP32 full
    rows = []
    for rate in full2p5_rates:
        rows.append(create_case_row(MODELS["dpsk"], DATASETS["long_full"], "dp32", rate,
                                   rate_phase="full2p5",
                                   reason="DPSK long_full DP32 complete"))
    write_csv(output_dir / "dpsk_long_full_dp32_full2p5.csv", rows)
    
    # DP16cp2 full
    rows = []
    for rate in full2p5_rates:
        rows.append(create_case_row(MODELS["dpsk"], DATASETS["long_full"], "dp16cp2", rate,
                                   rate_phase="full2p5",
                                   reason="DPSK long_full DP16cp2 complete"))
    write_csv(output_dir / "dpsk_long_full_dp16cp2_full2p5.csv", rows)
    
    # 4. Kimi: issue01 DP16cp2 rate=57
    rows = []
    rows.append(create_case_row(MODELS["kimi"], DATASETS["issue01random"], "dp16cp2", 57.0,
                                reason="Kimi issue01 DP16cp2 rate=57"))
    write_csv(output_dir / "kimi_issue01_dp16cp2_rate57.csv", rows)
    
    # 5. DPSK: issue01random, issue03random, issue05random with least_cache dispatch policy, DP32
    # issue01: start from 10, interval 5
    issue01_rates = list(range(10, 95, 5))  # 10, 15, 20, ..., 90
    rows = []
    for rate in issue01_rates:
        rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue01random"], "dp32", float(rate),
                                   dispatch_policy="least_cache",
                                   rate_phase="step5_from10",
                                   reason="DPSK issue01random DP32 least_cache"))
    write_csv(output_dir / "dpsk_issue01random_dp32_least_cache.csv", rows)
    
    # issue03: start from 2.5, interval 2.5
    issue03_rates = build_rate_range(2.5, 90.0, 2.5)
    rows = []
    for rate in issue03_rates:
        rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue03random"], "dp32", rate,
                                   dispatch_policy="least_cache",
                                   rate_phase="step2p5_from2p5",
                                   reason="DPSK issue03random DP32 least_cache"))
    write_csv(output_dir / "dpsk_issue03random_dp32_least_cache.csv", rows)
    
    # issue05: start from 2.5, interval 2.5
    issue05_rates = build_rate_range(2.5, 90.0, 2.5)
    rows = []
    for rate in issue05_rates:
        rows.append(create_case_row(MODELS["dpsk"], DATASETS["issue05random"], "dp32", rate,
                                   dispatch_policy="least_cache",
                                   rate_phase="step2p5_from2p5",
                                   reason="DPSK issue05random DP32 least_cache"))
    write_csv(output_dir / "dpsk_issue05random_dp32_least_cache.csv", rows)

if __name__ == "__main__":
    main()
