# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "offline_dp_profile"
    / "prepare_custom_lens_case.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_custom_lens_case",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prepare_custom_lens_case_writes_length_csv_and_casecsv(
        tmp_path: Path) -> None:
    module = load_module()
    base_case_csv = tmp_path / "base.casecsv"
    base_case_csv.write_text(
        (
            "enabled,name,cluster,model,dataset,strategy,dispatch_policy,"
            "request_rate,rate_phase,max_num_seqs,gpu_memory_utilization,"
            "max_requests,warmup_requests,max_model_len,data_parallel_rpc_port,"
            "reason,historical_reference\n"
            "1,base_case,cluster_a,model_a,dataset_a,dp32,"
            "waiting_x4_plus_running,40.0,offline_profile,256,0.87,256,32,"
            "1000000,29550,reason,\n"
        ),
        encoding="utf-8",
    )
    lens_json = tmp_path / "issue01.json"
    lens_json.write_text("[[11, 13], [17]]\n", encoding="utf-8")
    output_dir = tmp_path / "prepared"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(MODULE_PATH),
            "--base-case-csv",
            str(base_case_csv),
            "--lens-json",
            str(lens_json),
            "--output-dir",
            str(output_dir),
            "--output-len",
            "64",
            "--cluster",
            "4node_h200",
            "--strategy",
            "dp4dcp8",
            "--model",
            "deepseek_v3_1024k",
            "--dispatch-policy",
            "least_batch",
            "--warmup-requests",
            "8",
            "--max-requests",
            "3",
            "--request-rate",
            "45",
            "--max-num-seqs",
            "2048",
            "--gpu-memory-utilization",
            "0.91",
            "--data-parallel-rpc-port",
            "29600",
            "--case-name",
            "custom_issue01_case",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    length_rows = read_csv_rows(
        output_dir / "custom_lens.dispatch_least_batch.lengths.csv")
    case_rows = read_csv_rows(output_dir / "custom_lens.casecsv")

    assert length_rows == [
        {
            "prompt_len": "11",
            "output_len": "64",
        },
        {
            "prompt_len": "13",
            "output_len": "64",
        },
        {
            "prompt_len": "17",
            "output_len": "64",
        },
    ]
    assert len(case_rows) == 1
    assert case_rows[0]["name"] == "custom_issue01_case"
    assert case_rows[0]["cluster"] == "4node_h200"
    assert case_rows[0]["strategy"] == "dp4dcp8"
    assert case_rows[0]["model"] == "deepseek_v3_1024k"
    assert case_rows[0]["dispatch_policy"] == "least_batch"
    assert case_rows[0]["request_rate"] == "45"
    assert case_rows[0]["warmup_requests"] == "8"
    assert case_rows[0]["max_requests"] == "3"
    assert case_rows[0]["max_num_seqs"] == "2048"
    assert case_rows[0]["gpu_memory_utilization"] == "0.91"
    assert case_rows[0]["data_parallel_rpc_port"] == "29600"
    assert case_rows[0]["dataset"] == str(
        (output_dir / "custom_lens.dispatch_least_batch.lengths.csv").resolve())


def test_prepare_custom_lens_case_keeps_base_values_when_not_overridden(
        tmp_path: Path) -> None:
    module = load_module()
    base_case_csv = tmp_path / "base.casecsv"
    base_case_csv.write_text(
        (
            "enabled,name,cluster,model,dataset,strategy,dispatch_policy,"
            "request_rate,rate_phase,max_num_seqs,gpu_memory_utilization,"
            "max_requests,warmup_requests,max_model_len,data_parallel_rpc_port,"
            "reason,historical_reference\n"
            "1,base_case,cluster_a,model_a,dataset_a,dp32,least_cache,"
            "55.0,offline_profile,256,0.87,512,16,1000000,29550,reason,\n"
        ),
        encoding="utf-8",
    )
    lens_json = tmp_path / "custom.json"
    lens_json.write_text("[23, 29]\n", encoding="utf-8")
    output_dir = tmp_path / "prepared"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(MODULE_PATH),
            "--base-case-csv",
            str(base_case_csv),
            "--lens-json",
            str(lens_json),
            "--output-dir",
            str(output_dir),
        ]
        module.main()
    finally:
        sys.argv = old_argv

    case_rows = read_csv_rows(output_dir / "custom_lens.casecsv")

    assert len(case_rows) == 1
    assert case_rows[0]["name"] == "base_case__dp32__custom__dispatch_least_cache"
    assert case_rows[0]["dispatch_policy"] == "least_cache"
    assert case_rows[0]["request_rate"] == "55.0"
    assert case_rows[0]["warmup_requests"] == "16"
    assert case_rows[0]["max_requests"] == "512"
    assert case_rows[0]["dataset"] == str(
        (output_dir / "custom_lens.dispatch_least_cache.lengths.csv").resolve())


def test_prepare_custom_lens_case_applies_strategy_profile_defaults(
        tmp_path: Path) -> None:
    module = load_module()
    base_case_csv = tmp_path / "base.casecsv"
    base_case_csv.write_text(
        (
            "enabled,name,cluster,model,dataset,strategy,dispatch_policy,"
            "request_rate,rate_phase,max_num_seqs,gpu_memory_utilization,"
            "max_requests,warmup_requests,max_model_len,data_parallel_rpc_port,"
            "reason,historical_reference\n"
            "1,offline_profile_template,cluster_a,model_a,dataset_a,dp32,"
            "waiting_x4_plus_running,40.0,offline_profile,,,32,32,1000000,"
            "29550,reason,\n"
        ),
        encoding="utf-8",
    )
    lens_json = tmp_path / "issue01.json"
    lens_json.write_text("[31, 37]\n", encoding="utf-8")
    output_dir = tmp_path / "prepared"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(MODULE_PATH),
            "--base-case-csv",
            str(base_case_csv),
            "--lens-json",
            str(lens_json),
            "--output-dir",
            str(output_dir),
            "--strategy",
            "dp4dcp8",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    case_rows = read_csv_rows(output_dir / "custom_lens.casecsv")

    assert len(case_rows) == 1
    assert case_rows[0]["name"] == "offline_profile_template__dp4dcp8__issue01"
    assert case_rows[0]["strategy"] == "dp4dcp8"
    assert case_rows[0]["max_num_seqs"] == "1024"
    assert case_rows[0]["gpu_memory_utilization"] == "0.85"


def test_prepare_custom_lens_case_supports_csv_rows_max_requests(
        tmp_path: Path) -> None:
    module = load_module()
    base_case_csv = tmp_path / "base.casecsv"
    base_case_csv.write_text(
        (
            "enabled,name,cluster,model,dataset,strategy,dispatch_policy,"
            "request_rate,rate_phase,max_num_seqs,gpu_memory_utilization,"
            "max_requests,warmup_requests,max_model_len,data_parallel_rpc_port,"
            "reason,historical_reference\n"
            "1,base_case,cluster_a,model_a,dataset_a,dp32,least_cache,"
            "55.0,offline_profile,256,0.87,512,16,1000000,29550,reason,\n"
        ),
        encoding="utf-8",
    )
    lens_json = tmp_path / "custom.json"
    lens_json.write_text("[23, 29]\n", encoding="utf-8")
    output_dir = tmp_path / "prepared"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(MODULE_PATH),
            "--base-case-csv",
            str(base_case_csv),
            "--lens-json",
            str(lens_json),
            "--output-dir",
            str(output_dir),
            "--max-requests",
            "csv_rows",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    case_rows = read_csv_rows(output_dir / "custom_lens.casecsv")

    assert len(case_rows) == 1
    assert case_rows[0]["max_requests"] == "csv_rows"
