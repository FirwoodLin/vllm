# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = (REPO_ROOT / "benchmarks" / "offline_dp_profile" /
                "start_multinode_offline_profile.sh")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_fake_python(fake_python_path: Path, log_path: Path) -> None:
    fake_python_path.write_text(
        f"""#!{sys.executable}
import json
import subprocess
import sys
from pathlib import Path

log_path = Path({str(log_path)!r})
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")

if len(sys.argv) > 1 and sys.argv[1].endswith(
    "benchmarks/manual_multinode_poisson_runner.py"
):
    raise SystemExit(0)

raise SystemExit(subprocess.call([{sys.executable!r}, *sys.argv[1:]]))
""",
        encoding="utf-8",
    )
    fake_python_path.chmod(0o755)


def load_logged_calls(path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


@pytest.mark.benchmark
def test_start_multinode_offline_profile_help_mentions_manual_strategy_inputs(
        tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "zsh",
            str(WRAPPER_PATH),
            "--help",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Strategies with built-in offline-profile defaults:" in result.stdout
    assert "Additional runner-supported strategies:" in result.stdout
    assert "dp1tp8dcp2" in result.stdout
    assert "dp4tp8dcp2" in result.stdout
    assert "--max-num-seqs" in result.stdout
    assert "--gpu-memory-utilization" in result.stdout
    assert "--cluster 1node_h200" in result.stdout
    assert "--prompt-len N" in result.stdout
    assert "--requests-per-dp N" in result.stdout
    assert "--routing-mode internal_dplb|explicit_rank_replay" in result.stdout


@pytest.mark.benchmark
def test_start_multinode_offline_profile_defaults_to_dp32_for_lens_json(
        tmp_path: Path) -> None:
    lens_json = tmp_path / "short.json"
    lens_json.write_text("[[11, 13], [17]]\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "python_calls.jsonl"
    write_fake_python(fake_bin / "python3", log_path)

    generated_input_root = tmp_path / "generated_inputs"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GENERATED_INPUT_ROOT"] = str(generated_input_root)

    subprocess.run(
        [
            "zsh",
            str(WRAPPER_PATH),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--lens-json",
            str(lens_json),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    prepared_dir = (generated_input_root / "short" / "dp32" /
                    "dispatch_waiting_x4_plus_running")
    case_rows = read_csv_rows(prepared_dir / "custom_lens.casecsv")

    assert len(case_rows) == 1
    assert case_rows[0]["strategy"] == "dp32"
    assert case_rows[0]["cluster"] == "4node_h200"
    assert case_rows[0]["max_num_seqs"] == "256"
    assert case_rows[0]["gpu_memory_utilization"] == "0.87"
    assert case_rows[0]["dataset"] == str(
        (prepared_dir /
         "custom_lens.dispatch_waiting_x4_plus_running.lengths.csv").resolve())

    logged_calls = load_logged_calls(log_path)
    runner_call = next(call for call in logged_calls
                       if call[0].endswith(
                           "benchmarks/manual_multinode_poisson_runner.py"))
    case_csv_index = runner_call.index("--case-csv")
    assert runner_call[case_csv_index:case_csv_index + 2] == [
        "--case-csv",
        str(prepared_dir / "custom_lens.casecsv"),
    ]


@pytest.mark.benchmark
def test_start_multinode_offline_profile_isolates_lens_inputs_by_strategy(
        tmp_path: Path) -> None:
    lens_json = tmp_path / "issue01.json"
    lens_json.write_text("[31, 37]\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "python_calls.jsonl"
    write_fake_python(fake_bin / "python3", log_path)

    generated_input_root = tmp_path / "generated_inputs"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GENERATED_INPUT_ROOT"] = str(generated_input_root)

    subprocess.run(
        [
            "zsh",
            str(WRAPPER_PATH),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--cluster",
            "4node_h200",
            "--strategy",
            "dp4dcp8",
            "--model",
            "deepseek_v3_1024k",
            "--lens-json",
            str(lens_json),
            "--dispatch-policy",
            "least_cache",
            "--max-requests",
            "csv_rows",
            "--profile-delay-iterations",
            "32",
            "--pause-before-profile",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    prepared_dir = (generated_input_root / "issue01" / "dp4dcp8" /
                    "dispatch_least_cache")
    case_rows = read_csv_rows(prepared_dir / "custom_lens.casecsv")

    assert len(case_rows) == 1
    assert case_rows[0]["cluster"] == "4node_h200"
    assert case_rows[0]["strategy"] == "dp4dcp8"
    assert case_rows[0]["model"] == "deepseek_v3_1024k"
    assert case_rows[0]["dispatch_policy"] == "least_cache"
    assert case_rows[0]["max_requests"] == "csv_rows"
    assert case_rows[0]["max_num_seqs"] == "1024"
    assert case_rows[0]["gpu_memory_utilization"] == "0.85"

    logged_calls = load_logged_calls(log_path)
    prepare_call = next(call for call in logged_calls
                        if call[0].endswith(
                            "benchmarks/offline_dp_profile/prepare_custom_lens_case.py"
                        ))
    runner_call = next(call for call in logged_calls
                       if call[0].endswith(
                           "benchmarks/manual_multinode_poisson_runner.py"))

    assert "--strategy" in prepare_call
    assert "dp4dcp8" in prepare_call
    assert "--output-dir" in prepare_call
    assert str(prepared_dir) in prepare_call
    assert "--case-csv" in runner_call
    assert str(prepared_dir / "custom_lens.casecsv") in runner_call
    assert "--frontend-extra-arg=--pause-before-profile" in runner_call
    assert "--frontend-extra-arg=32" in runner_call
    assert "--headless-extra-arg=32" in runner_call


@pytest.mark.benchmark
def test_start_multinode_offline_profile_supports_explicit_rank_replay(
        tmp_path: Path) -> None:
    lens_json = tmp_path / "mix.json"
    lens_json.write_text(str(([524288] * 4) + ([2048] * 16)) + "\n",
                         encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "python_calls.jsonl"
    write_fake_python(fake_bin / "python3", log_path)

    generated_input_root = tmp_path / "generated_inputs"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GENERATED_INPUT_ROOT"] = str(generated_input_root)

    subprocess.run(
        [
            "zsh",
            str(WRAPPER_PATH),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--strategy",
            "dp8dcp4",
            "--lens-json",
            str(lens_json),
            "--dispatch-policy",
            "least_batch",
            "--routing-mode",
            "explicit_rank_replay",
            "--warmup-requests",
            "8",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    prepared_dir = (generated_input_root / "mix" / "dp8dcp4" /
                    "dispatch_least_batch" / "routing_explicit_rank_replay")
    case_rows = read_csv_rows(prepared_dir / "custom_lens.casecsv")
    length_rows = read_csv_rows(
        prepared_dir / "custom_lens.dispatch_least_batch.lengths.csv")

    assert len(case_rows) == 1
    assert case_rows[0]["strategy"] == "dp8dcp4"
    assert case_rows[0]["dispatch_policy"] == "least_batch"
    assert case_rows[0]["dataset"] == str(
        (prepared_dir / "custom_lens.dispatch_least_batch.lengths.csv").resolve())
    assert length_rows[:8] == [
        {
            "prompt_len": "2048",
            "output_len": "64",
            "data_parallel_rank": str(rank),
        }
        for rank in range(8)
    ]
    assert [row["data_parallel_rank"] for row in length_rows[8:12]] == [
        "0", "1", "2", "3"
    ]

    logged_calls = load_logged_calls(log_path)
    prepare_call = next(call for call in logged_calls
                        if call[0].endswith(
                            "benchmarks/offline_dp_profile/prepare_custom_lens_case.py"
                        ))
    runner_call = next(call for call in logged_calls
                       if call[0].endswith(
                           "benchmarks/manual_multinode_poisson_runner.py"))

    assert "--routing-mode" in prepare_call
    assert "explicit_rank_replay" in prepare_call
    assert "--data-parallel-size" in prepare_call
    assert "8" in prepare_call
    assert "--data-parallel-size-local" in prepare_call
    assert "2" in prepare_call
    assert "--warmup-short-rows" in prepare_call
    assert "--frontend-extra-arg=--routing-mode" in runner_call
    assert "--frontend-extra-arg=explicit_rank_replay" in runner_call


@pytest.mark.benchmark
def test_start_multinode_offline_profile_supports_qwen_dp4tp4_on_2node_1and3(
        tmp_path: Path) -> None:
    lens_json = tmp_path / "short.json"
    lens_json.write_text("[[11, 13], [17]]\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "python_calls.jsonl"
    write_fake_python(fake_bin / "python3", log_path)

    generated_input_root = tmp_path / "generated_inputs"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GENERATED_INPUT_ROOT"] = str(generated_input_root)

    subprocess.run(
        [
            "zsh",
            str(WRAPPER_PATH),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--cluster",
            "2node-1and3",
            "--strategy",
            "dp4tp4",
            "--model",
            "qwen3_235b_fp8_1024k",
            "--lens-json",
            str(lens_json),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    prepared_dir = (generated_input_root / "short" / "dp4tp4" /
                    "dispatch_waiting_x4_plus_running")
    case_rows = read_csv_rows(prepared_dir / "custom_lens.casecsv")

    assert len(case_rows) == 1
    assert case_rows[0]["cluster"] == "2node-1and3"
    assert case_rows[0]["strategy"] == "dp4tp4"
    assert case_rows[0]["model"] == "qwen3_235b_fp8_1024k"
    assert case_rows[0]["max_num_seqs"] == "384"
    assert case_rows[0]["gpu_memory_utilization"] == "0.85"

    logged_calls = load_logged_calls(log_path)
    prepare_call = next(call for call in logged_calls
                        if call[0].endswith(
                            "benchmarks/offline_dp_profile/prepare_custom_lens_case.py"
                        ))
    runner_call = next(call for call in logged_calls
                       if call[0].endswith(
                           "benchmarks/manual_multinode_poisson_runner.py"))

    assert "--cluster" in prepare_call
    assert "2node-1and3" in prepare_call
    assert "--strategy" in prepare_call
    assert "dp4tp4" in prepare_call
    assert "--case-csv" in runner_call
    assert str(prepared_dir / "custom_lens.casecsv") in runner_call


@pytest.mark.benchmark
def test_start_multinode_offline_profile_supports_uniform_qwen_dp4tp4_workload(
        tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "python_calls.jsonl"
    write_fake_python(fake_bin / "python3", log_path)

    generated_input_root = tmp_path / "generated_inputs"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["GENERATED_INPUT_ROOT"] = str(generated_input_root)

    subprocess.run(
        [
            "zsh",
            str(WRAPPER_PATH),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--cluster",
            "4node_h200",
            "--strategy",
            "dp4tp4",
            "--model",
            "/mnt/nvme1n1/ml_research/models/qwen3-235B-fp8",
            "--prompt-len",
            "2048",
            "--requests-per-dp",
            "512",
            "--dispatch-policy",
            "least_batch",
            "--max-requests",
            "csv_rows",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    prepared_dir = (generated_input_root / "uniform_prompt2048_perdp512" /
                    "dp4tp4" / "dispatch_least_batch")
    case_rows = read_csv_rows(prepared_dir / "custom_lens.casecsv")
    length_rows = read_csv_rows(
        prepared_dir / "custom_lens.dispatch_least_batch.lengths.csv")

    assert len(case_rows) == 1
    assert case_rows[0]["cluster"] == "4node_h200"
    assert case_rows[0]["strategy"] == "dp4tp4"
    assert case_rows[0]["model"] == (
        "/mnt/nvme1n1/ml_research/models/qwen3-235B-fp8"
    )
    assert case_rows[0]["max_requests"] == "csv_rows"
    assert case_rows[0]["max_num_seqs"] == "384"
    assert case_rows[0]["gpu_memory_utilization"] == "0.85"
    assert len(length_rows) == 2048
    assert length_rows[0] == {
        "prompt_len": "2048",
        "output_len": "64",
    }
    assert length_rows[-1] == {
        "prompt_len": "2048",
        "output_len": "64",
    }

    logged_calls = load_logged_calls(log_path)
    prepare_call = next(call for call in logged_calls
                        if call[0].endswith(
                            "benchmarks/offline_dp_profile/prepare_custom_lens_case.py"
                        ))
    runner_call = next(call for call in logged_calls
                       if call[0].endswith(
                           "benchmarks/manual_multinode_poisson_runner.py"))

    assert "--uniform-prompt-len" in prepare_call
    assert "2048" in prepare_call
    assert "--repeat-count" in prepare_call
    assert "2048" in prepare_call
    assert "--case-csv" in runner_call
    assert str(prepared_dir / "custom_lens.casecsv") in runner_call
