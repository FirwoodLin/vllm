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
