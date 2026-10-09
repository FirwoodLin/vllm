# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


MODULE_PATH = (Path(__file__).resolve().parents[2] / "benchmarks" /
               "manual_dpsk_issue01_random_dp32_ep_runner.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "manual_dpsk_issue01_random_dp32_ep_runner",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_case_rows_matches_requested_dpsk_issue01_dp32_ep_run() -> None:
    module = load_module()

    rows = module.build_case_rows()

    assert [(row.dataset, row.request_rate) for row in rows] == [
        ("issue01_random", 20.0),
        ("issue01_random", 50.0),
    ]
    assert {row.model for row in rows} == {"deepseek_v3_1024k"}
    assert {row.strategy for row in rows} == {"dp32"}
    assert {row.cluster for row in rows} == {"4node_h200"}
    assert {row.dispatch_policy for row in rows} == {
        "waiting_x4_plus_running"
    }
    assert {row.max_num_seqs for row in rows} == {196}
    assert {row.max_model_len for row in rows} == {1000000}
    assert {row.gpu_memory_utilization for row in rows} == {0.9}
    assert rows[0].max_requests == 12000
    assert rows[-1].max_requests == 30000


def test_write_case_csv_and_runner_argv(tmp_path: Path) -> None:
    module = load_module()
    csv_path = tmp_path / "cases.csv"

    module.write_case_csv(csv_path, module.build_case_rows())
    records = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    assert len(records) == 2
    assert records[0]["name"] == (
        "DPSK__issue01_random__dp32_ep__rate20__bs196"
    )
    assert records[0]["request_rate"] == "20"
    assert records[0]["max_num_seqs"] == "196"
    assert records[0]["gpu_memory_utilization"] == "0.9"
    assert records[0]["max_requests"] == "12000"
    assert records[-1]["request_rate"] == "50"
    assert records[-1]["max_requests"] == "30000"

    argv = module.build_runner_argv(
        case_csv=csv_path,
        run_label="probe",
        dry_run=True,
        ignore_historical_skips=True,
    )

    assert argv[:2] == [sys.executable, str(module.RUNNER_PATH)]
    assert "--case-csv" in argv
    assert str(csv_path) in argv
    assert "--artifact-root" in argv
    assert str(module.DEFAULT_ARTIFACT_ROOT) in argv
    assert "--run-label" in argv
    assert "probe" in argv
    assert "--dry-run" in argv
    assert "--ignore-historical-skips" in argv
    assert "--frontend-extra-arg=--enable-dbo" not in argv
    assert "--headless-extra-arg=--enable-dbo" not in argv


def test_apply_runner_overrides_sets_model_path_and_master_addr() -> None:
    module = load_module()

    @dataclass(frozen=True)
    class ClusterSpec:
        master_addr: str
        remote_hosts: tuple[str, ...]

    class RunnerModule:
        MODELS = {"deepseek_v3_1024k": "/old/path"}
        CLUSTERS = {
            "4node_h200": ClusterSpec(
                master_addr="10.0.0.1",
                remote_hosts=("h200-rjob0", "h200-rjob1", "h200-rjob2"),
            )
        }

    module.apply_runner_overrides(
        RunnerModule,
        model_path=Path("/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"),
        master_addr="10.102.237.21",
    )

    assert RunnerModule.MODELS["deepseek_v3_1024k"] == (
        "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
    )
    assert RunnerModule.CLUSTERS["4node_h200"].master_addr == "10.102.237.21"
    assert RunnerModule.CLUSTERS["4node_h200"].remote_hosts == (
        "h200-rjob0",
        "h200-rjob1",
        "h200-rjob2",
    )
