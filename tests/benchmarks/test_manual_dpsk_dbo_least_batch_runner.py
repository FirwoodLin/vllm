# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (Path(__file__).resolve().parents[2] / "benchmarks" /
               "manual_dpsk_dbo_least_batch_runner.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "manual_dpsk_dbo_least_batch_runner",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_case_rows_matches_requested_dpsk_dbo_sweep() -> None:
    module = load_module()

    rows = module.build_case_rows()

    assert [(row.dataset, row.request_rate) for row in rows] == [
        ("issue01_random", 10.0),
        ("issue01_random", 20.0),
        ("issue01_random", 30.0),
        ("issue01_random", 40.0),
        ("issue01_random", 50.0),
    ]
    assert {row.model for row in rows} == {"deepseek_v3_1024k"}
    assert {row.strategy for row in rows} == {"dp32"}
    assert {row.dispatch_policy for row in rows} == {"least_batch"}
    assert {row.max_num_seqs for row in rows} == {192}
    assert {row.max_model_len for row in rows} == {1000000}
    assert {row.gpu_memory_utilization for row in rows} == {0.9}
    assert rows[0].max_requests == 6000
    assert rows[-1].max_requests == 30000


def test_write_case_csv_and_runner_argv_enable_dbo(tmp_path: Path) -> None:
    module = load_module()
    csv_path = tmp_path / "cases.csv"

    module.write_case_csv(csv_path, module.build_case_rows())
    records = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    assert len(records) == 5
    assert records[0]["name"] == (
        "DPSK_DBO__issue01_random__dp32_least_batch__rate10__bs192"
    )
    assert records[0]["dispatch_policy"] == "least_batch"
    assert records[0]["max_requests"] == "6000"
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
    assert argv.count("--frontend-extra-arg=--enable-dbo") == 1
    assert argv.count("--frontend-extra-arg=--dbo-decode-token-threshold") == 1
    assert argv.count("--frontend-extra-arg=2") == 1
    assert argv.count("--headless-extra-arg=--enable-dbo") == 1
    assert argv.count("--headless-extra-arg=--dbo-decode-token-threshold") == 1
    assert argv.count("--headless-extra-arg=2") == 1


def test_apply_model_path_override_keeps_dpsk_alias() -> None:
    module = load_module()

    class RunnerModule:
        MODELS = {"deepseek_v3_1024k": "/old/path"}

    module.apply_model_path_override(
        RunnerModule,
        Path("/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"),
    )

    assert RunnerModule.MODELS["deepseek_v3_1024k"] == (
        "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
    )
    assert module.MODEL_PATH == Path(
        "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
    )
