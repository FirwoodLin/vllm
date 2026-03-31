# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path

from vllm.utils.system_utils import build_process_log_path, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_build_process_log_path():
    path = build_process_log_path("/tmp/engine-core-logs", "EngineCore/DP 1:0", 321)

    assert path == Path("/tmp/engine-core-logs/EngineCore_DP_1_0.pid321.log")
