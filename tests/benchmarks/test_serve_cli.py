# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import http.server
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests
import urllib3

from ..utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def write_random_csv(csv_path: Path, rows: list[tuple[int, int]]) -> None:
    csv_lines = ["prompt_len,output_len"]
    csv_lines.extend(f"{prompt_len},{output_len}" for prompt_len, output_len in rows)
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RecordingCompletionServer:
    def __init__(self, address: str = "127.0.0.1") -> None:
        self.address = address
        self.port = -1
        self.prompt_lens: list[int] = []
        self._lock = threading.Lock()
        self.server: http.server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass

            def do_GET(self):
                if self.path != "/v1/models":
                    self.send_error(404)
                    return

                body = json.dumps(
                    {"data": [{"id": MODEL_NAME, "root": MODEL_NAME}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return

                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                prompt = payload.get("prompt", [])
                prompt_len = len(prompt) if isinstance(prompt, list) else 0
                with outer._lock:
                    outer.prompt_lens.append(prompt_len)

                max_tokens = int(payload.get("max_tokens", 1))
                body = (
                    'data: {"choices":[{"text":"x"}]}\n\n'
                    + "data: "
                    + json.dumps({"usage": {"completion_tokens": max_tokens}})
                    + "\n\n"
                    + "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.port = _find_free_port()
        self.server = http.server.ThreadingHTTPServer(
            (self.address, self.port), Handler
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join()

    @property
    def base_url(self) -> str:
        return f"http://{self.address}:{self.port}"


def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed certificate for testing."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_file, key_file


class RemoteOpenAIServerSSL(RemoteOpenAIServer):
    """RemoteOpenAIServer subclass that supports SSL with self-signed certs."""

    @property
    def url_root(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _wait_for_server(self, *, url: str, timeout: float):
        """Override to use HTTPS with SSL verification disabled."""
        # Suppress InsecureRequestWarning for self-signed certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        start = time.time()
        while True:
            try:
                if requests.get(url, verify=False).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None


@pytest.fixture(scope="function")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager", "--load-format", "dummy"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="function")
def ssl_server():
    """Start a vLLM server with SSL enabled using a self-signed certificate."""
    with tempfile.TemporaryDirectory() as cert_dir:
        cert_file, key_file = generate_self_signed_cert(Path(cert_dir))
        args = [
            "--max-model-len",
            "1024",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--ssl-certfile",
            str(cert_file),
            "--ssl-keyfile",
            str(key_file),
        ]

        with RemoteOpenAIServerSSL(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.benchmark
def test_bench_serve(server):
    # Test default model detection and input/output len
    command = [
        "vllm",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_insecure(ssl_server):
    """Test --insecure flag with an HTTPS server using a self-signed certificate."""
    base_url = f"https://{ssl_server.host}:{ssl_server.port}"
    command = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
        "--insecure",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_chat(server):
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_random_csv(server, tmp_path: Path):
    csv_path = tmp_path / "random_lengths.csv"
    write_random_csv(csv_path, [(16, 4), (24, 5), (20, 3)])

    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--backend",
        "openai",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-csv-path",
        str(csv_path),
        "--num-prompts",
        "3",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_random_csv_chat_backend_fails(server, tmp_path: Path):
    csv_path = tmp_path / "random_lengths.csv"
    write_random_csv(csv_path, [(16, 4), (24, 5)])

    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-csv-path",
        str(csv_path),
        "--num-prompts",
        "2",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode != 0
    assert "random-csv-path" in (result.stdout + result.stderr)


@pytest.mark.benchmark
def test_bench_serve_random_csv_prompt_len_routing(tmp_path: Path):
    csv_path = tmp_path / "random_lengths.csv"
    result_path = tmp_path / "serve-routing-result.json"
    write_random_csv(csv_path, [(99, 2), (100, 2), (101, 2)])

    with RecordingCompletionServer() as short_server, RecordingCompletionServer() as long_server:
        command = [
            "vllm",
            "bench",
            "serve",
            "--model",
            MODEL_NAME,
            "--backend",
            "openai",
            "--dataset-name",
            "random",
            "--random-csv-path",
            str(csv_path),
            "--num-prompts",
            "3",
            "--routing-prompt-len-threshold",
            "100",
            "--routing-base-url-short",
            short_server.base_url,
            "--routing-base-url-long",
            long_server.base_url,
            "--save-result",
            "--save-detailed",
            "--disable-tqdm",
            "--result-filename",
            str(result_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    assert short_server.prompt_lens == [99]
    assert sorted(long_server.prompt_lens) == [100, 101]

    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_result["routing_enabled"] is True
    assert saved_result["routing_threshold_prompt_len"] == 100
    assert saved_result["routing_base_url_short"] == short_server.base_url
    assert saved_result["routing_base_url_long"] == long_server.base_url
    assert saved_result["routing_request_counts"] == {"short": 1, "long": 2}
    assert saved_result["request_routes"] == ["short", "long", "long"]


@pytest.mark.benchmark
def test_bench_serve_no_save_generated_texts(server, tmp_path: Path):
    result_path = tmp_path / "serve-result.json"
    command = [
        "vllm",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "3",
        "--save-result",
        "--save-detailed",
        "--no-save-generated-texts",
        "--result-filename",
        str(result_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert "generated_texts" not in saved_result
    assert "ttfts" in saved_result
    assert "itls" in saved_result
    assert "errors" in saved_result
