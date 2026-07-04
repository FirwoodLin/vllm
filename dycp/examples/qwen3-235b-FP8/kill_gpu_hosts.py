#!/usr/bin/env python3
"""Run the interactive kill_gpu.sh helper on multiple hosts."""

import argparse
import errno
import os
import pty
import re
import select
import shlex
import subprocess
import sys
import time


DEFAULT_SCRIPT = "/mnt/nvme1n1/ml_research/linbinbin1/scripts/kill_gpu.sh"
DEFAULT_HOSTS = ("dlh200-3", "dlh200-2")
DEFAULT_SUDO_PASSWORD = ""

SUDO_PASSWORD_RE = re.compile(r"(\[sudo\].*password|password for .+:)", re.IGNORECASE)
HOSTKEY_RE = re.compile(r"are you sure you want to continue connecting", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run kill_gpu.sh on remote hosts and answer its interactive prompts."
        )
    )
    parser.add_argument(
        "hosts",
        nargs="*",
        default=list(DEFAULT_HOSTS),
        help="Remote hosts to clean. Default: dlh200-3 dlh200-2.",
    )
    parser.add_argument(
        "--script",
        default=os.environ.get("GPU_CLEANUP_SCRIPT", DEFAULT_SCRIPT),
        help=f"Remote kill script path. Default: {DEFAULT_SCRIPT}.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("GPU_CLEANUP_USER", ""),
        help="SSH user. Defaults to the ssh config/current user.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("GPU_CLEANUP_TIMEOUT", "180")),
        help="Timeout per host in seconds. Default: 180.",
    )
    parser.add_argument(
        "--selection",
        default=os.environ.get("GPU_CLEANUP_SELECTION", "all"),
        help="Selection to send to kill_gpu.sh. Default: all.",
    )
    return parser.parse_args()


def build_ssh_command(host: str, user: str, script: str) -> list[str]:
    target = f"{user}@{host}" if user else host
    remote_cmd = f"bash {shlex.quote(script)}"
    return [
        "ssh",
        "-tt",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        target,
        "bash",
        "-lc",
        shlex.quote(remote_cmd),
    ]


def maybe_answer_prompts(
    master_fd: int,
    output_buffer: str,
    password: str,
    selection: str,
    state: dict[str, bool],
) -> bool:
    lowered = output_buffer.lower()
    if not state["accepted_hostkey"] and HOSTKEY_RE.search(lowered):
        os.write(master_fd, b"yes\r")
        state["accepted_hostkey"] = True
        return True

    if password and SUDO_PASSWORD_RE.search(lowered):
        os.write(master_fd, (password + "\r").encode())
        return True

    if not state["sent_selection"] and (
        "\n> " in output_buffer
        or "\r\n> " in output_buffer
        or output_buffer.endswith("> ")
    ):
        os.write(master_fd, (selection + "\r").encode())
        state["sent_selection"] = True
        return True

    return False


def run_host(host: str, args: argparse.Namespace, password: str) -> int:
    command = build_ssh_command(host, args.user, args.script)
    print(f"\n=== Cleaning GPU processes on {host} ===", flush=True)

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        command,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    state = {"accepted_hostkey": False, "sent_selection": False}
    output_buffer = ""
    deadline = time.monotonic() + args.timeout

    try:
        while True:
            if proc.poll() is not None:
                break
            if time.monotonic() > deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"\nTimed out cleaning {host}.", file=sys.stderr)
                return 124

            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if not ready:
                continue

            try:
                data = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not data:
                break

            text = data.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()

            output_buffer = (output_buffer + text)[-4000:]
            answered = maybe_answer_prompts(
                master_fd,
                output_buffer,
                password,
                args.selection,
                state,
            )
            if answered:
                output_buffer = ""
    finally:
        os.close(master_fd)

    return proc.wait()


def main() -> int:
    args = parse_args()
    password = os.environ.get("GPU_CLEANUP_SUDO_PASSWORD", DEFAULT_SUDO_PASSWORD)
    exit_code = 0

    for host in args.hosts:
        try:
            rc = run_host(host, args, password)
        except Exception as exc:
            rc = 1
            print(f"Cleanup on {host} failed: {exc}", file=sys.stderr)
        if rc != 0:
            exit_code = rc
            print(f"Cleanup on {host} exited with {rc}.", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
