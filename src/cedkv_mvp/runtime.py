from __future__ import annotations

import platform
import re
import socket
import subprocess
from datetime import datetime, timezone


def _run_git_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()


def get_git_metadata() -> tuple[str, bool]:
    git_sha = _run_git_command(["rev-parse", "--short", "HEAD"]) or "nogit"
    status = _run_git_command(["status", "--porcelain"]) or ""
    git_dirty = bool(status)
    return git_sha, git_dirty


def get_environment_metadata() -> dict[str, str]:
    torch_version = "not_installed"
    cuda_version = "not_installed"
    device_name = "cpu"

    try:
        import torch  # type: ignore

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "none")
        if torch.cuda.is_available():
            device_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass

    return {
        "host": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "device_name": device_name,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id(git_sha: str, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = git_sha or "nogit"
    suffix = re.sub(r"[^0-9A-Za-z_-]", "_", suffix)
    return f"{timestamp}_{suffix}"

