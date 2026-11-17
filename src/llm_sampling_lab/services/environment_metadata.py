"""Safe, non-identifying execution environment metadata for benchmarks."""

import platform
from pathlib import Path


def capture_environment_metadata() -> dict[str, str]:
    """Describe the Python runtime, without identifiers or secrets.

    Inside Docker these values describe the container, not the host OS.
    This snapshot intentionally excludes usernames, hostnames, IP addresses,
    environment variables and absolute paths.
    """

    return {
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "execution_context": (
            "container" if Path("/.dockerenv").is_file() else "unknown"
        ),
    }
