"""Resource snapshots and conservative training safety checks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ResourceSnapshot:
    generated_at: str
    memory_total_bytes: int
    memory_available_bytes: int
    disk_free_bytes: int
    gpu_memory_total_mib: int | None
    gpu_memory_used_mib: int | None


def _memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def _gpu_memory() -> tuple[int | None, int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        ).stdout.splitlines()[0]
        total, used = (int(part.strip()) for part in output.split(","))
        return total, used
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None, None


def snapshot(path: Path = Path("/")) -> ResourceSnapshot:
    memory_total, memory_available = _memory()
    gpu_total, gpu_used = _gpu_memory()
    return ResourceSnapshot(
        generated_at=datetime.now(UTC).isoformat(),
        memory_total_bytes=memory_total,
        memory_available_bytes=memory_available,
        disk_free_bytes=shutil.disk_usage(path).free,
        gpu_memory_total_mib=gpu_total,
        gpu_memory_used_mib=gpu_used,
    )


def assert_safe(
    current: ResourceSnapshot,
    *,
    minimum_memory_available_gib: float = 9,
    minimum_disk_free_gib: float = 15,
    maximum_gpu_used_gib: float = 20,
) -> None:
    gib = 1024**3
    if current.memory_available_bytes < minimum_memory_available_gib * gib:
        raise RuntimeError("available RAM is below the safety threshold")
    if current.disk_free_bytes < minimum_disk_free_gib * gib:
        raise RuntimeError("free disk is below the safety threshold")
    if (
        current.gpu_memory_used_mib is not None
        and current.gpu_memory_used_mib > maximum_gpu_used_gib * 1024
    ):
        raise RuntimeError("GPU memory use is above the safety threshold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    current = snapshot(args.path)
    assert_safe(current)
    rendered = json.dumps(asdict(current), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
