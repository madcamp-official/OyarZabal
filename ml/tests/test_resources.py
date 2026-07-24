from dataclasses import replace

import pytest
from oyarzabal.resources import ResourceSnapshot, assert_safe


def _snapshot() -> ResourceSnapshot:
    gib = 1024**3
    return ResourceSnapshot(
        generated_at="2026-07-24T00:00:00+00:00",
        memory_total_bytes=49 * gib,
        memory_available_bytes=45 * gib,
        disk_free_bytes=62 * gib,
        gpu_memory_total_mib=24 * 1024,
        gpu_memory_used_mib=0,
    )


def test_safe_snapshot_passes() -> None:
    assert_safe(_snapshot())


def test_low_memory_stops_new_work() -> None:
    with pytest.raises(RuntimeError, match="RAM"):
        assert_safe(replace(_snapshot(), memory_available_bytes=8 * 1024**3))
