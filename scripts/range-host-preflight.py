#!/usr/bin/env python3
"""Fresh hosted capacity receipt before allocating this job's loop filesystem."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def memory_capacity():
    """Physical availability intersected with every enclosing v2 cgroup."""
    with Path("/proc/meminfo").open() as source:
        text = source.read(65537)
    if len(text) > 65536:
        raise ValueError("host memory receipt bound")
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].split() for line in text.splitlines()}
    values = {}
    for name in ("MemTotal", "MemAvailable"):
        number, unit = fields[name]
        if not number.isdecimal() or unit != "kB":
            raise ValueError("invalid physical host memory receipt")
        values[name] = int(number) * 1024
    with Path("/proc/self/cgroup").open() as source:
        text = source.read(4097)
    unified = [line[3:] for line in text.splitlines() if line.startswith("0::/")]
    if len(text) > 4096 or len(unified) != 1:
        raise ValueError("missing bounded host unified cgroup receipt")
    root = Path("/sys/fs/cgroup").resolve()
    group = (root / unified[0].lstrip("/")).resolve()
    if not group.is_relative_to(root):
        raise ValueError("host cgroup escaped root")
    available, ancestors = values["MemAvailable"], []
    while True:
        if len(ancestors) >= 64:
            raise ValueError("host cgroup ancestry bound")
        maximum = group / "memory.max"
        if maximum.is_file():
            with maximum.open() as source:
                maximum_text = source.read(129).strip()
            with (group / "memory.current").open() as source:
                current_text = source.read(129).strip()
            if not current_text.isdecimal() or len(current_text) > 128:
                raise ValueError("invalid host cgroup usage")
            current = int(current_text)
            if maximum_text != "max" and (not maximum_text.isdecimal() or len(maximum_text) > 128):
                raise ValueError("invalid host cgroup limit")
            limit = None if maximum_text == "max" else int(maximum_text)
            if limit is not None:
                available = min(available, max(0, limit - current))
            ancestors.append({"path": str(group), "maximum_bytes": limit, "current_bytes": current})
        elif group != root:
            raise ValueError("host cgroup lacks memory accounting")
        if group == root:
            break
        group = group.parent
    return {"physical_total_bytes": values["MemTotal"], "physical_available_bytes": values["MemAvailable"],
            "effective_available_bytes": available, "cgroup_ancestors": ancestors}


def main():
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "linux" or len(sys.argv) != 2:
        raise SystemExit("requires the fixed GitHub Linux workflow")
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if Path(sys.argv[1]).resolve(strict=True) != runner_temp:
        raise SystemExit("wrong owned runner temporary root")
    constants = runpy.run_path("scripts/range-feedback.py")
    image, limit = constants["IMAGE"], constants["SCRATCH_BYTES"]
    # Image provisioning is control-plane work, outside compiler cgroup claims.
    subprocess.run(["docker", "pull", image], check=True, timeout=300)
    volume = os.statvfs(runner_temp)
    available = volume.f_bavail * volume.f_frsize
    reserve = 4 * 1024**3
    print(json.dumps({"phase": "fresh-host-storage-preflight", "image": image,
                      "available_bytes_after_image_pull": available,
                      "declared_loop_bytes": limit, "host_reserve_bytes": reserve}), flush=True)
    if available < limit + reserve:
        raise SystemExit("fresh hosted disk cannot reserve the declared scratch plus host margin")
    profile = os.environ.get("RANGE_PROFILE", "component")
    if profile not in constants["PROFILES"]:
        raise SystemExit("unknown fixed memory-admission profile")
    memory = (8 if profile in constants["HEAVY_PROFILES"] else 4) * 1024**3
    memory_reserve = 2 * 1024**3
    capacity = memory_capacity()
    print(json.dumps({"phase": "fresh-host-memory-preflight", "profile": profile,
                      "declared_guest_memory_bytes": memory, "host_reserve_bytes": memory_reserve,
                      **capacity}), flush=True)
    if capacity["effective_available_bytes"] < memory + memory_reserve:
        raise SystemExit("fresh hosted memory cannot admit the declared guest plus host reserve")


if __name__ == "__main__":
    main()
