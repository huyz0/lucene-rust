#!/usr/bin/env python3
"""Fresh hosted capacity receipt before allocating this job's loop filesystem."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


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


if __name__ == "__main__":
    main()
