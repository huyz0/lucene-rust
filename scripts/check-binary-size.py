#!/usr/bin/env python3
"""Fail when a tracked file is larger than LIMIT bytes.

Large build outputs (plugin zips, libraries, jars) must not be committed: every
clone downloads them forever. Generate them with the build scripts instead.
"""
import subprocess
import sys

LIMIT = 8 * 1024 * 1024


def main() -> int:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached"], check=True, capture_output=True
    ).stdout.decode()
    bad = []
    for path in filter(None, out.split("\0")):
        try:
            size = subprocess.run(
                ["git", "cat-file", "-s", f":{path}"], check=True, capture_output=True
            ).stdout
        except subprocess.CalledProcessError:
            continue
        if int(size) > LIMIT:
            bad.append((int(size), path))
    for size, path in sorted(bad, reverse=True):
        print(f"{path}: {size / 1048576:.1f} MiB exceeds {LIMIT // 1048576} MiB", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
