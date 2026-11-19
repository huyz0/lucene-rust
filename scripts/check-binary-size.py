#!/usr/bin/env python3
"""Keep compiled and generated binary files out of the repository.

Two rules over the tracked files:

1. No file is larger than LIMIT (8 MiB).
2. A binary file (git's own text/binary detection) larger than SMALL must be
   listed in `scripts/binary-baseline.txt`. The baseline is a ratchet: it only
   shrinks. A new large binary fails until it is generated instead -- by
   `scripts/fetch-data.py` for upstream-derived data, by a script for the rest
   -- and a baseline entry whose file is gone, or is no longer a large binary,
   fails until the line is deleted.

What the baseline holds is the Java-written differential fixtures
(`fixtures/data`, `fixtures/testframework`: Lucene stamps random segment ids
that the tests hardcode, so they cannot be regenerated), the tiny ML models
and the Painless corpus. Tracking something new there is a decision for the
review, not for this script.
"""
import pathlib
import subprocess
import sys

LIMIT = 8 * 1024 * 1024
SMALL = 64 * 1024
BASELINE = pathlib.Path(__file__).with_name("binary-baseline.txt")


def tracked_binaries():
    """(path, size) of every tracked file git considers binary and above SMALL."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--eol", "--cached"], check=True, capture_output=True
    ).stdout.decode()
    for rec in filter(None, out.split("\0")):
        info, path = rec.split("\t", 1)
        p = pathlib.Path(path)
        if not p.is_file():
            continue
        size = p.stat().st_size
        if info.split()[0] == "i/-text" and size > SMALL:
            yield path, size


def main() -> int:
    baseline = {
        line.strip()
        for line in BASELINE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached"], check=True, capture_output=True
    ).stdout.decode()
    bad = []
    for path in filter(None, out.split("\0")):
        p = pathlib.Path(path)
        if p.is_file() and p.stat().st_size > LIMIT:
            bad.append(f"{path}: {p.stat().st_size / 1048576:.1f} MiB exceeds {LIMIT // 1048576} MiB")
    seen = set()
    for path, size in tracked_binaries():
        seen.add(path)
        if path not in baseline:
            bad.append(
                f"{path}: tracked binary of {size // 1024} KiB (limit {SMALL // 1024} KiB). "
                "Generate it instead (scripts/fetch-data.py), do not commit it"
            )
    for path in sorted(baseline - seen):
        bad.append(f"{path}: in scripts/binary-baseline.txt but no longer a large tracked binary; delete the line")
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
