#!/usr/bin/env python3
"""The slowest tests of a nextest run, as a Markdown table.

    python3 scripts/ci-slowest-tests.py JUNIT_XML [N]

Reads the JUnit report `cargo nextest run --profile ci` writes
(.config/nextest.toml) and prints the N (default 25) slowest test cases, so a
CI job summary shows where the suite's time goes and a test that has crept up
is visible before it nears the 10-minute per-test limit. CI appends this to
$GITHUB_STEP_SUMMARY and never fails on it.
"""

import sys
import xml.etree.ElementTree as ET


def main(argv):
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    n = int(argv[1]) if len(argv) > 1 else 25
    root = ET.parse(argv[0]).getroot()
    cases = []
    total = 0.0
    for case in root.iter("testcase"):
        t = float(case.get("time", "0") or 0)
        total += t
        cases.append((t, case.get("classname", ""), case.get("name", "")))
    cases.sort(reverse=True)
    print(f"### Slowest {min(n, len(cases))} of {len(cases)} tests ({total:.0f}s of test time in all)")
    print()
    print("| seconds | binary | test |")
    print("|--:|---|---|")
    for t, cls, name in cases[:n]:
        print(f"| {t:.1f} | `{cls}` | `{name}` |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
