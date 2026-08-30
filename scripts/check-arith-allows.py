#!/usr/bin/env python3
"""Enforce the arithmetic gate's `#[allow]` convention.

`clippy::arithmetic_side_effects` is denied crate-wide in the crates that
decode bytes off disk (see `docs/arithmetic-gate.md`). Clippy enforces the
deny; nothing enforces the *rule for switching it off*, which is where the
value is -- an `#[allow]` with no stated reason is how a gate stops meaning
anything.

Every `#[allow(clippy::arithmetic_side_effects)]` under `crates/*/src/` must be
one of:

  1. a burn-down marker -- the line itself ends in `// TODO(arith-audit)`;
  2. justified -- the contiguous comment block immediately above it contains
     `ARITH:`, naming the invariant that makes the arithmetic safe;
  3. a test-module opt-out -- an inner `#![allow(...)]` inside a `#[cfg(test)]`
     module (or a `tests/`/`benches/`/`examples/` file, which this script does
     not scan at all).

The script also checks that `docs/arithmetic-gate.md`'s burn-down counts match
the markers actually present, since those are hand-maintained.

Exits non-zero with a per-site report on failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = "#[allow(clippy::arithmetic_side_effects)]"
INNER_ALLOW = "#![allow(clippy::arithmetic_side_effects)]"
MARKER = "// TODO(arith-audit)"


def check_file(path: Path, errors: list[str]) -> int:
    """Returns the number of TODO(arith-audit) markers in `path`."""
    lines = path.read_text().splitlines()
    markers = 0
    in_test_module = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#\[cfg\(test\)\]", stripped):
            in_test_module = True
        if stripped.startswith(INNER_ALLOW):
            if not in_test_module and i > 20:
                errors.append(
                    f"{path}:{i + 1}: module-scope `#![allow]` outside a "
                    f"`#[cfg(test)]` module. Narrow it, or move the opt-out to "
                    f"the module's declaration in lib.rs with a "
                    f"`{MARKER}` marker."
                )
            continue
        if not stripped.startswith(ALLOW):
            continue
        if stripped.endswith(MARKER):
            markers += 1
            continue
        if in_test_module:
            continue
        # Rule 2: walk back over the contiguous comment block.
        justified = False
        j = i - 1
        while j >= 0 and lines[j].strip().startswith("//"):
            if "ARITH:" in lines[j]:
                justified = True
                break
            j -= 1
        if not justified:
            errors.append(
                f"{path}:{i + 1}: `{ALLOW}` with no `// ARITH:` proof in the "
                f"comment block above it. State the invariant that makes the "
                f"arithmetic safe, or mark the module `{MARKER}` in lib.rs. "
                f"See docs/arithmetic-gate.md."
            )
    return markers


def main() -> int:
    errors: list[str] = []
    per_crate: dict[str, int] = {}
    for crate_dir in sorted((ROOT / "crates").glob("*/src")):
        per_crate[crate_dir.parent.name] = 0
    for path in sorted(ROOT.glob("crates/*/src/**/*.rs")):
        crate = path.relative_to(ROOT).parts[1]
        per_crate[crate] = per_crate.get(crate, 0) + check_file(path, errors)

    doc = (ROOT / "docs" / "arithmetic-gate.md").read_text()
    # Cross-check *every* crate that has its own burn-down row, including the
    # ones the tree says have zero markers.
    #
    # This used to skip `count == 0`, which made the check blind in the
    # direction that actually matters: deleting a module's marker without
    # auditing it turns the crate's gate red and silences this script at the
    # same time. c31 found `lucene-codecs` in exactly that state (197 live
    # errors, table still claiming 5 pending modules), and then found
    # `lucene-index` in it too.
    for crate, count in sorted(per_crate.items()):
        row = re.search(
            r"^\|\s*`" + re.escape(crate) + r"`\s*\|[^|]*\|\s*([^|]*?)\s*\|",
            doc,
            re.M,
        )
        if row is None:
            if count:
                errors.append(
                    f"docs/arithmetic-gate.md: no burn-down row for `{crate}`, "
                    f"which has {count} `{MARKER}` module(s)."
                )
            continue
        cell = row.group(1)
        match = re.match(r"(\d+)\b", cell)
        expected = int(match.group(1)) if match else (0 if "none" in cell else None)
        if expected is None:
            continue  # a row that names no count, e.g. the "off" crates
        if expected != count:
            errors.append(
                f"docs/arithmetic-gate.md: the burn-down table says "
                f"{expected} unaudited module(s) in `{crate}`, but the tree "
                f"has {count}. A marker deleted without finishing the audit "
                f"leaves the crate's clippy gate red and this check quiet."
            )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\ncheck-arith-allows: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    total = sum(per_crate.values())
    print(f"check-arith-allows: ok ({total} module(s) still unaudited)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
