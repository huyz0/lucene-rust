#!/usr/bin/env python3
"""The port inventory: every class of a Lucene 10.5.0 jar, and what became of it.

`docs/parity.md` records what someone chose to record; nothing recorded what
nobody did. This gate closes that. `docs/inventory/<module>.tsv` lists every
top-level class of the module's jar, one per line:

    <class>\t<status>\t<detail>

`<class>` is the path under `org/apache/lucene/` without `.class`
(`index/IndexWriter`). `<status>` is one of:

  ported              detail = `crates/<crate>/src/<file>.rs` and optionally
                      `::<symbol>`; the file must exist and, when a symbol is
                      named, contain it as a word.
  partial             detail = the same location, then ` -- M<n>: ` and what
                      is missing: the milestone that closes the gap, stated.
  not-needed          detail = why: Java infrastructure with no Rust
                      counterpart to write (SPI loading, reflection, JVM
                      intrinsics), or a class whose behaviour lives in a
                      different Rust shape -- which must then be `ported`
                      pointing at that shape, not `not-needed`.
  todo:M<n>           not ported yet; milestone M<n> takes it. detail = what it is.
  deferred:M<n>       the same, for a class a later milestone takes on purpose.

What the gate checks, always:

  - every top-level class of the jar appears exactly once, and nothing else;
  - every status is one of the above, with the detail it needs;
  - every `ported`/`partial` location exists (and names its symbol).

`--milestone M7` additionally fails while any class is `todo:M7` or a
`partial` whose gap is tagged `M7` -- the milestone's own "done". `--summary` prints the
counts per status and package.

The class list comes from the compiled jar (`scripts/lib-lucene-jars.sh`: the
container has it baked in, CI downloads it), top-level classes only (no `$`),
including the Java 21 multi-release variants. If no jar can be found the
jar-membership check is skipped and said so; the rest still runs.

Usage:
  scripts/check-port-inventory.py [--module core|backward-codecs] [--milestone M7] [--summary]
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = re.compile(r"^(ported|partial|not-needed|todo:M\d+|deferred:M\d+)$")
PARTIAL_GAP = re.compile(r" -- (M\d+): \S")
LOCATION = re.compile(r"^(crates/[\w-]+/src/[\w/]+\.rs)(?:::([\w:]+))?")
MODULES = {"core": "lucene-core", "backward-codecs": "lucene-backward-codecs"}


def jar_classes(module: str) -> set[str] | None:
    """Top-level classes of the module's compiled jar, or None if unavailable."""
    jar_env = os.environ.get("LUCENE_INVENTORY_JAR")
    if jar_env:
        path = jar_env
    else:
        script = (
            f'JARS="{ROOT}/fixtures/.jars"; '
            f'source "{ROOT}/scripts/lib-lucene-jars.sh"; '
            f"lucene_resolve_jar {MODULES[module]}"
        )
        try:
            path = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True, check=True
            ).stdout.strip()
        except subprocess.CalledProcessError:
            return None
    if not path or not Path(path).is_file():
        return None
    out = set()
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            # Multi-release variants (`META-INF/versions/<n>/...`) are real
            # classes of the jar too: the Panama implementations live there.
            m = re.match(r"^(?:META-INF/versions/\d+/)?org/apache/lucene/(.+)\.class$", name)
            if not m or "$" in m.group(1):
                continue
            cls = m.group(1)
            if cls.endswith(("module-info", "package-info")):
                continue
            out.add(cls)
    return out


def symbol_in(path: Path, symbol: str) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    last = symbol.split("::")[-1]
    return re.search(rf"\b{re.escape(last)}\b", text) is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="core", choices=sorted(MODULES))
    ap.add_argument("--milestone", help="also fail while this milestone's work is open")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    tsv = ROOT / "docs" / "inventory" / f"lucene-{args.module}.tsv"
    errors: list[str] = []
    rows: dict[str, tuple[str, str]] = {}
    for n, line in enumerate(tsv.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            errors.append(f"{tsv.name}:{n}: expected 3 tab-separated columns")
            continue
        cls, status, detail = (p.strip() for p in parts)
        if cls in rows:
            errors.append(f"{tsv.name}:{n}: {cls} listed twice")
        rows[cls] = (status, detail)
        if not STATUS.match(status):
            errors.append(f"{tsv.name}:{n}: {cls}: unknown status {status!r}")
            continue
        if not detail:
            errors.append(f"{tsv.name}:{n}: {cls}: {status} needs a detail")
            continue
        if status in ("ported", "partial"):
            m = LOCATION.match(detail)
            if not m:
                errors.append(f"{tsv.name}:{n}: {cls}: {status} must name crates/<crate>/src/<file>.rs")
                continue
            path = ROOT / m.group(1)
            if not path.is_file():
                errors.append(f"{tsv.name}:{n}: {cls}: {m.group(1)} does not exist")
            elif m.group(2) and not symbol_in(path, m.group(2)):
                errors.append(f"{tsv.name}:{n}: {cls}: {m.group(1)} has no `{m.group(2)}`")
            if status == "partial" and not PARTIAL_GAP.search(detail):
                errors.append(
                    f"{tsv.name}:{n}: {cls}: partial must end ' -- M<n>: <what is missing>'"
                )

    classes = jar_classes(args.module)
    if classes is None:
        print(f"check-port-inventory: no {MODULES[args.module]} jar; jar membership not checked",
              file=sys.stderr)
    else:
        for cls in sorted(classes - rows.keys()):
            errors.append(f"{cls}: in the jar, not in {tsv.name}")
        for cls in sorted(rows.keys() - classes):
            errors.append(f"{cls}: in {tsv.name}, not in the jar")

    if args.milestone:
        m = args.milestone
        for cls, (status, detail) in sorted(rows.items()):
            gap = PARTIAL_GAP.search(detail) if status == "partial" else None
            if status == f"todo:{m}" or (gap and gap.group(1) == m):
                errors.append(f"{cls}: {m} is not done: {status} {detail}")

    if args.summary:
        by_status = collections.Counter(s for s, _ in rows.values())
        print(f"{tsv.name}: {len(rows)} classes")
        for s, c in sorted(by_status.items()):
            print(f"  {s:<14} {c}")
        by_pkg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for cls, (s, _) in rows.items():
            by_pkg[cls.rsplit("/", 1)[0] if "/" in cls else "."][s] += 1
        for pkg in sorted(by_pkg):
            c = by_pkg[pkg]
            print(f"  {pkg:<34} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))

    if errors:
        print(f"check-port-inventory: {len(errors)} problem(s)", file=sys.stderr)
        for e in errors[:200]:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"check-port-inventory: {tsv.name} ok ({len(rows)} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
