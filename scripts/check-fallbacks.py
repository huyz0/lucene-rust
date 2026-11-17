#!/usr/bin/env python3
"""Fail when the plugin gains a Lucene/Java fallback reason nobody approved.

Every place the plugin hands a request back to Lucene or Java names a reason (a string
literal that ends up in the `fallbacks` map of `GET _plugins/accelerated/stats`). This script
extracts those literals from the plugin's Java sources and compares them with
`opensearch-plugin/fallback-reasons.txt`, the list of reasons that still exist.

  * A reason in the source but not in the list: the build fails. Port the path to native
    instead; if it truly cannot be, get it approved and add it to the list with a comment.
  * A reason in the list but gone from the source: the build fails too, so the list only
    shrinks as paths are ported and never carries stale entries.

Usage: check-fallbacks.py [--write]   (--write regenerates the list; review the diff)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "opensearch-plugin/src/main/java"
LIST = ROOT / "opensearch-plugin/fallback-reasons.txt"

# `return "x"`, `reason = "x"`, `fallback("x")`, `Encoded(null, "x")`, `Encoded.fallback("x")`,
# `x.fallback = "x"`, `Prepared(null, 0, "x")`
PATTERNS = [
    re.compile(r'\breturn\s+"([a-z][a-z0-9_]*)"\s*;'),
    re.compile(r'\breturn\s+"([a-z][a-z0-9_]*_)"\s*\+'),
    re.compile(r'[?:]\s*"([a-z][a-z0-9_]*_)"\s*\+'),
    re.compile(r'\breason\s*=\s*"([a-z][a-z0-9_]*)"'),
    re.compile(r'\bfallback\(\s*"([a-z][a-z0-9_]*)"'),
    re.compile(r'Encoded\(\s*null\s*,\s*"([a-z][a-z0-9_]*)"'),
    re.compile(r'Acquired\(\s*0L?\s*,\s*"([a-z][a-z0-9_]*)"'),
    # the highlighters: `prepared.fallback = "x"`, `String undescribed = "x"`, `Prepared(null, 0, "x")`
    re.compile(r'\.fallback\s*=\s*"([a-z][a-z0-9_]*)"'),
    re.compile(r'\bundescribed\s*=\s*"([a-z][a-z0-9_]*)"'),
    re.compile(r'\bPrepared\(\s*null\s*,\s*(?:0L?\s*,\s*)?"([a-z][a-z0-9_]*)"'),
]
# Files whose `return "..."` literals are not fallback reasons (names, formatting, REST).
SKIP = {
    "RestStatsAction.java", "RustSearchPlugin.java", "RustIndexWriter.java",
    "InfoStatsManager.java", "JoinQueries.java", "KnnRestHandlers.java",
    "KnnModelHandlers.java", "DocValuesEncoder.java",
}


def found():
    out = {}
    for p in sorted(SRC.rglob("*.java")):
        if p.name in SKIP:
            continue
        text = p.read_text(encoding="utf-8")
        for pat in PATTERNS:
            for m in pat.finditer(text):
                out.setdefault(m.group(1), set()).add(p.name)
    return out


def listed():
    if not LIST.exists():
        return set()
    return {
        l.split("#")[0].strip()
        for l in LIST.read_text().splitlines()
        if l.split("#")[0].strip()
    }


def main():
    f = found()
    if "--write" in sys.argv:
        LIST.write_text(
            "# Fallback reasons the plugin still has (scripts/check-fallbacks.py).\n"
            "# Shrinks as paths are ported; adding a line needs approval.\n"
            + "".join(f"{r}  # {', '.join(sorted(f[r]))}\n" for r in sorted(f))
        )
        print(f"wrote {len(f)} reasons")
        return 0
    have = listed()
    new = sorted(set(f) - have)
    stale = sorted(have - set(f))
    for r in new:
        print(f"NEW fallback reason {r!r} in {', '.join(sorted(f[r]))}: port the path to native, "
              f"or get approval and add it to {LIST.relative_to(ROOT)}")
    for r in stale:
        print(f"STALE entry {r!r} in {LIST.relative_to(ROOT)}: the fallback is gone, delete the line")
    return 1 if new or stale else 0


sys.exit(main())
