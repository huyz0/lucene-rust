#!/usr/bin/env python3
"""Fail when the plugin gains a Lucene/Java fallback reason nobody approved.

Every place the plugin hands a request back to Lucene or Java names a reason, a constant of the
`FallbackReason` enum whose key ends up in the `fallbacks` map of `GET _plugins/accelerated/stats`.
This script compares the enum with `opensearch-plugin/fallback-reasons.txt`, the list of reasons
that still exist, and finds reasons spelled as bare string literals.

  * A bare literal where a reason is returned: the build fails; name it in `FallbackReason`.
  * A constant not in the list: the build fails. Port the path to native instead; if it truly
    cannot be, get it approved and add it to the list with a comment.
  * A constant or list entry no source uses any more: the build fails too, so the list only
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
    "RestStatsAction.java", "AcceleratedSearchPlugin.java", "AcceleratedIndexWriter.java",
    "InfoStatsManager.java", "JoinQueries.java", "KnnRestHandlers.java",
    "KnnModelHandlers.java", "DocValuesEncoder.java",
}


ENUM = SRC / "org/lucenerust/opensearch/FallbackReason.java"


def constants():
    """key -> constant name, from the enum's `NAME("key"),` lines."""
    pairs = re.findall(r'^\s+([A-Z][A-Z0-9_]*)\("([a-z][a-z0-9_]*)"\)', ENUM.read_text(), re.M)
    return {k: n for n, k in pairs}


def found():
    """Reasons the sources use through the enum: key -> files; plus bare literals left over."""
    consts = constants()
    by_name = {n: k for k, n in consts.items()}
    used, literals = {}, {}
    for p in sorted(SRC.rglob("*.java")):
        if p.name in SKIP or p == ENUM:
            continue
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\bFallbackReason\.([A-Z][A-Z0-9_]*)\b", text):
            if m.group(1) in by_name:
                used.setdefault(by_name[m.group(1)], set()).add(p.name)
        for pat in PATTERNS:
            for m in pat.finditer(text):
                literals.setdefault(m.group(1), set()).add(p.name)
    return used, literals, consts


def listed():
    if not LIST.exists():
        return set()
    return {
        l.split("#")[0].strip()
        for l in LIST.read_text().splitlines()
        if l.split("#")[0].strip()
    }


def main():
    f, literals, consts = found()
    if "--write" in sys.argv:
        LIST.write_text(
            "# Fallback reasons the plugin still has (scripts/check-fallbacks.py).\n"
            "# Shrinks as paths are ported; adding a line needs approval.\n"
            + "".join(f"{r}  # {', '.join(sorted(f[r]))}\n" for r in sorted(f))
        )
        print(f"wrote {len(f)} reasons")
        return 0
    have = listed()
    bad = False
    for r in sorted(literals):
        print(f"BARE reason {r!r} in {', '.join(sorted(literals[r]))}: name it in FallbackReason.java")
        bad = True
    for r in sorted(set(consts) - set(f)):
        print(f"UNUSED constant for {r!r} in FallbackReason.java: the fallback is gone, delete it")
        bad = True
    for r in sorted(set(f) - have):
        print(f"NEW fallback reason {r!r} in {', '.join(sorted(f[r]))}: port the path to native, "
              f"or get approval and add it to {LIST.relative_to(ROOT)}")
        bad = True
    for r in sorted(have - set(f)):
        print(f"STALE entry {r!r} in {LIST.relative_to(ROOT)}: the fallback is gone, delete the line")
        bad = True
    return 1 if bad else 0


sys.exit(main())
