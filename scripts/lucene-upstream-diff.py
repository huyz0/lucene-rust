#!/usr/bin/env python3
"""List the upstream Apache Lucene commits that landed after the port's anchor.

The anchor lives in docs/lucene-port-anchor.json: the released Lucene tag the
port is pinned to, plus the PR numbers already ported or deliberately skipped
from the releases after it. This script walks `git log <anchor tag>..<target>`
in a Lucene checkout and prints every commit grouped by area and by relevance,
marking each with its status (ported / skipped / in the pin / referenced in the
repo / pending). Standard library only.

  scripts/lucene-upstream-diff.py --lucene ~/work/lucene
  scripts/lucene-upstream-diff.py --lucene ~/work/lucene --target releases/lucene/10.6.0
  scripts/lucene-upstream-diff.py --lucene ~/work/lucene --pending --format json
  scripts/lucene-upstream-diff.py --lucene ~/work/lucene --summary

A blobless clone is enough (`git clone --filter=blob:none --no-checkout
https://github.com/apache/lucene`): only commit and tree data is read.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "docs", "lucene-port-anchor.json")

# (area, path regex). First match wins for a file; a commit takes the area with
# the most files, ties broken by the order here.
AREAS = [
    ("vectors", r"(codecs/(hnsw|lucene\d+/.*(Vector|Hnsw|Quantiz|Flat))|util/hnsw|util/quantization|util/VectorUtil|util/VectorScorer|search/.*Knn|search/.*Vector|internal/vectorization|Vector|Hnsw|Quantiz|vectorization)"),
    ("codecs-formats", r"(/codecs/|backward-codecs|/store/|SegmentInfos|index/.*(Writer|Merge|Segment|Format))"),
    ("analysis", r"(/analysis/|lucene/analysis)"),
    ("util-simd", r"(/util/|/internal/|Panama|vectorization)"),
    ("index", r"/index/"),
    ("search-query", r"(/search/|lucene/(queries|sandbox|join|grouping|facet|highlighter|queryparser|suggest|spatial|expressions|misc|memory|monitor|classification))"),
]
BUILD = re.compile(r"^(gradle|build-tools|dev-tools|\.github|help/|lucene/(distribution|documentation|benchmark|luke|demo)|CHANGES|lucene/CHANGES|build\.gradle|settings\.gradle|versions|.*\.(md|txt|toml|lock|properties))")
TESTONLY = re.compile(r"(/src/test/|/test-framework/|/src/test-files/|lucene/test-framework)")
FIX = re.compile(r"\b(fix|bug|correct|incorrect|wrong|NPE|regression|corrupt|leak|race|overflow|crash|don't|doesn't|ensure|avoid|prevent|throw|validate|clamp|guard)\b", re.I)
PERF = re.compile(r"\b(optimi[sz]|speed ?up|faster|perf|vectoriz|avoid (redundant|unnecessary|copy)|reduce|skip|cache|faster|improve|intoBitSet|bulk|SIMD|batch)\w*", re.I)
FORMAT = re.compile(r"\b(format|codec|version_current|\.dvp|segments_N|file format|new field)\b", re.I)
PR = re.compile(r"#(\d{4,6})")
GH = re.compile(r"(?:GITHUB#|GH-|LUCENE-|#)(\d{3,6})")


def git(lucene, *args):
    return subprocess.run(["git", "-C", lucene, *args], check=True, capture_output=True, text=True).stdout


def load_anchor():
    with open(ANCHOR) as f:
        return json.load(f)


def repo_refs():
    """PR numbers the repo mentions (docs, code, tests, fixtures): a hint, not proof."""
    out = subprocess.run(
        ["git", "-C", ROOT, "grep", "-hoE", r"(GITHUB#|GH-|#)1[3-9][0-9]{3}\b", "--", "docs", "crates", "opensearch-plugin", "fixtures", "scripts", "AGENTS.md"],
        capture_output=True, text=True).stdout
    return {int(m) for m in re.findall(r"1[3-9]\d{3}", out)}


def classify(subject, files):
    non_build = [f for f in files if not BUILD.match(f)]
    if not files or not non_build:
        return "build-docs", "build/docs"
    if all(TESTONLY.search(f) for f in non_build):
        return "test-only", "test"
    src = [f for f in non_build if not TESTONLY.search(f)]
    votes = {}
    for f in src:
        for area, rx in AREAS:
            if re.search(rx, f):
                votes[area] = votes.get(area, 0) + 1
                break
    area = max(votes, key=lambda a: (votes[a], -[x for x, _ in AREAS].index(a))) if votes else "other"
    if FORMAT.search(subject) and area in ("codecs-formats", "index", "vectors"):
        kind = "format?"
    elif FIX.search(subject):
        kind = "bugfix"
    elif PERF.search(subject):
        kind = "perf"
    else:
        kind = "feature/refactor"
    return area, kind


def commits(lucene, rng):
    raw = git(lucene, "log", "--no-merges", "--name-only", "--date=short",
              "--format=%x01%H%x02%ad%x02%s", rng)
    out = []
    for chunk in raw.split("\x01"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        head, _, rest = chunk.partition("\n")
        sha, date, subject = head.split("\x02", 2)
        files = [x for x in rest.split("\n") if x]
        prs = [int(x) for x in PR.findall(subject)] or [int(x) for x in GH.findall(subject)]
        area, kind = classify(subject, files)
        out.append(dict(sha=sha[:10], date=date, subject=subject, prs=prs, area=area, kind=kind, files=len(files)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lucene", default=os.environ.get("LUCENE_CHECKOUT"), help="path to a Lucene git checkout")
    ap.add_argument("--target", default="origin/branch_10_6", help="upstream ref to compare against (default origin/branch_10_6; use origin/main for 11.0)")
    ap.add_argument("--pending", action="store_true", help="only commits not ported, skipped or in the pin")
    ap.add_argument("--line", choices=["10.6", "11.0"], help="with --target origin/main: only changes that are in 10.6.0 or only the 11.0 line")
    ap.add_argument("--summary", action="store_true", help="counts per area/status only")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--include-test-build", action="store_true", help="also list test-only and build/docs commits")
    a = ap.parse_args()
    if not a.lucene:
        sys.exit("pass --lucene <checkout> or set LUCENE_CHECKOUT")

    anchor = load_anchor()
    base = anchor["base"]["tag"]
    # Every PR number in the pinned release's history: a main commit whose PR is
    # here was backported to 10.x already, so it is in the pin under another sha.
    in_pin = {int(n) for line in git(a.lucene, "log", "--no-merges", "--format=%s", base).split("\n") for n in PR.findall(line)}
    ported = {int(k) for k in anchor["ported_prs"]}
    skipped = {int(k): v for k, v in anchor["skipped_prs"].items()}
    refs = repo_refs()
    part = anchor.get("partial", {}).get("tag")
    part_prs = {int(n) for line in git(a.lucene, "log", "--no-merges", "--format=%s", f"{base}..{part}").split("\n") for n in PR.findall(line)} if part else set()

    rows = commits(a.lucene, f"{base}..{a.target}")
    for r in rows:
        st = "pending"
        if any(p in ported for p in r["prs"]):
            st = "ported"
        elif any(p in skipped for p in r["prs"]):
            st = "skipped"
        elif any(p in in_pin for p in r["prs"]):
            st = "in-pin"
        elif any(p in refs for p in r["prs"]):
            st = "referenced"
        r["status"] = st
        r["line"] = "10.6" if (any(p in part_prs for p in r["prs"]) or a.target.endswith("branch_10_6")) else "11.0"
        r["note"] = next((skipped[p] for p in r["prs"] if p in skipped), "")
    if a.line:
        rows = [r for r in rows if r["line"] == a.line]
    inert = {"build-docs", "test-only"}
    if not a.include_test_build:
        hidden = sum(1 for r in rows if r["area"] in inert)
    else:
        hidden = 0
    shown = [r for r in rows if a.include_test_build or r["area"] not in inert]
    if a.pending:
        shown = [r for r in shown if r["status"] in ("pending", "referenced")]

    if a.format == "json":
        json.dump(dict(base=base, target=a.target, total=len(rows), commits=shown), sys.stdout, indent=1)
        print()
        return

    print(f"# Lucene upstream since {base} (anchor) -> {a.target}\n")
    print(f"{len(rows)} commits ({hidden} test-only/build/docs hidden). Anchor: {anchor['base']['release']} ({anchor['base']['commit']}).\n")
    by = {}
    for r in rows:
        by.setdefault(r["area"], {}).setdefault(r["status"], 0)
        by[r["area"]][r["status"]] += 1
    print("| area | total | ported | skipped | in-pin | referenced | pending |\n|---|---|---|---|---|---|---|")
    for area in sorted(by, key=lambda x: -sum(by[x].values())):
        c = by[area]
        print(f"| {area} | {sum(c.values())} | " + " | ".join(str(c.get(s, 0)) for s in ("ported", "skipped", "in-pin", "referenced", "pending")) + " |")
    if a.summary:
        return
    order = ["vectors", "codecs-formats", "search-query", "index", "analysis", "util-simd", "other"]
    kinds = ["bugfix", "format?", "perf", "feature/refactor"]
    for area in order + sorted(set(by) - set(order) - inert):
        group = [r for r in shown if r["area"] == area]
        if not group:
            continue
        print(f"\n## {area}\n")
        for kind in kinds:
            sub = [r for r in group if r["kind"] == kind]
            if not sub:
                continue
            print(f"**{kind}**\n")
            for r in sub:
                pr = ",".join(f"#{p}" for p in r["prs"]) or "-"
                note = f" ({r['note']})" if r["note"] else ""
                print(f"- `{r['sha']}` {r['date']} {pr} [{r['status']}] {r['subject']}{note}" + ("" if a.target.endswith("branch_10_6") else f" <{r['line']}>"))
            print()


if __name__ == "__main__":
    main()
