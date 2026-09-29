#!/usr/bin/env python3
"""Every Rust dependency the shipped library links is under a licence this
project may redistribute under Apache-2.0 (docs/licences.md).

The audit M6 asked for was done once, by hand; this keeps it true. A new
dependency -- or an upgrade that changes a licence -- fails here until
docs/licences.md records it, rather than being found at release time.

Only what ships is gated: the workspace's normal and build dependencies and
theirs. Test and benchmark dependencies never reach the plugin's
`liblucene_ffi.so`, and are listed for the record (`--all`).

A licence expression passes when one of its `OR` alternatives is made only
of allow-listed licences (`AND` parts all allowed); the alternative chosen is
printed with `--all`.

Usage: scripts/check-licences.py [--all]
"""
import json
import re
import subprocess
import sys

# Permissive licences compatible with redistribution under Apache-2.0.
ALLOWED = {
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "0BSD",
    "Zlib",
    "Unlicense",
    "Unicode-3.0",
    "Unicode-DFS-2016",
    "ISC",
}


def alternatives(expr):
    """An SPDX expression (or the older `A/B` form) as its OR alternatives,
    each a list of licences that must all hold."""
    expr = expr.replace("/", " OR ")
    # One level of parentheses is all crates.io uses: `(A OR B) AND C`.
    m = re.fullmatch(r"\((.+)\)\s+AND\s+(.+)", expr.strip())
    if m:
        return [[a.strip(), m.group(2).strip()] for a in m.group(1).split(" OR ")]
    return [[p.strip() for p in alt.split(" AND ")] for alt in expr.split(" OR ")]


def chosen(expr):
    for alt in alternatives(expr):
        if all(p in ALLOWED for p in alt):
            return " AND ".join(alt)
    return None


def main():
    show_all = "--all" in sys.argv[1:]
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--offline"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    ws = set(meta["workspace_members"])
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    pkgs = {p["id"]: p for p in meta["packages"]}

    def reachable(from_workspace_kinds):
        seen, stack = set(), list(ws)
        while stack:
            i = stack.pop()
            for d in nodes[i]["deps"]:
                kinds = [k["kind"] for k in d["dep_kinds"]]
                wanted = from_workspace_kinds if i in ws else (None, "build")
                if any(k in wanted for k in kinds) and d["pkg"] not in seen and d["pkg"] not in ws:
                    seen.add(d["pkg"])
                    stack.append(d["pkg"])
        return seen

    shipped = reachable((None, "build"))
    everything = reachable((None, "build", "dev"))
    failures = []
    for i in sorted(everything, key=lambda i: (pkgs[i]["name"], pkgs[i]["version"])):
        p = pkgs[i]
        expr = p.get("license")
        pick = chosen(expr) if expr else None
        ships = i in shipped
        if ships and pick is None:
            failures.append(f"{p['name']} {p['version']}: {expr or 'no licence field'}")
        if show_all:
            where = "ships" if ships else "test/bench"
            print(f"{p['name']:28} {p['version']:14} {where:10} {expr}  ->  {pick}")
    for i in ws:
        if pkgs[i].get("license") != "Apache-2.0":
            failures.append(f"workspace crate {pkgs[i]['name']}: licence {pkgs[i].get('license')}, want Apache-2.0")
    if failures:
        print("check-licences: dependencies the shipped library links without an allowed licence:")
        for f in failures:
            print("  " + f)
        print("Record the audit in docs/licences.md and extend ALLOWED only for a licence")
        print("compatible with redistribution under Apache-2.0.")
        sys.exit(1)
    print(f"check-licences: ok ({len(shipped)} shipped dependencies, {len(everything)} in all)")


if __name__ == "__main__":
    main()
