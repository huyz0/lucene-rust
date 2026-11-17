#!/usr/bin/env python3
"""Compares the plugin's forked files with upstream at the pinned tag: the bump worklist.

`opensearch-plugin/upstream.json` names, per fork, the upstream repository, tag and source roots.
A plugin file is a fork when its header carries "Copyright OpenSearch Contributors" (the files
written here carry this repository's own header). For each fork file the script fetches the same
relative path under each root at the tag (raw.githubusercontent.com), normalises both sides (the
licence header, `package` and `import` lines, blank lines, trailing whitespace) and counts the
lines that differ. Output: a Markdown report, forks first by diff size; `--json` prints the
numbers instead. A fork file with no counterpart at the mapped path is listed as unmapped.

  scripts/upstream-diff.py                  # report on stdout
  scripts/upstream-diff.py --fork neural-search
  scripts/upstream-diff.py --json

Needs network access to raw.githubusercontent.com. Not part of CI (hundreds of requests): run it
when bumping the OpenSearch version, after editing the tags in upstream.json.
"""
import argparse
import concurrent.futures
import difflib
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "opensearch-plugin"
SRC = PLUGIN / "src/main/java/org/lucenerust/opensearch"
MARKER = "Copyright OpenSearch Contributors"


def normalise(text: str) -> list[str]:
    text = re.sub(r"\A\s*/\*.*?\*/", "", text, count=1, flags=re.S)
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("package ", "import ")):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out


def fetch(repo: str, tag: str, path: str):
    url = f"https://raw.githubusercontent.com/{repo}/{tag}/{path}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError):
            pass
    raise SystemExit(f"upstream-diff: cannot fetch {url}")


def compare(fork, file: pathlib.Path):
    rel = file.relative_to(SRC / fork["plugin_dir"]).as_posix()
    mine = file.read_text()
    # `overrides` maps a plugin-relative path to its upstream path when the layouts differ.
    candidates = [fork.get("overrides", {}).get(rel)] if rel in fork.get("overrides", {}) else [f"{root}/{rel}" for root in fork["roots"]]
    for upstream in candidates:
        theirs = fetch(fork["repo"], fork["tag"], upstream)
        if theirs is not None:
            a, b = normalise(theirs), normalise(mine)
            changed = sum(1 for d in difflib.ndiff(a, b) if d[0] in "+-")
            return {"file": f"{fork['plugin_dir']}/{rel}", "upstream": upstream, "lines": len(b), "changed": changed}
    return {"file": f"{fork['plugin_dir']}/{rel}", "upstream": None, "lines": len(normalise(mine)), "changed": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((PLUGIN / "upstream.json").read_text())
    report = []
    for fork in manifest["forks"]:
        if args.fork and fork["name"] != args.fork:
            continue
        files = [f for f in sorted((SRC / fork["plugin_dir"]).rglob("*.java")) if MARKER in f.read_text()[:400]]
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            rows = list(pool.map(lambda f: compare(fork, f), files))
        report.append({"fork": fork["name"], "repo": fork["repo"], "tag": fork["tag"], "files": rows})
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        return
    for r in report:
        rows = r["files"]
        mapped = [x for x in rows if x["upstream"]]
        same = sum(1 for x in mapped if x["changed"] == 0)
        print(f"## {r['fork']} ({r['repo']} @ {r['tag']})\n")
        print(f"{len(rows)} fork files: {len(mapped)} with an upstream counterpart ({same} identical after normalising), "
              f"{len(rows) - len(mapped)} unmapped; {sum(x['changed'] for x in mapped)} changed lines over {sum(x['lines'] for x in mapped)}.\n")
        print("| file | lines | changed |\n|---|---|---|")
        for x in sorted(mapped, key=lambda x: -x["changed"]):
            if x["changed"]:
                print(f"| {x['file']} | {x['lines']} | {x['changed']} |")
        if len(rows) != len(mapped):
            print("\nUnmapped (no file at the mapped path upstream): " + ", ".join(x["file"] for x in rows if not x["upstream"]))
        print()


if __name__ == "__main__":
    main()
