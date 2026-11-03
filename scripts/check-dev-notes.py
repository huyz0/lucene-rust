#!/usr/bin/env python3
"""Fails when a code comment carries development-process notes: milestone or
task ids (`M12`, `T12.4`), sweep batch names, links into docs/milestones or
docs/sweep, PR or task numbers, or parity-ledger references. Comments say what
the code does; history belongs in git and docs/.

Usage: scripts/check-dev-notes.py [--list]
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTS = {".rs", ".java", ".py", ".sh", ".gradle", ".toml", ".yml", ".yaml"}
# Line-comment markers per extension; Java/Rust block comments are matched by
# their leading `*`.
COMMENT = {
    ".rs": re.compile(r"^\s*(//|/\*|\*)"),
    ".java": re.compile(r"^\s*(//|/\*|\*)"),
    ".gradle": re.compile(r"^\s*(//|/\*|\*)"),
}
HASH = re.compile(r"^\s*#(?!!)")
PATTERNS = [
    re.compile(r"\bM(?:1[0-4]|[1-9])(?:\.[a-z]|-e2e)?\b(?![-/]\d| \()"),
    re.compile(r"\bT\d{1,2}\.\d{1,2}[a-z]?\b"),
    re.compile(r"docs/(?:milestones|sweep)\b"),
    re.compile(r"\b(?:task|PR|pull request) #\d+"),
    re.compile(r"\bread path R\d\b"),
    re.compile(r"\bTrack [A-E]\b"),
    re.compile(r"`[cb]\d{1,2}-[a-z][a-z0-9-]+`"),
    re.compile(r"\b(?:sweep|batch) [a-z]\d{1,2}\b"),
    re.compile(r"\bledger (?:row|item)\b|\bTier-2 review\b"),
]
# Sweep batch names (`c25`, `b11`), outside code spans so `{@code b8}` fields
# and `c1=0` style labels pass.
BATCH = re.compile(r"(?<!= )(?<![\w/`{.@\"'-])[cb][1-4]?\d(?:'s)?(?![\w=`}/.-]| =| \()")
CODE_SPAN = re.compile(r"\{@code [^}]*\}|`[^`]*`")
# Files that legitimately talk about the process itself.
EXEMPT = {"scripts/check-dev-notes.py", "scripts/check-port-inventory.py"}


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        p = pathlib.Path(line)
        if p.suffix in EXTS and line not in EXEMPT and not line.startswith(("docs/", "target")):
            yield line


def is_comment(ext, line, in_docstring):
    if ext in COMMENT:
        return bool(COMMENT[ext].match(line))
    if ext in (".rs",):
        return False
    return bool(HASH.match(line)) or in_docstring


def main():
    hits = []
    for rel in tracked():
        ext = pathlib.Path(rel).suffix
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        in_doc = False
        for n, line in enumerate(text.splitlines(), 1):
            if ext == ".py":
                q = line.count('"""')
                starts = in_doc or q > 0
            else:
                starts = False
            if is_comment(ext, line, starts) and (
                any(p.search(line) for p in PATTERNS) or BATCH.search(CODE_SPAN.sub("", line))
            ):
                hits.append(f"{rel}:{n}:{line.strip()}")
            if ext == ".py" and line.count('"""') % 2 == 1:
                in_doc = not in_doc
    if hits:
        print("\n".join(hits))
        if "--list" not in sys.argv:
            print(f"check-dev-notes: {len(hits)} comment line(s) carry development-process notes", file=sys.stderr)
            return 1
    else:
        print("check-dev-notes: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
