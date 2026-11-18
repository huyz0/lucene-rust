#!/usr/bin/env python3
"""Every setting this plugin adds is named in the docs.

The plugin registers settings of three kinds: stock ones it replaces (k-NN, neural search,
ml-commons: the forks under knn/, ml/, neural/ and sparse/, documented upstream and checked
against upstream by scripts/upstream-diff.py) and ones of its own. A setting of its own that no
page under docs/ or README names is an undocumented switch; this fails on it.

Names are found by reading the first argument of each `Setting.<kind>Setting(...)` and
`new Setting<>(...)` in the Java sources, resolving `static final String` constants.

    scripts/check-settings-docs.py
"""
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "opensearch-plugin", "src", "main", "java")
FORKS = ("/knn/", "/ml/", "/neural/", "/sparse/")

CONST = re.compile(r"static\s+final\s+String\s+(\w+)\s*=\s*([^;]+);")
START = re.compile(r"(?:Setting\s*\.\s*(\w+Setting)\s*\(|new\s+Setting<[^>]*>\s*\()")


def first_arg(text, i):
    depth, quote, cur = 1, False, ""
    while i < len(text):
        c = text[i]
        if c == '"' and text[i - 1] != "\\":
            quote = not quote
        if not quote:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    return cur
            elif c == "," and depth == 1:
                return cur
        cur += c
        i += 1
    return cur


def evaluate(expr, consts):
    out = ""
    for part in re.split(r"\+", expr.strip()):
        part = part.strip()
        m = re.fullmatch(r'"((?:[^"\\]|\\.)*)"', part)
        if m:
            out += m.group(1)
        elif part.split(".")[-1] in consts:
            out += consts[part.split(".")[-1]]
        else:
            return None
    return out


def main():
    files = [os.path.join(r, f) for r, _, fs in os.walk(SRC) for f in fs if f.endswith(".java")]
    texts = {f: open(f, encoding="utf-8").read() for f in files}
    local = {f: {} for f in texts}
    consts = {}
    for _ in range(3):
        for f, text in texts.items():
            for m in CONST.finditer(text):
                v = evaluate(m.group(2), {**consts, **local[f]})
                if v is not None:
                    local[f][m.group(1)] = v
                    consts.setdefault(m.group(1), v)
    docs = ""
    for pattern in ("docs/**/*.md", "*.md", "opensearch-plugin/*.md"):
        for f in glob.glob(os.path.join(ROOT, pattern), recursive=True):
            docs += open(f, encoding="utf-8").read()
    missing = []
    seen = 0
    for f, text in texts.items():
        if any(k in f.replace(os.sep, "/") for k in FORKS):
            continue
        for m in START.finditer(text):
            if m.group(1) in ("Property",):
                continue
            key = evaluate(first_arg(text, m.end()), {**consts, **local[f]})
            if not key or "." not in key:
                continue
            seen += 1
            if key not in docs:
                missing.append((key, os.path.relpath(f, SRC)))
    for key, f in sorted(set(missing)):
        print(f"undocumented setting: {key} ({f})")
    print(f"check-settings-docs: {seen} settings of ours, {len(set(missing))} undocumented")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
