#!/usr/bin/env python3
"""Keep the OpenSearch plugin's Java files and methods from growing past a review-sized limit.

The Java twin of check-file-size.py. For every `opensearch-plugin/src/main/java/**.java` file it
counts code lines (no blank or comment-only lines) and finds each method's length (its header
through its closing brace). A file over FILE_LIMIT code lines, or a method over METHOD_LIMIT lines,
must be listed in `opensearch-plugin/java-size-baseline.json` with the size it had when listed; it
may not grow past that size. When a listed file or method shrinks, lower its entry (the check says
so), and when it drops under the limit, delete the entry. A new file or method over the limit
fails: split it before adding it.

  check-java-size.py            check against the baseline
  check-java-size.py --write    rewrite the baseline from the sources (review the diff: it may only shrink)
  check-java-size.py --list     print every file and method over its limit
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "opensearch-plugin/src/main/java"
BASELINE = ROOT / "opensearch-plugin/java-size-baseline.json"
FILE_LIMIT = 1000
METHOD_LIMIT = 100

NOT_METHODS = {"if", "for", "while", "switch", "try", "catch", "else", "do", "synchronized", "new", "return", "throw", "assert"}
ANNOTATION = re.compile(r"@\w+(?:\.\w+)*(?:\s*\([^)]*\))?")
TYPE_DECL = re.compile(r"\b(?:class|record|enum|interface)\s")
HEADER = re.compile(r"(\w+)\s*\([^;{}]*\)\s*(?:throws\s+[\w.,\s<>]+)?$", re.S)


def scan(text):
    """Yield (code_lines, methods) for a Java source: methods are (name, first_line, last_line)."""
    code_lines = set()
    methods = []
    i, n, line = 0, len(text), 1
    depth = 0
    seg_start_line = 1
    seg = []          # characters of the statement since the last ; { }
    stack = []        # per open brace: ("method", name, start_line) or None
    in_block = False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            if c == "\n":
                line += 1
            i += 1
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r":
            if seg:
                seg.append(" ")
            i += 1
            continue
        code_lines.add(line)
        if c == '"' and text[i:i + 3] == '"""':
            j = text.index('"""', i + 3) + 3
            line += text.count("\n", i, j)
            code_lines.update(range(line - text.count("\n", i, j), line + 1))
            seg.append("S")
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            seg.append("S")
            i = j + 1
            continue
        if not seg:
            seg_start_line = line
        if c == "{":
            header = ANNOTATION.sub("", "".join(seg)).strip()
            m = None if TYPE_DECL.search(header) else HEADER.search(header)
            kind = None
            if m and m.group(1) not in NOT_METHODS and "->" not in header and "=" not in header.split("(")[0]:
                kind = ("method", m.group(1), seg_start_line)
            stack.append(kind)
            seg = []
        elif c == "}":
            kind = stack.pop() if stack else None
            if kind:
                methods.append((kind[1], kind[2], line))
            seg = []
        elif c == ";":
            seg = []
        else:
            seg.append(c)
        i += 1
    return len(code_lines), methods


def sizes():
    files, methods = {}, {}
    for p in sorted(SRC.rglob("*.java")):
        rel = p.relative_to(SRC).as_posix()
        code, ms = scan(p.read_text(encoding="utf-8"))
        files[rel] = code
        seen = {}
        for name, a, b in ms:
            seen[name] = seen.get(name, 0) + 1
            key = f"{rel}#{name}" + (f"@{seen[name]}" if seen[name] > 1 else "")
            methods[key] = b - a + 1
    return files, methods


def main():
    files, methods = sizes()
    over = {f: n for f, n in files.items() if n > FILE_LIMIT}
    over.update({m: n for m, n in methods.items() if n > METHOD_LIMIT})
    if "--list" in sys.argv:
        for k, n in sorted(over.items(), key=lambda kv: -kv[1]):
            print(f"{n:6} {k}")
        return 0
    if "--write" in sys.argv:
        BASELINE.write_text(json.dumps(dict(sorted(over.items())), indent=1) + "\n")
        print(f"wrote {len(over)} entries")
        return 0
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    errors = []
    for k, n in sorted(over.items()):
        limit = FILE_LIMIT if "#" not in k else METHOD_LIMIT
        what = "code lines" if "#" not in k else "lines"
        if k not in base:
            errors.append(f"{k}: {n} {what}, over {limit}. Split it.")
        elif n > base[k]:
            errors.append(f"{k}: grew from {base[k]} to {n} {what}. Split it.")
        elif n < base[k]:
            errors.append(f"{k}: shrank to {n}; lower its entry from {base[k]} (check-java-size.py --write).")
    for k in sorted(base):
        if k not in over:
            errors.append(f"{k}: no longer over its limit; delete its entry (check-java-size.py --write).")
    for e in errors:
        print(e)
    if errors:
        print(f"check-java-size: {len(errors)} problem(s)")
        return 1
    print(f"check-java-size: ok ({len(over)} over a limit, none grown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
