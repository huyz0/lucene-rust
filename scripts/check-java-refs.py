#!/usr/bin/env python3
"""Check that Java identifiers cited in comments exist in the *pinned* tree.

A comment that names a Java class, method or exception message is evidence:
the `// ARITH:` proofs, the sweep reports and the "no new bound rejects a file
real Lucene wrote" tables all rest on "Java does X here". c31's Tier-2 review
found a proof citing `DataInput.readVInt`'s
`"Invalid vInt detected (too many bits)"` -- an exception that exists in
Lucene `main` and **not** in the pinned 10.5.0 tree this port targets. The
conclusion happened to survive on a different argument, but the evidence was
fiction, and the next person to touch that code would have trusted it.

This is the mechanical form of the sweep protocol's own rule ("compare against
`lucene-10.5.0`, NOT `lucene`"), and the same discipline as grepping the Java
for `>>>`.

Two kinds of citation are checked, both only inside comments:

  1. **`ClassName.methodName`** in backticks. The pinned tree must declare
     `ClassName` (as a class/interface/record/enum, so nested types count) and
     that declaration's file must mention `methodName`. JDK types are skipped:
     they are real, they are just not Lucene.
  2. **Quoted exception/message text** -- a double-quoted run of >= 12
     characters containing a space -- but only where the surrounding comment
     actually presents it as Java's: within two lines of a class reference
     *and* of one of `throw`/`throws`/`Exception`/`assert`. Anything looser
     drowns in authors' scare-quoted English, which is most of what quotes in
     this codebase are.

A comment that names a Java string in order to say it is **not** there (as
`fst.rs` and `terms_dict.rs` now do about the `readVInt` exception) marks
itself with `check-java-refs: absent`.

Neither is exhaustive and neither tries to be: the point is to catch a
citation of something that is not there at all.

Scope is incremental, exactly as the arithmetic gate's is: by default it scans
`AUDITED` below -- the files whose citations have been checked -- and fails on
any regression there. `--all` scans the whole workspace and shows the backlog
(98 citations at c31, mostly paraphrases naming a method that lives on a
sibling or parent class). Add a file to `AUDITED` when its citations are
clean.

Usage:
  scripts/check-java-refs.py [--java DIR] [--path P ...] [--all] [--list]

Exits non-zero with a per-citation report on failure. If the pinned tree is
absent the check skips rather than failing, so it stays runnable off the
development machine.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files whose Java citations are verified. Grows as modules are audited.
AUDITED = [
    "crates/lucene-codecs/src/fst.rs",
    "crates/lucene-codecs/src/vectors.rs",
    "crates/lucene-codecs/src/postings_writer.rs",
    "crates/lucene-codecs/src/hnsw.rs",
    "crates/lucene-codecs/src/hnsw_vectors.rs",
    "crates/lucene-codecs/src/terms_dict.rs",
    "crates/lucene-codecs/tests/fst_byte_flip_sweep.rs",
]
def _default_java() -> Path:
    """The pinned tree, wherever this is running.

    `scripts/docker-test.sh` mounts it read-only at `/lucene-10.5.0`; on the
    host it sits beside the repo. `LUCENE_JAVA_SRC` overrides both.
    """
    if "LUCENE_JAVA_SRC" in os.environ:
        return Path(os.environ["LUCENE_JAVA_SRC"])
    for candidate in (Path("/lucene-10.5.0"), Path.home() / "work" / "lucene-10.5.0"):
        if candidate.is_dir():
            return candidate
    return Path("/lucene-10.5.0")


DEFAULT_JAVA = _default_java()

COMMENT = re.compile(r"^\s*(?://[/!]?|\*)\s?(.*)$")
# A comment that names Java text precisely in order to say it is absent.
ABSENT_PRAGMA = "check-java-refs: absent"
# `Foo.java`, `AGENTS.md`: a filename, not a `Class.method`.
FILE_EXTS = {"java", "md", "rs", "txt", "py", "sh", "toml", "properties", "json"}
# What makes a quoted string a citation of Java's own text rather than prose.
MESSAGE_CUE = re.compile(r"\bthrows?\b|Exception|\bassert")
QUOTED = re.compile(r'"([^"\\]{12,})"')
CLASS_METHOD = re.compile(r"`([A-Z][A-Za-z0-9]{2,})\.([a-z][A-Za-z0-9]*)\(?\)?`")

# Names that are not Lucene classes: Rust types that happen to match the
# pattern, and JDK types (real, but outside the tree this checks). Extend with
# a reason.
IGNORE_CLASSES = {
    # Rust
    "Self", "Vec", "Ok", "Err", "Some", "None", "String", "Option", "Result",
    "Box", "Iterator", "Ordering", "Duration", "Instant", "HashMap", "HashSet",
    # JDK
    "Arrays", "Math", "System", "Objects", "Collections", "Comparator",
    "Integer", "Long", "Float", "Double", "Short", "Byte", "Boolean",
    "Character", "Optional", "List", "Map", "Set", "Thread", "Files", "Path",
    "Object", "Class", "Number", "StringBuilder", "ByteBuffer", "Stream",
    "SplittableRandom", "Random", "BitSet", "Iterable", "Comparable",
}


def comment_lines(path: Path):
    for i, raw in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        m = COMMENT.match(raw)
        if m:
            yield i, m.group(1)


DECL = re.compile(
    r"\b(?:class|interface|record|enum|@interface)\s+([A-Z][A-Za-z0-9_]*)"
)


def build_index(java: Path) -> tuple[dict[str, list[str]], str]:
    """Map every declared type name to the source of the files declaring it.

    Keyed on the *declaration*, not the file name, so a nested type
    (`Lucene99HnswVectorsReader.OffHeapHnswGraph`,
    `HnswGraphBuilder.GraphBuilderKnnCollector`) resolves the same as a
    top-level one -- those were the bulk of the first draft's false positives.
    """
    by_class: dict[str, list[str]] = {}
    blobs: list[str] = []
    for p in java.rglob("*.java"):
        text = p.read_text(errors="replace")
        blobs.append(text)
        for name in set(DECL.findall(text)):
            by_class.setdefault(name, []).append(text)
    return by_class, "\n".join(blobs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", type=Path, default=DEFAULT_JAVA)
    ap.add_argument("--path", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="scan the whole workspace")
    ap.add_argument("--list", action="store_true", help="print every citation checked")
    args = ap.parse_args()

    if not args.java.is_dir():
        print(f"check-java-refs: {args.java} not present, skipping")
        return 0

    scope = args.path or (["crates"] if args.all else AUDITED)
    paths: list[Path] = []
    for p in scope:
        base = ROOT / p
        paths.extend(sorted(base.rglob("*.rs")) if base.is_dir() else [base])
    paths = [p for p in paths if "/target" not in str(p)]

    # This port's own fixture generators are legitimately citable too.
    by_class, blob = build_index(args.java)
    fixtures = ROOT / "fixtures" / "src"
    if fixtures.is_dir():
        gen_by_class, gen_blob = build_index(fixtures)
        for name, srcs in gen_by_class.items():
            by_class.setdefault(name, []).extend(srcs)
        blob = blob + "\n" + gen_blob
    errors: list[str] = []
    checked = 0

    for path in paths:
        rel = path.relative_to(ROOT)
        lines = dict(comment_lines(path))
        near_class_ref = set()
        for line_no, text in lines.items():
            if any(
                c not in IGNORE_CLASSES and m not in FILE_EXTS
                for c, m in CLASS_METHOD.findall(text)
            ):
                near_class_ref.update({line_no - 2, line_no - 1, line_no, line_no + 1, line_no + 2})

        for line_no, text in sorted(lines.items()):
            for cls, method in CLASS_METHOD.findall(text):
                if cls in IGNORE_CLASSES or method in FILE_EXTS:
                    continue
                checked += 1
                if args.list:
                    print(f"  ref  {rel}:{line_no}: {cls}.{method}")
                sources = by_class.get(cls)
                if sources is None:
                    errors.append(
                        f"{rel}:{line_no}: cites `{cls}.{method}`, but the "
                        f"pinned tree at {args.java} declares no type named "
                        f"{cls}."
                    )
                elif not any(method in src for src in sources):
                    errors.append(
                        f"{rel}:{line_no}: cites `{cls}.{method}`, but {cls} "
                        f"in the pinned tree does not mention {method}."
                    )
            window = range(line_no - 2, line_no + 3)
            context = " ".join(lines.get(n, "") for n in window)
            if line_no not in near_class_ref:
                continue
            if ABSENT_PRAGMA in context or not MESSAGE_CUE.search(context):
                continue
            for quoted in QUOTED.findall(text):
                if " " not in quoted or quoted.startswith("http"):
                    continue
                checked += 1
                if args.list:
                    print(f"  msg  {rel}:{line_no}: {quoted!r}")
                if quoted not in blob:
                    errors.append(
                        f"{rel}:{line_no}: quotes {quoted!r} beside a Java class "
                        f"reference, but that text appears nowhere under "
                        f"{args.java}. Check it against the *pinned* tree -- "
                        f"`main` is not it."
                    )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\ncheck-java-refs: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print(f"check-java-refs: ok ({checked} citation(s) verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
