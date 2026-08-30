#!/usr/bin/env python3
"""Mechanical consistency check for docs/parity.md.

`parity.md` is the source of truth for what is ported (see the
`parity-tracking` skill), and it is maintained append-only by many
concurrent authors. Two failure modes have already been observed and cost
real review time:

  * A Rust path in a row no longer exists -- a module was renamed or moved
    and the row rotted. Batch c10 shipped a stale path that survived to its
    own Tier-2 review.
  * A ported source file has no row at all, so its status cannot be looked
    up.

Both are mechanical and have no false positives.

Verifying that a row's *Java* side names something real is deliberately not
done here: `scripts/check-java-refs.py` does it for the whole tree, against
the pinned 10.5.0 checkout, and resolves that checkout in both the host and
container layouts. This script used to print a warning about a
Java-counterpart check it never performed, which was its own small instance
of the defect both scripts exist to catch. Detecting two rows that
genuinely *contradict* each other is deliberately NOT automated: a class
routinely has several rows (read side and write side, a scoped-down first
cut and a later widening), and a heuristic over the status text flags
fourteen of those for every real problem it finds. `--verbose` lists the
multi-row classes for a human to scan instead.
"""
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files that are boundary or test infrastructure rather than a port of
# anything in Lucene. Each needs a reason, so the list cannot grow silently.
EXEMPT = {
    "lucene-ffi/src/raw.rs": "C-ABI pointer/length helpers",
    "lucene-ffi/src/registry.rs": "handle table; the boundary's own machinery",
    "lucene-ffi/src/legacy_boolean_abi.rs": "test-only bridge pinning the pre-c13 ABI",
    "lucene-util/src/test_support.rs": "shared test scratch-directory guard; compiled only under cfg(test)/the test-support feature",
}
PARITY = os.path.join(ROOT, "docs", "parity.md")

# A Rust path: `crate/src/path.rs`, optionally followed by `::item`.
RUST_PATH = re.compile(r"`(lucene-[a-z]+/(?:src|tests|benches|examples)/[A-Za-z0-9_/]+\.rs)(?:::[^`]*)?`")
# The same, capturing the `::item` suffix -- a single item, or a
# `::{a, b, C::d}` group. Validated since c41: c37's Tier-2 review found
# `parity.md` describing two *deleted* functions in the present tense, and
# this script pointedly checked only the file path.
RUST_ITEMS = re.compile(
    r"`(lucene-[a-z]+/(?:src|tests|benches|examples)/[A-Za-z0-9_/]+\.rs)::([^`]+)`"
)
# What an item name may look like once the `Type::method` and generic noise is
# stripped: the last path segment is what has to exist in the file.
ITEM_SPLIT = re.compile(r"[,\s]+")
# A Java class reference: `pkg/Class` or `pkg/Class.method`, inside backticks.
JAVA_REF = re.compile(r"`((?:[a-z0-9]+/)+[A-Z][A-Za-z0-9]*)")


def rows(text):
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] == "Java":
            continue
        yield lineno, cells


def item_names(suffix):
    """The identifiers a row's `::item` suffix names.

    `::write`, `::{a, b}`, `::Directory::create_output` and
    `::SliceInput::slice_input` all reduce to the *last* segment of each
    comma-separated entry -- the name that has to be defined in the file. A
    `Type::{a, b}` group expands to both.
    """
    suffix = suffix.strip()
    if suffix.startswith("{") and suffix.endswith("}"):
        suffix = suffix[1:-1]
    # `Directory::{create_output, sync}` -> `create_output, sync`
    suffix = re.sub(r"\w+::\{", "", suffix).replace("}", "")
    for part in ITEM_SPLIT.split(suffix):
        part = part.strip().strip(",")
        if not part:
            continue
        name = part.split("::")[-1]
        if re.fullmatch(r"[A-Za-z_]\w*", name):
            yield name


DEFINITION = (
    "fn {0}",
    "struct {0}",
    "enum {0}",
    "trait {0}",
    "type {0}",
    "const {0}",
    "static {0}",
    "mod {0}",
    "union {0}",
    "macro_rules! {0}",
)


def defines(source, name):
    """Whether `source` defines an item called `name`.

    Textual, deliberately: a real resolver would need the whole crate graph,
    and the failure this catches -- a row naming something a diff deleted --
    shows up as *no occurrence at all*. A `use` re-export counts, because a
    row may legitimately point at the module that publishes the name.
    """
    for shape in DEFINITION:
        if re.search(r"\b" + shape.format(re.escape(name)) + r"\b", source):
            return True
    # A re-export (`pub use foo::Bar;`) or an enum variant / struct field the
    # row names.
    return bool(
        re.search(r"pub use [^;]*\b" + re.escape(name) + r"\b", source)
        or re.search(r"^\s*" + re.escape(name) + r"\s*[,({]", source, re.M)
    )


def main():
    text = open(PARITY, encoding="utf-8").read()
    errors = []
    java_to_rows = defaultdict(list)

    for lineno, cells in rows(text):
        java_cell, rust_cell, status = cells[0], cells[1], cells[2]

        for path in RUST_PATH.findall(rust_cell):
            if not os.path.exists(os.path.join(ROOT, "crates", path)):
                errors.append(
                    f"{PARITY}:{lineno}: Rust path does not exist: {path}"
                )

        for path, items in RUST_ITEMS.findall(rust_cell):
            full = os.path.join(ROOT, "crates", path)
            if not os.path.exists(full):
                continue  # already reported above
            source = open(full, encoding="utf-8").read()
            for item in item_names(items):
                if not defines(source, item):
                    errors.append(
                        f"{PARITY}:{lineno}: {path} does not define `{item}` "
                        f"(the row's Rust column names it)"
                    )

        for ref in JAVA_REF.findall(java_cell):
            java_to_rows[ref].append((lineno, status))

    # Coverage: every ported source file should be described by at least one
    # row. A file with no row is a file whose port status nobody can look up,
    # which is the failure `parity.md` exists to prevent.
    mentioned = set(RUST_PATH.findall(text))
    crates = os.path.join(ROOT, "crates")
    for crate in sorted(os.listdir(crates)):
        src = os.path.join(crates, crate, "src")
        if not os.path.isdir(src):
            continue
        for dirpath, _, files in os.walk(src):
            for name in sorted(files):
                if not name.endswith(".rs"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), crates)
                if name in ("lib.rs", "error.rs"):
                    continue  # module facade / error enum: no Java counterpart
                if rel in EXEMPT:
                    continue
                if rel not in mentioned:
                    errors.append(
                        f"{PARITY}: no row describes {rel} -- add one, "
                        f"or say explicitly that it has no Java counterpart"
                    )

    # Informational only: a Java class with several rows is normal (read side
    # and write side, or a scoped-down first cut plus a later widening), so
    # this is reported, never failed. It is here because three genuinely
    # self-contradicting pairs reached review before anyone noticed.
    multi = {r: e for r, e in java_to_rows.items() if len(e) > 1}
    if multi and "--verbose" in sys.argv:
        print("classes with multiple rows (review by hand, not an error):")
        for ref, entries in sorted(multi.items()):
            print(f"  {ref}: lines {', '.join(str(ln) for ln, _ in entries)}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\ncheck-parity: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print("check-parity: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
