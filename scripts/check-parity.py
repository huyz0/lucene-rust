#!/usr/bin/env python3
"""Mechanical consistency check for the parity ledger.

The ledger is `docs/parity.md` (an index) plus one file per area under
`docs/parity/`. It is the source of truth for what is ported (see the
`parity-tracking` skill), and it is maintained by many concurrent authors.
Two failure modes have already been observed and cost real review time:

  * A Rust path in a row no longer exists -- a module was renamed or moved
    and the row rotted. Batch c10 shipped a stale path that survived to its
    own Tier-2 review.
  * A ported source file has no row at all, so its status cannot be looked
    up.

Both are mechanical and have no false positives. The same goes for a
backticked `scripts/..` or `tools/..` path in a row's Status column (a
generator or verifier the row cites as evidence): it must exist, `tools/`
under the repository or a crate the row's Rust column names.

The ledger also grew to 1.25 MB in one file, mostly dated history, before it
was split, so the layout itself is checked too:

  * the index links every `docs/parity/*.md` file, and every relative
    Markdown link in the ledger resolves to a file that exists;
  * budgets: no table row over `MAX_ROW` characters, no file over
    `MAX_FILE` bytes, the whole ledger under `MAX_TOTAL` bytes;
  * no `## ` heading appears in two files (a section split across files
    has no single place for its next row);
  * the row count the index's area table states for each file is the
    number of rows the file has.

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
PARITY_DIR = os.path.join(ROOT, "docs", "parity")

# Budgets. A row is one Java class or one coherent feature: what is ported,
# where, how it differs, the evidence, the benchmark ratio -- not its history
# (that is `git log` and `docs/sweep/`). Past these, split the row or the
# file; do not raise them to fit.
MAX_ROW = 2_000
MAX_FILE = 80_000
MAX_TOTAL = 400_000

# A line of the index's area table: its link and its row count.
AREA_ROW = re.compile(r"^\|\s*\[[^\]]*\]\(([^)]+\.md)\)\s*\|.*\|\s*(\d+)\s*\|\s*$")

# `[text](target)` with a relative target; anchors and absolute URLs aside.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")

# A Rust path: `crate/src/path.rs`, optionally followed by `::item`.
RUST_PATH = re.compile(r"`(lucene-[a-z]+(?:-[a-z0-9]+)*/(?:src|tests|benches|examples)/[A-Za-z0-9_/]+\.rs)(?:::[^`]*)?`")
# The same, capturing the `::item` suffix -- a single item, or a
# `::{a, b, C::d}` group. Validated since c41: c37's Tier-2 review found
# `parity.md` describing two *deleted* functions in the present tense, and
# this script pointedly checked only the file path.
RUST_ITEMS = re.compile(
    r"`(lucene-[a-z]+(?:-[a-z0-9]+)*/(?:src|tests|benches|examples)/[A-Za-z0-9_/]+\.rs)::([^`]+)`"
)
# What an item name may look like once the `Type::method` and generic noise is
# stripped: the last path segment is what has to exist in the file.
ITEM_SPLIT = re.compile(r"[,\s]+")
# A script or generator named in a row: `scripts/x.sh`, `tools/X.java`. The
# M11 part 2 review found the Snowball rows naming `tools/snowball_utf16_tables.py`,
# a file that never existed (the script is `snowball_utf16.py`).
TOOL_PATH = re.compile(r"`((?:scripts|tools)/[A-Za-z0-9_./-]+\.(?:sh|py|java|rs))`")
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


def check_rows(name, text, errors, java_to_rows):
    """Rust paths and `::items` in one ledger file's rows."""
    for lineno, cells in rows(text):
        java_cell, rust_cell, status = cells[0], cells[1], cells[2]

        for path in RUST_PATH.findall(rust_cell):
            if not os.path.exists(os.path.join(ROOT, "crates", path)):
                errors.append(f"{name}:{lineno}: Rust path does not exist: {path}")

        for path, items in RUST_ITEMS.findall(rust_cell):
            full = os.path.join(ROOT, "crates", path)
            if not os.path.exists(full):
                continue  # already reported above
            source = open(full, encoding="utf-8").read()
            for item in item_names(items):
                if not defines(source, item):
                    errors.append(
                        f"{name}:{lineno}: {path} does not define `{item}` "
                        f"(the row's Rust column names it)"
                    )

        # `scripts/..` is the repository's; `tools/..` is the repository's or
        # that of a crate the row's Rust column names (`crates/<crate>/tools/`).
        row_crates = {p.split("/", 1)[0] for p in RUST_PATH.findall(rust_cell)}
        for path in TOOL_PATH.findall(status):
            bases = [ROOT]
            if path.startswith("tools/"):
                bases += [os.path.join(ROOT, "crates", c) for c in sorted(row_crates)]
            if not any(os.path.exists(os.path.join(b, path)) for b in bases):
                errors.append(f"{name}:{lineno}: tool or script does not exist: {path}")

        for ref in JAVA_REF.findall(java_cell):
            java_to_rows[ref].append((f"{name}:{lineno}", status))


def ledger_files():
    """The index followed by every area file, as (path, text) pairs."""
    files = [PARITY]
    if os.path.isdir(PARITY_DIR):
        files += sorted(
            os.path.join(PARITY_DIR, f)
            for f in os.listdir(PARITY_DIR)
            if f.endswith(".md")
        )
    return [(f, open(f, encoding="utf-8").read()) for f in files]


def rel(path):
    return os.path.relpath(path, ROOT)


def check_layout(ledger, errors):
    """Index links, link targets, budgets and duplicated headings."""
    index_text = ledger[0][1]
    linked = {
        os.path.normpath(os.path.join(os.path.dirname(PARITY), target))
        for target in MD_LINK.findall(index_text)
    }
    for path, _ in ledger[1:]:
        if os.path.normpath(path) not in linked:
            errors.append(
                f"{rel(PARITY)}: the index does not link {rel(path)} -- "
                f"add it to the area table"
            )

    # The area table's row counts: `| [name](parity/name.md) | ... | N |`.
    counts = {}
    for line in index_text.splitlines():
        m = AREA_ROW.match(line)
        if m:
            counts[os.path.normpath(os.path.join(os.path.dirname(PARITY), m.group(1)))] = int(m.group(2))
    for path, text in ledger[1:]:
        actual = sum(1 for _ in rows(text))
        stated = counts.get(os.path.normpath(path))
        if stated is not None and stated != actual:
            errors.append(
                f"{rel(PARITY)}: the area table says {rel(path)} has {stated} "
                f"rows, it has {actual}"
            )

    total = 0
    headings = defaultdict(list)
    for path, text in ledger:
        size = len(text.encode("utf-8"))
        total += size
        if size > MAX_FILE:
            errors.append(
                f"{rel(path)}: {size} bytes, over the {MAX_FILE}-byte file "
                f"budget -- split the area"
            )
        for lineno, line in enumerate(text.splitlines(), 1):
            for target in MD_LINK.findall(line):
                if "://" in target or target.startswith("mailto:"):
                    continue
                full = os.path.normpath(os.path.join(os.path.dirname(path), target))
                if not os.path.exists(full):
                    errors.append(
                        f"{rel(path)}:{lineno}: link target does not exist: {target}"
                    )
            if line.startswith("|") and len(line) > MAX_ROW:
                errors.append(
                    f"{rel(path)}:{lineno}: row is {len(line)} characters, over "
                    f"the {MAX_ROW} budget -- drop history, or split the row"
                )
            if line.startswith("## "):
                headings[line.strip()].append(f"{rel(path)}:{lineno}")
    if total > MAX_TOTAL:
        errors.append(
            f"docs/parity*: {total} bytes in all, over the {MAX_TOTAL}-byte "
            f"ledger budget"
        )
    for heading, where in sorted(headings.items()):
        files = {w.rsplit(":", 1)[0] for w in where}
        if len(files) > 1:
            errors.append(
                f"heading `{heading}` appears in several files: {', '.join(where)}"
            )


# The benchmark runner's sources: a `use lucene_<crate>::...;` there names
# the items a bench case drives.
BENCH_SOURCES = os.path.join(ROOT, "benchmarks", "rust-runner", "src")
BENCH_USE = re.compile(r"\buse\s+(lucene_[a-z0-9_]+)::([^;]*);", re.S)
# The crates `check_bench_clauses` covers: M12's language modules, one crate
# per Lucene module (`lucene-analysis-kuromoji`, ...).
BENCH_CLAUSE_CRATE = re.compile(r"lucene-analysis-[a-z0-9-]+")
BENCH_CLAUSE = re.compile(r"\bBench\b")


def bench_imports():
    """`{crate: {item, ...}}`: what the benchmark runner imports, per crate."""
    imported = defaultdict(set)
    if not os.path.isdir(BENCH_SOURCES):
        return imported
    for name in sorted(os.listdir(BENCH_SOURCES)):
        if not name.endswith(".rs"):
            continue
        source = open(os.path.join(BENCH_SOURCES, name), encoding="utf-8").read()
        for crate, items in BENCH_USE.findall(source):
            imported[crate.replace("_", "-")].update(re.findall(r"\b[A-Za-z_]\w*\b", items))
    return imported


def check_bench_clauses(name, text, errors, imported):
    """A **ported** row of a language-module crate whose Rust column names an
    item a bench case imports must carry a `Bench` clause (its ratio, or a
    pointer to the row that has it): the M12 part 2 review found the Nori
    rows silent about a bench that existed and had been run.

    Scoped to the `lucene-analysis-<module>` crates: ledger-wide, the same
    rule flags 95 rows of other crates whose items a bench imports only to
    set up its input (`Directory`, `IndexWriter`, `BooleanQuery`, ...).
    """
    for lineno, cells in rows(text):
        # A `|` inside a backticked regex splits the status; keep all of it.
        rust_cell, status = cells[1], "|".join(cells[2:])
        if not status.startswith("**ported**") or BENCH_CLAUSE.search(status):
            continue
        hits = sorted(
            item
            for path, items in RUST_ITEMS.findall(rust_cell)
            if BENCH_CLAUSE_CRATE.fullmatch(path.split("/", 1)[0])
            for item in item_names(items)
            if item in imported.get(path.split("/", 1)[0], ())
        )
        if hits:
            errors.append(
                f"{name}:{lineno}: a bench case drives {', '.join(hits)}, but this "
                f"**ported** row has no Bench clause (its ratio, or `Bench: see` the row "
                f"that has it)"
            )


def main():
    ledger = ledger_files()
    errors = []
    java_to_rows = defaultdict(list)
    check_layout(ledger, errors)

    imported = bench_imports()
    for path, text in ledger:
        check_rows(rel(path), text, errors, java_to_rows)
        check_bench_clauses(rel(path), text, errors, imported)
    text = "\n".join(t for _, t in ledger)

    # Coverage: every ported source file should be described by at least one
    # row. A file with no row is a file whose port status nobody can look up,
    # which is the failure the ledger exists to prevent.
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
                relpath = os.path.relpath(os.path.join(dirpath, name), crates)
                if name in ("lib.rs", "error.rs"):
                    continue  # module facade / error enum: no Java counterpart
                if relpath in EXEMPT:
                    continue
                if relpath not in mentioned:
                    errors.append(
                        f"docs/parity: no row describes {relpath} -- add one, "
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
            print(f"  {ref}: {', '.join(where for where, _ in entries)}")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"\ncheck-parity: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print("check-parity: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
