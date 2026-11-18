#!/usr/bin/env python3
"""Keep source files from growing past a review-sized limit.

Counts the production lines of every `crates/*/src/**.rs` file (not the test framework crate's, nor a `tests.rs`): test code
(`#[cfg(test)]` items, and files that open with `#![cfg(test)]`) does not
count. A file over LIMIT lines must be listed in `OVER`, with the size it had
when it was listed; it may not grow past that size. When a listed file
shrinks, lower its entry (the check says so), and when it drops under LIMIT,
delete the entry. A new file over LIMIT fails: split it before adding it.

Run from anywhere: `python3 scripts/check-file-size.py`; `--list` prints every
file over LIMIT with its size.
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 3000

# Files over LIMIT production lines, with their size when listed.
OVER = {
    "crates/lucene-index/src/index_writer.rs": 13144,
    "crates/lucene-aggs/src/bucket_aggs.rs": 7314,
    "crates/lucene-codecs/src/postings.rs": 6844,
    "crates/lucene-index/src/merge.rs": 5741,
    "crates/lucene-index/src/check_index.rs": 5677,
    "crates/lucene-codecs/src/doc_values.rs": 6117,
    "crates/lucene-search/src/top_field.rs": 5754,
    "crates/lucene-search/src/lib.rs": 5723,
    "crates/lucene-codecs/src/blocktree.rs": 5698,
    "crates/lucene-search/src/directory_reader.rs": 4102,
    "crates/lucene-search/src/vector_query.rs": 4014,
    "crates/lucene-codecs/src/fst.rs": 3908,
    "crates/lucene-analysis/src/en/kstem_data.rs": 3791,
    "crates/lucene-codecs/src/stored_fields.rs": 3791,
    "crates/lucene-codecs/src/points.rs": 3684,
    "crates/lucene-analysis/src/snowball/algorithms/serbian.rs": 3602,
    "crates/lucene-codecs/src/postings_writer.rs": 3448,
    "crates/lucene-analysis/src/lib.rs": 3195,
    "crates/lucene-search/src/exec/extended.rs": 3123,
}


def load_invariants():
    spec = importlib.util.spec_from_file_location(
        "invariants", os.path.join(ROOT, "scripts", "check-port-invariants.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["invariants"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    inv = load_invariants()
    sizes = {}
    for rel, lines in inv.src_files():
        rel = rel.replace(os.sep, "/")
        if rel.startswith("crates/lucene-test-framework/") or os.path.basename(rel) in (
            "tests.rs",
            "unit_tests.rs",
        ):
            continue
        blanked = inv.blank_cfg_test(lines)
        # Blank lines inside production code count: a blank line that was a
        # test line is "" too, so count lines of `lines` not blanked away.
        n = sum(1 for a, b in zip(lines, blanked) if not (a != "" and b == ""))
        if blanked.count("") == len(blanked):
            n = 0
        sizes[rel] = n
    over = {f: n for f, n in sizes.items() if n > LIMIT}
    if "--list" in sys.argv:
        for f, n in sorted(over.items(), key=lambda kv: -kv[1]):
            print(f"{n:6} {f}")
        return 0
    errors = []
    for f, n in sorted(over.items()):
        if f not in OVER:
            errors.append(f"{f}: {n} production lines, over {LIMIT}. Split it.")
        elif n > OVER[f]:
            errors.append(f"{f}: grew from {OVER[f]} to {n} production lines. Split it.")
        elif n < OVER[f]:
            errors.append(f"{f}: shrank to {n}; lower its OVER entry from {OVER[f]}.")
    for f in sorted(OVER):
        if f not in over:
            errors.append(f"{f}: no longer over {LIMIT}; delete its OVER entry.")
    for e in errors:
        print(e)
    if errors:
        print(f"check-file-size: {len(errors)} problem(s)")
        return 1
    print(f"check-file-size: ok ({len(over)} file(s) over {LIMIT}, none grown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
