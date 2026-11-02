#!/usr/bin/env python3
"""Check that the plugin's wire constants equal the Rust ones they mirror.

The OpenSearch plugin and `lucene-ffi` speak a byte protocol: query node
kinds, aggregation kinds, sort flags, stored-field types, status codes, the
ABI version. Each number is written twice, once as a Java `static final`
in `opensearch-plugin/src/main/java` and once as a Rust `const` in
`crates/lucene-ffi/src`, and nothing but a comment ties the two together. A
kind renumbered on one side only decodes as some other node, which the
native self-test catches late (it needs the JDK 25 image) or not at all
when the two shapes happen to share a layout.

This check pairs every integer `static final` in the plugin's main sources
with its Rust counterpart and fails when the values differ:

  * The Rust candidates are the `const`s of the `lucene-ffi` files the Java
    file names (`jvm_aggs.rs`, ...) plus those `EXTRA_FILES` adds. A Rust
    constant pairs when its name is the Java name, or the Java name behind
    the Java file's `PREFIXES` entry (`AGG_TERMS` for `NativeAggregationTree`'s
    `TERMS`); `RENAMES` covers the rest. The exact name wins over a prefixed
    one, and candidates that disagree among themselves need a `RENAMES` entry.
  * `NativeBridge`'s status codes pair with the `FfiStatus` variants
    (`INVALID_ARGUMENT` for `InvalidArgument`), and `EXPECTED_ABI_VERSION`
    with `JVM_ABI_VERSION`.
  * A constant with no Rust counterpart (a Java-side batch size, say) must be
    listed in `JAVA_ONLY`, so a new wire constant cannot go unpaired by
    accident.

Run from anywhere: `python3 scripts/check-wire-constants.py`. `--list` prints
every pairing.
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAVA_DIR = os.path.join(ROOT, "opensearch-plugin", "src", "main", "java")
RUST_DIR = os.path.join(ROOT, "crates", "lucene-ffi", "src")

# Rust files a Java file encodes for without naming them.
EXTRA_FILES = {
    "NativeCollapse.java": ["jvm_reader.rs"],
    "NativeStoredFieldsReader.java": ["jvm_fetch.rs"],
    "NativeTopHits.java": ["jvm_aggs.rs"],
    "NativeSignificantTerms.java": ["jvm_aggs.rs"],
    "QueryEncoder.java": ["jvm_nodes.rs"],
}

# The prefix the Rust side puts in front of a Java file's names.
PREFIXES = {
    "NativeAggregationTree.java": "AGG_",
    "NativePercentiles.java": "AGG_",
    "NativeTopHits.java": "AGG_",
    "NativeSignificantTerms.java": "AGG_",
    "NativeAggregations.java": "METRIC_",
    "SortEncoder.java": "SORT_",
    "NativeCollapse.java": "COLLAPSE_",
}

# (Java file, Java name) -> (Rust file, Rust name), where the names differ by
# more than a prefix or the prefix rule is ambiguous.
RENAMES = {
    ("QueryEncoder.java", "MAX_DEPTH"): ("query.rs", "MAX_CLAUSE_DEPTH"),
    ("QueryEncoder.java", "MAX_NODES"): ("query.rs", "MAX_CLAUSE_COUNT"),
    ("GeoEncoder.java", "MAX_ORIGINS"): ("jvm_reader.rs", "MAX_GEO_ORIGINS"),
    ("SortEncoder.java", "NESTED_KEY"): ("jvm_reader.rs", "SORT_NESTED"),
    ("SortEncoder.java", "MAX_KEYS"): ("jvm_reader.rs", "MAX_SORT_KEYS"),
    ("SortEncoder.java", "GEO"): ("jvm_reader.rs", "SORT_GEO_DISTANCE"),
    ("AnalyzerSpec.java", "VERSION"): ("jvm_analysis.rs", "SPEC_VERSION"),
    ("NativeBridge.java", "EXPECTED_ABI_VERSION"): ("jvm_reader.rs", "JVM_ABI_VERSION"),
}

# Constants with no Rust counterpart, and why.
JAVA_ONLY = {
    ("QueryEncoder.java", "REGEXP_CACHE_SIZE"): "the Java side's parsed-regexp cache",
    ("LuceneChangesSnapshot.java", "DEFAULT_BATCH_SIZE"): "derived OpenSearch engine code",
    ("LuceneChangesSnapshot.java", "MIN_SEQUENTIAL_ACCESS_BATCH_SIZE"): "derived OpenSearch engine code",
    ("RustIndexWriter.java", "BUFFER_BYTES_PER_BLOB_BYTE"): "the Java side's RAM accounting estimate",
    ("ExtendedEncoder.java", "MAX_REPEATS"): "Java refuses earlier than Rust's clause limit",
}

JAVA_CONST = re.compile(
    r"static\s+final\s+(?:byte|short|int|long)\s+([A-Z][A-Z0-9_]*)\s*=\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*;"
)
RUST_CONST = re.compile(
    r"\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*(?:u8|i8|u16|i16|u32|i32|u64|i64|usize)\s*=\s*([^;]+);"
)
RUST_FILE_REF = re.compile(r"\b([a-z][a-z0-9_]*\.rs)\b")


def rust_value(expr):
    """The value of a literal const expression (`1024`, `1 << 16`, `64 << 20`), else None."""
    expr = expr.replace("_", "").strip()
    if not re.fullmatch(r"[0-9xa-fA-F<>\s()+*-]+", expr):
        return None
    try:
        return int(eval(expr, {"__builtins__": {}}))  # digits and operators only
    except Exception:
        return None


def snake(camel):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).upper()


def load_rust():
    consts = {}
    for name in sorted(os.listdir(RUST_DIR)):
        if not name.endswith(".rs"):
            continue
        with open(os.path.join(RUST_DIR, name), encoding="utf-8") as f:
            text = f.read()
        found = {}
        for m in RUST_CONST.finditer(text):
            v = rust_value(m.group(2))
            if v is not None:
                found.setdefault(m.group(1), v)
        consts[name] = found
    with open(os.path.join(RUST_DIR, "error.rs"), encoding="utf-8") as f:
        text = f.read()
    body = re.search(r"pub enum FfiStatus\s*\{(.*?)\n\}", text, re.S).group(1)
    status = {snake(m.group(1)): int(m.group(2)) for m in re.finditer(r"^\s*([A-Z]\w*)\s*=\s*(\d+)\s*,", body, re.M)}
    return consts, status


def load_java():
    out = []
    for dirpath, _, files in os.walk(JAVA_DIR):
        for name in sorted(files):
            if not name.endswith(".java"):
                continue
            with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                text = f.read()
            refs = sorted(set(RUST_FILE_REF.findall(text)))
            for m in JAVA_CONST.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                out.append((name, line, m.group(1), int(m.group(2), 0), refs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print every pairing")
    args = ap.parse_args()

    rust, status = load_rust()
    errors = []
    paired = 0
    seen_java_only = set()
    for jfile, line, name, value, refs in load_java():
        where = f"opensearch-plugin/.../{jfile}:{line} {name} = {value}"
        key = (jfile, name)
        if key in JAVA_ONLY:
            seen_java_only.add(key)
            continue
        if key in RENAMES:
            rfile, rname = RENAMES[key]
            if rname not in rust.get(rfile, {}):
                errors.append(f"{where}: RENAMES names {rfile} {rname}, which does not exist")
                continue
            matches = [(rfile, rname, rust[rfile][rname])]
        elif jfile == "NativeBridge.java" and name in status:
            matches = [("error.rs", f"FfiStatus::{name}", status[name])]
        else:
            files = [f for f in refs + EXTRA_FILES.get(jfile, []) if f in rust]
            names = {name, PREFIXES.get(jfile, "") + name}
            matches = sorted({(f, r, v) for f in files for r, v in rust[f].items() if r in names})
            exact = [x for x in matches if x[1] == name]
            if exact:
                matches = exact
        if not matches:
            errors.append(
                f"{where}: no Rust counterpart in {', '.join(refs + EXTRA_FILES.get(jfile, [])) or 'the files it names (none)'};"
                " name the Rust file in a comment, or add a RENAMES or JAVA_ONLY entry"
            )
            continue
        values = {v for _, _, v in matches}
        if len(values) > 1:
            listed = ", ".join(f"{f} {r} = {v}" for f, r, v in matches)
            errors.append(f"{where}: ambiguous ({listed}); add a RENAMES entry")
            continue
        (rvalue,) = values
        if rvalue != value:
            f, r, _ = matches[0]
            errors.append(f"{where}: Rust has {f} {r} = {rvalue}")
            continue
        paired += 1
        if args.list:
            print(f"{jfile} {name} = {value}  <->  {matches[0][0]} {matches[0][1]}")

    for key in sorted(set(JAVA_ONLY) - seen_java_only):
        errors.append(f"JAVA_ONLY lists {key[0]} {key[1]}, which is no longer a constant there")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"check-wire-constants: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print(f"check-wire-constants: {paired} plugin constants equal their Rust counterparts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
