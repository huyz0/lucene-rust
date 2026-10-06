#!/usr/bin/env python3
"""Turn one generated Snowball stemmer's strings into UTF-16 (called by
`gen_snowball.sh`).

The runtime (`src/snowball/program.rs`) works on Java's UTF-16 units, so
every string literal the backend emits -- `among` entries, `eq_s`/`eq_s_b`
literals, `slice_from`/`insert` replacements -- becomes a `&[u16]` array
(the string kept in a comment), and the two string variables an algorithm
keeps (`danish.sbl`'s `ch`, `finnish.sbl`'s `x`) become `Vec<u16>`.

`find_among`/`find_among_b` binary-search a table, so its entries must be
sorted in the order the runtime compares them. The Snowball compiler's Rust
backend sorts them by UTF-8 bytes -- a backward table by its strings'
reversed bytes -- while Lucene's Java stemmers, and this crate's runtime
(`src/snowball/program.rs`), compare UTF-16 units. For a backward table the
two orders differ whenever entries end in characters of different UTF-8
widths (Arabic `'وا'` sorts before `'تم'` by its last unit, after it by its
last byte). This rewrites each table in UTF-16 order -- forward tables by
their units, backward tables (used with `find_among_b`) by their reversed
units, as the Java backend sorts -- and renumbers every entry's
`substring_i` to its entry's new index. Results and routines move with their
entries.

The tables are also re-sorted.

With `--java FILE`, it then checks every rewritten table against the same
table of Lucene's Java stemmer: same strings, `substring_i`, results and
routines, in the same order.

Usage: snowball_utf16.py RUST_FILE [--java JAVA_FILE]
"""

import re
import sys

TABLE = re.compile(
    r"static (A_\d+): &'static \[Among<Context>; (\d+)\] = &\[\n(.*?)\n\];", re.S
)
ENTRY = re.compile(r'    Among\("((?:[^"\\]|\\.)*)", (-?\d+), (-?\d+), (None|Some\(&(\w+)\))\),')
JAVA_TABLE = re.compile(r"private static final Among (a_\d+)\[\] = \{(.*?)\};", re.S)
JAVA_ENTRY = re.compile(r'new Among\("((?:[^"\\]|\\.)*)", (-?\d+), (-?\d+)(?:, "(\w*)", methodObject)?\)')


def rust_unescape(s):
    s = re.sub(r"\\u\{([0-9A-Fa-f]+)\}", lambda m: chr(int(m.group(1), 16)), s)
    return s.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


def java_unescape(s):
    s = re.sub(r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return s.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


def units(s):
    b = s.encode("utf-16-be")
    return [int.from_bytes(b[i : i + 2], "big") for i in range(0, len(b), 2)]


def array(s):
    return "&[" + ", ".join(str(u) for u in units(s)) + "]"


LITERAL_CALL = re.compile(
    r'(env\.(?:eq_s|eq_s_b)\(&|env\.slice_from\(|env\.insert\(bra, ket, )"((?:[^"\\]|\\.)*)"\)'
)


def main():
    path = sys.argv[1]
    java = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--java" else None
    src = open(path, encoding="utf-8").read()
    tables = {}

    def rewrite(m):
        name, n, body = m.group(1), int(m.group(2)), m.group(3)
        lines = body.split("\n")
        entries = [ENTRY.fullmatch(l) for l in lines]
        if len(entries) != n or not all(entries):
            sys.exit(f"{path}: {name}: cannot parse")
        backward = f"find_among_b({name}," in src
        forward = f"find_among({name}," in src
        if backward == forward:
            sys.exit(f"{path}: {name}: used {'both ways' if backward else 'never'}")
        key = lambda i: (
            units(rust_unescape(entries[i].group(1)))[:: -1 if backward else 1]
        )
        order = sorted(range(n), key=key)
        new_index = {old: new for new, old in enumerate(order)}
        out = []
        rows = []
        for old in order:
            e = entries[old]
            sub = int(e.group(2))
            sub = new_index[sub] if sub >= 0 else -1
            s = rust_unescape(e.group(1))
            out.append(f'    Among({array(s)}, {sub}, {e.group(3)}, {e.group(4)}), // "{e.group(1)}"')
            rows.append((rust_unescape(e.group(1)), sub, int(e.group(3)), e.group(5) or ""))
        tables["a_" + name[2:]] = rows
        return f"static {name}: &'static [Among<Context>; {n}] = &[\n" + "\n".join(out) + "\n];"

    src = TABLE.sub(rewrite, src)
    src = LITERAL_CALL.sub(
        lambda m: m.group(1).rstrip("&") + array(rust_unescape(m.group(2))) + ")", src
    )
    for var in ("S_ch", "S_x"):
        src = src.replace(f"    {var}: String,", f"    {var}: Vec<u16>,")
        src = src.replace(f"        {var}: String::new(),", f"        {var}: Vec::new(),")
    if re.search(r'env\.\w+\([^)]*"', src) or "String" in src:
        sys.exit(f"{path}: a string literal or String is left")
    open(path, "w", encoding="utf-8").write(src)
    if java:
        jsrc = open(java, encoding="utf-8").read()
        jtables = {}
        for m in JAVA_TABLE.finditer(jsrc):
            jtables[m.group(1)] = [
                (java_unescape(e.group(1)), int(e.group(2)), int(e.group(3)), e.group(4) or "")
                for e in JAVA_ENTRY.finditer(m.group(2))
            ]
        if jtables != tables:
            for k in sorted(set(jtables) | set(tables)):
                if jtables.get(k) != tables.get(k):
                    sys.exit(f"{path}: table {k} differs from Lucene's: {tables.get(k)} vs {jtables.get(k)}")
            sys.exit(f"{path}: tables differ from Lucene's")


if __name__ == "__main__":
    main()
