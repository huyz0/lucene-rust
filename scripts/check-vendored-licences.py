#!/usr/bin/env python3
"""Every vendored resource the shipped library embeds has its licence text in
the root LICENSE.

`include_bytes!`/`include_str!` is how a third-party file -- a dictionary, a
rule table, a stop list -- is compiled into a crate, and so into
`liblucene_ffi.so` once the crate is linked; the root `LICENSE` (with
`NOTICE`) is what ships beside that library in the plugin zip
(`opensearch-plugin/build.gradle`'s `from(file("${repoRoot}/LICENSE"))`).
`scripts/check-licences.py` gates the crates the library links; this gates
the files any workspace crate embeds, linked yet or not.

It finds every directory such a macro reads from under `crates/*/src` and
requires each to be listed in `VENDORED` with a line of its licence's text,
which `LICENSE` must contain verbatim. A new vendored directory fails until
its licence is recorded here and in `LICENSE`; a licence text dropped from
`LICENSE` fails too.

What it cannot catch: a directory recorded against the wrong licence, or a
marker line another licence in `LICENSE` happens to contain; a resource read
at run time instead of embedded; an include whose path is not a string
literal (`concat!`, a macro argument); test-only includes (`tests/`,
`#[cfg(test)]` modules are scanned, so they must be listed too).

Usage: scripts/check-vendored-licences.py [--list]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APACHE = "Version 2.0, January 2004"

SNOWBALL = "Copyright (c) 2001, Dr Martin Porter"

# Directory (relative to the repository) -> every licence its files are
# under: (whose files, the file shipped with the library that holds the
# licence, a line of that licence's text). NOTICE stands in for LICENSE only
# where Apache Lucene itself ships the attribution there and the vendored
# file's own header carries the licence (Savoy's and Carrot2's stop lists).
VENDORED = {
    "crates/lucene-analysis/src/charfilter": [
        ("Lucene's HTMLStripCharFilter tables (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis/src/classic": [
        ("Lucene's ClassicTokenizer tables (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis/src/email": [
        ("Lucene's UAX29URLEmailTokenizer tables (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis/src/wikipedia": [
        ("Lucene's WikipediaTokenizer tables (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis/src/lang/stopwords": [
        ("Lucene's stop lists and RSLP rules (Apache-2.0)", "LICENSE", APACHE),
        ("the Snowball stop lists (BSD-3-Clause)", "LICENSE", SNOWBALL),
        ("Jacques Savoy's stop lists (BSD)", "NOTICE", "BSD-licensed created by Jacques Savoy"),
    ],
    "crates/lucene-analysis-phonetic/src/resources": [
        ("Commons Codec's Daitch-Mokotoff rules (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis-phonetic/src/resources/bm": [
        ("Commons Codec's Beider-Morse rules (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis-stempel/src/resources": [
        ("Lucene's Egothor stemmer table (Egothor Software License 1.00)", "LICENSE",
         "Egothor Software License version 1.00"),
        ("Carrot2's Polish stop list (BSD)", "NOTICE",
         "stopword list that is BSD-licensed created by the Carrot2 project"),
    ],
    "crates/lucene-analysis-morfologik/src/resources": [
        ("morfologik-polish 2.1.9's dictionary (BSD-2-Clause)", "LICENSE",
         "Copyright (c) 2016, Marcin Miłkowski"),
        ("morfologik-ukrainian-search 4.9.1's dictionary and Lucene's Ukrainian stop list "
         "(Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-analysis-kuromoji/src/resources": [
        ("Lucene's compiled mecab-ipadic-2.7.0-20070801 dictionary (NAIST, with ICOT's "
         "conditions; Lucene ships the notice in its NOTICE)", "NOTICE",
         "Nara Institute of Science and Technology (NAIST),"),
        ("Lucene's Japanese stop words, stop tags and romaji map (Apache-2.0)", "LICENSE", APACHE),
    ],
    "crates/lucene-ffi/fuzz/seeds/jvm_search": [
        ("this project's own fuzz seeds, read by a test (Apache-2.0)", "LICENSE", APACHE),
    ],
}

INCLUDE = re.compile(r'include_(?:bytes|str)!\(\s*"([^"]+)"')


def vendored_dirs():
    """Every directory an include under crates/*/src reads from, with one
    including file as the example."""
    found = {}
    crates = os.path.join(ROOT, "crates")
    for crate in sorted(os.listdir(crates)):
        src = os.path.join(crates, crate, "src")
        for dirpath, _, files in os.walk(src):
            for f in sorted(files):
                if not f.endswith(".rs"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                for m in INCLUDE.finditer(text):
                    target = os.path.normpath(os.path.join(dirpath, m.group(1)))
                    d = os.path.relpath(os.path.dirname(target), ROOT)
                    found.setdefault(d, os.path.relpath(path, ROOT))
    return found


def main():
    found = vendored_dirs()
    if "--list" in sys.argv[1:]:
        for d, example in sorted(found.items()):
            print(f"{d}\t(e.g. {example})")
        return 0
    shipped = {}
    for name in ("LICENSE", "NOTICE"):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            shipped[name] = fh.read()
    errors = []
    for d, example in sorted(found.items()):
        if d not in VENDORED:
            errors.append(
                f"{d}: embedded by {example} but not in VENDORED -- record its "
                "licence here and append its text to LICENSE"
            )
    for d, licences in sorted(VENDORED.items()):
        if d not in found:
            errors.append(f"{d}: in VENDORED but nothing embeds it any more -- drop the entry")
        for what, name, marker in licences:
            if marker not in shipped[name]:
                errors.append(f"{d}: {name} lacks the licence of {what} (no line {marker!r})")
    for e in errors:
        print(f"check-vendored-licences: {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"check-vendored-licences: {len(found)} vendored directories, each licence in LICENSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
