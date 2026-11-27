#!/usr/bin/env python3
"""Materialises the data files the repository does not store.

Compiled and generated files are not tracked: the analyzer dictionaries come
from pinned upstream jars (Maven Central), the ICU packs are written by the
Java tools under `crates/lucene-analysis-icu/tools/`, the tessellator corpus
by `scripts/gen-tessellator-corpus.py`, and the ml-commons tokenizer data is
downloaded. Every download is pinned by SHA-256. Outputs land at the paths the
crates `include_bytes!` and the tests read, all of them git-ignored.

    scripts/fetch-data.py                  # every pack
    scripts/fetch-data.py --pack nori      # one pack (repeatable)
    scripts/fetch-data.py --list           # pack names and their files
    scripts/fetch-data.py --check          # rebuild into a temp dir, compare
    scripts/fetch-data.py --force          # rebuild even if the files exist

Downloads are cached in `fixtures/.jars/` (override: `LUCENE_RUST_DATA_CACHE`),
so a second run, or a run on a machine with the cache restored, is offline.
`LUCENE_RUST_DATA_OFFLINE=1` makes a missing download an error. Files are
written atomically, so concurrent runs (cargo builds several crates at once)
are safe.

The ICU packs `coll.pack.z`, `uprops.bin.z` and `unames.bin.z` need a JDK
(`java` 11+ on PATH); everything else needs only Python 3.
"""

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path(os.environ.get("LUCENE_RUST_DATA_CACHE", ROOT / "fixtures/.jars"))
MAVEN = "https://repo1.maven.org/maven2"
# Serves the same artifacts as Central; tried alternately when Central throttles.
MAVEN_MIRROR = "https://maven-central.storage-download.googleapis.com/maven2"
ML_COMMONS = "https://raw.githubusercontent.com/opensearch-project/ml-commons/3.9/ml-algorithms/src/main/resources/analysis"

# name -> (url, sha256)
JARS = {
    "lucene-analysis-kuromoji-10.5.0.jar": (
        f"{MAVEN}/org/apache/lucene/lucene-analysis-kuromoji/10.5.0/lucene-analysis-kuromoji-10.5.0.jar",
        "1a6f68328268df5d6854921121327edad2227a6b6b78fc40abe54c29c3a5b28c",
    ),
    "lucene-analysis-nori-10.5.0.jar": (
        f"{MAVEN}/org/apache/lucene/lucene-analysis-nori/10.5.0/lucene-analysis-nori-10.5.0.jar",
        "01c3052bd62f7bfaece34aef36af527951fa30e3010c79fd6f61a813ed0277b9",
    ),
    "lucene-analysis-smartcn-10.5.0.jar": (
        f"{MAVEN}/org/apache/lucene/lucene-analysis-smartcn/10.5.0/lucene-analysis-smartcn-10.5.0.jar",
        "0660dfaf2d8e9966116ab0fc805c409a78cc67ac5515676b8bd1887a2dfbbcc1",
    ),
    "lucene-analysis-stempel-10.5.0.jar": (
        f"{MAVEN}/org/apache/lucene/lucene-analysis-stempel/10.5.0/lucene-analysis-stempel-10.5.0.jar",
        "f46699a4457e1cec1035737c37be1f061eaf32d96b6679968bbda19d1b12ed2b",
    ),
    "lucene-analysis-icu-10.5.0.jar": (
        f"{MAVEN}/org/apache/lucene/lucene-analysis-icu/10.5.0/lucene-analysis-icu-10.5.0.jar",
        "53d7f2e28016022a9bffe6435c716471b902e972551067ca2d761c587892608e",
    ),
    "icu4j-77.1.jar": (
        f"{MAVEN}/com/ibm/icu/icu4j/77.1/icu4j-77.1.jar",
        "b3640b9f416a4411fd33c59abbeea8fd57d024c23e1819bf9673220a97499fe3",
    ),
    "morfologik-polish-2.1.9.jar": (
        f"{MAVEN}/org/carrot2/morfologik-polish/2.1.9/morfologik-polish-2.1.9.jar",
        "e503682b3f4e8bb7a5d05820b0e2a4a19d4bad43dae20f64741786658a9cf478",
    ),
    "morfologik-ukrainian-search-4.9.1.jar": (
        f"{MAVEN}/ua/net/nlp/morfologik-ukrainian-search/4.9.1/morfologik-ukrainian-search-4.9.1.jar",
        "463d9054b8d4cfacb9cd69566395826a6fe32fb6e7da91dbf79d37ddf7d56ba0",
    ),
}
FILES = {
    "bert-uncased.zip": (f"{ML_COMMONS}/bert-uncased.zip", "e981ea0951b3d7572cf914f565ad016c148eda17a846e7a893d3f5d7c7314f72"),
    "mbert-uncased.zip": (f"{ML_COMMONS}/mbert-uncased.zip", "e9b220fa57a14305a18e441316767f641f248228f92da33ac650811af995d09a"),
}

# The nine dictionary files Lucene's DictionaryBuilder writes, by the names
# the crates use (zlib-compressed) and the entries of the jar.
MORPH = {
    "character_definition.dat.z": "CharacterDefinition.dat",
    "connection_costs.dat.z": "ConnectionCosts.dat",
    "token_info_buffer.dat.z": "TokenInfoDictionary$buffer.dat",
    "token_info_fst.dat.z": "TokenInfoDictionary$fst.dat",
    "token_info_pos_dict.dat.z": "TokenInfoDictionary$posDict.dat",
    "token_info_target_map.dat.z": "TokenInfoDictionary$targetMap.dat",
    "unknown_buffer.dat.z": "UnknownDictionary$buffer.dat",
    "unknown_pos_dict.dat.z": "UnknownDictionary$posDict.dat",
    "unknown_target_map.dat.z": "UnknownDictionary$targetMap.dat",
}
ICU4J_DATA = "com/ibm/icu/impl/data/icudata/"
LUCENE_ICU = "org/apache/lucene/analysis/icu/"

# pack -> list of (destination relative to the repository, recipe).
# Recipes: ("jar", jar, entry, compress) | ("file", name) | ("java", tool, jars)
# | ("script", script, args).
PACKS = {}


def _morph(pack, crate, jar, base):
    d = f"crates/{crate}/src/resources/"
    PACKS[pack] = [(d + dst, ("jar", jar, base + src, True)) for dst, src in MORPH.items()]


_morph("kuromoji", "lucene-analysis-kuromoji", "lucene-analysis-kuromoji-10.5.0.jar", "org/apache/lucene/analysis/ja/dict/")
_morph("nori", "lucene-analysis-nori", "lucene-analysis-nori-10.5.0.jar", "org/apache/lucene/analysis/ko/dict/")
PACKS["smartcn"] = [
    (
        f"crates/lucene-analysis-smartcn/src/resources/{n}.mem.z",
        ("jar", "lucene-analysis-smartcn-10.5.0.jar", f"org/apache/lucene/analysis/cn/smart/hhmm/{n}.mem", True),
    )
    for n in ("coredict", "bigramdict")
]
PACKS["stempel"] = [
    (
        "crates/lucene-analysis-stempel/src/resources/stemmer_20000.tbl.z",
        ("jar", "lucene-analysis-stempel-10.5.0.jar", "org/apache/lucene/analysis/pl/stemmer_20000.tbl", True),
    )
]
PACKS["morfologik"] = [
    (
        "crates/lucene-analysis-morfologik/src/resources/polish.dict.z",
        ("jar", "morfologik-polish-2.1.9.jar", "morfologik/stemming/polish/polish.dict", True),
    ),
    (
        "crates/lucene-analysis-morfologik/src/resources/ukrainian.dict.z",
        ("jar", "morfologik-ukrainian-search-4.9.1.jar", "ua/net/nlp/ukrainian.dict", True),
    ),
]
_icu = "crates/lucene-analysis-icu/src/resources/"
PACKS["icu"] = (
    [
        (_icu + n, ("jar", "icu4j-77.1.jar", ICU4J_DATA + n, False))
        for n in ("nfc.nrm", "nfkc.nrm", "nfkc_cf.nrm", "nfkc_scf.nrm", "uts46.nrm", "ucase.icu")
    ]
    + [(_icu + "word.brk", ("jar", "icu4j-77.1.jar", ICU4J_DATA + "brkitr/word.brk", False))]
    + [
        (_icu + n + ".z", ("jar", "icu4j-77.1.jar", ICU4J_DATA + "brkitr/" + n, True))
        for n in ("thaidict.dict", "laodict.dict", "khmerdict.dict", "burmesedict.dict", "cjdict.dict")
    ]
    + [
        (_icu + "Default.brk", ("jar", "lucene-analysis-icu-10.5.0.jar", LUCENE_ICU + "segmentation/Default.brk", False)),
        (_icu + "MyanmarSyllable.brk", ("jar", "lucene-analysis-icu-10.5.0.jar", LUCENE_ICU + "segmentation/MyanmarSyllable.brk", False)),
        (_icu + "utr30.nrm", ("jar", "lucene-analysis-icu-10.5.0.jar", LUCENE_ICU + "utr30.nrm", False)),
        (_icu + "coll.pack.z", ("java", "GenIcuCollPack.java", ["icu4j-77.1.jar"], "jar-arg")),
        (_icu + "uprops.bin.z", ("java", "GenIcuProperties.java", ["icu4j-77.1.jar"], None)),
        (_icu + "unames.bin.z", ("java", "GenIcuNames.java", ["icu4j-77.1.jar"], None)),
    ]
)
PACKS["tessellator"] = [("fixtures/corpus/real_polygons.z", ("script", "scripts/gen-tessellator-corpus.py"))]
PACKS["ml-analysis"] = [
    ("opensearch-plugin/src/main/resources/analysis/bert-uncased.zip", ("file", "bert-uncased.zip")),
    ("opensearch-plugin/src/main/resources/analysis/mbert-uncased.zip", ("file", "mbert-uncased.zip")),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def download(url, sha, dest):
    if dest.exists() and sha256_file(dest) == sha:
        return dest
    if os.environ.get("LUCENE_RUST_DATA_OFFLINE"):
        sys.exit(f"fetch-data: {dest.name} is not cached and LUCENE_RUST_DATA_OFFLINE is set")
    print(f"fetch-data: downloading {url}", file=sys.stderr)
    last = None
    data = None
    # Maven Central answers 429 to a busy runner pool for a minute or more, so
    # the waits grow well past a few seconds and a Retry-After is honoured.
    waits = [5, 15, 30, 60, 90]
    urls = [url]
    if url.startswith(MAVEN + "/"):
        urls.append(MAVEN_MIRROR + url[len(MAVEN) :])
    for attempt in range(len(waits) + 1):
        try:
            with urllib.request.urlopen(urls[attempt % len(urls)], timeout=120) as r:
                data = r.read()
            break
        except Exception as e:  # network error or throttling: retry with backoff
            last = e
            if attempt == len(waits):
                break
            import time

            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = waits[attempt]
            if retry_after and retry_after.isdigit():
                wait = max(wait, min(int(retry_after), 120))
            print(f"fetch-data: {e}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    if data is None:
        sys.exit(f"fetch-data: cannot download {url}: {last}")
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        sys.exit(f"fetch-data: {url}\n  sha256 {got}\n  expected {sha}")
    atomic_write(dest, data)
    return dest


def jar_path(name):
    url, sha = JARS[name]
    return download(url, sha, CACHE / name)


def produce(recipe, tmp_dir):
    kind = recipe[0]
    if kind == "jar":
        _, jar, entry, compress = recipe
        with zipfile.ZipFile(jar_path(jar)) as z:
            data = z.read(entry)
        return zlib.compress(data, 9) if compress else data
    if kind == "file":
        url, sha = FILES[recipe[1]]
        return pathlib.Path(download(url, sha, CACHE / recipe[1])).read_bytes()
    if kind == "java":
        _, tool, jars, mode = recipe
        if not shutil.which("java"):
            sys.exit("fetch-data: a JDK (java 11+) is needed to write " + tool.replace("Gen", "").replace(".java", ""))
        paths = [str(jar_path(j)) for j in jars]
        out = pathlib.Path(tmp_dir) / (tool + ".out")
        src = ROOT / "crates/lucene-analysis-icu/tools" / tool
        args = ["java", "-cp", os.pathsep.join(paths), str(src)]
        if mode == "jar-arg":
            args.append(paths[0])
        args.append(str(out))
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out.read_bytes()
    raise AssertionError(kind)


def run_script(script, dest):
    subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT, stdout=sys.stderr)
    if not dest.exists():
        sys.exit(f"fetch-data: {script} did not write {dest}")


def build_pack(name, force, out_root, check):
    """Builds the pack's files under out_root (the repository for a normal
    run). Returns the list of files that differ when `check`."""
    drift = []
    with tempfile.TemporaryDirectory() as tmp:
        for rel, recipe in PACKS[name]:
            dest = out_root / rel
            if check and recipe[0] == "script":
                continue  # the script writes in place; its output is pinned by its sources
            if dest.exists() and not force and not check:
                continue
            if recipe[0] == "script":
                run_script(recipe[1], dest)
                continue
            data = produce(recipe, tmp)
            if check:
                existing = ROOT / rel
                if existing.exists() and existing.read_bytes() != data:
                    drift.append(rel)
                continue
            atomic_write(dest, data)
    return drift


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", action="append", choices=sorted(PACKS), help="only this pack (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the files in place")
    args = ap.parse_args()
    if args.list:
        for n in sorted(PACKS):
            print(n)
            for rel, _ in PACKS[n]:
                print("   ", rel)
        return
    packs = args.pack or sorted(PACKS)
    bad = []
    for n in packs:
        bad += build_pack(n, args.force, ROOT, args.check)
    if bad:
        print("fetch-data: differs from a fresh build:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
