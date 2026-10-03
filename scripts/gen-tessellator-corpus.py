#!/usr/bin/env python3
"""Builds the real-world polygon corpus the Tessellator fixtures replay.

Writes `fixtures/corpus/real_polygons.z`: zlib-compressed UTF-8, one shape
per line, `source<TAB>name<TAB>wkt|geojson<TAB>text`, the text exactly as the
source spells it (numbers keep their original digits), so Lucene's own
parsers -- `SimpleWKTShapeParser` and `Polygon.fromGeoJSON` -- and this port's
ports of them read the same doubles. `fixtures/src/GenGeoTessellatorReal.java`
tessellates every polygon of it with Lucene; `crates/lucene-util/tests/
geo_fixtures.rs` does the same with this port and compares.

Sources, every download pinned by SHA-256 and cached under
`fixtures/.jars/tessellator-corpus/` (never committed):

* Apache Lucene 10.5.0 (Apache-2.0): the real-world shapes its own
  `TestTessellator` replays -- the 21 `.geojson.gz`/`.wkt.gz` resources of
  `lucene-test-framework-10.5.0.jar` (`org/apache/lucene/tests/geo/`), and
  every polygon written inline in `TestTessellator.java` (issue reports from
  OpenStreetMap-derived data: LUCENE-xxxx / GitHub issue numbers).
* Natural Earth v5.1.2 (public domain, naturalearthdata.com, via the
  `nvkelso/natural-earth-vector` repository's GeoJSON exports):
  1:50m admin-0 countries, admin-1 states/provinces and lakes, in full; at
  1:10m, every lake with an island (a hole) and ten countries chosen for what
  they stress -- Norway, Chile and Greece (fjords and archipelagos), Russia
  (22,908 vertices, split at the antimeridian), Antarctica (the pole and
  +-180), Italy, South Africa and Kyrgyzstan (holes: enclaves), France and
  Kazakhstan.

Run it only to change the corpus; the generator and the tests read the
committed file.
"""

import decimal
import gzip
import hashlib
import io
import json
import pathlib
import re
import sys
import urllib.request
import zipfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "fixtures/.jars/tessellator-corpus"
OUT = ROOT / "fixtures/corpus/real_polygons.z"

LUCENE_TAG = "releases/lucene/10.5.0"
NE_TAG = "v5.1.2"
NE = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{NE_TAG}/geojson"
SOURCES = {
    "lucene-test-framework-10.5.0.jar": "https://repo1.maven.org/maven2/org/apache/lucene/"
    "lucene-test-framework/10.5.0/lucene-test-framework-10.5.0.jar",
    "TestTessellator.java": f"https://raw.githubusercontent.com/apache/lucene/{LUCENE_TAG}/"
    "lucene/core/src/test/org/apache/lucene/geo/TestTessellator.java",
    **{
        f"{n}.geojson": f"{NE}/{n}.geojson"
        for n in [
            "ne_50m_admin_0_countries",
            "ne_50m_admin_1_states_provinces",
            "ne_50m_lakes",
            "ne_10m_admin_0_countries",
            "ne_10m_lakes",
        ]
    },
}
SHA256 = {}
SHA_FILE = ROOT / "fixtures/corpus/real_polygons.sha256"

NE_10M_COUNTRIES = ["NOR", "CHL", "GRC", "RUS", "ATA", "ITA", "ZAF", "KGZ", "FRA", "KAZ"]


def load_pins():
    for line in SHA_FILE.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            SHA256[name] = digest


def fetch(name):
    url = SOURCES[name]
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"tessellator-corpus: downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if SHA256.get(name) != digest:
        sys.exit(f"{name}: sha256 {digest}, pinned {SHA256.get(name)} ({SHA_FILE})")
    return data


def num(v):
    s = str(v)
    if "E" in s or "e" in s:
        raise ValueError(f"exponent in coordinate {s}")
    return s


def dump(v):
    """GeoJSON coordinates back to text with each number's original digits."""
    if isinstance(v, list):
        return "[" + ",".join(dump(x) for x in v) + "]"
    return num(v)


def geometry_json(g):
    return '{"type":"%s","coordinates":%s}' % (g["type"], dump(g["coordinates"]))


def natural_earth(name, key, keep=lambda props, geom: True):
    doc = json.loads(fetch(name), parse_float=decimal.Decimal, parse_int=decimal.Decimal)
    rows = []
    seen = {}
    for ft in doc["features"]:
        g = ft["geometry"]
        props = ft["properties"]
        if g is None or not keep(props, g):
            continue
        label = str(key(props)).replace("\t", " ")
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}#{seen[label]}"
        rows.append((name.removesuffix(".geojson"), label, "geojson", geometry_json(g)))
    return rows


def has_hole(_props, g):
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return any(len(p) > 1 for p in polys)


def java_literals(src):
    """The `String wkt|geoJson = ...;` declarations of TestTessellator, each
    concatenation of string literals and text blocks joined (comments skipped)."""
    out = []
    for m in re.finditer(r"public void (test\w+)\(|String (wkt|geoJson)\s*=", src):
        if m.group(1):
            method = m.group(1)
            continue
        i, parts = m.end(), []
        while src[i] != ";":
            if src.startswith('"""', i):
                j = src.index('"""', i + 3)
                body = src[src.index("\n", i) + 1 : j]
                parts.append(body)
                i = j + 3
            elif src[i] == '"':
                j = i + 1
                while src[j] != '"':
                    j += 2 if src[j] == "\\" else 1
                parts.append(bytes(src[i + 1 : j], "utf-8").decode("unicode_escape"))
                i = j + 1
            elif src.startswith("//", i):
                i = src.index("\n", i)
            else:
                i += 1
        out.append((method, "wkt" if m.group(2) == "wkt" else "geojson", "".join(parts)))
    return out


def lucene():
    rows = []
    jar = zipfile.ZipFile(io.BytesIO(fetch("lucene-test-framework-10.5.0.jar")))
    for entry in sorted(jar.namelist()):
        m = re.match(r"org/apache/lucene/tests/geo/(.+)\.(geojson|wkt)\.gz$", entry)
        if m:
            text = gzip.decompress(jar.read(entry)).decode("utf-8")
            text = " ".join(text.split())
            rows.append(("lucene-test-framework", f"{m.group(1)}.{m.group(2)}", m.group(2), text))
    src = fetch("TestTessellator.java").decode("utf-8")
    seen = {}
    for method, kind, text in java_literals(src):
        if not (text.lstrip().startswith(("POLYGON", "MULTIPOLYGON", "{"))):
            continue
        seen[method] = seen.get(method, 0) + 1
        label = method if seen[method] == 1 else f"{method}#{seen[method]}"
        rows.append(("TestTessellator", label, kind, " ".join(text.split())))
    return rows


def main():
    load_pins()
    rows = lucene()
    rows += natural_earth("ne_50m_admin_0_countries.geojson", lambda p: p["ADM0_A3"])
    rows += natural_earth("ne_50m_admin_1_states_provinces.geojson", lambda p: p["adm1_code"])
    rows += natural_earth("ne_50m_lakes.geojson", lambda p: p["name"] or "unnamed")
    rows += natural_earth(
        "ne_10m_admin_0_countries.geojson",
        lambda p: p["ADM0_A3"],
        lambda p, g: p["ADM0_A3"] in NE_10M_COUNTRIES,
    )
    rows += natural_earth("ne_10m_lakes.geojson", lambda p: p["name"] or "unnamed", has_hole)
    for r in rows:
        assert all("\t" not in c and "\n" not in c for c in r), r[:2]
    text = "".join("\t".join(r) + "\n" for r in rows).encode("utf-8")
    OUT.write_bytes(zlib.compress(text, 9))
    print(f"{OUT.relative_to(ROOT)}: {len(rows)} shapes, {len(text)} bytes, "
          f"{OUT.stat().st_size} compressed", file=sys.stderr)


if __name__ == "__main__":
    main()
