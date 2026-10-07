# Licences and attribution

The M6 audit (T6.5), 2026-09-29. `scripts/check-licences.py` keeps the
dependency half of it true: it runs in `scripts/gate.sh` and CI and fails when
a dependency the shipped library links is not under a licence listed here.

## This project

- **Licence: Apache-2.0.** Every workspace crate declares `license =
  "Apache-2.0"` (the checker enforces it).
- **Why Apache-2.0:** this is a port of Apache Lucene 10.5.0, a derivative work
  of it, so it carries Lucene's licence and its notices (`PLAN.md` §3).
- **[`LICENSE`](../LICENSE)** is Lucene 10.5.0's own `LICENSE.txt`, verbatim:
  the Apache License 2.0 followed by the licences of third-party code inside
  Lucene. What was ported touches one of them:
  - the Brics-derived `o.a.l.util.automaton` (BSD): `lucene-codecs`'
    `automaton` ports `Operations.determinize`, `UTF32ToUTF8` and
    `ByteRunAutomaton`.
  - the University of Massachusetts' KStem stemmer and dictionary
    (BSD-style): `lucene-analysis`' `en/kstem.rs` ports `KStemmer`, and
    `en/kstem_data.rs` is its dictionary, generated from Lucene's sources
    (M11).
  - the Snowball stemmers (BSD-3-Clause, Dr Martin Porter, Richard Boulton
    and contributors): `lucene-analysis`' `snowball/algorithms/` is the
    Snowball compiler's Rust output over the `.sbl` sources Lucene 10.5.0's
    Java stemmers were generated from (Snowball commit `34f3612e`,
    `tools/gen_snowball.sh`), and `snowball/program.rs` follows Snowball's
    runtime and Lucene's `SnowballProgram`. The test vocabulary
    (`fixtures/data/snowball/`) is synthetic, built by `GenSnowball.java`
    from the stemmers' own suffix tables, plus `fixtures/corpus/snowball-targeted/`
    (words this project wrote, or concatenations of those tables' strings)
    -- no Snowball test data is redistributed. The Snowball project's own
    vocabularies (`snowball-data`, partly CC BY-SA and GPL-3.0) are only
    fetched, pinned by SHA-256, by `scripts/check-snowball-vocabulary.sh`
    into a temporary directory.
  - the language packages' stop word lists and RSLP rule files
    (`lucene-analysis/src/lang/stopwords/`, vendored verbatim from the
    10.5.0 jar, headers kept -- `english_stop.txt` (Snowball, BSD) and
    `cjk_stopwords.txt` (Lucene's own) among them, which only
    `ClasspathResourceLoader` serves): Jacques Savoy's (BSD) and Snowball's (BSD)
    lists and the others Lucene ships under its own licence, attributed in
    Lucene's `NOTICE.txt`, which [`NOTICE`](../NOTICE) carries in full; they
    are compiled into the library (`include_str!`).
  - the Savoy light stemmers (BSD reference implementations, also in
    Lucene's `NOTICE.txt`): ported in `lucene-analysis/src/lang/`.
  - Two others this port does *not* carry, and the reason:
    - the moman Levenshtein tables (MIT): `fuzzy` matches by dynamic
      programming instead;
    - Lucene's LZ4 rewrite (BSD-2-Clause): LZ4 is decompressed by the
      `lz4_flex` crate, with a literal-only encoder of this port's own.
  - The whole file is kept anyway, so the notices stay those of the upstream
    the port follows.
- **[`NOTICE`](../NOTICE)** is this project's notice, then Lucene 10.5.0's
  `NOTICE.txt` in full, then OpenSearch 3.8.0's `NOTICE.txt`.
  - OpenSearch's notice is there because files in `opensearch-plugin/.../engine/`
    are derived from OpenSearch 3.8.0 by
    `opensearch-plugin/tools/derive_engine.py`, each keeping OpenSearch's SPDX
    header under a line saying so.
- **Distribution:** the plugin zip (`gradle -p opensearch-plugin bundlePlugin`)
  ships `LICENSE` and `NOTICE` beside the jar and the native library.

- **Ported third-party Java code** (M9 T9.5). Lucene's `lucene-spatial-extras`
  is built on two libraries outside Lucene; the subsets it exercises are
  ported, file by file, like Lucene itself:
  - **Spatial4j 0.8** (`org.locationtech.spatial4j:spatial4j:0.8`,
    Apache-2.0; LocationTech, formerly an ASF-licensed Lucene spin-off) ->
    `crates/lucene-util/src/spatial4j/`. Its optional JTS dependency (EPL/EDL)
    is **not** ported or used: Lucene does not ship JTS, and spatial-extras
    reaches polygons through Geo3D instead.
  - **S2 Geometry Library for Java** (`io.sgr:s2-geometry-library-java:1.0.0`,
    Apache-2.0, Google) -> `crates/lucene-util/src/s2/` (cell ids, cells'
    vertices, the quadratic projection).
  - Both are compatible with redistribution under Apache-2.0; `NOTICE` names
    them. Their jars are fetched by the fixture scripts
    (`SPATIAL_EXTRAS_DEPS` in `scripts/lib-lucene-jars.sh`) and the
    container image, never redistributed. Neither jar carries a `NOTICE`
    file of its own.

- **Ported third-party Java code** (M12 T12.3). Lucene's
  `lucene-analysis-phonetic` runs **Apache Commons Codec 1.17.2**
  (`commons-codec:commons-codec:1.17.2`, the version its 10.5.0 pom names;
  Apache-2.0, ASF): its `language` and `language.bm` encoders are
  reimplemented in `crates/lucene-analysis-phonetic/`, and the rule files
  they load (`dmrules.txt`, 123 `bm/*.txt`, each with its ASF licence
  header) are vendored verbatim and compiled in (`include_str!`). `NOTICE`
  carries Commons Codec's notice. The jar is fetched by the fixture scripts
  (`PHONETIC_DEPS` in `scripts/lib-lucene-jars.sh`) and the container image,
  never redistributed.

- **Ported third-party code and data** (M12 T12.2, T12.3).
  - **Egothor** (`org.egothor.stemmer`, inside Lucene's
    `lucene-analysis-stempel`; Egothor Software License 1.00, BSD-style:
    redistribution keeps the notice, binaries reproduce it, the
    acknowledgement "This product includes software developed by the
    Egothor Project" is requested) -> `crates/lucene-analysis-stempel/src/egothor.rs`
    and `diff.rs`, the table reader only; the licence's full text is
    appended to the root `LICENSE` (which ships beside the library in the
    plugin zip), its conditions are quoted in `egothor.rs`'s module docs,
    and `NOTICE` carries the acknowledgement.
    Lucene's `stemmer_20000.tbl` (Apache-2.0, Lucene's) and the Carrot2 Polish
    stop words (BSD, already in Lucene's `NOTICE.txt`) are vendored.
  - **Morfologik 2.1.9** (`org.carrot2:morfologik-fsa`/`-stemming`, BSD) ->
    `crates/lucene-analysis-morfologik/`, reading and lookup only.
  - **Dictionaries**: Morfologik's Polish dictionary (`morfologik-polish`
    2.1.9, BSD-2-Clause, its licence vendored as
    `src/resources/polish.LICENSE.txt`, in the `.info` header and appended to
    the root `LICENSE`) and the
    Ukrainian dictionary (`ua.net.nlp:morfologik-ukrainian-search:4.9.1`,
    Apache-2.0) are vendored zlib-compressed (1.9 MB, 4.1 MB) so that
    `MorfologikAnalyzer::default()` and `UkrainianMorfologikAnalyzer::default()`
    work without files, as Lucene's no-argument constructors do. Both are
    licence-clean for redistribution; `NOTICE` attributes them. The
    fixture generator also uses `morfologik-fsa-builders` 2.1.9 (BSD) and
    HPPC 0.7.2 (Apache-2.0) to build test dictionaries; none of them is
    redistributed (`MORFOLOGIK_DEPS` in `scripts/lib-lucene-jars.sh`).
  - `scripts/check-vendored-licences.py` (in the gate) keeps this true:
    every directory an `include_bytes!`/`include_str!` reads from is listed
    with its licences, each of whose text the shipped `LICENSE` (or, for
    the Savoy and Carrot2 stop lists, `NOTICE`, as Lucene ships them) must
    hold.

- **Dictionaries of the M12 morphological analyzers** (T12.1).
  `crates/lucene-analysis-kuromoji/src/resources/` carries the nine binary
  files of Lucene 10.5.0's `analysis-kuromoji` jar -- mecab-ipadic-2.7.0-20070801
  compiled by Lucene's `DictionaryBuilder` -- zlib-compressed (9.2 MB ->
  4.6 MB in the repository), plus its `stopwords.txt`, `stoptags.txt` and
  `romaji_map.txt` (Apache-2.0, Lucene's). IPADIC's terms (NAIST, with
  ICOT's free-software conditions) allow redistribution, original or
  modified, provided the no-warranty section accompanies it; Lucene ships
  the data under them, and `NOTICE` reproduces Lucene's NOTICE with the full
  mecab-ipadic notice. Licence-clean, so vendored: `JapaneseTokenizer`'s
  default constructors work as Java's do; every dictionary class also loads
  a caller's files (`from_paths`). The fixture corpus
  (`fixtures/corpus/analysis-japanese*.txt`) is written for this project,
  apart from short public-domain literary quotations (Sōseki, Bashō, the
  Heike Monogatari, Sei Shōnagon, Miyazawa Kenji).

- **Test data from third parties** (M9). `fixtures/corpus/real_polygons.z`,
  the Tessellator's real-world polygon corpus (`scripts/gen-tessellator-corpus.py`),
  holds:
  - polygons from **Apache Lucene 10.5.0**'s own tests (Apache-2.0):
    `lucene-test-framework`'s geo resources and `TestTessellator.java`'s
    inline shapes;
  - **Natural Earth** v5.1.2 vector data (public domain:
    naturalearthdata.com/about/terms-of-use, "No permission is needed to use
    Natural Earth"), as exported to GeoJSON by
    `github.com/nvkelso/natural-earth-vector`.
  - Test data only: nothing under `fixtures/` ships in the plugin zip or the
    native library. `NOTICE` names both anyway.

- **Unicode data** (M11). `crates/lucene-analysis/src/java_character_tables.rs`
  (the general category of every code point, the decimal digits, `White_Space`)
  is generated from the **Unicode Character Database 16.0.0** by
  `crates/lucene-analysis/tools/gen_java_character_tables.py`, reading it
  through PyPI's `unicodedata2` 16.0.0 (Apache-2.0 code over the UCD files).
  The data is under the Unicode licence (Unicode-3.0, allowed above) and
  ships in the native library; `NOTICE` names it. It is deliberately *not*
  extracted from the JDK (GPL-2.0 with the Classpath Exception): Java's
  `Character.getType` is specified as the UCD's `General_Category`, and the
  generated tables equal JDK 25's code point for code point.
- **Hunspell charset tables** (M11),
  `crates/lucene-analysis/src/hunspell/charsets.rs`: the byte-to-character
  mapping of seven single-byte charsets (ISO/IEC 8859-2, -7, -13, -15, KOI8-R
  of RFC 1489, windows-1251, TIS 620) and the list of charset names the JDK
  accepts. Each mapping is fixed by its standard; they are read off the JDK
  by `crates/lucene-analysis/tools/GenHunspellCharsets.java` (written for
  this project) only so that the port decodes exactly as Lucene does on the
  JDK, including which bytes the JDK refuses. No JDK code is copied. Ships in
  the native library.
- **The analysis-common corpus** (M11), `fixtures/corpus/analysis-common.txt`
  (and its frozen copy `snowball-seed.txt`, which seeds `GenSnowball`),
  is written for this project (Apache-2.0); the few well-known pangrams in it
  are short phrases in common use. Test data only.
- **The synonym, classic, language and compound fixtures' inputs** (M11), `fixtures/corpus/analysis-classic.txt`, `analysis-compound.txt`, `hyphenation-test.xml` (a toy grammar, not a real language's patterns), `analysis-lang.txt`, `analysis-synonym.txt`,
  `synonyms-solr.txt` and `synonyms-wordnet.txt`, are written for this project
  (Apache-2.0); the WordNet file uses only WordNet's `wn_s.pl` line *format*,
  none of its synsets. Test data only.
- **The factory fixtures' inputs** (M11 T11.7), `fixtures/corpus/analysis-factories.txt`,
  `analysis-factories.conf` and the word, rule and mapping files under
  `analysis-factories/`, are written for this project (Apache-2.0). Test data only.

## What ships, and under what

- **The plugin zip** holds three things:
  - the plugin's own jar;
  - `plugin-descriptor.properties`;
  - `native/<platform>/liblucene_ffi.so`.
- **No third-party jar is bundled:** OpenSearch and Lucene are `compileOnly`,
  supplied by the node at run time (both Apache-2.0).
- **The native library** statically links the workspace and these 26 crates
  (the resolved graph of normal and build dependencies at the time of the
  audit; `scripts/check-licences.py --all` prints the current one):

| Crate | Version | Licence (declared) | Taken as |
|---|---|---|---|
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | 0BSD |
| aho-corasick | 1.1.5 | Unlicense OR MIT | Unlicense (via `regex`, M11) |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 | MIT |
| crc32fast | 1.5.2 | MIT OR Apache-2.0 | MIT |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 | MIT |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 | MIT |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 | MIT |
| either | 1.18.0 | MIT OR Apache-2.0 | MIT |
| libc | 0.2.189 | MIT OR Apache-2.0 | MIT |
| lz4_flex | 0.11.6 | MIT | MIT |
| memchr | 2.8.3 | Unlicense OR MIT | Unlicense (via `regex`, M11) |
| memmap2 | 0.9.11 | MIT OR Apache-2.0 | MIT |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 | MIT |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | MIT (build time only) |
| quote | 1.0.47 | MIT OR Apache-2.0 | MIT (build time only) |
| rayon | 1.12.0 | MIT OR Apache-2.0 | MIT |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | MIT |
| regex | 1.13.1 | MIT OR Apache-2.0 | MIT (M11: `java.util.regex` for analysis-common's `pattern` package) |
| regex-automata | 0.4.18 | MIT OR Apache-2.0 | MIT |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 | MIT (also direct, M11: parses the `java.util.regex` patterns) |
| syn | 2.0.119 | MIT OR Apache-2.0 | MIT (build time only) |
| thiserror | 1.0.69 | MIT OR Apache-2.0 | MIT |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 | MIT (build time only) |
| twox-hash | 2.1.4 | MIT | MIT |
| unicode-ident | 1.0.26 | (MIT OR Apache-2.0) AND Unicode-3.0 | MIT AND Unicode-3.0 (build time only) |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | MIT |

All are permissive and compatible with redistribution under Apache-2.0.

The list M6's work order named has drifted since it was written:
- **no longer used:** `jni` (the bridge moved to Panama FFM) and `zstd`;
- **test and benchmark only:** `proptest` and `criterion`.

## Tests, benchmarks and tooling

- **Test and benchmark dependencies:** 76 more crates (102 in all). They never
  reach the shipped library.
  - All are permissive: MIT, Apache-2.0, BSD-2-Clause, Unlicense, Zlib.
  - Two, `r-efi` 5 and 6, offer LGPL-2.1-or-later *as one alternative* beside
    MIT and Apache-2.0; MIT is the one taken.
- **The Java fixture generators and verifiers** (`fixtures/src`, `scripts/verify-*.sh`)
  run against Lucene 10.5.0's jars, Spatial4j 0.8 and s2-geometry-library-java
  1.0.0 (for spatial-extras) and OpenSearch 3.8.0's distribution, all
  Apache-2.0. None of them is redistributed; the scripts fetch them.

## Prior art

- **Tantivy** (MIT) is prior art to study, never a dependency (`PLAN.md` §1).
  No Tantivy crate is in `Cargo.lock`.
- The single mention of it in the source is a comment in
  `crates/lucene-codecs/src/regexp.rs` crediting the *idea* of caching a
  compiled pattern across queries; the code there is this port's own.
- The port's reference is Lucene's Java source, file by file, as
  the parity ledger (`docs/parity.md` and `docs/parity/`) records.
- **Hunspell test dictionaries** (M11 T11.4). Lucene's own Hunspell test
  dictionaries (`lucene/analysis/common/src/test/.../hunspell/*.aff`, `.dic`)
  are not redistributed: most are copied from Hunspell's test suite
  (MPL-1.1/GPL-2.0/LGPL-2.1 tri-licence) with no per-file licence to rely on.
  The fixtures run on dictionaries written for this project instead
  (`fixtures/corpus/hunspell/`, Apache-2.0), each exercising named features.
  Test data only.
