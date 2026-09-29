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

## What ships, and under what

- **The plugin zip** holds three things:
  - the plugin's own jar;
  - `plugin-descriptor.properties`;
  - `native/<platform>/liblucene_ffi.so`.
- **No third-party jar is bundled:** OpenSearch and Lucene are `compileOnly`,
  supplied by the node at run time (both Apache-2.0).
- **The native library** statically links the workspace and these 21 crates
  (the resolved graph of normal and build dependencies at the time of the
  audit; `scripts/check-licences.py --all` prints the current one):

| Crate | Version | Licence (declared) | Taken as |
|---|---|---|---|
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 | 0BSD |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 | MIT |
| crc32fast | 1.5.2 | MIT OR Apache-2.0 | MIT |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 | MIT |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 | MIT |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 | MIT |
| either | 1.18.0 | MIT OR Apache-2.0 | MIT |
| libc | 0.2.189 | MIT OR Apache-2.0 | MIT |
| lz4_flex | 0.11.6 | MIT | MIT |
| memmap2 | 0.9.11 | MIT OR Apache-2.0 | MIT |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 | MIT |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | MIT (build time only) |
| quote | 1.0.47 | MIT OR Apache-2.0 | MIT (build time only) |
| rayon | 1.12.0 | MIT OR Apache-2.0 | MIT |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | MIT |
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

- **Test and benchmark dependencies:** 81 more crates (102 in all). They never
  reach the shipped library.
  - All are permissive: MIT, Apache-2.0, BSD-2-Clause, Unlicense, Zlib.
  - Two, `r-efi` 5 and 6, offer LGPL-2.1-or-later *as one alternative* beside
    MIT and Apache-2.0; MIT is the one taken.
- **The Java fixture generators and verifiers** (`fixtures/src`, `scripts/verify-*.sh`)
  run against Lucene 10.5.0's jars and OpenSearch 3.8.0's distribution, both
  Apache-2.0. None of them is redistributed; the scripts fetch them.

## Prior art

- **Tantivy** (MIT) is prior art to study, never a dependency (`PLAN.md` §1).
  No Tantivy crate is in `Cargo.lock`.
- The single mention of it in the source is a comment in
  `crates/lucene-codecs/src/regexp.rs` crediting the *idea* of caching a
  compiled pattern across queries; the code there is this port's own.
- The port's reference is Lucene's Java source, file by file, as
  `docs/parity.md` records.
