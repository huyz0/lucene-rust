# lucene-rust: Porting Apache Lucene to Rust with FFI integration into OpenSearch

This is the master plan for porting Apache Lucene (Java, ~1.5M LOC in `lucene/core` alone)
to Rust, exposed over a JNI/FFI boundary so OpenSearch (JVM) can use it as a drop-in
engine for the hot paths. Source of truth for the Java side: `/home/tuong/work/lucene`.
OpenSearch checkout: `/home/tuong/work/OpenSearch`.

---

## 0. Strategy and non-goals

### Guiding decisions

1. **Port by on-disk format, not by class hierarchy.** The contract that matters is the
   Lucene index format (segments, postings, doc values, stored fields, points, vectors).
   If lucene-rust reads and writes bit-identical (or at least format-compatible) segments
   for one pinned codec version (e.g. `Lucene103`), Java Lucene and lucene-rust can
   coexist on the same index. This is the property that makes incremental adoption in
   OpenSearch possible.
2. **Pin one Lucene version.** Pick the version OpenSearch `main` ships (check
   `buildSrc/version.properties` → `lucene = ...`). Do not chase Lucene trunk during the
   port. Backward-codecs support is explicitly out of scope for v1; old segments are
   handled by the Java engine until force-merged.
3. **Read path first, write path second.** A read-only Rust searcher over
   Java-written segments delivers value early (query CPU is the OpenSearch hot path),
   is verifiable byte-for-byte against Java results, and avoids the hardest correctness
   risks (merge policy, deletes, transactional commits) until the foundations are proven.
4. **FFI = JNI via the `jni` crate + a thin handle-based C ABI.** OpenSearch is JVM;
   "FFI" concretely means a `cdylib` loaded by a JNI wrapper. Design the Rust-side API
   as a C ABI (opaque handles, no Rust types across the boundary) so the same library
   also works from Panama/FFM (`java.lang.foreign`), which is the better long-term
   binding (JDK 21+, which OpenSearch already requires).
5. **Differential testing is the correctness backbone.** Every milestone gates on
   comparing lucene-rust output against Java Lucene on the same input: same segments,
   same queries, same top-k docs and scores (within float tolerance), same term stats.

### Non-goals (v1)

- No port of: `luke`, `benchmark`, `demo`, `monitor`, `replicator`, `expressions`,
  `classification`, `spatial3d`, `spatial-extras`, `queryparser` (OpenSearch has its own
  query DSL; we accept pre-parsed query trees over FFI).
- No backward-codecs (old index versions).
- No index sorting, no soft-deletes semantics beyond what OpenSearch requires
  (OpenSearch **does** require soft-deletes for replication — this lands in Phase 6,
  it is required before write-path integration, just not before read-path integration).
- No scoring pluggability beyond BM25 + constant score + a similarity trait.

---

## 1. Architecture: crate layout

Cargo workspace, one crate per Java module boundary (roughly):

| Crate | Java counterpart | Contents |
|---|---|---|
| `lucene-util` | `o.a.l.util`, `o.a.l.internal` | BitSet/FixedBitSet, BKD utils, FST, packed ints, `BytesRef`, LSB radix sorter, SIMD kernels |
| `lucene-store` | `o.a.l.store` | `Directory`, `IndexInput/Output`, mmap (memmap2), buffered/NIOFS, checksums, locking |
| `lucene-codecs` | `o.a.l.codecs` | Codec trait + the one pinned default codec: postings (PFOR-delta), doc values, stored fields (LZ4/zstd blocks), points (BKD), KNN vectors (HNSW), norms, live docs, segment infos |
| `lucene-index` | `o.a.l.index` | SegmentReader/DirectoryReader, Terms/PostingsEnum, IndexWriter, DWPT, merge policy/scheduler, deletes, commits |
| `lucene-analysis` | `o.a.l.analysis` + `analysis/common` subset | TokenStream trait, StandardTokenizer (from Unicode segmentation), lowercase/stop/ascii-folding; everything else stays JVM-side long-term |
| `lucene-search` | `o.a.l.search` | Query/Weight/Scorer, Boolean (WAND/BMW), term/phrase/points ranges, collectors, BM25, ConstantScore, MatchAll |
| `lucene-core` | — | Facade crate re-exporting the above; the "public API" |
| `lucene-ffi` | — | `cdylib`: C ABI + JNI export layer, handle registry, panic → error-code mapping |
| `opensearch-plugin/` | — | Java: OpenSearch engine plugin (`EngineFactory`) + JNI binding class, native lib loading, CI packaging |

Rationale: matches Lucene's own dependency DAG (`util ← store ← codecs ← index ← search`),
lets phases parallelize, and keeps `lucene-ffi` as the only `unsafe`-heavy crate.

Key crates from the ecosystem to use rather than re-port: `memmap2` (mmap directory),
`zstd`/`lz4_flex` (stored fields), `crc32fast` (checksums), `unicode-segmentation`
(StandardTokenizer is UAX#29), `rayon` (concurrent merge/search), `jni` (JNI layer).
Study but do not depend on: **Tantivy** (license-compatible, MIT — prior art for nearly
every component; where a design question comes up, check how Tantivy solved it, but its
index format is NOT Lucene-compatible, which is exactly what we need to be).

---

## 2. Phases

### Phase 1 — Foundations: `lucene-util` + `lucene-store` (est. 6–8 weeks)

Port order within phase:

1. `BytesRef`/`BytesRefBuilder` → mostly `&[u8]`/`Vec<u8>` idioms; keep a thin newtype
   where ordering semantics (unsigned byte compare) matter.
2. `DataInput/DataOutput` primitives: vint/vlong, zigzag, `readGroupVInt` (group-varint —
   note Lucene 9.9+ uses this in postings), string (Java-modified-UTF8 **only** where the
   format demands; segment metadata uses standard UTF-8).
3. `Directory` + `IOContext`, `FSDirectory`, `MMapDirectory` (memmap2, madvise hints
   mirroring Java's `ReadAdvice`), `IndexInput` slicing/cloning — model clones as cheap
   offset-cursors over an `Arc<Mmap>`.
4. Checksums: `BufferedChecksumIndexInput` (CRC32), footer/header verification
   (`CodecUtil.checkHeader/checkFooter` — get the magic numbers and version framing exact).
5. `FixedBitSet`, `SparseFixedBitSet`, `LongBitSet`, `PackedInts`/`DirectWriter`/
   `DirectMonotonicReader` (doc values depend on these heavily — port with exhaustive
   round-trip tests against Java-generated fixtures).
6. FST reader (writer can wait until Phase 5): the terms index (`.tip`) is an FST.
   This is one of the two hardest data structures in the port (the other is BKD).
7. Locking: `NativeFSLockFactory` semantics via `flock`/`OpenOptions`.

**Progress (task #14):** item 3's `IndexInput` slicing/cloning landed —
`SliceInput::slice_input(description, offset, length)` in `lucene-store/src/data_input.rs`
returns a new, independent-file-pointer `SliceInput` over `[offset, offset+length)` of the
callee's own addressing (slice-of-a-slice supported, offsets always relative to the
callee); `Clone` (already derived, since `SliceInput` is just `(&[u8], usize)`) gives the
same independent-pointer duplicate as Java's `clone()`. This was deferred once already
(originally task #9, "no real caller exists yet") — the caller motivating it now is
segment merging (task #15, not yet started), which will need its own cursor per
source-segment sub-range read out of a shared file. `compound_format.rs::open_input` was
refactored onto this primitive (previously a hand-rolled `data.get(offset..offset+len)`
byte-range) since it's the same shape and gets a real cursor instead of a raw slice for
free. See `docs/parity.md`'s `IndexInput.slice`/`.clone()` row.

**Test harness (build in this phase, use forever):** a Java "fixture generator" module in
this repo (`fixtures/`, Gradle, depends on the pinned Lucene) that writes: raw
packed-ints blobs, FSTs, and small single-segment indexes with known content. Rust tests
read fixtures and assert exact decoded values. Plus proptest round-trips for every
encoder we do implement.

**Exit criteria:** open a Java-written segment directory, verify all file checksums,
decode `segments_N` and `.si` files, dump the terms index of a text field.

### Phase 2 — Read-only codec: decode the pinned default codec (est. 10–14 weeks)

The heart of the port. For the pinned codec (e.g. `Lucene103Codec`), implement readers for:

1. **SegmentInfos / FieldInfos** (`segments_N`, `.si`, `.fnm`) — already started in P1.
2. **Postings** (`Lucene103PostingsFormat`: `.tim/.tip/.tie` terms dict + FST index,
   `.doc/.pos/.pay`): block-decoded FOR/PFOR (the `ForUtil`/`PForUtil` generated code —
   port the generator output, then vectorize with `std::simd` or explicit AVX2 behind
   feature flags; scalar fallback first), skip data (impacts!), `PostingsEnum` with
   positions/offsets/payloads.
3. **Impacts** (`ImpactsEnum`): required for WAND/MAXSCORE in Phase 3 — do not skip.
4. **Doc values** (`.dvd/.dvm`): numeric (direct-monotonic + gcd/table compression),
   sorted/sorted-set (term dicts + ordinals), binary, sorted-numeric. OpenSearch
   aggregations live on these — treat as first-class, not an afterthought.
5. **Stored fields** (`.fdt/.fdx/.fdm`): LZ4/zstd(actually DEFLATE in BEST_COMPRESSION
   mode — check pinned version) block decompression, prefetch-friendly random access.
6. **Norms** (`.nvd/.nvm`).
7. **Points / BKD** (`.kdd/.kdi/.kdm`): BKD tree reader + `intersect` visitor. Hard;
   needed for numeric/date range queries which OpenSearch uses constantly. Read
   side and a write side supporting one dimension (any number of leaves,
   multi-leaf packed-index tree construction included) are done -- see
   `docs/parity.md`'s points row. Multi-dimension write (`LatLonPoint`-style)
   remains a future slice.
8. **Live docs** (`.liv`) and per-commit deletes/DV-updates generations.
9. **KNN vectors** (`.vec/.vex/.vem`, HNSW): schedule **last within the phase** and be
   willing to punt to Phase 8 — big, self-contained, and OpenSearch's k-NN often uses
   its own engines (faiss/nmslib) anyway.
10. `SegmentReader` + `DirectoryReader.open(commit)` + `MultiTerms`/leaf abstraction.

**Verification:** `CheckIndex`-equivalent in Rust (`lucene-rust check <dir>`), run against
indexes generated by the Java fixture generator from randomized documents (use Lucene's
own `RandomIndexWriter`-style randomization in the fixture generator, many codec
parameter combinations). Golden test: dump every posting, every doc value, every stored
doc from both sides and diff.

**Exit criteria:** Rust `CheckIndex` passes on randomized Java-written indexes including
deletes and DV updates; full-corpus dump diff is empty on a real dataset
(e.g. a Wikipedia sample indexed by Java Lucene).

### Phase 3 — Search: queries, scoring, collectors (est. 8–10 weeks)

**Progress so far:** a first, deliberately narrow slice landed in
`lucene-search` — single-segment `TermQuery` **matching** (no scoring):
`query::TermQuery` (field + exact term) executed by `search_term_query`
against an already-opened `blocktree::BlockTreeFields` (+ optional `.doc`
`DocInput`, optional `.liv`-derived `FixedBitSet`), feeding matching live doc
IDs to a `Collector` (`VecCollector`/`CountCollector`). Differential-tested
against the real `IndexWriter`-produced fixture in
`fixtures/data/blocktree_index/` (`crates/lucene-search/tests/term_query_fixtures.rs`).
Deliberately does **not** yet cover: relevance scoring/`Similarity` (item 2
below), dynamic pruning/`TopScoreDocCollector` (items 4–5), or multi-segment
`IndexSearcher`/`IndexReader` federation (item 6) — see `docs/parity.md`'s
`lucene-search` section for the exact scope line and the design rationale (no
`Weight`/`Scorer` trait hierarchy yet either — a single query type and a
single segment gave it no second implementation to justify the abstraction).

A second slice landed `BooleanQuery` **matching** (still no scoring): flat
`must`/`should`/`must_not: Vec<TermQuery>` clauses
(`query::BooleanQuery`/`search_boolean_query`), built on new
`docid_set::{Conjunction, Disjunction, Excluding}` merge combinators — plain
`Iterator<Item = i32>` adapters over each clause's doc-ID list (leapfrog
conjunction, min-scan disjunction, AND-NOT exclusion), not a bespoke
`DocIdSetIterator` trait (see that module's doc comment for why). Matching
semantics (a pure-`MUST_NOT` query matches nothing; `SHOULD` is non-filtering
once `MUST` exists) were verified against real `BooleanQuery.rewrite()` source
rather than assumed. Differential-tested in
`crates/lucene-search/tests/boolean_query_fixtures.rs` against the same
fixture segment. Still deferred at that point: nested `BooleanQuery` clauses,
`minimumNumberShouldMatch`, and relevance scoring.

A third slice (task #13) landed **BM25 relevance scoring**: `similarity.rs`
ports the pure `BM25Similarity` formula (`idf`/`tfNorm`/`score`, defaults
`k1 = 1.2`/`b = 0.75`, verified against `BM25Similarity.java` and independently
hand-computed in its unit tests), `search_term_query_scored`/
`search_boolean_query_scored` wire it into a new `ScoringCollector` trait
(deliberately *not* a breaking change to the existing `Collector` trait — see
`collector.rs`'s module doc), and `TopDocsCollector` is the ported
`TopScoreDocCollector`-equivalent (tie-break verified against real
`HitQueue.lessThan`: lower doc ID wins a score tie). **Important caveat, item
2 below is only partially closed**: this port has no norms *reader* yet (only
the `.nvd`/`.nvm` write side exists, see that format's `docs/parity.md` row),
so every score currently uses a constant field-length substitution
(`fieldLength == avgFieldLength == 1.0`) instead of this segment's real
per-document lengths — scores are internally consistent and correctly ordered
but **not** expected to numerically match real Lucene's BM25 output for the
same query. Closing that gap (a norms reader + a Java-side BM25-score fixture
generator for true cross-engine verification) remains future work. Verified
in `crates/lucene-search/tests/scoring_fixtures.rs`. The items below remain as
originally scoped except where superseded above.

A fourth slice (task #19) landed **`PhraseQuery` matching**, exact adjacent
positions only (`slop == 0`): `query::PhraseQuery { field, terms: Vec<Vec<u8>> }`
implicitly places each term at consecutive positions `0..terms.len()`, no
`PhraseQuery.Builder.add(Term, int position)`-style arbitrary/sloppy positions.
`search_phrase_query` computes the doc-level conjunction across every term first
(reusing `docid_set::Conjunction`, since phrase match implies term match), then
checks position alignment per candidate doc via a new `phrase_matches_in_doc`
function — every position in the first term's list is a candidate base `p`,
checked against every other term's sorted position list via binary search for
`p+i` (a straightforward candidate-and-check, not real `ExactPhraseScorer`'s
stateful per-postings merge — this port's positions are already fully
materialized per doc by the existing `postings::read_positions`/
`FieldTerms::positions`, so there's no lazy iterator state to replicate). A
single-term "phrase" degenerates to a plain `search_term_query` call (never
needs an opened `.pos` file); an empty `terms` list matches nothing (mirrors
real `PhraseQuery.Builder.build()`'s `MatchNoDocsQuery` for zero terms); a
missing term matches nothing, not an error; a repeated term ("the the") needs
no special-casing. **No relevance scoring for phrase queries yet** — BM25
phrase scoring needs a per-doc "phrase freq" concept from the same alignment
walk, deferred alongside a `ScoringCollector` sibling for `search_phrase_query`.
Differential-tested in `crates/lucene-search/tests/phrase_query_fixtures.rs`,
reusing the existing `pos` field already in `fixtures/data/blocktree_index/`
(no fixture generator changes needed — its real occurrences already have an
adjacent pair in one doc and a non-adjacent/absent pair in another). See
`docs/parity.md`'s new `PhraseQuery`/`ExactPhraseScorer` row for full detail.

1. Traits: `Query → Weight → Scorer/ScorerSupplier`, `DocIdSetIterator`,
   `TwoPhaseIterator`, `BulkScorer`. Use enums where the closed set allows
   (DISI is called per-doc — keep it monomorphizable; `Box<dyn>` only at Weight level).
2. Similarity: BM25 (exact same float math as Java — same order of operations, `f32`
   where Java uses float, precomputed norm cache tables) + constant score. **Formula
   ported** (`lucene-search/src/similarity.rs`, task #13) — **norms reading/precomputed
   per-doc norm cache tables still not ported**, so the formula currently runs on a
   constant field-length substitution rather than real per-document norms; see the
   note above and `docs/parity.md`'s BM25Similarity/norms rows.
3. Queries, in order: `MatchAllDocs`, `TermQuery`, `BooleanQuery` (conjunction DISI,
   disjunction heap, minimum-should-match), `PointRangeQuery` (BKD intersect),
   `PhraseQuery` (exact + sloppy), `TermInSetQuery`, `PrefixQuery`/`WildcardQuery`
   (needs Levenshtein/automaton machinery — port `o.a.l.util.automaton` here; consider
   the `fst`/`regex-automata` crates for internals but keep Lucene semantics),
   `FunctionScore`-shaped hooks deferred.
4. Dynamic pruning: `WANDScorer`/block-max, `ImpactsDISI`, `MaxScoreCache`. This is
   where Lucene's search performance comes from; without it the port is not competitive.
5. Collectors: `TopScoreDocCollector` (with after/searchAfter), `TotalHitCountCollector`,
   early termination, `CollectorManager` + intra-query concurrency via rayon over leaves
   (mirror Lucene's leaf-slice model). **`TopScoreDocCollector`'s core (fixed `top_n`,
   no `searchAfter`) ported** as `collector::TopDocsCollector` (task #13, sorted-`Vec`
   first cut, not a binary heap — see `docs/parity.md`) — `searchAfter`, early
   termination, and `CollectorManager`/rayon concurrency remain unported.
6. `IndexSearcher` facade + query cache (LRU on filter bitsets, like `LRUQueryCache`) —
   cache can be a later sub-milestone.

**Verification:** differential query harness — a Java CLI (in `fixtures/`) and Rust CLI
that both run a query file against the same index and emit `(docid, score)` top-1000;
diff with score tolerance 1e-5 relative and **exact** doc-set equality. Fuzz with
randomly generated boolean trees over randomized indexes. Also compare `explain()`-level
term stats for a sample.

**Exit criteria:** differential harness green over 100k randomized queries on randomized
indexes; luceneutil-style benchmark (wikimedium terms/phrases/booleans/ranges) shows
Rust ≥ Java on p50 and p99 for the ported query types.

### Phase 4 — FFI layer + read-only OpenSearch integration (est. 6–8 weeks, overlaps P3)

**Progress (task #20):** the first real FFI surface now exists in `lucene-ffi`,
wrapping `lucene-search`'s existing `search_term_query`/`search_boolean_query`/
`search_phrase_query` (unscored matching only, no BM25 scoring yet) behind opaque
`u64` handles: `ffi_open_directory`/`ffi_close_directory` (a real `FsDirectory`),
`ffi_open_segment`/`ffi_close_segment` (one segment's term dictionary plus
optional `.doc`/`.pos` postings files, from already-known file names/segment
ID/suffix/`maxDoc` — no `.si`/`segments_N` parsing on the Rust side yet),
`ffi_search_term_query`/`ffi_search_boolean_query`/`ffi_search_phrase_query`
(each collecting matches into a results handle via a plain
`lucene_search::VecCollector`, entirely Rust-side), and
`ffi_results_len`/`ffi_results_copy`/`ffi_close_results` to read them back out.
Every exported function is `catch_unwind`-guarded (`error::guard`) and returns
an `FfiStatus` code; `ffi_get_last_error_message` reads the thread-local
last-error string. See `docs/parity.md`'s new `## lucene-ffi` section for the
exact surface and what's deferred (scored queries, `.liv`/`.pay`, the unified
`.si`-driven segment-open entry point, the JNI wrapper class itself, and the
query-tree serialization / OpenSearch plugin work below, all still not
started). `crates/lucene-ffi/src/*.rs` is unit-tested (≥95% line coverage per
file) calling the exact exported `extern "C" fn` entry points against the real
`fixtures/data/blocktree_index/` fixture, including stale/closed-handle
rejection and a genuine caught-panic-surfaces-as-a-status-code test.

**C ABI design (`lucene-ffi`):**

- Opaque `u64` handles (generation-tagged slotmap) for: `Directory`, `IndexReader`,
  `IndexSearcher`, `Query`, prebuilt `TopDocs` result buffers. No Rust pointers cross
  the boundary; no callbacks from Rust into Java in v1 (collectors run entirely in Rust).
- All calls return `i32` status; results via out-buffers. `catch_unwind` at every entry
  point → error code + last-error message TLS slot. **A Rust panic must never unwind
  into the JVM.**
- Query representation across the boundary: a compact binary query tree (flatbuffer or
  hand-rolled tag-length-value — benchmark both; avoid protobuf/JSON per-query cost).
  OpenSearch already builds Lucene `Query` objects; we add a
  `QueryVisitor`-based serializer on the Java side for the supported subset, with a
  "unsupported → fall back to Java engine" escape hatch **per query**.
- Results: top-k `(doc, score)` + total hits written into a Java-owned direct
  ByteBuffer / MemorySegment to avoid JNI array copies. Stored-field fetch as a separate
  call (doc → bytes of the `_source` field).
- Two binding front-ends over the same C ABI: JNI (`jni` crate, works everywhere) and a
  Panama FFM `MethodHandle` layer (preferred at runtime on JDK 21+; measure — FFM
  downcalls are typically faster and avoid JNI-local-ref churn).

**OpenSearch plugin (`opensearch-plugin/`):**

- An `EnginePlugin` providing a custom `EngineFactory`. First deliverable: a
  **shadow-read mode** — the plugin opens the same shard directory read-only in Rust on
  each refresh (`DirectoryReader` handle refreshed on Lucene commit/refresh points),
  serves eligible search requests through Rust, everything else through the normal
  engine. Deletes visible via `.liv` per commit; near-real-time (in-memory) segments are
  NOT visible to Rust in this mode — acceptable for search-after-refresh semantics only
  if the shard is search-only/replica or `refresh` forces a commit; otherwise route
  NRT-sensitive requests to Java. Document this loudly.
- Native library packaging: per-platform `cdylib` (linux-x64/arm64 gnu, macOS arm64)
  inside the plugin zip, extracted and loaded at plugin init; crash-safety review
  (a segfault in Rust kills the whole node — this is why handle validation and no-raw-
  pointers matter).
- Benchmark with OpenSearch Benchmark (`nyc_taxis`, `pmc`, `big5` workloads), Rust vs
  Java engine on the same shards.

**Exit criteria:** an OpenSearch node serving term/bool/range/match queries for a real
workload through lucene-rust in shadow-read mode, with automatic per-query fallback,
and a benchmark report.

### Phase 5 — Write path: analysis chain + indexing (est. 12–16 weeks)

**Progress so far:** every single-segment write primitive (stored fields, `FieldInfos`,
`SegmentInfo`, points, term vectors, doc values, norms, live docs, compound format, real
LZ4 compression, and the `segments_N` commit file) lands one complete, real-Lucene-openable
segment — verified end-to-end by
`crates/lucene-index/examples/write_segment_infos_fixture.rs` +
`fixtures/src/VerifySegmentInfos.java` (`DirectoryReader.open`). On top of that,
`lucene-index/src/segment_writer.rs::flush_stored_only_segment` is a small, deliberately
minimal "flush an in-memory batch of documents to one new segment" building block: call it
more than once against the same `Directory` with distinct segment names, collect the
resulting `SegmentCommitInfo`s, and pass all of them to one `segment_infos::write` call —
that's a real multi-segment commit. Proven by
`crates/lucene-index/examples/write_multi_segment_commit_fixture.rs` (two independent
flushes, `_0`/`_1`, one `segments_N`) opened successfully by real Lucene's
`DirectoryReader.open` via `fixtures/src/VerifySegmentInfos.java` (unchanged — it was
already segment-count-agnostic). This did **not** require any change to
`segment_infos::write`/`parse`: `SegmentInfos::segments` was already `Vec<SegmentCommitInfo>`
with a plain loop on both the encode and decode side, so describing N segments in one
commit was already mechanical before this slice — the actual gap closed here was the
reusable per-segment flush helper, not the commit format. Still stored-fields-only (no
indexed fields), and still missing everything below: no RAM accounting/flush-triggering, no
merging, no deletes/updates during indexing, no NRT, no concurrency, and no unified
multi-segment read path on this port's own side (real Lucene's reader federates the
Rust-written segments fine; this port's own `SegmentReader`/`IndexSearcher` does not yet).

**Progress (task #15):** `lucene-index/src/merge.rs::merge_stored_only_segments` merges N
already-flushed, stored-fields-only segments into one new stored-fields-only segment: reads
each source's `FieldInfos` + `Document`s back out (via `stored_fields::open`/`.document()`,
already read-only ported in Phase 2), drops non-live docs per an optional per-source
`FixedBitSet` (via `live_docs::parse`), reconciles field numbering across sources by field
name (`reconcile_field_numbers`, the merge-time slice of `FieldInfos.FieldNumbers` — a
segment's own field number is local to that segment, so two sources naming the same field
differently is a real case, not a hypothetical one), renumbers surviving docs contiguously by
concatenating sources in order, and hands the result to the existing
`flush_stored_only_segment` to write the merged `.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si`. No new
read-side decoders were needed — `stored_fields`'s reader and `live_docs::parse` already
existed from Phase 2. Still missing, and still item 6 below: `TieredMergePolicy`-style
merge *selection* (this is caller-picks-the-sources merge *execution* only), background/
concurrent merging, merge-time codec upgrades, and merging anything beyond stored fields
(doc values/points/norms/term vectors/postings) — none of those have a write-side caller
producing a full segment yet in this port, so there's nothing there to merge.

**Progress (task #16):** `lucene-index/src/deletes.rs` adds the doc-ID-level delete
mechanics real Lucene's `ReadersAndUpdates.writeLiveDocs` performs: `mark_deleted` clears
given doc IDs out of a segment's live-docs bitset (from "all live" when `del_gen == -1`, or
from an existing bitset otherwise), idempotently (re-deleting an already-deleted doc doesn't
double-count) and with a hard `Err` on an out-of-range doc ID; `apply_deletes` wraps that
around a `SegmentCommitInfo`, writes the resulting bitset as that segment's next-generation
`.liv` file via the existing `live_docs::write`, and returns an updated `SegmentCommitInfo`
with `del_gen` incremented (`-1` -> `1` first time, else `+1`) and `del_count` bumped by the
newly-deleted count. This task also **establishes the `.liv` filename convention** this port
was missing: `deletes::liv_file_name` produces `_<segment>_<delGen in base36>.liv`, matching real
Lucene's `IndexFileNames.fileNameFromGeneration` (base-36-encoded generation, same convention
`live_docs.rs`'s own index-header suffix already uses) — `del_gen` was previously tracked in
`segment_infos.rs` purely as an opaque integer, with no filename derived from it anywhere.
**Explicitly deferred, and why:** delete-by-term/delete-by-query resolution (real
`BufferedUpdates`/`ReaderPool` — which doc IDs a `Term`/`Query` names, resolved by running
it against each segment's live postings/doc-values reader) is a genuinely separate, larger
feature this task does not attempt: this port's write path has no live per-segment reader
wired to a query executor across segments yet (search and index/write are still unconnected
halves), so there's nothing correct to resolve against, and a fake in-memory-scan version
wouldn't match real generation-ordered `BufferedUpdates` semantics. `updateDocument` is
deferred for the same reason: it's defined in real Lucene as delete-by-term + `addDocument`,
so without delete-by-term a faithful version doesn't exist — a doc-ID-keyed "replace" helper
would silently diverge from real `Term`-keyed semantics, so none was added; a caller needing
that shape today composes `apply_deletes` with a separate `flush_stored_only_segment`/merge
call by hand. See `docs/parity.md`'s new row for full detail, including test coverage
(99.5% lines).

1. `lucene-analysis`: `TokenStream` as an iterator-of-token-structs (skip Java's
   AttributeSource reflection design entirely — a plain
   `Token { bytes, position_increment, offset, ... }` struct), StandardTokenizer via
   UAX#29 (`unicode-segmentation`), lowercase, stop, ASCII-folding.
   **Long-term stance:** analysis mostly stays on the JVM side in OpenSearch
   (analyzers are configured there, plugins provide them). So ALSO support
   "pre-analyzed" ingestion over FFI: Java runs the analyzer, ships tokens to Rust.
   This makes the Rust analysis chain a fast path, not a compatibility burden.
2. Codec **writers** for everything Phase 2 reads: postings writer (FOR/PFOR encode,
   skip/impacts writer), FST builder (hard — port `FSTCompiler` carefully; fixture:
   build FST from same term set in Java and Rust, require byte-identical output),
   doc values writers, stored fields (LZ4 fast mode first), points (BKD writer with
   offline sort for large fields), norms, `.si`/`segments_N`/`.fnm` writers, compound
   files (`.cfs/.cfe`).
3. Indexing chain: `IndexWriter`, DWPT-per-thread with in-memory hash (bytes → postings
   builder mirroring `BytesRefHash` + parallel arrays), flush-by-RAM accounting,
   `flush()` → segment.
4. Deletes/updates: delete-by-term/query queues, `BufferedUpdates`, frozen deletes
   applied on flush; doc-values updates can be deferred to 5b.
5. Commits: `SegmentInfos` two-phase commit (pending_segments_N → fsync → rename),
   `IndexFileDeleter` refcounting, `prepareCommit/commit/rollback` (OpenSearch translog
   recovery depends on 2-phase commit + commit user-data — must be exact).
6. Merging: `TieredMergePolicy` (port the math faithfully), `ConcurrentMergeScheduler`
   on a rayon/thread pool, merge readers reusing Phase 2, optimized bulk-merge paths
   (stored fields raw-chunk copy) later.
7. NRT: `DirectoryReader.openIfChanged(writer)` — reader from uncommitted flushed
   segments + in-memory deletes. Required for real OpenSearch refresh semantics.

**Verification:** the killer test — **cross-engine round-trip**: index corpus with Rust
→ open with *Java* Lucene → Java `CheckIndex` passes and Java search results match; and
the reverse. Then interleaved: Java writes segments, Rust merges them, Java reads the
result. Randomized crash-consistency tests (kill during commit, reopen, verify).

**Exit criteria:** Java `CheckIndex` clean on Rust-written randomized indexes;
cross-engine differential search green; sustained indexing throughput ≥ Java on
luceneutil `wikimediumall` ingest.

### Phase 6 — Full OpenSearch engine integration (est. 10–14 weeks)

1. Soft-deletes + `Lucene*SoftDeletesRetentionMergePolicy` equivalent — required for
   OpenSearch peer recovery / CCR-style retention leases.
2. Engine implementation: an `InternalEngine` alternative where IndexWriter lives in
   Rust — translog interplay (OpenSearch translog stays Java; Rust engine must expose
   sequence numbers, local checkpoint, commit user data exactly as
   `InternalEngine` does), refresh → Rust NRT reader, flush → Rust commit.
3. Segment replication mode (simpler than document replication for us: only primaries
   index; replicas use the Phase 4 read path) — recommend shipping this first.
4. `_source`, `_id`, `_seq_no`, `_primary_term`, `_version` field handling parity;
   get-by-id (term lookup on `_id`) fast path over FFI.
5. Aggregations: keep OpenSearch agg framework on Java initially, feed it via
   FFI doc-value cursors (batch columnar reads into shared buffers); native Rust
   terms/histogram/stats aggs as a follow-on performance phase.
6. Ops: memory accounting bridged to OpenSearch circuit breakers (Rust side reports RAM
   usage), stats APIs, slow log hooks, graceful shutdown, panic → shard-failed (not
   node-down) hardening where possible.

**Exit criteria:** OpenSearch integration test suite (`:server` engine tests adapted +
full REST test suite for search/index/get/delete) green on the Rust engine for a
supported feature matrix; multi-day soak test with random restarts, no index corruption.

### Phase 7 — Performance and SIMD hardening (continuous, dedicated 6–8 weeks)

- Vectorize: PFOR decode, dot-product/cosine (if vectors in scope), BKD compare loops,
  bitset ops — `std::simd` with runtime feature detection (AVX2/AVX-512/NEON).
- Profile-guided: flamegraphs vs Java async-profiler on identical workloads; close gaps.
- Memory: arena allocation in DWPT (Lucene's `ByteBlockPool` design translates well),
  `IOContext`-driven madvise, optional `io_uring` experiment for cold stored-field reads.
- FFI overhead budget: < 1µs per search call overhead; batch APIs wherever per-doc
  calls could occur.

### Phase 8 — Long tail (post-v1, prioritized backlog)

KNN/HNSW (if not done in P2), highlighting (needs term vectors — add `.tvd/.tvx` codec
support), suggesters (FST-based, reuse P5 FST builder), join/grouping/facets (OpenSearch
mostly reimplements these as aggs — likely never needed), index sorting, backward-codecs.

---

## 3. Cross-cutting engineering rules

- **Unsafe policy:** `unsafe` allowed only in `lucene-util` (SIMD), `lucene-store`
  (mmap access), and `lucene-ffi` (C ABI); `#![forbid(unsafe_code)]` in all other
  crates. Miri on util/store tests.
- **Float discipline:** scoring must match Java: `f32` math in the same order; no FMA
  contraction in scoring paths (verify codegen); document every place we intentionally
  diverge.
- **Java-isms translation guide** (write `docs/porting-conventions.md` early):
  IndexReader lifecycle/refcounting → `Arc` + explicit `close` for mmap determinism;
  checked IOException → `thiserror` error enums per crate; `IndexInput.clone()` →
  cursor structs; ThreadLocal DWPT pools → per-thread slots keyed by rayon/thread id;
  Java unsigned-byte compares → `u8` slices (free win).
- **Fixture pinning:** the Java fixture generator pins the exact Lucene version; CI
  regenerates fixtures and runs the differential suites on every PR (Linux x64 + arm64).
- **Licensing:** this is a derivative work of Apache Lucene → Apache-2.0, keep NOTICE
  attribution.
- **Progress tracking:** a `docs/parity.md` matrix — every Java file in `core` mapped to
  ported / partial / not-needed / deferred, updated per PR.

## 3.5 Rust-first design: where we deliberately do NOT mirror Java

The on-disk **format** is the compatibility contract; the **in-memory design** is ours.
Rule of thumb: *port the bytes, not the objects.* Concretely:

1. **No GC-shaped object graphs.** Java Lucene's design is heavily driven by avoiding
   allocation/GC (ByteBlockPool, parallel arrays, AttributeSource reuse). In Rust we get
   deterministic memory for free, so: plain structs, arenas (`bumpalo`) per-DWPT and
   per-query where lifetimes are scoped, and struct-of-arrays layouts chosen for cache
   behavior — not to dodge a garbage collector.
2. **Monomorphization over virtual dispatch in per-doc loops.** Java pays a megamorphic
   call on every `DocIdSetIterator.nextDoc()`. We keep `dyn` only at Query/Weight level;
   scorers and DISIs are enums or generic so the per-doc loop inlines. Target: zero
   virtual calls inside `collect()` inner loops.
3. **Zero-copy reads end-to-end.** `IndexInput` over mmap yields `&[u8]` views;
   `BytesRef`-style copies only at true ownership boundaries. Stored fields / `_source`
   returned to FFI as borrowed slices into decompression buffers owned by the call
   context — never intermediate `Vec` churn like Java's `byte[]` copies.
4. **SIMD from the start, not as a retrofit.** Java's Panama vector code
   (`PanamaVectorUtilSupport`, generated `ForUtil`) fights the JIT; we write the PFOR /
   bitset / BKD-compare kernels once with `std::simd` + runtime dispatch
   (`is_x86_feature_detected!`), scalar fallback for correctness testing. The Java
   generated-code files are treated as *specs*, not sources to transliterate.
5. **Bounds checks engineered away, not ignored.** Hot decode loops operate on
   fixed-size arrays (`&[u8; 128*4]` blocks) or use iterator patterns the compiler can
   prove; `get_unchecked` only in `lucene-util` behind cargo-fuzz + Miri coverage.
6. **Thread model: ownership instead of synchronized.** Java Lucene is littered with
   locks/volatiles because everything is shared. We structure it as: immutable segment
   readers (`Arc`, lock-free), one owner per DWPT (no locking in the indexing hot path,
   channel-based handoff to flush), rayon leaf-slices for query concurrency. `Mutex`
   allowed only on control-plane state (commits, merge scheduling, deleters).
7. **io_uring / madvise as first-class citizens**, not JNI-gated afterthoughts:
   `IOContext` maps directly to `madvise` per file type (RANDOM for term dicts,
   SEQUENTIAL for merges, WILLNEED prefetch for BKD leaves), and the cold-read path is
   abstracted so an io_uring backend can drop in (Linux) without touching codecs.
8. **Async-free core.** No async runtime in the library — search/indexing are
   CPU-bound; blocking + rayon is simpler and faster. FFI callers get plain blocking
   calls (OpenSearch already dispatches on its own thread pools).
9. **Error paths off the hot path.** `Result` in APIs, but decode inner loops validate
   per-block (checksums, bounds) rather than per-value, so the happy path is
   branch-predictable.
10. **Skip Java's abstraction taxes entirely**: AttributeSource reflection (plain token
    structs), `IndexInput.clone()` object churn (Copy cursor structs over `Arc<Mmap>`),
    boxed `Integer`/autoboxing in collectors (never exists), `ThreadLocal` pools
    (scoped ownership), finalizers/`Cleaner` (Drop).

Each phase's exit criteria implicitly include: profile the ported component and confirm
it beats Java on the same workload *before* moving on — a slower "faithful" port is a
bug, and finding out early is the point of the phased structure.

## 4. Sequencing summary and effort

Rough serial-critical-path estimate (small senior team, 3–5 people who know both
Lucene internals and Rust): P1 (2mo) → P2 (3mo) → P3 (2.5mo, P4 overlaps) → P4 (1.5mo)
→ P5 (3.5mo) → P6 (3mo) → P7 (ongoing). **~14–18 months to a production-candidate
read+write engine**, with a demonstrable read-only OpenSearch speedup at ~7–8 months
(end of P4). The read-only milestone is the natural go/no-go checkpoint: if Rust search
isn't decisively faster there, stop before paying for the write path.

Biggest technical risks, in order:
1. **FST builder byte-compatibility** (P5) — mitigate: reader-only first, and consider
   accepting non-byte-identical-but-format-valid output (Java can read it; only golden
   tests need loosening).
2. **Two-phase commit / translog recovery semantics** (P6) — mitigate: segment
   replication first, exhaustive crash fuzzing.
3. **JNI crash blast radius** — a Rust bug can kill a node, not just a shard —
   mitigate: handle validation, fuzzing of the FFI surface, optional
   process-isolation mode (sidecar over shared memory) as a fallback design we keep
   sketched but don't build unless needed.
4. **Scoring drift** breaking top-k parity — mitigate: float discipline + differential
   harness from day one of P3.
