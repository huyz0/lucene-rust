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
- No soft-deletes semantics beyond what OpenSearch requires (OpenSearch **does**
  require soft-deletes for replication — this lands in Phase 6, it is required
  before write-path integration, just not before read-path integration).
- Index sorting: multi-field NUMERIC index sort is now supported at flush time
  (`segment_info.rs`'s `IndexSortField`/`SortMissingValue` with
  `SegmentInfo::index_sort: Option<Vec<IndexSortField>>`, `segment_writer.rs`'s
  `flush_sorted_stored_only_segment` taking a `&[SortKeySpec]`) -- a priority-
  ordered `Sort` of one or more `SortField`s, each with its own independent
  `reverse`/missing-value policy, ties broken field-by-field just like real
  Lucene's `Sort` array. Merges of sorted segments now also preserve global
  sort order (`merge.rs`'s `merge_sorted_stored_only_segments`, a genuine
  k-way merge by sort key across sources reusing `segment_writer.rs`'s
  `sort_key_rank` comparator, not a concatenation of source A's docs then
  source B's docs) -- see `docs/parity.md` for the exact scope. Still
  explicitly out of scope: the k-way merge only reorders stored fields (doc
  values/norms/term vectors are never reordered during a merge, matching
  this port's existing write-side limits), and the `.si` index-sort byte
  encoding remains this port's own internal format, NOT verified
  byte-compatible with real Lucene's `Lucene99SegmentInfoFormat` (no
  real-Lucene-written sorted-segment `.si` fixture exists to derive the true
  `SortFieldProvider` wire format from) -- true for single-field and remains
  true now that multiple fields and merges are supported.
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
| `lucene-analysis` | `o.a.l.analysis` + `analysis/common` subset | TokenStream trait, StandardTokenizer (from Unicode segmentation), lowercase/stop/ascii-folding/Porter-stem; everything else stays JVM-side long-term |
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
   side and a write side supporting any number of dimensions and any number of
   leaves (`LatLonPoint`-shaped multi-dimension fields included, via a
   widest-range split-dimension heuristic) are done -- see `docs/parity.md`'s
   points row. Real query-driven pruning execution (an actual `PointRangeQuery`/
   bounding-box `IntersectVisitor` on the search side) remains a future slice --
   this port's own reader still decodes every leaf rather than pruning.
8. **Live docs** (`.liv`) and per-commit deletes/DV-updates generations.
9. **KNN vectors** (`.vec/.vex/.vem`, HNSW): schedule **last within the phase** and be
   willing to punt to Phase 8 — big, self-contained, and OpenSearch's k-NN often uses
   its own engines (faiss/nmslib) anyway.
10. `SegmentReader` + `DirectoryReader.open(commit)` -- minimal version now ported,
    see task #45 in Phase 3's write-up below; `MultiTerms`/a unified leaf abstraction
    still not started.

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
fixture segment. Still deferred at that point: nested `BooleanQuery` clauses
(closed by task #25, see below), `minimumNumberShouldMatch` (closed by task
#24, see below), and relevance scoring.

Task #24 closed the `minimumNumberShouldMatch` gap: `query::BooleanQuery`
gained a `minimum_should_match: usize` field (default `0`, via
`with_minimum_should_match`, an additive builder method — no existing call
site needed to change). Verified against real `BooleanWeight.scorer`/
`bulkScorer`/`explain` source rather than assumed: `should` is gated by
`minimum_should_match` **regardless of whether `must` is also non-empty** —
the interaction is easy to get backwards, since the pre-#24 rule ("`should` is
score-only once `must` exists") only applies at `minimum_should_match == 0`.
A new `should_match_counts` helper (`HashMap<i32, usize>` tally across each
`should` clause's doc-ID list) gives `matched_boolean_docs` (the merge logic
shared by `search_boolean_query`/`search_boolean_query_scored`, unified in the
same task to avoid implementing the new gating twice) the per-doc
"how many `should` clauses agreed" count a plain `Disjunction` can't answer.
`minimum_should_match` exceeding `should.len()` needs no special case — no
doc's count can ever reach an unreachable threshold, so the same comparison
naturally yields real Lucene's `MatchNoDocsQuery` outcome. Scoring is
unaffected: `search_boolean_query_scored` still sums every `must`/`should`
clause a matched doc satisfies, not just `minimum_should_match`-worth.
Differential-tested in `crates/lucene-search/tests/boolean_query_fixtures.rs`
and `scoring_fixtures.rs` against the same fixture segment.

Task #25 closed the nested-`BooleanQuery`-clauses gap: `query::BooleanQuery`'s
`must`/`should`/`must_not` fields changed from `Vec<TermQuery>` to
`Vec<Clause>`, where `Clause` is a new closed two-variant enum
(`Clause::Term(TermQuery)` / `Clause::Boolean(Box<BooleanQuery>)`) — an enum
rather than a `Weight`/`Scorer`-style trait object, since `TermQuery` and
`BooleanQuery` are the only two shapes that actually need to nest today (see
the `rust-performance` skill's "enums where the closed set allows" guidance;
`PhraseQuery` deliberately isn't a `Clause` variant yet, tracked as a future
extension). `Clause`'s `From<TermQuery>`/`From<BooleanQuery>` impls, combined
with `with_must`/`with_should`/`with_must_not` taking `impl Into<Clause>`
items, kept every existing `with_must([TermQuery::new(...)])` call site
(including `lucene-ffi`'s `ffi_search_boolean_query`, which still only ever
constructs flat `Clause::Term` clauses from its four-parallel-array wire
format) compiling unchanged. `lib.rs` gained two recursive helpers:
`resolve_clause_docs` (matching: a `Clause::Boolean` recurses into a fresh
`matched_boolean_docs` call on the nested query, respecting that query's own
`must`/`should`/`must_not`/`minimum_should_match` completely independently of
the parent's) and `clause_scores` (scoring: a nested `BooleanQuery`'s own
score contribution is the sum of *its own* matching `must`/`should`
sub-clauses' scores, restricted to the doc set the nested query itself
matched — mirroring real Lucene's additive `BooleanScorer` recursion).
Neither helper hardcodes a nesting-depth limit — a `Clause::Boolean` nested
inside another `Clause::Boolean` resolves the same way, recursively.
Differential-tested (2–3 levels of nesting, both matching and scoring) in
`crates/lucene-search/tests/boolean_query_fixtures.rs`/`scoring_fixtures.rs`
against the same fixture segment, plus unit tests in `lib.rs`/`query.rs`
proving a nested clause's own `minimum_should_match` is evaluated
independently of the parent's (no cross-contamination in either direction).

A third slice (task #13) landed **BM25 relevance scoring**: `similarity.rs`
ports the pure `BM25Similarity` formula (`idf`/`tfNorm`/`score`, defaults
`k1 = 1.2`/`b = 0.75`, verified against `BM25Similarity.java` and independently
hand-computed in its unit tests), `search_term_query_scored`/
`search_boolean_query_scored` wire it into a new `ScoringCollector` trait
(deliberately *not* a breaking change to the existing `Collector` trait — see
`collector.rs`'s module doc), and `TopDocsCollector` is the ported
`TopScoreDocCollector`-equivalent (tie-break verified against real
`HitQueue.lessThan`: lower doc ID wins a score tie). A follow-on task closed
the norms gap this slice originally left open: `search_term_query_scored`/
`search_boolean_query_scored` now take an optional opened
`field_norms::FieldNorms` (real per-doc field length, decoded from `.nvd`/
`.nvm` via `norms::norm_value` plus a new `lucene_util::small_float`
`SmallFloat.byte4ToInt`-equivalent decode, with `avgFieldLength` computed once
per field per query by averaging every live doc's decoded length) instead of
always substituting a constant. Passing `None` (a field with no opened norms —
disabled for that field, or a caller that hasn't wired norms opening yet)
still falls back to the constant `fieldLength == avgFieldLength == 1.0`
approximation, now a deliberate, documented fallback rather than the only
option. Differential-verified in `crates/lucene-search/tests/scoring_fixtures.rs`
against this fixture's real `_0.nvm`/`_0.nvd`: decoded per-doc lengths match
hand-derived values from the fixture's own known per-term postings
frequencies, and real-norms scores differ from the constant-fallback scores
for the same query. Remaining gap: no cross-engine BM25-score fixture
generator exists yet (a Java-side one comparing this port's final scores
byte-for-byte against real Lucene's), so exact numeric parity for a full
query (not just the length-normalization term being live) is still unverified
— future work. The items below remain as originally scoped except where
superseded above.

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
no special-casing. (BM25 phrase scoring landed later, task #29 — see below.)
Differential-tested in `crates/lucene-search/tests/phrase_query_fixtures.rs`,
reusing the existing `pos` field already in `fixtures/data/blocktree_index/`
(no fixture generator changes needed — its real occurrences already have an
adjacent pair in one doc and a non-adjacent/absent pair in another). See
`docs/parity.md`'s new `PhraseQuery`/`ExactPhraseScorer` row for full detail.

**Task #28** closed the `slop`-deferred gap above: `PhraseQuery` now carries a
`slop: u32` field (default `0`, `with_slop` builder method), and
`search_phrase_query` dispatches to the existing exact-adjacency
`phrase_matches_in_doc` fast path when `slop == 0`, or a new
`phrase_matches_in_doc_sloppy(term_positions, slop)` otherwise. That function
implements an **in-order-only** subset of real Lucene's sloppy semantics: for
an alignment `p_0 < p_1 < ... < p_{n-1}` (one position per term, strictly
increasing, in phrase order), the total "moves" needed is
`(p_{n-1} - p_0) - (n - 1)` (telescoped sum of each adjacent gap's slack,
per real `PhraseQuery`'s Javadoc description of slop as "moves to line up in
order"); a doc matches iff some such alignment has moves `<= slop`. Real
Lucene's general `SloppyPhraseMatcher` additionally allows term **reordering**
within the slop budget via a priority-queue-based edit-distance computation —
that general algorithm was not confidently re-derivable/verifiable against
real Lucene's source within this task's scope, so it's deliberately out of
scope here (documented on `phrase_matches_in_doc_sloppy` and in
`docs/parity.md`), not silently guessed at. Tested with hand-computed unit
tests (exact/boundary/gap-of-N slop values, multiple candidate base
positions, repeated terms) plus one `search_phrase_query` wiring test against
the real `pos` fixture. **Cross-engine gap now closed**: `GenBlockTree.java`'s
`pos` field gained doc7 (alpha@0, beta@3, a real non-adjacent gap needing 2
moves), and its generator now actually runs real `IndexSearcher`/
`PhraseQuery.setSlop(n)` against it for `n` in `{0,1,2,3,5}`, recording real
Lucene's match/no-match verdicts in `manifest.properties`.
`phrase_query_fixtures.rs::sloppy_phrase_gap_matches_real_lucenes_phrase_query_set_slop_at_every_tested_value`
confirms this port's sloppy path agrees with real Lucene at all five slop
values — the sloppy-match formula is now cross-engine verified, not just
self-consistent.

**Task #29** closed two related deferred gaps at once: **`PhraseQuery` BM25
scoring** and **`PhraseQuery` as a `BooleanQuery` clause**. `search_phrase_query_scored`
mirrors `search_phrase_query`'s matching, additionally computing a per-doc
"phrase frequency" fed through the same `similarity::tf_norm`/`FieldNorms`
machinery `search_term_query_scored` already uses; the phrase's `idf` is the
sum of each constituent term's own `idf(docFreq, docCount)` (real
`BM25Similarity.idf(CollectionStatistics, TermStatistics[])`'s actual
combined-term behavior, verified against source, not guessed). Exact
(`slop == 0`) phrase frequency (`phrase_freq_exact`) counts every valid base
position in the first term's own position list — one match per distinct
starting position, matching `ExactPhraseScorer`'s own counting granularity, no
double-counting of overlapping repeats. Sloppy (`slop > 0`) phrase frequency
is **deliberately simplified** to a matches-or-not `1`/`0` signal rather than
real Lucene's graduated `1.0 / (matchLength + 1)` per-match `SloppyPhraseMatcher`
weighting — that exact formula (layered on an alignment-enumeration algorithm
already scoped down to in-order-only, task #28) could not be confidently
re-derived/verified within this task's scope, so graduated sloppy match-quality
scoring is deliberately deferred (documented in `docs/parity.md`), consistent
with this port's "scope down honestly" practice. Separately, `query::Clause`
grew a third variant, `Clause::Phrase(PhraseQuery)` (alongside the existing
`Clause::Term`/`Clause::Boolean`), making `PhraseQuery` composable inside a
`BooleanQuery`'s `must`/`should`/`must_not`. Wiring this in required a
signature change threaded through the whole recursive chain
(`search_boolean_query`, `search_boolean_query_scored`, `matched_boolean_docs`,
`resolve_clause_docs`, `clause_scores`): each now additionally takes
`pos_in`/`pay_in` (the segment's opened `.pos`/`.pay` files), since resolving
a `Clause::Phrase` needs them — `resolve_clause_docs` delegates matching to
`search_phrase_query`, `clause_scores` delegates scoring to
`search_phrase_query_scored`, both via small local collectors rather than
duplicating either function's logic. `None`/`None` is fine for a query with no
multi-term phrase clause; passing `None` for a query that turns out to need it
surfaces as `Error::MissingPosInput`, same convention `search_phrase_query`
already established. Tested: unit tests for `phrase_freq_exact` (single
occurrence, repeated/overlapping occurrences, no-alignment, empty/missing-term
edge cases), a fixture test hand-deriving the expected BM25 score for the
`pos` field's real alpha/beta phrase from `manifest.properties`' real `docFreq`
values, unit tests for `Clause::Phrase` matching/scoring inside a
`BooleanQuery` (including one clause nested inside a `Clause::Boolean`), and a
`scoring_fixtures.rs` differential test proving a `Clause::Phrase`'s score sums
correctly alongside a sibling `Clause::Term`'s. **Deliberately not touched**:
`lucene-ffi`'s `ffi_search_boolean_query`/`ffi_search_boolean_query_scored`
(the latter added by task #30) still only construct flat `Clause::Term`
clauses from their C-ABI wire format — exposing `Clause::Phrase`/
`Clause::Boolean` construction over FFI remains deferred (see
`docs/parity.md`).

**Progress (task #21):** doc-values-driven range query and single-key sort now
exist in `lucene-search/src/doc_value_query.rs`, built directly on
`lucene-codecs`' already-complete doc-values read side (`numeric_value`,
`sorted_ord`). `search_numeric_range` full-sweeps `[0, max_doc)` checking each
live doc's `NumericEntry` value against an inclusive `[min, max]` (no BKD/
skip-index pruning exists to do better); `search_sorted_ord_range` is the same
shape over a single-valued SORTED field's ordinal. `sort_by_numeric_doc_value`
sorts an already-collected candidate doc-ID list (e.g. `search_term_query`'s
output) ascending by numeric value, ties broken by ascending doc ID, with an
explicit `MissingValue::{Exclude, Default}` policy for candidates lacking a
value — implemented as a standalone function rather than a new `Collector`
variant, since sorting needs the whole candidate set before it can produce
its first output pair (unlike `Collector`'s streaming per-doc callback or
`TopDocsCollector`'s incremental top-`N` heap). **Scoped to single-valued
NUMERIC/SORTED fields only** — multi-valued SORTED_NUMERIC/SORTED_SET
range/sort needs a `SortedNumericSelector`/`SortedSetSelector`-equivalent this
port doesn't have yet (deferred; `doc_values::sorted_numeric_values` is
already the read-side building block for that future slice). Verified against
real Lucene by reusing the already-checked-in `fixtures/data/doc_values_index/`
and `fixtures/data/sorted_dv_index/` fixtures (no new Java generator needed).
See `docs/parity.md`'s new row for the full accounting.

**Progress (task #31):** the multi-valued gap task #21 deferred is now closed.
`ValueSelector` (`Min`/`Max`) reduces a SORTED_NUMERIC/SORTED_SET field's
multiple per-doc values to one comparable value (real Lucene's
`SortedNumericSelector.Type`/`SortedSetSelector.Type`, scoped to MIN/MAX —
`MIDDLE_MIN`/`MIDDLE_MAX` remain deferred, a small follow-up if ever needed).
`search_multi_valued_range`/`sort_by_multi_valued_doc_value` are the
multi-valued siblings of task #21's two range/sort functions, built on
`doc_values::sorted_numeric_values` (confirmed to genuinely decode a doc's
*entire* value list, not just one). Both take a `SortedNumericEntry`, which —
since `sorted_numeric_values` reads SORTED_NUMERIC values and a multi-valued
`SortedSetKind::Multi` field's ordinals identically — means the same two
functions serve **both** field types with no separate sorted-set code path
(pass the `Multi` variant's `ords` entry for SORTED_SET). Verified against
real Lucene via the already-checked-in `fixtures/data/multi_valued_dv_index/`
fixture (`fixtures/src/GenMultiValuedDocValues.java`, already used by
`lucene-codecs`' own read-side tests — no new Java generator needed): a
SORTED_NUMERIC field with 0-3 values/doc and a SORTED_SET field with 0-2
ordinals/doc sharing a terms dictionary, confirming (among other cases) that
a doc whose MIN falls in range but MAX doesn't (and vice versa) is decided by
the selector alone. See `docs/parity.md`'s updated row for the full
accounting.

**Progress (task #32):** `DisjunctionMaxQuery` is ported — matching (a pure
union of `disjuncts`, no `minimum_should_match`-style gate) and scoring (real
`DisjunctionMaxScorer`'s exact `max(disjunct scores) + tie_breaker *
sum(rest)` formula). `Clause` gains a fourth variant,
`Clause::DisjunctionMax(Box<DisjunctionMaxQuery>)`, nesting the same way
`Clause::Boolean` already does (either direction: a `BooleanQuery` clause can
be a `DisjunctionMax`, and a `DisjunctionMaxQuery` disjunct can be a
`Boolean`, to arbitrary depth). Verified against real Lucene both by reusing
`fixtures/data/blocktree_index/`'s already-fixture-verified `body` postings
and by a genuine `IndexSearcher.search(new DisjunctionMaxQuery(...), 10)` run
against that exact segment (`fixtures/src/AppendDismaxManifest.java`,
appended to the manifest without perturbing the segment's committed random
ID), asserting doc-for-doc, score-for-score agreement with real Lucene's own
`TopDocs` output. **This cross-engine test caught a pre-existing BM25 formula
bug**: `similarity::tf_norm` carried a spurious `(k1 + 1)` numerator factor
(the textbook BM25 term, not real Lucene 10.5.0's actual `BM25Scorer.doScore`
formula) that every earlier self-consistency test had independently
reimplemented instead of catching — fixed, with dependent tests' hand-computed
expected values updated to match. See `docs/parity.md`'s new row and the
`BM25Similarity` row's updated note for the full accounting.

**Progress (task #33):** `ConstantScoreQuery`/`BoostQuery` are ported —
`Clause` gains two more variants, `ConstantScore(Box<ConstantScoreQuery>)` and
`Boost(Box<BoostQuery>)`, wrapping any other `Clause` (nesting either
direction, arbitrary depth, same pattern as `Boolean`/`DisjunctionMax`).
Matching is always the wrapped clause's own matching set; scoring replaces the
inner score with a fixed constant (`ConstantScoreQuery`) or multiplies it by a
boost factor (`BoostQuery`). No new Java fixture generator was added — both
wrappers are a trivial arithmetic composition (a literal constant, a single
`f32` multiply) over an inner clause whose own scoring is already cross-engine
verified (task #30/#32's fixtures), so the tests instead use
`search_term_query_scored`'s already-real BM25 score as ground truth for the
"known real" inner score and assert the wrapped result exactly matches the
constant/product — see `lib.rs`'s test module doc comment for the full
reasoning.

**Progress (task #34):** `WildcardQuery` is ported — a leaf `Clause::Wildcard(WildcardQuery)`
matching every doc containing at least one term (for `query.field`) that
`lucene_codecs::wildcard::WildcardPattern` accepts, unioned across every
matching term (`wildcard_doc_ids` in `lib.rs`, reusing task #1's
`FieldTerms::intersect`/`WildcardPattern` machinery rather than building a
second parallel term-matching path). Unscored: every matching doc scores a
flat `1.0` (real `MultiTermQuery`'s default constant-score-style rewrite for
a bare, unwrapped multi-term query — a caller wanting BM25-shaped scoring
would wrap it in the existing `ConstantScoreQuery`/`BoostQuery`, same as real
Lucene's own `rewrite()` chain). This task also closed task #1's one
remaining documented gap in `WildcardPattern`: `\`-escaping of a literal
`*`/`?`/`\` byte, matching real `WildcardQuery.toAutomaton`'s
`WILDCARD_ESCAPE` handling exactly (a `\` forces the following byte to be a
plain literal even if it's itself special; a trailing, unpaired `\` falls
back to matching a literal `\`). Verified against real Lucene via
`fixtures/src/AppendWildcardManifest.java` (same append-only pattern as
`AppendDismaxManifest.java` — opens the already-committed
`fixtures/data/blocktree_index/` directory read-only and runs eight real
`org.apache.lucene.search.WildcardQuery` patterns against `body`'s real terms,
recording real Lucene's own matched doc IDs), asserting doc-for-doc agreement
with this port's own `Clause::Wildcard` matching for every recorded case
(literal, prefix-`*`, `?`, bare `*`, no-match, escaped `*`, escaped
non-special byte, `?` matching a literal `d` in `bird`). See
`docs/parity.md`'s new row for the full accounting.

**Progress (task #35):** `PrefixQuery` is ported — a leaf
`Clause::Prefix(PrefixQuery)` matching every doc containing at least one term
(for `query.field`) starting with `query.prefix`'s literal bytes, unioned
across every matching term (`prefix_doc_ids` in `lib.rs`, structurally
identical to task #34's `wildcard_doc_ids` but built on
`lucene_codecs::wildcard::WildcardPattern::prefix` instead of
`WildcardPattern::new`). Unscored, same flat `1.0` per match as
`Clause::Wildcard` and for the same reason. **Design decision**: rather than
building `PrefixQuery` as a thin wrapper that escapes `prefix`'s bytes and
appends an unescaped `*` to reuse `WildcardQuery`'s glob parser,
`PrefixQuery` calls `WildcardPattern::prefix` directly — a constructor that
already existed (task #1) and builds its token list straight from `prefix`'s
raw bytes as literals plus a trailing `AnyMany`, never touching
`WildcardPattern::new`'s escape/glob-parsing loop at all. This sidesteps the
escaping-edge-case risk entirely instead of mitigating it: a prefix
containing a literal `*`/`?`/`\` byte (e.g. `a*b`) is matched as the 3 literal
bytes it is, with no escaping step that could get it wrong. Verified against
real Lucene via `fixtures/src/AppendPrefixManifest.java` (same append-only
pattern as `AppendWildcardManifest.java`) running six real
`org.apache.lucene.search.PrefixQuery` cases against `body`'s real terms — a
prefix matching one term, a prefix matching several, the empty prefix
(matches every term), a prefix equal to a full existing term, a no-match
prefix, and a prefix containing literal `*`/`?` bytes — asserting doc-for-doc
agreement with this port's own `Clause::Prefix` matching for every recorded
case. See `docs/parity.md`'s updated row for the full accounting.

**Progress (task #36):** delete-by-BKD-point-range is ported —
`crates/lucene-index/src/points_delete.rs`'s `resolve_points_range_doc_ids`
(field number + inclusive `[min_packed, max_packed]` byte range →
matching live doc IDs, per-dimension unsigned-byte-wise comparison,
de-duplicated/ascending) and `resolve_and_apply_points_range_delete` (same
resolve-then-`deletes::apply_deletes` shape as task #27's
`term_delete::resolve_and_apply_term_delete`), the BKD-range analog of that
task's delete-by-term flow. **Design decision / honest gap**: no BKD
range-query matcher existed anywhere in the workspace to reuse — unlike
task #27, where `lucene-search::term_doc_ids` already existed and
`term_delete.rs` just reimplemented its handful of lines locally to avoid an
upward `lucene-index → lucene-search` dependency (verified again here:
`crates/lucene-search/Cargo.toml` depends on `lucene-index`, not the other
way around) — `lucene-search`'s only range-shaped queries
(`search_numeric_range`/`search_sorted_ord_range`/`search_multi_valued_range`
in `doc_value_query.rs`) walk doc-values, not the BKD tree, and
`lucene_codecs::points` itself has no intersection/pruning logic, only
`PointsReader::decode_all_points` (decodes every point in a field
unconditionally). `points_delete.rs` therefore decodes every point via that
existing primitive and filters in memory rather than porting
`BKDReader.intersect`'s tree-pruning traversal — correct, not sublinear; a
real perf gap against `BKDReader.intersect`, tracked in `docs/parity.md`, not
hidden. Verified with new hand-built in-memory fixtures (via the existing
`points::write`, mirroring how `term_delete.rs`'s own tests build a segment
in-memory) rather than a new checked-in real-Lucene fixture: the existing
`fixtures/data/points_index/` fixture (task #18/#22, `GenPoints.java`) is
single-dimension, and `points_delete.rs`'s unit tests already need a
hand-built 2D fixture to cover the multi-dimension per-dimension-AND
semantics that fixture can't exercise anyway, so extending it with a new
`Append*Manifest.java` would have added Java-side machinery without covering
anything the in-memory fixture doesn't already cover byte-for-byte (same
`points::write`/`points::open` round-trip task #18/#22's own tests already
verify against real Lucene). Unit tests cover: exact range match, inclusive
boundaries on both ends, zero-match range (no-op), all-docs range,
unknown-field-number no-op, live-docs filtering, and 2D multi-dimension
"every dimension must independently be in range" semantics. See
`docs/parity.md`'s updated row for the full accounting.

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
   (both **ported** — `WildcardQuery` task #34, `PrefixQuery` task #35 — glob/
   prefix matching via the existing `WildcardPattern`/`FieldTerms::intersect`
   machinery, not real automaton/`IntersectTermsEnum` block-skipping),
   `FuzzyQuery` (**ported**, task #42 — edit-distance DP, not a
   `LevenshteinAutomata`), `RegexpQuery` (**ported**, task #43 — a hand-built
   parser/backtracking matcher over a restricted Lucene-regexp syntax subset,
   not `o.a.l.util.automaton`/`CompiledAutomaton`; see `docs/parity.md`'s row
   for exactly which operators are supported vs deferred),
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
last-error string. `crates/lucene-ffi/src/*.rs` is unit-tested (≥95% line
coverage per file) calling the exact exported `extern "C" fn` entry points
against the real `fixtures/data/blocktree_index/` fixture, including
stale/closed-handle rejection and a genuine caught-panic-surfaces-as-a-status-
code test.

**Progress (task #30):** relevance-scored query execution now has a C-ABI
surface too, closing the gap task #20 deferred: `ffi_search_term_query_scored`/
`ffi_search_boolean_query_scored`/`ffi_search_phrase_query_scored` mirror their
unscored siblings' parameter shapes plus a `top_n` (feeding a
`lucene_search::TopDocsCollector`), storing `(doc_id, score)` hits in a new
`ScoredResultsHandle`/`RegistryTag::ScoredResults` registry (kept separate from
the unscored `ResultsHandle` so a handle from the wrong search flavor is
rejected by the handle-tag check, not misread) — read back via
`ffi_scored_results_len`/`ffi_scored_results_copy` (two parallel caller-
allocated buffers, doc IDs and scores, not one interleaved buffer) and released
via `ffi_close_scored_results`. `ffi_open_segment` grew optional `nvm_name`/
`nvd_name` parameters so a caller can open the segment's real `.nvm`/`.nvd`
norms pair, giving the scored functions real per-doc/avg field lengths instead
of always falling back to the unnormed constant approximation; `SegmentHandle`
now also carries the parsed `field_infos` needed to map a scored query's field
name to the field number norms are keyed by. Same test rigor as task #20:
real fixture round-trips (including a differential real-norms-vs-unnormed-
fallback test), stale/wrong-registry-tag handle rejection, and a
mutex-poisoning regression test for the scored path — using a `thread_local!`-
scoped panic-injection switch rather than reusing task #20's process-wide
`AtomicBool` one, since the latter is exposed to a cross-test race under
`cargo test`'s default parallel execution (flagged by task #29's review;
fixing the pre-existing one is tracked as a separate follow-up, not touched by
this task). See `docs/parity.md`'s `## lucene-ffi` section for the exact
surface and what's still deferred (`.liv`/`.pay`, the unified `.si`-driven
segment-open entry point, the JNI wrapper class itself, nested/phrase
`BooleanQuery` clause construction over the C ABI, and the query-tree
serialization / OpenSearch plugin work below, all still not started).

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

**Progress (task #15, extended by task #26):** `lucene-index/src/merge.rs::merge_stored_only_segments`
merges N already-flushed segments into one new segment: reads each source's `FieldInfos` +
`Document`s back out (via `stored_fields::open`/`.document()`, already read-only ported in
Phase 2), drops non-live docs per an optional per-source `FixedBitSet` (via `live_docs::parse`),
reconciles field numbering across sources by field name (`reconcile_field_numbers`, the
merge-time slice of `FieldInfos.FieldNumbers` — a segment's own field number is local to that
segment, so two sources naming the same field differently is a real case, not a hypothetical
one), renumbers surviving docs contiguously by concatenating sources in order, and writes the
merged `.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si`. Task #26 extended this to also merge doc values,
norms, and term vectors whenever a caller supplies them per source (`MergeSource`'s optional
`numeric_doc_values`/`norms`/`term_vectors` fields): each format's per-doc data is decoded via
the existing read-side functions (`doc_values::numeric_value`, `norms::norm_value`,
`term_vectors::TermVectorsReader::document`), filtered/renumbered/concatenated the same way
stored fields are, and re-encoded via the existing single-field write-side encoders
(`doc_values::write_single_dense_numeric_field`, `norms::write_single_dense_field`,
`term_vectors::write_best_speed`) into `.dvm`/`.dvd`/`.dvs`, `.nvm`/`.nvd`, and
`.tvd`/`.tvx`/`.tvm`. Because those write-side encoders are single-field-only, at most one
numeric-doc-values field and one norms field can be merged per call; because they're
dense-only, a doc-values/norms field can only be merged if every source contributing live docs
has full data for it (an `Error` otherwise, not a silent drop) — term vectors have neither
limit. **Important**: this remains mergeable-if-a-caller-has-the-data, not a real
end-to-end scenario — `flush_stored_only_segment` (this port's only write path that produces a
full segment) still only ever writes stored-fields-only segments, so no real caller can yet
*produce* doc-values/norms/term-vectors sources to merge; only this module's own tests do.
Still missing, and still item 6 below: `TieredMergePolicy`-style merge *selection* (this is
caller-picks-the-sources merge *execution* only), background/concurrent merging, merge-time
codec upgrades, multi-field `.dvd`/`.nvd`, and merging points/postings/binary-or-sorted doc
values/term-vector offsets-payloads.

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
**Progress (task #27):** `lucene-index/src/term_delete.rs` closes the delete-by-term half of
the gap above, scoped to **one already-opened segment**: `resolve_term_doc_ids` takes a
segment's `BlockTreeFields` + opened `.doc` file + a `(field, term)` pair and returns the
matching **live** doc IDs ascending, using only `lucene-codecs` primitives (`field.postings`
+ a `live_docs` filter) — the same lookup `lucene-search::term_doc_ids` already does, kept at
the `lucene-codecs` layer rather than depending on `lucene-search` from `lucene-index` (that
would invert the intended `util ← store ← codecs ← index ← search ← core ← ffi` dependency
graph, since `lucene-search` already depends on `lucene-index`). `resolve_and_apply_term_delete`
composes that with `deletes::apply_deletes` for the full per-segment resolve-then-apply flow.
**Still explicitly deferred, and why:** multi-segment resolution (a real `IndexWriter`'s
`BufferedUpdates`/`ReaderPool` resolves a delete against *every* currently-open segment, not
one already-opened one — this port has no multi-segment reader/writer orchestration); delete-
by-query beyond a single exact term; and `updateDocument` (real Lucene defines it as delete-by-
term + `addDocument` — now that delete-by-term exists for one segment, a caller can compose
`resolve_and_apply_term_delete` with a separate `flush_stored_only_segment`/merge call by hand,
but a first-class `updateDocument` wrapper is left for when multi-segment resolution exists, so
it composes correctly rather than silently only covering one segment). See `docs/parity.md`'s
updated row for full detail and test coverage.
**Progress (task #37):** `lucene-index/src/update_document.rs::update_document` adds that
first-class `updateDocument` wrapper: it composes task #27's `resolve_and_apply_term_delete`
(fanned out over every segment in a `SegmentInfos` that the caller supplies an opened
`SegmentDeleteSource` for) with task #11's `flush_stored_only_segment` (the new document), then
commits both as one atomic `segment_infos::write` call — the single durable state transition,
performed only after every earlier fallible step has already succeeded, so a reader can never
observe a `segments_N` reflecting only the delete or only the add. This closes task #27's
explicitly-deferred `updateDocument` gap by composing already-verified primitives rather than
building anything new byte-format-wise; a real reader-pool-driven true multi-segment
*resolution* (opening/searching every live segment automatically) is still out of scope, same
as task #27, since this port has no reader pool — the caller still supplies whichever segments'
`BlockTreeFields` it already has open. See `docs/parity.md`'s new row for full detail, the exact
atomicity argument, and test coverage.
**Progress (task #38):** `flush_stored_only_segment` can now actually pack a flushed
segment's files into a `.cfs`/`.cfe` pair, closing the gap `compound_format.rs`'s own
parity row flagged (the write-side codec existed but nothing in the writer pipeline called
it). A new `use_compound_file: bool` parameter picks the on-disk layout: `false` keeps the
pre-existing loose-file behavior byte-for-byte (regression-tested); `true` packs the same
`.fdt`/`.fdx`/`.fdm`/`.fnm` bytes through `compound_format::write` into
`<segment_name>.cfs`/`.cfe` instead, and the `.si` correctly records
`is_compound_file: true`. Real Lucene only calls `Lucene90CompoundFormat.write` once
`TieredMergePolicy`'s size heuristic (`noCFSRatio`/`maxCFSSegmentSizeMB`) says a flushed
segment is small enough — this port has no merge policy or segment-size accounting at all
(see item 6 below), so a size-based heuristic would have nothing real to compare against;
a direct boolean is simpler and equally correct for both of this port's current callers
(`update_document.rs` passes `false`, keeping its existing loose-file commits unchanged).
No new Java fixture was needed: `compound_format.rs` was already differentially verified
write→read against real Lucene (see its parity row), so this task is pure Rust-side
wiring/composition, not new byte-format decoding — verified instead by a unit test that
flushes with `use_compound_file: true` and recovers the original `.fnm`/`.fdt`/`.fdx`/`.fdm`
bytes *through* `compound_format::parse_entries`/`open_input`, then confirms
`stored_fields::open` can read the documents back out through those recovered slices (not
the original in-memory buffers) — an end-to-end check that a byte-offset bug in the new
wiring would actually fail.
**Progress (task #39):** `lucene-search/src/term_vectors_query.rs` surfaces task #3/#26's
already-decoded term vectors through a query-facing read API — real Lucene's
`IndexReader.getTermVector(int doc, String field)` equivalent. `term_vector_for_doc` resolves
a caller-friendly `(doc, field name)` pair to the codec's `(doc, field number)` shape via
`FieldInfos`, then returns whichever fields/terms/frequencies/positions/offsets/payloads
`term_vectors::TermVectorsReader::document` already decoded — no new byte-format work, purely
a thin adapter, following the same "None for no match, propagate the codec's own Err" contract
every other `lucene-search` query function already uses. On top of that, `matched_term_offsets`
adds one real, scoped-down highlighting primitive: given a field's already-decoded term vector
and a set of matched terms, it computes `(term, start_offset, end_offset)` spans — exactly what
a `Highlighter`/`UnifiedHighlighter` needs to slice source text at match boundaries — returning
`None` (not a guess) when that field has no stored offsets. No fragment selection, snippet
assembly, or scoring is attempted; this is the offset-lookup primitive a highlighter sits on
top of, not a highlighter. No new Java fixture was generated — per the `differential-testing`
skill's precedent for composition/wiring tasks (#36-38), this task's own logic touches no new
byte format, so its tests reuse the real, already cross-engine-verified
`fixtures/data/term_vectors_index/` fixture (task #3's) instead. See `docs/parity.md`'s new row
for the full scoping detail and test list.

**Progress (task #40):** task #21/#31's `sort_by_numeric_doc_value`/
`sort_by_multi_valued_doc_value` (`lucene-search/src/doc_value_query.rs`) now have a
C-ABI surface, following task #30's exact FFI pattern: `ffi_sort_by_doc_value`/
`ffi_sort_by_multi_valued_doc_value` (`lucene-ffi/src/sort.rs`) take an already-known
candidate doc-ID list plus a field name, resolve it to a NUMERIC/SORTED_NUMERIC
doc-values entry via the segment's opened `.dvm` (new `dvm_name`/`dvd_name`/
`dv_suffix` parameters on `ffi_open_segment` — doc-values files carry their own
codec-suffix component, independent of the postings suffix, so this is a separate
parameter rather than reusing `segment_suffix`), and store the resulting ascending
`(doc_id, value)` pairs in a new `SortedResultsHandle`/`RegistryTag::SortedResults`
registry — kept separate from both the unscored `ResultsHandle` and the scored
`ScoredResultsHandle` since a sort's second element is an arbitrary `i64` doc-value,
not a BM25 `f32` score, and conflating the three would let a handle from the wrong
call be silently misread. Read back via `ffi_sorted_results_len`/
`ffi_sorted_results_copy` (two parallel buffers, doc IDs and values) and released via
`ffi_close_sorted_results` — the same shape `results_scored.rs` established. The
missing-value policy (`MissingValue::Exclude`/`Default(i64)`) crosses the wire as a
plain `missing_is_default: bool` + `missing_default: i64` rather than a tagged union;
the multi-valued entry point's `ValueSelector` (MIN/MAX) crosses as an `i32` (`0`/`1`,
anything else is `InvalidArgument`). Ascending-only, single-sort-key-only, matching
`doc_value_query.rs`'s own documented scope (no descending sort or multi-key `Sort`
composition exists in this port). Same test rigor as tasks #20/#30: real-fixture
round-trips against `fixtures/data/doc_values_index/`/`multi_valued_dv_index/` (the
same fixtures task #31's own differential tests already established as
cross-engine-correct — no new Java fixture was needed since this is FFI wiring
around an already-verified sort, not new decoding logic), unknown-field/wrong-entry-
kind/no-doc-values-opened rejection (`InvalidArgument`), stale/wrong-registry-tag
handle rejection, and a mutex-poisoning regression test using a `thread_local!`-
scoped panic-injection switch (not task #20's process-wide `AtomicBool`, for the
same cross-test-flakiness reason task #30 already documented). See `docs/parity.md`'s
`## lucene-ffi` section for the exact surface and what's still deferred.

**Progress (task #41, final task in this batch):** multi-segment search —
`lucene-search/src/multi_segment.rs` ports real `IndexSearcher.search`'s top-level
fan-out (score/collect each segment locally, translate local doc IDs to global via
`doc_base`, merge into one globally-ranked `TopDocs`-equivalent). One generic
`merge_multi_segment_scored` core (not one copy per query type) drives thin wrappers
`search_term_query_multi_segment`/`search_boolean_query_multi_segment`, since every
scored query function already reduces to the same `Vec<ScoreDoc>` shape via
`TopDocsCollector` — the merge step needs nothing query-type-specific. Each segment
is bounded to its own local top-`top_n` before the cross-segment merge (matching real
per-leaf `TopFieldCollector` behavior, provably lossless), and the merge itself reuses
`TopDocsCollector`'s own score-desc/doc-id-asc tie-break comparator a second time
rather than reimplementing it. **Explicit scope decision on idf**: real `BM25Similarity`
computes `idf` from index-wide `CollectionStatistics`/`TermStatistics` aggregated across
every segment; this port's existing scored query functions (unchanged by this task)
compute `idf` from each segment's own `docFreq`/`docCount` only — there is no
index-wide aggregation anywhere in this port, and this task does not add one (that's a
`DirectoryReader`-level concept out of scope here). This task therefore claims "correct
matching + correct per-segment-relative scoring + correct global merge order," **not**
byte-for-byte parity with real multi-segment Lucene's BM25 scores whenever a term's
`docFreq`/`docCount` genuinely differ across segments. **Verification**: unit tests in
`multi_segment.rs` isolate the merge core (interleaved scores across 2-3 synthetic
segments, truncation, a zero-match segment, cross-segment tie-break, `top_n == 0`, a
single segment with nonzero `doc_base`) plus end-to-end real-fixture calls through both
public wrappers. Cross-engine verification (`tests/multi_segment_fixtures.rs`): no
genuine 2+-segment Java fixture exists yet in this repo, so — the documented
next-best alternative — the real `fixtures/data/blocktree_index/` segment is opened
twice as two segments of one index, and this test independently concatenates/sorts each
"segment"'s own real, already fixture-proven BM25 scores (translated by `doc_base`) to
confirm `search_term_query_multi_segment`'s actual output matches, doc-for-doc and
score-for-score — proving the merge/doc-base-translation/tie-break logic against real
recorded scores. See `docs/parity.md`'s new row for the full scoping detail, what's
still deferred (real index-wide idf, a genuine multi-segment fixture, multi-segment
wrappers for the other query types, an `IndexReader`/`DirectoryReader` object model,
FFI exposure), and the exact test list.

**Progress (task #42):** `FuzzyQuery` is ported — a leaf `Clause::Fuzzy(FuzzyQuery)`
matching every doc containing at least one term (for `query.field`) within
`max_edits` edit distance of `term`, restricted to terms sharing `term`'s first
`prefix_length` bytes exactly, unioned across every matching term
(`fuzzy_doc_ids` in `lib.rs`, structurally identical to task #34/#35's
`wildcard_doc_ids`/`prefix_doc_ids` but built on the new
`lucene_codecs::fuzzy::{edit_distance, FuzzyMatch}` and
`FieldTerms::fuzzy_intersect` instead of `WildcardPattern`). Defaults mirror real
`FuzzyQuery` exactly: `max_edits = 2`, `prefix_length = 0`, `transpositions = true`.
Unscored, same flat `1.0` per match as `Clause::Wildcard`/`Clause::Prefix`.
**No `LevenshteinAutomata`/automaton machinery** — this port computes edit
distance directly with a plain `O(n*m)` DP each time a candidate term is tested
(narrowed first by `prefix_length`'s literal-prefix binary-search range, same
"narrow then filter" shape `FieldTerms::intersect` established for
`WildcardQuery`/`PrefixQuery`), closing `wildcard.rs`'s last remaining
explicitly-deferred gap from task #1. **The transpositions subtlety, gotten
right and verified, not assumed**: `edit_distance(a, b, transpositions)`
implements restricted/OSA Damerau-Levenshtein when `transpositions = true` (real
`FuzzyQuery`'s own default — an adjacent-character swap costs 1 edit, not 2) and
plain Levenshtein when `false`. **Byte-vs-codepoint scope decision, stated
explicitly**: real Lucene's `LevenshteinAutomata` operates on UTF-32 codepoints;
this port's `edit_distance` operates on raw bytes instead — a deliberate,
documented shortcut (not an oversight) since every fixture/test term is ASCII,
where one byte and one codepoint coincide; full codepoint decoding for non-ASCII
terms is deferred, see `fuzzy.rs`'s module doc. Verified against real Lucene via
`fixtures/src/AppendFuzzyManifest.java` (same append-only pattern as
`AppendWildcardManifest.java`/`AppendPrefixManifest.java`) running eleven real
`org.apache.lucene.search.FuzzyQuery` cases against `body`'s real terms — exact
match, single substitution/insertion/deletion, prefix-length exclusion,
max-edits boundary, a no-match case, and **the single most important case**: a
transposition (`"cta"` vs. target `"cat"`, `maxEdits=1`) matching with
`transpositions=true` and not matching with `transpositions=false`, confirmed
against real Lucene's own output on the first run — asserting doc-for-doc
agreement with this port's own `Clause::Fuzzy` matching for every recorded case.
See `docs/parity.md`'s new row for the full accounting.

**Progress (task #43):** `RegexpQuery` is ported — a leaf
`Clause::Regexp(RegexpQuery)` matching every doc containing at least one term
(for `query.field`) that a compiled regexp pattern accepts **in full** (real
`RegexpQuery`'s whole-term-match convention — never a substring match),
unioned across every matching term (`regexp_doc_ids` in `lib.rs`, structurally
identical to task #34/#35/#42's `wildcard_doc_ids`/`prefix_doc_ids`/
`fuzzy_doc_ids` but built on the new `lucene_codecs::regexp::RegexpPattern`
and `FieldTerms::regexp_intersect`). Unscored, same flat `1.0` per match as
`Clause::Wildcard`/`Clause::Prefix`/`Clause::Fuzzy`.
**Scope decision: a hand-built recursive-descent parser plus a backtracking
matcher, not the `regex` crate** (no `Cargo.toml` in this workspace depends
on it) — real Lucene's `RegExp` syntax is deliberately not PCRE/Perl regex
(no anchors, no lookahead, its own `~`/`&` operators standard `regex` lacks
entirely), so reusing `regex` would either silently accept syntax Lucene
rejects or need a translation/validation layer nearly as large as writing a
purpose-built parser — continuing the `fuzzy.rs`/`wildcard.rs` precedent of
a small, scoped, from-scratch matcher instead. **Exact subset supported**:
literals (with `\`-escaping), `.` (any single byte), `*`/`+`/`?` postfix
quantifiers, `[...]` character classes (with ranges and `^`-negation),
`(...)` grouping, `|` alternation. **Exact subset deliberately NOT
supported** (rejected with a parse error, not silently mis-parsed):
`{n,m}` bounded repetition, `~` complement, `&` intersection, named classes
— all would need real automaton machinery (complementation/intersection)
materially beyond this slice's backtracking-matcher scope; see
`regexp.rs`'s module doc for the full writeup, including the same
byte-vs-codepoint tradeoff `fuzzy.rs` already documents. Verified against
real Lucene via `fixtures/src/AppendRegexpManifest.java` (same append-only
pattern as `AppendFuzzyManifest.java`) running eleven real
`org.apache.lucene.search.RegexpQuery` cases against `body`'s real terms —
exact literal, **the single most important case**: the whole-term-match
convention (`ca` must not match `cat` as a substring, confirmed against real
Lucene's own output), `.`/`*`/`+`/`?` quantifiers, a `[...]` class,
two-and-three-way alternation, a no-match case, and a missing-field case —
asserting doc-for-doc agreement with this port's own `Clause::Regexp`
matching for every recorded case on the first run. See `docs/parity.md`'s
new row for the full accounting.

**Progress (task #44):** a minimal query-string parser —
`lucene-search/src/query_parser.rs::parse_query(input, default_field) ->
Result<Clause, ParseError>` — turning a hand-picked subset of classic
Lucene `QueryParser` syntax straight into this port's existing `Clause`
tree, reusing every already-ported query constructor
(`TermQuery`/`PhraseQuery`/`BooleanQuery`/`WildcardQuery`/`PrefixQuery`/
`FuzzyQuery`/`RegexpQuery`/`BoostQuery`) rather than adding a new query
shape. **This is not a port of `QueryParser.java`/`StandardQueryParser`**
(both are large JavaCC-derived grammars with range queries, configurable
operator precedence, and per-field analyzers) — it's a small,
from-scratch recursive-descent parser inspired by that syntax, scoped
down explicitly. **Boolean-operator style: `+`/`-`/bare-is-SHOULD only**,
not `AND`/`OR`/`NOT` — real Lucene supports both simultaneously with
precedence rules; picking one avoids the "half-supported mix" trap
(`AND`/`OR`/`NOT` parse as ordinary terms, not operators, in this
parser). Supports `field:term` and bare terms (an explicit
`default_field: Option<&str>` parameter, `None` making a bare term a
clean `ParseError::MissingField` rather than a silent guess), quoted
`"phrase terms"`, `field:term~`/`~N` fuzzy (`N` in `0..=2`, matching
`FuzzyQuery`'s own supported range), `field:prefix*` vs. `field:c?t`
wildcard disambiguation (a single trailing unescaped `*` and nothing
else special → `PrefixQuery`, anything else with `*`/`?` → `WildcardQuery`,
mirroring real `QueryParser`'s own split), `field:/pattern/` regexp
(Lucene's own delimiter convention), `(...)` grouping to arbitrary
nesting depth, and a `^boost` suffix on any atom. **Deferred with a parse
error, not silent misinterpretation**: range queries (`[a TO b]`/
`{a TO b}`), `AND`/`OR`/`NOT` as real operators, fractional-similarity
fuzzy (`term~0.8`), and any escaping beyond a single `\`-then-any-byte
rule. Verified by unit tests covering every grammar case plus malformed-
input cases (unclosed quote/paren/regexp, unmatched close paren, missing
field, invalid boost/fuzziness, unsupported range syntax) confirming a
clean `Err`, never a panic; a fixture-backed integration test
(`crates/lucene-search/tests/query_parser_fixtures.rs`) parses queries
against the real `fixtures/data/blocktree_index/` segment and confirms
they execute (via `search_boolean_query`) to the same doc sets as the
equivalent hand-built `Clause` values — the meaningful correctness check
for parser *syntax*, since there's no "real Lucene bytes" to
differentially decode here. See `docs/parity.md`'s new row for the exact
grammar accounting.

**Progress (task #45):** a minimal `DirectoryReader`/`SegmentReader`
now exists — `lucene-search/src/directory_reader.rs::{DirectoryReader,
SegmentReader, OpenedSegments}`. `DirectoryReader::open(dir)` finds the
latest `segments_N` (new `lucene_index::segment_infos::read_latest`, a
thin wrapper over the already-existing `lucene_store::directory::
read_latest_commit` + `parse`), opens one `SegmentReader` per listed
segment, and computes each segment's `doc_base` as the running sum of
previous segments' `maxDoc` — the two things task #41's `OpenSegment::
doc_base` doc comment left entirely to the caller. Per segment, only the
files that segment actually has get opened (checked against `SegmentInfo.
files`, not assumed): `.tim`/`.tip`/`.tmd`/`.doc`/`.pos`/`.pay` are opened
together or not at all (a segment with none of them, e.g. this port's
stored-fields-only fixtures, gets an empty `BlockTreeFields::empty()`);
`.liv` is opened only when `del_gen != -1`, reusing `lucene_index::
deletes::liv_file_name`. **Crate placement**: this lives in `lucene-search`,
not `lucene-index` — it has to hand back `lucene_search::multi_segment::
OpenSegment` values, and `lucene-index` cannot depend on `lucene-search`
(confirmed again here, same constraint as tasks #27/#36). **Two-call API**:
`DirectoryReader::open_segments()` returns an `OpenedSegments` owning the
freshly-opened `DocInput`/`PosInput`/`PayInput` values, and `OpenedSegments::
as_open_segments()` returns the final `Vec<OpenSegment>` — two calls, not
one, because `OpenSegment` holds `&'a DocInput<'a>` (a reference to an
already-constructed value), and storing that value inside `SegmentReader`
itself would be self-referential (illegal without `unsafe`, forbidden in
this crate). **Compound-file segments (`.cfs`/`.cfe`) were out of scope at
the time this task landed**: `SegmentReader::open` returned
`Error::CompoundFileUnsupported` rather than silently mis-reading —
packing/unpacking compound sub-files into this reader was more scope than
this task's "centralize what callers already did by hand" brief called
for. **Superseded by task #76 below**, which wires real compound-file
reads into `SegmentReader::open` and removes `Error::CompoundFileUnsupported`
entirely — see that entry for the current, accurate state. **Verified**: opens the real single-segment
`fixtures/data/blocktree_index/` fixture end-to-end and reproduces task
#41's `search_term_query_multi_segment` result; opens the real
`fixtures/data/live_docs_index/` fixture and confirms `.liv` is read and
cross-checked; opens the real `fixtures/data/compound_index/` fixture and
confirms it's rejected with a typed error; a hand-built stored-fields-only
segment (via `segment_writer::flush_stored_only_segment`) opens with no
postings files and no panic; `doc_base` running-sum computation checked by
opening the same real fixture segment twice under one hand-built
two-segment `SegmentInfos`; missing `.fnm` and partial `.tim`/`.tip`/`.tmd`
(some but not all three present) both surface as typed errors, not panics.
**Still deferred, same list as before plus this task's own scope line**:
soft deletes, compound-file segments (above), real index-wide
`CollectionStatistics`-based idf (task #41's own gap, unchanged), and
`lucene-ffi` exposure of this reader. NRT/reopen was closed by task #46
(`DirectoryReader::open_if_changed`, see its own `docs/parity.md` row) --
that task's own remaining scope line still applies (no reader-pool-wide
sharing, no warm-up hooks; each call only reuses its own receiver's
segments).

**Progress (task #47):** a `TieredMergePolicy`-equivalent merge-selection
function, `lucene-index/src/merge_policy.rs::{find_merges,
find_forced_merges}` — decides which segments should merge next; it does
not execute merges (still `merge.rs`'s job) or schedule them in the
background (`MergeScheduler` is out of scope). `SegmentStat` is a new,
deliberately unit-agnostic per-segment stat struct (name/doc_count/
del_count/size_bytes) rather than `SegmentCommitInfo` directly, because
`SegmentCommitInfo` (`segments_N`) carries `del_count` but not doc count or
byte size — those live in the separate per-segment `.si` file
(`segment_info::SegmentInfo`). **Size-unit decision**: real `TieredMergePolicy`
sizes segments by on-disk byte size; this port adds `segment_byte_size(dir,
info)`, which sums real file lengths for a segment's `.si`-listed files via
the existing `Directory` trait (no new trait method needed) — the honest,
byte-accurate option when a `Directory` is available. A caller without one
may instead approximate `size_bytes` via `doc_count`
(`SegmentStat::from_segment_info`), documented explicitly in the module doc
as an approximation, not real bytes; the algorithm itself doesn't care
which unit it's given. **Kept from real `TieredMergePolicy`**: excluding
already-oversized segments from further merge input; a reclaim-weighted
score (`size * (1 - reclaim_weight * del_count/doc_count)`) so a
heavily-deleted segment is preferred over an equally-sized clean one, not
naive size-only bin-packing; a `segments_per_tier` target that suppresses
merges once segment count is already at/below it; a `max_merge_at_once`
cap no proposed group ever exceeds; preferring smaller/more-deleted
segments first via sorting before grouping. **Simplified/dropped**: real
Lucene's exact `MergeScore` formula (log-based skew penalty, floor/ceiling
tier smoothing, iterative multi-candidate search with rollback) — this
uses one simpler, real-shaped score instead; one greedy pass building
merge groups rather than an iterative multi-merge search; no compound-file
awareness; `find_forced_merges` merges the excess down to a target count
in one group rather than real Lucene's own chunked, multi-pass forced-merge
walk. `MergePolicyConfig` defaults mirror real `TieredMergePolicy`'s own
(`maxMergeAtOnce=10`, `segmentsPerTier=10`, `maxMergedSegmentMB=5000`).
**Closing the loop with `merge.rs`**: a new integration test,
`lucene-index/tests/merge_policy_to_merge_integration.rs`, flushes three
real stored-fields-only segments, calls `find_merges` to pick all three,
then feeds that chosen name group straight into
`merge::merge_stored_only_segments` (re-reading each segment's files off
disk by name, as a real caller resolving names would) and confirms the
merged segment holds all six docs — proving `find_merges`' output shape
plugs into the existing merge-execution machinery with no adapter needed.
Not wired into an automatic merge-triggering pipeline (no `IndexWriter`
integration) — that's real Lucene's `MergeScheduler`, explicitly out of
scope for this task.

**Progress (task #48):** soft-delete **visibility**,
`lucene-search/src/soft_deletes.rs::{SoftDeletesField, is_soft_deleted,
is_live, effective_live_docs}` — real Lucene's
`IndexWriterConfig.setSoftDeletesField` convention: a document is invisible
to search if EITHER its hard-delete bit is cleared in `.liv` OR its
soft-deletes-field doc-values value is *present* (real Lucene's actual rule
is `DocValuesFieldExistsQuery`-shaped presence, not a marker-value compare).
`effective_live_docs` computes one combined `FixedBitSet` (hard-live AND NOT
soft-deleted) that plugs straight into any of this crate's existing
`live_docs: Option<&FixedBitSet>` parameters unchanged — no query function
needed a new parameter, and every pre-existing hard-delete-only call site is
unaffected. **Honest scope call, made explicitly rather than faked**: real
Lucene's soft-delete *write* path (`IndexWriter.softUpdateDocument`) relies
on `NumericDocValuesFieldUpdates` — an incremental per-doc-values-generation
delta file, not a full rewrite. This port's only doc-values write primitive
(`lucene_codecs::doc_values::write_single_dense_numeric_field`) always
writes a brand-new, complete, single-dense-field `.dvm`/`.dvd`/`.dvs` triple
from scratch; there is no incremental-update codec here at all. Rather than
build a fake "cheap incremental marking" shim on top of a full-rewrite
primitive, this task ships the read-side (**visibility**) half only, and
documents marking-a-doc-soft-deleted as deferred to whatever wrote the
segment's doc-values in the first place. Verified against the real,
checked-in `fixtures/data/doc_values_index/` fixture's genuinely sparse
(`IndexedDISI`) `sparse` numeric field (docs 0/2/4 present, 1/3 absent) —
not a hand-built/dense-encoded stand-in, since a dense field can't represent
"no value at all" and so can't stand in for real soft-deletes presence
semantics — plus an end-to-end composition test reusing the real
`blocktree_index` fixture's `body`/`bird` term query (docs `[1, 4]`)
together with that same real sparse field, confirming a soft-deleted doc is
excluded from real term-query results. See `docs/parity.md` for the full
row and scope writeup.

**Progress (task #54):** a numeric doc-values **update overlay** —
`lucene-codecs/src/doc_values_updates.rs::{write_numeric_updates,
read_numeric_updates, numeric_value_with_updates}` — closes task #48's
documented write-side gap above. Real Lucene's `NumericDocValuesFieldUpdates`
marks a doc's doc-values field with a new value by appending a small
"generation" file of sparse `(docId, value)` deltas rather than rewriting a
whole segment's `.dvd`/`.dvm` triple, with `SegmentCommitInfo.docValuesGen`
tracking generations and newest-generation-wins semantics across many update
rounds. This task ships exactly the core, single-generation mechanism: write
a sorted, de-duplicated `(docId -> newValue)` map to its own small file
(reusing `codec_util`'s header/footer/CRC shell for structural integrity);
read it back to a `HashMap`; and read a base numeric doc-values value
*through* the overlay (overlay wins, else fall back to the existing
`doc_values::numeric_value` decode) — proving the real "incremental update,
no full rewrite" property. **Byte format is this port's own invention**, not
a port of real Lucene's actual generation-file bytes — same honest,
documented situation as task #49/#52's index-sort format, and given the
scope decision below, there's no plan to derive a byte-exact format either.
**Scope, made explicit**: multiple sequential overlay generations with
newest-wins semantics across many rounds, and `SegmentCommitInfo`/`.si`
`docValuesGen` metadata wiring, are both deliberately deferred; this is a
single overlay round, not the full commit-lifecycle integration. **Wired
into task #48's soft-deletes flow**:
`lucene-search/src/soft_deletes.rs::{mark_soft_deleted_via_overlay,
is_soft_deleted_with_overlay, effective_live_docs_with_overlay}` mark a doc
soft-deleted via *only* an overlay write (zero base-file I/O) and extend the
existing presence check / combined-bitset computation to consult the
overlay first, falling back to the base decode — task #48's own write-side
gap is now partially closed (one overlay round, not the full incremental-
update lifecycle). Verified: overlay round-trip (including unsorted input,
duplicate-doc last-write-wins, empty overlay), corruption rejection (wrong
segment id, truncated file, hand-built out-of-order doc ids), overlay-vs-base
composition (override, fallback, no-op-when-empty) against a base field
built through the real `write_single_dense_numeric_field`/`parse_meta` round
trip, and the soft-deletes integration against the real checked-in
`fixtures/data/doc_values_index/` sparse fixture: marking doc 3 soft-deleted
via the overlay alone (no base rewrite) makes the overlay-aware checks
correctly exclude it while the plain base-only check still correctly reports
it as not soft-deleted, proving the base bytes were genuinely untouched. See
`docs/parity.md` for the full row and scope writeup.

**Progress (task #50):** basic faceting over a SORTED_SET doc-values field,
`lucene-search/src/facets.rs::{facet_counts, resolve_labels, top_n_facets,
FacetCount}` — a simplified port of real Lucene's
`SortedSetDocValuesFacetCounts` (`lucene-facet` module): for every doc in a
caller-supplied matching-doc-ID slice, `facet_counts` increments a counter
for every one of that doc's SORTED_SET ordinals (multi-valued docs increment
more than one counter, not just a "primary" one), `resolve_labels` turns
those per-ordinal counts into `FacetCount { ord, label, count }` via the
field's existing terms dictionary (`terms_dict::decode_all_terms`, the
`lookupOrd`-equivalent already in this port), and `top_n_facets` sorts
descending by count (ties broken by ascending ordinal, this crate's usual
determinism convention) and truncates — real Lucene's
`Facets.getTopChildren`. Thin aggregation only: no new format decoding, built
entirely on the already-decoded `sorted_numeric_values`/`decode_all_terms`
this port already had from tasks #4/#21/#31.

**Two scope calls, made explicitly:**
- **Single-segment only.** Real Lucene's faceting is index-wide because
  `SortedSetDocValuesReaderState` builds one merged ordinal map across every
  segment up front. This port has no such merged map — each segment's
  SORTED_SET terms dictionary assigns ordinals independently, so summing raw
  ordinal counts across segments would silently conflate unrelated terms
  that happen to share an ordinal number in different segments. `facet_counts`
  therefore counts one already-opened segment only; multi-segment callers
  must merge per-segment results **by resolved string label** (via
  `resolve_labels`), not by raw ordinal — a straightforward follow-up once a
  caller needs it, not implemented here.
- **Query-scoped counting is the only mode; "count everything" is the
  caller's trivial special case**, not a separate code path — pass every live
  doc ID as the matching-doc-ID slice to count the whole segment, exactly
  how real Lucene's `FacetsCollector` has no separate API distinct from
  running `MatchAllDocsQuery`.

Verified against real Lucene ground truth from the existing
`multi_valued_dv_index` fixture (task #31,
`fixtures/src/GenMultiValuedDocValues.java`): its manifest already records
each doc's real `SortedSetDocValues.nextOrd()` output
(`field.tags.ords`/`field.tags.terms`), written via a straightforward
per-doc iteration over real Lucene's own `SortedSetDocValues` — a genuine,
real-Lucene-computed ground truth without depending on the `lucene-facet`
module (not a project dependency). No new Java fixture generator was needed
since the existing manifest already had everything required. Coverage:
`facets.rs` 98.74% lines.

**Progress (task #58):** numeric range faceting, `lucene-search/src/facets.rs
::{NumericRange, range_facet_counts}` — a companion to task #50's SortedSet
facet counting, extending the same module rather than starting a new one. A
simplified port of real Lucene's `LongRangeFacetCounts`/
`DoubleRangeFacetCounts`: given a caller-supplied list of `NumericRange`s
(each with independently inclusive/exclusive `min`/`max` bounds and a label)
and a matching-doc-ID slice, `range_facet_counts` decodes each doc's value
via the already-existing `doc_values::numeric_value` (task #21/#31's numeric
decode, reused as-is — no new decoding), then checks it against every range
independently and increments that range's counter on a match. Output is
`(label, count)` pairs in the **same order as the input `ranges` slice** —
real Lucene's `FacetResult.labelValues` preserves caller-specified range
order rather than sorting by count, unlike `top_n_facets`'s deliberate
sort-and-truncate for the string-facet case.

**Semantics carried over from real Lucene, not simplified away:**
- **Ranges may overlap, and a doc in two ranges is counted in both.** Real
  Lucene never requires ranges to partition the value space; `range_facet_
  counts` makes one independent containment check per range per doc, with no
  notion of "the" bucket a doc belongs to.
- **A doc missing the field never counts, in any range — including an
  unbounded one.** Same missing-value rule `doc_value_query::search_numeric_
  range` already documents (`numeric_value` returning `None` skips the doc
  entirely), applied per-range here instead of to one range.
- **Empty `matching_docs` yields every range at count 0**, matching task
  #50's own empty-set convention (`facet_counts`'s all-zero-but-present
  result) rather than an empty `Vec`.

**Boundary handling** (`NumericRange::contains`) checks each end
independently: `min_inclusive`/`max_inclusive` each switch between `>=`/`>`
and `<=`/`<` on their own, so all four inclusive/exclusive combinations are
representable and were each tested precisely at the boundary value itself
using the real `doc_values_index` fixture's `varying` field (values -100, 7,
42, 1000, -3 for docs 0..4): `[42,42]` inclusive-inclusive matches (doc 2's
exact value), `[42,42)` inclusive-exclusive and `(42,42]` exclusive-inclusive
both correctly exclude it, and `(7,1000)` exclusive-exclusive still matches
doc 2 (42) while correctly excluding both endpoint docs (7 and 1000
themselves).

Verified with the same real-Lucene-recorded ground truth `doc_value_query.rs`
already established for the `doc_values_index` fixture's `varying`
(-100/7/42/1000/-3) and `sparse` (5/NONE/15/NONE/25) fields — reused directly
rather than re-deriving decode correctness, since that decode is already
differentially verified. Tests cover: non-overlapping ranges partitioning
docs correctly; an overlapping-ranges case where doc 2 (42) is counted in
both of two overlapping ranges; all four boundary combinations at the exact
value; the `sparse` field's missing docs never counting even under an
unbounded `[i64::MIN, i64::MAX]` range; a doc excluded from `matching_docs`
contributing nothing even under an unbounded range; empty `matching_docs`
producing all-zero counts; caller-specified range order preserved in the
output regardless of count; and decode-error propagation surfacing as `Err`.
Coverage: `facets.rs` 99.10% lines (up from 98.74% pre-task, both new
functions fully exercised).

**Progress (task #51, final task in this batch):** `lucene-ffi` exposure of
task #41's multi-segment search and task #45's `DirectoryReader` — the last
"lucene-ffi exposure" gap those two tasks' own doc comments (and this file's
task #45 write-up above) flagged as deferred. New module
`lucene-ffi/src/directory_reader.rs`, following tasks #20/#30/#40's exact FFI
pattern: a new `RegistryTag::DirectoryReader` handle/registry
(`registry::DirectoryReaderHandle`/`registry::directory_readers()`),
`ffi_open_directory_reader(path, path_len, out_handle)` (opens an internal,
short-lived `FsDirectory` at `path` and calls `DirectoryReader::open` on it —
a path string, not an already-open `ffi_open_directory` handle, since a
`DirectoryReader` copies every segment's bytes into its own owned buffers at
open time and needs the directory for no longer than that one call, unlike
`ffi_open_segment`'s directory-handle-reuse case), `ffi_close_directory_reader`,
and `ffi_search_term_query_multi_segment`/`ffi_search_boolean_query_multi_segment`
(same wire formats as their single-segment `_scored` siblings in `query.rs` —
one `(field, term)` pair / the same flat four-parallel-array clause lists —
plus a `top_n`).

**Results-handle reuse, not a new type**: both multi-segment entry points
write into the *existing* `ScoredResultsHandle`/`ffi_scored_results_len`/
`ffi_scored_results_copy`/`ffi_close_scored_results` trio task #30 already
shipped, rather than inventing a fourth results registry — multi-segment
search returns exactly the same `Vec<ScoreDoc>` shape single-segment scored
search already does (`multi_segment.rs`'s own module doc makes this explicit:
the merge step's output is indistinguishable in shape from any single
collector's `top_docs()`), so a new type would be a pure duplicate of an
already-correct wire contract.

**No norms**: task #45's `DirectoryReader`/`SegmentReader` carry no
`.nvm`/`.nvd` data at all (unchanged by this task), so every per-segment
norms slot passed to the two multi-segment search functions here is `None`
— the same documented `UNNORMED_FIELD_LENGTH` fallback this crate's
single-segment scored queries already use for a bare `None`, not a new
approximation.

**Avoiding task #29's flakiness pattern**: the new panic-injection
regression test (`registry_mutex_recovers_from_poisoning_after_a_panic_mid_multi_segment_query`)
uses a **thread-local** `Cell<bool>` switch
(`PANIC_ON_NEXT_MULTI_SEGMENT_QUERY`), armed and fired on the same test's own
thread only — following `query.rs`'s `PANIC_ON_NEXT_SCORED_TERM_QUERY`
precedent (added after task #29's process-wide-`AtomicBool` flakiness
history) rather than `query.rs`'s older, still-process-wide
`PANIC_ON_NEXT_TERM_QUERY`, since `cargo test`'s parallel thread pool can run
this crate's tests concurrently and a shared atomic armed by one test can
fire inside an unrelated, concurrently-running test's call to the same
function.

**Verified**: happy-path multi-segment term and boolean search against the
real single-segment `blocktree_index` fixture (reused as-is — task #41's own
tests already establish the "open one real fixture segment twice as two
segments" pattern for genuine multi-segment coverage, and this task's own
scope is the FFI wiring around already-verified `lucene-search` logic, not
new multi-segment correctness proof); wrong-tag handle rejection both
directions (a `ScoredResultsHandle` id passed as a reader handle, and a
`DirectoryReader` handle passed to `ffi_scored_results_len`); null-pointer,
unknown-handle, and double-close cases for every new entry point; the
poison-recovery regression test above. Coverage: `directory_reader.rs`
96.04% lines (13 new tests; `lucene-ffi` crate total 156 passing tests).

**Progress (task #55):** the `SpanQuery` family --
`lucene-search/src/query.rs::SpanQuery` (`SpanTerm`/`SpanNear`/`SpanOr`) plus
`Clause::Span`, wired into `resolve_clause_docs`/`clause_scores` following the
`Wildcard`/`Prefix`/`Fuzzy`/`Regexp` precedent (flat `1.0`-per-match scoring,
no new scoring machinery) -- a genuinely different query family from
`PhraseQuery` (task #19/#28): instead of "does this doc match", a span
query's result is the actual matching **span ranges** (`[start, end)`
position pairs), composable (a `SpanNear` of `SpanNear`s, etc).

**Scope decision, made explicitly**: real Lucene's `Spans` is a lazy
iterator API (`nextStartPosition`/`nextDoc`/`advance`, buffered
`NearSpansOrdered`/`NearSpansUnordered` merge state) -- substantial machinery
out of scope here. This port instead computes span matches **directly
against a doc's already-decoded position lists**
(`lucene-search/src/lib.rs::span_matches_in_doc`), the same "compute matches
directly against decoded data" shape `phrase_matches_in_doc`/
`phrase_matches_in_doc_sloppy` already use for `PhraseQuery` -- an
honestly-scoped MVP, not a lazy-iterator redesign. `span_doc_ids` (the
`Clause::Span` doc-ID resolver) takes every leaf `SpanTerm`'s doc list as a
safe, simple over-approximation of the candidate set (rather than a
tighter, per-variant candidate computation) -- correctness first, profile
before optimizing, same call this crate's other multi-term matchers already
make.

**The `in_order == false` differentiator**: `SpanNearQuery`'s `inOrder`
flag genuinely supports both in-order and any-order proximity search --
`in_order == false` allows sub-spans in **any** relative order within the
`slop` budget, a capability `PhraseQuery`'s own sloppy matching (task #28)
deliberately does *not* have (that was explicitly scoped to in-order-only).
`span_near_matches` implements both: `in_order == true` requires sub-spans
non-overlapping and increasing in the query's own clause order;
`in_order == false` sorts the chosen sub-spans by start position first, then
applies the same non-overlapping/slop check -- any relative order among
clauses is accepted. The total-slack formula
(`sum(next.start - prev.end)` over adjacent arranged spans) generalizes
`phrase_matches_in_doc_sloppy`'s single-position "moves needed" accounting
to `[start, end)` span ranges.

**Cross-engine verified** (`crates/lucene-search/tests/span_query_fixtures.rs`,
reusing the `blocktree_index` fixture's `pos` field): `GenBlockTree.java`
gained doc8 (`"delta"@0`, `"gamma"@1` -- occurrence order deliberately
reversed relative to a `SpanNearQuery` built with clauses in `[gamma,
delta]` order) plus `field.pos.span.*` manifest keys, recorded by *actually
running* real `org.apache.lucene.queries.spans.SpanNearQuery`/`SpanOrQuery`
against this fixture at generation time (`lucene-queries` module, not
`lucene-core`). Real Lucene's own verdict: `SpanNearQuery([gamma, delta],
slop=0, inOrder=true)` does NOT match doc8;
`SpanNearQuery([gamma, delta], slop=0, inOrder=false)` DOES match -- exactly
the `in_order` differentiator this task's own scoping flagged as most
likely to be subtly wrong if hand-derived, and this port's implementation
agrees with real Lucene on both verdicts. Unit tests
(`lucene-search/src/lib.rs`) additionally cover: `SpanTerm` matching every
occurrence in a multi-occurrence doc; `SpanNear` slop-boundary exactness
(exactly-at-limit matches, one-over doesn't); `SpanOr` union semantics
(either/neither/both sub-spans); nested `SpanNear`-of-`SpanNear` composition.
Coverage: `lucene-search/src/lib.rs` 96.10% lines, `query.rs` 98.34% lines
(workspace total 97.23% lines, `cargo llvm-cov --fail-under-lines 95`
passing). See `docs/parity.md` for the full row and scope writeup.

**Progress (task #56):** `lucene-search/src/highlighter.rs` -- fragment
assembly on top of task #39's `matched_term_offsets` primitive, the
follow-up that primitive's own doc comment explicitly deferred. Given the
original field text (read separately, e.g. from
`lucene-codecs/src/stored_fields.rs`'s `StoredFieldsReader`) plus a set of
`TermOffsetSpan`s, `assemble_fragments` slices out fixed-size character
windows around each match (or cluster of nearby matches), wraps the matches
in caller-configurable `pre`/`post` markers (default `<b>`/`</b>`, real
Lucene's `PassageFormatter` default), and merges overlapping windows into
one fragment instead of emitting duplicates -- the one piece of this
logic that's genuinely easy to get wrong, since inserting multiple
highlight markers into one merged fragment means later insertions must not
invalidate earlier ones (`render_cluster` inserts back-to-front, by match
position, so each insertion's byte offsets stay valid for the next).

**Scope decision, made explicitly**: this is a simplified passage-boundary
heuristic, not real Lucene's `BreakIterator`-based sentence detection --
window edges are snapped outward to the nearest whitespace so a fragment
doesn't start/end mid-word, but there is no sentence awareness and no
term-density passage scoring; fragments are emitted in left-to-right
document order and truncated at `max_fragments`.

**Offset-unit finding**: term-vector offsets are decoded verbatim off disk
by task #3 and never reinterpreted by task #39, so they carry whatever unit
real Lucene's indexing-time `Analyzer` wrote (UTF-16 code units in real
Lucene). This port's checked-in fixture is ASCII-only, so UTF-16-code-unit/
UTF-8-byte/Unicode-scalar counts are indistinguishable there. `highlighter.rs`
picks Unicode-scalar (`char`) count as its contract going forward (matches
UTF-16 for the entire Basic Multilingual Plane) and converts to UTF-8 byte
offsets via `char_indices()` before ever slicing `full_text`, so a match
spanning a multi-byte UTF-8 character cannot panic even on out-of-range or
mis-unitted input (offsets are clamped to the text's char count first).

Unit-tested: single match windowed/highlighted correctly; two nearby
matches merge into one fragment with both terms highlighted, verified
precisely (markers around both terms, neither original word corrupted);
two far-apart matches stay in separate fragments; window clamping at text
start/end without panicking; `max_fragments` truncation; a multi-byte UTF-8
match; out-of-range/invalid spans dropped defensively rather than panicking;
one test composes task #39/#3's real, cross-engine-verified fixture offsets
(`fixtures/data/term_vectors_index/`, doc 0's "text" field -- terms "cat"/
"car"/"cat" at char offsets 0..3/4..7/8..11) with the real text those exact
offsets denote (`"cat car cat"`, per `fixtures/src/GenTermVectors.java`'s
`CannedTokenStream`), rather than a made-up string. No new Java fixture was
generated for the fragment-assembly logic itself, per the
`differential-testing` skill's precedent for presentation-layer composition
over already-differentially-verified data (there is no "real Lucene bytes"
to check string-slicing/highlighting against). Coverage:
`lucene-search/src/highlighter.rs` 98.67% lines (workspace total 97.25%
lines, `cargo llvm-cov --fail-under-lines 95` passing). See
`docs/parity.md` for the full row.

**Progress (task #57):** `lucene-index/src/check_index.rs` -- a
`CheckIndex`-equivalent: a standalone consistency verifier that opens a
segment and cross-checks internal relationships a normal single-purpose
open never bothers to verify. Deliberately *not* built on top of
`lucene-search`'s `DirectoryReader`/`SegmentReader` (task #45) (those types
only expose the curated subset of state a query needs and hide exactly
what a self-check needs to cross-reference -- `SegmentInfo.files`,
per-field flags, raw `.si`/`.fnm`/stored-fields bytes). Lives in
`lucene-index`, not `lucene-search` (which it has no actual dependency on
-- every type it composes is already available here), and reuses
`crate::segment_infos::read_latest` for the shared "find the latest
commit, list its segments" piece, otherwise opening each segment's files
directly through the same lower-level decoders `lucene-search`'s
`directory_reader.rs` itself calls (`segment_info::parse`,
`field_infos::parse`, `live_docs::parse`, `stored_fields::open`), since
those are exactly the values worth comparing against each other.

Checks implemented, each reported as an independent named pass/fail (not
a single boolean, matching real `CheckIndex`'s per-check `Status` style):
every file `.si` lists opens and has a structurally valid codec footer;
`.si` doc_count vs `.liv`'s byte-size-implied word count (computed
independently of `si.doc_count`, not by construction); `live_docs`
cardinality vs `SegmentCommitInfo.del_count` (surfaced via
`live_docs::parse`'s own `DelCountMismatch`, since that cross-check is
already enforced at decode time -- this module reports it under its own
check name rather than re-deriving it); `.fnm`'s per-field flags (doc
values, norms, term vectors, postings via `index_options`) cross-checked in
both directions against which of `.dvd`/`.dvm`/`.nvd`/`.nvm`/`.tvd`/`.tvx`/
`.tvm`/`.tim`/`.tip`/`.tmd` the segment's file list actually includes (a
field claiming a feature with no matching files is flagged, and so is a
file group present with no field claiming it); stored-fields reader's own
`max_doc()` vs `.si`'s declared `doc_count`. A `.si` that fails to
open/parse short-circuits every other check for that segment (nothing else
can be trusted without a valid file list), reported as a single `si.open`
failure.

**Deliberately deferred** (this port's honest scope, not an oversight):
postings term-by-term re-derivation (recomputing docFreq/totalTermFreq from
raw postings and cross-checking against the term dictionary's own recorded
stats -- real `CheckIndex`'s single most expensive check), doc-values
value-range sanity, points-tree structural invariants, and vectors-graph
structural invariants. Each requires walking per-format internals in a
different shape (blocktree iteration, points-tree traversal, HNSW graph
traversal) -- a separate task per format, not a natural extension of this
module's cross-file bookkeeping checks.

Unit-tested (no new Java fixture -- this is self-consistency logic over
already-differentially-verified decoders, not new byte decoding, per the
`differential-testing` skill's precedent): the real `blocktree_index` and
`live_docs_index` fixtures pass every check cleanly; deliberately corrupted
inputs confirm each failure mode reports clearly rather than panicking or
false-passing -- a hand-mutated `SegmentCommitInfo.del_count`, a truncated
`.liv` file, a `.si` listing a file that doesn't exist, a `.si` that won't
parse at all, a `.liv` whose byte size disagrees with `si.doc_count`, a
file with a corrupted footer, a partial `.fdt`/`.fdx`/`.fdm` file set, and a
`.si.doc_count` that disagrees with a real stored-fields reader's `max_doc()`.
Coverage: `lucene-index/src/check_index.rs` 96.97% lines (workspace total
97.25% lines, `cargo llvm-cov --fail-under-lines 95` passing). See
`docs/parity.md` for the full row.

**Progress (task #59):** query `explain()` — `lucene-search/src/explain.rs`,
a new `Explanation`/`explain_clause` pair mirroring real
`IndexSearcher.explain(query, doc)` / `org.apache.lucene.search.Explanation`
exactly: `{ matched: bool, value: f32, description: String, details:
Vec<Explanation> }`, with `Explanation::match_(value, description)` /
`.with_details(vec)` / `Explanation::no_match(description)` matching real
Lucene's own `Explanation.match(...)`/`Explanation.noMatch(...)` factory
split (`matched` is this port's stand-in for real Lucene's `isMatch()`).

**This task changes no scoring behavior** — it is purely introspection over
the already-verified BM25/boolean/dismax math from tasks #13/#22/#23/#29/#32.
`explain_clause` recomputes each node's `value` by calling the *exact same*
`similarity::idf`/`similarity::tf_norm` functions (and the same
`term_doc_freqs`/`term_doc_positions`/`phrase_freq_exact`/
`matched_boolean_docs`/`resolve_clause_docs` helpers) `lib.rs`'s
`search_term_query_scored`/`search_boolean_query_scored`/
`search_phrase_query_scored`/`search_disjunction_max_query_scored` already
call, in the same argument order — so its reported top-level `value` is
bit-for-bit identical to those functions' own output for the same doc, not a
second, independently-computed approximation. Verified directly: this
module's own unit tests build a query, run it through the real scored
search function to get ground-truth `(doc, score)` pairs, then call
`explain_clause` for that exact doc and `assert_eq!` (not an epsilon
compare) the two values — for `Clause::Term` (with its nested `idf`/
`tfNorm` sub-explanations further asserted to multiply back to the same
top-level value), `Clause::Phrase` (multi-term, using the real
`"alpha beta"`-in-doc-8555 fixture match `lib.rs`'s own phrase tests
already established), `Clause::Boolean` (must+should, sub-clause values
summing to the top-level value), `Clause::DisjunctionMax` (max +
tie-breaker*sum(rest)), `Clause::ConstantScore`, and `Clause::Boost`.

**Which `Clause` variants get a real vs flat explanation** (see
`explain.rs`'s own module doc comment): `Term`/`Boolean`/`Phrase`/
`DisjunctionMax`/`ConstantScore`/`Boost` get a full breakdown (weight →
score → idf/tfNorm sub-trees for `Term`/`Phrase`; sum-of-clauses for
`Boolean`; max-plus-tie-breaker for `DisjunctionMax`; wrap-and-relabel for
`ConstantScore`/`Boost`). `Wildcard`/`Prefix`/`Fuzzy`/`Regexp`/`Span` get a
flat one-level "matches, constant score 1.0" or "no match" explanation —
these have no single term's frequency/idf to break down further (same
"unscored, flat 1.0" rationale each query type's own `query.rs` doc comment
already states for scoring).

No new Java fixture generator was needed (per the `differential-testing`
skill's precedent for presentation-layer logic over already-differentially-
verified scoring: there is no "real Lucene bytes" to check a description
string against, only the numeric equality this task's tests already assert
directly against this crate's own ground truth). Coverage:
`lucene-search/src/explain.rs` 97.82% lines (workspace total 97.28% lines,
`cargo llvm-cov --fail-under-lines 95` passing). See `docs/parity.md` for
the full row.

**Progress (task #60):** `BooleanQuery`/`BooleanWeight` edge cases --
investigation only, no production fix landed. Six real-Lucene corner cases
were each independently checked against this port's actual
`matched_boolean_docs`/`should_match_counts`/`clause_scores` code (not
assumed correct): a pure `must_not`-only query matches nothing (confirmed
before `must_not` is ever consulted); `minimum_should_match > 0` with zero
`should` clauses matches nothing, not "trivially satisfied"; `minimum_
should_match` exceeding `should.len()` has no distinct code path from the
in-range case, so no off-by-one is possible past the boundary; a doc
matching every required clause plus a `must_not` clause is still excluded;
a nested `Clause::Boolean`'s own `must_not` doesn't leak into or get
leaked into by an outer level's `must_not` (independently verified, not a
case where both correct and buggy behavior would coincide); and a literal
duplicate `should` clause double-counts toward `minimum_should_match` and
double-scores -- confirmed as real Lucene's own actual (non-deduping)
behavior, not a bug this port needed to suppress. All six were already
correct; new regression tests lock each one in (`lib.rs`, `explain.rs`) so
a future change to this recursion can't silently regress them. Coverage:
`lucene-search/src/lib.rs` 96.23% lines, `lucene-search/src/explain.rs`
98.54% lines (workspace total 97.31% lines). See `docs/parity.md`'s
`BooleanQuery` row for the itemized findings.

**Progress (task #61, final task in this batch):** the analyzer chain --
`lucene-analysis/src/lib.rs` fills in the previously-empty
`crates/lucene-analysis` crate: `Token { term, start_offset, end_offset,
position_increment }`, a simplified word-boundary `tokenize()` (split on
non-alphanumeric boundaries, keep alphanumeric runs, char offsets -- the
core algorithm of `StandardTokenizer`/`WhitespaceTokenizer`, not full UAX#29
segmentation, which stays out of scope), `LowerCaseFilter`, `StopFilter`,
and `Analyzer::standard(stopwords)` composing them, mirroring
`StandardAnalyzer`. The crate stays dependency-free, sitting below both
`lucene-index` and `lucene-search` in the workspace's downward graph so
either could depend on it without a cycle -- neither does yet, since
wiring an `Analyzer` into `query_parser.rs` or a not-yet-built indexing
tokenization step is separate follow-on work; every existing "terms" input
in this port (query parser, term-vector fixtures) still takes
already-tokenized terms directly, unchanged by this task.
`StopFilter`'s position-increment-preservation rule (the subtle,
easy-to-invert one): a removed stopword's own `position_increment` is
carried onto the *next surviving* token instead of being dropped, so
`PhraseQuery`/`SpanNear` slop math stays correct across stopword removal.
Verified against real Lucene: `fixtures/src/GenAnalysis.java` runs a real
`StandardAnalyzer` with a real stopword set over six cases (stopword
mid-sentence, leading, trailing, three consecutive, all-stopwords, and a
mixed-case/punctuation sentence with none removed), and
`crates/lucene-analysis/tests/analysis_fixtures.rs` asserts this port's
`Analyzer` produces byte-identical (term, position_increment, offsets)
sequences -- all six passed on the first real-Lucene run. Coverage:
`lucene-analysis/src/lib.rs` covered by 12 unit tests plus the fixture
test; see `docs/parity.md`'s new `lucene-analysis` section for the full
scope table (ported vs. deferred: stemming, synonyms, ASCII-folding, and
per-field analyzer configuration are all out of scope for this slice).

**Progress (task #62):** wired task #61's `Analyzer` into
`crates/lucene-search/src/query_parser.rs`, the first real consumer of
`lucene-analysis` outside `lucene-core`. `lucene-search/Cargo.toml` gained a
`lucene-analysis` path dependency (a clean downward edge -- `lucene-analysis`
has zero workspace deps, so no cycle). `parse_query` is unchanged and now
just delegates to a new, additive entry point,
`parse_query_with_analyzer(input, default_field, analyzer: Option<&Analyzer>)`;
`None` (every existing call site, via `parse_query`) preserves the exact
pre-task-#62 literal-term behavior byte-for-byte -- confirmed by running the
full existing `query_parser.rs` test suite unmodified (all 41 pre-existing
tests still pass) plus a new test that directly compares `parse_query` and
`parse_query_with_analyzer(.., None)` output on the same input. When an
analyzer is supplied, it runs over: (a) a plain bareword's text before
deciding `Clause::Term`, and (b) each whitespace-separated word of a quoted
phrase's text before building `Clause::Phrase` (real `QueryParser` analyzes
phrase text word-by-word too, not as one blob, so the original phrase-word
boundaries are preserved and each word gets independent zero/one/multi-token
handling, then results are spliced flat into the phrase's term sequence in
order). It deliberately does **not** run over wildcard/prefix (`c*t`),
fuzzy (`cat~`), or regexp (`/ca.*/`) pattern text -- verified with tests using
uppercase letters and a stopword-shaped substring (`the`) inside pattern text
that must survive untouched, since real Lucene's classic `QueryParser` never
analyzes those either (tokenizing/lowercasing/stopword-filtering glob or
regex syntax would corrupt the pattern). Zero/one/multi-token handling per
analyzed bareword or phrase word (`clause_from_analyzed_terms`/inline splice
in `parse_phrase`), a deliberately simplified subset of real
`QueryParserBase.newFieldQuery`'s fuller multi-token handling (which can also
build position-aware `SynonymQuery`s in some cases -- out of scope): exactly
one token becomes a `Clause::Term`/one phrase position; zero tokens (the
bareword or phrase-word analyzed away entirely, e.g. it was itself a
stopword) becomes `no_match_clause()`, an empty `BooleanQuery` (no
`must`/`should`/`must_not`) -- already a well-established "matches nothing"
shape in this crate per `matched_boolean_docs`'s own doc comment, so no new
`Clause` variant was needed; more than one token becomes a `Clause::Phrase`
in order (for a bareword) or is spliced into the surrounding phrase's term
list (for one word of an already-multi-word phrase). New unit tests in
`query_parser.rs`: lowercase-only analyzer lowercases a bareword; a
stopword bareword yields the no-match empty-`BooleanQuery` clause, not a
panic; a hyphenated bareword (`state-of-the-art`) that the tokenizer splits
into multiple tokens, with one token also a stopword, produces a
`Clause::Phrase` in the correct order; wildcard/prefix/fuzzy/regexp pattern
text is untouched by the analyzer; a quoted phrase's words are analyzed
per-word (one word dropped as a stopword, others lowercased); a phrase made
entirely of stopwords collapses to the no-match clause. Coverage:
`query_parser.rs` 96.92% lines (workspace total 97.33% lines, gate is 95%).
No new Java fixture -- this is composition of two already independently
cross-engine-verified pieces (task #44's parser grammar, task #61's analyzer
chain, both already verified against real Lucene separately); the
zero/one/multi-token control flow this task adds is new Rust-level logic
verified directly by the unit tests above, not new byte-format decoding that
would need a fixture.

**Progress (task #63):** an in-memory tokenize-and-invert *builder*,
`crates/lucene-index/src/indexing_chain.rs::invert_documents`, real Lucene's
`DocumentsWriterPerThread`/`IndexingChain`'s job of running a document's
indexed field text through an `Analyzer` and grouping the result into
`InMemoryInvertedIndex { terms: BTreeMap<(field, term), Vec<PostingEntry>> }`
-- each `PostingEntry` a doc's `doc_id` plus its occurrences' resolved
absolute positions and character offsets (`term_freq()` is
`occurrences.len()`). `lucene-index/Cargo.toml` gained the same clean
downward `lucene-analysis` path dependency `lucene-search` already has (no
cycle, since `lucene-analysis` has zero workspace deps). **Scope reality,
stated explicitly so this doesn't overclaim:** `segment_writer.rs` still has
no write-side postings encoder at all (every flushed field is
`IndexOptions::None`, per that module's own "what this deliberately is not"
section, unchanged by this task) -- so there is no path from this new
in-memory structure to any file on disk. Nothing in this port today is
indexed/searchable via analyzed text as a *result* of this task; what exists
now is the tested tokenize-and-group logic a future postings writer (task
#75) will need as its input, with an output shape (doc-ID-sorted
`Vec<PostingEntry>` per term, each carrying freq/positions/offsets) chosen
to match what `Lucene104PostingsWriter`'s `.doc`/`.pos`/`.pay` encode needs
directly, so that writer can consume it without re-deriving doc ordering or
re-grouping occurrences into frequencies itself. Verified by unit tests
only (no new Java fixture -- composition of task #61's already
cross-engine-verified analyzer plus this task's own new Rust-level grouping
logic, same precedent as task #62): single doc/field exact shape, multiple
docs sharing a term produce a doc-ID-sorted list, a repeated term's
`term_freq`/positions are all recorded (not just the first), independent
per-field entries for the same term text, and stopword-filtered text
excludes the stopword while preserving surviving tokens' positions.
Coverage: `indexing_chain.rs` 100% lines/functions/regions (workspace total
97.35% lines, gate is 95%).

**Progress (task #64):** `AsciiFoldingFilter`, a third `lucene-analysis`
filter alongside `LowerCaseFilter`/`StopFilter` from task #61, mirroring
real `org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter`.
**Scope, itemized rather than "some diacritics"**: the full Latin-1
Supplement letter block (U+00C0-U+00DE / U+00E0-U+00FE, skipping `×`/`÷`),
plus a documented Latin Extended-A subset -- Polish (Ą/ą, Ć/ć, Ę/ę, Ł/ł,
Ń/ń, Ś/ś, Ź/ź, Ż/ż) and Czech/Slovak/Baltic caron forms (Š/š, Č/č, Ž/ž,
Ď/ď, Ť/ť, Ň/ň). `Æ`/`æ` and `Œ`/`œ` fold to **two** ASCII characters
(`AE`/`ae`, `OE`/`oe`), and `ß` folds to `ss` -- both real Lucene's actual
special-case behavior, verified against real `ASCIIFoldingFilter`, not
guessed. **Deferred**: the rest of real Lucene's table (remaining Latin
Extended-A/B, Latin Extended Additional, and non-Latin-script folding) --
a character outside this table passes through unchanged, never dropped,
never a panic. **Offsets are not adjusted for folding-driven length
changes**: `æther` folds to `aether` (5 chars -> 6), but `start_offset`/
`end_offset` still denote the *original* source span, matching real
Lucene's `ASCIIFoldingFilter` (it never touches `OffsetAttribute`).
**Filter ordering decision**: `Analyzer::with_ascii_folding()` inserts
folding *before* lowercasing (fold -> lowercase -> stopwords), so an
uppercase accented letter (`É`) folds straight to its ASCII letter (`E`)
and is lowercased in the same subsequent pass as every other token, and
stopword matching (last in the chain) always sees the fully
folded-and-lowercased form. Folding is **off by default** --
`Analyzer::standard(stopwords)` is unchanged, so `query_parser.rs` (task
#62) and `indexing_chain.rs` (task #63) keep their exact prior behavior;
callers opt in via the new `.with_ascii_folding()` builder method.
Verified against real Lucene: `fixtures/src/GenAnalysis.java` gained two
new cases using a hand-built `Analyzer` subclass wiring real
`StandardTokenizer` + `ASCIIFoldingFilter` (`fold_only`, case preserved,
over "café naïve Müller cœur straße") and the same plus real
`LowerCaseFilter` (`fold_then_lower`, over "Café Naïve ÉCOLE") --
`crates/lucene-analysis/tests/analysis_fixtures.rs` asserts this port's
`AsciiFoldingFilter`/`Analyzer::with_ascii_folding()` produce the same
(term, position_increment, offset-span) sequences, both passing on the
first real-Lucene run. **Offset-unit reconciliation needed for this
fixture specifically** (documented in the test, `char_offsets_to_byte_offsets`):
this crate's `tokenize()` emits UTF-8 *byte* offsets (despite its own doc
comment calling them "character offsets" -- a pre-existing, harmless
mislabel that only becomes visible once non-ASCII text is involved, since
every prior fixture was ASCII-only where the two units coincide), while
real Lucene reports `char`/UTF-16-code-unit offsets; the test converts
real Lucene's char offsets to byte offsets via the fixture's own text
before comparing -- the same kind of documented byte-vs-codepoint scope
call `fuzzy.rs`/`wildcard.rs` already make elsewhere in this port, not a
new bug. Unit tests (`lib.rs`) cover: each Latin-1 spot-check
(café/naïve/Müller/ñ), the eszett special case, a ligature growing the
term's character count while offsets stay put, a plain-ASCII token passing
through untouched, mixed diacritic+ASCII in one token, a non-table
character (Cyrillic) passing through unchanged, the composed
fold-then-lowercase order, and the unchanged no-folding default. Coverage:
`lucene-analysis/src/lib.rs` 100% lines (28 unit tests + 3 fixture tests).

**Progress (task #65):** `PorterStemFilter`, a fourth `lucene-analysis`
filter alongside `LowerCaseFilter`/`StopFilter`/`AsciiFoldingFilter`,
mirroring real `org.apache.lucene.analysis.en.PorterStemFilter`. **All five
steps of the classic 1980 Porter algorithm are ported, not a subset**: step
1a (`-sses`->`-ss`, `-ies`->`-i`, `-s`-> delete), step 1b (`-eed`->`-ee`
under `m>0`; `-ed`/`-ing` deleted only if the stem has a vowel, with the
at/bl/iz-append, double-consonant-drop, and CVC-append cleanup that
follows), step 1c (`-y`->`-i` if the stem has a vowel), step 2 (the
`-ational`/`-tional`/... 20-entry suffix table, `m>0`), step 3
(`-icate`/`-ative`/... , `m>0`), step 4 (`-al`/`-ance`/`-ion` (only after
s/t)/... removal, `m>1`), step 5a (final `-e` dropped under `m>1`, or `m==1`
and not CVC), step 5b (`-ll`->`-l` under `m>1`). Implemented as a private
`porter` submodule (`is_consonant`/`measure`/`contains_vowel`/
`ends_double_consonant`/`cvc`/`try_step` helpers plus one function per
step), operating on `Vec<char>` for correct Unicode-scalar indexing.
**Domain of definition, stated explicitly**: the algorithm (and Lucene's own
port of it) is only defined over lowercase ASCII alphabetic words -- a term
containing any non-ASCII-alphabetic character or uppercase letter passes
through unchanged (never panics). In the normal analyzer chain this is a
non-issue since `PorterStemFilter` runs after `LowerCaseFilter`. **Filter
ordering**: `Analyzer::with_stemming()` inserts stemming *last* (tokenize ->
fold -> lowercase -> stopwords -> stem), matching real Lucene's
`EnglishAnalyzer` (its stop set holds unstemmed words like `"the"`, so
stopword matching must see pre-stem terms). Off by default -- existing
callers (`query_parser.rs`, `indexing_chain.rs`) are unaffected; opt in via
`.with_stemming()`. **Verification approach**: direct unit tests against
known Porter-algorithm input/output pairs rather than a new Java fixture --
this is a purely algorithmic, non-file-format task (no on-disk bytes to
diff), and the test vocabulary is drawn from the algorithm's own canonical
worked examples (Porter's 1980 paper's step 2-4 illustration list, e.g.
"relational"->"relat", "operator"->"oper", "triplicate"->"triplic"),
independently traceable step-by-step against the implementation rather than
guessed. Unit tests cover: step 1a plurals (caresses/ponies/cats/caress),
step 1b's `m`/vowel guards including words that must **not** stem
(feed/bled/sing all fail their respective conditions and stay unchanged;
agreed->agre, plastered->plaster, motoring->motor do stem), the full step
2/3/4 suffix-family table (47 canonical pairs), step 5's final-e/double-l
edge cases (rate keeps its `e` since `m==1` and it is CVC; roll keeps one
`l` since `m==1` not `>1`), offsets/position-increment left untouched,
non-lowercase-ASCII pass-through (uppercase, accented, empty, digit-only
terms), and the composed `Analyzer::with_stemming()` running after
`StopFilter`. Coverage: `lucene-analysis/src/lib.rs` 99.11% lines (61 unit
tests total + 3 fixture tests; workspace total 97.36% lines, gate is 95%).

**Progress (task #66):** `SynonymFilter`, a fifth `lucene-analysis` filter
alongside `LowerCaseFilter`/`StopFilter`/`AsciiFoldingFilter`/
`PorterStemFilter`, a scoped-down version of real
`org.apache.lucene.analysis.synonym.SynonymFilter`/`SynonymGraphFilter`:
single-word-to-single-word synonym injection only. **Scope, stated
explicitly**: real Lucene's full `SynonymGraphFilter` also handles
multi-word synonym *phrases* (`"New York"` <-> `"NYC"`) via a graph token
stream -- legitimately out-of-scope machinery for this task. This filter
takes a caller-supplied `HashMap<String, Vec<String>>` and, for each token
whose term is a map key, injects one additional token per configured
replacement immediately after the original, with `position_increment == 0`
and the same `start_offset`/`end_offset` as the original -- real Lucene's
own convention for representing "these two tokens are alternatives at the
same spot" so `PhraseQuery`/`SpanNear` built against either term still
aligns with surrounding words. This is the first token in the crate with
`position_increment == 0` (every prior token, including `StopFilter`'s
carried-over increments, has been `>= 1`). **Bidirectionality is explicitly
NOT automatic**, matching real Lucene's `SynonymMap`: configuring
`"quick" -> ["fast"]` does not also expand `"fast"` to `"quick"`; a caller
wanting symmetric synonyms configures both directions themselves. **Filter
ordering**: `Analyzer::with_synonyms()` inserts synonym expansion *last*
(tokenize -> fold -> lowercase -> stopwords -> stem -> synonyms), for two
reasons -- (1) real Lucene's convention is that synonym expansion runs over
already-normalized terms, so the caller-supplied map's keys are expected to
already be lowercased/stemmed; (2) running after `StopFilter` means a term
that is itself a stopword (and thus removed) never gets its synonym
expanded, since expanding a term about to be dropped would produce an
orphaned synonym token with no corresponding original. **Verification
approach**: no new Java fixture -- the position_increment==0 injection is
new Rust-level control flow over an already cross-engine-verified position
system (`StopFilter`'s increment mechanics were differentially verified in
task #61), not new byte-format decoding, so unit tests are the right tool
here. Unit tests cover: a token with one configured synonym produces two
tokens (original at its own increment, synonym at increment 0); a token with
multiple synonyms produces the original plus all of them, all at increment
0; a token with no configured synonym passes through unchanged; synonym
expansion is confirmed NOT automatically bidirectional; injected-token
offsets match the original's exactly; composed with `StopFilter` (a
stopword's synonym is never expanded since the stopword is removed first,
and a surviving term's synonym is correctly carried through with the
right accumulated position_increment); and composed with `PorterStemFilter`
(the synonym map's key matches the *stemmed* form, proving synonyms see
post-stemming terms). Coverage: `lucene-analysis/src/lib.rs` 99.44% lines
(39 unit tests total + 3 fixture tests; workspace total 97.38% lines, gate
is 95%).

**Progress (task #67):** `TermInterner`, a new standalone byte-sequence
interning pool in `lucene-util/src/term_interner.rs`
(`TermInterner`/`TermId`) -- **not** a byte-for-byte port of
`org.apache.lucene.util.BytesRefHash` and **not yet wired into any indexing
or query path**. Real `BytesRefHash` is bound to Lucene's `ByteBlockPool`
arena allocator and carries sort/compact/rehash machinery for the indexing
chain's per-field term dictionaries; that machinery is out of scope here.
This module keeps only the core value proposition -- deduplicating recurring
byte sequences into a stable, cheap-to-copy handle -- via a from-scratch
`HashMap`-backed pool: `TermInterner::intern(&[u8]) -> TermId` returns the
same ID for byte-identical input across calls and a fresh one for new
input, `TermInterner::get(TermId) -> Option<&[u8]>` looks the original bytes
back up, and `TermId` is a plain `Copy` `u32` wrapper. Lives in `lucene-util`
(zero workspace dependencies, sits under every other crate per the
`architecture` skill's downward dependency graph) so any future consumer
(indexing chain, query term dictionaries) can depend on it without a cycle.
No `unsafe` -- the workspace only permits it in `lucene-util`/`lucene-store`/
`lucene-ffi`, and nothing here needed it. Unit tests cover: interning
identical bytes twice returns the same ID; distinct byte sequences get
distinct IDs; ID → bytes round trip; the empty byte string is a valid,
distinct term; looking up an ID this interner never produced returns `None`
rather than panicking; and a stress case interning 20,000 calls over a
50-word vocabulary asserting `TermInterner::len() == 50` (dedup actually
collapsing storage, not just handing back arbitrary IDs), plus a separate
5,000-all-distinct-terms case confirming no false collisions. **Explicitly
deferred**: wiring this into `lucene-index`'s indexing chain or
`lucene-search`'s query term handling (a real future task, once there's a
concrete allocation hot path to point it at), `ByteBlockPool`-style
bulk/arena allocation, sort/compaction, and any on-disk format tie-in --
purely an in-memory primitive today.

**Progress (basic query cache):** `QueryCache`, a new standalone primitive in
`lucene-search/src/query_cache.rs` -- analogous in spirit to real Lucene's
`LRUQueryCache` (`org.apache.lucene.search.LRUQueryCache`), **not** a
byte-for-byte port of it. Real `LRUQueryCache` tracks per-segment
`IndexReader.CacheKey` identity via weak references, bounds itself by both
entry count *and* estimated RAM usage, and decides per-query whether caching
is even worthwhile (`shouldCache`). None of that is implemented; this module
keeps only the core value proposition -- given a `(segment, query)` pair,
hand back a previously computed `FixedBitSet` of matching doc IDs instead of
re-running the query's scorer/matcher. `QueryCache<S, Q>` is generic over any
segment identifier `S: Eq + Hash + Clone` and any query representation
`Q: Eq + Hash + Clone` (this port has no `IndexReader.CacheKey`-style segment
identity object yet, so a caller-supplied key -- a segment name, a generation
number -- stands in); `query::TermQuery` picked up an additive `Hash` derive
(alongside its existing `PartialEq + Eq`) so it can be used directly as `Q`,
rather than inventing a parallel query representation just for caching. API:
`QueryCache::new(max_entries)`, `get_or_compute(segment, query, || ->
FixedBitSet)` (computes and inserts on a miss, returns the cached bitset
unchanged on a hit), `invalidate_segment(&segment) -> usize` (removes that
segment's entries only), `clear()`, `len()`/`is_empty()`. Eviction is bounded
by entry count only, least-recently-used-first, tracked via a monotonic
access counter stamped on every hit/insert (no external LRU-list crate in
the workspace, and a linear scan over a small bounded cache to find the
minimum is the right amount of machinery for this scope). No `unsafe`
(`lucene-search` is `#![forbid(unsafe_code)]`). **Not wired into any live
search path** -- `IndexSearcher`-equivalent query execution
(`directory_reader.rs`/`multi_segment.rs`) still always re-evaluates a
query's scorer/matcher on every call, exactly as before this module existed;
wiring this cache in (with real segment-lifecycle-triggered invalidation and
a cache-worthiness heuristic) is future work. Unit tests cover: a cache miss
calls the compute closure and stores the result; a cache hit reuses the
stored bitset without calling compute again (verified via a call counter);
distinct queries against the same segment get distinct entries; the same
query against different segments gets distinct entries; inserting past
`max_entries` evicts the correct least-recently-used entry (verified by
re-touching one entry to change eviction order, then confirming the
untouched one is evicted and the touched one survives); `invalidate_segment`
removes only that segment's entries, leaving other segments' entries and
cache hits intact; a `max_entries == 0` cache never actually retains
anything; `clear()` empties every segment's entries; and `TermQuery` used as
a concrete `Q` end-to-end. **Explicitly deferred** (see `docs/parity.md`):
RAM-based cache sizing (bounded by count alone here), automatic per-segment
invalidation hooks tied to real segment open/close/merge lifecycle events
(`invalidate_segment` exists and is correct, but nothing in this port calls
it yet -- no segment lifecycle to hook into), cache-worthiness heuristics
like real `LRUQueryCache.shouldCache`, and wiring into `IndexSearcher`-
equivalent live query execution.

**Progress (concurrent segment search):** `multi_segment.rs` gained a `rayon`-
based parallel sibling of its existing sequential fan-out/merge core --
analogous in spirit to real Lucene's `IndexSearcher` constructed with an
`ExecutorService` (each `LeafReaderContext` searched on the executor, partial
`TopDocs` merged once every leaf finishes), **not** a port of that
`Executor`/`LeafSlice` machinery itself. This port has no thread-pool
abstraction of its own, so `merge_multi_segment_scored_concurrent` uses
`rayon::prelude::*`'s `par_iter` (rayon is already a workspace dependency used
elsewhere in this crate) over segments instead of inventing one: each
segment's own `TopDocsCollector`, per-segment search call, and doc-base
translation happen independently inside the parallel closure (no shared
mutable state, so no locking), and the final merge across segments' results
runs sequentially through the same `TopDocsCollector` type the existing
sequential path already uses -- the exact same merge logic, not a
reimplementation of it, since `rayon`'s `.collect()` preserves input order
regardless of which worker thread computed which element. Two thin
concurrent wrappers, `search_term_query_multi_segment_concurrent`/
`search_boolean_query_multi_segment_concurrent`, mirror the existing
`search_term_query_multi_segment`/`search_boolean_query_multi_segment`
exactly, one call to the new core instead of the old one. The existing
sequential functions are unchanged -- this is a pure addition, not a
replacement. **Correctness property tested directly**: sequential and
concurrent results are asserted byte-for-byte identical (same doc IDs, same
order, same scores) for the same input, across an empty index, a single
segment, 16 synthetic segments (enough for rayon's pool to plausibly
parallelize), a top-N-truncation case, a same-score-tie-across-segments case,
and both real-fixture end-to-end query types (term, boolean). **Explicitly
deferred**: any thread-pool configuration/sizing knobs (real Lucene's
`Executor` lets a caller supply its own pool size; this uses rayon's global
pool as-is, no equivalent knob exposed), work-stealing tuning beyond what
rayon's global pool already provides, and any I/O-bound async concern -- this
is CPU-parallel scorer evaluation only, no FFI entry point for it yet either.

**Progress (IndexWriter facade):** `IndexWriter`, a new struct in
`lucene-index/src/index_writer.rs` -- analogous in spirit to real Lucene's
`org.apache.lucene.index.IndexWriter` as the single entry point for
add/update/delete/commit, **composed entirely out of already-built
primitives, not a reimplementation of any of them**. Before this task, a
caller wanting to add documents, commit them as a segment, delete/update by
term, or fold in a merge result had to hand-thread a `SegmentInfos`, a
segment-name counter, and a buffered-document list across separate calls
into `segment_writer::flush_stored_only_segment`, `segment_infos::write`/
`read_latest`, `update_document::update_document`,
`term_delete::resolve_and_apply_term_delete` + `deletes::apply_deletes`, and
`merge::merge_stored_only_segments` -- exactly the manual orchestration this
task's tests (and `update_document.rs`'s/`directory_reader.rs`'s own tests)
previously did by hand. `IndexWriter::open(dir, fields, codec_name,
lucene_version)` resumes an existing commit or starts fresh; `add_document`
buffers in memory; `commit()` flushes the buffer through the existing
`flush_stored_only_segment` and writes the updated segment list through the
existing `segment_infos::write`; `update_document`/`delete_documents`
delegate straight to the existing atomic delete/update primitives,
unchanged; `apply_merge` folds an already-executed
`merge::merge_stored_only_segments` result back into the writer's committed
state. **Update (automatic merge triggering task):** `commit()` now *does*
call `merge_policy::find_merges`/`merge::merge_stored_only_segments`
automatically -- see the follow-up entry directly below; the paragraph above
describes `apply_merge`'s own scope, which is unchanged (still purely "fold
an already-executed merge result into committed state," now just also
called internally by `commit()`'s own auto-merge step, not only by a manual
caller). Segment/commit ids are generated by hashing a monotonic counter with
the current time (`DefaultHasher`, not a CSPRNG) since this workspace has no
`rand` dependency and the only property needed here is per-session
uniqueness, not statistical randomness. **Explicitly deferred, and why:** no
RAM-based flush triggering (only an explicit `commit()` flushes, matching
`segment_writer.rs`'s own long-documented "no RAM accounting" stance). No
two-phase commit/rollback. No multi-threaded `DocumentsWriterPerThread`-style
pooling. `update_document`/`delete_documents` still only resolve against
segments the caller supplies an opened `SegmentDeleteSource` for -- inherited
directly from `update_document.rs`/`term_delete.rs`, which still have no
reader pool that opens every existing segment's postings automatically.
**No new Java fixture** (composition of already-verified primitives, same
precedent as `update_document.rs`/`merge.rs`'s own composition tasks): unit
tests cover opening a fresh directory, add-then-commit producing one
readable segment (verified both through the returned struct and by
re-reading the on-disk `segments_N`), a no-pending-documents commit still
producing a valid next generation, multiple commits producing multiple
independent non-colliding segment names, resuming a writer against an
already-committed directory without a segment-name collision,
`update_document`/`delete_documents` against the same real-Lucene postings
fixture `term_delete.rs`/`update_document.rs` already use (including a
failing delete leaving both the writer's in-memory state and the on-disk
`segments_N` untouched), and `apply_merge` folding a real
`merge_stored_only_segments` result over two committed segments back into
the writer. `SegmentInfos`/`SegmentCommitInfo` picked up a `PartialEq` derive
(needed to assert a failed call left state byte-for-byte unchanged) -- a
mechanical addition, no behavior change.

**Follow-up task: automatic merge triggering.** `IndexWriter::set_merge_policy(Some(config))`
opts a writer into automatic merging: every subsequent `commit()` call, right
after writing its own `segments_N`, loops calling `merge_policy::find_merges`
against the writer's now-committed segments (stats built from each segment's
`.si` via `segment_info::parse` plus `merge_policy::segment_byte_size`, and
`del_count` already in memory from `SegmentCommitInfo`) and, for every
proposed group, executes it via `merge::merge_stored_only_segments` (opening
each source segment's stored fields/live-docs straight off `dir`) and folds
the result in via the existing `apply_merge` -- repeating until `find_merges`
proposes nothing further. Terminates because each executed merge strictly
reduces the segment count by at least one. With no merge policy set (the
default), `commit()` is byte-for-byte the same as before this task, so none
of the 10 pre-existing `index_writer.rs` tests needed changes. Both
`find_merges` (`merge_policy.rs`) and `merge_stored_only_segments`
(`merge.rs`) are called as-is, not reimplemented. New tests: no-merge-policy
commits never auto-merge; a commit sequence that stays at/under
`segments_per_tier` stays unmerged; a sequence that crosses it merges down to
fewer segments with every original document still readable; and 20
consecutive single-document commits under a tight policy converge without
panicking or looping, with segment count never exceeding commit count. Still
deferred: no merge-policy configurability from `IndexWriter::open` itself
(only via the separate `set_merge_policy` setter), no concurrent/background
merging, and no multi-tier scheduling beyond whatever one
`merge_policy::find_merges` call already does -- `update_document`/
`delete_documents` do not trigger this check, only `commit()` does.

**Progress (task #76): compound-file read wiring in `DirectoryReader`.**
`lucene-search/src/directory_reader.rs::SegmentReader::open` now opens a
segment's `.cfs`/`.cfe` pair when `SegmentInfo.is_compound_file` is set,
instead of returning `Error::CompoundFileUnsupported`. This is read-path
*wiring*, not new codec work: the new `CompoundArchive` type calls
`lucene_codecs::compound_format::{parse_entries, check_data_header_footer,
open_input}` exactly as already written and already differentially verified
(see that module's `docs/parity.md` row) -- nothing about the `.cfs`/`.cfe`
byte format is touched. Every sub-file lookup this reader needs (`.fnm`,
`.tim`, `.tip`, `.tmd`, `.doc`, `.pos`, `.pay`) now goes through one shared
helper, `open_segment_file` (plus `find_segment_file_name` for the
name-only lookup `.tim`'s embedded codec-suffix parsing needs), so there's a
single branch point -- "compound: read out of the archive, by extension
suffix; loose: `dir.open` the `SegmentInfo.files` entry ending in that
extension, exactly as before" -- rather than duplicating that decision at
each of the seven call sites. Verified against a real Java-written compound
segment, not just this port's own writer: `fixtures/data/compound_index/`
(`GenCompoundFormat.java`, a real `IndexWriter` with `useCompoundFile=true`,
5 docs, real `.tim`/`.tip`/`.tmd`/`.doc` postings packed inside the `.cfs`
alongside `.fnm`/stored-fields/doc-values files) now opens through
`DirectoryReader::open` and answers a real term query correctly -- replacing
the old test that merely checked for a rejection error. A second test
flushes the same documents through this port's own
`segment_writer::flush_stored_only_segment` twice, loose and compound, and
confirms the compound segment's field infos/doc count match the loose one
with no loose `.fnm` present to silently fall back to. Every pre-existing
`directory_reader.rs` test (real fixtures, stored-fields-only, missing
`.fnm`, partial blocktree files, `open_if_changed` reuse/reopen) still
passes unchanged. **Honestly still not compound-aware**: doc-values, norms,
and term-vector reading elsewhere in `lucene-search`
(`doc_value_query.rs`, `field_norms.rs`, `term_vectors_query.rs`,
`soft_deletes.rs`) each open their files directly off the outer `Directory`
by name, independent of `SegmentReader`; none of them check
`SegmentInfo.is_compound_file` or use this task's new helpers, so a segment
with doc-values/norms/term-vectors packed into its `.cfs` would still fail
to open through those call sites. This port's own write side doesn't
produce such a segment yet, so nothing exercises that gap today, but it's a
real, unclosed one, not silently papered over.

**Progress (task #77): doc-values write-side generalization.** Before this
task, `lucene-codecs/src/doc_values.rs::write_single_dense_numeric_field`
was the only doc-values write function -- one kind (NUMERIC), dense-only,
plain delta-compressed. This task adds two siblings built on the same
dense/no-terms-dict scope: `write_single_dense_binary_field` (BINARY --
fixed-length, `ordinal * length` indexing, and variable-length via a
`direct_monotonic::write`-backed end-offset array, both dense) and
`write_single_dense_sorted_numeric_field` (SORTED_NUMERIC -- every doc has
>= 1 value, flattened into one shared value array plus a per-doc address
range, with the same one-value-per-doc collapse real Lucene's own reader
does: when every doc has exactly one value, `read_sorted_numeric_entry`
infers "no address array" from `num_docs_with_field == numeric.num_values`
rather than a stored flag, so the writer must detect and match that case
rather than always writing addresses). The three functions now share one
extracted helper, `write_dense_numeric_entry_body`, for the NUMERIC-entry
layout SORTED_NUMERIC's flat value array reuses verbatim. **SORTED and
SORTED_SET still have no write side, and it's not an oversight**: both need
a terms-dictionary write side (`terms_dict.rs`'s 64-term LZ4-compressed,
prefix-compressed blocks plus an FST reverse index) that doesn't exist in
this port at all -- `terms_dict.rs` is decode-only (see its parity row) --
so writing either would mean building an entire new codec, not extending
today's dense/no-compression-tricks scope this task's generalization
otherwise stayed inside. That remains a real, separately-sized future slice.
**Not wired into any writer pipeline**: same as `write_single_dense_numeric_field`
before it, nothing in `flush_stored_only_segment`/`IndexWriter` calls the new
functions; only this module's own tests do. Verified by round-tripping
through this port's own unmodified read side (`parse_meta`/`binary_value`/
`sorted_numeric_values`/`check_data_header_footer`), the read function's own
correctness oracle per this task's brief: BINARY fixed-length and
variable-length (including an empty-string value), the non-dense-input
rejection path; SORTED_NUMERIC with varying per-doc value counts (1-3),
the all-single-valued collapse case (confirms no address array is written,
matching what the reader infers), the all-same-value case (confirms the
constant-value/`bitsPerValue == 0` encoding still applies to the flattened
array), and the empty-per-doc-value rejection path. See `docs/parity.md`'s
updated row for the full accounting.

**Progress (task #78): postings write side, single-field first cut.** New
module `crates/lucene-codecs/src/postings_writer.rs::write_single_field`
writes `.doc`/`.tim`/`.tip`/`.tmd` for exactly one field: one `.tim` block
(single `SIGN_NO_CHILDREN` `.tip` root, no floor/multi-child trie), every
term's `docFreq < BLOCK_SIZE` (256, the group-varint "tail block" shape
only — no full `ForUtil`/`PForUtil` blocks), `IndexOptions::Docs`/
`DocsAndFreqs` only (no positions/offsets/payloads, no `.pos`/`.pay`), and
`docFreq == 1` pulsed into the term dictionary exactly like the real writer
(no `.doc` bytes for a singleton). No read-side decode logic was
reimplemented -- the writer only emits bytes, promoting a handful of
previously-private format constants (`postings::DOC_CODEC`/
`VERSION_CURRENT`, blocktree's codec-name/version/trie-sign constants) to
`pub(crate)` so the writer references the exact same wire constants the
reader checks, and promoting the pre-existing test-only `write_group_vints`
encoder (`postings.rs`) to a real `pub(crate)` production function. **Not
wired into `flush_stored_only_segment`/`IndexWriter` at all** -- this closes
the "can this port's own code produce `.doc`/`.tim`/`.tip`/`.tmd` bytes the
existing reader accepts" gap for one narrow shape, not the "is a document
added via `IndexWriter` now searchable" gap (that still needs multi-field
support, wiring into the segment flush path, and a `.si`/`.fnm` record that
actually points at postings files, none of which this task touched).
**Required end-to-end proof**: `crates/lucene-search/tests/
postings_writer_round_trip.rs::term_query_finds_correct_docs_over_freshly_written_postings`
writes a field with singleton and multi-doc terms, opens it via the
existing unmodified `blocktree::open`/`postings::DocInput::open`, and runs
the existing unmodified `lucene_search::search_term_query` for every term
(plus a missing term and a live-docs-filtered case in a sibling test),
asserting the correct doc IDs come back through the whole stack -- not just
a byte-level decode check. `postings_writer.rs`'s own unit tests cover the
byte layer beneath that (mixed singleton/multi-doc, `IndexOptions::Docs`
no-freqs aliasing, all-singleton field needing no `.doc` file, 20
terms x 5 docs each for running-`doc_start_fp`-delta correctness across
more than a couple of terms) plus one negative test per structural
invariant. **Deferred, explicitly**: multiple fields per call/segment,
multi-block `.tim` fields (block-splitting/floor sub-blocks/multi-level
`.tip` tries), `docFreq >= BLOCK_SIZE` (full blocks/skip data/impacts),
positions/offsets/payloads, and any wiring into the segment writer/
`IndexWriter`. See `docs/parity.md`'s new row for the full accounting.

**Progress (task #78 follow-up): postings write side wired into `IndexWriter::commit()`.**
`IndexWriter::set_postings_field(Some(field_name))` (new,
`crates/lucene-index/src/index_writer.rs`) opts a writer into building and
writing real postings for exactly one field, using
`crate::indexing_chain::invert_documents` (already-built tokenize-and-invert
builder, unchanged) to turn that field's `FieldValue::String` values into
`postings_writer::TermPostings`, then calling
`postings_writer::write_single_field` unmodified to encode the bytes --
**no postings-encoding logic was reimplemented**. `commit()` now, when a
postings field is set and there are pending docs: builds that field's
postings entirely in memory first, then flushes stored fields via the
unchanged `flush_stored_only_segment`, then writes `<segment>.doc`/`.tim`/
`.tip`/`.tmd` and patches `<segment>.si`'s file list to include them. This
closes the "is a document added via `IndexWriter` now searchable" gap task
#78 explicitly left open, but for the **exact same narrow scope**
`postings_writer.rs` already has: one field indexed with postings at a
time (no per-field file-suffix machinery to fan out to more than one),
one `.tim` block per commit (`docFreq < BLOCK_SIZE` = 256), term-frequency
only (no positions/offsets/payloads, so no phrase queries over
`IndexWriter`-produced postings yet). A term reaching `docFreq >= 256` in
one commit's pending-document batch makes the **whole `commit()` call
fail** with `Error::PostingsWriter(postings_writer::Error::DocFreqTooLarge)`
-- checked before anything is written to `dir`, so `dir`/`pending_docs`/
`segment_infos` are all left completely unchanged, never a partially-written
segment (same atomicity `IndexWriter::update_document` already guarantees).
Backward compatible: a writer that never calls `set_postings_field`
(`None`, the default) produces byte-identical stored-only segments to
before this feature existed -- every pre-existing `IndexWriter` test still
passes unchanged. **Required end-to-end proof**:
`crates/lucene-search/tests/index_writer_postings_fixtures.rs::
documents_added_via_index_writer_are_searchable_by_term_query` adds 3
documents via `IndexWriter::add_document`/`commit()` (not a hand-built
fixture), opens the resulting segment through the existing unmodified
`blocktree::open`/`postings::DocInput`, and runs the existing unmodified
`lucene_search::search_term_query` for 5 distinct terms (shared and
singleton) plus a missing term, asserting the exact doc IDs a real
`IndexSearcher` would return. A sibling test,
`commit_rejects_a_term_at_the_256_doc_freq_boundary`, drives 256 docs
sharing one term through `IndexWriter` itself and asserts `commit()`
returns `Err` rather than silently writing wrong/truncated postings.
`crates/lucene-index/src/index_writer.rs`'s own unit tests cover the same
boundary from inside the crate (`commit_rejects_and_leaves_state_unchanged_
when_a_term_reaches_doc_freq_256`, `commit_succeeds_below_the_doc_freq_
boundary`), plus `set_postings_field` misuse (unknown field name, a field
with `IndexOptions::None`), a doc missing the field/holding a non-`String`
value, text that tokenizes to zero terms, and an empty-pending-docs commit
with a postings field set -- all "skip postings, don't error" cases.
**Interaction with automatic merge triggering (task #71), fixed during
review**: `execute_merge`/`merge_stored_only_segments` only know how to
merge stored fields -- they have no `.doc`/`.tim`/`.tip`/`.tmd` awareness at
all. Feeding a postings-carrying segment into `find_merges` would let an
automatic merge silently drop that segment's postings with no error
(the merged segment's `.si` would list only stored-fields files, and the
source's real postings files would become orphaned on disk). `segment_stats()`
now excludes any segment whose `.si` lists a `.doc` file from
`find_merges`' candidate pool entirely, so such a segment is permanently
un-mergeable rather than mergeable-with-silent-data-loss, until
postings-aware merging exists. Covered by
`segments_with_postings_are_never_automatically_merged_away`: enables both
`set_postings_field` and `set_merge_policy` at once, crosses the tight
policy's merge threshold with three postings-carrying one-doc commits, and
asserts the segment count stays at 3 (no auto-merge fires) with every
segment's `.tim` file still present and correctly listed in its own `.si`.

**Still explicitly deferred**: multiple fields indexed with postings in one
commit, multi-block `.tim` fields, positions/offsets/payloads (so no
`PhraseQuery` support over `IndexWriter`-produced segments), postings-aware
segment merging (a segment with postings can never be auto-merged today,
see above), and any RAM-threshold/auto-flush triggering (unchanged from
`IndexWriter`'s existing scope). See `docs/parity.md`'s updated row for the
full accounting.

**Progress (task #78 follow-up #2): term-vector write side wired into
`IndexWriter::commit()`.** `IndexWriter::set_term_vector_field(Some(field_name))`
(new, `crates/lucene-index/src/index_writer.rs`) opts a writer into building
and writing real term vectors for exactly one field, using
`crate::indexing_chain::invert_documents` (same tokenize-and-invert builder
`set_postings_field` already reuses, unchanged) and regrouping its
term-keyed inverted index by doc ID (term vectors need
per-document `term -> (freq, positions)`, the transpose of what a postings
writer wants), then calling `term_vectors::write_best_speed` unmodified to
encode the bytes -- **no term-vector-encoding logic was reimplemented**.
`commit()` now, when a term-vector field is set and there are pending docs:
builds that field's term vectors entirely in memory first (alongside, and
independently of, any `set_postings_field` output), then flushes stored
fields via the unchanged `flush_stored_only_segment`, then writes
`<segment>.tvd`/`.tvx`/`.tvm` and patches `<segment>.si`'s file list to
include them (reusing the same read-modify-write-then-resync `.si`-patching
helper shape `write_postings_files` already established, so a segment with
both postings and term vectors ends up with one `.si` correctly listing all
seven files regardless of which write happened first). Scope matches
`term_vectors.rs::write_best_speed`'s own documented scope exactly: one
field opted into term vectors at a time, single chunk, positions only (no
offsets/payloads yet). `set_term_vector_field` validates the field exists
and has `store_term_vectors == true` on its `FieldInfo` (an `Err` otherwise,
mirroring `set_postings_field`'s own fail-fast validation and
`field_infos::FieldInfo::check_consistency`'s own "non-indexed field cannot
store term vectors" invariant). Backward compatible: a writer that never
calls `set_term_vector_field` (`None`, the default) produces byte-identical
segments to before this feature existed. **Required end-to-end proof**:
`crates/lucene-search/tests/index_writer_term_vectors_fixtures.rs::
documents_added_via_index_writer_have_readable_term_vectors` adds 3
documents via `IndexWriter::add_document`/`commit()` (not a hand-built
fixture), opens the resulting segment through the existing unmodified
`term_vectors::open`/`TermVectorsReader::document`, and reads them back via
the existing unmodified `lucene_search::term_vector_for_doc`, asserting the
exact per-document term/frequency/position data a real
`IndexReader.getTermVector` would return (plus that a field never opted in
has none). `crates/lucene-index/src/index_writer.rs`'s own unit tests cover
`set_term_vector_field` misuse (unknown field name, a field without
`store_term_vectors`), a doc missing the field/holding a non-`String` value,
text that tokenizes to zero terms, a doc-count mismatch check (a doc with no
term-vector text for this field still gets a `TermVectorsDocument` entry so
doc IDs stay aligned with the segment, even though
`TermVectorsReader::document` itself then decodes that doc as `None`), and
an empty-pending-docs commit with a term-vector field set.
**Interaction with automatic merge triggering (task #71), applied
proactively from the postings-feature review finding above**: the exact
same class of bug applies here -- `execute_merge`/`merge_stored_only_segments`
have no `.tvd`/`.tvx`/`.tvm` awareness either, so `segment_stats()` now also
excludes any segment whose `.si` lists a `.tvd` file from `find_merges`'
candidate pool, keeping it permanently un-mergeable rather than
mergeable-with-silent-data-loss. Covered by
`segments_with_term_vectors_are_never_automatically_merged_away`, mirroring
the postings version exactly. **Postings + term vectors together, tested**:
`a_field_with_both_postings_and_term_vectors_configured_together_produces_both_correctly`
(in `lucene-search`'s fixture file) and
`a_field_with_both_postings_and_term_vectors_configured_at_once_writes_both_correctly`
(in `lucene-index`'s own unit tests) both enable
`set_postings_field(Some("body"))` and `set_term_vector_field(Some("body"))`
on the same field in the same commit and assert both write sides land
correctly in one `.si` and are both independently readable -- no
interaction bug found; the two build/write passes are fully independent
(separate in-memory builds before either touches `dir`, separate file sets,
and the `.si`-patching helper for each reads back whatever the other has
already written rather than overwriting it).

**Still explicitly deferred**: offsets/payloads in term vectors, multiple
fields with term vectors in one commit, multi-chunk `.tvd`, term-vector-aware
segment merging (a segment with term vectors can never be auto-merged today,
same as postings), and any RAM-threshold/auto-flush triggering (unchanged
from `IndexWriter`'s existing scope). See `docs/parity.md`'s updated row for
the full accounting.

**Follow-up task: NUMERIC doc values wired into `IndexWriter::commit()`.**
`IndexWriter::set_doc_values_field(Some(field_name))` (new,
`crates/lucene-index/src/index_writer.rs`) opts a writer into calling
`doc_values::write_single_dense_numeric_field` unmodified for exactly one
field of every segment `commit()` flushes, writing `.dvd`/`.dvm`/`.dvs` and
patching that segment's `.si` to list them -- same "one field per call, build
in memory before touching `dir`" shape `set_postings_field`/
`set_term_vector_field` already established. Only NUMERIC is wired: BINARY
and SORTED_NUMERIC write sides exist in `doc_values.rs` (task #77) but
`set_doc_values_field` rejects any field whose `FieldInfo.doc_values_type`
isn't `DocValuesType::Numeric` with `Error::UnsupportedDocValuesType`.

**Dense-only, enforced at `commit()` time, atomically**: unlike
`set_postings_field`/`set_term_vector_field`'s "best effort, skip that doc"
handling of a missing/wrong-typed value, `write_single_dense_numeric_field`
has no missing-value encoding at all, so *every* pending doc must carry a
`FieldValue::Int`/`FieldValue::Long` value for the opted-in field or the
whole `commit()` call fails -- `Error::MissingDenseDocValue` for a doc with
no value at all, `Error::NonNumericDocValue` for a doc whose value isn't
`Int`/`Long` -- leaving `dir`/`pending_docs`/`segment_infos` completely
unchanged, same atomicity guarantee as the `docFreq >= 256` postings
rejection. Backward compatible: a writer that never calls
`set_doc_values_field` produces byte-identical output to before this feature
existed (covered by `commit_with_no_doc_values_field_configured_stays_stored_only`).

**Interaction with automatic merge triggering (task #71), applied
proactively from the postings/term-vector features' own review findings**:
`execute_merge`/`merge_stored_only_segments` have no `.dvd`/`.dvm`/`.dvs`
awareness either, so `segment_stats()` now also excludes any segment whose
`.si` lists a `.dvd` file from `find_merges`' candidate pool, keeping it
permanently un-mergeable rather than mergeable-with-silent-data-loss. Covered
by `segments_with_doc_values_are_never_automatically_merged_away`, mirroring
the postings/term-vector versions exactly.

**Postings + term vectors + doc values together, tested**:
`postings_term_vectors_and_doc_values_configured_together_all_write_correctly`
enables `set_postings_field`, `set_term_vector_field`, and
`set_doc_values_field` (different fields) in the same commit and asserts all
ten files land correctly in one `.si` and are all independently readable --
no ordering bug found between the three independent `.si`-patching passes.

Required end-to-end proof:
`commit_with_doc_values_field_writes_readable_numeric_values_for_multiple_docs`
(`crates/lucene-index/src/index_writer.rs`) adds documents via
`IndexWriter::add_document`/`commit()` and reads the written `.dvm`/`.dvd`
back via the existing unmodified `lucene_codecs::doc_values::{parse_meta,
numeric_value}`, asserting the exact per-document values a real
`NumericDocValues.longValue` would return.

**Still explicitly deferred**: BINARY/SORTED_NUMERIC doc values wired into
`IndexWriter`, multiple doc-values fields in one commit, sparse (missing-value)
doc values, and doc-values-aware segment merging (a segment with doc values
can never be auto-merged today, same as postings/term vectors). See
`docs/parity.md`'s updated rows for the full accounting.

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
   - **Postings writer: single-field, single-block first slice landed**
     (`lucene-codecs/src/postings_writer.rs::write_single_field`) — one field,
     one `.tim` block/trie node, `docFreq < BLOCK_SIZE` (no full FOR/PFOR
     blocks, no skip/impacts data), term-frequency-only (no positions). Proven
     correct by round-tripping through the existing unmodified
     `blocktree::open`/`postings::DocInput` and, end-to-end, through
     `lucene_search::search_term_query`
     (`crates/lucene-search/tests/postings_writer_round_trip.rs`) — see
     `docs/parity.md`'s row for the precise scope/deferred list. Multi-block
     terms, multi-block/multi-field term dictionaries, and positions/offsets/
     payloads remain unimplemented.

**Progress (task #79): `BooleanQuery`/`Clause` rewrite pass.** New
`crates/lucene-search/src/query.rs::{BooleanQuery::rewrite, Clause::rewrite}`
-- a pure, standalone simplification pass, **opt-in only**, not wired into
`search_boolean_query`/`search_boolean_query_scored` (neither function
changed). Rules implemented, precisely: (1) single-clause unwrap -- a
`BooleanQuery` with exactly one clause total and no `must_not` collapses to
that clause directly, but only in the two cases that provably don't change
matching: `must.len() == 1` with `should` empty and `minimum_should_match ==
0`, or `should.len() == 1` with `must` empty and `minimum_should_match <=
1` (both `minimum_should_match > 0` against an empty `should`, and `> 1`
against a single `should` clause, are deliberately excluded -- either would
turn "matches nothing" into a positive match, see `BooleanQuery::rewrite`'s
doc comment); (2) zero-clause/`must_not`-only "matches nothing" -- confirmed
as a no-op in code, since `matched_boolean_docs` already treats that case as
matching nothing with no `MatchNoDocsQuery`-equivalent `Clause` variant
needed; (3) recursion -- every clause is rewritten bottom-up before a parent
checks its own collapse condition, reaching into nested `Clause::Boolean`/
`DisjunctionMax`/`ConstantScore`/`Boost`. **Deliberately NOT implemented:
duplicate-clause deduplication** -- task #60 already confirmed, against this
port's real executor, that a duplicate `should` clause double-counts toward
`minimum_should_match` and double-scores (real Lucene's actual behavior, not
a bug), and the same sum-based scoring applies to duplicate `must` clauses;
deduplicating either would silently change scores or matched sets, the
opposite of what this pass promises, so it's skipped rather than guessed at.
Scoring-equivalence is proven end-to-end, not just structurally: three new
tests in `crates/lucene-search/tests/boolean_query_fixtures.rs`
(`rewrite_produces_identical_scored_results_for_single_must_clause`,
`_for_single_should_clause`, `_for_nested_single_clause_boolean`) run the
same query pre- and post-`rewrite()` through `search_boolean_query_scored`
against the real `blocktree_index` fixture and assert identical
`TopDocsCollector::top_docs()` output (doc IDs and scores both), plus 15
structural unit tests in `query.rs` covering each rule and its boundary
(`minimum_should_match` too high/one-past, `must_not` present, more than one
clause, leaf clauses passed through unchanged). `cargo test -p lucene-search`
passes in full (426 lib tests, including the 15 new `query::tests::rewrite_*`
cases, plus every integration test, including `boolean_query_fixtures.rs`'s
13, up from 10 pre-existing).

**Progress (task #80): `TopFieldCollector` -- search-time sort-by-field.**
New `crates/lucene-search/src/collector.rs::{SortDirection, FieldValueDoc,
TopFieldCollector}` -- the first general SEARCH-time "sort matched query
results by a doc-value field" collector, as opposed to the earlier
segment-level index-sort infrastructure (`sort_by_numeric_doc_value`, task
#21/#31) or `IndexSort`-by-field write-side ordering. `TopFieldCollector` is
structurally identical to the existing `TopDocsCollector` (same bounded,
always-sorted `Vec` design, same tradeoff rationale), but ranks by an
already-decoded `i64` doc-value instead of an `f32` score, and supports both
`SortDirection::Ascending` and `Descending` (real `SortField.setReverse`).
Ties break by ascending doc ID, the same convention `TopDocsCollector`
already uses for a score tie. It intentionally does **not** implement
`Collector`/`ScoringCollector` -- reading a doc's sort value is fallible
(propagates `doc_values::Error`), and neither trait's `collect` signature can
carry a `Result`; instead callers decode each candidate's value themselves
and call `TopFieldCollector::offer(doc_id, value)` with the plain, already-
decoded value.

Two new functions in `doc_value_query.rs` provide the actual end-to-end
usable entry points: `sort_top_n_by_numeric_doc_value` (the general
composition point -- takes any already-collected `&[i32]` candidate list,
same contract `sort_by_numeric_doc_value` already has, plus a
`SortDirection` and a `top_n` bound, decoding via the existing
`doc_values::numeric_value` primitive, no new decode logic) and
`search_numeric_range_sorted_by_field` (a concrete wiring of that composition
point onto an existing query execution path, `search_numeric_range`: run the
range query into a `VecCollector`, then sort the matches by a second numeric
field, ascending or descending, top-N truncated). **Scope, stated precisely**:
numeric doc-value fields only (`SortField.Type.LONG`/`INT` -- no `DOUBLE`
bit-reinterpret step, no String/`SortedDocValues`-based sort), single sort
key (no secondary `Sort` composition), ties broken by ascending doc ID.
**Missing-value handling**: governed by the same `MissingValue::Exclude`/
`Default(i64)` enum `sort_by_numeric_doc_value` already established --
`Exclude` drops a candidate with no value from the top-N entirely,
`Default(v)` substitutes `v` and lets it compete normally. **Standalone,
not wired into every existing scored-search caller**: `search_term_query`/
`search_boolean_query`/`search_term_query_scored`/`search_boolean_query_scored`
are all unchanged; a caller wanting field-sorted results explicitly calls one
of the two new functions above, the same "additive, not a breaking change"
posture this crate has kept for every collector addition so far (see
`collector.rs`'s own module doc on `ScoringCollector`). Tests: 8 new unit
tests in `collector.rs` (`top_field_collector_*`, `field_rank_order_*` --
empty/zero-`top_n`, ascending/descending ordering, top-N truncation both
directions, ascending-doc-ID tie-break) plus 9 new unit tests in
`doc_value_query.rs` (`sort_top_n_*`, `search_numeric_range_sorted_by_field_
end_to_end_real_fixture`) reusing the real, already-checked-in
`fixtures/data/doc_values_index/` fixture's `varying`/`gcd`/`sparse` fields --
the end-to-end test queries `gcd in [1000, 1100]` (matching real docs 0, 1,
2, 4 out of the fixture's 5), then sorts those matches by the `varying`
field both ascending and descending, asserting the exact real-value-derived
doc-ID order (`[(0,-100),(4,-3),(1,7),(2,42)]` ascending, reversed
descending) plus a top-2-truncated descending case, all hand-computed
against the fixture's own recorded values, not just "sorted somehow". A
separate hand-built constant-value case confirms the ascending-doc-ID
tie-break under both directions. `cargo test -p lucene-search`: 442 lib
tests pass (up from 426), plus every pre-existing integration test still
green. `cargo clippy --workspace --all-targets -- -D warnings` clean. See
`docs/parity.md`'s updated row for the exact scope statement.

**Progress (Faceted search FFI exposure):** `lucene-ffi` C-ABI wrappers for
tasks #50/#58's `lucene-search/src/facets.rs`, the last "no FFI exposure"
gap that module's own scope notes left open. Two new modules following
tasks #20/#30/#40's exact pattern (opaque `u64` handles, `catch_unwind`
via `guard`, per-call status codes, handle validation before use):

- `crates/lucene-ffi/src/facets.rs`: `ffi_facet_counts_sorted_set` (wraps
  `facets::facet_counts` + `resolve_labels`, then `top_n_facets` when
  `top_n > 0` -- `top_n == 0` returns every facet in ordinal order,
  untruncated, simply by not calling `top_n_facets`) and
  `ffi_range_facet_counts` (wraps `facets::range_facet_counts`). Field-name
  -> field-number/doc-values-entry lookup follows `sort.rs`'s
  `numeric_entry_for` pattern exactly; a SORTED_SET field written as
  `SortedSetKind::Single` is `FfiStatus::InvalidArgument` since
  `facet_counts` itself has no counting path for that shape (not a gap this
  FFI layer introduces -- see `facets.rs`'s own module doc).
- `crates/lucene-ffi/src/results_facets.rs`: `ffi_facet_results_len`/
  `ffi_facet_results_copy` (parallel `i64`/`u64` `(ord, count)` buffers, same
  shape `results_sorted.rs` established) plus a new
  `ffi_facet_result_label` per-index accessor (labels are variable-length
  strings resolved from the index, so they don't fit the fixed-size
  parallel-buffer shape -- reuses this crate's existing
  `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract from
  `ffi_get_last_error_message` rather than inventing a new wire encoding),
  and `ffi_close_facet_results`. New `RegistryTag::FacetResults` /
  `registry::FacetResultsHandle` (own registry, not folded into
  `SortedResultsHandle` -- a facet result carries a resolved label
  `SortedResultsHandle`'s element has no room for).
- `ffi_range_facet_counts` needs **no output handle at all**: every range's
  label is caller-supplied input (not resolved from the index), so counts
  are written straight into a caller-allocated `out_counts: *mut u64` buffer
  in the same order as the input ranges. Range inputs cross the wire as
  five parallel arrays (`range_mins`/`range_min_inclusive`/`range_maxs`/
  `range_max_inclusive`/label bytes sliced by `range_label_lens`) -- the
  concatenated-buffer encoding is only usable on the input side, where the
  caller already knows every length up front; see `facets.rs`'s module doc
  for why the output side (`ffi_facet_result_label`) had to take a different
  shape.

**No facet-counting logic was touched or duplicated** -- every new function
is a thin marshal-in/call-into-`facets.rs`/marshal-out wrapper; `facets.rs`
itself is unchanged. Tests: `facets::tests` (20 cases -- a real fixture
cross-check against calling `lucene_search::facets` directly for the
SortedSet path, `top_n` truncation, unknown/wrong-kind field, unknown
segment handle, null out-handle, null candidates with nonzero length, empty
candidates, a field with no SORTED_SET entry; a real fixture cross-check for
the range path, zero-ranges no-op, unknown field, a field with no NUMERIC
entry, null out-counts, null candidates/mins/label-data with nonzero
lengths, an invalid-UTF-8 label, unknown segment handle) and
`results_facets::tests` (13 cases -- len/copy/label round-trip,
buffer-too-small on both copy and label, a null label buffer, empty-results
no-op, null-pointer variants, out-of-bounds label index, unknown/
double-close handle). `cargo test -p lucene-ffi`: 189 tests pass (up
from 156). `cargo clippy --workspace --all-targets -- -D warnings` clean.
**Deferred, not a gap in this task**: `SortedSetKind::Single` counting (no
underlying `facets.rs` support to wrap); index-wide/multi-segment facet
aggregation (same scope boundary tasks #50/#58 already documented);
hierarchical/taxonomy facets, drill-down/drill-sideways (no `lucene-facet`-
module equivalent exists anywhere in this port).

**Progress (Highlighter FFI exposure):** `lucene-ffi` C-ABI wrapper for task
#56's `lucene-search/src/highlighter.rs`, closing that module's own
"lucene-ffi C-ABI exposure" deferred item. Two new modules, same
handle/registry/error-code pattern as the Faceted search FFI exposure task
above:

- `crates/lucene-ffi/src/highlighter.rs`: one function,
  `ffi_assemble_fragments`, wraps `highlighter::assemble_fragments` directly
  -- no fragment-assembly logic reimplemented. Unlike every other function in
  this crate it needs no segment/directory handle at all: `assemble_fragments`
  only takes a field's full text plus a set of `TermOffsetSpan`s, both
  supplied as plain input buffers. Spans cross the wire as four parallel
  arrays (`span_start_offsets`/`span_end_offsets` plus a concatenated
  `span_term_data` buffer sliced by `span_term_lens`), the same
  concatenated-buffer convention `facets.rs`'s `ranges_from_raw` already
  established for per-range labels. `window_chars`/`pre`/`post`/
  `max_fragments` build a `FragmentConfig` directly; `max_fragments == 0` is
  rejected as `FfiStatus::InvalidArgument` (not silently zero fragments)
  since `highlighter.rs`'s `assemble_fragments` never documents that as a
  meaningful input.
- `crates/lucene-ffi/src/results_fragments.rs`: `ffi_fragment_results_len`,
  `ffi_fragment_result_text` (per-fragment highlighted text), plus
  `ffi_fragment_result_matched_terms_len`/`ffi_fragment_result_matched_term`
  (per-fragment matched-term list) -- all per-index string accessors reusing
  the `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract from
  `ffi_get_last_error_message`, no `_copy` bulk call: unlike a facet result's
  `(ord, count)` half, a `Fragment` has no fixed-size field at all (both
  `text` and `matched_terms` are variable-length), so there is nothing to
  bulk-copy into parallel buffers. New `RegistryTag::FragmentResults` /
  `registry::FragmentResultsHandle` (own registry, not folded into
  `FacetResultsHandle` -- a fragment's two-level variable-length shape has no
  fixed-size element `FacetResultsHandle`'s accessors assume).

**No highlighting logic was touched or duplicated** -- `ffi_assemble_fragments`
is a thin marshal-in/call-into-`highlighter.rs`/marshal-out wrapper;
`highlighter.rs` itself is unchanged. Tests: `highlighter::tests` (12 cases --
a real fixture cross-check against calling `lucene_search::highlighter`
directly, reusing task #39/#3's `fixtures/data/term_vectors_index/`-derived
offsets the same way `highlighter.rs`'s own differential test does; empty
spans; empty full_text with a null pointer and zero length; out-of-range
spans dropped rather than erroring; null out-handle, null full_text with
nonzero length, null pre/post, null span_term_lens/span_term_data with
nonzero counts, invalid-UTF-8 term, zero `max_fragments`) and
`results_fragments::tests` (15 cases -- len/text/matched-terms-len/
matched-term round-trip, out-of-bounds fragment index and out-of-bounds
term index on every accessor, buffer-too-small on both text and
matched-term calls, null buffers, unknown handles, unknown/double-close
handle). `cargo test -p lucene-ffi`: 221 tests pass (up from 189).
`cargo clippy --workspace --all-targets -- -D warnings` clean. New-file
coverage: `highlighter.rs` 99.76% lines, `results_fragments.rs` 97.60% lines
(both above the 95% bar; workspace `lucene-ffi` total 98.34%, up from
98.27%). **Deferred, not a gap in this task**:
`term_vectors_query::matched_term_offsets` has no FFI wrapper of its own yet
-- a JNI-only caller with no direct Rust-side access to `lucene-search` would
need that exposed too before it could compute real `TermOffsetSpan`s itself;
`ffi_assemble_fragments` takes spans as plain caller-supplied input rather
than assuming that gap closed. `BreakIterator` sentence-boundary detection
and term-density passage scoring remain out of scope, matching
`highlighter.rs`'s own already-documented scope boundary -- unchanged by
this FFI-only task.

**Progress (Explain FFI exposure):** `lucene-ffi` C-ABI wrappers for the
Query explain() task's `lucene-search/src/explain.rs`, closing that module's
own "no FFI exposure yet" gap. Two new modules, same
handle/registry/error-code pattern as the Faceted search/Highlighter FFI
exposure tasks above:

- `crates/lucene-ffi/src/explain.rs`: `ffi_explain_term_query`/
  `ffi_explain_phrase_query`/`ffi_explain_boolean_query`, each building
  exactly the same `Clause` `query.rs`'s matching `ffi_search_*_query_scored`
  sibling already builds (same wire formats and norms-map helpers, reused
  directly via newly-`pub(crate)` `query::open_field_norms`) and handing it
  straight to `lucene_search::explain::explain_clause` -- no explain logic
  reimplemented. **Scope deliberately matches `query.rs`'s existing
  construction surface**: only `Clause::Term`/`Clause::Phrase`/flat-
  `Clause::Term`-only `Clause::Boolean` are exposed, since those are the only
  three clause shapes `query.rs` can build from FFI input at all --
  `DisjunctionMax`/`ConstantScore`/`Boost`/`Wildcard`/`Prefix`/`Fuzzy`/
  `Regexp`/`Span`/truly-nested-`Boolean` explanations have no wrapper because
  those clauses can't be *searched* through this ABI either (not a gap this
  task introduced).
- **Recursive-tree flattening scheme**: `Explanation` is a recursive tree
  (`details: Vec<Self>`, e.g. a single term explanation is already three
  levels deep: `weight(...)` → `score(freq=...)` → `idf`/`tfNorm`) -- a
  fundamentally different shape from every prior FFI result (`facets.rs`/
  `highlighter.rs` both produced flat lists). Chosen scheme: depth-first,
  pre-order flattening into `Vec<registry::ExplainNode>` at construction
  time -- each node keeps its own `value`/`matched`/`description` plus a
  `Vec<usize>` of **its children's indices into that same flat `Vec`** (a
  child-index list per node, not a parent-index-per-node scheme, so "give me
  this node's Nth child" is an O(1) index into a small per-node list rather
  than an O(total nodes) scan). Pre-order guarantees the root explanation is
  always flattened first, so **node index `0` is always the root** -- a
  caller walks the whole tree starting at `0` and recursively following
  `ffi_explain_node_child_at`. `crates/lucene-ffi/src/results_explain.rs`
  reads it back: `ffi_explain_results_len`, `ffi_explain_node_value`/
  `ffi_explain_node_matched` (fixed-size per-node fields), a per-node
  `ffi_explain_node_description` string accessor (same
  `buf`/`buf_len`/`out_written`/`BufferTooSmall` contract as
  `ffi_get_last_error_message`), and `ffi_explain_node_child_count`/
  `ffi_explain_node_child_at` (the "length first, then per-index accessor"
  shape `results_fragments.rs` already established, applied to a node's
  *children* instead of a fragment's *matched terms*), then
  `ffi_close_explain_results`. New `RegistryTag::ExplainResults` /
  `registry::ExplainResultsHandle` (own registry -- an explain node's shape
  has no correspondence to any existing handle's element).

**No explain logic was touched or duplicated** -- every new function is a
thin marshal-in/call-into-`explain_clause`/marshal-out wrapper;
`lucene-search/src/explain.rs` itself is unchanged. Tests: `explain::tests`
(18 cases -- a real-fixture differential cross-check against calling
`lucene_search::explain::explain_clause` directly for the term, phrase, and
boolean paths, walking the **entire** flattened tree via the FFI accessors
node-by-node (not just the root) and also independently verifying every
`ffi_explain_node_child_at` link matches a from-scratch Rust-side
re-flattening of the same `Explanation`; a non-matching doc collapsing to a
single no-match node; a single-term phrase delegating to the same tree a
direct term-explain call produces; a missing-`.pos`-input multi-term phrase
surfacing as `FfiStatus::Search`; an empty boolean query as a single
no-match node; unknown segment handle, null out-handle, null field/terms
with nonzero length, invalid-UTF-8 field, for each of the three functions)
and `results_explain::tests` (27 cases -- full tree round-trip covering
value/matched/description/child-count/child-at on both an internal and a
leaf node, out-of-bounds node/child index on every accessor, buffer-too-small
on the description accessor, null-pointer variants, unknown/double-close
handle). `cargo test -p lucene-ffi`: 261 tests pass (up from 221). `cargo
clippy --workspace --all-targets -- -D warnings` clean. New-file coverage:
`explain.rs` 97.05% lines / 96.71% regions, `results_explain.rs` 100% lines
(both above the 95% bar; workspace `lucene-ffi` total 98.41% lines, up from
98.34%). The handful of missed lines in `explain.rs` are `.doc`/`.pos`
reopen-decode-error branches identical in shape (and already accepted as
untested) to `query.rs`'s own equivalent branches -- not reachable without
hand-corrupting already-validated-on-open segment bytes. **Deferred, not a
gap in this task**: every clause kind `query.rs` itself doesn't expose (see
above) has no explain wrapper either -- a follow-up to `query.rs`'s own wire
format, not to `explain.rs` (either crate).

**Progress (task #81): search-side BKD points range query.** New
`crates/lucene-search/src/points_query.rs::search_points_range`, the
read-only, non-deleting sibling of task #36's delete-side
`lucene-index/src/points_delete.rs::resolve_points_range_doc_ids` -- "which
live doc IDs does a `PointRangeQuery`-shaped search actually match, in one
already-opened segment", fed through this crate's existing `Collector`
trait (the same shape `search_term_query`/`doc_value_query`'s `search_*`
functions already use), rather than a standalone `Vec<i32>` function.
**No BKD read/traversal logic reimplemented, not even the filtering
logic**: `search_points_range` calls `resolve_points_range_doc_ids`
directly (the dependency graph already has `lucene-search -> lucene-index`,
confirmed by `crates/lucene-search/Cargo.toml`) and only adapts its
`Result<Vec<i32>>` onto `Collector`/this crate's own `Error` type -- the
per-dimension unsigned-byte-wise range comparison, the `decode_all_points`
call, and the ascending/deduplicated doc-ID ordering are all task #36's
code, used as-is. A new `Error::Points(#[from] lucene_codecs::points::Error)`
variant was added to `lucene-search`'s crate-level `Error` enum (mirroring
the existing `Error::DocValues`) since `resolve_points_range_doc_ids` can
surface a `.kdd` decode failure; its sibling
`lucene_index::points_delete::Error::Deletes` variant is matched but
`unreachable!()` here, since `resolve_points_range_doc_ids` (unlike its
`resolve_and_apply_*` sibling) never calls `deletes::apply_deletes`.
**Deliberately out of scope**: a scored variant (`PointRangeQuery` is a
`ConstantScoreQuery`-shaped match-only query in real Lucene too, so there is
no `ScoredCollector` sibling to add); the sublinear `BKDReader.intersect`
tree-pruning traversal (same honest `O(field's point count)` gap task #36's
row already documents, inherited unchanged here); multi-segment federation
(single already-opened segment's `PointsReader`, same scope as every other
query module in this crate). **This port's already-built multi-dimension
BKD points support is exercised, not just single-dimension**: one new test
(`two_dimension_range_checks_every_dimension_independently`) builds a 2D
`LatLonPoint`-shaped fixture in-memory via the existing `points::write` and
confirms a doc whose first dimension alone would match but whose second
dimension doesn't is correctly excluded -- the same AND-across-dimensions
semantics task #36's own 2D test already proved at the delete layer, now
proven again at this new search-side entry point. Tests (9, all in
`points_query.rs`, same hand-built-fixture-via-`points::write` approach
task #36's tests use rather than a new checked-in Java fixture, for the
same reason that task's row gives: the existing
`fixtures/data/points_index/` fixture is single-dimension only and can't
exercise the 2D AND semantics anyway): exact range match, inclusive
boundaries on both ends, a zero-match range, a range matching every doc, an
unknown field number, `live_docs` filtering an already-deleted doc out, the
2D multi-dimension case above, and a corrupt-`.kdd`-leaf-data case
(scrambling bytes strictly between the codec header and footer so
`points::open` itself still succeeds but `decode_all_points`'s leaf read
fails) confirming the new `Error::Points` surfaces correctly all the way
through `search_points_range` itself, not just through `points::open`.
`cargo test -p lucene-search`: all tests pass (450 lib tests, up from 442).
`cargo clippy --workspace --all-targets -- -D warnings` clean. New-file
coverage: `points_query.rs` 98.72% lines (above the 95% bar). See
`docs/parity.md`'s updated row for the exact scope statement.

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
mostly reimplements these as aggs — likely never needed), backward-codecs. Merge-time
re-sorting of already-sorted segments (stored fields only, via k-way merge) and
multi-field NUMERIC index sort at flush time are both done, see `docs/parity.md`;
reordering doc values/norms/term vectors during a merge remains a long-tail item.

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
