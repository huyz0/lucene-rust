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
- Index sorting: single-field NUMERIC index sort is now supported at flush time
  (`segment_info.rs`'s `IndexSortField`/`SortMissingValue`, `segment_writer.rs`'s
  `flush_sorted_stored_only_segment`) -- see `docs/parity.md` for the exact
  scope. Still explicitly out of scope: multi-field/compound sorts, re-sorting
  during segment merges (merge always produces an unsorted `.si`), and the
  `.si` index-sort byte encoding is this port's own internal format, NOT
  verified byte-compatible with real Lucene's `Lucene99SegmentInfoFormat`
  (no real-Lucene-written sorted-segment `.si` fixture exists to derive the
  true `SortFieldProvider` wire format from).
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
this crate). **Compound-file segments (`.cfs`/`.cfe`) are out of scope**:
`SegmentReader::open` returns `Error::CompoundFileUnsupported` rather than
silently mis-reading — packing/unpacking compound sub-files into this
reader was more scope than this task's "centralize what callers already
did by hand" brief called for. **Verified**: opens the real single-segment
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
mostly reimplements these as aggs — likely never needed), backward-codecs. Multi-field
index sorts and merge-time re-sorting of already-sorted segments remain long-tail items
(single-field NUMERIC index sort at flush time is done, see `docs/parity.md`).

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
