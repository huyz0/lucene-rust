# b15-ffi-core

Files swept: `crates/lucene-core/src/lib.rs`, and all 22 modules of
`crates/lucene-ffi/src/` (`lib`, `error`, `handle`, `registry`, `raw`,
`directory`, `directory_reader`, `segment`, `writer`, `query`,
`points_query`, `sort`, `range_sort`, `facets`, `explain`, `highlighter`,
`results`, `results_scored`, `results_sorted`, `results_explain`,
`results_facets`, `results_fragments`).

## Java counterparts

**There is no Java counterpart for `lucene-ffi`.** It *is* the C-ABI/JNI
boundary — a layer real Lucene does not have, because real Lucene *is* the
JVM library. Per the protocol's rule 1, no Java path is invented for the
boundary machinery (`error`/`handle`/`registry`/`raw`/`results_*`). The
protocol's "compare against Java" step is applied only where an exported
function wraps a Lucene concept, and then against the Java class the
*wrapped* `lucene-search`/`lucene-index` function was itself ported from:

| Exported surface | Wraps | Java concept behind it |
|---|---|---|
| `ffi_open_directory`/`ffi_close_directory` | `lucene_store::FsDirectory` | `store/FSDirectory` |
| `ffi_open_segment`/`ffi_close_segment`/**`ffi_segment_set_live_docs`** | `blocktree::open`, `norms`, `doc_values`, `points`, **`live_docs::parse`** | `index/SegmentReader` (assembled by the caller), **`Lucene90LiveDocsFormat`** |
| `ffi_open_directory_reader`/`ffi_close_directory_reader` | `lucene_search::directory_reader::DirectoryReader` | `index/DirectoryReader.open` |
| `ffi_search_term_query{,_scored,_scored_maxscore,_scored_with_similarity}` | `search_term_query*` | `search/TermQuery` + `BM25Similarity` + `MaxScoreBulkScorer` |
| `ffi_search_boolean_query{,_scored,_scored_maxscore}` | `search_boolean_query*` | `search/BooleanQuery`/`BooleanScorer` |
| `ffi_search_phrase_query{,_scored}` | `search_phrase_query*` | `search/PhraseQuery` |
| `ffi_search_*_multi_segment{,_concurrent,_maxscore*}` | `multi_segment::*` | `search/IndexSearcher` leaf-slice fan-out + `TopDocs.merge` |
| `ffi_search_points_range` | `points_query::search_points_range` | `search/PointRangeQuery` |
| `ffi_sort_by_doc_value`/`ffi_sort_by_multi_valued_doc_value`/`ffi_numeric_doc_value_for_doc` | `doc_value_query::*` | `search/SortField`/`SortedNumericSelector`/`NumericDocValues` |
| `ffi_search_numeric_range_sorted_by_field{,_multi_segment}` | `doc_value_query`/`multi_segment` | `search/TopFieldCollector` |
| `ffi_facet_counts_sorted_set`/`ffi_range_facet_counts` | `facets::*` | `facet/SortedSetDocValuesFacetCounts`/`LongRangeFacetCounts` |
| `ffi_assemble_fragments` + `ffi_fragment_result_*` | `highlighter::assemble_fragments` | `uhighlight/DefaultPassageFormatter` + `PassageScorer` |
| `ffi_explain_*` + `ffi_explain_node_*` | `explain::explain_clause` | `search/Weight.explain` / `Explanation` |
| `ffi_open_writer`/`ffi_writer_*`/`ffi_close_writer` | `lucene_index::index_writer::IndexWriter` | `index/IndexWriter` (+ `TieredMergePolicy` config) |
| `ffi_get_last_error_message`, `guard`, `SlotMap`, the registries, `results_*` | — | **no Java counterpart** (boundary machinery) |

`crates/lucene-core/src/lib.rs` is a two-line placeholder
(`#![forbid(unsafe_code)]` + a doc comment pointing at `PLAN.md`). No Java
counterpart, nothing exported, nothing to sweep. **Verdict: swept-clean
(empty by design).**

## Findings

22 findings: 12 CORRECTNESS (all fixed), 5 MISSING (all fixed), 3 PERF (1
fixed and A/B measured, 2 recorded), 4 INTENTIONAL (2 of which are "this looks
wrong but is right" pins, deliberately recorded so the next reader does not
undo them). Findings 19-22 came out of the Tier-2 `quality-reviewer` pass on
this batch's own diff, run after the gate was green; all four were real and
all four are fixed.

### 1. [CORRECTNESS] Deleted documents were visible to every single-segment query

Java: `IndexSearcher` never returns a deleted document — every `Scorer` is
filtered through `LeafReader.getLiveDocs()`.

Us: `SegmentHandle` had no live-docs field and every single-segment call
passed `live_docs: None`. This *was* an honestly-documented deferral when
written, but it is no longer one: `lucene_codecs::live_docs::parse` exists,
`lucene_search`'s functions all take `Option<&FixedBitSet>`, and b13's
`DirectoryReader` already reads each segment's `.liv` and passes it through
`OpenedSegments::as_open_segments`. So the *multi-segment* FFI path had
silently started honouring deletions while the single-segment path had not —
two entry points to the same index disagreeing about which documents exist.

Consequence: a JNI caller using `ffi_open_segment` got deleted documents back
as live hits, with real scores, in query, points-range, explain and
range-sort results.

Resolution: **fixed.** New exported `ffi_segment_set_live_docs(segment_handle,
dir_handle, liv_name, liv_name_len, del_gen, del_count)` (`segment.rs`)
decodes the `.liv` via `live_docs::parse` — which cross-checks cardinality
against the commit's `del_count` — onto `SegmentHandle::live_docs`; a null
`liv_name` clears it. Threaded into all 9 `query.rs` searches,
`points_query.rs`, the 3 `explain.rs` calls and both `range_sort.rs` calls.
It is a separate additive call rather than four more `ffi_open_segment`
parameters because a `.liv`'s name is generation-suffixed, needs
`del_gen`/`del_count` to validate, and changes *without* the segment being
rewritten — so deletions can be refreshed without reopening the term
dictionary and postings, and the existing 30-parameter C signature stays
stable. Tests: `segment::live_docs_tests` (9 tests) against the real
Java-written `fixtures/data/live_docs_index/` — a term query matches the
deleted doc before attaching and not after, the decoded bitset equals the
fixture's `manifest.properties` exactly, clearing restores the old behavior,
and a wrong `del_gen`/`del_count`/negative value is rejected.

`sort.rs`/`facets.rs` deliberately still pass no live docs: they consume a
caller-supplied *candidate list* rather than producing a doc set, so
filtering it is the caller's decision. Documented on the new function.

### 2. [CORRECTNESS] `open_field_norms`' `live_docs: None` is right — pinned, not "fixed"

While fixing #1 the obvious next step is to pass live docs to
`FieldNorms::open` too. That would be **wrong**: Java's BM25 `avgdl` comes
from `CollectionStatistics` (`Terms.getSumTotalTermFreq()`/`getDocCount()`),
field-level term statistics that still count deleted documents. Subtracting
deletions there would make every score on a segment with deletions diverge
from Java's. Left as `None` with a doc comment explaining why, so the next
reader does not "fix" it. No code change; recorded because the reasoning is
non-obvious and the wrong change is one line away.

### 3. [CORRECTNESS] A failing call could hand back a *previous* call's error message

Not a Java divergence — a boundary-contract one. 81 non-`Ok` return paths
(every `NullPointer`, most `BufferTooSmall`, some `InvalidArgument`) returned
a status without calling `set_last_error`, so
`ffi_get_last_error_message` reported whatever unrelated failure last ran on
that thread — read by the caller as a diagnosis of the call that just failed.

Resolution: **fixed centrally in `error.rs`** rather than by 81 scattered
calls, so the invariant also holds for every path added later: `guard` clears
a thread-local `ERROR_RECORDED` flag before running the body and, on
`Err(status)` with the flag still clear, writes `status.default_message()`.
Tests: stale-message-does-not-leak, body's-own-message-is-kept, and every
variant has a non-empty default. Thread-locality confirmed: `LAST_ERROR`,
`CAPTURING_PANIC` and `ERROR_RECORDED` are all `thread_local!`, never shared
statics — criterion 4 holds.

### 4. [CORRECTNESS] `Vec::with_capacity` on caller-supplied lengths could **abort** the JVM

Nine production sites (`read_term_clauses`, two phrase term lists, the
explain phrase list, `ranges_from_raw`, `spans_from_raw`, two `range_sort`
segment lists, three `writer.rs` field lists) plus four `.to_vec()` helpers
allocated straight from a length the JNI caller supplied. A bad length (a
negative `int` widened to `usize`) makes `Vec::with_capacity` call
`handle_alloc_error`, which **aborts** — and an abort is not an unwind, so
`guard`'s `catch_unwind` cannot contain it. One caller bug would take the
OpenSearch node down instead of returning a status code.

Resolution: **fixed.** `raw::try_with_capacity`/`raw::try_to_vec`
(`try_reserve_exact`, failure → `FfiStatus::InvalidArgument` + message)
replace every production site. Also fixed the two `lens.iter().sum()` length
totals in `facets.rs`/`highlighter.rs`, which wrap silently in release and
would then back a shorter slice than the per-element slicing indexes into.
Tests: `try_with_capacity` rejects `usize::MAX / 4` instead of aborting, plus
a source-scanning invariant test that fails the build if any production line
in this crate reintroduces `Vec::with_capacity`.

### 5. [CORRECTNESS] Handle-index overflow could alias a live handle (type confusion)

`SlotMap::insert` packed `slots.len() as u32` into a 24-bit field guarded
only by a `debug_assert!`. At 2^24 live handles of one kind — reachable by a
caller leaking results handles — release builds truncated the index, so a new
handle silently aliased a *different* live entry in the same registry. The
generation tag cannot catch this (the aliased slot's generation is whatever
it happens to be), so the caller would get someone else's result set.

Resolution: **fixed.** `insert` returns `Option<u64>` and refuses past
`MAX_SLOTS`; `insert_checked` maps that to a new `FfiStatus::HandleLimit`
(code 11, additive — no existing code changed) plus a message. All 36 call
sites migrated. Tested via a `#[cfg(test)]`-only `set_max_slots_for_test` so
the exhaustion path is exercised without allocating 16.7M slots.

### 6. [CORRECTNESS/MISSING] Unvalidated numerics reaching scorers and decoders

Raised by the coordinator for `Bm25Params`, then swept across the whole
surface. Java's `BM25Similarity` constructor throws `IllegalArgumentException`
on `k1 < 0`/non-finite `k1`/`b` outside `0..=1`; b12 added the equivalent
`Bm25Params::new`, but `ffi_search_term_query_scored_with_similarity` still
built `Bm25Params { k1, b }` from raw FFI floats, so `new` had no non-test
caller. This is not cosmetic: a `b` outside `0..=1` makes BM25's length
normalization non-monotonic in the norm, which invalidates the
impacts-derived bounds the MAXSCORE paths use — **dropping matching
documents**, not merely scoring them oddly.

Resolution: **fixed**, and the same class swept everywhere:

| Input | Was | Now |
|---|---|---|
| `k1`/`b` (`ffi_search_term_query_scored_with_similarity`) | raw struct literal | `Bm25Params::new`, `InvalidArgument` + Lucene's verbatim message |
| candidate doc IDs (`ffi_sort_by_doc_value`, `ffi_sort_by_multi_valued_doc_value`, `ffi_facet_counts_sorted_set`, `ffi_range_facet_counts`) | unchecked | `sort::validate_candidates` against `max_doc` |
| `doc` (all three `ffi_explain_*`) | unchecked | `0..max_doc` or `InvalidArgument` |
| `max_doc` (`ffi_open_segment`) | unchecked | must be `>= 0` |
| `doc_bases[i]` (`ffi_search_numeric_range_sorted_by_field_multi_segment`) | unchecked | must be `>= 0` |
| `del_gen`/`del_count` (new `ffi_segment_set_live_docs`) | — | `del_gen >= 1`, `del_count >= 0`, then `live_docs::parse`'s own cross-check |
| merge-policy percentages / `target_search_concurrency` | — (newly exposed) | `0..=100`, `>= 1` |
| `PassageScorer` `k1`/`b`/`pivot` | — (newly exposed) | `k1 >= 0`, `b` in `0..=1`, `pivot > 0` |

The candidate-list case is the same hole
`ffi_numeric_doc_value_for_doc` already documents: the doc-values decoders
bounds-check against `max_doc` only for DENSE entries, so an out-of-range doc
falls through a SPARSE entry's docs-with-field bitset as "no value" — which a
`MissingValue::Default` sort then ranks as a document that does not exist and
a facet count silently ignores. Tests: one per rejected input, plus the
accepted boundary values (`k1 = 0`, `b = 0`, `b = 1`) still working.

Numerics deliberately left unvalidated, checked and found harmless: `top_n`
(`TopDocsCollector::new` preallocates nothing; `0` correctly yields no hits),
`missing_default`/range `min`/`max` (any `i64` is meaningful; `min > max`
yields empty), `window_chars` (clamped to the text). `selector`, `direction`,
`index_options`, `doc_values_types` were already validated.

### 7. [MISSING] Writer capabilities added after this wrapper was written

`IndexWriter` gained `delete_all()`, `set_live_commit_data`/
`live_commit_data`, and (in b10's `TieredMergePolicy` rewrite)
`deletes_pct_allowed`/`target_search_concurrency`, with
`force_merge_deletes_pct_allowed` gaining a real `find_forced_delete_merges`
to configure. None were reachable over the C ABI;
`ffi_writer_set_merge_policy` silently substituted defaults for three knobs,
and `writer.rs`'s module doc still claimed `MergePolicyConfig` "actually has
five knobs today" and that `forceMergeDeletesPctAllowed` had nothing to
configure — both stale.

Resolution: **fixed.** New `ffi_writer_delete_all`,
`ffi_writer_set_live_commit_data` (four parallel `(key, value)` arrays),
`ffi_writer_live_commit_data_len`/`ffi_writer_live_commit_data_entry`
(length-then-per-index accessor, both halves size-checked before either is
written so a `BufferTooSmall` never leaves a half-copied entry).
`ffi_writer_set_merge_policy` now takes all eight knobs. Module doc
corrected. Tests: `deleteAll` drops buffered *and* committed docs and the
emptiness is durable through a real `segments_N` read-back; it is refused
mid-two-phase-commit; live commit data round-trips and lands in the real
`SegmentInfos.user_data`; empty list clears; buffer-too-small writes nothing;
out-of-range knobs rejected.

`IndexWriter::apply_merge`/`add_postings_field`/
`set_custom_freq_postings_field`/`set_norms_field` remain unwrapped —
`apply_merge` for the reason `writer.rs` already documents (no FFI way to
*run* a merge, so it would be undrivable), the other three as a genuinely
open item (below).

### 8. [MISSING] Highlighter knobs and `Fragment` fields unreachable

b14 gave `FragmentConfig` `ellipsis`, `escape` and a `PassageScorer`, and
`Fragment` gained `start_offset`/`end_offset`/`score`. `ffi_assemble_fragments`
could not set any of the three (a JNI caller always got Java's defaults), and
the `PassageScorer` in particular decides *which* passages survive
`max_fragments` truncation — so the FFI caller could not influence which
highlights it got. The new `Fragment` fields had no accessor.

Resolution: **fixed.** `ffi_assemble_fragments` gained
`ellipsis`/`ellipsis_len` (null = Java's `"... "`, non-null-empty = the empty
string, so "no separator" stays expressible), `escape`, and
`scorer_k1`/`scorer_b`/`scorer_pivot` (validated per #6). New
`ffi_fragment_result_span` exposes `start_offset`/`end_offset`/`score`, any
out-parameter nullable. Tests: escaping escapes the text but not the
markers, every rejected scorer value, and the accessor's values/bounds/handle
paths.

### 9. [MISSING] Eight exported functions missing from `lib.rs`'s `pub use`

`ffi_search_boolean_query_multi_segment_maxscore{,_concurrent}`,
`ffi_writer_committed_doc_count`, `ffi_writer_delete_documents`,
`ffi_writer_update_document`, `ffi_writer_set_{postings,term_vector,doc_values}_field`.
The C symbols were always emitted (`#[no_mangle]`), so no caller was broken —
but the crate's Rust-level surface and `cargo doc` view understated what
exists, which is how #7's staleness went unnoticed. Resolution: **fixed**,
all re-exported (plus the new functions).

### 10. [PERF] The registries serialized every search across every JVM thread

Each query holds its registry guard for the *whole* search: `SlotMap::get`
hands back a `&SegmentHandle`/`&DirectoryReaderHandle` borrowed from the
guard, and the term dictionary, postings bytes and live docs are read out of
it while the collector runs. Under the `Mutex` these were, an N-core node ran
**one query at a time** regardless of thread or segment count — which also
made `ffi_search_*_multi_segment_concurrent`'s rayon fan-out pointless, since
caller threads queued outside the boundary instead. This is strictly worse
than Java, where `IndexSearcher` is explicitly concurrent-safe.

Resolution: **fixed.** The registries are `RwLock`; `registry::read_recovering`
takes the shared guard for the ~85 read-only lookup sites, `lock_recovering`
keeps the exclusive guard for `insert_checked`/`remove`/`get_mut`. Handle
validation is unchanged, and closing a handle still cannot race a call using
it (`ffi_close_*` needs the write guard the in-flight read guard holds off).

Measured A/B with `benchmarks/rust-runner/src/ffi_overhead.rs`'s new section D
(`ffi_search_term_query_multi_segment`, `fixtures/data/blocktree_index`, 400k
calls):

| | 1 thread | 4 threads | speedup |
|---|---|---|---|
| exclusive guard everywhere (before) | 620 ns/call | 2164 ns/call wall | **0.29x** — 4 threads were 3.5x *slower* |
| shared guard for lookups (after) | 405 ns/call | 347 ns/call wall | **1.17x** |

6.2x the four-thread throughput on the same work. Single-threaded FFI
boundary cost is unchanged at 47 ns/call against the <1µs budget.

Method: the same binary built twice with only `read_recovering`'s guard kind
changed, then run alternately — so the comparison survives whatever else the
machine is doing. The table above is from an otherwise-idle machine. Repeated
later at load average ~2 (sibling batches building), three alternating rounds
gave 2712/2782/2799 ns/call exclusive against 1075/1009/976 ns/call shared —
2.7x rather than 6.2x, because fewer cores are free for four threads to
spread over. The direction and order of magnitude hold; the exact ceiling is
a property of how much of the machine the caller owns, which is why both are
recorded here and in `registry.rs`'s module doc.

Also confirmed by unit test: two threads can hold read guards on one registry
simultaneously (a deadlock under the old `Mutex`), poison recovery still
works for the write guard, and the registries are independent locks.

One hazard the change introduces and that is guarded against: `std`'s
`RwLock` is neither reentrant nor upgradable, so holding a read guard while
asking for the same lock's write guard deadlocks. A script over the whole
crate found exactly one function that touches a registry both ways
(`ffi_segment_set_live_docs`); its read guard is block-scoped and dropped
before the write guard, with a comment saying why so a later edit does not
widen it.

Side effect on the existing poisoning tests: a panic under a *read* guard no
longer poisons the lock at all (std only poisons on write-guard panics) —
strictly better, since a shared borrow cannot leave the map half-written. The
two tests now assert the property that actually matters (a later call is not
wedged) rather than the poisoning mechanism.

### 11. [PERF] Remaining concurrency ceiling: the results registries' exclusive sections

1.17x rather than ~4x, because each call still takes the *exclusive* guard
twice on the results registries — once to insert the results handle, once for
the caller's `ffi_close_*` — and this fixture's query is only ~0.4µs, so
those two acquisitions dominate what is left. **Recorded, not fixed**:
sharding those registries (or an epoch/lock-free free list) is a
data-structure change rather than a lock-mode change, and real queries are
orders of magnitude longer than the guard sections, so the payoff is much
smaller than #10's. Noted in `registry.rs`'s module doc with the numbers.

### 12. [PERF] `open_segments()` repeated per multi-segment call

Every `ffi_search_*_multi_segment` call reconstructs each segment's
`DocInput`/`PosInput`/`PayInput` (header/footer checks). Measured at 120
ns/call on the fixture — real but small, and it is `DirectoryReaderHandle`'s
shape (owning bytes, not self-referential open inputs) that forces it.
**Recorded, unchanged**; the bench has decomposed this since it was written
precisely so it is not miscounted as boundary cost.

### 13. [INTENTIONAL] `pos_in`/`pay_in` always `None` for boolean queries

`ffi_search_boolean_query*` pass `pos_in: None`/`pay_in: None` even when the
segment has a `.pos`. Correct as long as clauses are flat `Clause::Term`,
which `read_term_clauses` guarantees — nested/phrase clause *construction*
over the C ABI remains the documented open item. No change.

### 14. [INTENTIONAL] `.pay` (payloads) still not openable

`ffi_open_segment` has no `pay_name`; `search_phrase_query` gets `pay_in:
None`, which is correct for a field without payloads and a clean
`FfiStatus::Search` for one that needs them. Unchanged — unlike `.liv`, no
consumer in this crate can use payloads yet, so wiring the file would add a
parameter nothing reads.

### 15. [INTENTIONAL] `WriterHandle`'s lifetime-erasing `unsafe`

`IndexWriter<'d>` borrows `&'d dyn Directory`, so the handle owns a
`Box<FsDirectory>` and constructs the writer against a `'static`-erased
borrow into it. Re-checked: soundness rests on (a) the `Box`'s heap address
being stable across moves and (b) struct field declaration order (`writer`
before `dir`) making the borrow drop first. Both hold, both are documented,
and the `unsafe impl Send`/`Sync` are justified by the registry lock being
the only access path. No change; the `Mutex` → `RwLock` move does not weaken
it (the lock is still the sole access path, and `RwLock<T>: Sync` needs
`T: Send + Sync`, which the explicit impls provide).

### 16. [INTENTIONAL] Ownership of returned buffers is consistent

Audited every `*_len`/`*_copy`/`ffi_close_*` triple: `results`,
`results_scored`, `results_sorted`, `results_facets` (+ `label` accessor),
`results_fragments` (+ per-index string accessors and the new `span`),
`results_explain` (per-node accessors), and the handle-less
`ffi_range_facet_counts`. Every one writes into a *caller*-allocated buffer;
no Rust-allocated memory crosses the boundary; each results kind has exactly
one matching `ffi_close_*`; each registry has its own `RegistryTag`, so a
handle of the wrong kind is rejected before it can be misread. No handle is
inserted on any error path (the `out_handle` null check is the first
statement of every producer, and `insert_checked` is the last), so no leak
is reachable from a failing call. No change.

### 17. [CORRECTNESS] `avgFieldLength` averaged lossy decoded norms (b12/b13 carry-over, owner b15)

Carried to this batch by b12 (F-7) and b13: `query::open_field_norms` was the
last production caller of `FieldNorms::open`, which derives `avgdl` by
decoding each doc's `SmallFloat`-quantized norm and averaging *those*. Java's
`BM25Similarity` uses `avgdl = sumTotalTermFreq / docCount` from
`CollectionStatistics`. The average of the lossy values sits 0.1-0.6% off the
average of the true lengths — enough to reorder documents at the top-k
boundary; M1's benchmark cross-check attributed 19 of 20 disagreeing queries
to this alone (`docs/benchmarks/verdict.md`). `open` also scanned only *live*
docs, while Java's `docCount` counts deleted ones.

Consequence: every FFI-served scored search (term, boolean, phrase, MAXSCORE,
custom-similarity, and all three explains, which share this helper) used a
slightly wrong `avgdl` in every score's denominator.

Resolution: **fixed.** `open_field_norms` now calls
`FieldNorms::from_field_stats(data, entry, field_terms.sum_total_term_freq,
field_terms.doc_count)` with the field's own `.tmd` aggregates via
`BlockTreeFields::field(name)`. Both divergences go at once: the exact Java
formula, and a `doc_count` that (like Java's) includes deleted documents — so
this stays correct alongside finding 1's deletion support. Bonus: O(1)
instead of O(maxDoc) per query, so the old "recomputed on every call" caveat
in the doc comment no longer needs justifying.

### 18. [MISSING] No `maxClauseCount` (b12 F-17, owner b15)

Java's `BooleanQuery.Builder.add` throws `TooManyClauses` past
`IndexSearcher.getMaxClauseCount()` (default 1024). This port's
`BooleanQuery` has no builder and no cap, and b12 flagged the FFI boundary as
the right place for the policy — it is the only place a clause list is built
from untrusted input (a caller-supplied `count`). Without it, a caller (or a
JVM-side rewritten prefix/wildcard query) could hand over a million clauses
and have a million-clause query actually executed: a denial-of-service shape
Java refuses outright.

Resolution: **fixed.** `query::MAX_CLAUSE_COUNT = 1024`, enforced in
`read_term_clauses` — so every `ffi_search_boolean_query*` and
`ffi_explain_boolean_query`, plus the multi-segment wrappers that reuse the
same decoder, are covered by one check. Per clause list, matching Java's
per-`Builder` counter. Surfaced as `InvalidArgument` with Java's own message
shape. Tested: 1025 clauses rejected with a `maxClauseCount` message and no
handle issued; exactly 1024 still accepted (Java's check is `>`, not `>=`).

### 19. [CORRECTNESS] `deletes_pct_allowed` was validated against the wrong range

Introduced by finding 7's own fix, caught by the Tier-2 review. Java's three
`TieredMergePolicy` setters do **not** share a range, and this treated two of
them as one uniform "0..=100 percentage":

| Java setter | Java's check |
|---|---|
| `setForceMergeDeletesPctAllowed` | `v < 0.0 \|\| v > 100.0` |
| `setDeletesPctAllowed` | `v <= 0 \|\| v > 50` |
| `setSegmentsPerTier` | `v < 2.0` |
| `setTargetSearchConcurrency` | `< 1` |

So the boundary accepted `0.0` and everything in `50.1..=100.0` for
`deletes_pct_allowed`, which Java refuses outright — while the doc claimed
"Java's own setters throw `IllegalArgumentException` for the same
out-of-range values". `segments_per_tier` was not validated at all, though
the same doc claimed every knob was.

Resolution: **fixed.** Each bound is now Java's own, separately, with Java's
own message text, and the doc spells all four out and says explicitly that
they are not uniform. Tests assert each edge in both directions — including
`base(10.0, 101.0, 1)` as a named regression guard on the value that used to
be accepted.

### 20. [CORRECTNESS] The clause cap was per clause list, not per query

Also introduced by this batch (finding 18) and caught by the review. Java's
counter lives on the `BooleanQuery.Builder` and every clause goes through the
same `add` regardless of `Occur`
(`if (clauses.size() >= IndexSearcher.maxClauseCount) throw new TooManyClauses();`).
Checking inside `read_term_clauses` — called once per `must`/`should`/
`must_not` — let a caller pass `3 x 1024` clauses in one query, three times
Java's ceiling, defeating the point of a DoS guard.

Resolution: **fixed.** `query::check_clause_count(must, should, must_not)`
sums the three (saturating) and is called once per entry point, before any
decoding, at all eight boolean entry points (three in `query.rs`, one in
`explain.rs`, four multi-segment ones in `directory_reader.rs`). Test asserts
`3 x 512 = 1536` split across the three lists is rejected, and that exactly
1024 summed is still accepted. The doc's ">, not >=" aside was also wrong
about Java (Java's is `>=` against the pre-add size, which yields the same
1024 maximum) and now says so correctly.

### 21. [CORRECTNESS] A deleted document explained as "no matching term"

Finding 1 pushed `live_docs` down into `explain_clause`, which produces the
right *verdict* (no match) but the wrong *reason*: the doc falls out at the
postings lookup and the tree says `no matching term`, indistinguishable from
a live document that simply lacks the term. Java answers this one layer up,
at exactly the layer `explain.rs` is:

```java
final Bits liveDocs = ctx.reader().getLiveDocs();
if (liveDocs != null && liveDocs.get(deBasedDoc) == false) {
  return Explanation.noMatch("Document " + doc + " is deleted");
}
```

`ffi_explain_node_description` is the string an OpenSearch `_explain`
response renders, so collapsing the two cases is a real diagnosis
regression.

Resolution: **fixed.** `explain::deleted_doc_explanation` reproduces Java's
branch (message text included) ahead of any weight, in all three
`ffi_explain_*`. Test against the real `live_docs_index` fixture asserts
deleted docs 1 and 3 explain as `Document N is deleted`, a live document
lacking the term still gets the ordinary no-match reason, and a live matching
document is unaffected.

### 22. [CORRECTNESS] The `unsafe impl Send/Sync for WriterHandle` safety comment became false

Finding 10's `Mutex` → `RwLock` change invalidated the comment's load-bearing
premise ("at most one thread ever touches a given `WriterHandle` at a time"):
`read_recovering(writers())` now hands `&WriterHandle` to many threads at
once. The impls are still sound, but for a *different* reason than the one
written down — and on the one `unsafe impl` at the JVM boundary, a safety
comment that argues the wrong thing is the defect.

Resolution: **fixed.** The comment now rests the argument where it actually
holds: `IndexWriter` lives in a `#![forbid(unsafe_code)]` crate and has no
interior mutability at all (no `Cell`/`RefCell`/`UnsafeCell`), so every
mutation needs `&mut self`, which only the write guard can produce — making
concurrent shared `&WriterHandle` data-race-free. It also records what the
old wording said and why it no longer applies, so the change is not silently
reverted.

### Tier-2 advisories, also addressed

- `ffi_assemble_fragments`' `pivot > 0` guard was aimed at the wrong formula:
  the singularity is `PassageScorer.norm = 1 + 1/ln(pivot + passageStart)`,
  `+Inf` at `pivot + passageStart == 1` and sign-flipped below — and the test
  explicitly accepted `pivot = 0.001`. Tightened to `pivot > 1.0` (the
  weakest bound that holds for every `passageStart >= 0`), with the real
  reason documented.
- `a_different_scorer_can_change_which_fragment_survives_truncation` asserted
  only that a fragment came back, which would pass if the three scorer floats
  were accepted and dropped. Added
  `scorer_parameters_reach_the_passage_score`, which reads the score back
  through the new `ffi_fragment_result_span` and asserts it *changes* with
  `pivot` and with `k1` — a direct proof the values reach `PassageScorer`.
- The ellipsis test claimed to distinguish null from empty but asserted
  neither. Renamed to what it can actually prove and made to assert it
  (all three spellings accepted, none changes fragment assembly), with a note
  that `ellipsis` is only consumed by `format_fragments`, which no FFI
  function reaches yet.
- The `Vec::with_capacity` invariant test split production from test code at
  the *first* `#[cfg(test)]`, which is wrong for the three files in this diff
  that add a second test module after an existing one. Rewritten with
  brace-depth tracking (plus a test for the scanner itself), extended to
  `String::with_capacity` and `vec![x; n]`, with an `// alloc-ok: <reason>`
  opt-out for the reader-derived lengths in `directory_reader.rs`. A
  `clippy.toml` `disallowed-methods` entry would be a stronger gate but is
  workspace-scoped, and `Vec::with_capacity` is legitimate in every other
  crate here.
- `doc_bases` checked negativity but not `doc_base + max_doc` overflow, which
  wraps back into the negative global doc IDs the check exists to exclude.
  Now `checked_add`.
- `a_del_count_that_disagrees_with_the_file_is_a_decode_error` started from a
  segment with nothing attached, so it could not observe whether a *failed
  re-attach* preserves the previous bitset. Added
  `a_failed_reattach_leaves_the_previous_bitset_intact`.
- `error.rs`'s step-4 invariant is stated globally but
  `ffi_get_last_error_message` deliberately sits outside it (backfilling
  there would destroy the message the caller is retrying to read). The
  exception and its reason are now in the doc.

### Panic safety and `unsafe` scope (criteria 1 and 2), re-verified

- Every one of the 79 exported `extern "C" fn`s wraps its body in
  `guard(...)`. The only exception is `ffi_get_last_error_message`, which
  forwards to `error::get_last_error_message` — a function that allocates
  nothing and cannot panic (it is a bounds check plus a `copy_nonoverlapping`),
  so a `catch_unwind` there would be decoration. Verified by script, not by eye.
- Every raw pointer is null-checked before deref, all through `raw.rs`'s two
  helpers or an explicit `is_null()`. No `as`-cast of a `*const c_char`
  anywhere (the `c_char`-signedness trap); confirmed by a clean
  `cargo clippy --target aarch64-unknown-linux-gnu --all-targets -D warnings`
  as well as the native one.
- Every `unsafe` block carries a SAFETY comment (audited by script; the
  handful of apparent misses are blocks covered by one comment above a group
  of consecutive lines).

## Verdict

| File | Verdict |
|---|---|
| `lucene-core/src/lib.rs` | swept-clean (empty placeholder, no Java counterpart) |
| `lucene-ffi/src/lib.rs` | swept-clean (docs corrected: `.liv` closed, stale multi-segment claim removed, 8 missing re-exports added) |
| `error.rs` | swept-clean (finding 3 fixed) |
| `handle.rs` | swept-clean (finding 5 fixed) |
| `registry.rs` | swept-clean (findings 10, 22 fixed + measured); finding 11 open |
| `raw.rs` | swept-clean (finding 4's helpers added here) |
| `directory.rs` | swept-clean |
| `segment.rs` | swept-clean (findings 1, 6 fixed) |
| `query.rs` | swept-clean (findings 1, 2, 4, 6, 17, 18, 20 fixed) |
| `points_query.rs` | swept-clean (finding 1 fixed) |
| `sort.rs` | swept-clean (findings 4, 6 fixed) |
| `range_sort.rs` | swept-clean (findings 1, 4, 6 fixed + the doc_base-overflow advisory) |
| `facets.rs` | swept-clean (findings 4, 6 fixed) |
| `explain.rs` | swept-clean (findings 1, 4, 6, 20, 21 fixed) |
| `highlighter.rs` | swept-clean (findings 4, 6, 8 fixed + the pivot-bound and test-meaningfulness advisories) |
| `directory_reader.rs` | swept-clean (already honoured `.liv`; docs corrected) |
| `writer.rs` | swept-clean (findings 4, 7, 19 fixed); `add_postings_field`/`set_custom_freq_postings_field`/`set_norms_field`/`apply_merge` open |
| `results*.rs` (6 files) | swept-clean (finding 8's accessor added to `results_fragments.rs`) |

## Open items

- **Results-registry exclusive sections cap concurrency at ~1.2x** (finding
  11). Needs sharding or a lock-free free list, not a lock-mode change.
- **`IndexWriter::add_postings_field`/`set_custom_freq_postings_field`/
  `set_norms_field`/`add_term_vector_field`/`add_document_with_custom_freq_terms`
  have no FFI wrapper.** Mechanical, same shape as the setters already
  wrapped; left out of this batch to keep its diff reviewable.
- **`apply_merge` / manual merge execution** — unchanged deferral, needs an
  FFI way to *run* a merge first.
- **`.pay` payloads, nested/phrase `BooleanQuery` clause construction,
  multiple sort fields** — unchanged deferrals, all blocked on wire-format
  design rather than plumbing.
- **`term_vectors_query::matched_term_offsets` has no wrapper**, so a
  JNI-only caller cannot compute the spans `ffi_assemble_fragments` consumes.
  Unchanged.

## Tier-2 review

The `quality-reviewer` subagent was run on this batch's diff after the gate
was green (AGENTS.md's workflow step). It returned four gating findings —
all four real, all four this batch's own regressions rather than pre-existing
ones — and eight advisories. Findings 19-22 and the advisory list above are
the record; nothing from that review is left open.

## Gate

`cargo fmt --all`, `cargo clippy -p lucene-ffi -p lucene-core --all-targets
-- -D warnings` (and the same for `--target aarch64-unknown-linux-gnu`), and
`cargo test -p lucene-ffi -p lucene-core` all pass: 439 tests, 0 failures.
`cargo llvm-cov -p lucene-ffi --summary-only`: every file ≥ 95% lines
(lowest `directory_reader.rs` 95.75%; total 97.97%).
