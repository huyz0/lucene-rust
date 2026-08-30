# b13 — search readers, doc-ID sets, query cache

Batch: `crates/lucene-search/src/{multi_segment,directory_reader,docid_set,field_norms,query_cache,soft_deletes}.rs`.
Java read at `/home/tuong/work/lucene` (Lucene 10.5.0).

**Totals: 29 findings — 8 CORRECTNESS (all fixed), 9 MISSING (7 fixed, 2
recorded), 8 PERF (6 fixed, 2 recorded with measurements), 4 INTENTIONAL.**

F-25 through F-30 came out of the Tier 2 semantic review of this batch's own
diff, not out of the first read of the Java — they are listed under the file
they belong to.

---

## `crates/lucene-search/src/directory_reader.rs`

Java counterparts:
`lucene/core/src/java/org/apache/lucene/index/StandardDirectoryReader.java`,
`DirectoryReader.java`, `SegmentReader.java`, `SegmentCoreReaders.java`,
`SegmentDocValues.java`, `BaseCompositeReader.java`, `LeafReaderContext.java`,
`IndexWriter.getReader` (via `StandardDirectoryReader.open(IndexWriter, ...)`).

| Rust | Java | Verdict |
|---|---|---|
| `DirectoryReader::open` | `DirectoryReader.open(Directory)` → `StandardDirectoryReader.open` | equivalent (scoped) |
| `DirectoryReader::open_at` | `StandardDirectoryReader.open(dir, infos, oldReaders, ...)` with `oldReaders == null` | equivalent |
| `DirectoryReader::open_at_reusing` | `StandardDirectoryReader.createSegmentReaders` + `getOldSegmentReader` + `createOrReuseSegmentReader` | **was divergent (F-1), fixed** |
| `DirectoryReader::open_if_changed` | `DirectoryReader.openIfChanged` → `doOpenIfChanged` → `doOpenNoWriter`/`isCurrent` | divergent on the currency test (F-11, INTENTIONAL) |
| `DirectoryReader::segment_readers` | `CompositeReader.getSequentialSubReaders()` | equivalent |
| `DirectoryReader::open_segments` / `OpenedSegments::as_open_segments` | `IndexReader.leaves()` → `List<LeafReaderContext>` | equivalent (two-step for lifetime reasons, documented in the module) |
| `SegmentReader::open` | `new SegmentReader(SegmentCommitInfo, ...)` + `SegmentCoreReaders` ctor | **was missing norms (F-3), fixed** |
| `SegmentReader::clone_reader` | `oldReader.incRef()` | **was a deep copy (F-2), fixed** |
| `SegmentReader::reopen_with_new_live_docs` | `new SegmentReader(commitInfo, oldReader, liveDocs, hardLiveDocs, numDocs, false)` | **new (F-2)** |
| `SegmentReader::{points_files, segment_id, field_infos, doc_values_meta, doc_values_data}` | `SegmentCoreReaders.pointsReader` / `SegmentInfo.getId` / `FieldInfos` / `SegmentDocValues` | equivalent |
| `SegmentReader::{live_docs, norms_entry, norms_data, field_norms, soft_deletes_field}` | `SegmentReader.getLiveDocs`, `getNormValues`, `FieldInfos.getSoftDeletesField` | **new (F-3, F-4)** |
| `CompoundArchive`, `open_segment_file`, `find_segment_file_name`, `codec_suffix_of`, `find_file_ending` | `CompoundDirectory` / `IndexFileNames.stripSegmentName` / per-format file lookup | equivalent; `codec_suffix_of` is a Rust-only dedup of three copies of the same derivation |

Java methods with **no** Rust counterpart (all recorded, none fixed):
`IndexReader.incRef/decRef/close/tryIncRef` (refcounting is `Arc`/`Drop` here),
`registerParentReader`/`notifyReaderClosedListeners`, `IndexReader.CacheHelper`
(`getReaderCacheHelper`/`getCoreCacheHelper`), `getIndexCommit`,
`listCommits`/`indexExists`, `doOpenFromCommit` (open at a *specific* commit),
`leafSorter`, `ExitableDirectoryReader`, `FilterDirectoryReader`,
`SoftDeletesDirectoryReaderWrapper` (the reader-level wrapper; the read-side
rule itself is `soft_deletes.rs`), `StandardDirectoryReader.open(IndexWriter, …)`
NRT-from-writer, `isCurrent()` against a writer, `SegmentDocValues`' per-generation
doc-values-producer sharing, `SegmentReader.getTermVectorsReader`/`storedFields`
(this port opens those directly, not through the reader).

### F-1 `[CORRECTNESS]` reuse ignored `fieldInfosGen`/`docValuesGen`

Java's reuse test is
`oldReader.getSegmentInfo().getDelGen() == commitInfo.getDelGen() &&
oldReader.getSegmentInfo().getFieldInfosGen() == commitInfo.getFieldInfosGen()`
(`StandardDirectoryReader.createOrReuseSegmentReader`). We matched on
`segment_name`, `segment_id` and `del_gen` only.

A numeric doc-values update bumps `docValuesGen` and `fieldInfosGen` and leaves
`delGen` alone (`ReadersAndUpdates.writeFieldUpdates` →
`SegmentCommitInfo.advanceDocValuesGen`/`advanceFieldInfosGen`). Every
`openIfChanged` after such a commit therefore reused the reader holding the
*previous* generation's parsed `.dvm`/`.dvd` — a doc-values update was
permanently invisible to any reader that had been open across it. `SegmentCommitInfo`
already carried both generations; the predicate simply never looked.

**Fixed**: `SegmentReader` now records `field_infos_gen`/`doc_values_gen` and
`open_at_reusing` matches on all three generations.
**Tests**: `open_if_changed_reopens_a_segment_whose_doc_values_gen_changed`
(the segment's `.fnm` is deleted before the reopen, so reuse would succeed and a
fresh open must fail — failure is the assertion) and its opposite,
`open_if_changed_reuses_a_segment_whose_generations_all_match`. The pair pins
both directions, so neither "always reuse" nor "never reuse" can regress in.

### F-2 `[PERF]` reopen deep-copied the segment core instead of sharing it

Java shares `SegmentCoreReaders` by reference count across every reopen that
keeps a segment: the "no change" branch is `oldReader.incRef()`, and even the
"liveDocs changed" and "DV changed" branches call
`new SegmentReader(commitInfo, oldReader, …)`, which re-reads *only* the `.liv`
and doc values and carries the terms/postings/field-infos core over by pointer.

We did neither. `clone_reader` was `#[derive(Clone)]`, so reusing a segment
deep-copied its decoded `BlockTreeFields` (the FST and every field's term
metadata), its `FieldInfos`, its `FixedBitSet` live docs and its parsed
`DocValuesMeta`; and a `del_gen` bump fell all the way through to a fresh
`SegmentReader::open`, re-reading and re-decoding `.si`/`.fnm`/`.tim`/`.tip`/
`.tmd` from disk to change one bitset.

**Fixed**, in two parts:
1. `fields`, `field_infos`, `live_docs` and `dv_meta` are now `Arc`, so
   `clone_reader` is a handful of refcount bumps — Java's `incRef` — instead of
   an FST copy. (The `.doc`/`.pos`/`.pay`/`.dvd` buffers were already `Arc`.)
2. New `SegmentReader::reopen_with_new_live_docs`, Java's "only liveDocs
   changed" branch: shares the core and reads only the new `.liv`. It needs
   nothing from `.si` (`max_doc` and `segment_id` are already on the reader), so
   a delete-only reopen now touches exactly one file.

**Cost**: a delete-only commit went from re-reading and re-decoding every file
of the segment to reading one `.liv`; a reuse went from copying the FST to a
refcount bump. **Tests**:
`a_delete_only_reopen_shares_the_core_and_rereads_only_the_liv_file` deletes
*all* of the segment's own files before reopening (so a full open cannot
succeed) and still asserts the new deletions are visible;
`a_reused_segment_shares_its_core_rather_than_copying_it` asserts
`Arc::ptr_eq` on the term dictionary and field infos across the reopen.

### F-3 `[MISSING]` norms were never opened

`SegmentCoreReaders` opens the segment's `NormsProducer` alongside its postings
and field infos. `SegmentReader::open` read `.fnm`/`.tim`/`.tip`/`.tmd`/`.doc`/
`.pos`/`.pay`/`.liv`/`.dvm`/`.dvd`/`.kdm`/`.kdi`/`.kdd` and simply skipped
`.nvm`/`.nvd`. A caller who opened an index the normal way therefore had no way
to reach norms at all and had to pass `None` to every `*_scored` entry point,
scoring every document at `UNNORMED_FIELD_LENGTH`. That is a wrong BM25
length-normalization term, not a missing feature — and it is invisible, because
`None` is also the legitimate value for a field with `omitNorms`.

**Fixed**: `SegmentReader::open` reads `.nvm`/`.nvd` (present together or not at
all, same contract `.tim`/`.dvm` already have, and the same codec-suffix
derivation, now factored into `codec_suffix_of`), validates the `.nvd`
header/footer, and exposes `norms_entry`/`norms_data` plus a
`field_norms(field)` convenience that builds a `FieldNorms` through
`from_field_stats` — i.e. `avgFieldLength = sumTotalTermFreq / docCount` from
the field's real `.tmd` counters, which is `BM25Similarity`'s own formula.
**Test**: `opens_real_norms_and_scores_differently_from_the_no_norms_fallback`,
against the real Java-written `blocktree_index` fixture, asserts the same
documents match with and without norms but that the *ranking changes* — on that
fixture real norms swap the top two hits. `a_segment_without_norms_files_reports_none`
covers the no-norms segment.

This also hands b15 the API the ledger's `avgFieldLength` carry-over needs: the
FFI call site can now reach a `from_field_stats`-built `FieldNorms` through the
reader instead of constructing a lossy one itself.

### F-4 `[MISSING]` no `getLiveDocs`/`getSoftDeletesField` accessor

`live_docs` was private, so `soft_deletes::effective_live_docs` could not be
composed with a reader-opened segment at all without going through
`as_open_segments`, and the `.fnm`'s soft-deletes field name (which
`FieldInfos::soft_deletes_field()` already parsed) was unreachable.
**Fixed**: `SegmentReader::live_docs()` and `SegmentReader::soft_deletes_field()`.
Covered by the tests above.

### F-11 `[INTENTIONAL]` `isCurrent()` compares generation, not version

Java's `isCurrent()` is
`SegmentInfos.readLatestCommit(dir).getVersion() == segmentInfos.getVersion()`;
we compare `generation`. For a directory-backed reader the two move together
(every commit writes a new `segments_N` *and* bumps the version), so this is
equivalent for everything this port can currently do. It stops being equivalent
for an NRT reader obtained from a writer, which this port does not have.
Recorded, not changed.

### F-12 `[INTENTIONAL]` no "same name, different id" guard

`StandardDirectoryReader.getOldSegmentReader` throws `IllegalStateException`
when a candidate has the segment's name but a different id — a best-effort
detector for "the app `rm -rf`'d the index under an open reader". We match on
name *and* id, so such a candidate simply isn't reused and the segment is opened
fresh, which is safe rather than diagnostic. Recorded.

### F-25 `[MISSING]` a half-present `.nvm`/`.nvd` pair degraded to "no norms"

The first cut of F-3 followed the `.dvm`/`.dvd` precedent: `(Some, Some)` opens
norms, anything else means none. But `.tim`/`.tip`/`.tmd` raises
`PartialBlockTreeFiles` for exactly that condition, and F-3's own argument is
that scoring without norms is *silently wrong* rather than merely absent — so
degrading silently on a corrupt half-pair undercuts the fix it is part of.
**Fixed**: a `PartialNormsFiles { segment }` error variant mirroring
`PartialBlockTreeFiles`. **Test**:
`a_segment_with_only_half_a_norms_pair_is_an_error` writes a `.nvm` with no
`.nvd`, lists it in the segment's `.si`, and asserts the typed error.

### F-26 `[MISSING]` `field_norms()`'s `avgFieldLength` is per-leaf, Java's is reader-wide

`IndexSearcher.fieldStats` sums `getSumTotalTermFreq()`/`getDocCount()` **over
every leaf**, so Java's `avgdl` is reader-wide the same way its
`docFreq`/`docCount` are. `SegmentReader::field_norms()` reads *this segment's*
`.tmd` counters. That is the same class of divergence as F-5/F-6 — half fixed,
since this batch made `docFreq`/`docCount` reader-wide and left `avgdl` alone.

**Recorded, not fixed**: a `SegmentReader` sees only itself, so the sum cannot
be taken there. The reader-wide sibling belongs next to
`multi_segment::global_term_stats`, which has the whole leaf list, and would
need `FieldNorms` to carry an externally-supplied `avg_field_length` — a change
to `field_norms.rs`'s constructor surface that would ripple into `explain.rs`
and the FFI (b14/b15's files) mid-sweep. `field_norms()`'s doc comment now
states the divergence and where the fix belongs. The new norms test uses a
single segment, where the two coincide, so it cannot show it either — noted
there too.

### Verdict

Swept clean. Open items are F-26 and the enumerated no-counterpart list above —
chiefly
`FilterDirectoryReader`/`ExitableDirectoryReader`, opening at a specific
`IndexCommit`, and the NRT-from-`IndexWriter` reader, none of which have a
caller in this port yet.

---

## `crates/lucene-search/src/multi_segment.rs`

Java counterparts: `lucene/core/src/java/org/apache/lucene/search/IndexSearcher.java`
(`search`, `searchAfter`, `slices`, `LeafSlice`, `search(List<LeafReaderContext>, Weight, …)`,
`termStatistics`/`collectionStatistics`), `TopDocs.merge`, `TaskExecutor.java`,
`CollectorManager.java`, `TimeLimitingBulkScorer.java`, `ReaderUtil.java`.

| Rust | Java | Verdict |
|---|---|---|
| `OpenSegment` | `LeafReaderContext` (`reader`, `docBase`) | equivalent, flattened |
| `merge_multi_segment_scored` | `IndexSearcher.search(leaves, weight, collectorManager)` + `TopDocs.merge` | equivalent |
| `merge_multi_segment_scored_with_deadline` | `IndexSearcher` + `TimeLimitingBulkScorer`/`QueryTimeout` | divergent granularity (F-9, INTENTIONAL) |
| `merge_multi_segment_scored_concurrent` | `IndexSearcher.search` with an `Executor` + `TaskExecutor.invokeAll` | divergent slicing (F-8, PERF/INTENTIONAL) |
| `merge_multi_segment_scored_concurrent_with_deadline` | same, plus `QueryTimeout` | as above |
| `global_term_stats` | `IndexSearcher.termStatistics` + `collectionStatistics` | equivalent |
| `global_boolean_stats` | the same, walked over a query's leaf terms (`Weight` construction time in Java) | equivalent for the clause shapes it walks |
| `search_term_query_multi_segment` | `IndexSearcher.search(TermQuery, n)` | **was divergent (F-5), fixed** |
| `search_term_query_multi_segment_concurrent` | same, with an `Executor` | **was divergent (F-6, F-5), fixed** |
| `search_boolean_query_multi_segment{,_concurrent,_maxscore,_maxscore_concurrent}` | `IndexSearcher.search(BooleanQuery, n)` (± `WANDScorer`) | equivalent |
| `merge_multi_segment_by_field`, `search_numeric_range_sorted_by_field_multi_segment`, `DocValueSegment` | `TopFieldCollector` + `TopFieldDocs.merge` | equivalent, scoped to one numeric key |
| `maxscore_keeps_global_stats`, `single_term_global_stats` | no Java counterpart | Rust-only guard for F-5 |

Java methods with **no** Rust counterpart: `IndexSearcher.slices`/`LeafSlice`/
`slicesWithSegmentPartitions` (F-8), `searchAfter`, `count(Query)`,
`rewrite`/`createWeight`, `getTopReaderContext`, `setSimilarity`,
`CollectorManager.reduce` as a user-facing extension point,
`TaskExecutor`'s caller-runs work stealing (rayon's is used instead),
`ReaderUtil.subIndex`/`getTopLevelContext`.

### F-5 `[CORRECTNESS]` reader-wide term statistics were silently dropped on every fallback

`search_term_query_scored_maxscore_with_stats` (in `lib.rs`) takes the
reader-wide `CollectionStats` and then, on each of its three fallback paths —
no `.doc` input, `docFreq <= 1`, and an index option `LazyDocsCursor` rejects —
returns `search_term_query_scored(...)`, the **no-stats** entry point. The leaf
silently reverts to its own `docFreq`/`docCount`.

This is the cross-segment idf bug this module's own doc comment records, alive
on a path nothing tested. On a two-segment index where `fox` appears in 1 of 1
documents in one leaf and 1 of 4 in the other, the three candidate idfs are
0.288 (leaf A), 1.204 (leaf B) and 0.876 (reader-wide): the merged top-k fills
from whichever leaf makes the term look rarest.

The existing `..._concurrent_matches_sequential` test cannot see it. It
duplicates one fixture segment whose `docFreq` is exactly half its `docCount`,
and `idf(d, 2d) == idf(2d, 4d) == ln 2`, so summing the counters is numerically
invisible there.

**Fixed at the caller**, because the defect is in `lib.rs`, another sweep
batch's in-flight file (b12): `multi_segment.rs` now routes a leaf through the
eager `search_term_query_scored_with_stats` — which honours `global` correctly —
whenever `maxscore_keeps_global_stats` says the MAXSCORE path would drop them,
and keeps the pruned path otherwise. The remaining fallback (`LazyDocsCursor`
rejecting the index options) is unreachable for a scored term query: it only
fires for `IndexOptions::None`, which carries no postings to score.

**The real fix belongs in `lib.rs`**: those three `return`s should forward
`global` to `search_term_query_scored_with_stats`. Recorded for b12 below and
in `docs/sweep/findings.md`.

**Tests**: `global_stats_across_concurrent_leaves::concurrent_term_query_uses_reader_wide_idf_like_the_sequential_path`
builds two deliberately lopsided real segments through `IndexWriter`, asserts
sequential and concurrent agree, and asserts the score equals the *reader-wide*
idf rather than either leaf's own — so "both wrong the same way" cannot pass it.
Plus a tripwire, `the_maxscore_entry_point_still_ignores_global_stats_on_its_fallback`:
the guard re-derives another function's private control flow from the outside,
so if b12's fix lands (or its conditions move) the guard silently mis-routes and
nothing else would fail. That test hands the MAXSCORE entry point a `global`
nothing like the segment's own counters and asserts it is discarded — when that
stops being true it fails, pointing straight at the guard to delete.
`search_boolean_query_scored_maxscore_with_stats` already forwards `global` on
its fallback, so the term-query one is the outlier.

### F-6 `[CORRECTNESS]` the concurrent term-query fan-out never computed global stats at all

Independently of F-5, `search_term_query_multi_segment_concurrent` called
`search_term_query_scored_maxscore` (no stats) where its sequential twin called
`..._with_stats(global)`. The module doc claims the two are "provably
byte-for-byte identical"; they were not, for any multi-segment index.
**Fixed**: it now computes `global_term_stats` and threads it through, exactly
as the sequential path does. Same test as F-5.

### F-8 `[PERF]` no `LeafSlice` grouping

`IndexSearcher.slices` sorts leaves by descending `maxDoc`, gives any leaf over
`maxDocsPerSlice` (250 000) its own slice, and groups the rest until
`maxSegmentsPerSlice` (5) or `maxDocsPerSlice` is reached — one `Collector` per
slice, reduced by a `CollectorManager`. We hand `doc_bases.par_iter()` to rayon
and give every leaf its own `TopDocsCollector`.

Reasoned rather than measured, because the shapes differ in kind: rayon's
`par_iter` over an indexed range does *adaptive* recursive splitting with work
stealing, so it already batches adjacent leaves into per-thread runs and
rebalances when one leaf turns out expensive — the load-balancing problem
Java's descending-`maxDoc` sort is a static approximation of. What we do not
reproduce is (a) the per-slice collector (we allocate one bounded
`TopDocsCollector` per *leaf*, which is `top_n` `ScoreDoc`s each — for a
1000-segment index at `top_n = 10` that is 1000 small vectors instead of 200),
and (b) `slicesWithSegmentPartitions`, Java's intra-segment partitioning for a
single huge leaf, which this port cannot express at all since its per-segment
search functions take no doc-ID range. Recorded, not fixed: (a) is bounded and
small, (b) needs a range parameter on every scorer entry point.

### F-9 `[INTENTIONAL]` per-segment timeout granularity, `bool` instead of `TotalHits.Relation`

`TimeLimitingBulkScorer` checks the clock every `~grow`-doubling interval of
documents *within* a leaf and throws `TimeExceededException`, which
`IndexSearcher` turns into a `TopDocs` with
`totalHits.relation == GREATER_THAN_OR_EQUAL_TO`. We check once per segment and
return a `bool` alongside the hits, because this port has no `TotalHits` type.
Already documented in the module; recorded here for completeness.

### F-10 `[INTENTIONAL]` `global_boolean_stats` skips multi-term clause shapes

Wildcard/prefix/fuzzy/regexp clauses expand to terms elsewhere and still score
from per-segment counters. Already recorded in `docs/sweep/findings.md`;
unchanged by this batch.

### Verdict

Swept clean for the fan-out and merge. Open: F-8's per-slice collector and
intra-segment partitioning, F-10's multi-term clauses, and F-5's real fix in
`lib.rs`.

---

## `crates/lucene-search/src/docid_set.rs`

Java counterparts: `lucene/core/src/java/org/apache/lucene/search/ConjunctionDISI.java`,
`DisjunctionDISIApproximation.java`, `DocIdSetIterator.java`, `DocIdSet.java`,
`BitDocIdSet.java`, `NotDocIdSet.java`, and
`lucene/core/src/java/org/apache/lucene/util/RoaringDocIdSet.java`.

| Rust | Java | Verdict |
|---|---|---|
| `Conjunction` | `ConjunctionDISI.doNext` | equivalent minus `TwoPhaseIterator` |
| `Disjunction` | `DisjunctionDISIApproximation` | equivalent result, `O(n)` vs `O(log n)` per step (pre-existing, documented) |
| `Excluding` | `ReqExclScorer` / `Occur.MUST_NOT` filtering | equivalent |
| `RoaringDocIdSet`, `RoaringBuilder`, `Block`, `RoaringIter` | `RoaringDocIdSet` + `Builder` + its five private `DocIdSet` block classes | **new (F-7)** |
| `CachedDocIdSet`, `CachedDocIdSetIter` | `LRUQueryCache.cacheImpl`'s bitset-vs-Roaring choice + `BitDocIdSet` | **new (F-7)** |

Java classes with **no** Rust counterpart: `DocIdSetBuilder` (the
buffer-of-doc-ids that upgrades to a `FixedBitSet` at a threshold) — recorded,
not ported: nothing in this port builds a doc-ID set that way, since every
producer here already yields an ascending iterator, and `RoaringBuilder` covers
the only consumer that materializes one. `DocIdSetIterator.advance`/`cost`,
`BitSetIterator`, `SparseFixedBitSet` likewise have no caller.

### F-7 `[MISSING]` `RoaringDocIdSet` and the cache's representation choice

`LRUQueryCache.cacheImpl` picks a `BitDocIdSet` when
`scorer.cost() * 100 >= maxDoc` (≥1% density, where random access pays and is
what makes a cached filter usable as a conjunction lead) and a
`RoaringDocIdSet` otherwise. `query_cache.rs` had neither: it stored a raw
`FixedBitSet` sized to `maxDoc`.

That is `maxDoc / 8` bytes per cached `(segment, query)` pair **regardless of
how few documents matched** — 1.25 MB for a 100-hit query on a 10M-document
segment — and it made iterating a cache hit an `O(maxDoc)` bit-by-bit scan.

**Fixed**: ported `RoaringDocIdSet` with all five of Java's block encodings and
their exact thresholds — all-set, contiguous range, `u16` array up to
`MAX_ARRAY_LENGTH` (4096), the *inverse* `u16` array for a full block missing
fewer than 4096 documents, and a block-local bitset otherwise — plus the
builder's buffer-then-upgrade strategy (documents accumulate in a 4096-slot
`u16` buffer and only spill into a bitset when it overflows, so a sparse block
never allocates one). `CachedDocIdSet` is the cache's stored type and applies
Java's 1% rule.

**Measured**: 100 documents in a 10 000 000-document space cost **4 928 bytes**
as Roaring against **1 250 000** as a `FixedBitSet` — 254x. Per-block overhead
is `size_of::<Block>() == 32` bytes against Java's 8-byte `DocIdSet[]` slot
(boxing the dense variant would halve it, at the cost of an indirection on the
dense path; not done, and negligible against the 254x).

**Tests**: every block encoding round-trips and `contains` agrees with iteration
for every document in range (including the documents *between* matches, where a
block-boundary off-by-one shows); the encoding the builder *chooses* is pinned
per density at each threshold boundary; a trailing partial block cannot take the
two full-block encodings; out-of-order adds panic as Java throws; RAM accounting
covers every encoding; block equality distinguishes encodings and contents; and
both cached representations agree on contents across five densities.

### F-29 `[CORRECTNESS]` `contains` could index past a trailing partial block

`RoaringDocIdSet::contains` guarded a negative doc and a missing block, but a
trailing partial block's bitset is shorter than 65536, so a document past
`max_doc` *inside the last block* reached `FixedBitSet::get` — which indexes its
word array behind a `debug_assert!`. That is a panic in test builds and an
out-of-bounds read in release. `CachedDocIdSet::Bits` had the same hole. The
existing test only probed a doc whose *block* was missing, which takes the safe
arm.
**Fixed**: both arms check the bitset's own length first. **Test**: extended
`cached_bitset_edge_cases` with a dense trailing block probed past `max_doc`,
and a `Bits` set probed at and past its length.

### F-30 `[PERF]` `from_bitset`'s sparse conversion was the scan this file calls a bug

`CachedDocIdSet::from_bitset` walked `for doc in 0..max_doc { bits.get(doc) }` to
feed the Roaring builder — 10M bounds-checked loads to emit 100 documents — while
`CachedDocIdSetIter::Bits`' own doc comment two hundred lines below calls that
"the `O(maxDoc)` bit-by-bit scan a naive `for doc in 0..len` would be".
**Fixed**: a `next_set_bit` word-at-a-time helper, now the single implementation
behind all three consumers (the conversion, the cached-bitset iterator, and the
dense-block iterator, which had a third copy of the same loop). It lives here
rather than on `FixedBitSet` because `lucene-util` is another batch's file;
recorded as a follow-up.

Also corrected a doc overclaim caught in the same pass: `cacheImpl`'s threshold
is reproduced at Java's value, but Java consults the scorer's `cost()` *estimate*
before materializing anything, where this port takes the exact cardinality of an
already-built bitset. The stored representation matches; the transient
allocation does not, and the comment now says so instead of "reproduced
exactly".

### Verdict

Swept clean. Open: `DocIdSetBuilder` (no caller), the disjunction min-heap
(pre-existing, documented), and moving `next_set_bit` onto `FixedBitSet` where
it belongs.

---

## `crates/lucene-search/src/query_cache.rs`

Java counterparts: `lucene/core/src/java/org/apache/lucene/search/LRUQueryCache.java`
(including `LRUQueryCachePartition`, `CachingWrapperWeight`, `CacheAndCount`,
`MinSegmentSizePredicate`), `QueryCache.java`, `QueryCachingPolicy.java`,
`UsageTrackingQueryCachingPolicy.java`,
`lucene/core/src/java/org/apache/lucene/util/FrequencyTrackingRingBuffer.java`.

| Rust | Java | Verdict |
|---|---|---|
| `QueryCache::new` / `with_ram_limit` | `LRUQueryCache(maxSize, maxRamBytesUsed)` | **was count-only (F-13), fixed** |
| `QueryCache::get_or_compute` | `CachingWrapperWeight.scorerSupplier`'s cache lookup + `tryPopulateCache` | **was copying on hit (F-14) and poisoning on error (F-16), fixed** |
| `QueryCache::get_or_compute_with_policy` | `CachingWrapperWeight` consulting `policy.onUse`/`shouldCache` | **new (F-15)** |
| `QueryCache::insert` | `put` + `evictIfNecessary` | **was evict-then-insert, now Java's order (F-13)** |
| `QueryCache::requires_eviction` | `LRUQueryCachePartition.requiresEviction` | equivalent |
| `QueryCache::evict_lru` | `evictIfNecessary`'s `uniqueCacheKeys` iterator | **was `O(n)` (F-17), fixed** |
| `QueryCache::{len,is_empty,ram_bytes_used,hit_count,miss_count,cache_count,eviction_count}` | `getCacheSize`, `ramBytesUsed`, `getHitCount`, `getMissCount`, `getCacheCount`, `getEvictionCount` | equivalent |
| `QueryCache::{invalidate_segment,clear,remove}` | `clearCoreCacheKey`, `clear`, `clearQuery` | equivalent (manual, see F-18) |
| `QueryCachingPolicy`, `UsageTrackingPolicy`, `CachingCost`, `FrequencyRingBuffer` | `QueryCachingPolicy`, `UsageTrackingQueryCachingPolicy`, its three static shape predicates, `FrequencyTrackingRingBuffer` | **new (F-15)** |
| `leaf_is_worth_caching`, `DEFAULT_MIN_LEAF_SIZE` | `LRUQueryCache.MinSegmentSizePredicate(10000)` | **new (F-15)** |
| `search_term_query_cached` | no Java counterpart (`IndexSearcher` wires the cache into `Weight`, not into a per-query function) | Rust-only entry point |

Java methods with **no** Rust counterpart: the 16-way partitioning and
`ReentrantReadWriteLock` (F-19), `skipCacheFactor` (this port has no `cost()`
estimate to compare against a lead cost), the `onQueryCache`/`onQueryEviction`/
`onDocIdSetCache`/`onCacheEntryInserted`/`onCacheEntryEvicted` callback surface,
`assertConsistent`, `CachingWrapperWeight.count`'s "cache has no accurate count
under deletions" branch, `getChildResources`.

### F-13 `[MISSING]` no `maxRamBytesUsed`

Java bounds itself by entry count **and** by the summed `ramBytesUsed` of every
cached `DocIdSet`, evicting least-recently-used until neither bound is exceeded
(`requiresEviction()` is `size > maxSize || ramBytesUsed > maxRamBytesUsed`).
We bounded by count alone — so `max_entries` of 1000 over a 10M-document index
was an unannounced 1.25 GB ceiling.

**Fixed**: `with_ram_limit(max_entries, max_ram_bytes)`, exact per-entry
accounting charged on insert and returned on every removal path, and Java's
insert-then-`evictIfNecessary` order (so an entry larger than the whole budget
is inserted and immediately evicted, as Java's is, rather than bypassing the
accounting). `new(max_entries)` is `LRUQueryCache(maxSize, Long.MAX_VALUE)`.
**Tests**: the RAM bound evicts while the count bound is far away; accounting
returns to exactly zero through eviction, targeted removal *and* per-segment
invalidation; an entry bigger than the whole budget does not stay.

### F-14 `[PERF]` a cache hit deep-copied the whole bitset

`get_or_compute` returned `entry.bits.clone()`. A hit therefore memcpy'd
`maxDoc / 8` bytes — 1.25 MB on a 10M-document segment — which is more work than
re-running many queries would have been, and inverted the entire point of the
cache. Java hands back a reference to the shared immutable `DocIdSet`.
**Fixed**: entries are `Arc<CachedDocIdSet>` and a hit is a refcount bump.
Combined with F-7, iterating a hit is now `O(matches)` for a sparse set and
`O(maxDoc/64)` word-at-a-time for a dense one, instead of an `O(maxDoc)`
bit-by-bit scan.
**Test**: `a_hit_shares_the_cached_set_rather_than_copying_it` asserts
`Arc::ptr_eq` across two hits.

### F-15 `[MISSING]` no caching policy

Java never caches unconditionally. `UsageTrackingQueryCachingPolicy` tracks the
hashes of the last 256 used queries in a `FrequencyTrackingRingBuffer` and only
caches once a query's frequency in that window reaches `minFrequencyToCache` —
2 for costly queries (multi-term, term-in-set, point), 4 for composite ones
(`BooleanQuery`/`DisjunctionMaxQuery`, cached a step earlier so "A OR B" wins
over caching A and B separately), 5 otherwise — and `shouldNeverCache` refuses
`TermQuery`, `FieldExistsQuery`, `MatchAllDocsQuery`, `MatchNoDocsQuery` and
empty boolean/dismax queries outright. `LRUQueryCache` separately refuses leaves
under 10 000 documents or under half the reader's average leaf size.

**Fixed**: `QueryCachingPolicy` trait, `UsageTrackingPolicy` with the ring
buffer and all three thresholds, a `CachingCost` trait by which a query type
declares its own shape (implemented for `TermQuery` as `never_cache`, matching
Java's first case verbatim), `leaf_is_worth_caching` for
`MinSegmentSizePredicate`, and `get_or_compute_with_policy` to apply it.
**Tests**: each of the three thresholds pinned at its exact boundary; a
`TermQuery` never cached however often used and not even tracked; the history
window forgetting a displaced query; the gate actually withholding the *store*
until the threshold, then serving hits; and `MinSegmentSizePredicate`'s two
conditions including the `maxDoc * 2 > average` boundary.

Note the tension this exposes, now documented in the module: the one wired
entry point, `search_term_query_cached`, caches exactly the query type Java
refuses to cache. It exists because `TermQuery` is the only query in this port
whose representation satisfies `Eq + Hash`; a caller wanting Java's judgement
uses `get_or_compute_with_policy` and gets the uncached path.

### F-16 `[CORRECTNESS]` a failed computation briefly inserted a poisoned entry

`compute` had to return a bare `FixedBitSet`, so `search_term_query_cached`
captured the error out of band, handed back an empty placeholder that
`get_or_compute` dutifully inserted, and then removed it again. Between the two
there was a cached "matches nothing" entry for a key whose real answer was
unknown; any path that returned early, panicked, or looked the key up in between
would have served it.
**Fixed**: `get_or_compute` is generic over the error and takes
`impl FnOnce() -> Result<FixedBitSet, E>`, so a failure propagates and nothing
is ever inserted. The `RefCell` dance and the compensating `remove` are gone.
**Test**: `a_failed_computation_stores_nothing` asserts the cache is untouched
and the next attempt is a genuine miss.

### F-17 `[PERF]` LRU selection was a linear scan

`evict_lru` was `entries.iter().min_by_key(last_used)` — `O(n)` per eviction,
where Java's access-ordered `LinkedHashMap` is `O(1)`. Harmless while the only
bound was entry count (one eviction per insert); quadratic once the RAM bound
(F-13) can evict many entries in a single insert.
**Fixed**: a `BTreeMap<u64, CacheKey>` recency index kept in lockstep with the
entry map, so eviction is `O(log n)`. Covered by the existing LRU-order tests
plus the new RAM-bound eviction test.

### F-27 `[CORRECTNESS]` `on_use` was called per leaf, not per query execution

`CachingWrapperWeight.scorerSupplier` guards its own call:
`if (used.compareAndSet(false, true)) policy.onUse(getQuery())` — one `onUse`
per `Weight`, across every leaf that `Weight` is asked about. F-15's first cut
called it inside `get_or_compute_with_policy`, i.e. once per `(segment, query)`.
On an N-leaf reader that bumps a query's tracked frequency N times per
execution, so the ported thresholds are crossed after ~5/N executions instead of
5 — the policy would cache almost everything on a many-segment index, which is
the opposite of what it exists for. Worse, the module doc claimed the port *was*
`UsageTrackingQueryCachingPolicy` and listed no such divergence.

**Fixed**: `get_or_compute_with_policy` no longer calls `on_use`;
`QueryCachingPolicy::on_use` is the caller's, once per query execution, and both
its doc comment and the method's say so. **Test**:
`searching_many_leaves_does_not_inflate_a_querys_tracked_frequency` runs one
execution across 20 leaves and asserts the tracked frequency is 1 and nothing
was cached.

### F-28 `[CORRECTNESS]` deletions were baked into the cached set

`search_term_query_cached` passed `live_docs` into the `compute` closure, so the
stored set had deletions already applied — and a *hit* never looks at
`live_docs` at all. The same `(segment, query)` key under a different `live_docs`
therefore returned the first call's doc set, silently.

Java deliberately caches deletion-agnostic sets: both `cacheIntoBitSet` and
`cacheIntoRoaringDocIdSet` score with `acceptDocs == null`, and the cache keys
on the segment's *core* cache helper, precisely so an entry survives a new
`.liv` generation.

**Fixed**: the entry is computed with `live_docs == None` and `live_docs` is
applied while iterating it — which also makes the entry reusable across delete
generations instead of stale. **Test**:
`a_cached_entry_is_deletion_agnostic_and_live_docs_apply_on_the_way_out`
populates with no deletions, then re-queries the same segment key with one hit
deleted and `.doc` input withheld (so it can only be served from the cache) and
asserts the deleted document is filtered out and the cache still holds one entry.

### F-18 `[INTENTIONAL]` no automatic invalidation

Java keys on `IndexReader.CacheHelper` identity and registers a close listener,
so closing a segment drops its entries. This port has no such lifecycle object;
`invalidate_segment` is called by hand. Pre-existing and already documented;
restated here because F-15's arrival makes the cache look more complete than its
invalidation story is.

### F-19 `[INTENTIONAL]` `&mut`-exclusive instead of lock-partitioned

Java's cache is one shared object behind a `ReentrantReadWriteLock`, split into
16 partitions to cut contention. Ours is `&mut`-exclusive: the borrow checker
makes the data race impossible rather than a lock making it unlikely
(`rust-performance`'s "ownership over locks"), and a per-thread cache needs no
synchronization at all. The cost is that a caller wanting one cache shared
across rayon leaf threads must wrap it. Recorded.

### Verdict

Swept clean. Open: F-18's lifecycle hook (needs a segment-identity object this
port does not have), F-19's shared-cache ergonomics, `skipCacheFactor` (needs
iterator `cost()` estimates), and the observability callbacks.

---

## `crates/lucene-search/src/field_norms.rs`

Java counterparts:
`lucene/core/src/java/org/apache/lucene/codecs/lucene90/Lucene90NormsProducer.java`
(read path), `search/similarities/BM25Similarity.java` (`avgFieldLength`, the
`cache[]` norm-inverse table), `index/NumericDocValues`.

| Rust | Java | Verdict |
|---|---|---|
| `FieldNorms::from_field_stats` | `BM25Similarity.scorer` + `CollectionStatistics` (`sumTotalTermFreq / docCount`) | equivalent |
| `FieldNorms::open` | no Java counterpart (Java never derives `avgdl` from norms) | divergent by construction (F-21) |
| `FieldNorms::norm_inverse` | `BM25Scorer.cache[norm]` | equivalent |
| `FieldNorms::field_length` | `SmallFloat.byte4ToInt(norm)` | equivalent |
| `FieldNorms::sparse_norm`, `sparse_doc_ids`, `dense_norm_bytes` | `Lucene90NormsProducer.DenseNormsIterator`/`SparseNormsIterator`/`getNorms` | equivalent shape, caller-side caching (documented, pre-existing) |
| `norm_inverse_table`, `norm_length_table` | `BM25Similarity`'s `cache[]` build | equivalent |

Re-verified against b6's norms work: all five `numBytesPerValue` widths
(0/1/2/4/8) are handled — the fast path claims only the dense one-byte shape and
`fast_path_declines_shapes_it_cannot_serve` asserts it declines the others,
which then take `norms::norm_value`'s general decode. Sparse fields and
multi-field `.nvm` entries decode through `lucene_codecs::norms` unchanged.

### F-20 `[CORRECTNESS]` `dense_rank_power: 0` in test metadata

Flagged by b6. `0` is not a legal `IndexedDISI` `denseRankPower`: the only valid
values are 7..=15 and `0xFF` (Java's `-1`, "no rank table"), and
`dense_rank_bytes` rejects everything else. Two of this file's test fixtures
described metadata no Lucene writer can produce.

It happened to be invisible: dense norms entries never consult the field, and
the one sparse fixture had four documents, which `indexed_disi::write` emits as
a SPARSE block — and only DENSE blocks read the rank table. That is exactly the
kind of almost-right test input that hides a real decode bug.

**Fixed**: both fixtures use `0xFF`, named as a local `NO_RANK` constant with
the explanation. **Tests**: a new fixture crosses the 4096-document threshold so
`indexed_disi::write` emits a real DENSE block, and asserts the fast and general
paths agree across it; and `an_illegal_dense_rank_power_is_rejected_rather_than_guessed`
asserts that an illegal value makes the once-decoded fast path decline *and*
surfaces an error per lookup rather than inventing a norm.

### F-21 `[MISSING]` `FieldNorms::open`'s `avgFieldLength` is not Java's

The ledger's carry-over from b12. `open` averages *decoded* norms, which are
`SmallFloat`-quantized and lossy above length 24, where Java divides two exact
counters. Unfixable from `.nvd`/`.nvm` alone — `from_field_stats` exists
precisely because it reads the counters instead.

**Addressed on this file's side**: `open`'s doc comment now enumerates the three
divergences in order of how much each moves a score (quantization; population,
since Java's `docCount` counts deleted documents too and `live_docs` here does
not; the empty-field divide-by-zero), states its `O(maxDoc)` cost against
`from_field_stats`' two integer reads, and points every caller at
`from_field_stats`. The population divergence is currently *latent*: every
caller in the workspace passes `live_docs == None`.

**The call sites remain b14's (`explain.rs`) and b15's (`lucene-ffi/src/query.rs`).**
`SegmentReader::field_norms()` (F-3) now gives both of them a
`from_field_stats`-built `FieldNorms` straight off the reader, which is what
that carry-over needs to close.

### Verdict

Swept clean for the decode. F-21's call sites are open and owned by b14/b15.

---

## `crates/lucene-search/src/soft_deletes.rs`

Java counterparts: `lucene/core/src/java/org/apache/lucene/index/PendingSoftDeletes.java`
(`applySoftDeletes`, `onDocValuesUpdate`, `onNewReader`), `PendingDeletes.java`,
`SoftDeletesDirectoryReaderWrapper.java`,
`search/FieldExistsQuery.getDocValuesDocIdSetIterator`,
`index/DocValuesFieldUpdates.java` (`Iterator.hasValue`, `reset`).

| Rust | Java | Verdict |
|---|---|---|
| `is_soft_deleted` | `FieldExistsQuery`'s presence test on the soft-deletes field | equivalent |
| `is_live` | `PendingDeletes.getLiveDocs` combined with the soft-delete bit | equivalent |
| `effective_live_docs` | `PendingSoftDeletes.onNewReader` → `applySoftDeletes(iterator, getMutableBits())` | **was per-document (F-22), fixed** |
| `is_soft_deleted_with_overlay` | `applySoftDeletes`' `hasValue()` branch pair | equivalent for presence; divergent for the reset-on-hard-deleted case (F-23) |
| `effective_live_docs_with_overlay` | `PendingSoftDeletes.onDocValuesUpdate` | as above |
| `mark_soft_deleted_via_overlay` | `IndexWriter.softUpdateDocument`'s `NumericDocValuesFieldUpdates` (single generation) | scoped, pre-existing |
| `present_docs`, `hard_live_bits`, `clear_present` | `FieldExistsQuery.getDocValuesDocIdSetIterator` + `applySoftDeletes`' loop | **new (F-22)** |

Java methods with **no** Rust counterpart: `PendingSoftDeletes.delete` (write
side), `writeLiveDocs`, `dropChanges`, `numDeletesToMerge`/`isFullyDeleted`/
`ensureInitialized` (all `IndexWriter`-side), the `dvGeneration` bookkeeping and
its "we have seen this generation already" assertion (this port has no
`docValuesGen` wiring yet — see b6's scope note), and
`SoftDeletesDirectoryReaderWrapper` itself (this is the per-segment rule, not
the reader wrapper).

Built on b6's `Option<i64>` change: an overlay `Some(None)` is
`DocValuesFieldUpdates.reset`, a value *removal*, and both overlay-aware
functions treat it as "not soft-deleted, shadowing the base" — Java's
`hasValue() == false` branch.

### F-22 `[PERF]` live-docs construction asked per document

`applySoftDeletes` drains the field-exists iterator **once**, clearing a bit per
present document. Both `effective_live_docs` functions instead looped
`0..max_doc` calling `is_soft_deleted`, and `doc_values::numeric_value`'s sparse
branch builds a fresh `DisiCursor` and walks block headers from the start of the
region on every call. Building a live-docs bitset was therefore
`O(maxDoc x blocks)` where Java pays `O(maxDoc + cardinality)` — the same shape
of defect `field_norms.rs` already fixed at its own caller.

**Fixed**: `present_docs` resolves the presence encoding once (empty / dense /
one `IndexedDISI` decode) and `clear_present` clears in a single pass, with the
overlay shadowing applied by skipping documents the overlay speaks for. The
public per-document functions are unchanged for single-lookup callers.
**Test**: `one_pass_live_docs_agree_with_the_per_document_check` cross-checks the
bulk build against the per-document reference over the real sparse doc-values
fixture, across five hard-delete patterns x five overlay shapes including
`reset` entries — the two are implementations of one rule, so the only thing
worth asserting about the fast one is that it is not also different.

### F-23 `[INTENTIONAL]` a `reset` never resurrects a hard-deleted document

`PendingSoftDeletes.applySoftDeletes`' value-removal branch is
`bits.getAndSet(docID)` against the *combined* live-docs bitset, so in Java a
`reset` on a hard-deleted document sets its bit back and resurrects it.
`IndexWriter` never emits that pair, so it is unreachable in practice; this port
keeps the stronger invariant instead — a `.liv`-deleted document stays deleted
whatever the soft-deletes field says.
Deliberate, and now pinned rather than incidental:
`an_overlay_reset_never_resurrects_a_hard_deleted_doc`. The *reachable* half of
the same branch — a reset un-deleting a document the base marked soft-deleted —
does match Java and is pinned by
`an_overlay_reset_undeletes_a_doc_the_base_marked_soft_deleted`.

### F-24 `[MISSING]` no coverage for the dense and empty presence shapes

`isFullyDeleted`'s degenerate case (a *dense* soft-deletes field has a value for
every document, so every document is soft-deleted) and the empty-field case
(`docsWithFieldOffset == -2`) were both unexercised; the sparse fixture cannot
express either.
**Fixed** by construction in F-22 (`present_docs` handles all three shapes
explicitly) and covered by `a_dense_soft_deletes_field_marks_every_doc`,
`an_empty_soft_deletes_field_marks_nothing`,
`overlay_entries_outside_the_doc_range_are_ignored` (an overlay record for a
document outside `0..max_doc` must not index past the bitset) and
`a_presence_region_past_the_end_of_the_data_is_an_error` (a corrupt
offset/length pair is a typed error, not a panic or a silently empty result).

### F-31 `[CORRECTNESS]` corrupt presence metadata became "delete every document"

F-22's first cut mapped a negative `docsWithFieldOffset`/`docsWithFieldLength`
to `Eof { offset: 0 }` — an error describing something that did not happen — and,
worse, wrote `usize::try_from(count).unwrap_or(max_doc)` for a dense entry's
`numValues`, turning a corrupt negative count into *every document is
soft-deleted*: a silently empty segment from one bad byte.
**Fixed**: `PresentDocs::Every` carries a `usize` validated at construction, and
every conversion failure and out-of-range region is a
`lucene_store::Error::Corrupted` naming the field and the offending value.
**Test**: `a_negative_dense_doc_count_is_an_error_not_a_fully_deleted_segment`,
alongside the existing `a_presence_region_past_the_end_of_the_data_is_an_error`.

### Verdict

Swept clean for the read side. The write side (`PendingSoftDeletes`'
`IndexWriter` half and `docValuesGen` wiring) remains out of scope, as b6
recorded.

---

## Handed to other batches

- **b12 (`lib.rs`)**: `search_term_query_scored_maxscore_with_stats` drops its
  `global` argument on all three fallback `return`s, calling the no-stats
  `search_term_query_scored`. See F-5 for the failing scenario and the numbers.
  Worked around at this batch's caller; the fix belongs in `lib.rs`.
- **b14 (`explain.rs`) / b15 (`lucene-ffi/src/query.rs`)**: the ledger's
  `avgFieldLength` carry-over. `SegmentReader::field_norms()` (F-3) now provides
  the `from_field_stats`-built `FieldNorms` those call sites need.

## Gate

`cargo fmt --all`, `cargo clippy -p lucene-search --all-targets -- -D warnings`,
`cargo test -p lucene-search` green (842 tests). Per-file line coverage after
this batch: `directory_reader.rs` 99.09%, `docid_set.rs` 99.84%,
`field_norms.rs` 98.46%, `multi_segment.rs` 96.77%, `query_cache.rs` 96.36%,
`soft_deletes.rs` 96.91% — all above the 95% bar.

A workspace-wide `clippy`/`llvm-cov` could not be run to completion: `lib.rs`,
`highlighter.rs`, `query_parser.rs` and `lucene-ffi/src/query.rs` were mid-edit
by b12/b14/b15 throughout, and their errors are not this batch's.
