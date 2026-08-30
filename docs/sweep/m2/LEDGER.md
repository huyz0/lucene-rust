# M2 sweep ledger

Every ported source file, its batch, and its sweep status.
Status: `pending` → `running` → `swept` (report written, gates green).
Checkbox legend: `- [ ]` open, `- [x]` done, `- [~]` obsolete.

| Batch | Files | Status |
|---|---|---|
| b1-util-store | lucene-util/{lib,base36,fixed_bit_set,small_float,term_interner,zigzag}, lucene-store/{lib,codec_util,data_input,data_output,directory,error,index_output} | swept (22 findings: 3 CORRECTNESS, 7 MISSING, 3 PERF fixed) |
| b2-packed | codecs/{for_util,packed_ints,block_packed,direct_reader,direct_monotonic,indexed_disi} | swept (18 findings: 9 CORRECTNESS, 5 MISSING, 3 PERF) |
| b3-compression | codecs/{lz4,deflate,compound_format,stored_fields,lib} | swept (21 findings: 4 CORRECTNESS, 4 MISSING, 7 PERF) |
| b4-fst-blocktree | codecs/{fst,blocktree,terms_dict} | swept (21 findings: 6 CORRECTNESS all fixed, 6 MISSING (1 fixed), 4 PERF (1 fixed), 5 INTENTIONAL) |
| b5-postings | codecs/{postings,postings_writer} | swept (12 findings: 4 CORRECTNESS all fixed, 0 MISSING, 3 PERF (1 fixed+benchmarked), 5 INTENTIONAL) |
| b6-docvalues | codecs/{doc_values,doc_values_updates,norms,live_docs} | swept (20 findings: 5 CORRECTNESS, 8 MISSING, 2 PERF) |
| b7-points-fields | codecs/{points,field_infos,term_vectors,vectors} | swept (28 findings: 3 CORRECTNESS all fixed, 7 MISSING (6 fixed; HNSW recorded), 10 PERF (2 fixed with measurements), 8 INTENTIONAL) |
| b8-automata-analysis | codecs/{regexp,wildcard,fuzzy,suggest}, lucene-analysis/lib | swept (34 findings: 19 CORRECTNESS all fixed, 6 MISSING (4 fixed), 4 PERF (2 fixed+benchmarked), 5 INTENTIONAL) |
| b9-index-write | index/{index_writer,segment_writer,indexing_chain,update_document,lib} | swept (20 findings: 6 CORRECTNESS (5 fixed), 6 MISSING (1 fixed), 5 PERF (2 fixed+benchmarked, 2.5x), 3 INTENTIONAL) |
| b10-merge | index/{merge,merge_policy} | swept (38 findings: 9 CORRECTNESS all fixed, 12 MISSING (9 fixed, 3 recorded with blockers), 8 PERF (2 fixed, 3 carried over), 9 INTENTIONAL) -- includes a new 33-scenario `GenMergePolicy` Java fixture; all 33 match |
| b11-index-meta | index/{segment_info,segment_infos,check_index,checksum_verify,deletes,term_delete,points_delete} | swept (26 findings: 5 CORRECTNESS all fixed, 12 MISSING (11 fixed, 1 recorded), 4 PERF (2 fixed+measured, 220x on points delete), 5 INTENTIONAL) -- the `.si` index-sort encoding was this port's own invention and is now the real `SortFieldProvider` layout, proven byte-identical against a new real-Lucene `fixtures/data/_2.si` and through `VerifySegmentInfo`; `check_index` gained `testStoredFields`' real document scan, `testTermVectors`, `testDocValues`, `testSort`, `checkSoftDeletes`, `testPostings`' field summaries and every `SegmentInfos.readCommit` cross-`.si` validation |
| b12-search-core | search/{lib,query,collector,similarity} | swept (28 findings: 5 CORRECTNESS (4 fixed, 1 handed to b13/b15), 16 MISSING (5 fixed), 4 PERF (2 fixed+benchmarked, 5.0-6.8x), 3 INTENTIONAL); Tier-2 review run, all four gating items resolved |
| b13-search-readers | search/{multi_segment,directory_reader,docid_set,field_norms,query_cache,soft_deletes} | swept (29 findings: 8 CORRECTNESS all fixed, 9 MISSING (7 fixed, 2 recorded), 8 PERF (6 fixed, 2 recorded with measurements), 4 INTENTIONAL) -- ported `RoaringDocIdSet` and `LRUQueryCache`'s bitset-vs-Roaring rule (254x smaller for a 100-hit query on a 10M-doc segment), `maxRamBytesUsed`, `UsageTrackingQueryCachingPolicy` and `MinSegmentSizePredicate`; `SegmentReader` now shares its core across reopens by `Arc` (Java's `incRef`) and opens `.nvm`/`.nvd`, which real norms had never been reachable through |
| b14-search-features | search/{doc_value_query,points_query,facets,highlighter,term_vectors_query,explain,query_parser} | swept (24 findings: 9 CORRECTNESS all fixed, 8 MISSING (all fixed; 6 further gaps recorded with blockers), 2 PERF (1 was already fixed upstream by b7 -- 577x -- with only stale docs left, 1 recorded), 5 INTENTIONAL) -- every `Explanation` description string is now real Lucene's verbatim (`weight(f:t in D) [BM25Similarity]`, `boost * idf * tf`, `N`/`n`, `dl`/`avgdl`, `sum of:`, `max of:`) plus `Explanation.toString()`; the query parser gained real `AND`/`OR`/`NOT`/`&&`/`||`/`!` operators via a port of `QueryParserBase.addClause`, `setDefaultOperator`, and phrase slop `"a b"~n`; the highlighter gained `PassageScorer` (so `max_fragments` keeps the *best* passages, not the first) and the rest of `DefaultPassageFormatter` (ellipsis, HTML escaping, overlapping-match coalescing); facets gained `getTopChildren`/`getAllChildren`'s `FacetResult` semantics (zero-count children dropped) and `LongRange`/`DoubleRange` with `NumericUtils.doubleToSortableLong`; `SortedNumericDocValuesRangeQuery`'s real any-value-in-range semantics were absent (a selector reduction was standing in for them); `PointInSetQuery` ported |
| b15-ffi-core | lucene-ffi/*, lucene-core/lib | swept (22 findings: 12 CORRECTNESS all fixed, 5 MISSING all fixed, 3 PERF (1 fixed + A/B measured 6.2x four-thread throughput, 2 recorded), 4 INTENTIONAL) -- `.liv` deletions now honoured by every single-segment query/explain/range-sort path (new `ffi_segment_set_live_docs`, proven against the real `live_docs_index` fixture); every non-`Ok` status now leaves its own error message instead of a stale one; no `Vec::with_capacity` on a caller-supplied length can abort the JVM; handle-index overflow refused instead of aliasing a live handle; the registries are `RwLock` so concurrent JVM search threads stop serializing; and b12's two b15-owned carry-overs are closed (`avgdl` now `sumTotalTermFreq/docCount`, `maxClauseCount` enforced per *query* at the boundary). Tier-2 review run on the diff: four gating findings, all this batch's own regressions (`deletesPctAllowed`'s range is `(0,50]` not `0..=100`; the clause cap was per-list not per-query; a deleted doc explained as `no matching term` instead of Java's `Document N is deleted`; the `unsafe impl Send/Sync for WriterHandle` safety comment was invalidated by the `RwLock` change), all fixed with tests |

## Follow-up batches (opened from the carry-overs below)

| Batch | Scope | Status |
|---|---|---|
| c1-lazy-blocktree | A1: lazy frame-based `SegmentTermsEnum`, 155x segment-open gap | swept (19 findings: 6 CORRECTNESS all fixed, 5 MISSING all fixed, 4 PERF (3 fixed+measured, 1 recorded), 4 INTENTIONAL) -- `SegmentTermsEnum`/`SegmentTermsEnumFrame`/`TrieReader.lookupChild` ported in full, `open` no longer touches a `.tim` block: **35.4 ms -> 0.175 ms (202x)** and **+39.0 MB -> +4.7 MB** per segment, with the hot seek loop measured against real Lucene on the same 2000 terms in the same order, interleaved A/B (a cold hit 495 ns vs Lucene's 440 ns, 1.13x; a miss 215 ns vs 263 ns, i.e. cheaper than Lucene's); `regexp_intersect`'s dead-prefix skip now skips *loading* the blocks too (131x/1768x over a real-Lucene 1M-term dictionary, up from b8's 88x/1065x); two new exhaustive `seekCeil` differentials (~30k targets) over the depth-6 nested-block and floor-split real-Lucene fixtures |
| c2-sparse-lookup | `IndexedDISI` incremental cursor + `NumericReader`'s O(cardinality) `Vec` | swept (13 findings: 3 CORRECTNESS all fixed, 3 MISSING (2 fixed, jump table recorded), 4 PERF (2 fixed+measured: ~56-450x forward, ~95-1200x single lookup, 2.0 MB -> 0 B; 1 recorded, 1 handed to b13), 3 INTENTIONAL) -- `DisiCursor` is now a field-for-field port of Java's `IndexedDISI` including `rankSkip`; `createRank` ported and the sparse doc-values/norms writers now emit Lucene's default rank table, proven by three new strided passes in `VerifySparseNumericDocValues` with a negative control |
| c3-writer-lifecycle | `IndexFileDeleter` (orphan-file leak) + RAM accounting / flush trigger | swept (9 findings: 2 CORRECTNESS both fixed, 4 MISSING all fixed, 2 PERF both fixed+measured, 1 INTENTIONAL) -- `index_file_deleter.rs` ports `IndexFileDeleter`/`FileDeleter`/`KeepOnlyLastCommitDeletionPolicy`, so superseded commits, superseded `.liv` generations, merge sources, rolled-back prepares, failed flushes and a crashed session's leftovers are all reclaimed; `ramBufferSizeMB`/`maxBufferedDocs` + an automatic flush cut writer peak RSS from 862 MB to 128 MB at 200k docs (6.7x, and O(n) -> O(1)) while *improving* throughput 29.0 -> 21.0 us/doc |
| c4-merge-fastpath | stored-fields/term-vectors bulk-copy merge, streaming postings merge, `BKDWriter.merge` | swept (23 findings: 4 CORRECTNESS all fixed, 3 MISSING all fixed, 11 PERF (8 fixed+measured, 3 recorded), 5 INTENTIONAL) -- all three of `Lucene90CompressingStoredFieldsWriter.merge`'s strategies ported behind a `MatchingReaders` port (**BULK 520x, DOC 26x, VISITOR 29x** on 4 x 20 000 documents, A/B against the pre-c4 algorithm re-run in-process); `merge_postings` is a real streaming k-way `TermsEnum` merge decoding postings from the cursor's own position (10.8x); `BKDWriter.merge`'s one-pass 1-D leaf plan, with the sortedness precondition *verified* rather than assumed (2.6x); a new `write_merged_segment_fixture`/`VerifyMergedSegment` case hands a merged segment to real Lucene and caught two defects nothing in-port could see -- a merged `.fnm` carrying no `PerFieldPostingsFormat` attributes (postings silently invisible, `CheckIndex` clean) and one promising norms that made the index unopenable; the bulk path now runs Java's `checkIntegrity` on every source, without which it would launder a corrupt source into a permanently valid segment. Term-vectors bulk merge is the one carry-over left, blocked on the term-vectors writer emitting one chunk per segment |
| c5-vectors | `Lucene99FlatVectorsFormat` interop (`.vec`/`.vemf`) + HNSW graph (`.vem`/`.vex`) | swept (21 findings: 5 CORRECTNESS all fixed, 6 MISSING all fixed, 5 PERF (3 fixed+measured, 2 recorded), 5 INTENTIONAL) -- the invented `.vec`/`.vem` layout is gone: `vectors.rs` is now a real port of `Lucene99FlatVectors{Format,Reader,Writer}` + `OrdToDocDISIReaderConfiguration` + `OffHeap{Float,Byte}VectorValues` (both encodings, all four similarities including Java's *different* byte transforms, `alignOutput` padding, the sparse `IndexedDISI` + `DirectMonotonicWriter` ord<->doc pair built on c2's cursor), and new `hnsw.rs`/`hnsw_vectors.rs` port the whole graph stack including a bit-exact `java.util.SplittableRandom`. **Evidence**: over a real Lucene 4000-vector graph this port's searcher reproduces `KnnFloatVectorQuery`/`KnnByteVectorQuery` doc-for-doc in order across 80 queries; a graph built here has the same levels, the same entry node (171; 46601 at 50k x 128) and **recall@10 0.9250 against Lucene's 0.9250**; and real Lucene searches a Rust-written graph at recall 0.91-1.00 (`scripts/verify-write-path.sh` 15/15 -> **16/16** with a new `VerifyVectors`). **Measured** 50k x 128 clustered, M=16/beam=100: 26.8k q/s vs 572 brute force (**47x**), 205 distance computations/query vs 50 000 (**244x fewer**), recall@10 0.914; graph build 5.5 s against real Lucene's 6.4 s on identical data. Tier-2 review run: no gating findings, six advisories, five acted on -- writer/reader parameter-bound symmetry, `VectorUtil.checkFinite` (whose absence made `f32::max`'s NaN-dropping reachable, unlike Java's `Math.max`), a per-seek `Vec` allocation on the hottest search loop, a dedicated error for builder-parameter mistakes, and -- the important one -- an **arc-for-arc comparison of the Rust-built graph against Lucene's**, which is **4273/4273 nodes identical** and, unlike the recall assertion, actually discriminates: mutating the diversity rule takes agreement to 1/4273 while recall *rises* to 0.9350. The sixth (moving `SplittableRandom`/`TernaryLongHeap`/`NumericUtils` helpers into `lucene-util`) is a carry-over |
| c6-search-followups | dropped `global` stats, `FieldNorms` cursor, reader-wide `avgdl`, `BooleanScorer` bulk-OR | swept (8 findings: 5 CORRECTNESS all fixed, 2 MISSING both fixed+measured, 1 PERF fixed+measured; 1 INTENTIONAL recorded) -- b13's F-5 fixed at the defect and its caller-side guard deleted (the tripwire is kept, inverted); `FieldNorms` gives each scan its own `FieldNormsCursor` (Lucene's per-scorer `NumericDocValues`) so the eager `Vec<i32>` is gone: **construct 140 us -> 493 ns at 100k and O(cardinality) -> O(1); a whole query's scan 1.74 ms -> 799 us; 400 KB -> 0 B per query per leaf**. Honest counterweight: the *isolated* random lookup is 14x **slower** (171 ns one-shot cursor vs 12 ns binary search), which the scan amortises away (a warm cursor step is 9.8 ns and flat, against a logarithmic search) and only `explain` pays. The first cut of the bench claimed 1,183x on that lookup by rebuilding the `Vec` inside `b.iter()` while the cursor arm reused a structure built outside -- caught by the batch's own Tier-2 review. A test proves the rayon fan-out is still genuinely concurrent; `avgdl` is reader-wide now (b13's F-26) and proven bit-for-bit against a **new genuinely two-segment real-Lucene fixture** (`GenMultiSegmentScoring.java`, per-leaf avgdl 1.75 and 40.0 vs reader-wide 20.875) with a negative control; the six multi-segment FFI entry points, which passed `None` for norms and so scored every hit unnormed, now pass real ones; `BooleanScorer`'s 4,096-doc window/bucket bulk OR ported as `docid_set::WindowedDisjunction` (**3.1x / 9.4x / 24.8x**, no regression when sparse), and b12's F-22 premise corrected -- 10.5.0 picks `MaxScoreBulkScorer`, not `BooleanScorer`, under `ScoreMode.TOP_SCORES`, so porting it onto the scored top-k path would have been a regression. The batch's own Tier-2 review then caught a **hang** in that new code (`i32` overflow computing the window end at `doc == i32::MAX` saturated so that no clause advanced and the loop re-derived the same window forever); fixed with `i64` window arithmetic and pinned by three top-of-doc-id-space shapes under a 30-second deadline, since an assertion cannot observe non-termination |
| c8-tv-chunking | term-vectors chunking + its bulk merge (c4's carry-over), `PostingsEnum` flags (b5's F6) | swept (20 findings: 4 CORRECTNESS all fixed, 9 MISSING all fixed, 6 PERF (4 fixed+measured, 2 assessed and recorded), 1 INTENTIONAL) -- `term_vectors::TermVectorsWriter` is now a full port of `Lucene90CompressingTermVectorsWriter`: both flush triggers (4096 bytes / 128 docs), all nine per-chunk header writers with *both* flags encodings, `startTerm` prefix compression, `flushOffsets`' derived `charsPerTerm`, sorted `flushFieldNums`, `blockShift = 10`, dirty-chunk accounting, `tooDirty`, `with_geometry`, and `copyChunks`; `merge.rs`'s `write_merged_term_vectors` is `Lucene90CompressingTermVectorsWriter.merge`'s two-way `MatchingReaders`-gated shape with c4's `checkIntegrity`-before-byte-copy fix. **Merge 289 292 ms -> 113.5 ms (2 548x) from chunking + a `ChunkCursor` alone, then -> 0.6 ms (469 076x) with the byte copy**, random-access `document()` **195x**, flush write **1.28x faster** and only 0.5% larger despite 160 chunks instead of 1. `PostingsFlags` + `for_util::pfor_skip` close b5's F6 (docs-only decode 1.07-1.32x, and the honest reason it is not more is that doc deltas dominate a block). Two corrupt-file panics in `postings.rs` turned into typed errors (c9's cross-batch finding), which also took c9's negative-control test from ~10 minutes to 0.36 s. Real Lucene reads a 400-document multi-chunk segment with offsets/payloads occurrence by occurrence, and a bulk-merged term-vector segment passes `CheckIndex`; `verify-write-path.sh` 18/18 |
| c7-delete-queue | `DocumentsWriterDeleteQueue` + sequence numbers, the four APIs it blocked, `BinaryDocValuesFieldUpdates`, `inflateGens`' per-segment half, `sci_id` | swept (28 findings: 9 CORRECTNESS (8 fixed, 1 recorded with a named blocker), 11 MISSING (10 fixed, 1 declined with reasoning), 4 PERF (1 fixed, 3 recorded with measurements/reasons), 4 INTENTIONAL) -- new `buffered_updates.rs` ports `DocumentsWriterDeleteQueue`/`BufferedUpdates`/`FrozenBufferedUpdates`/`DocValuesUpdate`/`BufferedUpdatesStream`; every mutating `IndexWriter` method now returns a `long` seqNo starting at 1, deletes buffer with a `docIDUpto` and resolve through a `delGen`-stamped packet, and the contract is pinned by tests that assert a **specific final visible document set** for an interleaved add/update/delete ordering and that a delete issued after a flush leaves the later segment's identically-termed documents alive (`del_count` 2 and 0 on the two segments) while one issued before still reaches the segment that flush produces. All four blocked APIs landed: `softUpdateDocument(s)` (proven to mark the original and *not* the replacement, with nothing hard-deleted), `updateDocValues`/`updateNumericDocValue`/`updateBinaryDocValue` (null value = `reset(doc)`, not zero), `deleteDocuments(Query)` over a closed `DeleteQuery` enum the crate can resolve without inverting the `index <- search` edge (incl. LUCENE-6379's `MatchAllDocsQuery` -> `deleteAll` specialisation), and `addDocuments`/`updateDocuments` block adds -- **real Lucene confirms `hasBlocks`** via a new `write_block_segment_fixture`/`VerifyBlockSegment` case asserting `LeafMetaData.hasBlocks()`, parent doc IDs at 0/4/8/... and a clean `CheckIndex` (`verify-write-path.sh` 16/16 -> **17/17**). Also closed: `BinaryDocValuesFieldUpdates`, `inflateGens`' per-segment half (all four of Java's transient generation fields now on `SegmentCommitInfo` + `IndexFileNames.parseGeneration`), and b9's `sci_id`. The FFI delete path changed semantics deliberately -- `ffi_writer_update_document`/`ffi_writer_delete_documents` are now **buffered** like Java's instead of committing a `segments_N` per updated document; ABI unchanged, ~160 lines of FFI-side segment reopening deleted, five tests pin the new timing. **Measured**: the seqNo machinery is below `index-bench`'s noise floor in an interleaved 8-pair A/B (median 20.6 vs 19.9 us/doc, sign alternating). Five of the nine CORRECTNESS findings were defects this batch introduced -- the honest shape of a change that makes a previously immutable structure mutable -- three of them caught by the Tier-2 review: a rollback rewound the delete generation but not the segments carrying it, so every later delete silently reached only the oldest segment; the `.dvu` overlay name encoded its generation in decimal where Java (and this port's own `liv_file_name`) uses base 36; and a flush between `prepare_commit` and `finish_commit` discarded both its segment and the deletes it had resolved. All three fixed, each with a test verified to fail against the pre-fix code. Real Lucene now also validates the `.liv` the buffered-delete path writes (the block fixture buffers a delete before committing; `CheckIndex` cross-checks the bitset against `segments_N`'s `delCount`). Carry-overs: a segment carrying doc-values updates is not readable by real Lucene, because the overlay format is this port's own (b6's declared scope) -- the semantics above it are ported and tested, the byte format is not |
| c11-occur-filter | search/{query,lib,explain} -- `Occur.FILTER` | swept (8 findings) -- see `docs/sweep/m2/c11-occur-filter.md` |
| c12-search-features-2 | search/{lib,facets,ordinal_map,highlighter,query_parser,doc_value_query,directory_reader} + `fixtures/src/GenFacets.java` | swept (2 CORRECTNESS both fixed, 14 MISSING (11 fixed, 3 recorded with named blockers), 3 PERF all measured, 4 INTENTIONAL) -- b14's whole remaining facet layer landed (`OrdinalMap`, `FacetsConfig`, `SortedSetDocValuesReaderState` with flat `OrdRange` *and* hierarchical `DimTree`, `getAllDims`/`getSpecificValue`/`getTopDims`, `FacetResult.value`'s `-1`, multi-valued range counting + `totCount`), pinned against real `lucene-facet` output over a new three-segment `GenFacets` fixture (18 differential tests, all green first run); `MultiFieldQueryParser` with the disjunction at the leaf and per-field boosts; the highlighter's hand-rolled abbreviation list turned out to be a **CORRECTNESS divergence** from the JDK's `BreakIterator` (which suppresses nothing) and is replaced by UAX #29 via `unicode-segmentation`, with `SplittingBreakIterator` and the `FieldOffsetStrategy` selection + postings/analysis offset sources ported (postings offsets verified against real Lucene's own occurrence list); `FieldExistsQuery` + `IndexOrDocValuesQuery`'s planner ported at the layer holding the readers, with b14's "needs a `Clause` variant" diagnosis corrected to "needs a doc-values input in `resolve_clause_docs`"; c11's dead MAXSCORE body verified dead two ways and **deleted**; c1's F-13 closed at **4.8x** (`DirectoryReader::open` 579 us -> 120.7 us) via a new `blocktree::SharedBytes`, since `Arc<[u8]>` could never have aliased the mapping; and c8's `PostingsFlags::DocsOnly` wired through every unscored path with the contract enforced *structurally* (a bare `Vec<i32>`, a `next_doc`-only newtype, and a `LegRole::Scoring` variant holding the scoring inputs) rather than by convention |
| c9-check-index | b11 carry-over: `check_index`/`checksum_verify` vs `CheckIndex.java` method by method -- norms values, positions/offsets/payloads, seek/intersect/skip agreement, term-vectors-vs-postings, vectors + HNSW | swept (19 findings: 1 CORRECTNESS fixed, 12 MISSING (11 fixed, 1 recorded), 3 PERF (2 fixed/bounded + measured, 1 reasoned), 3 INTENTIONAL) -- `testFieldNorms`, `checkFields`' positional/statistical/seek/intersect/skip blocks, `testTermVectors`' slow-level cross-check, `testVectors` + `testHnswGraphs` (expressible now that c5 landed the vectors codec), `testPoints`' `docCount` half and `testDocValues`' ordinal-space half all ported; the `file:*` check upgraded from `retrieveChecksum` (footer shape) to c4's `checksumEntireFile`, which `checksum_verify` now shares instead of duplicating; counts cross-checked against real Java `CheckIndex -level 3` output on five fixtures (414 terms / 8968 pairs / 13547 tokens etc, all equal); 24.7 ms for the whole 8959-doc segment vs Java's 197 ms, and 16x the documents for 11x the time |
| c10-vectors-wiring | c5/c9 carry-over: make the vector subsystem reachable -- `IndexWriter` vector fields, the codec-level merge entry points, and c5's `lucene-util` move | swept (45 findings: 5 CORRECTNESS all fixed, 23 MISSING (19 fixed, 4 recorded with named blockers), 3 PERF (1 fixed, 2 measured), 13 INTENTIONAL) -- **`IndexWriter` can now index vector fields**: `set_vector_field`/`add_vector_field`/`add_document_with_vectors`/`set_hnsw_parameters`, both encodings, all four similarities, dense and sparse, `shouldCreateGraph`'s tiny-segment skip, real `PerFieldKnnVectorsFormat` file names and `.fnm` attributes. **Evidence**: a new `write_vector_segment_fixture`/`VerifyVectorSegment` case opens a 3000-document Rust-written index with real `DirectoryReader`, runs `KnnFloatVectorQuery`/`KnnByteVectorQuery` against **Lucene's own brute-force top-k** (recall@10 0.9083/0.9750/0.9750, and the sub-threshold field exact), and runs real `CheckIndex` (`verify-write-path.sh` 17/17 -> **18/18** with this batch's case; green at 19/19 once a concurrent batch added its own); this port's own `check_index` also runs over a writer-produced vector segment for the first time (c9 had no producer). Two negative controls: dropping the `PerFieldKnnVectorsFormat.format` attribute makes all four fields read back empty **with no error from Lucene**, and a `.fnm` claiming `vector_dimension > 0` for a field the flush wrote nothing for likewise opens and passes `CheckIndex` -- which is why both are asserted directly. **The merge entry points are ported**: `Lucene99FlatVectorsWriter.mergeOneFlatVectorField` (new `FlatVectorsWriter` consumer; the merged `.vec`/`.vemf` are proven **byte-identical** to a flush of the same documents, and each unbroken run of surviving ordinals is one `memcpy` where Java re-encodes per vector), and `Lucene99HnswVectorsWriter.mergeOneField` + `IncrementalHnswGraphMerger` + `MergingHnswGraphBuilder` + `InitializedHnswGraphBuilder` + `UpdateGraphsUtils.computeJoinSet` -- **11,124 similarity computations against a rebuild's 122,473 (11.0x fewer)**, asserted at a 3x floor so it cannot silently regress into a rebuild, and merging one undeleted graph reproduces it **arc for arc** (the assertion that discriminates; recall does not). The `write_vectors_fixture` case gained a fifth field produced by the *merge* path, so real Lucene reads and searches merge-written bytes too (recall@10 0.9750). One real CORRECTNESS fix in c5's code: `HnswGraphSearcher` sized its `visited` bitset from `graph.size()` where Java uses `maxNodeId() + 1` -- equal for a finished graph, which is why every c5 test passed, and an out-of-bounds panic the first time a merge searches a partly-built graph. Also found: **Java's own merge is not reproducible** when deletions are present (`InitializedHnswGraphBuilder.rebalanceGraph` draws from an *unseeded* `SplittableRandom`), and `MergingHnswGraphBuilder` hands hppc's hash order to the search -- so an arc-for-arc differential against Java's *merged* graph is not achievable, which is why the evidence is exact equality against the source graph plus real Lucene searching merge-written bytes. **Measured**: below `HNSW_GRAPH_THRESHOLD` a 128-dim vector per document is free (median 22.0 vs 20.6 us/doc, the vector arm faster in two of five interleaved pairs); above it the graph is the whole cost -- 50k docs at dim 16/64/128 cost 85.5/142.9/203.8 us/doc (interleaved medians) against a 25.8 us/doc vector-free baseline, on uniform-noise vectors (the ANN worst case; c5 measured clustered data at 2.4x less work). c5's `lucene-util` carry-over closed: `SplittableRandom`, `TernaryLongHeap` and the `NumericUtils` float helpers moved to `lucene-util`, 100% covered; `hnsw.rs` 96.75%, `hnsw_vectors.rs` 95.60%, `vectors.rs` 96.94%, `index_writer.rs` 98.29%. **Tier-2 review run on the diff: no gating findings** -- it walked `computeJoinSet`/`copyGraphStructure`/`rebalanceGraph`/`updateGraph`/`addReader` against the Java line by line and confirmed them faithful, and independently verified that Java's `tryPromoteNewEntryNode` assertion makes the guarded Rust form equivalent rather than divergent, and that the flush path's `shouldCreateGraph(threshold, final_count)` is equivalent to Java's *incremental* form plus `replayBufferedVectors` (an equivalence c5 had assumed). All nine advisories acted on, three of them real: `copy_graph_structure` bounds-checked the node ordinal but indexed the **neighbour** ordinal unchecked (a short ordinal map panicked instead of erroring -- and the existing test only took the error path by accident of which ordinal the level assignment put on top); a checksum-valid `.vex` with an over-full upper-level node reached a `NeighborArray` `assert!` (panic in release) rather than a `CorruptMeta`; and `rebalance_graph` drew from the module's default seed instead of the builder's, so the "merged graph is a function of its inputs" property held for the inputs but not for the caller's seed. Also caught: a **tautological** assertion inside `assert_well_formed` (which runs on every merged graph, so it read as coverage that was not there), the flat merge leaving the writer half-written on an error path, and a `parity.md` row still pointing `SplittableRandom` at `lucene-codecs/src/hnsw.rs` after this batch moved it. `merge.rs` wiring is a precisely-described handoff (c8 owns that file); FLOAT16 is recorded with a named blocker. **The index-sorted flush is now a live cross-batch hazard rather than a dormant gap**: a concurrent batch is adding `set_index_sort` + `pending_sort_map` to `index_writer.rs`, and `pending_sort_map` is currently declared but never read, so nothing is broken yet -- but the moment the sorted flush is wired, `build_vectors_output` assigns ordinals in *buffer* order and would attach every vector to the wrong document, silently and `CheckIndex`-clean. The fix is `Lucene99FlatVectorsWriter.writeSortingField` + `reconstructAndWriteGraph`, and the graph half is already available as this batch's `HnswGraphBuilder::init_graph` (tested arc-for-arc under a reversed ordinal map). Flagged in `c10-vectors-wiring.md` finding 11 for whoever lands the sorted flush |
| c14-dv-updates-format | c7's F-15 carry-over: replace the invented `.dvu` doc-values-update overlay with Lucene's real generational representation (`ReadersAndUpdates.writeFieldUpdates`, `Lucene90DocValuesConsumer`, `SegmentDocValuesProducer`) | swept (18 findings: 7 CORRECTNESS all fixed, 7 MISSING (5 fixed, 2 recorded with named owners), 2 PERF (1 fixed, 1 recorded with a cost statement), 2 INTENTIONAL) -- **the third invented format is gone** (after b11's `.si` index sort and c5's `.vec`/`.vem`). A doc-values update now rewrites the updated field's *whole column* into a generation-suffixed `_<seg>_<base36 gen>_Lucene90_0.{dvm,dvd,dvs}` triple plus a `FieldInfos` generation `_<seg>_<base36 gen>.fnm`, recorded in `SegmentCommitInfo` (`docValuesGen`/`fieldInfosGen`, `dvUpdatesFiles` **replacing** rather than accumulating). New `lucene-index/src/field_updates.rs` is `writeFieldUpdates`; `doc_values_updates.rs` gained the merge + generation writers; `SegmentReader::open` reads the generational `.fnm` and resolves each field to its own generation (`SegmentDocValuesProducer`), so a never-updated field still comes out of the base column. **Proven in both directions**: a new `GenDocValuesUpdates.java` fixture (three update rounds, three fields in the three states a reader must tell apart) is decoded value-for-value against Lucene's own answers, and a new `VerifyDocValuesUpdates` case has real Lucene read every document's updated value out of a **Rust-written** index and run `CheckIndex` (`verify-write-path.sh` 18/18 -> **19/19**). Also fixed on the way: `Lucene90DocValuesConsumer`'s `numDocsWithValue == 0` shape (`meta[-2, 0]`), which an all-reset generation reaches; `advanceFieldInfosGen`/`advanceNextWrite*Gen` were missing from `SegmentCommitInfo`; and a segment carrying *only* an update generation (a field with no base column) was admitted to auto-merge, which would have dropped the update silently. Five CORRECTNESS findings came from the Tier-2 review, three of them defects this batch introduced -- the honest shape of replacing a format wholesale: the per-field codec suffix was hardcoded instead of read off the field (wrong the moment an update lands on a Lucene-written segment), the failure path *rewound* the next-write generation counters so a retry could reuse a partial file's name, and the base column was located by "first `.dvm` in the `.si`" with a miss degrading to an all-absent column -- which would have dropped every untouched document's value into a well-formed, checksummed file. The review also caught the column merge calling the free `numeric_value`/`binary_value` once per document (`O(maxDoc x blocks)` on a sparse column, the same defect b13 fixed in `effective_live_docs`); fixed with `NumericReader` and a new `BinaryReader`. Recorded: the `.dvu` encoding survives only as `soft_deletes.rs`' standalone marking serialization, referenced by no index -- removing it is c12's call; `SegmentReader::doc_values_for_field` is correct and tested but has no production callers yet (`lucene-ffi`'s sort/facets still read the base pair -- c12/c13); and `check_index` reads the base `.fnm`, so it never opens a generation column (c9, a coverage gap, not a false failure -- the generational files' checksums *are* verified) |
| c13-ffi-surface | ffi/{handle,registry,raw,segment,query,explain,directory_reader,writer,lib} + new ffi/{vectors,legacy_boolean_abi} | swept (24 findings: 9 CORRECTNESS all fixed, 12 MISSING (9 fixed, 3 recorded with named blockers), 1 PERF fixed+A/B measured, 2 INTENTIONAL) -- the C-ABI surface gaps later batches opened but could not close. **The boolean wire format is rebuilt**: three `must`/`should`/`must_not` four-array buckets replaced by one flat, `Occur`-tagged, parent-indexed clause array, so c11's `Occur.FILTER` (44% cheaper than the equivalent `MUST`), arbitrarily nested `Clause::Boolean`, and `minimumNumberShouldMatch` (which had no wire representation *at all*, at any level) are now expressible -- and an `Occur` or a `(field, term)`-shaped leaf kind is a **value**, so the next of either costs no ABI break. Acyclicity is structural (`parent < i`, one no-recursion reverse pass) and depth is capped at 32, because caller-controlled nesting is caller-controlled stack depth and a stack overflow *aborts* where `catch_unwind` cannot reach. **KNN crosses the boundary**: `ffi_open_vectors`/`ffi_knn_{float,byte}_vector_search` expose c5's flat vectors + HNSW graph over their own handle (a vector field needs no term dictionary, so `SegmentHandle` was the wrong home), differentially verified by running **every query the vectors fixture records through the exported C symbols** and matching real Lucene's `KnnFloatVectorQuery`/`KnnByteVectorQuery` doc-for-doc and score-for-score, plus the graphless exact branch and the sparse ord->doc mapping; deletions are honoured exactly (beam widened by the deleted count) with the divergence from Java's lazy `acceptOrds` walk recorded and its blocker named. **c7's whole delete-queue surface landed** -- block adds, `updateDocuments`, `softUpdateDocument`, numeric/binary doc-values updates, and `deleteDocuments(Query...)` over the same parent-indexed node-array format -- and every mutating writer call now returns Java's `long` seqNo (c7's A7), starting at 1. b15's four unwrapped field setters landed too; without `add_postings_field` a writer could index exactly one searchable field and without `set_norms_field` it wrote no norms at all, so `ffi_open_segment`'s `nvm_name` had nothing to open. b15's `.pay` deferral closed (the multi-segment path had already started honouring payloads while the single-segment one structurally could not). **b15's finding 11 measured and removed**: the results registries are sharded 16 ways with the shard in four handle bits taken from the *generation* field, not the index one -- paired A/B, same binary built twice with one line changed, **2.08x at 16 threads and 2.19x at 32** (1.00x at 1, ~1.1x at 4, which is why b15's four-thread 1.17x looked like a ceiling: four threads on a 20-core box barely contend, and an OpenSearch search pool is sized to the core count). Also fixed: `map_writer_error` was sending every c7/c10 caller-misuse error to `Io` through a `_` arm, so a JNI caller would fail a shard for a bad argument -- the arm is now gone, both sides enumerated, so the next variant added to `lucene-index` fails to compile here. The ABI break spans 11 exported functions and is proved behaviour-preserving by a test-only bridge that replays the whole pre-c13 boolean suite through the real exported symbols. Tier-2 review: four gating findings, three fixed (a self-contradicting `parity.md`, a comment describing a fix that had not been made, a caller error surfacing as `Decode`) and one open -- Java's *query-level* KNN policy sits in `lucene-ffi` because `lucene-search` (a concurrent batch's files) has no vector module, so no non-FFI consumer can run a vector query yet |
| c17-index-sort | c10's carry-over: make the **index-sorted flush** reachable -- index/{index_writer,segment_writer,indexing_chain} + `field_updates`, `codecs/doc_values` (one variant), new `fixtures/src/{GenSortedIndex,VerifySortedSegment}.java` | swept (19 findings: 4 CORRECTNESS all fixed, 10 MISSING (9 fixed, 1 recorded with a named blocker), 1 PERF measured, 4 INTENTIONAL) -- **`IndexWriter` can now index-sort a flush**: `set_index_sort`/`index_sort`, with all of Java's validation (`validateIndexSortDVType`, `validateIndexSort`/`isCongruentSort`, the doc-values-update guard, and the blocks-without-a-parent-field `CorruptIndexException`) plus one this port needs because its doc values are opt-in -- a sort field must actually be *written*, since `DocValues.getNumeric` substitutes an all-missing column rather than failing and would make `CheckIndex.testSort` vacuous. **Design divergence, deliberate**: Java threads a `Sorter.DocMap` into each of eight per-format writers; this port permutes the **document buffer** once, before any format is built, so stored fields, postings, term vectors, norms, doc values and vectors land in sort order together and by construction consistently -- the per-format remap is exactly where a format gets forgotten (c10 found the vector one unreachable here for that reason). c10's finding 11 said to call `segment_writer::flush_sorted_stored_only_segment` from `flush`; that turns out to be **wrong** -- it sorts the stored fields only, so every other format would still address the original doc ids. **The headline CORRECTNESS fix**: `sort_key_rank` pinned missing values to one end *regardless of `reverse`*, where Lucene substitutes a `Long.MIN_VALUE`/`MAX_VALUE` **sentinel** and compares it like any other value, so `reverseMul` applies to it too. Every reversed sort with a missing key therefore produced a segment physically ordered one way and described the other way by its own `.si`. Proven both directions: the new `fixtures/data/sorted_index/` (a real `IndexWriter` with `setIndexSort`) puts its six missing-`rank` documents **first** under a missing-*last* descending sort, and the old comparator rejects that real index; and with the old comparator real `CheckIndex` rejects what this port writes (`docID=1944 sorts after docID=1945`). **Evidence**: new `write_sorted_segment_fixture`/`VerifySortedSegment` (`verify-write-path.sh` 19/19 -> **20/20**) -- Java re-derives the expected permutation *itself* and checks, per doc id, the stored value, both doc-values columns, the unique postings term, the norm and every vector component, plus `LeafMetaData.sort()` tier for tier, a real `KnnFloatVectorQuery` over the sorted ordinal space, and `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS`. Two negative controls: **dropping the vector buffer's permutation attaches every vector to the wrong document and is `CheckIndex`-clean** (the hazard c10 flagged mid-batch), and the pre-batch comparator fails Lucene's own `testSort`. Two enabling MISSING fixes came out of needing a *multi-field* sort: `IndexWriter` could write only **one** doc-values field per segment (`add_doc_values_field` now, one `.dvm`/`.dvd` for all of them), and `doc_values::write_dense_fields` was dense-only, so a sort tier with missing values was inexpressible (`DenseField::SparseNumeric`). Also: a segment-private delete's `docIDUpto` is a **pre-sort** buffer position and is now mapped through `newToOld` at all three comparison sites (Java's `ReadersAndUpdates.sortMap`), and `field_updates::check_updatable` now accepts a field with no base column (`verifyOrCreateDvOnlyField`'s create half). **Measured**: the sort is free at the default 16 MB RAM buffer (21.2 vs 21.0 us/doc against a same-doc-values control, seven interleaved runs) and **+0.3 us/doc (~1%)** for a single flush of all 50 000 documents. **A merge does not preserve the sort** and is not wired: `merge_sorted_stored_only_segments` writes no postings and no vectors, so routing `execute_merge` through it would trade a lost sort for lost data -- instead `segment_stats` refuses to offer a sorted segment to the merge policy, and the wiring is a precise handoff. **Tier-2 review: three gating findings, all fixed**, one of them a real defect this batch introduced -- a flush failing *after* the sort left the buffer permuted, so the retry took the identity short-circuit, produced no sort map, and compared pre-sort delete limits against sorted doc ids, deleting the exact complement of the right documents with a valid `.liv` and a clean `CheckIndex`; fixed by extracting `build_and_write_segment` (one error path) which restores insertion order. The other two were docs that still stated the semantics finding 1 corrected (`SortMissingValue`'s own enum comment, and `set_doc_values_field`'s dense-only claim). Coverage: `index_writer.rs` 98.10%, `segment_writer.rs` 99.49%, `doc_values.rs` 98.27%, `field_updates.rs` 96.32% |
| c20-postings-skip | c15's F10 carry-over: wire up `.doc`'s level-0/level-1 `.pos`/`.pay` skip pointers -- codecs/{postings,postings_writer}, new `fixtures/src/GenPostingsSkip.java` | swept (9 findings: 1 CORRECTNESS fixed, 3 MISSING all fixed, 3 PERF all fixed (two measured), 2 INTENTIONAL) -- **`advance(doc)` no longer walks the doc list.** `read_full_block_header`/`read_level1_entry` retain `posEndFPDelta`/`posBufferUpto`/`payEndFPDelta` instead of discarding them; `LazyDocsCursor` accumulates them with `BlockPostingsEnum`'s exact state machine (`skipLevel1To`'s `level0PosEndFP = level1PosEndFP`, `skipLevel0To`'s pre-header snapshot, `readLevel0PosData`) and `position_origin()` hands out `seekPosData`'s arguments plus `accumulatePendingPositions`' in-block frequency sum; `read_occurrences_for_doc` (behind `FieldTerms::occurrences_for_doc`, which the highlighter calls) is `advance` + `skipPositions` + `nextPosition`. **Measured** on `benchmarks/.corpus/merged` (5.2 M docs), one document of `t0` (`docFreq` 4 997 130 / `ttf` 56 600 329), A/B/C in one build: whole-term reader 2.77 s -> c15's frequency-sum `advance` 34.4 ms -> **1.57 us** for the first document (a shared, heavily loaded machine, so read the ratios and not the absolutes; re-measured after the Tier-2 review at 1.19 us). The write side had to come first: `postings_writer` emitted **no** pos/pay sub-fields and refused `docFreq >= BLOCK_SIZE` on a positions field for exactly that reason, so this port could not write the shape it needed to test -- `PosSkipWriter`/`PositionLayout` reconstruct the samples Lucene takes live (exactly, since a `.pos` block closes every 256 occurrences regardless of doc boundaries) and the ceiling is gone. `TermMetadata::last_pos_block_offset` becomes load-bearing on the read side for the first time (b5 F4's defect is now visible to a test). New real-Lucene fixture `postings_skip_index` (`GenPostingsSkip.java`): 8 500 documents, one term, offsets and payloads, so a level-1 entry + 33 level-0 headers + a vint tail all carry the pointers -- no other fixture in the tree contains a single byte of them. b5's F7 closed on the way (level-1 impacts were decoded and allocated for every span *skipped*, which with the pointers wired up became the dominant residual: reaching `t0`'s last document crosses 610 spans). **Tier-2 review found one gating item, and it was the important one**: the fixture's frequency cycle had period 4, so `sum(1 + d % 4)` over the first 8192 documents was exactly 80 whole `.pos` blocks and the level-1 `posBufferUpto` was `0` -- the level-1 pointer byte was indistinguishable from a hardcoded zero, on both sides, by every test the batch added. Both generators now cycle 1..5 (coprime with 256, giving 253), the manifest carries the value so the test pins non-degeneracy, and a second sparser term was added because the dense one is in every document and so takes Lucene's degenerate all-consecutive doc-delta encoding in all 33 of its blocks. Five advisories acted on, including a `level0NumBytes` cross-check the level-0 header never had (which found four hand-built test headers writing the block body's length where Lucene writes the metadata region plus the header fields), an occurrence budget against the `.pos` bytes that exist (an allocation-abort class, not a panic one), and `code >> 1` where Java has `>>>`. Coverage: `postings.rs` 97.33%, `postings_writer.rs` 99.55% |
| c21-hnsw-seeded | c16's three blocked items: seed the optimistic re-entry pass, push `acceptOrds`/`filteredDocCount` into the codec API, and expose filtered KNN over the C ABI -- codecs/{hnsw,hnsw_vectors}, search/vector_query, ffi/vectors, new `fixtures/src/{GenVectorsSeeded,GenVectorsFiltered}.java` | swept (19 findings: 2 CORRECTNESS both fixed, 7 MISSING all fixed, 1 PERF measured, 9 INTENTIONAL; Tier-2 review run, three gating findings and four advisories, all acted on) -- **`SeededHnswGraphSearcher` is ported** as `HnswGraphSearcher::search_seeded` (a second entry point on the one searcher, since Java's class exists only to override `findBestEntryPoint` with a constant and delegate `searchLevel`), reached through `hnsw_vectors::search`'s new `SearchOptions`, which also carries Java's `acceptOrds` and `filteredDocCount` -- so `vector_query::hnsw_search`, the ~40-line copy of that dispatch c16 had to write, **is deleted** and there is one walk again (the c15 lesson: two copies of one decode loop drift exactly where no test compares them). **A real CORRECTNESS defect found in c16's own code**: Java recomputes `perLeafTopK` from `ctx.parent` inside `getLeafResults` on *every* call, so phase 2's `cost <= perLeafTopK` short circuit and its `scoreDocs.length >= perLeafTopK` fall-back stay **pro-rata**; only the collector becomes full `k` (it arrives through `ReentrantKnnCollectorManager`'s delegate, which `getLeafResults` never inspects). c16 raised both thresholds with the collector, which takes the exact-search branch where Java walks the graph on any filtered re-entered leaf. Also fixed: pushing `acceptOrds` down made a caller-supplied `FixedBitSet` shorter than the field reachable from a public codec API, i.e. an out-of-bounds panic where Java's unbounded `Bits` has nothing to check -- now a typed error, as is a negative `filteredDocCount`. **Filtered KNN is exposed over the C ABI** (`ffi_knn_{float,byte}_vector_search_filtered`) reusing c13's occur-tagged clause array verbatim -- the first clause-shaped addition since that ABI break, and it cost no new encoding -- plus the same segment's `SegmentHandle` for the term dictionary `ffi_open_vectors` deliberately does not open, cross-checked on `maxDoc` because a filter resolved against a different segment is silently wrong. **Two new fixtures, both forced rather than chosen**: `vectors_multi_index` reaches the re-entry pass but only on its 40-document *graphless* leaf, where Java ignores the search strategy entirely -- so none of c16's 80 queries said anything about seeding. A graph-bearing leaf can only be re-entered when `perLeafTopK < k` (`p < 0.039` at `k = 10`, under ~156 documents) *and* `shouldCreateGraph` holds (~660 vectors), which are only compatible at a larger `k`: `vectors_seeded_index` is 1400/700/700/40 at `k = 100`, where the clustered 700-document leaf has `perLeafTopK = 93` and is re-entered on all 20 queries with 93 seeds. `vectors_filter_index` is one 1200-document segment *with postings*, because the C ABI opens one segment per handle and single-leaf ground truth (`perLeafTopK == k`, no re-entry) cannot come from one leaf of a four-leaf index. **Measured**: seeding removes `findBestEntryPoint`'s descent -- 216 vector comparisons against 292 (26%) on the 4000-vector fixture at `k = 10`, but only 530 against 553 (4.1%) for the `k = 100` phase 2, and **no measurable query-level change** (130.3 vs 130.5 us/query, p = 0.88). So it does *not* claw back c16's 7x fan-out gap: that gap is the pro-rata collectors, which Java pays too. Honest limit recorded: seeding changes no answer on either fixture, so its discriminating evidence is structural (a seeded walk is a fixpoint on its own answer, does strictly fewer comparisons, and under a one-visit-per-seed cap returns exactly its seeds) rather than differential. c16's 16 differential tests and the 25 pre-existing `lucene-ffi` vector tests all pass **unmodified**; `verify-write-path.sh` confirmed green by running it (20/20 at the start of the batch, 21/21 at the end -- a concurrent batch added a case) |
| c23-positions-writer | c20's two carry-overs: make `IndexWriter` index positions/offsets/**payloads** end to end and get real Lucene to read them back -- index/{indexing_chain,index_writer,segment_writer}, codecs/field_infos (`write` only), new `fixtures/src/VerifyPositionsSegment.java` + `examples/write_positions_segment_fixture.rs` + `tests/positions_write_path.rs` | swept (14 findings: 3 CORRECTNESS all fixed, 6 MISSING all fixed, 2 PERF both measured, 3 INTENTIONAL) -- **real Lucene 10.5.0 now reads back a `.pos`/`.pay` this port wrote, occurrence by occurrence.** `verify-write-path.sh` 21/21 -> **22/22**: a 20 000-document, six-field index (one field per `IndexOptions` rung plus offsets-without-payloads and payloads-without-offsets, sharing one `.doc`/`.pos`/`.pay` set) whose dense term closes **two** `LEVEL1_NUM_DOCS` spans, with a frequency period coprime with 256 so the level-1 `posBufferUpto` cannot pass for a hardcoded zero, terms at `ttf` 256/257 (`lastPosBlockOffset`'s `-1` sentinel branch), a singleton, and a document whose occurrences straddle a `.pos` block boundary; Java walks 51 boundary-chosen documents with a **fresh `PostingsEnum` each** -- so every sample is reached through `advance` and c20's skip records -- comparing position, both offsets and payload, then runs a `PhraseQuery` (and its reverse, which must match nothing), the term vector, and `CheckIndex`. The manifest is re-derived from the document text by an independent scan, never from `invert_documents`, because a manifest built from the structure under test agrees with it however wrong both are (b4/b11's failure shape). **Negative controls, all run**: the level-1 `posBufferUpto` written as a constant `0` fails at doc **8192**, the level-0 one at doc **256**, `position + 1` at doc 0, and c20's F9 is confirmed from the other side -- the level-0 `payloadByteUpto` written as `0` is correctly **not** caught, because Java overwrites it from the landing block. **Three CORRECTNESS fixes.** (1) `build_postings_output` passed `has_payloads: false` unconditionally while the `.fnm` carried the caller's `STORE_PAYLOADS` bit, so a `store_payloads` field produced a segment real Lucene either cannot open (no `.pay` at all, without offsets) or reads garbage from (measured: every offset and payload wrong). (2) Term vectors recorded positions only whatever the field indexed, which made `CheckIndex.testTermVectors`' offset and payload cross-checks **skip** rather than fail -- and `CheckIndex` stays clean under that mutation, which is the finding. (3) `field_infos::write` put `omitNorms`/`storePayloads`/`storeTermVectors` on the wire for a **non-indexed** field, where Java's `FieldInfo` constructor coerces them away: real Lucene coerces on read too (so every cross-engine verifier stayed green) but this port's own `parse` rejects it, so `check_index`'s `fnm.open` failed and **every postings check in the segment was silently skipped** -- `all_passed()` returning true over a term dictionary that had never been opened. Found by doing what the brief asked: running our own `check_index` over a writer-produced positions segment. Also fixed: a `#[test]` duplicated on one function and absent on the next, so one doc-values test had never run. Payloads reach the indexing chain through `PayloadSource`, this port's stand-in for `PayloadAttribute` (`Token` has none), gated per field on `FieldInfo.store_payloads` because payload presence is a field property in Lucene and never a per-token one; `BoxedPayloadSource` is `Send + Sync` because `lucene-ffi`'s `unsafe impl Send for WriterHandle` rests on `IndexWriter` having no interior mutability. **Measured** (`index-bench`, 50k docs x 40 tokens, interleaved, new `LUCENE_RUST_INDEX_OPTIONS` arm): positions **+1.3 us/doc** over `DocsAndFreqs`, offsets **+3.4**, payloads **+26.1 and +190 MB** -- and an all-empty-payload control costs the same, so the payload cost is the per-occurrence `Vec<u8>` slot, not the bytes: a fresh instance of the block-pool item below, whose fix (a flat `(bytes, lengths)` shape in **both** `TermPostings::payloads` and `PostingEntry::payloads`) is a codec-side API change. Coverage: `indexing_chain.rs` **97.06%** (all 8 misses are assert-message arms), `segment_writer.rs` 99.49% -- swept and deliberately unchanged, the positional flush lives in `index_writer`, not here; `index_writer.rs` 85.63% in this snapshot, with 415 of its 1 319 missed lines in `execute_merge` alone and none in this batch's code |

| c19-coverage-hardening | c15's carry-overs: the last two files under the ≥95% bar (`index/{check_index,checksum_verify}`), a **mechanical gate** for the arithmetic-on-a-file-value panic/abort class (workspace `[lints]` + `docs/arithmetic-gate.md`), and c15's weak-floor negative-control shape applied to five `caught > 0` assertions | swept (5 CORRECTNESS all fixed with tests, 1 MISSING fixed, 3 INTENTIONAL) -- `clippy::arithmetic_side_effects` is now denied crate-wide in `lucene-store` (fully audited), `lucene-codecs` and `lucene-index`, with a per-module `TODO(arith-audit)` burn-down in each `lib.rs` so a *new* module is gated from its first line; the audit found `.kdm`'s `numDims x bytesPerDim` overflowing an `i32` (Java wraps to `NegativeArraySizeException`, we panicked), two `.kdm`-sized allocations that **abort**, `.nvm`'s `docsWithFieldOffset + docsWithFieldLength` overflowing an `i64` in both `norms.rs` and `check_index.rs`, `checkDocValueSkipper`'s `maxDocID(0) + 1`, and `.tim`'s term-stat accumulators; the five negative controls now re-sign the footer and assert measured floors (85/99 `.nvm`, 44/99 `.tip`, 138/318 `.vex`, 69/261 `.dvd` numeric+binary, 18/99 `.dvd` sorted, 5/2034 `.doc`) plus an isolation case; `vectors.dimension_positive` withdrawn as unfalsifiable in Java as well as here; `checksum_verify.rs` 93.00% -> **97.11%** (over the bar), `check_index.rs` 89.19% -> **94.19%** (still 0.81 under it; the residual is ~100 individual never-fired check arms with no block larger than 12 lines, enumerated in the report) |

| c25-check-index-coverage | c19's one unmet requirement: `check_index.rs` at 94.19%, and the ~110 individual `Check::fail`/`problems.push` arms behind that number that had **never once fired** -- index/{check_index, checksum_verify} | swept (2 CORRECTNESS both fixed, 10 arms deleted as unreachable, 21 new tests) -- **`check_index.rs` 94.19% -> 97.25%**, over the bar. The valuable half is the deletions: **ten arms could not fire at all**, in two shapes. (a) `x == x` -- the check compares two values this port's decoder derived from the *same* `.fnm`, where Java's codec could in principle disagree: all three of `checkFields`' frequency-flag cross-checks (`!hasFreqs => totalTermFreq == docFreq`, `=> freq == 1`, `=> sumTotalTermFreq == sumDocFreq`) fell this way, and removing them removed the **last use of `has_freqs` in `check_postings`** -- the `.fnm`'s freq flag has *no* independent on-disk witness in this port, unlike `hasOffsets`/`hasPayloads`, which have one (whether the segment carries a `.pay`). Also `postings.field_in_fnm`'s no-`.fnm`-entry arm (`blocktree::open` names every field *from* the `FieldInfos` it validates against). (b) **the decoder already rejected it**: `.tmd` `docCount > maxDoc` and `sumTotalTermFreq < sumDocFreq` are `blocktree::open` errors, so the file is reported as `postings.open` and never reaches the check; `hnsw.neighbors_sorted`'s out-of-order half is impossible because `neighbors_into` decodes a running sum of **unsigned** deltas (only the *repeat* can fire, which is why c19's 318-corruption `.vex` sweep never tripped it); the `.nvm`-vs-`.nvd` version mismatch is impossible because `norms::VERSION_START == VERSION_CURRENT == 0`; a fresh `DocValuesSkipper` sets `max_doc_id` to `-1` unconditionally one line above the check that asserts it. **Two CORRECTNESS defects, both found by this batch's own new re-signed `.tvd` sweep on its first run, both in `lucene-codecs/src/term_vectors.rs` -- a module the arithmetic gate had already been through**: a `prefixLength` off disk slicing the *previously decoded* term (**panic**), and a chunk's claimed decompressed length sizing `vec![0u8; n]` (**SIGABRT** -- `memory allocation of 1011335590898973 bytes failed`, the one shape `catch_unwind` cannot intercept). The gate missed both because it covers arithmetic and shifts and *not* indexing or allocation sizing -- the two rows `docs/arithmetic-gate.md` itself records as uncovered, now with a concrete case each. Fixed with a bounded `get(..)`, a `checked_add`, and an LZ4-max-expansion (255x) ceiling on the decompressed length. **New negative-control rows, extending c19's table**: `.fdt` -> `stored_fields.every_doc_decodes` **33/47** with **0 caught by anything else** (nothing else reads the stored-fields payload, so what this walk misses nothing catches; the 14 it misses are stored-field *values*, which have no second copy on disk); `.tvd` -> `term_vectors.every_doc_decodes` **15/43** with 21 caught elsewhere -- term vectors being the one place two independent copies of the same data exist, so the cross-checks see most corruptions first. Two reusable helpers make the term-dictionary arms falsifiable at all: `patch_tim_stats` and `patch_tmd` decode the written `.tim` stats region and `.tmd` field summary, change one claim and re-encode with the footer re-signed -- `postings_writer` computes every statistic from the postings it is given, so it cannot be *asked* for a dictionary that lies. Their identity round-trip is asserted byte-for-byte, which is what caught `.tmd`'s trailing `indexLength`/`termsLength` being **little-endian** (`writeLong`, Lucene 9+) -- a big-endian round trip is byte-identical and silently wrong. The HNSW degenerate-entry-point arm is driven through this port's own `.vem`/`.vex` writer with a hand-built `OnHeapHnswGraph` whose entry node has no neighbours, against a connected control differing in exactly one edge set. Verifier runtime **0.23 s** (c19: 0.28 s) over every fixture in release -- no new pass over any document, term or byte. `verify-write-path.sh` **22/22** confirmed by running it. **Late addition, handed over by c23 and the most valuable thing in the batch**: c23's `.fnm` defect made `fnm.open` fail, and *every postings check in the segment was then silently skipped* -- and three term-vector families were reported as **passes**, because their problem lists were empty for the one reason that must never read as agreement (nothing had looked). `Check` now carries a three-valued `Outcome` (`Passed`/`Failed`/**`Skipped`**) instead of `passed: bool`; a skip names the prerequisite that took it down, `all_passed()` is false for it, and `CheckResult::skipped()` lists what went unguarded. Only a *failed* prerequisite produces a skip -- a format the segment legitimately lacks still produces no check, which is what keeps the real-fixture pass green. All **17** prerequisites audited (every `*.open` arm plus the per-field "metadata parsed but has no entry for this field" arms), **43** distinct families, pinned by `every_prerequisite_names_the_families_it_takes_down` so a new `*.open` without its `skip_families` call fails the suite. Two cases where `all_passed()` returned **true** over an unverified segment are closed: a `.si` declaring an index sort with no doc-values files (the sort order every merge and every early-terminating query trusts, verified by nothing, reported healthy -- `fnm.doc_values_vs_files` does not cover it because a sort field absent from `.fnm` leaves no field claiming doc values), and a `.fnm` naming a soft-deletes field with no column behind it. The three "0 caught by another check" rows (`.fdt`, `.dvd`, `.nvd`) are why this needed a type change and not a comment: a skipped family there is the file's only reader |
| c26-merge-completeness | c22's Tier-2 root cause: **nothing mechanically checks that `execute_merge` opens every format the flush can write** (the root of c22's findings 14/22/23/24), plus its two merge carry-overs -- index/{merge, index_writer, merge_policy}, two `field_updates` fns made `pub(crate)` | swept (3 CORRECTNESS all fixed, 4 MISSING (3 fixed, 1 recorded with a named `lucene-codecs` blocker), 1 PERF measured, 3 INTENTIONAL) -- **the gate**: `merge::SegmentFormat` (an exhaustive enum over the seven per-segment formats) + `merge::check_format_coverage`, called from `execute_merge` on the real path, so every format a source's own `SegmentCommitInfo::files` lists must be one the caller opened, and an extension no variant claims is an error -- which is the anti-rot loop (a new format cannot be satisfied except by adding a variant, whose two exhaustive `match`es then force the caller to open it). Verified by stubbing each of the five formats out of `execute_merge` in turn -- c22's findings 14, 22 and 23 reproduced deliberately and all caught by name. Run over the whole existing corpus it found **no** unknown drop, which is reported as the real result. Both carry-overs closed: **multi-field norms** on the merge *and* the flush (`Error::TooManyNormsFields` gone, `IndexWriter::add_norms_field` added; c22's stated `lucene-codecs` blocker was stale -- `norms::write_fields` already existed from b6, so **no `lucene-codecs` file was touched**), and **generational doc values merge** (each field resolved to its newest generation via `field_updates::read_current_field_infos`/`read_current_column`, merged segment folds back to `doc_values_gen == -1`, so `segment_stats` now **withholds nothing**). Along the way: `merge_numeric_doc_values` refused a source with no column where `DocValuesConsumer.getMergedNumericDocValues` simply drops that reader's sub -- fixed for NUMERIC, recorded for the other four types (blocked on sparse `DenseField` variants, `c24` owns the crate). Coverage: `index_writer.rs` **98.44%**, `merge.rs` **98.54%**, `merge_policy.rs` **99.06%**; the 46 uncovered regions in `execute_merge` are all `?` error edges on directory reads. `verify-write-path.sh` 22/22, `cargo clippy --workspace --all-targets -- -D warnings` at zero, both scripts exit zero. |
| c28-arith-index | c19's `lucene-index` burn-down: the 8 modules still carrying `#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)` that no other in-flight batch owned -- index/{segment_infos, segment_info, index_file_deleter, deletes, term_delete, update_document, indexing_chain, buffered_updates} | swept (6 CORRECTNESS all fixed, 3 MISSING (2 fixed, 1 recorded), 2 PERF, 2 INTENTIONAL) -- **11 -> 3 modules marked** (the 3 left are c26's `index_writer`/`merge`/`merge_policy`), 42 lint sites resolved. The headline is **`segment_infos::MAX_GENERATION`**: all six generational/commit counters (`delGen`, `fieldInfosGen`, `docValuesGen`, `version`, `counter`, and the `N` parsed out of the `segments_N` *file name* in base 36) were read with no bound and then stepped with a bare `+ 1` in seventeen places -- `SegmentCommitInfo`'s three `advance_*` and three `advance_next_write_*` methods, `derive_next_gen`, `deletes::apply_deletes`, `field_updates`' whole write path, and `update_document`'s commit bump. Java wraps a `long` silently; here it **panics in a debug build**, and in release wraps `i64::MAX` to a *negative* generation that `liv_file_name` formats as `_0_-9223372036854775808.liv`. Capped at `i64::MAX / 2` on the way in, which is what makes every surviving `+ 1` provable (2^62 further index writes to reach the top) -- deliberately **not** `saturating_add`, since a saturated generation makes two successive delete rounds write the *same* `.liv` name, which is data loss rather than a crash. **Three of the six CORRECTNESS findings came from the allocation/indexing hand-check, not the lint** (c25's scope correction, which arrived mid-batch): `numSegments` sized a `Vec<SegmentCommitInfo>` (~150 bytes each) straight off a 4-byte header field -- `i32::MAX` is a **~300 GB reservation and an abort**, in the very first file a reader opens -- with `numDVFields` and the `.si`'s `numSortFields` the same shape; and `term_delete`'s live-docs filter fed a doc ID from a corrupt `.doc` straight into `FixedBitSet::get`, which indexes `words[i >> 6]` and **panics in release too**, where the identical value with `live_docs == None` would have been reported as `DocOutOfRange` -- so whether corruption surfaced as an error or a crash depended on whether the segment happened to have deletions yet. In `index_file_deleter`, which decides what to **delete**: `_0_1y2p0ij32e8e7.liv` is a well-formed base-36 `i64::MAX` in Lucene's own file-name shape, so any process that can create a file in the index directory could panic `inflate_gens` at open time; such names are now discarded as trash exactly as Java discards unparsable ones, and `max_segment_name.saturating_add(1)` became a proved `+ 1` (the saturation it replaced would have pinned `counter` at `i64::MAX` and made every later segment name collide). Also ported both of Java's position guards in `IndexingChain.PerField.invert`, neither of which existed here (A/B on `index-bench`, 8 runs each: 24 693 vs 26 733 docs/s -- inside the noise band, no regression), and stopped `skip_sequence_numbers` accepting a *backwards* jump that reissues a sequence number. **Tier-2 review verified all 23 `// ARITH:` proofs individually** and found one gating defect plus five advisories, all addressed: the cap was inclusive on the way in, so `_<base36(MAX_GENERATION)>.si` drove `infos.counter` to `MAX_GENERATION + 1`, which `write` serialized and `parse` then refused -- **an index this port wrote and could no longer open**, and the original trash-name tests missed it by using `i64::MAX` rather than the boundary. Closed from both ends (`usable_generation` exclusive of the cap; a new `check_writable_generations` on the write path) and pinned by `a_commit_this_port_writes_is_always_one_it_can_read_back`, the round-trip property the cap has to satisfy. Also from the review: six proofs said "`parse` caps it" when the fields are `pub` and the setters take a bare `i64` -- reworded to what is actually enforced, with `debug_assert_generation` added to the three setters as the missing enforcer; `deletes::mark_deleted` still carried the exact panic shape F6 removed from `term_delete` (bounding on `max_doc` but indexing a *separate* caller-supplied bitset), now bounded on `bits.len()`; and the F6 test used doc ID 2 against a 2-bit bitset, where `2 >> 6 == 0` only trips the `debug_assert` -- rewritten to cover both real modes, an empty bitset (`words` empty, so any ID is a release index panic) and a short one (a silent ghost-bit read past `num_bits`). Per-file coverage 97.20-99.20% across all eight; `verify-write-path.sh` **22/22**, both scripts exit zero. |
| c22-sorted-merge | c17's + c10's carry-overs: make the **sort-preserving merge** complete and wire it -- index/{merge, index_writer (`execute_merge`/`segment_stats`)}, additive edits to codecs/{vectors, stored_fields, term_vectors}, new `write_sorted_merged_segment_fixture` | swept (26 findings: 11 CORRECTNESS all fixed, 8 MISSING (7 fixed, norms recorded), 3 PERF (2 fixed, 1 measured), 4 INTENTIONAL) -- **there is now one merge.** `merge::merge_segments` is the single implementation and `sort_fields: Option<..>` is the only difference between the two shapes (`MergeState.buildDocMaps`'s two branches: concatenation, or `MultiSorter.sort`'s k-way merge). c17 could not wire the sorted merge because it wrote no postings, no points and no vectors, so routing `execute_merge` through it would have traded a lost sort for lost data; collapsing the two entry points into one makes it structurally impossible for a format to be written by one and forgotten by the other. **Three CORRECTNESS defects the sorted path exposed**: `build_doc_id_maps` derived `MergeState.DocMap` from the *concatenation* rule, so postings/points/vectors would have addressed a different document space from stored fields/doc values/norms/term vectors (every file valid, terms on the wrong documents); `merge_postings` concatenated a term's postings in source order, which delta-encodes negatively once the sources interleave (Java uses `DocIDMerger`'s `SortedDocIDMerger`); and `merge_one_flat_vector_field` assigned merged ordinals in source order, which `.vemf`'s `IndexedDISI` cannot encode -- it now assigns them in merged-document order (`DocIDMerger.of(subs, needsIndexSort)`) while keeping the per-run `memcpy`. **A fourth, found by running this port's own `CheckIndex` over a merged segment for the first time**: a merged `.si` never listed itself in `SegmentInfo.files` (`Lucene99SegmentInfoFormat.write`'s `si.addFile`) -- a `CheckIndex` failure and an `IndexFileDeleter` reference nothing held, live in every merged segment this port has ever written. **Two enabling MISSING fixes**: the doc-values merge took one field per type and one type per call (six error variants, all removed) so a **multi-tier index sort was inexpressible in a merge**, and a NUMERIC column with a missing value could not be merged at all -- `SortField.setMissingValue`'s normal case, and the reason c17's own fixture was unmergeable. **`execute_merge` now opens doc values, norms and vectors** (closing `c4-merge-fastpath`'s standing carry-over that an automatic merge silently dropped norms and changed every BM25 score), validates the shared sort (`IndexWriter.validateIndexSort`), and reads each tier's keys out of the very column the merged segment will carry; `segment_stats`' `.dvd` and index-sort exclusions are gone, leaving only doc-values *generations*. **PERF**: `copy_chunks` allocated a fresh `ChunkCursor` per call, so a sorted merge decompressed a whole stored-fields chunk **per document** -- 2 004 ms for 4x20 000 documents; with the per-source cursor Java gets free from its reader's cached `BlockState`, **13.2 ms (152x)**. The honest remaining cost of a sort is **13-18x** over the byte-copy path (0.6-1.5 ms -> 10.9-23.2 ms over six runs), which is BULK becoming illegal (a compressed chunk encodes a contiguous run) and landing on c4's DOC path, exactly as Java does. **Evidence**: `verify-write-path.sh` **20 -> 21** (22/22 by the end of the session, another batch having added one) -- the new case runs the **same** `VerifySortedSegment` as the flushed one over a segment merged from eight overlapping sorted flushes with one document in fifty-three deleted, so real Lucene checks `LeafMetaData.sort()`, a permutation it re-derives itself, both doc-values columns (including the sparse one's absences), every unique postings term's doc id, every norm, every vector component, a real `KnnFloatVectorQuery` over the merged graph, that deleted terms are gone from the dictionary, and `CheckIndex.testSort`. **Negative controls as tests, not by hand**: the same three segments with the sort stripped from their `.si` concatenate into a segment that is fully self-consistent and `CheckIndex`-clean (only comparing it against the sorted result shows the loss; stamping the sort onto its `.si` then fails `testSort`), and `the_doc_id_maps_invert_the_doc_order_for_both_merge_orders` pins the one invariant findings 3/4/19 all rest on. **Tier-2 review: three gating findings, all fixed, two of them defects this batch introduced.** (a) An **all-missing** merged NUMERIC column was dropped rather than written -- for a *sort tier* that leaves a `.fnm` with no doc values under a `.si` that declares the sort over them, which real Lucene's `DocValues.getNumeric` throws on (so `testSort` fails rather than degrades) and which this port could then never merge again, propagating `MergeSortColumnMissing` out of every `commit`; Java writes `docsWithFieldOffset = -2` and so does this now. (b) `execute_merge` opened all five doc-values types but wired only **NUMERIC** into the merge sources, so with the `.dvd` exclusion gone a BINARY/SORTED/SORTED_NUMERIC/SORTED_SET column was merged away into a `CheckIndex`-clean segment. (c) The same shape for **positional postings**, whose flush had just landed while `execute_merge`'s filter still said `Docs|DocsAndFreqs`. Plus `has_blocks`, dropped by every merge since b10 (`IndexWriter.mergeMiddle` ORs it across sources; a merged segment reporting `false` silently invalidates every join query), a `numeric_value`-per-document walk in two whole-column loops where `NumericReader`'s own doc comment names a sort as the caller that must not do that, and `merge_segments`' `assert!`s turned into named errors. Every one has a test; the first two were verified to fail with their fix stubbed back out. Coverage: `merge.rs` 98.50%, `merge_policy.rs` 99.06%, `index_writer.rs` 98.25% |
| c29-search-carryovers | the recorded search-crate carry-overs: b14/c12's `FieldExistsQuery` sources + `PointValues.estimateDocCount`, c12's `PhraseHelper` / `OrdinalMap` input / `FacetsConfig.build`, **c14's A1** and **c23's F13** -- search/{doc_value_query,highlighter,facets,field_norms,points_query,ordinal_map,multi_segment,term_vectors_query}, ffi/{segment,registry,sort,facets,range_sort} | swept (5 CORRECTNESS all fixed, 6 MISSING (5 fixed, 1 handed off), 2 PERF (1 measured + handed off, 1 changed and honestly reported as a wash), 4 INTENTIONAL) -- **both wrong-answer items closed with the pre-fix behaviour asserted as a negative control**. c14's A1: after `updateNumericDocValue` the FFI read every doc-values consumer off the *base* column, so a sort returned 100 documents in doc-id order where Lucene returns 60 ranked by the updated values and a `[7000,7000]` range facet counted 0 instead of 30; closed by `SegmentHandle::doc_values_for_field` + the additive `ffi_segment_add_doc_values_generation` (the `.liv` attach pattern -- a generation changes without the segment being rewritten), with **every** doc-values read now resolving the entry and its bytes *together* so the mismatch is unrepresentable, which forced `search_numeric_range_sorted_by_field`/`DocValueSegment` to take separate range/sort buffers (two fields of one segment need not share a column). Found on the way: `ffi_open_segment` cannot open a generational `.fnm` at all (its codec suffix is the generation in base 36). c23's F13: the highlighter read Lucene's UTF-16 offsets as **Unicode scalars** (wrong by one per preceding supplementary-plane character -- an emoji mis-marked every later term) and `offsets_from_analysis` fed it `lucene_analysis`'s **UTF-8 byte** offsets unconverted (wrong for *any* non-ASCII text); the whole module is now Java `char`s, pinned against a real `StandardAnalyzer`'s offsets over text separating all three units (new `GenBreakIterator` section + `highlighter_utf16_offsets_fixtures.rs`, with a negative control asserting the fixture contains a token a scalar reader gets wrong). Also: `FieldExistsQuery`'s norms and vector sources ported (the vector one iterates **ordinals**, so 1 334 steps rather than 4 000 `IndexedDISI` lookups on the sparse fixture field) and c12's "points source" **withdrawn as a misreading** -- 10.5.0's `scorerSupplier` never touches `PointValues`; `PhraseHelper`'s offset half ported and checked against real Lucene's own per-slop match results; `FacetsConfig.build`'s SSDV half ported after establishing it needs **no** `IndexWriter` (it is a pure transformation), checked against eleven real `FacetsConfig.build` configurations, which also makes the existing read-side facet tests non-circular; and `estimate_doc_count`'s arithmetic ported with the planner question answered -- a real estimate changes the plan **only** when another clause leads with fewer than `cost/8` documents, never when this query leads. `OrdinalMap`'s materialized input measured before touching anything: **267 MB of a 319 MB peak** on 5 segments x 1 M terms (the map itself is 52 MB), so the streaming form is worth ~84% and is handed to `lucene-codecs` with the cursor shape spelled out. 34 new tests; per-file coverage 96.85-100% across every file touched. |
| c32-fixture-tooling | the tooling the sweep's evidence base rests on: c29's `gen-fixtures.sh` footgun, c28's `/tmp`-full test-hygiene defect, and whether the two gates are actually run -- `scripts/{gen-fixtures.sh,fixture-segment-ids.py,check-parity.py}`, `fixtures/{README.md,segment-ids.txt}`, `.githooks/pre-commit`, `.github/workflows/ci.yml`, `lucene-util/src/test_support.rs` + 30 `tempdir()` call sites | swept (5 CORRECTNESS all fixed, 3 MISSING all fixed, 1 PERF, 1 INTENTIONAL; **no Java counterpart** -- this is the port's own build/test infrastructure). **The headline is that both gate scripts the sweep added, `check-parity.py` and `check-arith-allows.py`, have never been committed** (`git ls-files scripts/` lists neither, `git log` for both is empty); `.githooks/pre-commit` *is* tracked and calls `check-arith-allows.py` under `set -euo pipefail`, so on any fresh clone the whole pre-commit gate aborts on a missing file *before* `cargo llvm-cov` runs -- that, not c25's stale table, is why the gates were not being run as reliably as assumed. Both are now also wired into CI's `gate` job, where a hook cannot be skipped with `--no-verify`; `setup-hooks.sh` was checked and does install them correctly (it sets `core.hooksPath`). **The commit that lands this must `git add` both scripts** or CI fails on the runner. `gen-fixtures.sh`: a bare full run now **refuses** (needs `--all`), `--only <Gen*>` regenerates one generator plus all six idempotent appenders, and both of c29's damage modes are caught by name rather than as "files differ" -- dropped `Append*Manifest` keys via a per-manifest **key-set** comparison (`blocktree_index/manifest.properties` is *non-deterministic*, measured, so the old byte check was blind to precisely the file c29 damaged: 468 -> 239 keys passed every check), and a changed segment id via `fixtures/segment-ids.txt`, a committed baseline parsed out of each `.si`/`segments_N` index header. Preserving ids across a regeneration was investigated and is **not possible** without patching Lucene, so the refusal plus a readable one-line-per-index diff is the honest fix. Proven by use, twice, against a `sha256sum` of all 684 fixture files: `--only Primitives` left the tree byte-identical (including the appender-rewritten manifest) and `--only Norms` changed exactly its own 17 files. Also found: eight fixture directories are **untracked**, so c29's `git checkout` recovery would not have worked for them at all. Temp dirs: one shared RAII guard in `lucene-util` (the architecture skill's downward-only graph puts it at the bottom; a `lucene-test-support` sibling would be the forbidden edge five times over), gated behind a `test-support` feature on `[dev-dependencies]` edges exactly as `lucene-search`/`lucene-codecs` already do. It removes on `Drop` but **keeps while `thread::panicking()`** -- a failing test's scratch bytes are the evidence for the failure -- asserted by dropping a guard inside a real `catch_unwind`. 30 of 33 sites migrated: a four-crate run leaks **26** dirs against roughly a thousand before, of which 24 are the three files other batches hold (`check_index.rs` 68 sites, `checksum_verify.rs` 3, `fst.rs` 1) and 2 are the deliberate `#[should_panic]` keeps. `verify-write-path.sh` **22/22**; `test_support.rs` 97.92% lines. |
| c33-analysis-offsets | c23's F13 / c29's §2.2 handoff: the **producer** half of the offset unit -- `lucene-analysis/{src/lib.rs,tests/analysis_fixtures.rs}` + `fixtures/src/GenAnalysis.java`, with the consumer compensation and the codec-level consequence chased into `lucene-search/src/highlighter.rs`, `lucene-index/src/indexing_chain.rs` and `lucene-codecs/src/{block_packed,term_vectors}.rs` | swept (12 findings: 5 CORRECTNESS all fixed, 3 MISSING all fixed, 1 PERF measured, 1 INTENTIONAL, 2 recorded with named owners) -- `Token`'s offsets are now **UTF-16 code units** (Java `char`s, `OffsetAttribute`'s own unit) instead of UTF-8 bytes: `tokenize` converts the segmenter's byte indices in one running pass behind a whole-text `is_ascii` fast path, and `Analyzer::keyword` ends its token at the `char` count Java's `KeywordTokenizer` uses (`correctOffset(upto)`). Since c23 those offsets are written verbatim into `.pos`/`.pay`/`.tvd` and `CheckIndex` never compares an offset against the text it indexes, so a Rust-written index of non-ASCII text told real Lucene each term sat where it did not. **The double-fix check found a real compensation**: c29's boundary conversion in `offsets_from_analysis` is deleted (its test now asserts the two sides are *identical*, not merely that the highlight is right), and so is the `char_offsets_to_byte_offsets` reconciliation the fixture test applied to every Lucene-derived expectation -- a compensation inside the one file whose job is to catch this, and itself wrong for astral text. `GenAnalysis` gained **12 `utf16_*` cases** (Latin-1, CJK, combining mark, astral symbol, astral *letter*, emoji, all-units, plus one per filter: keyword, fold, porter, n-gram, edge-n-gram, synonym graph), each with negative controls asserting a byte-offset and (where astral) a scalar-offset producer would fail it; the manifest was regenerated from the real 10.5.0 jars this run and is **identical** to the committed copy. Two findings are new and codec-level: the unit change makes `block_packed::encode_all`'s negative-value path **live** (a term-vector length is `span - prefixLength - suffixLength`, so every multi-byte term is negative -- `caf\u{e9}` is -1), whose doc comment still claimed no caller fed it negatives; and no real-Lucene reader had ever seen a non-ASCII offset this port wrote, so `write_term_vectors_fixture`'s multi-chunk segment gained a non-ASCII document that `VerifyTermVectors` reads back offset-for-offset, with a negative control (the same line rewritten with byte offsets) failing. Also fixed: the interrupted attempt's own `indexing_chain` test asserted a position an untokenized astral symbol does not consume. **Measured** (interleaved A/B, min-of-25, against the pre-c33 producer): ASCII text pays **nothing** (delta negative in all three runs); non-ASCII pays **+0.17 us/doc, +1.8-2.0%** of tokenization and under 1% of the ~21 us/doc indexing cost. Coverage: `lucene-analysis/src/lib.rs` 99.27%. Recorded: real `StandardTokenizer` emits an emoji token this port does not (b8's F40, the `utf16_emoji` case pins the exact shape of the gap), and `IndexingChain`'s `invertState.offset` multi-value accumulation is unported but unreachable (`IndexWriter` indexes only a field's first value) |
| c30-finish-index | the last two open `lucene-index` items: c25's residual `CheckIndex` failure arms, and the workspace's last three `TODO(arith-audit)` modules -- index/{check_index, checksum_verify, index_writer, merge, merge_policy}, cross-batch codecs/vectors, ffi/writer | swept (7 CORRECTNESS all fixed, 1 defensive, 1 arm deleted, 5 kept as unreachable *error handling* with the proof at the site) -- **`check_index.rs` 97.25% -> 98.58%** and **`docs/arithmetic-gate.md` 3 -> none: the whole workspace is now audited**. Item 1 closed c25's last open D-list question in both directions and drew the line c25's rule does not: *a `Check::fail` arm that cannot fire is a false claim of coverage; an `Err(e) =>` arm that cannot fire is total error handling* -- one arm deleted (`connected_nodes_on_level`'s entry-point guard, proved unreachable because `OffHeapHnswGraph::new` rejects the only shape that could produce it), five kept with proofs. Its own negative control found a `.vemf` region offset slicing the `.vec` behind `if a + b > len` where `a` came off disk -- the gate doc's named shape, a panic on the KNN query path. Item 2's two headline findings **both came from the hand-check of indexing/slicing/allocation, not from the lint**: `execute_merge` took `maxDoc` from the `.fdm` where Java's `SegmentMerger` takes it from the `.si`, and `stored_fields::open` checks only that it is non-negative -- a four-byte edit sizes two per-source `Vec`s at ~8.6 GB each, **verified SIGABRT** (`memory allocation of 4294967296 bytes failed`, `signal: 6`) under `ulimit -v`; and `merge_segments` indexed the `.liv` `FixedBitSet` with a bound off the `.fdm`, c28's crate rule for the third module running. `build_doc_id_maps` even carried an `// ARITH:` proof claiming that `max_doc` was "already-validated" -- the c19 failure mode (right conclusion, wrong reasoning), now naming the caller that supplies the bound. Three more in `merge_policy`, all from `as i64` on a `pub` unsigned config: `usize::MAX as i64 == -1` made `max_allowed_docs` return **-100** where Java's `Math.ceilDiv` returns 1 (a negative budget every merge exceeds, so the policy silently stops merging), and a `size_bytes` above `i64::MAX` became a *negative* byte count -- the largest segment in the index packed into every merge as if it cost nothing, and two of them overflowing `bytes_this_merge + seg_bytes` outright. Plus the norms field length wrapping a `u32` where `IndexingChain.PerField.invert` uses `Math.addExact` (a wrapped length encodes to a *small* norm: the longest document scoring as one of the shortest, in every BM25 query over the field). Every fix has a test that fails against the unfixed code except the last, whose reachable input needs a ~2 GB field value (c28-F7's precedent, stated rather than glossed). Verifier runtime **0.18 s** in release (c25 0.23, c19 0.28); `index-bench` **21.53 us/doc** and the stored-fields BULK merge **532x**, both unchanged; `verify-write-path.sh` **22/22**; **`scripts/docker-test.sh gate` -> `gate: ok`**, workspace lines 98.10%, no file below 95%. Answered a carry-over by measuring rather than assuming: `--lib --tests` gives *identical* `check_index.rs` coverage to `--lib`, so c23's integration tests were not silently covering the residual arms after all |
| c36-merge-metadata | the last two wrong-answer findings (**1** `MergeSource`'s missing `min_version`/`has_blocks`, **2** `SegmentCommitInfo.sci_id` never regenerated) plus the two adjacent items in the same files (**11** zero-doc merges committed, **19** the `.si` rewritten five times per commit) -- index/{merge, index_writer, segment_infos, segment_writer, deletes}, new `GenMergeMetadata`/`VerifyMergedMetadata`/`write_merged_metadata_fixture` | swept (7 findings: 2 CORRECTNESS both fixed, 2 MISSING (1 fixed, 1 recorded with its blocker), 2 PERF (1 fixed, 1 recorded and now unblocked), 1 hygiene fixed) -- **`minVersion` was the live half of item 1**: the merged `.si` claimed the merging writer's version where `SegmentMerger`'s constructor takes the minimum over the readers (`null` if any reader has none); `has_blocks` was already correct via c22's `MergeOptions` and has been **moved onto `MergeSource`** so the two halves of Java's `LeafMetaData` record travel together -- 85 exhaustive struct literals, exactly c34's estimate. **The `sci_id` is a change token**: all four of Java's generation-advancing mutators call `generationAdvanced() -> id = randomId()`, and none of this port's did, so two commits of the same segment reported the same id and every NRT/replication/cache consumer keyed on it was told "unchanged" across a delete or a doc-values update; `deletes::apply_deletes` was the live site (it hand-rebuilt the `SegmentCommitInfo` and carried `sci_id` across) and now mutates through `advance_del_gen`. **Zero-doc merges** now take `mergeMiddle`'s `shouldMerge() == false` path -- the merge is skipped before a single file is written and the new `IndexWriter::drop_merge` (`commitMerge` with `dropSegment`) retires the sources with nothing published. **The `.si` is written once**, from an in-memory `SegmentInfo` (`segment_writer::FlushedSegment` + `seal_flushed_segment` = `sealFlushedSegment`), replacing up to **seven** writes / six `open`+`parse` round trips / seven fsyncs per commit; the wall-time saving is below this host's noise floor (`fsync` is effectively free on the container mount -- 20 000 extra read-modify-write cycles per flush produced no separation), so the result is stated as the I/O it removes, asserted by a counting `Directory` rather than timed. **Evidence**: `verify-write-path.sh` **22 -> 23** -- `GenMergeMetadata` writes three *real Lucene* segments whose `.si` files record `minVersion` 10.2.0/**10.0.0**/10.1.0 (oldest in the middle) with `hasBlocks` set by Lucene itself on one of them, this port merges them, and real Lucene reads both fields back off `LeafMetaData`; verified to fail both assertions against the unfixed code. Coverage: `merge.rs` 98.70%, `index_writer.rs` 98.37%, `segment_infos.rs` 98.50%, `segment_writer.rs` 99.49%, `deletes.rs` 98.48%; workspace 98.11% lines, no file below 95% |

## Open work, prioritised (reconciled by `c34-ledger-reconcile`)

**This is the list to plan from.** Everything below it in this file is the
historical record: 68 items were carrying an unticked `- [ ]` box when c34
started, and each was re-verified against the tree -- reading the code,
grepping the symbol, running the test -- rather than against the batch report
that raised it. **29 were already done and never ticked**, **4 are obsolete**,
and **35 boxes were genuinely open** -- **33 distinct findings**, since two
findings each have two boxes (the term-vector field-order item, and the
filter-only pruning item). Two of the 33 were trivially closable and c34 closed
them, leaving **31 distinct open findings** in 32 boxes, all listed below.
Every historical entry now carries
a note naming what closed it and the evidence, or -- where it is still open --
what the current tree actually looks like, because several described a world
two batches out of date.

**`c35-norms-and-sort` closed items 3 and 4** (the two largest B-tier
entries), leaving **29 distinct open findings**. **`c36-merge-metadata` then
closed items 1, 2, 11 and 19** -- the whole of tier A -- and raised two new
ones it met on the way (**11b**, 100%-deleted segments not dropped;
**25b**, the deleter re-reading the `.si`; and **26b**, two checks that are not
in the gate and had both gone quietly red), leaving **28**.
**`c37-search-behaviours` then closed items 6, 7, 23 and 29 outright, plus two
of item 5's four sub-entries** (`searchAfter`/`MaxScoreAccumulator` and
`Weight.count`), leaving **24**. Two of the four it closed had been recorded as
*blocked on a fixture* that already existed -- see item 29 -- so the batch's
most transferable finding is procedural, not technical. Closed entries are
struck through in place rather than deleted, and new ones take a lettered
suffix, so the numbering below stays stable.

**Tier A is empty.** Every wrong answer this sweep found on a path this port
exposes is now fixed.

Ordering below is by what a user can observe, not by effort:
**(A) wrong-answer bugs**, **(B) missing Lucene behaviour a caller can reach**,
**(C) performance/memory divergence**, then **(D) tooling and hygiene**.

### A. Wrong answers a caller can reach

**Both entries are closed** (`c36-merge-metadata`). They are kept below with
what closed them, because the reasoning c34 recorded here -- that neither was a
*silent* wrong answer on a path this port currently exposes, that both were
wrong-answer-shaped and gated by something -- is what made them takeable as one
batch, and because the "latent" one turned out to be only half-broken, which is
worth knowing about the next entry that reads the same way.

1. ~~**`MergeSource` carries no per-source `min_version`/`has_blocks`.**~~
   **CLOSED by `c36-merge-metadata`.** `MergeSource` now carries both fields of
   Java's `LeafMetaData`, and `merge_segments` folds them exactly as
   `SegmentMerger`'s constructor and `IndexWriter.mergeMiddle` do
   (`merged_min_version`: minimum over the sources, `None` if any source has
   none, seeded with the writer's version; `has_blocks`: the disjunction).
   `MergeOptions::has_blocks` is gone. The `has_blocks` half was in fact
   **already correct** -- c22 finding 24 had put the disjunction in
   `execute_merge` -- so only `minVersion` was a live wrong answer; the move
   was made anyway, because keeping the two halves of one Java record in two
   different parameters is how the `minVersion` half stayed missing.
   Evidence in both directions: `fixtures/data/merge_metadata/` is three
   segments a real `IndexWriter` wrote, carrying `minVersion` 10.2.0/10.0.0/
   10.1.0 and `hasBlocks` on one of them, which this port merges and real
   Lucene reads back through `LeafMetaData` (`verify-write-path.sh` case 23,
   verified to fail on both fields against the unfixed code).

2. ~~**`SegmentCommitInfo.sci_id` is not regenerated when a generation
   advances.**~~ **CLOSED by `c36-merge-metadata`.**
   `SegmentCommitInfo::generation_advanced` is Java's `generationAdvanced()`
   and is called from all four mutators; `advanceNextWrite*Gen` correctly still
   is not. The live site was `deletes::apply_deletes`, which hand-rebuilt the
   `SegmentCommitInfo` with `sci_id: sci.sci_id` and now mutates the real one
   through `advance_del_gen`. The id is **derived, not random** -- a hash of
   `(segment_id, del_gen, field_infos_gen, doc_values_gen,
   buffered_deletes_gen)` -- which gives the only property a consumer reads it
   for ("different iff the segment-commit is different") *better* than a random
   draw does, since two runs reaching the same commit state produce the same
   token instead of a spurious "changed". **What the id is used for here**:
   nothing reads it inside this port -- `segment_infos::parse`/`write` carry it
   and no validator touches it. Its consumers are all outside: Java's
   `DirectoryReader.openIfChanged` per-segment reuse, an NRT/replication
   client's "must I re-fetch this segment", and any cache keyed on it. That is
   precisely why the defect was invisible and why the fix has to be checked by
   asserting the id *changes*, which five new tests do.

### B. Missing Lucene behaviour a caller can reach

3. ~~**`segment_info::IndexSortField` cannot represent most real Lucene
   sorts.**~~ **CLOSED by `c35-norms-and-sort`.** `IndexSortField` is now
   `(field, reverse, IndexSortKind)` covering all four `SortFieldProvider`s,
   every `SortField.Type` that can be an index sort, both selector enums and
   every missing-value form including "none"; `write` is `parse`'s byte-level
   inverse for all of them. Evidence both directions:
   `fixtures/data/sorted_index_wide/` (a real `IndexWriter` index sorted by
   `LONG` descending with missing value 42, `SortedNumericSortField` MAX with
   no missing value, and `STRING` by ordinal) is opened, ordered and
   `CheckIndex`-ed by this port; `write_segment_info_fixture`'s `_3` writes
   all four providers and `VerifySegmentInfo` compares real Lucene's
   `Sort.toString()`. The *writers* stay narrower on purpose and say so
   (`Error::UnsupportedIndexSortKind` for ordinal/byte sorts) -- see
   `docs/sweep/m2/c35-norms-and-sort.md`.

4. ~~**Norms are opt-in per field.**~~ **CLOSED by `c35-norms-and-sort`.**
   Norms follow `IndexingChain.writeNorms` exactly: every indexed field whose
   `omitNorms` is false gets a column, with no opt-in anywhere, and
   `IndexWriter::omit_norms_field` is the opt-out. The column is sparse, as
   `NormValuesWriter`'s is. Verified by real Lucene reading back every
   document's norm for a field nothing configured
   (`VerifyFullSegment.checkNorms`). Measured: +0.65 us/doc (19.85 vs 19.19),
   and one byte per document per normed field in the `.nvd` (zero when every
   document's length is equal -- the constant encoding).

5. **`Occur`-shaped search primitives that do not exist.** Four entries, one
   milestone:
   - **`TwoPhaseIterator`/`matchCost`/`ScorerSupplier.cost()`** -- no
     cheap-approximation-then-verify split anywhere, so a conjunction verifies
     an expensive phrase clause on every candidate and clause order is the
     caller's problem. Needs every clause expressed as
     `(approximation, matches(), matchCost())`, i.e. turning the per-shape free
     functions into a scorer enum. c6 and c11 both assessed it: **a milestone,
     not a batch.** The contained part -- ordering conjunction clauses by
     `docFreq` ascending -- is batch-sized on its own.
   - ~~**`searchAfter` and `MaxScoreAccumulator`**~~ **CLOSED by
     `c37-search-behaviours`.** `TopDocsCollector::with_after` is
     `TopScoreDocCollector`'s own test (`score > afterScore || (score ==
     afterScore && doc <= afterDoc)`, applied after `totalHits` is
     incremented); `collector::MaxScoreAccumulator` is one `AtomicI64` folded
     with `fetch_max` over `DocScoreEncoder`'s packing, and lives where the
     fan-out does
     (`multi_segment::merge_multi_segment_scored_concurrent_shared_max_score`).
     Java's `modInterval` is deliberately not modelled: it exists to keep an
     atomic read off a *push* path, and this port pulls the threshold once per
     block. Verified against three real `IndexSearcher.searchAfter` pages on
     both a single-segment and a two-segment fixture
     (`AppendSearchAfterManifest.java`).
   - ~~**`Weight.count(LeafReaderContext)`**~~ **CLOSED by
     `c37-search-behaviours`.** `weight_count::{count_term_query,
     count_match_all_docs, count_field_exists_leaf}` plus
     `ffi_count_term_query`. Measured on the 5M-document corpus: a count that
     took **13.93 ms** of postings walk takes **72.2 ns**. Ground truth from
     real `IndexSearcher.count` on both a deletion-free index and one with a
     `.liv` (`AppendCountManifest.java`), because the deletions gate is the
     only part that can be silently wrong.
   - **Only one `Similarity`** (BM25), with no `Similarity`/`SimScorer` trait:
     `TFIDF`/`Classic`/`Boolean`/`LMDirichlet`/`IndependenceStandardized`,
     `k3` and `computeQueryTermWeight` are all unported. b12 already split BM25
     into `idf`/`norm_inverse`/`do_score`, which are the three pieces the trait
     needs.

6. ~~**`FieldExistsQuery.count` and `rewrite`'s whole-reader decision.**~~
   **CLOSED by `c37-search-behaviours`.** `weight_count::FieldExistsLeaf` is
   the "leaf list" the entry said was missing -- the counts Java reads off a
   `LeafReader`, gathered by `SegmentReader::field_exists_leaf` -- and
   `count_field_exists_leaf`/`field_exists_rewrites_to_match_all_docs` are the
   two rules over it. No reader-level query object was needed: the rules are
   pure functions of the counts, and the reader only has to produce them.
   Java's norms asymmetry (top-level `getDocCount(field)`/`maxDoc()` inside a
   per-leaf loop) is reproduced, with the two reader-wide values as explicit
   parameters so a caller cannot pass the leaf's by accident. Ground truth over
   five committed indexes, covering the complete-norms, partial-norms,
   no-doc-count-available and skipper arms
   (`AppendCountManifest.java`); the live-doc arithmetic was verified to fail
   with `numDocs` replaced by `maxDoc`.

7. ~~**Sloppy phrase matching is in-order only.**~~ **CLOSED by
   `c37-search-behaviours`.** `lucene-search/src/sloppy_phrase.rs` is
   `SloppyPhraseMatcher` ported statement for statement -- the shifted-position
   window, the `PhraseQueue` walk that emits the `matchLength` *sequence*
   `PhraseScorer.score()` sums, and the `rptGroups` machinery for repeated
   terms including the `hasMultiTermRpts` union a `MultiPhraseQuery` can reach.
   *The recorded blocker was false*: `GenBlockTree` has had a reordered pair
   since task #55 (doc 8558, `delta`@0 `gamma`@1, added for the
   `SpanNearQuery` `in_order` test), and doc 8555's `alpha beta` is a reordered
   document for the query `"beta alpha"`. No index was regenerated; the ground
   truth is six new `AppendScoringManifest` entries recorded against the
   committed segment, all matched bit-for-bit on the first run, and verified to
   fail (zero hits where Lucene has one) against the in-order matcher.
   **One half remains**: `highlighter::phrase_match_offsets`' match enumeration
   is still in-order, so a reordered occurrence is scored but is not offered a
   highlight fragment. That is now the only in-order-only phrase path, and it
   is recorded on the function.

7b. **The highlighter's phrase match enumeration is still in-order only.**
    `highlighter::phrase_match_offsets` walks one greedy in-order alignment per
    starting position of the first slot; with item 7 closed it is now the only
    in-order-only phrase path in the crate, so a reordered occurrence is
    *scored* (`sloppy_phrase`) but is not offered a highlight fragment -- the
    two halves of one query disagree about what matched. *Cost*: a missing
    highlight, never a wrong hit set. *What it needs*: `phrase_match_offsets`
    to enumerate through `sloppy_phrase`'s walk, which means the matcher
    growing a "what were the raw positions of this match" accessor
    (`SloppyPhraseMatcher.startPosition()`/`endPosition()`/`startOffset()`/
    `endOffset()`, all of which Java has and this port skipped because nothing
    consumed them). Contained; roughly the size of item 7's second half.
    (Raised by c37.)

8. **`FieldInfo` is a plain struct where Java's is a validating constructor.**
   `field_infos::write` applies the one coercion that was producing unreopenable
   files, and `check_consistency` exists, but a caller can still build
   combinations Java makes unrepresentable and find out at `parse` time or not
   at all. A `FieldInfo::new -> Result` closes the class; it touches ~197
   construction sites.

9. **`FuzzyTermsEnum`'s `MaxNonCompetitiveBoostAttribute` feedback loop** (swap
   to a lower-edit automaton once the top-terms queue is full) is unported, and
   `fuzzy_doc_scores` blends `docFreq` within one segment where
   `BlendedTermQuery` blends across the whole reader -- the fuzzy clause has no
   `GlobalStats` plumbing.

10. **`lucene-analysis` has no `TokenStream` lifecycle.** `end()`'s trailing
    position increment is dropped by `StopFilter` and both n-gram filters, so a
    document whose last tokens were all filtered out does not advance the
    position counter -- **the one gap here a caller can observe as a wrong
    position**. Alongside it: no case-insensitive `CharArraySet` (the port
    matches a lowercase set against already-lowercased terms, right for the
    standard chain, wrong for a caller-supplied mixed-case set) and no
    `maxTokenLength`. The UTF-16-offsets gap that used to lead this entry is
    **closed** (c29 read side, c33/`tokenize` write side).

11. ~~**Zero-doc merges are committed rather than dropped.**~~
    **CLOSED by `c36-merge-metadata`.** `execute_merge` sums the sources' live
    document counts right after their stored-fields readers open -- where
    `mergeMiddle`'s `if (merger.shouldMerge()) merger.merge();` sits -- and on
    zero calls the new `IndexWriter::drop_merge`, which is `commitMerge` with
    `dropSegment` set: `SegmentInfos.applyMergeChanges`' "remove every
    merged-away source, insert nothing". No file is written, so there is
    nothing for Java's `deleteNewFiles(merge.info.files())` to do on this path.
    `apply_merge` and `drop_merge` share one private `commit_merge`; the public
    API is unchanged.

11b. **100%-deleted segments are not dropped when deletes are applied.**
    `IndexWriter.finishApply` drops them
    (`closeSegmentStates` collects a segment iff `rld.isFullyDeleted()` --
    hard deletes only, `getDelCount() == maxDoc()` -- and
    `MergePolicy.keepFullyDeletedSegment` is false; then
    `dropDeletedSegment` + `checkpoint`). This port keeps them in the commit
    forever. *Cost*: a segment nothing can ever match, carried by every later
    open, merge and `CheckIndex` -- the same cost as item 11's zero-doc merge,
    one step earlier. *Blocked on*: three things this port does not have.
    (a) `MergePolicyConfig` has no `keepFullyDeletedSegment` hook, and it is
    not decoration -- `SoftDeletesRetentionMergePolicy` returns `true` from it,
    so a drop without the hook is only correct for the default policy.
    (b) There is no `adjustPendingNumDocs`/reader-pool bookkeeping to update.
    (c) At least one existing test
    (`a_rollback_after_a_buffered_delete_was_applied_restores_the_committed_segment_list`)
    asserts on `segments[0]` *after* deleting that segment's only document, so
    the change has an observable ripple that needs its own reasoning rather
    than a mechanical edit. Met and recorded by c36 while building item 11's
    test, which had to hand-build fully-deleted segments precisely because of
    this. (Raised by c36.)

12. **`Util.shortestPaths`/`TopNSearcher`/`readCeilArc` unported**, so
    `top_n_completions` walks the prefix subtree with a bounded heap and cannot
    skip a subtree that provably cannot beat the current worst candidate.
    `SegmentTermsEnum` `TermState` seeking is likewise unported -- no
    `TermStates` caller exists yet.

13. **`PointValues.estimatePointCount`'s BKD walk.** `estimate_doc_count`'s
    arithmetic is ported and is what `IndexOrDocValuesQuery`'s planner consumes,
    but the walk that produces its input is not: `IntersectVisitor::compare`
    sees cell bounds but no subtree size, and the node-id walk is private.
    *Owner*: `lucene-codecs` -- a `PointsReader::estimate_point_count(field,
    &mut V)` beside `intersect`, reusing `IntersectCtx`/`intersect_node`'s
    `node_id` bookkeeping plus `BKDReader.IndexTree.size()`. Nothing in
    `lucene-search` moves.

### C. Performance and memory divergence

14. **`DirectoryReader::open` is still the largest reader-side gap.**
    `verdict-m1.6.md`: **52.7 ms** on the merged corpus, ~155x Lucene, RSS
    70 MB. c1's "2.0 ms of 2.2 ms is `open_segments`" diagnosis predates c12's
    4.8x and the mmap change and must be re-measured before it is planned
    against -- `micro reader_open` builds and runs again, so that is cheap.

15. **`indexing_chain` allocates per token/term/posting** where Java uses
    `BytesRefHash` + `ByteBlockPool`/`IntBlockPool`/`ByteSlicePool`. Measured
    by c3: 8.3 MB of document text becomes 78.5 MB of `InMemoryInvertedIndex`
    (**9.4x**), and the single largest term is the `Vec<Occurrence>` whose
    first `push` reserves capacity 4 -- 48 bytes of allocation for 12 bytes of
    payload. `shrink_to_fit` was tried and reverted (structure drops to 5.98x,
    indexing costs 25-60% more, peak RSS does not move: glibc keeps the freed
    chunks). *Contained version*: an inline-capacity-1 occurrence
    representation, which needs `PostingEntry`'s public shape to change.
    This is also what keeps `ram_bytes_used()` measuring the arena rather than
    Java's inverted-form bytes.

16. **Payload slots cost ~26 us/doc and ~190 MB per 50 000 documents, and the
    cost is the slot, not the bytes** (c23 F9, measured with an
    all-empty-payload control). The fix is a flat `(bytes, lengths)`
    representation in **both** `postings_writer::TermPostings::payloads`
    (still `Vec<Vec<Vec<u8>>>`, `postings_writer.rs:337`) and
    `indexing_chain::PostingEntry::payloads` (still `Vec<Vec<u8>>`); doing only
    the second is *slower*, because `build_postings_output` would
    re-materialize the nested form. An instance of item 15 with a number on it.

17. **`OrdinalMap::build` materializes every segment's term list.** Java streams
    `TermsEnum`s and never holds a dictionary. Measured by c29
    (`lucene-search/examples/ordinal_map_memory.rs`, 17-byte terms): 5 segments
    x 1 M terms costs **267 MB** for the term lists against **52 MB** for the
    map, 319 MB peak -- the input is ~5x the output and ~84% of the peak.
    *Blocked on*: `lucene-codecs/src/terms_dict.rs` exposing a `TermsCursor`
    that yields one term at a time over the prefix-compressed blocks
    `decode_all_terms` already walks; then `facets.rs` and `OrdinalMap::build`
    take iterators.

18. **Impacts are computed against norm 1.** `FieldPostingsInput` carries no
    norms, so every level-0 and level-1 impact this writer emits is
    `(maxFreq, 1)` where `Lucene104PostingsWriter` feeds real per-doc norms
    into `CompetitiveImpactAccumulator`. **Sound but loose** -- norm 1 is the
    highest-scoring norm, so it costs pruning, never a wrong answer. The
    `lucene-index` half is unblocked (one shared invert pass already computes
    every document's field length at the `write_fields` call site); what
    remains is entirely in `lucene-codecs`.

19. ~~**`IndexWriter` rewrites the `.si` once per file group -- five times, not
    four.**~~ **CLOSED by `c36-merge-metadata`.** It was seven, counting the
    stored-fields flush's own write and `write_index_sort_to_si`.
    `segment_writer` now splits into `write_stored_only_segment_files` (every
    codec file, no `.si`) + `seal_flushed_segment` (`sealFlushedSegment`'s
    single `.si` write and one fsync of the whole set), carrying a
    `FlushedSegment` -- the in-memory `SegmentInfo` -- in between. The five
    `write_*_files` helpers return their file names instead of patching the
    file; `write_index_sort_to_si` is deleted; `flush_sorted_stored_only_segment`
    lost its own second write. The recorded blocker ("a signature change to
    `flush_stored_only_segment`, which `merge.rs` also calls") did not
    materialise: `flush_stored_only_segment{,_with_blocks}` keep their exact
    signatures and are now build + seal. *Measured*: **below this host's noise
    floor** -- `fsync` is effectively free on the container's mount, so even
    20 000 extra read-modify-write-then-resync cycles per flush produced no
    separation (see the batch report's table). What the change removes is
    countable instead of timed, and is what matters on a filesystem that
    honours `fsync`: 4 writes, 4 fsyncs and 4 `open`+`parse` round trips per
    flushed segment in the configuration the new test drives, 6 of each in the
    maximum case. Asserted by `one_commit_writes_the_segments_si_exactly_once`,
    which runs the writer through a counting `Directory`.

20. **Migrate the blocktree lookups to their `try_*` forms** (c1 F-11). A
    corrupt `.tim` block is discovered at lookup time, as in Java, and
    `try_seek_exact`/`try_next`/`try_seek_ceil`/`try_current` exist and are
    tested -- but the infallible spellings are still what 136 call sites use
    (against 61 `try_*` uses), and they degrade a corrupt block to "no such
    term"/end-of-terms. *Cost*: a corrupt index reads as an empty one.

21. **Split term iteration from stats in `TermsEnum`** (c1 F-14). Java's
    `next()` decodes only the term bytes and defers `decodeMetaData` to
    `docFreq()`; `blocktree.rs:1780`'s `next()` returns `(&[u8], TermStats)`
    and so always decodes. Full-field enumeration is 27 ns/term against
    Lucene's 20.5 ns. Wants a `next_term()` + `stats()` split, which changes
    `check_index`'s and the intersect iterators' call shape.

22. **`highlighter`/postings read path aside**: `IndexedDISI`'s **block jump
    table** is still not read (`createJumpTable`/`advanceBlock`'s
    two-blocks-ahead shortcut). Cost is O(maxDoc/65536) four-byte header reads
    -- 16 for a million documents -- and this port's writers emit
    `jumpTableEntryCount = 0`/`-1`, so our own files have no table to read.
    Worth revisiting only alongside a `nextDoc`-shaped iterator API.

23. ~~**A filter-only query cannot prune under a top-`n` collector.**~~
    **CLOSED by `c37-search-behaviours`.** The guard was removable, and the
    reasoning behind it was wrong rather than merely cautious: Java skips the
    tie too. `TopScoreDocCollector.updateMinCompetitiveScore` publishes
    `Math.nextUp(topScore)`, which for a bottom of `0f` is `Float.MIN_VALUE`,
    and the scorers skip a block whose maximum is `< minCompetitiveScore` -- so
    Java's `bound < nextUp(bottom)` and this port's `bound <= bottom` are the
    same rule for every finite bottom. It is also independently correct:
    documents arrive ascending and `HitQueue` gives a score tie to the lower
    doc id, so nothing skipped could have entered a full queue. *The recorded
    blocker was half false*: `body` is indeed too small (its two-document
    postings live in the vint tail, where `current_block_last_doc_id()` is
    still `-1`, so no bound exists to prune against), but the same fixture's
    `big` field has 300 documents and fills a top-20 queue fifteen times over.
    **Measured, both sides re-run in one session**: `#body:t0 #body:t1` at
    top-50 on `benchmarks/.corpus/merged` went **44.20 ms -> 2.00 ms** (22x),
    against 7.02 ms for the all-`MUST` form -- so the filter shape is now the
    cheaper one under a top-`n` collector, as it already was without one.
    (Two ledger entries, one finding.)

24. **`StoredFieldsReader::document()` materializes a whole `Document`** where
    Java's `StoredFieldVisitor` lets a caller take one field and skip the rest.
    Read path only; the write side was made streaming by c4.

25. **DEFLATE encoder has no preset dictionary.** `miniz_oxide` exposes no
    `deflateSetDictionary`. Compression ratio only -- the decode side is
    correct and fixture-verified. *Blocked on a dependency*: revisit if a
    raw-deflate crate with dictionary support becomes acceptable.

25b. **`IndexFileDeleter`'s checkpoint re-reads and re-parses every segment's
    `.si`.** Java reference-counts from the in-memory
    `SegmentCommitInfo.files()`, which already holds the `SegmentInfo`; this
    port re-opens and re-parses the file, once per segment per checkpoint. It
    is the one `.si` read c36 left in the flush path (that batch's counting
    test pins it at exactly one, so a regression is visible). *Blocked on*:
    handing the deleter live `SegmentInfo`s rather than segment *names* -- a
    signature change across `index_file_deleter.rs` and every checkpoint call
    site. **Now unblocked in one respect**: `segment_writer::FlushedSegment` is
    exactly the in-memory `SegmentInfo` that call would take. (Raised by c36.)

### D. Tooling, gates and tidy-ups

26. **A rustdoc pass belongs in the gate.** `rustdoc::broken_intra_doc_links`
    is warn-by-default and caught by none of `cargo fmt`/`clippy`/`test`/
    `llvm-cov`; c4 shipped a broken link that survived a green gate. Turning it
    on needs the pre-existing broken links cleaned up first.

26b. **Two checks that exist but are not in the gate, and had both gone
    quietly red.** c36 found `scripts/gen-fixtures.sh --check` failing on a
    committed fixture generated outside the container (`break_iterator`'s
    manifest recorded `java_version=25` against the pinned JDK 21) and
    `benchmarks/rust-runner` not compiling at all (two half-finished API
    reshapes from an earlier batch), with a stale binary in
    `target-docker/release/` still producing plausible benchmark numbers from
    pre-change code. Both are fixed, but neither is *gated*: the benchmark
    crate is outside the workspace so `clippy --workspace` never sees it, and
    `--check` is run by hand. **The benchmark half is now closed**: c36 added
    `cargo check --manifest-path benchmarks/rust-runner/Cargo.toml
    --all-targets` to `scripts/gate.sh` (and to AGENTS.md's table), so the next
    API reshape fails the gate instead of producing plausible numbers from a
    stale binary. What is left is `gen-fixtures.sh --check`, which is expensive
    (it generates the whole tree twice) and probably belongs on a schedule
    rather than per-push. (Raised and half-closed by c36.)

27. **Mechanical gates for three defect shapes this sweep keeps re-finding.**
    (a) a clippy `disallowed_methods` entry on the free
    `doc_values::numeric_value`/`binary_value`, naming
    `NumericReader`/`BinaryReader` as the sanctioned multi-lookup API -- the
    "call the re-deriving free function once per document" defect has appeared
    twice (b13's `soft_deletes::effective_live_docs`, c14's column merge) and
    is grep-able; (b) a gate on `Lucene90_\d`-shaped string literals outside
    `index_writer::per_field_codec_suffix` and test modules, which is how c14's
    hardcoded per-field suffix got in; (c) for every `fn`/`struct` a diff
    *removes*, grep `crates/`, `docs/parity.md` and `PLAN.md` (excluding
    `docs/sweep/`, which is an archive) and fail on a hit -- c37's Tier-2 review
    found `parity.md` describing two deleted functions in the present tense,
    and `check-parity.py` deliberately validates only the file *paths* in the
    Rust column, never identifiers in the prose. No false positives for
    `pub(crate)`-and-above symbols. (Raised by c37.) There is no `clippy.toml` in the tree
    today, so (a) starts from zero. Repo-wide tooling, not any one batch's
    files.

28. **`norms::parse_meta`'s signature still differs from Java's
    `readFields(meta, infos)`.** *Behaviour* is closed (c15's additive
    `norms::validate_fields`, called from the segment reader's open and from
    `check_index`). What is left is a pure tidy-up across 34 call sites in four
    crates, and c7's reasoning still holds: the right moment is when a shared
    `FieldInfos` open lands, at which point the parameter is free instead of
    34 mechanical edits that refactor would rewrite. **Do not schedule this on
    its own.**

29. ~~**`lucene-analysis`/`lucene-search` fixture debt for items 7 and 23.**~~
    **CLOSED by `c37-search-behaviours`, and worth reading as a cautionary
    entry.** Neither fixture had to be built: `GenBlockTree` already contained
    a reordered pair (doc 8558, added by task #55 for `SpanNearQuery`) and a
    300-document field (`big`, added for the postings block tests). Both items
    had been carrying a "blocked on a fixture" note for several batches while
    the fixture sat in the tree, having been added for a different purpose.
    The same shape as c34's four "actively misleading" entries -- a blocker
    that stopped being true and nothing noticed, because the thing that made it
    untrue was somebody else's work.

### What was actively misleading (the class that cost c26 and c29 time)

Four entries claimed a blocker that no longer existed, or described as
structural something that had since become local. Each is annotated in place;
listed together here because the pattern, not the individual entry, is the
thing to watch:

- **Norms opt-in** said "blocked on a multi-field `.nvd`/`.nvm` writer". c26
  wrote one. The item had been unschedulable for a reason that had been false
  for several batches.
- **`Lucene104PostingsWriter` impacts** was read alongside a
  `postings_writer.rs` module doc asserting "impacts are always an empty byte
  region" and that positions can never co-occur with a full block. Both were
  made untrue by c20/c23 and neither was updated; c34 corrected the module doc.
- **Stored-fields writer API** said the writer "takes `&[Document]` rather than
  streaming". c4 made it a streaming object; only the *reader* half was left.
- **`docs/parity.md`'s FFI row** still described `open_field_norms` as ending
  in `FieldNorms::open` -- the production-path scoring divergence b12 raised as
  F-7 -- after b15 had moved it onto `from_field_stats`. The code was right and
  the record was the last thing still claiming otherwise. Corrected by c34.

Two more were stale rather than misleading and are worth the same watchfulness:
the `DirectoryReader::open` entry carried c1-era numbers three batches out of
date, and the `.si`-rewrite entry undercounted its own cost (four file groups,
now five).

## Carry-over items (assign to a later batch)

> **Historical record from here down.** Plan from "Open work, prioritised"
> above instead. Every box below was re-verified by `c34-ledger-reconcile`
> against the tree, not against the report that raised it: `- [x]` is done
> and carries a note naming what closed it and the evidence checked, `- [~]`
> is **obsolete** (the code it referred to is gone, or a later decision
> superseded it) and says why, and `- [ ]` is genuinely open and carries a
> note on what the current tree actually looks like wherever the original
> entry had gone stale.

- [x] **`search_term_query_scored_maxscore_with_stats` drops its reader-wide
      statistics on every fallback path** (`lib.rs`). **Fixed by c6**; b13's
      caller-side guard deleted and its tripwire inverted. All three fallback
      `return`s call the no-stats `search_term_query_scored`, so a leaf with no
      `.doc` input or a `docFreq <= 1` term silently scores from its own
      `docFreq`/`docCount`. On a two-segment index where a term is in 1 of 1
      documents in one leaf and 1 of 4 in the other, that is idf 0.288 vs 1.204
      against the reader-wide 0.876. Worked around at b13's caller
      (`multi_segment::maxscore_keeps_global_stats`); the fix is to forward
      `global` to `search_term_query_scored_with_stats` instead. Owner: b12.
      (Raised by b13, F-5.)
- [x] **`avgFieldLength` is not Java's at the constructor every caller uses.**
      *(b13 done on the `field_norms.rs` side: `FieldNorms::open`'s doc comment
      now enumerates all three divergences and every caller in the workspace
      passes `live_docs == None`, so only the unfixable `SmallFloat`
      quantization is active. `SegmentReader::field_norms()` now hands out a
      `from_field_stats`-built `FieldNorms` straight off the reader -- the API
      the remaining call sites need. Owners now: b14 `explain.rs`, b15 FFI.)*
      `FieldNorms::from_field_stats` is Java's `sumTotalTermFreq / docCount`
      exactly, but `FieldNorms::open` -- used by `lucene-ffi`, `explain.rs` and
      every test -- sums each *live* doc's *decoded* (lossy above length 24,
      one-directional) norm and divides by the count of live docs with a norm.
      Java's `docCount` also counts deleted docs. Both shift `avgdl`, which is
      in every score's denominator; invisible on b12's short-document fixtures,
      real on any corpus with long fields. Owners: b13 (`field_norms.rs`), b15
      (the FFI call site). (Raised by b12, F-7.) **b15 half done**:
      `lucene-ffi/src/query.rs::open_field_norms` now calls
      `FieldNorms::from_field_stats` with the field's `.tmd`
      `sum_total_term_freq`/`doc_count`, so the FFI path (and the three
      `explain.rs` FFI wrappers that share the helper) matches Java exactly.
      `FieldNorms::open` itself still exists for b13/tests.
      **CLOSED, verified by c34-ledger-reconcile.** Every *production* call site now uses `FieldNorms::from_field_stats` (Java's `sumTotalTermFreq / docCount`): `lucene-ffi/src/query.rs::open_field_norms` (b15) and `benchmarks/rust-runner/src/main.rs:478`. The nine surviving `FieldNorms::open` call sites are all inside `#[cfg(test)]` modules (`explain.rs:1870`, `lib.rs:8709`, `doc_value_query.rs:2479/2508`, three integration suites) -- checked against each file's own `#[cfg(test)]` line. The reader-wide half was closed by c6 (`DirectoryReader::avg_field_length`).
- [x] **`BooleanScorer`'s window/bucket bulk OR is unported** -- the one
      mechanism in Lucene's disjunction path this project has never tried, and
      the one that matches M1.6's own conclusion after six failed pruning
      attempts ("has to come from not reaching the documents at all, at a
      coarser granularity than a document"). 2048-doc windows into 1024 score
      buckets. Highest-value untried item for the boolean queries. (Raised by
      b12, F-22.)
      **CLOSED by c6.** `crates/lucene-search/src/docid_set.rs:227::WindowedDisjunction`, chosen the way `BooleanScorerSupplier.booleanScorer` chooses it. c6 also corrected the premise: under `ScoreMode.TOP_SCORES` Lucene 10.5.0 picks `MaxScoreBulkScorer`, not `BooleanScorer`. The `[x]` c6 entry further down says the same thing; ticked here so the two agree.
- [ ] **No `TwoPhaseIterator` / `matchCost` / `ScorerSupplier.cost()`** -- no
      cheap-approximation-then-verify split anywhere, so a conjunction verifies
      an expensive phrase clause on every candidate and clause order is the
      caller's problem. Needs a `Scorer`-shaped abstraction, so it should be
      weighed together with `Occur.FILTER` (F-16) as one milestone. The
      contained part -- ordering conjunction clauses by `docFreq` ascending --
      is b13-sized. (Raised by b12, F-20.)
- [x] **`Occur.FILTER` does not exist** (three clause lists, not four), which
      makes four of `BooleanQuery.rewrite`'s twelve rules unreachable and
      leaves no way to express "required but non-scoring". Wants
      `ScoreMode::CompleteNoScores`'s postings half to be worth anything.
      (Raised by b12, F-16.)
      **CLOSED by c11.** `crates/lucene-search/src/query.rs:861::BooleanQuery::filter` is the fourth clause list; matching, zero-contribution scoring, `explain`'s `# clause` arm and the five `FILTER` rewrite rules are pinned bit-for-bit against real `IndexSearcher`. Same finding as the `[x]` c11 entry below.
- [x] **No `maxClauseCount`**: a prefix query expanding to a million terms
      builds a million-clause query where Java throws `TooManyClauses` past
      1024. Denial-of-service guard; the FFI boundary is the right place for
      the policy. (Raised by b12, F-17; owner b15.) **Closed by b15**:
      `lucene-ffi/src/query.rs::MAX_CLAUSE_COUNT = 1024`, enforced in
      `read_term_clauses`, so every boolean-query and explain entry point
      (single- and multi-segment) rejects an over-long clause list as
      `FfiStatus::InvalidArgument` with Java's own message shape.
      **CLOSED** -- the entry's own body already recorded b15's fix and the box was never ticked. Verified: `crates/lucene-ffi/src/query.rs:253::MAX_CLAUSE_COUNT = 1024`, enforced at `:279` with Java's message shape, over the occur-tagged clause array c13 introduced.
- [x] **Sloppy phrase matching is in-order only** -- `SloppyPhraseMatcher`
      admits term reordering within the slop budget. b12 fixed the *frequency*
      (`1/(1+matchLength)` summed, was a flat 1) but not the reordering. The
      concrete blocker is a fixture: `GenBlockTree` has no reordered-phrase
      document, so there is nothing to check an implementation against.
      (Raised by b12, F-19.)
      **CLOSED by c37-search-behaviours** (`lucene-search/src/sloppy_phrase.rs`).
      The blocker was false: `GenBlockTree`'s doc 8558 (`delta`@0 `gamma`@1) has
      been a reordered document since task #55 added it for `SpanNearQuery`, and
      any `alpha beta` document is one for the query `"beta alpha"`. See open
      item 7 above.
- [x] `searchAfter` (`TopScoreDocCollector`'s `after`) and
      `MaxScoreAccumulator` (one min-competitive score shared across
      concurrently-searched leaves) are both unported. The second is b13's,
      since it has to live where the fan-out does. (Raised by b12, F-14/F-15.)
      **CLOSED by c37-search-behaviours**: `TopDocsCollector::with_after`,
      `collector::MaxScoreAccumulator`,
      `multi_segment::{merge_multi_segment_scored_after,
      search_term_query_multi_segment_after,
      merge_multi_segment_scored_concurrent_shared_max_score}`,
      `ffi_search_term_query_scored_after`. See open item 5 above.
- [x] `Weight.count(LeafReaderContext)`: a `TermQuery` can answer "how many
      match" from `docFreq` with no deletions, `MatchAllDocsQuery` from
      `numDocs`. `CountCollector` always iterates. Belongs with b13's
      multi-segment count path. (Raised by b12, F-23.)
      **CLOSED by c37-search-behaviours** (`lucene-search/src/weight_count.rs`,
      `ffi_count_term_query`). 13.93 ms -> 72.2 ns on the 5M-document corpus.
      See open item 5 above.
- [ ] Only one `Similarity` exists (BM25), with no `Similarity`/`SimScorer`
      trait: `TFIDF`/`Classic`/`Boolean`/`LMDirichlet`/`IndependenceStandardized`
      are all unported, as are `k3`/`computeQueryTermWeight`. b12 split BM25
      into `idf`/`norm_inverse`/`do_score`, which are the three pieces a future
      trait needs. (Raised by b12, F-11/F-12.)

- [x] `segment_writer::flush_stored_only_segment` writes a `.si` whose `files`
      set omits `<segment>.si`. Real Lucene's
      `Lucene99SegmentInfoFormat.write` always adds it (`si.addFile`) and
      `merge.rs` already does; `IndexFileDeleter` reference-counts from that
      set. `check_index`'s new `si.files_lists_itself` check flags it. One
      line, in b9's file. (Raised by b11.)
      **CLOSED in the main session.** `crates/lucene-index/src/segment_writer.rs:224` now pushes `si_name` into `files` before building the `SegmentInfo`, with the `Lucene99SegmentInfoFormat.write`/`si.addFile` citation in place; `check_index`'s `si.files_lists_itself` arm (`check_index.rs:399/939`) is the tripwire.
- [x] `lucene-search/src/points_query.rs::corrupt_kdd_leaf_data_surfaces_as_points_error`
      fails on the current tree (`unwrap_err()` on an `Ok`). `points_query.rs`
      is unmodified; `lucene-codecs/src/points.rs` grew ~879 lines in b7/b8, so
      the corrupt-`.kdd` input the test builds now decodes cleanly. Needs a
      corruption the new `intersect` path actually rejects. (Raised by b11.)
      **CLOSED.** c34 *ran* it: `cargo test -p lucene-search --lib corrupt_kdd` -> `points_query::tests::corrupt_kdd_leaf_data_surfaces_as_points_error ... ok`. The corruption the test builds is rejected by the current `intersect` path.
- [x] `check_index` still does not verify norms *values*
      (`CheckIndex.testFieldNorms`), postings positions/offsets/payloads
      ordering, or term-vectors-vs-postings agreement (Java's slow level).
      HNSW/vector graph checks remain blocked on there being no vector write
      path at all. (Raised by b11.) **Closed by c9**: all three ported
      (`norms.*`, `postings.positions_valid`/`offsets_valid`,
      `term_vectors.self_consistent`/`match_postings`), plus `vectors.*` and
      `hnsw.*` -- c5 landed the vectors/HNSW codec, and `check_index` reads
      through `lucene-codecs` directly, so the "no write path" premise no
      longer holds at the level this module works at. c9 also found that the
      `file:*` check was `retrieveChecksum` (footer *shape*) rather than
      `checksumEntireFile`, so a mid-file byte flip passed it.
- [x] `crates/lucene-codecs/src/postings.rs`: `read_postings`' full-block
      loop ends with `debug_assert_eq!(r.position(), header.body_end)`, so a
      corrupt `.doc` block **panics** in a debug build instead of returning
      `Error::Corrupted` like every other malformed-input path. `CheckIndex`
      is precisely the tool one runs on a corrupt index; c9's
      `corrupting_the_doc_skip_data_is_caught_by_the_advance_check` has to
      `catch_unwind` around it. Owned by c8. (Raised by c9.)
      **CLOSED by c8.** The `debug_assert_eq!` is gone: `postings.rs:2887::check_wire_position` returns `Error::Corrupted` ("decode ended at N but the file's own length field claims M"), called at `postings.rs:666/702/3987`. `check_index.rs:5039` records that the `catch_unwind` + silenced panic hook came out with it -- and that deleting the workaround is what exposed a fifth panic nobody had found by hand.
- [x] `segment_info::IndexSortField` models only
      `(field, reverse, missing-first-or-last)`, so `parse` *rejects* valid
      Lucene sorts it cannot represent (numeric sort with no missing value,
      arbitrary missing sentinel, non-`MIN` multi-value selector). Widening it
      means adding fields/variants whose construction and `match` sites live in
      `segment_writer.rs` and `merge.rs`. (Raised by b11.)
      **c34 restated.** Still open, but the failure mode is now *honest* rather than silent: `segment_info::parse` rejects a sort it cannot represent as `Error::UnsupportedSortField { field, reason }` naming exactly what it was (`segment_info.rs:113`, and the `//!` list at `:79-81`), instead of guessing. What is missing is the representation itself -- `IndexSortField` is still `(field, reverse, missing: SortMissingValue{First,Last})` at `segment_info.rs:167-173`. Cost: this port **cannot open** a real Lucene index whose sort uses a numeric field with no missing value, an arbitrary missing sentinel, or a non-`MIN` multi-value selector. Blocked on nothing but the ripple: widening the enum changes construction and `match` sites in `segment_writer.rs` and `merge.rs`. **CLOSED by `c35-norms-and-sort`**, ripple and all: `IndexSortKind` covers all four providers, `SortKeyComparator` replaced `sort_key_rank` in the flush, the merge and `CheckIndex.testSort`, and `fixtures/data/sorted_index_wide/` is a real Lucene index with a sort the old model rejected that this port now opens, orders and checks.

- [x] `scripts/gen-fixtures.sh --check` was not run end to end for b10 (it
      regenerates every fixture twice and other batches were mid-edit).
      `GenMergePolicy` itself was verified deterministic and byte-identical to
      the committed manifest by generating twice by hand. Re-run the full
      `--check` once the tree is quiet. (Raised by b10.)
      **CLOSED.** c34 ran `scripts/gen-fixtures.sh --check` end to end, exit 0: 47 deterministic files byte-identical, 629 non-deterministic (random segment id), **0** deterministic mismatches, **0** missing, **0** unexplained extras, **0** manifests with a wrong key set, **0** segment-id baseline lines disagreeing.

- [x] Stored-fields **bulk merge** path. **Done by c4**: `StoredFieldsWriter`
      is now the streaming object Java's writer is, with all three of
      `Lucene90CompressingStoredFieldsWriter.merge`'s strategies
      (`copy_chunks`/`add_serialized_document`/`add_document`), a
      `MatchingReaders` port, `tooDirty`, a `BlockState`-equivalent
      `ChunkCursor`, and `checkIntegrity` before any byte copy. BULK 520x,
      DOC 26x, VISITOR 29x. (Raised by b10, closed by c4.)
- [x] **Streaming postings merge.** **Done by c4**: `merge_postings` is a k-way
      merge over one forward `TermCursor` per source, decoding each term's
      postings (and positions) from the cursor's own frame through the new
      `TermsEnum::try_current_postings`/`try_current_postings_and_positions` —
      zero dictionary seeks, no `BTreeSet` of terms, `docsSeen` as a
      `FixedBitSet`. 10.8x. The output is still buffered before writing
      (`postings_writer::write_fields` takes materialised `TermPostings`), which
      is b5's API to change. (Raised by b10, closed by c4.)
- [x] **`BKDWriter.merge`.** **Done by c4**: `merge_point_streams` is Java's
      priority-queue loop (value bytes, then docID) and `points::presorted_leaf_plan`
      is `OneDimensionBKDWriter` — no sort, no per-node `widest_dim` scan, and
      the leaves are slices of the caller's own vector rather than copies. The
      sortedness precondition Java takes on trust is *verified* here in one
      linear scan, with the general path as the fallback. 2.6x. (Raised by b10,
      closed by c4.)
- [x] **Term-vectors bulk merge**, and the chunking it needs first.
      **Closed by c8-tv-chunking.** `term_vectors::TermVectorsWriter` is now
      `Lucene90CompressingTermVectorsWriter` -- both flush triggers (4 096
      bytes / 128 documents), all nine per-chunk header writers, prefix
      compression, the derived `charsPerTerm`, dirty-chunk accounting,
      `tooDirty` and `copyChunks` -- and `merge.rs`'s
      `write_merged_term_vectors` is that writer's `merge`, `MatchingReaders`
      gate and `checkIntegrity`-before-byte-copy included. Merge
      289 292 ms -> 113.5 ms from the chunking and a `ChunkCursor` alone, then
      -> 0.6 ms with the byte copy; random-access `document()` 195x. (Raised by
      c4.)
- [x] **`IndexWriter::execute_merge` supplies no norms**, so an automatic merge
      silently drops them and changes every BM25 score in the merged segment.
      c4 made the merged `.fnm` honest about it (which is what keeps the index
      openable), but the loss is real; the fix is to open each source's
      `.nvm`/`.nvd` and populate `MergeSource::norms`, the way `execute_merge`
      already does for postings and term vectors. `index_writer.rs`.
      (Raised by c4.)
      **CLOSED by c22/c26.** `index_writer.rs:5107` carries `RawNormsFiles` ("Raw `.nvm`/`.nvd` bytes for a source that has norms") on the merge source, and `merge.rs:2022` writes them back through `norms::write_fields`. c26's `check_format_coverage` plus the `merge-format-completeness` test (`index_writer.rs:15409`) now mechanically refuse a merge that fails to open a format a source's `.si` lists, which is what stops this class recurring.
- [ ] **A rustdoc pass belongs in the gate.**
      `rustdoc::broken_intra_doc_links` is warn-by-default and caught by none
      of `cargo fmt`/`clippy`/`test`; c4 shipped one broken link that survived
      a green Tier 1 gate. Turning it on needs the pre-existing broken links in
      `doc_values.rs`, `for_util.rs` and others cleaned up first. (Raised by c4.)
- [x] `MergeSource` carries no per-source `min_version`/`has_blocks`, so a merged
      `.si` claims the caller's version and `has_blocks = false` instead of
      Java's `min over readers` / propagated flag. Unreachable while this port
      only merges segments it wrote itself; the fix is an exhaustive-struct-literal
      change across ~85 call sites plus `index_writer.rs`. (Raised by b10.)
      **CLOSED by c36-merge-metadata** -- see open-work item 1 above. Both
      fields are on `MergeSource`, `merged_min_version` is `SegmentMerger`'s
      fold, and `verify-write-path.sh` case 23 checks both through real
      Lucene's `LeafMetaData`. The `has_blocks` half turned out to have been
      correct since c22 (via `MergeOptions`); only `minVersion` was wrong.
      **c34 restated.** Confirmed still open: `MergeSource` (`merge.rs:790-825`) carries `field_infos`/`reader`/`live_docs`/the five doc-values kinds/`norms`/`term_vectors`/`postings`/`points`/`vectors` and no `min_version` or `has_blocks`; the merged `.si` takes `min_version: Some(lucene_version)` (`merge.rs:2142`) and `has_blocks: options.has_blocks` (`:2151`), i.e. the caller's version and the caller's flag, not Java's `min over readers` / propagated flag. Still unreachable while this port only merges segments it wrote itself, so this is a **latent** correctness item, not a live one. Cost of the fix is unchanged: an exhaustive-struct-literal change across ~85 call sites.
- [x] Zero-doc merges should be dropped by `IndexWriter::apply_merge` the way
      `SegmentMerger.shouldMerge()` + `IndexWriter.commitMerge` do, rather than
      committing a real 0-doc segment. Belongs to b9/b11. (Raised by b10.)
      **CLOSED by c36-merge-metadata**: the guard is in `execute_merge` (where
      Java's `shouldMerge()` is, so no file is written at all) and
      `IndexWriter::drop_merge` is `commitMerge`'s `dropSegment` branch.
      `merge.rs`'s `no_sources_produces_an_empty_segment` still stands -- it
      tests `merge_segments` itself, which is `SegmentMerger`, not the writer.
      **c34 restated.** Confirmed still open: `IndexWriter::apply_merge` (`index_writer.rs:6491`) folds the merged `SegmentCommitInfo` in unconditionally -- there is no `doc_count == 0` guard anywhere in it -- and `merge.rs`'s own `no_sources_produces_an_empty_segment` test asserts a genuinely well-formed 0-doc segment is what comes out. Java's `SegmentMerger.shouldMerge()` + `IndexWriter.commitMerge` drop it instead. Cost: a committed 0-doc segment per empty merge, which every later open, merge and `CheckIndex` then pays for.

- [x] `lucene-codecs` privately duplicates primitives that now exist in
      `lucene-store`: zigzag i32, group-varint write, `header_length` /
      `index_header_length` / `retrieve_checksum_with_expected_length`.
      Migrate the call sites. (Raised by b1; files are outside b1's scope.)
      **CLOSED by c34.** zigzag and group-varint *reading* had already moved (`block_packed.rs` uses `lucene_util::zigzag`, `postings.rs` reads through `DataInput::read_group_vints`). c34 removed the last two: `compound_format.rs`'s private `index_header_length`/`vint_len` pair now calls `lucene_store::codec_util::index_header_length(codec, "")` -- the two formulas agree for every codec name `write_index_header` accepts (ASCII under 128 bytes, so the length prefix is one byte), and the old `vint_len` unit test is replaced by one pinning the shared helper against the bytes `write_index_header` actually emits -- and `postings.rs`'s `write_group_vints` free function, a line-for-line copy of `DataOutput::write_group_vints`, is deleted with its 19 call sites moved onto the trait method (`hnsw_vectors.rs` already used it).
- [ ] DEFLATE encoder has no preset dictionary (`miniz_oxide` exposes no
      `deflateSetDictionary`): compression ratio only, decode side correct.
      Revisit if a raw-deflate crate with dictionary support is acceptable.
      (Raised by b3.)
- [ ] Stored-fields writer API takes `&[Document]` rather than streaming, and
      `document()` materializes a whole `Document` instead of Java's
      `StoredFieldVisitor`. Memory-shape divergence. (Raised by b3.)
      **c34 restated -- half of this is no longer true.** The *streaming* half was closed by c4: `stored_fields::StoredFieldsWriter` (`stored_fields.rs:1161`) is a real streaming object with `add_document(&Document)` (`:1373`) and `finish()` (`:1601`); `write_best_speed(&[Document])` survives as a convenience wrapper, not as the only API. What remains is the **read** side: `StoredFieldsReader::document()` (`stored_fields.rs:412`) materializes a whole `Document` where Java's `StoredFieldVisitor` lets a caller take one field and skip the rest. Memory-shape divergence, read path only.
- [x] `IntersectTermsEnum`'s **skipping** is ported for regexp (b8). Blocker
      (a) is gone: a full `Automaton`/`CompiledAutomaton` was assessed and not
      built, because what `IntersectTermsEnum` needs from one is a single bit --
      *is this prefix dead?* -- which a backtracker can answer directly by
      treating "ran out of input" as success. `RegexpPattern::dead_prefix_len`
      does that and `blocktree::RegexpIntersect` jumps past the dead run with a
      galloping search. Measured on a million-term dictionary
      (`benches/regexp_intersect.rs`): `t1[0-9]` 88x, `t1*z` -- b4's shape --
      **1065x**; an adaptive give-up keeps the shapes it would lose on at
      1.00-1.01x. Blocker (b) stands: Lucene's other win is *not loading* the
      pruned blocks, which needs A1. Blocker (b) is gone too (c1): with lazy frames the
      skip's `seekCeil` no longer loads the blocks it jumps over, re-measured
      over a *real-Lucene* `t0..t999999` dictionary at `t1[0-9]` **131x** and
      `t1*z` **1768x** (up from 88x/1065x), with the prefix-closed shapes
      still at 1.00-1.17x. Still to do: the same treatment for
      `fuzzy_intersect` (needs a Levenshtein automaton's dead states, not a
      prefix test) and `intersect` (wildcard).
- [~] `RegExp`'s `CASE_INSENSITIVE`/`CASE_INSENSITIVE_RANGE` *match* flags, and
      `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT`'s reject-at-construction
      semantics (this port bounds matching instead and reports "no match").
      Neither is reachable from `RegexpQuery(Term)`. (Raised by b8.)
      **OBSOLETE (an intentional divergence, not outstanding work).** The `DEFAULT_DETERMINIZE_WORK_LIMIT` half is superseded: `regexp.rs`'s own gap list (`regexp.rs:45-52`) now names exactly two gaps and that is not one of them -- this port bounds *matching* instead, a different mechanism with the same visible contract. The `CASE_INSENSITIVE`/`CASE_INSENSITIVE_RANGE` half stands but is unreachable: `RegexpQuery(Term)` passes match flags `0` and this port exposes no constructor taking them, so no caller can observe it. It belongs in the module doc, where it is, not on a work list.
- [ ] **Payload slots cost ~26 us/doc and ~190 MB per 50 000 documents, and the
      cost is the slot, not the bytes** (c23 F9, measured with an
      all-empty-payload control). The fix is a flat `(bytes, lengths)`
      representation in **both** `postings_writer::TermPostings::payloads` and
      `indexing_chain::PostingEntry::payloads`; doing only the second is
      slower, because `build_postings_output` would re-materialize the nested
      form. An instance of the block-pool item above, now with a number.
- [ ] **`FieldInfo` is a plain struct where Java's is a validating
      constructor** (c23 F4). `field_infos::write` now applies the one coercion
      that was producing files this port could not re-open, but a caller can
      still build combinations Java makes unrepresentable and find out only at
      `parse` time, or not at all. A `FieldInfo::new -> Result` would close the
      class; it touches every construction site in the workspace.
- [ ] `lucene-analysis` has no `TokenStream` lifecycle, so `end()`'s trailing
      position increment is dropped by `StopFilter` and both n-gram filters; no
      case-insensitive `CharArraySet`; `StandardTokenizer` emits no emoji tokens
      and has no `maxTokenLength`; `Token` offsets are UTF-8 byte offsets where
      Lucene's are UTF-16 code-unit offsets. All four are structural rather
      than local. (Raised by b8.) **The offset one is no longer latent as of
      c23**: those offsets now reach `.pos`/`.pay` and are read back by real
      Lucene's `startOffset()`/`endOffset()`, and nothing catches the
      divergence -- `CheckIndex` checks only that offsets are ordered and in
      range, never that they index the stored text. **c29 fixed the read half
      and left the write half here**: `lucene-search`'s highlighter now
      measures in UTF-16 code units (it had been using Unicode scalars, a third
      unit again, wrong for every supplementary-plane character) and
      `offsets_from_analysis` converts bytes to UTF-16 at the boundary --
      pinned against a real `StandardAnalyzer`'s offsets in
      `fixtures/src/GenBreakIterator.java`. What remains is
      `lucene-analysis/src/lib.rs::tokenize` itself, whose four affected call
      sites (its own doc comment, `analysis_fixtures.rs`'s
      `char_offsets_to_byte_offsets` reconciliation -- itself scalar-based and
      so wrong for astral text, `offsets_from_analysis`'s now-redundant
      conversion, and a non-ASCII case for `GenTermVectors`/the positions
      write-path verifier) are listed in `c29-search-carryovers.md` §2.2.
      **c34 restated -- the offsets half is closed.** `lucene-analysis/src/lib.rs::tokenize` now emits **UTF-16 code-unit** offsets (`lib.rs:150-203`, `utf16_len`), so the unit matches Java's `char` indices end to end; `lucene-search`'s highlighter measures in the same unit and `offsets_from_analysis` no longer converts (`highlighter.rs:1754-1788` pins that it must not). Three structural gaps remain, all in `lucene-analysis`: (a) **no `TokenStream` lifecycle**, so `end()`'s trailing position increment is dropped by `StopFilter` and both n-gram filters (`lib.rs:2306-2310` records it at the site); (b) **no case-insensitive `CharArraySet`** -- the port matches a lowercase set against already-lowercased terms, which is right for the standard chain and wrong for a caller-supplied mixed-case set; (c) **no `maxTokenLength`** on `StandardTokenizer`, and no grapheme-cluster emoji tokenization (`lib.rs:121-126`). (a) is the one a caller can observe as a wrong position.
- [ ] `FuzzyTermsEnum`'s `MaxNonCompetitiveBoostAttribute` feedback loop (swap
      to a lower-edit automaton once the top-terms queue is full) is unported,
      and `fuzzy_doc_scores` blends `docFreq` within one segment where
      `BlendedTermQuery` blends across the whole reader -- the fuzzy clause has
      no `GlobalStats` plumbing. (Raised by b8, owner b12/b13.)
- [x] **A1 is closed (c1-lazy-blocktree).** `blocktree.rs` now opens a segment
      by reading only the `.tmd` records and each field's root trie node, and
      navigates with a ported `SegmentTermsEnum`/`SegmentTermsEnumFrame` frame
      stack (`TrieReader.lookupChild` one label at a time, `scanToFloorFrame`,
      `loadBlock` for the one block a seek reaches, `binarySearchTermLeaf`/
      `scanToTerm*`, `decodeMetaData` only up to the term landed on).
      Measured on the M1 corpus' 579k-term real-Lucene segment:
      `blocktree::open` **35.4 ms -> 0.175 ms (202x)**, now below Lucene's
      whole `DirectoryReader.open` (0.310 ms); live heap per segment
      **+39.0 MB -> +4.7 MB** (the shared `.tim`/`.tip` buffers, nothing per
      term); and the hot seek loop holds up against Lucene's own, interleaved
      A/B on the same 2000 terms in the same order -- a cold hit 495 ns
      against Lucene's 440 ns (1.13x), a miss 215 ns against 263 ns (cheaper
      than Lucene's). See `docs/sweep/m2/c1-lazy-blocktree.md`.
- [ ] **Migrate the blocktree lookups to their `try_*` forms** (c1, F-11).
      A corrupt `.tim` block is now discovered at lookup time, as in Java, so
      `FieldTerms::try_seek_exact`/`TermsEnum::try_next`/`try_seek_ceil`/
      `try_current` exist and are tested; the pre-existing infallible
      spellings are kept and degrade such a block to "no such
      term"/end-of-terms. `seek_exact` alone has 64 call sites across
      `lucene-search`, `lucene-index`, `lucene-ffi` and `benchmarks/`, all of
      which were being edited by b13/b14/b15 while c1 ran, which is why the
      migration was not done there. (Raised by c1.)
- [x] **`directory_reader` should call `blocktree::open_shared`** (c1, F-13).
      **Closed by c12, and the "three-line change" estimate was wrong for a
      reason worth recording**: `open_shared` took an `Arc<[u8]>`, which owns
      its allocation, so there is no zero-copy route to one from
      `lucene_store::Input` (a mapping cannot move into an `Arc`, and
      `Arc::<[u8]>::from(Vec)` copies too) -- migrating the caller alone would
      have moved the copy, not removed it. Fixed by erasing the owner:
      `blocktree::SharedBytes` is `Arc<dyn AsRef<[u8]> + Send + Sync>`, and
      `Input` gained an `AsRef<[u8]>` impl so the `Arc<Input>`
      `open_segment_file` already returns -- for a `MmapDirectory`, the mapping
      itself -- coerces straight in. Re-measured on the M1 corpus:
      `DirectoryReader::open` **579 us -> 120.7 us (4.8x)**, larger than the
      199 us copy alone because the copy also first-touched 4.7 MB of the
      mapping. No measurable seek regression from the erasure's virtual
      `as_ref` (both accessors are hoisted per lookup; the machine's spread on
      those cases is +/-25%).
- [ ] **Split term iteration from stats in `TermsEnum`** (c1, F-14). Java's
      `next()` decodes only the term bytes and defers `decodeMetaData` to
      `docFreq()`; this port's `next()` returns `(term, TermStats)` so it
      always decodes. Full-field enumeration is 27 ns/term against Lucene's
      20.5 ns on the same field. Wants a `next_term()` + `stats()` split,
      which changes `check_index`'s and the intersect iterators' call shape.
      (Raised by c1.)
- [x] Re-take `cargo llvm-cov --workspace --summary-only` once the tree is
      quiet. c1 could only get a trustworthy per-file reading by pointing
      `CARGO_TARGET_DIR` at a scratch directory and restricting to
      `-p lucene-codecs` (`blocktree.rs` 95.80% lines): four batches
      rebuilding the shared `target/` kept emptying `target/llvm-cov-target`
      mid-run, which llvm-cov reports as coverage rather than as an error.
      (Raised by c1.)
      **CLOSED.** `scripts/gate.sh` now ends with `cargo llvm-cov --workspace --fail-under-lines 95`, so the reading is re-taken on every commit rather than by hand. c34 re-took it anyway: **97.55% regions / 98.10% lines** workspace-wide, and **no file is below 95% lines**.
- [x] `benchmarks/rust-runner`'s `micro` binary does not compile
      (`src/micro.rs:84`, `for_util::for_encode` now takes `&mut [i64]`), so
      the `reader_open`/`postings_iter`/`stored_fields` micros cannot be run.
      c1 had to derive its `DirectoryReader::open` "after" number instead of
      measuring it; re-run
      `micro reader_open benchmarks/.corpus/merged` once fixed. Owner: whoever
      changed `for_encode` (b2/b5). (Raised by c1.)
      **CLOSED in the main session.** `benchmarks/rust-runner/src/micro.rs:93` encodes from a scratch copy (`for_util::for_encode(&mut scratch, bits, &mut bytes)`) so `values` stays the pristine round-trip expectation. The binary builds.
- [ ] `DirectoryReader::open` is now dominated by everything *except* the term
      dictionary: ~2.0 ms of the ~2.2 ms is `open_segments`' file handling
      against Lucene's 0.310 ms for the whole open. That is the next
      reader-open item now that A1 is gone. Owner: b13. (Raised by c1.)
      **c34 restated -- the numbers here are three batches stale.** The 2.0 ms/2.2 ms figures predate c12 and the mmap work. Current state: c12's `open_shared`/`SharedBytes` change took `DirectoryReader::open` on the M1 fixture corpus **579 us -> 120.7 us (4.8x)**, and `88ebd47 perf(search): stop copying mmap'd postings files into the heap on reader open` landed after that. The standing whole-corpus number is `docs/benchmarks/verdict-m1.6.md`: reader open **52.7 ms** against Lucene's, i.e. **~155x**, with RSS 1,690 MB -> 70 MB. So this is still the largest single reader-side gap, but the diagnosis ("`open_segments`' file handling") needs re-measuring before anyone plans against it -- `benchmarks/rust-runner`'s `micro reader_open` now builds and runs (see the closed item above), so measuring it is cheap.
- [x] No `PostingsEnum`-flags plumbing: `DocInput::read_postings`/`lazy_cursor`
      always decode freqs, where Lucene's `needsFreq == false` path
      `PForUtil.skip`s the freq block entirely. Fixing it changes those
      signatures and every call site in `blocktree.rs` and `lucene-search`
      (owners: b4, b12/b13). (Raised by b5.) **b12 update**: the search-side
      half is done -- `collector::ScoreMode` now exists with Java's
      `needsScores()`/`isExhaustive()` predicates, and `TopDocsCollector`
      reports its mode. What remains is entirely the codec-side flags path.
      **CLOSED.** The codec-side half landed: `postings::PostingsFlags` with `DocsOnly`/`Freqs`, `DocInput::read_postings_with_flags` (`postings.rs:566`) and `lazy_cursor_with_flags` (`:776`), surfaced as `blocktree::FieldTerms::postings_with_flags`/`lazy_postings_with_flags` (`blocktree.rs:2105/2145`) and actually *used* by the unscored search paths (`lucene-search/src/lib.rs:411`, `:575`, `:2849`). `lucene-search/benches/docs_only_postings.rs` measures what it buys. b12's search-side half (`collector::ScoreMode`) was already done.
- [ ] `Lucene104PostingsWriter` takes a `NumericDocValues norms` per term and
      feeds real per-doc norms into `CompetitiveImpactAccumulator`;
      `postings_writer::FieldPostingsInput` carries no norms, so impacts are
      computed against norm 1. Sound (norm 1 is the highest-scoring norm) but
      loose, costing pruning. (Raised by b5.) **b9 update**: the
      `lucene-index` half is unblocked -- one shared invert pass per commit now
      computes every document's field length at the `write_fields` call site.
      What remains is entirely in `lucene-codecs` (a norms input on
      `FieldPostingsInput` plus a real `CompetitiveImpactAccumulator`), so the
      owner is whoever next owns `postings_writer.rs`, not b9.
      **c34 restated -- the "impacts are empty" premise is gone; the norms premise stands.** This writer *does* emit impacts now: `write_full_block` writes one `(maxFreq, norm = 1)` impact per level-0 block and `write_level1_span` (`postings_writer.rs:1283`) writes the span-wide maximum, because real Lucene rejects a segment with "Got empty list of impacts". What is still missing is the *norms input*: `FieldPostingsInput` (`postings_writer.rs:342-360`) has `field_number`/`index_options`/`doc_count`/`has_payloads`/`terms` and no norms, so every impact is computed against norm 1 where `Lucene104PostingsWriter` feeds real per-doc norms into `CompetitiveImpactAccumulator`. Norm 1 is the highest-scoring norm, so the bound is **sound but loose**: it costs query-time pruning, never a wrong answer. c34 also corrected `postings_writer.rs`'s module doc, which still claimed empty impacts and that positions can never co-occur with a full block -- both untrue since c20/c23.
- [x] Points: `lucene-search/src/points_query.rs` and
      `lucene-index/src/points_delete.rs` still decode every point and filter
      in memory. b7 ported `PointsReader::intersect`/`range_query` (the
      `PointValues.intersect` pruning traversal, fixture-verified against real
      `BKDWriter` `.kdi` bytes) and measured 577x on a 200k-point tree at 0.1%
      selectivity; migrating the callers is b14/b11 work. (Raised by b7.)
      **CLOSED.** `points_query.rs:287` calls `PointsReader::intersect` and `:295` `range_query`; `points_delete.rs:96` calls `reader.range_query(field_number, min_packed, max_packed)`. `decode_all_points` now survives only in `lucene-codecs` tests and benches.
- [x] Vectors: `vectors.rs` was **not** Lucene's `.vec`/`.vem` format and had
      **no HNSW graph**. Closed by c5: `Lucene99FlatVectorsFormat` (with
      `OrdToDocDISIReaderConfiguration`, both encodings, all four
      similarities) and `Lucene99HnswVectorsFormat` + `util/hnsw/*` are
      ported, verified in both directions against real Lucene, and measured at
      47x/244x fewer distance computations with recall@10 matching Lucene's
      exactly (0.9250 vs 0.9250). See `docs/sweep/m2/c5-vectors.md`.
      *Still open from it*: the merge write path
      (`mergeOneField`/`IncrementalHnswGraphMerger`,
      `mergeOneFlatVectorField`), `FLOAT16`, the index-sort write path, and
      `IndexWriter` wiring -- nothing in `lucene-index` can add a vector field
      to a document yet.
- [x] Term vectors: callers of `write_best_speed` must pass each document's
      fields in ascending field-*name* order, or real `CheckIndex` rejects the
      segment (`TVFields.iterator()` yields document field order and
      `checkFields` requires it sorted). Enforce in b9's flush path, where
      names are known. (Raised by b7.)
      **CLOSED by c34.** Enforced where the names are known: `IndexWriter::build_term_vectors_output` sorts its `TermVectorFieldConfig`s by name before building `per_doc`, so a document's `fields` list comes out ascending by field *name* whatever order `add_term_vector_field` was called in. Pinned by `term_vector_fields_are_written_in_ascending_field_name_order`, which declares `zeta` as field 1 and `alpha` as field 2, configures them in that (wrong) order and asserts the written field numbers are `[2, 1]` -- verified to fail (`left: [1, 2]`) with the sort removed. `write_best_speed`'s own caller contract is unchanged and still documented in `docs/parity.md`.
- [ ] `Util.shortestPaths`/`TopNSearcher`/`readCeilArc` unported (b8 confirmed
      and left it: `top_n_completions` walks the prefix subtree with a bounded
      heap, which cannot skip a subtree that provably cannot beat the current
      worst candidate). `SegmentTermsEnum` `TermState` seeking unported (owner: the
      search batches -- no `TermStates` caller exists yet). (Raised by b4.)
- [~] `postings_writer` should hold one `ForUtil` across blocks rather than
      constructing one per block (b2 made `encode` in-place + scratch-by-ref;
      the caller side is b5/b9 territory).
      **OBSOLETE.** There is no per-block `ForUtil` to hoist. b2 turned the encoder into free functions that pack in place with caller-supplied scratch (`for_util::for_encode`/`pfor_encode`), and `postings_writer.rs` calls them directly (`:1189`, `:1225`) -- it never constructs a `ForUtil`. The `ForUtil` struct still exists at `for_util.rs:887` for the decode side; the writer does not touch it.
- [x] Validate `bits_per_value` at `doc_values::read_varying_block`'s parse
      site -- covered by b2's `direct_reader::get` validation; b6 confirmed.
- [x] `PostingsEnum`-flags plumbing: freqs are always decoded where Lucene
      `PForUtil.skip`s them; and `FieldPostingsInput` carries no norms, so
      impacts are `(maxFreq, 1)` -- exactly Java's output for a norm-less
      field, a sound-but-loose bound otherwise. Needs signature changes in the
      search crate. (Raised by b5, owner b12/b13.)
      **CLOSED as written** -- this is the flags half of the entry above (closed: `PostingsFlags`) plus the impacts-vs-norms half, which is tracked as its own open item (`Lucene104PostingsWriter` norms / `FieldPostingsInput`). Two ledger entries for one finding; folded into that one so it stops being counted twice.

- [x] No `IndexFileDeleter`. **Closed by c3**:
      `crates/lucene-index/src/index_file_deleter.rs` ports `IndexFileDeleter`
      + `util/FileDeleter` + `KeepOnlyLastCommitDeletionPolicy`/
      `NoDeletionPolicy`, and `IndexWriter` checkpoints it at every lifecycle
      point. Windows delete-on-close (`deletePendingFiles`,
      `getPendingDeletions`, the `Constants.WINDOWS` branch) is deliberately
      omitted -- Linux target. The follow-on from the same area --
      `SegmentCommitInfo` had no `next_write_del_gen`/`next_write_field_infos_gen`/
      `next_write_doc_values_gen`, so `inflateGens`' per-segment half was
      unported -- is **closed by c7** (F-18/F-19): all four of Java's transient
      generation fields (including `bufferedDeletesGen`) are on
      `SegmentCommitInfo`, `IndexFileNames.parseGeneration` is ported, and
      `deletes::apply_deletes` derives its next `.liv` generation from
      `getNextDelGen()` so the inflation actually takes effect. (Raised by
      b9, F-11; closed by c3, F-1/F-2 and c7, F-18/F-19.)
- [x] No `DocumentsWriterDeleteQueue`/sequence numbers. **Closed by c7**:
      `crates/lucene-index/src/buffered_updates.rs` ports
      `DocumentsWriterDeleteQueue`/`BufferedUpdates`/`FrozenBufferedUpdates`/
      `DocValuesUpdate`/`BufferedUpdatesStream`, every mutating `IndexWriter`
      method returns a `long` seqNo starting at 1, deletes are buffered with a
      `docIDUpto` and resolved through a `delGen`-stamped packet, and all four
      blocked APIs landed: `softUpdateDocument(s)`, `updateDocValues`/
      `updateNumericDocValue`/`updateBinaryDocValue`, `deleteDocuments(Query)`
      and `addDocuments`/`updateDocuments` block adds with `hasBlocks` set
      (proven against real Lucene by a new `VerifyBlockSegment` case,
      `verify-write-path.sh` 16/16 -> 17/17). The lock-free linked list and the
      `DeleteSlice` heads are deliberately not ported (one indexing thread; the
      equivalence argument is in the batch report). Measured: the seqNo
      machinery's per-document cost is below `index-bench`'s noise floor in an
      interleaved 8-pair A/B (median 20.6 vs 19.9 us/doc, sign alternating).
      (Raised by b9, F-7/F-12; closed by c7, F-7 through F-13.)
- [x] No RAM accounting and no flush trigger. **Closed by c3**:
      `set_ram_buffer_size_mb` (default 16.0) / `set_max_buffered_docs`
      (default `DISABLE_AUTO_FLUSH`) with Java's exact validation,
      `ram_bytes_used()` as a real incremental byte count, and an automatic
      `flush()` from `add_document` (which is now fallible). Measured on
      `index-bench` at 200k docs: writer peak RSS 862 MB -> 128 MB, and flat in
      document count instead of linear; throughput 29.0 -> 21.0 us/doc.
      **Divergence recorded**: the counter measures the buffered-document arena,
      not Java's inverted-form bytes, because this port inverts at flush rather
      than per document -- the transient inverted structure is 9.4x the arena on
      that corpus (`InMemoryInvertedIndex::ram_bytes_used()` measures it). The
      constant only comes down with the block-pool redesign below. (Raised by
      b9, F-10; closed by c3, F-5/F-6/F-8.)
- [ ] `indexing_chain` still allocates per token/term/posting where Java uses
      `BytesRefHash` + `ByteBlockPool`/`IntBlockPool`/`ByteSlicePool`. b9
      removed the two worst constant factors (2.5x measured) but the shape is
      unchanged; closing it needs a borrowed-token `Analyzer` API
      (`lucene-analysis`) and a byte-pool `InMemoryInvertedIndex`. (Raised by
      b9, F-9.) **c3 measured the cost**: 8.3 MB of document text becomes
      78.5 MB of `InMemoryInvertedIndex` on `index-bench`'s corpus, 9.4x, and
      the single largest term is the `Vec<Occurrence>` whose first `push`
      reserves capacity 4 -- 48 bytes of allocation for 12 bytes of payload.
      c3 tried `shrink_to_fit` on it: the structure drops to 5.98x but
      indexing costs 25-60% more time and peak RSS does not move at all
      (glibc keeps the freed 48-byte chunks in its arena), so it was reverted.
      An inline-capacity-1 occurrence representation is the contained version
      of the fix and needs `PostingEntry`'s public shape to change.
- [x] `SegmentCommitInfo.sci_id` was written as absent (`marker 0`) by every
      construction site in the port. **Closed by c7** (F-20): the two sites that
      *create* a segment (`segment_writer::flush_stored_only_segment*` and
      `merge.rs`'s two merged-segment constructors) now emit a `sci_id` derived
      from the segment's own already-unique id -- distinctness is the only
      property anything reads it for, so no CSPRNG is needed, as b9 predicted.
      Every other site propagates what it was given. Real Lucene reads marker
      byte `1` + 16 bytes on every segment this port writes
      (`verify-write-path.sh` 17/17). (Raised by b9, F-13; closed by c7, F-20.)
- [x] Norms are opt-in per field (`set_norms_field`) and every other indexed
      field gets `omit_norms` forced on in the `.fnm`; Java writes norms for
      every indexed non-`omitNorms` field. Blocked on a multi-field
      `.nvd`/`.nvm` writer (`norms::write_single_dense_field` is
      single-field-only). The `IndexWriter` half is free now that one invert
      pass computes every field's lengths. (Raised by b9, F-15; owner b6.)
      **c34 restated -- the recorded blocker no longer exists.** "Blocked on a multi-field `.nvd`/`.nvm` writer" was true when b9 wrote it and is not true now: `norms::write_fields` (`norms.rs:392`) writes one or more norms fields into one pair, c26 removed the one-field cap on both the flush and the merge, and `IndexWriter::add_norms_field` (`index_writer.rs:1809`) accumulates. What is left is only the **opt-in itself**: `index_writer.rs:4789-4793` still forces `omit_norms = true` onto every indexed field the caller did not name, where Java writes norms for every indexed non-`omitNorms` field. The forcing is deliberate and correct as far as it goes (promising norms a segment does not carry is what `DirectoryReader.open` throws on), but it means this port's `.fnm` describes a different schema than the caller asked for. Nothing blocks closing it now. **CLOSED by `c35-norms-and-sort`.** `set_norms_field`/`add_norms_field` and the `.fnm` coercion are gone; `IndexWriter::norms_field_configs` is Java's `writeNorms` loop condition and `omit_norms_field` is the opt-out. `build_norms_output` writes a **sparse** column, matching `NormValuesWriter` (a document that does not carry the field gets no norm; one that carries it but produced no tokens gets an explicit `0`), and `merge_norms` -- now driven by the merged `FieldInfos` like `NormsConsumer.merge` -- carries the gaps instead of raising `Error::NormsFieldMissingInSource`. Real Lucene reads every value back in `VerifyFullSegment`.
- [x] `IndexWriter` re-reads, re-parses and rewrites the `.si` once per file
      group (postings/TV/DV/norms) instead of accumulating
      `SegmentInfo.files` and writing it once, as `sealFlushedSegment` does --
      four redundant fsyncs per commit. Needs a signature change to
      `flush_stored_only_segment`, which `merge.rs` (b10, in flight) also
      calls. (Raised by b9, F-16.)
      **CLOSED by c36-merge-metadata** -- see open-work item 19 above. The
      recorded blocker was wrong twice over: the count was seven, not four, and
      `flush_stored_only_segment`'s signature did not have to change (it is now
      `write_stored_only_segment_files` + `seal_flushed_segment`, which is also
      what `merge.rs` needed nothing of).
      **c34 restated -- it is worse than recorded.** The count is now **five** file groups, not four: `write_postings_files`, `write_term_vector_files`, `write_doc_values_files`, `write_norms_files` and `write_vector_files` (`index_writer.rs:4601/4638/4673/4707/4870`) each do the same read-`.si` / `segment_info::parse` / extend `files` / `segment_info::write` / `dir.sync` cycle. So a commit that writes every format re-parses and rewrites its own `.si` five times and fsyncs it five times, where `sealFlushedSegment` accumulates `SegmentInfo.files` and writes it once.

## Cross-batch fixes made in the main session

- `lucene_store::directory::last_commit_generation` was lenient where Java's
  `SegmentInfos.getLastCommitGeneration` throws: it skipped an unparsable
  `segments*` name. A directory whose only commit file is corrupt therefore
  read as generation -1, and `IndexWriter::open` would create a fresh index
  over it. Now `Result<i64>`, strict, and `read_latest_commit` shares it.
  The `segments.gen` guard is now Java's prefix test, not equality.
- The `segments_index` fixture directory shipped a `segments_2.raw` byte
  reference beside `segments_2` -- a name real Lucene throws on, so the
  fixture was a directory Lucene itself cannot open. Renamed to
  `expected_segments_2.bin` in `GenSegmentInfos.java` and both test suites.
- [x] `NumericReader` holds an O(cardinality) doc-id `Vec` where Java's
      `IndexedDISI` is O(1). **Done by c2-sparse-lookup.** `DisiCursor` now
      carries Java's per-block state and its DENSE rank-table jump, and
      `NumericReader` reads through it: 0 bytes of heap where it held 2.0 MB at
      1M docs / 50% density, and *faster* rather than a trade -- ~56x on a
      forward DENSE walk, ~1200x on a single lookup into a 100,000-doc field.
      A backward doc id now panics (with `reset()` as the supported way to go
      back) instead of silently answering `None`.
- [x] `lucene_search::field_norms::FieldNorms` still holds the same
      O(cardinality) `Vec<i32>` that `NumericReader` just shed, for the same
      reason (it needs random access). The fix is now a two-line one: hold a
      `DisiCursor`, `reset()` when `doc < cursor.doc_id()`. Owner: b13.
      (Raised by c2.)
      **CLOSED by c6.** `field_norms.rs` has no `sparse_doc_ids` any more: `FieldNorms` keeps only the region slice and stays `Sync`, and `FieldNorms::cursor()` (`field_norms.rs:319`) hands each scan its own `FieldNormsCursor` wrapping a `lucene_codecs::indexed_disi::DisiCursor`. See the "NOT a two-line fix" section below for the design that made it work.
- [ ] `IndexedDISI`'s **block jump table** is still not read
      (`createJumpTable`/`advanceBlock`'s two-blocks-ahead shortcut). Cost is
      O(maxDoc/65536) four-byte header reads -- 16 for a million documents --
      and this port writes `jumpTableEntryCount = 0`, so our own files have no
      table to read. Worth revisiting only alongside a `nextDoc`-shaped
      iterator API. (Raised by c2.)
- [x] `lucene-search/src/explain.rs:1687` has an invalid `dense_rank_power: 0`
      test literal (`0` is not in Java's legal set). Unreachable -- the entry is
      dense -- but wrong data. One line; `lucene-search` was held by b13/b15.
      (Raised by c2, last of b2's finding-15 list.)
      **CLOSED by c6.** `explain.rs:1865` reads `dense_rank_power: 0xFF` (Java's "no rank table"), with a comment naming `field_norms.rs`'s deliberate `0` -- the input to `an_illegal_dense_rank_power_is_rejected_rather_than_guessed` -- so the two are not confused again.
- [ ] `SegmentCommitInfo.sci_id` is not *regenerated* when a generation
      advances. Java's `advanceDelGen`/`advanceDocValuesGen`/
      `advanceFieldInfosGen`/`setBufferedDeletesGen` all call
      `generationAdvanced()`, which does `id = StringHelper.randomId()`; the
      javadoc says the id "changes each time the segment changes due to a
      delete, doc-value or field update". c7 closed b9's F-13 (a *fresh*
      segment now gets an id) but not this: two commits of the same segment
      still report the same id. Nothing in Lucene validates it; what is lost is
      its use as a change token (segment replication). Wants one pass across
      every generation-advancing site. (Raised by c7's Tier-2 review, A4.)
- [x] The sequence number does not cross the FFI boundary.
      `ffi_writer_add_document`/`ffi_writer_update_document`/
      `ffi_writer_delete_documents` all discard the `SeqNo` c7 gave them.
      Surfacing it needs an `out_seq_no: *mut i64` parameter on three exported
      functions, i.e. an **ABI change** -- deliberately not done inside a sweep
      batch. The value is real: OpenSearch's `InternalEngine` uses Lucene's
      returned seqNo directly. (Raised by c7's Tier-2 review, A7.)
      **CLOSED by c13.** The ABI change was made: `out_seq_no: *mut i64` is a parameter on all five writer entry points (`lucene-ffi/src/writer.rs:505`, `:886`, `:963`, `:1622`, `:1675`, `:1740`), each writing through `write_seq_no`, and null is accepted.
- [x] `norms::parse_meta` does not validate field numbers against
      `FieldInfos` (22 call sites, 5 crates). Missed diagnostic only, never a
      wrong value. **c7 considered and declined it**, with reasons (c7 F-23):
      c7 adds no new `parse_meta` call sites so it is not in fact "in the
      area"; the sites are in `merge.rs` (c8), `check_index.rs` (c9) and
      `lucene-search` (c11), i.e. the same collision that made b6 defer it;
      threading `FieldInfos` into `norms` alone leaves the port half-validated
      (`doc_values::parse_meta` already does, no other per-field meta parser
      does); and the right moment is when the reader side gains a
      `SegmentReader`-equivalent that parses `.fnm` once and threads it into
      every per-format open, at which point the parameter is free instead of 22
      mechanical edits the refactor would rewrite. **Assign to that batch.**
      (Raised by b6; reasoning added by c7.)
      **CLOSED by c15** at the level that matters. `norms::validate_fields(&Norms, &FieldInfos)` (`norms.rs:190`) is called from `lucene-search/src/directory_reader.rs:475` and `lucene-index/src/check_index.rs:3926` -- the two places Java's diagnostic fires. The remaining `parse_meta`-signature tidy-up is a separate, still-open item and is *only* a tidy-up.
- [x] `BinaryDocValuesFieldUpdates` -- a whole second update type -- unported.
      **Closed by c7** (F-21): `write_binary_updates`/`read_binary_updates`/
      `binary_value_with_updates`/`binary_value_with_generations` mirror the
      numeric side exactly (ascending by doc, last write per doc wins, newest
      generation wins, `None` is `reset(doc)`), with `Some(vec![])` deliberately
      distinguishable from `None` (an empty `BytesRef` is a legal value) and a
      distinct codec name so a numeric overlay handed to the binary reader fails
      the header check. Reachable from the public API via
      `IndexWriter::update_binary_doc_value`. (Raised by b6; closed by c7.)
- [x] **HNSW was not ported at all**, and `.vec`/`.vem` were *this port's own
      format*, so vector fields did not interoperate with real Lucene in
      either direction. Closed by c5 (batch report:
      `docs/sweep/m2/c5-vectors.md`): both formats ported read and write,
      proven with a Java-writes/Rust-reads fixture (`GenVectors.java`, 4000
      documents, five fields) and a Rust-writes/Java-reads verifier
      (`VerifyVectors.java`, `verify-write-path.sh` now 16/16). Measured
      50k x 128: **47x** queries/sec, **244x** fewer distance computations,
      recall@10 0.9250 against Lucene's 0.9250 on the same data and
      parameters. (Raised by b7.)
- [x] `points_query.rs` still calls `decode_all_points` rather than b7's new
      `PointsReader::intersect` (577x on a 200k-point tree). Owner: b14.
      (`points_delete.rs` assigned to b11.)
      **CLOSED** -- duplicate of the points item above. `points_query.rs:287/295` uses `intersect`/`range_query`; `points_delete.rs:96` uses `range_query`.
- [x] Term-vector callers must supply each document's fields in ascending
      field-*name* order or real `CheckIndex` rejects the segment. Verify
      b9's flush path does. (Raised by b7.)
      **CLOSED by c34** -- duplicate of the term-vectors ordering item above; `IndexWriter::build_term_vectors_output` now sorts by field name and a tripwire test pins it.
- [x] b9's `segment_writer::flush_stored_only_segment` wrote a `.si` whose
      `files` set omitted the `.si` itself, where
      `Lucene99SegmentInfoFormat.write` calls `si.addFile(fileName)` before
      encoding. Every consumer that walks `SegmentInfo.files`
      (`IndexFileDeleter`, `CheckIndex`, our `checksum_verify`) was therefore
      blind to the file that names all the others. Fixed in the main session;
      `verify-write-path.sh` 14/14 against real Lucene 10.5.0.
- [x] **F-7 (from b12), production-path scoring divergence.** `docs/parity.md`
      has claimed since M1.5 that the search paths no longer use
      `FieldNorms::open`'s non-Java `avgFieldLength`. That is true only of the
      benchmark runner: `lucene-ffi`'s production entry point and `explain.rs`
      still use it, so every FFI-served search averages lossy decoded norms
      over live docs, where Java divides `sumTotalTermFreq` by a `docCount`
      that includes deleted documents. Parity row corrected by b12; the code
      fix is b13 (`field_norms.rs`) + b15 (`lucene-ffi`). **b15 half done**:
      the `lucene-ffi` production entry point now uses
      `FieldNorms::from_field_stats`; `explain.rs`'s own (non-FFI) call site
      and the benchmark runner remain for b13/b14.
      **CLOSED by c34.** The code half was already done (b15 moved `open_field_norms` onto `from_field_stats`; c6 made `avgdl` reader-wide; `explain.rs`'s and the benchmark runner's remaining sites are test-only or already migrated). What was still live was the *record*: `docs/parity.md`'s FFI row still described `open_field_norms` as ending in `FieldNorms::open`. c34 corrected that row. This entry was the last carrier of b12's F-7 claim.
- [x] `BooleanScorer` window/bucket bulk-OR **ported by c6** as
      `docid_set::WindowedDisjunction`, chosen the way
      `BooleanScorerSupplier.booleanScorer` chooses it, and measured (3.1x on a
      4-clause dense disjunction, 9.4x at 16 clauses, 24.8x with
      `minimum_should_match = 2`, no regression when sparse). c6 also corrected
      b12's premise: under `ScoreMode.TOP_SCORES` -- what
      `searcher.search(q, n)` uses -- Lucene 10.5.0 picks `MaxScoreBulkScorer`,
      **not** `BooleanScorer`, so this was never the mechanism behind the
      scored `or t0 t1 t2 t3` benchmark gap.
- [x] `Occur.FILTER` (b12's F-16). c6 checked: reachable **without** the scorer
      abstraction (a `filter` clause is a `must` clause whose score is dropped),
      roughly one focused batch across ~9 touch points. `TwoPhaseIterator` + a
      cost model (F-20) is not: it needs every clause expressible as
      `(approximation, matches(), matchCost())`, i.e. turning b12's per-shape
      free functions into a scorer enum. A milestone, not a batch.
      **CLOSED by c11** -- duplicate of the `Occur.FILTER` entry above. `TwoPhaseIterator` + a cost model is *not* closed and stays its own open item.

## `field_norms.rs`'s sparse `Vec<i32>` -- NOT a two-line fix (checked in main session; **fixed by c6**)

c2 handed over `lucene_search::field_norms::FieldNorms::sparse_doc_ids`
(`Option<Vec<i32>>`, `field_norms.rs:75`) as a two-line swap now that
`DisiCursor` is a real incremental cursor. It is not, and the blocker is worth
recording so the next owner does not rediscover it:

`DisiCursor::advance_exact` takes `&mut self`, but `FieldNorms::sparse_norm`
takes `&self`, and `FieldNorms` is passed as `&FieldNorms<'_>` into `rayon`
`par_iter` closures by `multi_segment.rs` (see its `norms: &[Option<&FieldNorms>]`
parameters and the `Sync` requirement documented at `multi_segment.rs:229`).
So the cursor cannot simply be stored behind a `RefCell` -- that would make
`FieldNorms` non-`Sync` and break the concurrent fan-out.

Three viable designs, in preference order:
1. Give the cursor to the *caller*: each rayon task owns its leaf's cursor for
   the duration of that leaf's scan, and `FieldNorms` stays immutable shared
   state. This matches Lucene, where `NumericDocValues` is a per-leaf,
   per-scorer object and the shared thing is the `SegmentCoreReaders` entry.
   Ripples through `explain.rs`, `multi_segment.rs`, `lib.rs`, `lucene-ffi`.
2. `&mut self` on `sparse_norm` and hold `FieldNorms` mutably per leaf --
   smaller diff, but wrong shape: it makes a cache look like state.
3. A thread-local or `Mutex` cursor -- rejected, it reintroduces per-lookup
   cost for a structure whose whole point is being free.

Option 1 is the right one and belongs in a batch that holds the whole search
crate, not a drive-by edit.

**c6 built option 1.** `FieldNorms` keeps only the region slice and stays
immutable/`Sync`; `FieldNorms::cursor()` (Lucene's
`LeafReader.getNormValues`) hands each scan its own `FieldNormsCursor`, and the
`&self` one-shot lookups survive as thin wrappers -- so `explain.rs` and
`lucene-ffi` needed no change at all, less ripple than predicted. The fan-out
is proven still concurrent by a test in which two rayon tasks share one
`&FieldNorms` and neither may finish until both have started. See
`docs/sweep/m2/c6-search-followups.md` finding F-2.

## Also still open in `lucene-search`

- [x] F-26 (b13): `field_norms()` computes `avgdl` per leaf where Java's is
      reader-wide. **Fixed by c6**: `DirectoryReader::avg_field_length` sums
      `IndexSearcher.fieldStats`' two counters over every leaf and
      `DirectoryReader::field_norms` builds every leaf's `FieldNorms` from the
      one value (`multi_segment::global_avg_field_length` is the
      `OpenSegment`-level sibling b13 asked for). It could not live in the
      search functions the way `global_term_stats` does: `avgdl` is consumed
      when the `FieldNorms` is *constructed*, before any search function sees
      it. Proven bit-for-bit against a new two-segment real-Lucene fixture.
- [x] `explain.rs:1687`'s invalid `dense_rank_power: 0` literal -- **fixed by
      c6** (`0xFF`, Java's "no rank table"). (`field_norms.rs`'s `0` was
      checked and is deliberate -- it is b13's rejection test, and c6 left it,
      extending that test to the `field_length` path as well.)
- [x] `lucene-ffi/src/writer.rs`'s two delete/update call sites were left on
      `IndexWriter`'s pre-`DeleteQueue` API for several hours after
      `lucene-index` moved to the `Term`-based one, so `lucene-ffi` did not
      compile. c6 noticed it (it blocked half c6's gate) and deliberately did
      **not** migrate it: the change deletes
      `open_all_segment_sources`/`build_delete_sources` (~150 lines of eager
      delete resolution) and alters when a delete takes effect, which is the
      owning batch's design decision, not a mechanical edit. They have since
      landed it; `lucene-ffi` is green (441 tests).
- [x] `crates/lucene-index/src/check_index.rs:1698-1699` tripped two
      `clippy::doc_lazy_continuation` lints, which fails `-D warnings` for
      **every** crate depending on `lucene-index` -- clippy lints path
      dependencies as well as the named package. Raised by c6 (it was the last
      blocker on c6's gate) and left for that file's owner rather than fixed
      mid-edit; since resolved. c6's gate is green, exit 0.
- [x] `lucene-search/src/lib.rs` sits at **90.5% line coverage** with both
      `lucene-search`'s and `lucene-ffi`'s suites running (measured by c6 after
      a `cargo llvm-cov clean --workspace`; a stale profile reports it ~26
      points lower still). Every other file in the two crates is above the 95%
      bar. It is b12's file and 8,650 regions; the gap predates c6, which added
      only covered lines. Worth its own item.
      **CLOSED.** Measured by c34 on the current tree: `lucene-search/src/lib.rs` is at **96.78% lines / 95.75% regions**, and no file in the workspace is below 95% lines. c11's 95.20% has held and grown.
- [x] The six multi-segment **FFI** entry points passed `vec![None; n]` for
      norms, so every multi-segment FFI search scored unnormed where real
      Lucene always applies the field's real per-document lengths. Found and
      fixed by c6 while wiring F-26; the module doc that justified it ("task
      #45's `DirectoryReader` carries no `.nvm`/`.nvd`") had been made untrue
      by b13.
- [x] `benchmarks/rust-runner/src/micro.rs` did not compile after b2 made
      `for_util::for_encode` pack in place (`&mut [u32; BLOCK_SIZE]`), which
      blocked c1 from *measuring* its `DirectoryReader::open` "after" number
      rather than deriving it. Fixed in the main session: the fixture is
      encoded from a scratch copy so `values` stays the pristine expectation
      the round-trip guard compares against. Binary builds and runs.

## Next follow-up batches (queued, blocked only on a free slot)

- [x] **c10 -- vectors end-to-end wiring.** c5 ported the flat format and the
      HNSW graph and proved both against real Lucene, but nothing in
      `lucene-index` can add a vector field yet, and the codec-level merge
      write path (`mergeOneField`/`IncrementalHnswGraphMerger`,
      `mergeOneFlatVectorField`) is unported -- so the subsystem is correct but
      unreachable end to end. Also: `FLOAT16`, the index-sort write path, and
      moving `SplittableRandom`/`TernaryLongHeap`/`NumericUtils` into
      `lucene-util`. Blocked on c7 (owns `index_writer.rs`) and c8 (owns
      `merge.rs`).
      **CLOSED.** `IndexWriter::set_vector_field`/`add_vector_field` exist (`index_writer.rs:1883` and its sibling), the codec merge write path is ported (`vectors::FlatVectorsWriter::merge_one_flat_vector_field`, `hnsw_vectors::merge_one_field` at `hnsw_vectors.rs:272`) and is reached from `lucene-index` (`merge.rs:3339`, and the `GraphMergeSource` block at `:3378`), and `SplittableRandom`/`TernaryLongHeap`/`NumericUtils` now live in `lucene-util` (`lucene-util/src/lib.rs:18/20`, `numeric_utils.rs`). **`FLOAT16` was a `main`-ism**: 10.5.0's `VectorEncoding` has exactly `BYTE` and `FLOAT32` (c18's version audit, recorded at `check_index.rs:268`), so there is no third encoding to port.
- [x] **c11 -- `Occur.FILTER`.** Done. `BooleanQuery` has a fourth clause list;
      matching, scoring (zero contribution, no perturbation of the `f32`
      summation order), `explain`'s Java-verbatim `# clause` arm, and the five
      `FILTER`-specific `rewrite` rules are ported and pinned bit-for-bit
      against real `IndexSearcher` (`scoring.boolean.filter*`).
      `TwoPhaseIterator` + a cost model remains NOT reachable and is untouched:
      it needs every clause as `(approximation, matches(), matchCost())`, i.e.
      turning b12's per-shape free functions into a scorer enum -- that is a
      milestone, not a batch. See `docs/sweep/m2/c11-occur-filter.md`.
- [x] **c12 -- b14's remaining feature gaps.** Done. Facets gained the whole
      missing layer: `OrdinalMap` (so cross-segment faceting exists at all --
      it was *unavailable*, not wrong), `FacetsConfig`/`DimConfig`,
      `pathToString`/`stringToPath`, `SortedSetDocValuesReaderState` with both
      the flat `OrdRange` and the hierarchical `DimTree`, and
      `getTopChildren`/`getAllChildren`/`getSpecificValue`/`getAllDims`/
      `getTopDims` including Java's `-1` for a multi-valued dim without
      `requireDimCount`; plus multi-valued range counting and `totCount`. All
      pinned against real `lucene-facet` output over a new three-segment
      `GenFacets.java` fixture (18 differential tests). `MultiFieldQueryParser`
      ported with the disjunction at the *leaf* (so `AND_OPERATOR` gives Java's
      `+(title:t body:t)`, not `+(title:t1 title:t2)`) and per-field boosts.
      The highlighter's `BreakIterator` turned out to be a **CORRECTNESS**
      finding: the hand-rolled abbreviation list diverged from the JDK, which
      applies no abbreviation suppression at all -- replaced with UAX #29 via
      `unicode-segmentation` (already a workspace dep), six JDK outputs pinned;
      `SplittingBreakIterator` and the `FieldOffsetStrategy` selection +
      postings/analysis offset sources ported. `FieldExistsQuery` (doc-values
      source) and `IndexOrDocValuesQuery`'s planner ported at the layer that
      holds the readers -- and b14's diagnosis corrected: `Occur::FILTER`
      changes nothing here, and the blocker was never a `Clause` variant but
      `resolve_clause_docs`' signature carrying no doc-values/norms/vector
      input. See `docs/sweep/m2/c12-search-features-2.md`.
- [ ] **Mechanical gates for two defect shapes this sweep keeps re-finding.**
      (a) A clippy `disallowed_methods` entry on the free
      `doc_values::numeric_value`/`binary_value`, naming
      `NumericReader`/`BinaryReader` as the sanctioned multi-lookup API: the
      "call the re-deriving free function once per document" defect has now
      appeared twice (b13's `soft_deletes::effective_live_docs`, c14's column
      merge) and is grep-able. (b) A gate on `Lucene90_\d`-shaped string
      literals outside `index_writer::per_field_codec_suffix` and test
      modules, which is how c14's hardcoded per-field suffix (F-12) got in.
      Repo-wide tooling, not any one batch's files. (Raised by c14's Tier-2
      review.)
- [x] **c14 -- the doc-values-update on-disk format.** Done. c7's F-15 is
      closed: the `.dvu` overlay this port invented is no longer written into
      any index, replaced by Lucene's own generational `.dvm`/`.dvd`/`.dvs` +
      `.fnm` representation, with a Java fixture in the read direction and a
      `CheckIndex`-running verifier in the write direction. See
      `docs/sweep/m2/c14-dv-updates-format.md`.
- [~] **c13 -- c1's caller migration.** Move callers to blocktree's `try_*`
      forms so a corrupt block surfaces as an error rather than "not found";
      ~~`directory_reader` -> `open_shared`~~ (**done by c12**, 4.8x); split
      term iteration from stats.
      **OBSOLETE as a batch entry.** Its three parts are tracked individually and two are still open: `directory_reader -> open_shared` is **done** (c12, 4.8x), while the `try_*` caller migration and the term-iteration/stats split are each their own open item above. Keeping a batch-shaped duplicate of two live items is how the same work gets planned twice.
- [x] `lucene-search/src/lib.rs` is at 90.52% line coverage, below the 95% bar
      -- a pre-existing gap in b12's file. (Raised by c6.) **Closed by c11 at
      95.20%**; every file in `lucene-search` is now above the bar. The two
      holes were the single-scoring-clause fast path plus the
      `stream_constant_score_clause`/`expanded_terms` streaming union it is the
      only caller of (nothing reached them: the wildcard-family fixture suites
      all call `search_prefix_query_scored` and friends directly, never through
      a `BooleanQuery`), and both lazy paths' block-max pruning branches (the
      5-document `body` field can never fill a top-`n` queue; the 300-document
      `big` and 8,250-document `l1` fields can).

## Open items raised by c15

- [x] **`.doc`'s level-0/level-1 skip records carry the `.pos`/`.pay` file
      pointers (`posEndFPDelta`, `posBufferUpto`) and this port parses and
      discards them.** Java's `advance(doc)` seeks `.pos` straight to the block
      holding the target document's occurrences and never sums a frequency;
      c15's wanted-documents walk still starts at the term's `pos_start_fp`,
      steps block by block (one token byte and a seek per 256 occurrences,
      after c15 -- it used to `PForUtil`-decode every one), and needs the
      term's whole doc list to know which occurrence range a document owns.
      That doc-list decode is the whole of the residual cost: 6.87 ms of the
      ~26 ms a five-million-document term's single-document highlight now
      takes, against 1.30 s before. The blocker is that using the pointers
      means accumulating them across every level-0 header from the term's
      start and threading a `.pos`/`.pay` cursor through the shared block
      header reader and `LazyDocsCursor`. (Raised by c15, §F10.)
      **Closed by c20**: the pointers are now read *and* written, and
      `read_occurrences_for_doc` / `FieldTerms::occurrences_for_doc` decode no
      doc list at all -- 34.4 ms -> 1.57 us for the first document of a
      five-million-document term, A/B/C in one build. Two things c15 could not
      have known: the write side had to be fixed first (it emitted no pos/pay
      sub-fields at all and refused `docFreq >= BLOCK_SIZE` on a positions
      field because of it), and once the walk was gone the *level-1 impacts
      decode* (b5 F7, recorded there as "bounded and small") became the
      dominant residual, because reaching a late document crosses hundreds of
      spans. Both fixed. See `docs/sweep/m2/c20-postings-skip.md`.
- [ ] **`norms::parse_meta`'s signature still differs from Java's**
      `readFields(meta, infos)`. c15 closed the *behaviour* gap (b6 #4, c7
      F-23) with an additive `norms::validate_fields(&Norms, &FieldInfos)`,
      called from the segment reader's open and from `check_index` -- the two
      places Java's diagnostic fires. What is left is a pure tidy-up across 23
      call sites in four crates, worth doing when a shared `FieldInfos` open
      lands and not before. (Raised by c15, §F14.)

## Open items raised by c12

- [x] **`highlighter::offsets_from_postings` decodes the whole term's
      postings** where Java's `PostingsOffsetStrategy` does
      `postingsEnum.advance(doc)` and walks one document. Invariant #3, and the
      fixture (docFreq 2-3) cannot see it; on a real corpus it is a full
      postings sweep per highlighted field. **Blocked outside
      `lucene-search`**: `FieldTerms::positions` is the only accessor in this
      port that returns offsets and it returns them for every document, while
      `FieldTerms::positions_for_docs` -- which does skip -- returns positions
      *only*, because `read_positions_for_docs` decodes each block's offset
      pairs into a scratch buffer and discards them. Offsets are delta-coded
      cumulatively *within* a document, so a skipping decoder has to carry
      per-document start-offset state it deliberately does not keep. The fix is
      an offsets-carrying sibling of `read_positions_for_docs` in
      `lucene-codecs/src/postings.rs` (c8's file, in flight while c12 ran).
      c12 removed the duplicate doc-list decode on top of it and documented the
      remainder on the function. (Raised by c12's Tier-2 review.)
      **Closed by c15**: `postings::read_occurrences_for_docs` +
      `FieldTerms::occurrences_for_doc`, 43x measured on a
      five-million-document term. c12's stated blocker was wrong in a way
      worth recording -- the start-offset accumulator resets at every
      document's *first* occurrence (`Lucene104PostingsWriter.startDoc`), so a
      document's offsets are self-contained and nothing carried across the
      skipped documents is needed. That reset is also what makes skipping
      sound at all.
- [ ] **`OrdinalMap::build` materializes every segment's term list**, where
      Java's `OrdinalMap.build(owner, TermsEnum[], weights, ratio)` streams
      `TermsEnum`s and never holds a dictionary. Needs a cursor API over the
      doc-values terms dictionary, which does not exist --
      `terms_dict::decode_all_terms` reads it whole. (Raised by c12's Tier-2
      review.) **Measured by c29** (`lucene-search/examples/ordinal_map_memory.rs`,
      17-byte terms): 5 segments x 1 M terms costs **267 MB** for the term
      lists against **52 MB** for the map, 319 MB peak -- so the input is ~5x
      the output and ~84% of the peak, i.e. this is the *larger* of c12's two
      recorded `OrdinalMap` divergences, not the aside it was filed as. Owner
      `lucene-codecs/src/terms_dict.rs`: a `TermsCursor` yielding one term at a
      time over the prefix-compressed blocks `decode_all_terms` already walks,
      then `facets.rs` and `OrdinalMap::build` take iterators.
- [~] **`docs/parity.md` has no mechanical staleness check.** c12's review
      found three rows still declaring "still out of scope" for symbols a later
      row claimed as ported (`OrdinalMap`/`FacetsConfig`/hierarchical dims,
      `BreakIterator`-grade segmentation, `MultiFieldQueryParser`) -- the
      predictable consequence of the append-only convention. All three are
      fixed, but the next batch will recreate the pattern. Worth an `xtask`
      that flags a row saying "out of scope"/"not ported" about a symbol
      another row claims as ported. (Raised by c12's Tier-2 review.)
      **OBSOLETE -- superseded by a design decision, and the decision is written down.** `scripts/check-parity.py` exists and *deliberately declines* to automate contradiction detection: its module docstring says a heuristic over the status text "flags fourteen of those for every real problem it finds", because a class routinely has several honest rows (read side and write side, a scoped first cut and a later widening). What it does mechanically instead has no false positives -- every Rust path in a row must exist, and every ported file must have a row -- and `--verbose` lists the multi-row classes for a human to scan. c12's three stale rows were fixed by hand; c34 fixed a fourth the same way.

## Open items raised by c29

- [ ] **`PointValues.estimatePointCount`'s BKD walk.** `estimate_doc_count`'s
      arithmetic is ported (`lucene-search/src/points_query.rs`) and is what
      `IndexOrDocValuesQuery`'s planner consumes, but the walk that produces
      its input is not: it needs each node's subtree `size()` *mid-traversal*,
      and `lucene-codecs/src/points.rs` exposes none --
      `IntersectVisitor::compare` sees cell bounds but no size, `intersect`
      visits every document of a fully-inside cell (exactly the cost the
      estimate exists to avoid), and the node-id walk is private. Owner
      `lucene-codecs`: a `PointsReader::estimate_point_count(field, &mut V)`
      beside `intersect`, reusing `IntersectCtx`/`intersect_node`'s existing
      `node_id` bookkeeping plus `BKDReader.IndexTree.size()` (subtree leaf
      count derived from the node id, times `max_points_in_leaf_node`, clamped
      at `point_count`). Nothing in `lucene-search` moves. (Raised by c29;
      originally b14 §1.4, then c12 §5.4.)
- [x] **`FieldExistsQuery.count`'s live-doc arithmetic, and `rewrite`'s
      whole-reader decision.** The per-leaf predicate is ported
      (`doc_value_query::field_exists_leaf_is_complete`); what is missing is
      the layer above it, which needs a leaf list and a `numDocs` -- i.e. a
      reader-level query object this port does not have (its queries are
      functions). Note when taking it that Java's norms branch reads
      `getDocCount(field)`/`maxDoc()` off the **top-level** `IndexReader` while
      the other two branches read the *leaf*'s, inside the same per-leaf loop;
      that asymmetry is Java's and is documented on the ported function.
      (Raised by c29.)
      **CLOSED by c37-search-behaviours** (`weight_count::{FieldExistsLeaf,
      count_field_exists_leaf, field_exists_rewrites_to_match_all_docs}`,
      `SegmentReader::field_exists_leaf`). No reader-level query object was
      needed -- the rules are pure functions of counts the reader already has.
      See open item 6 above.
- [x] **`scripts/gen-fixtures.sh` cannot regenerate one generator.**
      **Closed by c32.** `--only <Gen*>` (repeatable, `Gen` prefix optional,
      `--list` names them) runs one generator plus all six appenders and
      touches nothing else -- proven against a `sha256sum` of all 684 files
      under `fixtures/data`, twice. A bare full run now **refuses** and needs
      `--all`; preserving segment ids was checked and is not possible without
      patching Lucene (`randomId()` is called inside `IndexWriter`), so the
      honest fix is the refusal plus `fixtures/segment-ids.txt`, a committed
      record of every index's id that `--check` re-derives and diffs. `--check`
      also now compares each manifest's **key set** -- `blocktree_index/
      manifest.properties` is non-deterministic, so the old byte check was
      blind to exactly the damage c29 hit. Both damage modes demonstrated on a
      scratch copy. (Raised by c29.)

## Open items raised by c11

- [x] **`search_boolean_query_scored_maxscore_with_stats`' body is dead code**
      (~97 lines, the largest remaining uncovered block in
      `lucene-search/src/lib.rs`). **Closed by c12: deleted.** The claim was
      verified both by reading the two predicates (they are exactly
      complementary, and `try_disjunction_lazy` *accepts* the absent-field /
      absent-term shapes the body declined, so those never reached it either)
      and by a coverage report showing all 78 executable lines uncovered by the
      whole pre-batch 899-test suite. Deleted rather than revived because the body's own
      doc comment recorded it as 4-5x slower than the lazy union, which has
      since grown block-max pruning. The public entry points and the FFI are
      unchanged; a 10-shape x 4-`top_n` matrix test pins byte-identical output
      against the plain scored path. Every shape it can handle,
      `try_disjunction_lazy` has already handled by the time control reaches
      it, and every shape it declines it declines for a reason that also makes
      the body fall back -- including the pulsed-term case, which both bail on.
      Its own doc comment already says it is 4-5x slower than the lazy union
      and "prefer the plain scored entry point". Either delete the body (the
      public entry points keep working, they just become
      `try_disjunction_lazy`-then-fallback) or revive it as block-max WAND.
      Left alone by c11 as out of scope for a `FILTER` batch.
- [x] **A filter-only conjunction cannot prune.** c11 made it take the lazy
      leapfrog (129.5ms -> 58.7ms on the benchmark corpus) but deliberately
      switched block-max pruning off for it: every document scores 0, so the
      summed bound is 0 and `bound <= threshold` would skip on a *tie*. Java
      does prune this shape (`FilterScorer`'s `getMaxScore() == 0` against
      `TopScoreDocCollector`'s `Math.nextUp(bottom)`), and this port's
      `bound <= threshold` is algebraically the same rule -- so the guard is
      probably removable, but it wants a differential test against real
      `IndexSearcher` for a filter-only top-`n` query, which the current
      fixture segment is too small to produce.
      **CLOSED by c37-search-behaviours.** The guard is gone; Java prunes the
      tie too. The fixture segment was not too small -- `body` is, `big` (300
      documents) is not. 44.20 ms -> 2.00 ms. See open item 23 above.
- [x] **`lucene-ffi`'s boolean-query entry points have no `FILTER` list.** Four
      exported functions take three flat `(fields, field_lens, terms,
      term_lens, count)` array groups, one per occur. Adding a fourth is a
      C-ABI signature change plus matching Java/Panama bindings -- b15's call,
      not a `lucene-search` change.
      **CLOSED by c13.** The four entry points now take one occur-tagged, parent-indexed clause array: `clause_occurs: *const u8` with `OCCUR_MUST=0`/`OCCUR_FILTER=1`/`OCCUR_SHOULD=2`/`OCCUR_MUST_NOT=3` (`lucene-ffi/src/query.rs:292-299`, dispatched at `:336-338`). The pre-c13 three-bucket shape survives as `legacy_boolean_abi.rs`, a test-only bridge that proves the change behaviour-preserving.
- [x] `search_boolean_query_scored_maxscore_with_stats`'s body is provably dead
      code (~97 lines): c11 reports it and `try_disjunction_lazy` are exactly
      complementary. **Verified and deleted by c12** -- see the item above.
      (Raised by c11.)
- [x] `lucene-ffi`'s boolean entry points still take three clause arrays and
      cannot express a FILTER clause -- a C-ABI change, so it is b15's call
      rather than c11's. (Raised by c11.)
      **CLOSED by c13** -- duplicate of the entry above.
- [x] A filter-only query cannot prune under a top-`n` collector (no score to
      bound): 54.0 ms vs 10.0 ms measured. That gap is pruning, not filtering.
      (Raised by c11.) **CLOSED by c37-search-behaviours** -- duplicate of the
      entry above.

## A defect found in the main session by DELETING a test workaround

c8 converted four `debug_assert` panics in `postings.rs` to typed errors, so
c9's negative-control test no longer needed its `catch_unwind` + silenced
panic hook. Removing that workaround also removed its `Err(_) => continue`
arm -- which had been *excluding* every panicking byte-flip from the
assertion -- and the test immediately failed on a fifth, separate panic c8's
sweep had missed: `postings.rs`'s lazy-cursor tail block sliced its fixed
256-entry array by `doc_count_left`, which starts from a `docFreq` read off
disk. `Lucene104PostingsReader.refillRemainder` asserts
`docCountLeft >= 0 && docCountLeft < BLOCK_SIZE`; here that is not an
invariant of our own arithmetic, so a corrupt `.tim` panicked
("range end index 744 out of range for slice of length 256") instead of
reporting corruption. Now a typed `Corrupted` error.

The lesson worth keeping: a test that swallows a failure mode stops testing
it. That `continue` made the corruption scan look exhaustive while silently
skipping the cases that crashed. The test now runs every byte flip -- and in
0.36 s rather than ~10 CPU-minutes.

## Follow-up batches, continued

| Batch | Scope | Result |
|---|---|---|
| c15-postings-api | occurrences API, `checkDocValueSkipper`, norms field validation | swept (4C/3M/4P); highlight 1.32 s -> 18.3 ms (72x) |
| c16-knn-query | `KnnFloatVectorQuery` in `lucene-search`, multi-segment, filtered | swept (4C/7M/3P); 16 differential tests green first run |
| c17-index-sort | `setIndexSort` + sorted flush reachable | swept (4C/10M/1P); sort free at 16 MB buffer |
| c18-version-audit | 32 classes differing 10.5.0 vs `main` | 26 CLEAN, 2 CORRECTNESS + 2 doc defects fixed |
| c19-coverage-hardening | arithmetic gate + weak-floor controls | swept (5C/2M); gate found 6 defects 8 hand audits missed |
| c20-postings-skip | `.pos`/`.pay` skip pointers | swept (1C/3M/3P); highlight 2.77 s -> 1.57 us |
| c21-hnsw-seeded | seeded re-entry, `acceptOrds` down, filtered KNN ABI | swept (2C/7M/1P); found a defect in c16's code |
| c22-sorted-merge | sort-preserving merge across every format | swept (11C/8M/3P); sorted stored-fields merge 2004 ms -> 13.2 ms (152x) |
| c23-positions-writer | `IndexWriter` positions + cross-engine verifier | running |
| c24-arith-codecs | burn down 26 unaudited `lucene-codecs` modules | swept (11C/1M/2P/3I); **26 -> 12** modules marked, ~272 sites resolved; F1: two flipped bytes in a `.fdm`/`.tvm` header reserved 51 GB and **aborted** |
| c25-check-index-coverage | ~110 never-fired `CheckIndex` failure arms | swept (3C fixed); **97.25%**, 45 arms driven, **10 deleted as unfirable**, and a check that could not run no longer reads as one that passed |
| c26-merge-completeness | a mechanical gate for "the merge opens every format the flush writes" | swept (3C/4M/1P); gate reproduces c22's 14/22/23 by name; both merge carry-overs closed |
| c28-arith-index | burn down 8 unaudited `lucene-index` modules | swept (6C/3M/2P/2I); **11 -> 3** modules marked, 42 sites resolved; every generation off `segments_N` capped **on the read, file-name and write paths** (a cap applied only on the way in still let a trash file name manufacture a commit this port writes and cannot reopen -- caught by Tier-2, closed by a round-trip property test); **3 of 6 CORRECTNESS came from the allocation/indexing hand-check the lint cannot do**, and the two `FixedBitSet` instances of it produced a new greppable crate rule in `docs/arithmetic-gate.md` |
| c30-finish-index | the last check-index arms + the last 3 arithmetic-gate modules | swept (7C fixed, 1 defensive, 1 arm deleted, 5 kept as total error handling); **`check_index.rs` 98.58%**, **arith burn-down 3 -> none (workspace fully audited)**, 1 verified SIGABRT and 1 `FixedBitSet` panic in the *merge* path, both from the hand-check the lint cannot do; full gate green |

## Two cross-cutting findings worth carrying forward

**1. The Java source of truth was the wrong version for most of the sweep.**
`/home/tuong/work/lucene` sits on `main`, 4574 commits and 1261 files past the
pinned `releases/lucene/10.5.0`. 32 ported classes differ. c14 found this
independently when `main`'s `ReadersAndUpdates` turned out to use a completely
different design; c18 then audited all 32 and found 26 clean, 2 real
correctness defects (`TieredMergePolicy`'s `maxMergeAtOnce`, which `main`
deleted entirely, and `Lucene104PostingsWriter`'s dense-block choice, which an
earlier batch had actively moved onto `main`'s form and pinned with a test).

What limited the damage: `scripts/lib-lucene-jars.sh` pulls **10.5.0 jars**, so
every fixture and verifier was ground truth for the right version regardless of
what source a batch read. Paths a fixture covers were safe; paths without
fixture coverage were not -- which is exactly where both defects were.
The pinned source now lives read-only at `/home/tuong/work/lucene-10.5.0`.

**2. Manual auditing does not catch the panic-on-corrupt-input class.**
Found by hand in b1, b2, b4, b6, b7, c8, c15, c19 and twice in the main
session. Two data points settle it: c15 ran a deliberate audit of `postings.rs`,
fixed eight sites, and its Tier-2 review found **eight more it had missed**; and
the main session found a fifth `debug_assert` panic in that same file only by
*deleting a test workaround* whose `Err(_) => continue` arm had been excluding
every panicking byte-flip from the assertion. c19's mechanical gate then found
six more, including two allocation aborts -- which `catch_unwind` cannot
intercept, so they take the JVM down through the FFI.

## Queued after c24 releases `lucene-codecs`

- [x] **c26 — merge format-completeness gate.** c22's Tier-2 review traced
      three of its findings to one root: *nothing mechanically checks that
      `execute_merge` opens every format the flush can write*. Before c22,
      `execute_merge` supplied no norms (so every merged BM25 score was
      wrong -- c4's standing carry-over), no doc values, no vectors, and
      dropped `has_blocks`. A test that enumerates the formats `flush` can
      write and asserts `execute_merge` handles each would have caught all
      of it, and would catch the next format added. Same shape as c19's
      arithmetic gate and `scripts/check-parity.py`.
      **CLOSED by c26.** `merge.rs::check_format_coverage` refuses a merge whose caller never opened a format a source's own `.si` lists, `IndexWriter::execute_merge` carries the matching `debug_assert!`, and `index_writer.rs:15409`'s `merge-format-completeness` test asserts one permutation reaches every format the flush can write (`:15474`) and that the merged `.si` carries every format its sources' did (`:15488`).
- [x] Multi-field norms in a merge (needs `norms::write_dense_fields`).
      **CLOSED by c26.** `norms::write_fields` (`norms.rs:392`) writes one or more norms fields into one `.nvm`/`.nvd` pair, `merge.rs:2022` calls it, and `Error::TooManyNormsFields` is gone (`merge.rs:2897` records its removal). `IndexWriter::add_norms_field` is the accumulating flush-side counterpart.
- [x] Generational doc values are still un-mergeable. (Raised by c22.)
      **CLOSED by c26.** `merge.rs:2449` documents the change: a source without the field's column now contributes no value (Java's own behaviour) instead of raising `DocValuesFieldMissingInSource`, which is exactly what had kept a doc-values **update**'s generational column out of the merge policy. `check_format_coverage` is the replacement detector for the class the old error was standing in for.

## Test-hygiene defect found by c28 (closed by c32, three files handed off)

- [x] **The test suites leak a temp directory per test and never clean up.**
      **Closed by c32**: `lucene_util::test_support::TempDir`, one shared RAII
      guard that removes on `Drop` but **keeps on panic** so a failing test
      stays debuggable, in `lucene-util` because the architecture skill's
      downward-only graph puts a helper every crate's tests need at the bottom
      (a `lucene-test-support` sibling would be the forbidden sibling edge, five
      times over). 30 of 33 call sites migrated; a four-crate test run leaks
      **26** directories where it used to leak roughly a thousand, and 2 of the
      26 are the deliberate `#[should_panic]` keeps. Still to do, additive so
      they keep working as-is: `check_index.rs` (c30, 68 call sites -- the
      biggest single leaker), `checksum_verify.rs` (c30, 3), `fst.rs` (c31, 1
      inline site). Details in `c32-fixture-tooling.md` §2.4.

      The original report:
- [x] **(original) The test suites leak a temp directory per test.**
      c28 hit `/tmp` (a 16 GB tmpfs) at **100% full** from ~21,000 leftover
      `lucene-*-test-*` directories, which blocked its own work; the main
      session removed 7,016 dirs older than 30 minutes and left the newer ones
      alone in case a running batch owned them.
      **CLOSED.** The three files c32 handed off have landed: `check_index.rs:4703`, `checksum_verify.rs:270` and `fst.rs:5421` all use `lucene_util::test_support::TempDir`. The `[x]` c32 entry above is now true of the whole tree.

      Root cause: every crate has its own ad-hoc `fn tempdir()` returning a
      bare `PathBuf` built from `std::env::temp_dir()`, with nothing that
      removes it. At least ten files define or use one
      (`lucene-store/src/{directory,index_output}.rs`,
      `lucene-index/src/{check_index,term_delete,deletes,index_file_deleter}.rs`,
      `lucene-search/src/multi_segment.rs`, `lucene-codecs/src/fst.rs`,
      `lucene-ffi/src/writer.rs`, plus several integration tests).

      Fix: one shared RAII guard in a test-support module that removes the
      directory on `Drop` (and keeps it on panic, so a failing test is still
      debuggable), then migrate every call site. Deferred only because the
      call sites were spread across four crates that concurrent batches held;
      it is a small, mechanical change once the tree is quiet.

      Worth doing rather than recording: a full `/tmp` fails tests for reasons
      unrelated to the code under test, which is the most expensive kind of
      flake to diagnose.
