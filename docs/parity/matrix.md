# Supported matrix (M6 sweep, 2026-09-29; M7/M8 updates)

[Index](../parity.md). What this port supports end to end, and what it
deliberately does not. Each entry names the rows that carry the detail (by
their Java column); where a row and this summary disagree, the row is right.
In OpenSearch's vocabulary (query types, aggregations, settings, fallback
reasons): [`opensearch-native-queries.md`](../opensearch-native-queries.md),
[`opensearch-engine.md`](../opensearch-engine.md).

**Codec: Lucene 10.5.0's default, `Lucene104`, and only it.** R = read,
W = written by `IndexWriter`; "verified" = real Lucene reads the Rust bytes
(`scripts/verify-write-path.sh`, `scripts/verify-interop.sh`).

| Format (10.5.0 class) | Files | R | W | What differs or is missing (rows) |
|---|---|---|---|---|
| `Lucene104PostingsFormat` + `Lucene103BlockTreeTermsReader`/`Writer` | `.doc` `.pos` `.pay` `.psm` `.tim` `.tip` `.tmd` | yes | yes, verified; byte-identical to Lucene (`blocktree_byte_identity_fixture.rs`: two fixtures, suffix compression and the zigzag singleton branch included) | nothing for the format (M7); the whole-term *materialising* position readers refuse a stream past `u32::MAX` entries (16 GiB in memory), where the cursor and per-document walks take any `totalTermFreq` (rows `codecs/lucene104/Lucene104PostingsReader`..., `...(.doc/.tim/.tip/.tmd write side)`) |
| `Lucene90StoredFieldsFormat`, both modes | `.fdt` `.fdx` `.fdm` | yes | yes, verified; `BEST_COMPRESSION` byte-identical to Lucene (M7) | nothing (rows `...Lucene90CompressingStoredFieldsWriter`) |
| `Lucene90DocValuesFormat`, all five types, skip index | `.dvm` `.dvd` `.dvs` | yes | yes, verified | nothing: `doBlocks` byte-identical since M7 (rows `codecs/lucene90/Lucene90DocValuesConsumer`...) |
| doc-values updates (generational `.fnm`/`.dvm`/`.dvd`) | `_N_gen.*` | yes | yes, verified | `check_index` decodes every field from its current generation's column (row `index/CheckIndex`) |
| `Lucene90NormsFormat` | `.nvm` `.nvd` | yes | yes, verified | -- (rows `codecs/lucene90/Lucene90NormsFormat`, `...NormsConsumer`) |
| `Lucene90PointsFormat` (BKD version 10) | `.kdm` `.kdi` `.kdd` | yes | yes, verified; byte-identical to Lucene per field (M7) | older BKD versions rejected; a flush writes fields in `FieldInfos` number order where Java walks `IndexingChain`'s field hash (rows `codecs/lucene90/Lucene90PointsFormat`..., `...PointsWriter`...) |
| `Lucene90TermVectorsFormat` | `.tvd` `.tvx` `.tvm` | yes | yes, verified | refused by the OpenSearch engine; opened by `SegmentReader` (`term_vectors_reader`, `LeafReader::term_vectors`) since M7 (rows `codecs/lucene90/Lucene90TermVectorsFormat`..., `index/IndexReader.getTermVector`) |
| `Lucene90LiveDocsFormat` | `.liv` | yes | yes, verified | -- |
| `Lucene90CompoundFormat` | `.cfs` `.cfe` | yes | yes, verified: flushes follow `useCompoundFile`, merges the merge policy's `useCompoundFile` (M7) | this writer's `useCompoundFile` defaults to `false` (Java's to `true`); equal-length sub-files may be packed in a different order than Java's heap pops them (the `.cfe` maps names, so readers do not care) (rows `codecs/lucene90/Lucene90CompoundFormat.write`, `...CompoundReader`) |
| `Lucene99SegmentInfoFormat` (with index sort) | `.si` | yes, every `SortFieldProvider` | yes, verified | -- (every indexable sort kind, both directions; rows `index/SegmentInfo.indexSort`..., `index/IndexWriterConfig.setIndexSort`...) |
| `Lucene94FieldInfosFormat` | `.fnm` | yes | yes, verified | -- |
| `SegmentInfos` | `segments_N`, `pending_segments_N` | yes | yes, two-phase commit | -- (rows `index/SegmentInfos`, `index/SegmentInfos.write`) |
| `Lucene99HnswVectorsFormat` + `Lucene99FlatVectorsFormat` | `.vec` `.vemf` `.vem` `.vex` | yes | yes, verified | `FLOAT32` and `BYTE` (all of 10.5.0's encodings); no scalar-quantized formats; refused by the OpenSearch engine (rows `codecs/lucene99/...`) |

**M7 writer benchmarks** (`scripts/bench-micro.sh`; quiet machine 2026-10-01,
[`docs/benchmarks/m7-2026-10.md`](../benchmarks/m7-2026-10.md)):
`term_dict_write` 1.23x (`ids_1m`) / 1.44x (`words_200k`); `stored_fields_write`
over `GenStoredFieldsDeflate`'s documents 1.11x `BEST_COMPRESSION`, 2.22x
`BEST_SPEED`. From the loaded 2026-09-30 run (noise floors 2.5-4.3x):
`points_write` (flush/merge through `BkdWriter`) 6.27x/3.76x, `dv_merge` (with
`doBlocks`) 3.02x.

**Search (Lucene level).** The scorer tree (R1) runs `TermQuery`,
`BooleanQuery` (every occur, nesting, `minimumNumberShouldMatch`),
`ConstantScoreQuery`, `BoostQuery`, `DisjunctionMaxQuery`, match-all/none,
`PhraseQuery` (exact and sloppy), `TermInSetQuery`, `PrefixQuery`,
`WildcardQuery`, `RegexpQuery`, a one-dimension points range and
`FieldExistsQuery`, with bulk scorers, block-max pruning and a per-segment
query cache (rows `search/BooleanWeight`..., `search/Weight`
(`DefaultBulkScorer`)..., `search/PhraseScorer`...,
`search/AbstractMultiTermQueryConstantScoreWrapper`...,
`search/PointRangeQuery` in the scorer tree, `search/LRUQueryCache`
(`CachingWrapperWeight`...)). M7 added: `FuzzyQuery` and `MultiPhraseQuery`
(scored), explicit phrase positions, `SynonymQuery`, `CombinedFieldQuery`,
`NGramPhraseQuery`, `BlendedTermQuery`, every `MultiTermQuery` rewrite method,
`TermRangeQuery`, `AutomatonQuery`, the Indri, log-odds fusion and Bayesian
queries, N-dimensional point ranges and sets, doc-values and index-sort
ranges, KNN results as a clause, seeded, patience and similarity-threshold
vector queries ([search-queries.md](search-queries.md)). Outside the tree, as standalone
functions: three span queries and the classic query parser subset. Scoring is
BM25 with configurable `k1`/`b` (row `search/similarities/Similarity`...).
Collection: top hits with a total-hits threshold, `TopFieldCollector` (score,
doc, numeric and keyword keys; `searchAfter`), `Weight.count`, early
termination, `min_score`, concurrent slices. Also: explain, a
`UnifiedHighlighter` subset, SSDV and range facets, OpenSearch's shard-side
aggregations (metrics, keyword `terms`, `histogram`, `date_histogram`,
`range`/`date_range`, `filter(s)`, `global`, `cardinality`, any nesting),
stored-field fetch.

**Write path.** `IndexWriter` (row `index/IndexWriter`: adds, updates, soft
updates, deletes by term and by a closed query set, doc-values updates, flush
triggers, two-phase commit, `IndexFileDeleter`, `TieredMergePolicy` at commit,
force merge, index sort of every kind, `MAX_DOCS`), the concurrent writer (row
`index/DocumentsWriter`...), and pre-inverted documents for the OpenSearch
engine (`lucene-index/src/index_writer/explicit.rs`; [engine.md](engine.md)).

**Deliberately out of scope**, and what a caller sees:

- **Lucene 8.** Since M8 ([backward-codecs.md](backward-codecs.md)) every
  format Lucene 9.0-10.4 wrote is read, by default or per field: the retired
  `Lucene90`..`Lucene95` HNSW vector formats and the opt-in `Lucene99` scalar
  and `Lucene102` binary quantized KNN formats included. Lucene 8 formats are
  not read (neither Lucene 10 nor OpenSearch 3.x opens a Lucene 8 index).
  Since M8 T8.5 the plugin serves an index with retired-format postings
  natively (`NativeReaders.SUPPORTED_POSTINGS_FORMATS`); only a non-default
  postings format (`completion`) still falls back (`postings_format`).
- **Other codecs and per-field formats**: `SimpleText`, the scalar-quantized
  KNN formats, completion (suggester) postings, and any per-field postings or
  doc-values format other than `Lucene104`/`Lucene90`. (Several *instances*
  of those two are fine: postings fields route to `Lucene104PostingsFormat`
  instances of their own since M7, and segments with several of either are
  read and merged -- row `codecs/perfield/PerFieldPostingsFormat`.) The
  OpenSearch engine refuses them per document; see `opensearch-engine.md`,
  "What it refuses".
- **Features with no OpenSearch consumer**: taxonomy facets and
  `DrillSideways`, span queries beyond term/near/or, `NGramTokenizer` and
  most of `lucene-analysis` (the engine analyses in Java), the suggester
  beyond the FST shape (rows `facet/...`, `queries/spans/...`,
  `analysis/...`, `search/suggest/fst/WFSTCompletionLookup`).
