# Supported matrix

[Index](../parity.md). What this port supports end to end, and what it
deliberately does not; where a row elsewhere and this summary disagree, the
row is right. In OpenSearch's vocabulary:
[`opensearch-native-queries.md`](../opensearch-native-queries.md),
[`opensearch-engine.md`](../opensearch-engine.md).

**Codec: Lucene 10.5.0's default, `Lucene104`, and only it.** R = read,
W = written by `IndexWriter`; "verified" = real Lucene reads the Rust bytes
(`scripts/verify-write-path.sh`, `scripts/verify-interop.sh`).

| Format (10.5.0 class) | Files | R | W | What differs or is missing (rows) |
|---|---|---|---|---|
| `Lucene104PostingsFormat` + `Lucene103BlockTreeTermsReader`/`Writer` | `.doc` `.pos` `.pay` `.psm` `.tim` `.tip` `.tmd` | yes | yes, verified; byte-identical (`blocktree_byte_identity_fixture.rs`) | materialising position readers refuse a stream past `u32::MAX` entries; cursors take any `totalTermFreq` (codecs.md) |
| `Lucene90StoredFieldsFormat`, both modes | `.fdt` `.fdx` `.fdm` | yes | yes, verified; `BEST_COMPRESSION` byte-identical | nothing |
| `Lucene90DocValuesFormat`, all five types, skip index | `.dvm` `.dvd` `.dvs` | yes | yes, verified | nothing (`doBlocks` byte-identical) |
| doc-values updates (generational `.fnm`/`.dvm`/`.dvd`) | `_N_gen.*` | yes | yes, verified | -- (row `index/CheckIndex`) |
| `Lucene90NormsFormat` | `.nvm` `.nvd` | yes | yes, verified | -- |
| `Lucene90PointsFormat` (BKD version 10) | `.kdm` `.kdi` `.kdd` | yes | yes, verified; byte-identical per field | older BKD versions rejected; a flush writes fields in number order, Java in field-hash order |
| `Lucene90TermVectorsFormat` | `.tvd` `.tvx` `.tvm` | yes | yes, verified | refused by the OpenSearch engine |
| `Lucene90LiveDocsFormat` | `.liv` | yes | yes, verified | -- |
| `Lucene90CompoundFormat` | `.cfs` `.cfe` | yes | yes, verified | `useCompoundFile` defaults to `false` (Java: `true`); equal-length members may pack in another order (the `.cfe` maps names) |
| `Lucene99SegmentInfoFormat` (with index sort) | `.si` | yes, every `SortFieldProvider` | yes, verified | -- |
| `Lucene94FieldInfosFormat` | `.fnm` | yes | yes, verified | -- |
| `SegmentInfos` | `segments_N`, `pending_segments_N` | yes | yes, two-phase commit | -- |
| `Lucene99HnswVectorsFormat` + `Lucene99FlatVectorsFormat` | `.vec` `.vemf` `.vem` `.vex` | yes | yes, verified | refused by the OpenSearch engine; quantized formats: codecs.md |

Writer benchmarks: the codecs.md rows and
[`docs/benchmarks/m7-2026-10.md`](../benchmarks/m7-2026-10.md).

**Search (Lucene level).** The scorer tree runs every core query type
(boolean, constant score, boost, dis-max, match-all/none, phrase and
multi-phrase, term-in-set, prefix, wildcard, regexp, fuzzy, points and
doc-values ranges, field exists, synonym, combined field, every
`MultiTermQuery` rewrite, KNN as a clause, the Indri/log-odds/Bayesian
queries) with bulk scorers, block-max pruning and a per-segment query cache
([search-queries.md](search-queries.md),
[search-execution.md](search-execution.md)); spans, intervals and payloads
in [queries.md](queries.md). Scoring: BM25 and every Lucene similarity.
Collection: top hits with a total-hits threshold, `TopFieldCollector`,
`Weight.count`, early termination, `min_score`, concurrent slices. Also:
explain, a `UnifiedHighlighter` subset, facets, OpenSearch's shard-side
aggregations, stored-field fetch.

**Write path.** `IndexWriter` (row `index/IndexWriter`), the concurrent
writer (row `index/DocumentsWriter`...), and pre-inverted documents for the
OpenSearch engine ([engine.md](engine.md)).

**Deliberately out of scope**, and what a caller sees:

- **Lucene 8.** Every format Lucene 9.0-10.4 wrote is read
  ([backward-codecs.md](backward-codecs.md)); Lucene 8 formats are not
  (neither Lucene 10 nor OpenSearch 3.x opens them). The plugin serves
  retired-format postings natively; only a non-default postings format
  (`completion`) falls back (`postings_format`).
- **Other codecs and per-field formats**: `SimpleText`, completion
  postings, and per-field postings or doc-values formats other than
  `Lucene104`/`Lucene90` (several instances of those two are fine: row
  `codecs/perfield/PerFieldPostingsFormat`). The OpenSearch engine refuses
  them per document (`opensearch-engine.md`, "What it refuses").
- **Features with no OpenSearch consumer**: taxonomy facets and
  `DrillSideways`, most of `lucene-analysis` (the engine analyses in Java),
  the suggester beyond the FST shape (rows `facet/...`, `analysis/...`,
  `search/suggest/fst/WFSTCompletionLookup`).
