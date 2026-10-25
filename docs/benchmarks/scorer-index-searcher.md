# ScorerIndexSearcher and DisablingBulkScorerQuery

The pair searches real three-leaf indexes with Lucene 10.5.0's actual
`ScorerIndexSearcher` or `DisablingBulkScorerQuery`, and the Rust operational
wrappers. Each input has 768 documents, committed in three equal segments, with
192 deletions. Both writers use `StandardAnalyzer` on the same twelve ASCII
texts repeated 64 times. Index creation and reader/query setup are outside timing.

Each measured step searches nine queries: a term, disjunction, conjunction,
required/optional clauses, exclusion, exact phrase, term with a phrase filter,
constant-scored phrase and match-all. The four cases select either wrapper under
`COMPLETE` or `COMPLETE_NO_SCORES`. Preparation/rewriting, weight/scorer creation
and every hit's global document id and score bits are timed. The collectors feed
query order, hit counts and hits into a digest; all four checks must agree in
every repetition before the report accepts the pair.

The wrappers deliberately use `DefaultBulkScorer` over the ordinary scorer tree.
Their Java supplier wrappers also omit forwarding `setTopLevelScoringClause`, so
the top-level-dependent Boolean WAND/block-max specializations are suppressed;
WAND for minimum-should-match greater than one and other score-mode-dependent
scorer behavior remain available. Fused constant-scored term legs currently
also require top-level notification for competitive pruning, whereas Java's
ConstantScoreScorer prunes in TOP_SCORES without it. This is a missed
optimization with correct final top hits and scores. The nine-query pair uses
a constant-scored phrase and does not measure this fused constant-term gap.
Supplier notification omission is intended behavior,
not an optimization to recover. Rust carries the scorer-only choice on the
searcher itself, so manager worker threads see it and the default BM25 top-hits
shortcut cannot bypass it. Field-sort/facet APIs and leaf subranges are not
exposed by the Rust searcher wrapper.

The correctness fixture separately checks three small committed leaves with
live-document filtering, BM25 and Boolean similarity, complete/no-score/TopDocs
collectors (including TOP_DOCS_WITH_SCORES), top hits/search-after, counting, manager reduction, selected leaves
and query timeouts. A Java trap query fails if either wrapper requests its
specialized bulk scorer; it also records that top-level supplier notification is
not forwarded. A top-scores collector publishes positive infinity before and
after collection; terms, disjunction-max (including absent legs and nonzero ties),
blended terms, one-term phrases, required/filter and required/exclusion queries
still yield every live match because their suppliers were not marked top-level.
An ordinary term scorer obtained directly from each leaf under
TOP_DOCS_WITH_SCORES separately proves that its iterator remains exhaustive
after publishing a score threshold. This observation covers the scorer API;
specialized bulk scorers can separately filter collector thresholds. Java `DisablingBulkScorerQuery` throws `NullPointerException` when
the enclosed supplier is null, while the Rust optional scorer returns no hits.
That boundary is recorded explicitly and is excluded from the successful pair.

Memory accounting is the retained shallow increment of `ScorerIndexSearcher`
over the same existing `IndexSearcher`. The Rust wrapper adds no size beyond its owned searcher; the Java subclass
adds no fields. Neither wrapper adds a heap allocation. The feature-gated
selection bit belongs to Rust's existing searcher. This narrow comparison
excludes complete reader/searcher graphs, query-wrapper memory, temporary scorer
and collector allocations, allocator metadata, GC and RSS.

Run through the serialized capped container:

```sh
scripts/docker-test.sh bash scripts/bench-scorer-index-searcher.sh 1000 1500
```

The script alternates three Rust/Java repetitions, pins both to the same two CPUs,
uses Java assertions and a fixed 128 MiB heap, checks load before building, and
prints every raw repetition alongside median ratios and measured spread.

Validation on 2026-10-25 passed the 375-row Java fixture comparison, its separate
configuration/timeout test, both disabling-wrapper integration tests, and each
wrapper's module unit test. The infinity-threshold cases cover both similarities
and both wrappers. Java's null-supplier exception remains an explicit boundary.

The closest-to-Java implementation produced these median batch times across
three alternating repetitions. Each batch performs all nine searches.

| Case | Rust ns | Java ns | Java / Rust | Noise |
| --- | ---: | ---: | ---: | ---: |
| Scorer searcher, complete | 141733.501 | 683710.068 | 4.82 | 1.05 |
| Scorer searcher, no scores | 118069.008 | 579548.523 | 4.91 | 1.01 |
| Disabling query, complete | 147084.450 | 658071.925 | 4.47 | 1.02 |
| Disabling query, no scores | 123293.668 | 567077.274 | 4.60 | 1.02 |

Every required cross-engine digest agreed in every repetition. Scored batches
produce `6942781673998663008`; no-score batches produce `9230219897851981920`.
The median ratio is 4.71; the median within-engine noise is 1.02. Each case
exceeds parity beyond its observed spread, so stage 3 retains the faithful port
without an additional optimization. The selected retained wrapper increment is
0 Rust bytes and 0 Java bytes, within the narrow memory scope described above.

Raw batch times in ns, in repetition order:

| Case | Engine | Rep 1 | Rep 2 | Rep 3 |
| --- | --- | ---: | ---: | ---: |
| Scorer searcher, complete | Rust | 141609.517 | 142518.573 | 141733.501 |
| Scorer searcher, complete | Java | 675586.060 | 710707.689 | 683710.068 |
| Scorer searcher, no scores | Rust | 117770.087 | 118882.768 | 118069.008 |
| Scorer searcher, no scores | Java | 579383.769 | 579548.523 | 582571.209 |
| Disabling query, complete | Rust | 146759.023 | 147632.931 | 147084.450 |
| Disabling query, complete | Java | 658071.925 | 646384.959 | 661735.576 |
| Disabling query, no scores | Rust | 122999.667 | 123293.668 | 123671.225 |
| Disabling query, no scores | Java | 565439.960 | 567077.274 | 576311.539 |
