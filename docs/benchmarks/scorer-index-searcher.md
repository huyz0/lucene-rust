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
query order, hit counts and hits into a digest. Two additional cases use
`TOP_SCORES` on a constant-scored term. All six checks must agree in every
repetition before the report accepts the pair.

The wrappers deliberately use `DefaultBulkScorer` over the ordinary scorer tree.
Their Java supplier wrappers also omit forwarding `setTopLevelScoringClause`, so
the top-level-dependent Boolean WAND/block-max specializations are suppressed;
WAND for minimum-should-match greater than one and other score-mode-dependent
scorer behavior remain available. Java's `ConstantScoreScorer` and Rust's fused
constant-term scorer prune in `TOP_SCORES` without top-level notification.
The two constant-term cases explicitly cover that handshake.
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
Another 660 reference rows check constant terms, boosts, nesting, filters,
exclusions and DisMax with both wrappers and similarities, all five score modes,
and minimum scores of 1, 2 and positive infinity. The Rust collector translates
these inclusive scorer minima into its exclusive collector-bound convention.
Thresholds equal to the constant retain matches; higher thresholds prune where
the query's scorer forwards them. Genuine terms retain their separate supplier
notification requirement, including when mixed with constant terms.
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

Validation on 2026-10-25 passed the 1,035-row Java fixture comparison, its separate
configuration/timeout test, disabling-wrapper integration tests, and
scorer-searcher configuration tests. The infinity-threshold cases cover both similarities
and both wrappers. Java's null-supplier exception remains an explicit boundary.

The new fixture was run against the implementation before the pruning fix.
At the constant-term `TOP_SCORES` minimum of 2, Rust collected six score-1 hits
and Java collected none. After correcting pruning eligibility, all reference
rows agree. A separate iterator unit test covers both supplier-notification
states, exhaustive and top-score modes, thresholds equal to the constant,
threshold changes before and after positioning, and `nextDoc`/`advance`.

After the faithful pruning fix, the six-case pair on 2026-10-25 produced these
median batch times across three alternating repetitions. The first four batches
each perform nine searches; the constant-term batches each perform one search.

| Case | Rust ns | Java ns | Java / Rust | Noise |
| --- | ---: | ---: | ---: | ---: |
| Scorer searcher, complete | 143032.368 | 693580.979 | 4.85 | 1.04 |
| Scorer searcher, no scores | 117559.134 | 585229.938 | 4.98 | 1.02 |
| Disabling query, complete | 148334.923 | 662900.979 | 4.47 | 1.02 |
| Disabling query, no scores | 121926.185 | 577193.218 | 4.73 | 1.04 |
| Scorer searcher, constant term TOP_SCORES | 2434.576 | 4662.143 | 1.91 | 1.04 |
| Disabling query, constant term TOP_SCORES | 2687.305 | 4587.281 | 1.71 | 1.05 |

Every required cross-engine digest agreed in every repetition. Original scored
batches produce `6942781673998663008`; no-score batches produce
`9230219897851981920`. Each constant-term batch produces `7857554394914901`
and asserts exactly one live hit. The median ratio across all six cases is
4.60; the median within-engine noise is 1.04. Each case exceeds parity beyond
its observed spread, so stage 3 retains the faithful implementation without an
additional optimization. The selected retained wrapper increment is 0 Rust
bytes and 0 Java bytes, within the narrow memory scope described above.

Raw batch times in ns, in repetition order:

| Case | Engine | Rep 1 | Rep 2 | Rep 3 |
| --- | --- | ---: | ---: | ---: |
| Scorer searcher, complete | Rust | 143389.802 | 143032.368 | 142887.976 |
| Scorer searcher, complete | Java | 688607.073 | 714418.243 | 693580.979 |
| Scorer searcher, no scores | Rust | 117995.817 | 116968.707 | 117559.134 |
| Scorer searcher, no scores | Java | 575405.175 | 589741.091 | 585229.938 |
| Disabling query, complete | Rust | 148334.923 | 147701.593 | 148510.912 |
| Disabling query, complete | Java | 652992.835 | 662900.979 | 666052.970 |
| Disabling query, no scores | Rust | 121926.185 | 121773.695 | 122365.340 |
| Disabling query, no scores | Java | 563829.498 | 583574.783 | 577193.218 |
| Scorer searcher, constant term TOP_SCORES | Rust | 2434.576 | 2445.385 | 2387.592 |
| Scorer searcher, constant term TOP_SCORES | Java | 4535.345 | 4662.143 | 4697.486 |
| Disabling query, constant term TOP_SCORES | Rust | 2634.070 | 2687.305 | 2703.524 |
| Disabling query, constant term TOP_SCORES | Java | 4426.993 | 4587.281 | 4638.118 |

## Constant-term competitive-threshold extension

The `scorer_searcher_constant_term_top_scores` and
`disabling_query_constant_term_top_scores` cases each search one
`ConstantScoreQuery(TermQuery("body", "a"))` with score 1.0 over the same index.
Each collector starts without a competitive threshold. After hashing the first
live match it publishes positive infinity, and publishes that threshold when
receiving every later leaf scorer. This probes threshold changes on a current
scorer and thresholds present before later leaves begin, while the wrappers
continue to omit top-level supplier notification.

Each query must collect exactly one global live hit. Both engines hash query
ordinal, count, global document ID and raw score bits using the existing digest
format. The report requires the two new digest/count checks in every repetition;
a ratio is not accepted when a query collects extra hits. These checks cannot
measure internal iteration that produces the same hits. Each
new batch performs one search, while each original batch performs nine, so batch
times must not be compared directly across those groups. Preparation, scorer
construction, collection and checking use the same timing scope as the original
cases. The custom collector deliberately raises an infinite threshold; this is
a bounded scorer-handshake case, not a realistic top-k workload.

Run the actual-Java correctness fixtures and the Rust direct threshold
regressions before executing this extension. The unfixed fused leg collects
extra hits and must fail the correctness baseline; it is not a valid benchmark
baseline. After the faithful fix passed, the serialized six-case pair above
confirmed parity with Lucene and identical collected hits. Its 1.91/1.71 ratios
for the bounded constant-term handshake exceed the observed 1.04/1.05 noise.
Stage 3 retains the faithful fix without a further optimization; these timings
do not establish a general top-k speedup or prove the absence of internal
iteration that produces no additional collected hits.
