# OpenSearch: which searches run native

The lucene-rust OpenSearch plugin (`opensearch-plugin/`, milestone
[M2](milestones/m2-opensearch-read-path.md)) runs the **query phase** of a shard
search in Rust when the request is inside the table below, and on Lucene
otherwise — per query, transparently. The fetch phase (`_source`, highlighting,
`docvalue_fields`, …), indexing, refresh and every other API are OpenSearch's own
code, unchanged.

Pinned versions: **OpenSearch 3.8.0**, **Lucene 10.5.0**.

Everything here is enforced by `scripts/verify-opensearch.sh`: each row below
is backed by requests it sends (the `size: 0`/paging/`track_total_hits` row by
`match` and keyword `term` variants, with totals both above and below the
threshold), with the plugin's switch off (the Lucene reference) and on,
and it requires identical hits, scores (to 1e-5), totals and max score, and
that the plugin's own counters show the query ran where this table says.

## Runs native

A request runs native when **all** of these hold:

- the index has `index.lucene_rust.search.enabled: true` (the default);
- it is a plain top-hits-by-score request — no `sort`, `aggs`, `post_filter`,
  `min_score`, `terminate_after`, `scroll`, `search_after`, `collapse`,
  `rescore`, `timeout` or `profile`, not `search_type=dfs_query_then_fetch`,
  and no other plugin replacing the top-docs collector;
- the rewritten Lucene query is built only from the shapes below --
  `TermQuery`, `BooleanQuery` (any `Occur`, any `minimum_should_match`,
  nested), `ConstantScoreQuery`, `BoostQuery`, `DisjunctionMaxQuery`,
  `MatchAllDocsQuery`, `MatchNoDocsQuery` -- at most 32 deep and 1,024 nodes,
  over fields that score with the default BM25 (`k1 = 1.2`, `b = 0.75`), with
  every `TermQuery` scoring from the reader's own statistics (not blended
  `TermStates`, as `multi_match` `cross_fields` builds);
- every field in the index uses Lucene 10.5.0's default postings format
  (`Lucene104`). An index with a `completion` field does not.

Totals follow OpenSearch's own shortcut (`shortcutTotalHitCount`): a
`match_all` reports the reader's `numDocs`, and a `term` its document
frequency when nothing is deleted, without counting.

| Request (DSL) | Rewritten Lucene query | Measured, REST, native vs Lucene |
|---|---|---|
| `match` one term | `TermQuery` | 1.51× |
| `match` a rare term | `TermQuery` | 1.40× |
| `match` two terms (`or`) | `BooleanQuery` of `SHOULD` terms | 1.15× |
| `match` four terms | `BooleanQuery` of `SHOULD` terms | 1.21× |
| `match` with `operator: and` | `BooleanQuery` of `MUST` terms | 1.12× |
| `match` with `minimum_should_match: 2` | `BooleanQuery` with `minimumNumberShouldMatch` | 0.97× |
| `term` on a `keyword` field | `ConstantScoreQuery(TermQuery)` | 1.24× |
| `term` on a missing value | same | 1.22× |
| `bool` `must` + `should` | mixed `BooleanQuery` | 1.13× |
| `bool` with `must_not` | `BooleanQuery` with `MUST_NOT` | 1.15× |
| `bool` `must` + `filter` | `BooleanQuery` `MUST`/`FILTER` | 1.19× |
| `bool` with only `filter` | `BoostQuery(ConstantScoreQuery(…), 0)` | 1.21× |
| nested `bool` | `BooleanQuery` in `BooleanQuery` | 1.04× |
| `query_string` `a OR b` | `BooleanQuery` of `SHOULD` terms | 1.16× |
| `bool` `must` + `should`, `minimum_should_match: 1` | `BooleanQuery` `MUST` + `SHOULD` with a minimum | 0.99× |
| any query with `boost` ≠ 1 | `BoostQuery` | 1.13× |
| `bool` with a boosted clause | `BooleanQuery` of `BoostQuery` | 1.06× |
| `constant_score` | `BoostQuery(ConstantScoreQuery(…))` | 1.08× |
| `dis_max` (`tie_breaker: 0.3`) | `DisjunctionMaxQuery` | 1.28× |
| `multi_match` (`best_fields`) | `DisjunctionMaxQuery` of terms | 1.00× |
| `match_all` | `MatchAllDocsQuery` | 1.21× |
| `bool` `match_all` + `filter` | `BooleanQuery` with `MatchAllDocsQuery` | 1.04× |
| `constant_score` of `match_all` | `BoostQuery(ConstantScoreQuery(MatchAllDocsQuery))` | 1.04× |
| any of the above with `size: 0`, `from`/`size` paging, or any `track_total_hits` | — | 0.99–1.30× |

"Measured" is the median REST round trip over 40 requests per engine on one
node, both engines on the same 100k-document, two-segment index
(`scripts/verify-opensearch.sh --docs 100000 --bench-out FILE --bench-rounds
40`); above 1.0 means native is faster. At this size a round trip is 2-9 ms,
most of it OpenSearch's own request handling, so REST ratios compress toward
1.0 and move by about ±0.05 between runs; the in-process numbers for the same
shapes on a 1M-document corpus built like this index are in
[`milestones/m5-6-native-read.md`](milestones/m5-6-native-read.md).
Three rows are not yet clearly above 1.0 over REST: `minimum_should_match: 2`
(0.97×), `must` + `should` with a minimum (0.99×) and `multi_match`
(1.00× in this run, measured before per-clause skipping for a tie-breaker
of 0 landed, which took it from 0.36× to 1.30× in process). In process they
run 1.10×, 1.3× and 1.3–3.5× Lucene; the REST margin is inside the spread,
and they stay open under the milestone's R7 acceptance.

`index.lucene_rust.search.native_shapes` (`fast`/`all`) predates read path
R1, when mixed booleans measured slower and were routed to Lucene
(`slower_shape`); since R1 every encodable shape runs native under both
values.

## Falls back to Lucene

Each fallback is counted by reason at `GET /_plugins/lucene_rust/stats`.

| Reason | What it means |
|---|---|
| `disabled` | `index.lucene_rust.search.enabled: false` |
| `aggregations`, `post_filter`, `min_score`, `terminate_after`, `collectors` | the request adds a collector to the query phase |
| `sort`, `search_after`, `scroll`, `collapse`, `rescore`, `profile`, `timeout` | the request needs something the native top-hits path does not produce |
| `query_<Class>` | the rewritten query's root is not a supported shape — e.g. `query_PhraseQuery` (`match_phrase`), `query_IndexOrDocValuesQuery` (`range`), `query_MultiTermQueryConstantScoreBlendedWrapper` (`prefix`, `wildcard`) |
| `clause_<Class>` | the same, for a clause anywhere below the root (inside a `bool`, `constant_score`, `dis_max`, a boost) |
| `query_too_deep`, `query_too_large` | more than 32 levels, or more than 1,024 nodes counting wrappers (Lucene counts only leaves, and `indices.query.bool.max_clause_count` can raise its limit) |
| `boolean_msm_negative` | a `BooleanQuery` with a negative `minimumNumberShouldMatch` |
| `field_similarity` | a field scores with anything but default-parameter BM25 |
| `term_states` | a term query carries its own statistics (`multi_match` `cross_fields`), or is a `TermQuery` subclass |
| `dfs` | `search_type=dfs_query_then_fetch`: scoring uses statistics aggregated across shards |
| `collector_spec` | another plugin registered a replacement top-docs collector |
| `cancelled` | the task was cancelled before the native call |
| `boost_invalid` | a negative or non-finite boost |
| `slower_shape` | a correct native shape routed to Lucene by measurement; none since read path R1 |
| `postings_format` | some field of the index uses a postings format other than `Lucene104` |
| `reader_*`, `directory_*` | the searcher's reader or directory is not a local, standard one (remote store, a wrapped reader the plugin cannot see through) |
| `native_open_failed`, `native_error` | the native side refused the reader or the query; the node log has the message. Never a failed search: the query re-runs on Lucene |

## Known limits

- **One `QueryPhaseSearcher` per node.** OpenSearch accepts exactly one plugin
  providing it. `neural-search` provides one too, so the two cannot be
  installed together; the test image removes the bundled plugins.
- **Compound segments are copied, not mapped.** The native reader reads a
  `.cfs` segment into memory (OpenSearch's small, freshly flushed segments are
  compound); only non-compound segments are memory-mapped. Released with the
  reader either way.
- **A native query phase is not interruptible.** Cancellation is checked before
  the native call, not during it; the Java path checks it per segment. A
  native query runs to completion (one call, no per-document crossings), and
  requests with an explicit `timeout` fall back.
- **Linux only**, x86_64 and aarch64; the library needs glibc ≥ 2.34 (the
  OpenSearch 3.8.0 image has 2.34, and `verify-opensearch.sh` checks it).
