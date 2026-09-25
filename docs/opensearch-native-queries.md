# OpenSearch: which searches run native

The lucene-rust OpenSearch plugin (`opensearch-plugin/`, milestone
[M2](milestones/m2-opensearch-read-path.md)) runs the **query phase** of a shard
search in Rust when the request is inside the table below, and on Lucene
otherwise — per query, transparently. The fetch phase (`_source`, highlighting,
`docvalue_fields`, …), indexing, refresh and every other API are OpenSearch's own
code, unchanged.

Pinned versions: **OpenSearch 3.8.0**, **Lucene 10.5.0**.

Everything here is enforced by `scripts/verify-opensearch.sh`: every row is a
request it sends, with the plugin's switch off (the Lucene reference) and on,
and it requires identical hits, scores (to 1e-5), totals and max score, and
that the plugin's own counters show the query ran where this table says.

## Runs native

A request runs native when **all** of these hold:

- the index has `index.lucene_rust.search.enabled: true` (the default);
- it is a plain top-hits-by-score request — no `sort`, `aggs`, `post_filter`,
  `min_score`, `terminate_after`, `scroll`, `search_after`, `collapse`,
  `rescore`, `timeout` or `profile`;
- the rewritten Lucene query is one of the shapes below, over fields that score
  with the default BM25 (`k1 = 1.2`, `b = 0.75`);
- every field in the index uses Lucene 10.5.0's default postings format
  (`Lucene104`). An index with a `completion` field does not.

| Request (DSL) | Rewritten Lucene query | Native by default | Measured, REST, native vs Lucene |
|---|---|---|---|
| `match` one term | `TermQuery` | yes | 1.46× |
| `match` several terms (`or`) | `BooleanQuery` of `SHOULD` terms | yes | 1.15–1.31× |
| `match` with `operator: and` | `BooleanQuery` of `MUST` terms | yes | 1.05× |
| `term` on a `keyword` field | `ConstantScoreQuery(TermQuery)` | yes | 0.99× (noise) |
| `term` on a missing value | same | yes | 1.30× |
| `bool` of `must` term + `filter` term | `BooleanQuery` `MUST`/`FILTER` | yes | 1.22× |
| `query_string` `a OR b` | `BooleanQuery` of `SHOULD` terms | yes | 1.08× |
| any of the above with `size: 0`, `from`/`size` paging, or any `track_total_hits` | — | yes | 1.00–1.23× |
| `match` with `minimum_should_match` ≥ 2 | `BooleanQuery` with `minimumNumberShouldMatch` | **no** (`slower_shape`) | 0.23× |
| `bool` mixing `must` and `should` | mixed `BooleanQuery` | **no** (`slower_shape`) | 0.25× |
| `bool` with `must_not` | `BooleanQuery` with `MUST_NOT` | **no** (`slower_shape`) | 0.30× |
| nested `bool` | `BooleanQuery` in `BooleanQuery` | **no** (`slower_shape`) | 0.21× |
| `bool` with only `filter` clauses | `BoostQuery(ConstantScoreQuery(…), 0)` | **no** (`slower_shape`) | 0.95× |
| any query with `boost` ≠ 1 | `BoostQuery` | **no** (`slower_shape`) | 0.19–0.20× |
| `constant_score` | `BoostQuery(ConstantScoreQuery(…))` | **no** (`slower_shape`) | 0.26× |

"Measured" is the median REST round trip over 40 requests on one node, both
engines on the same 100k-document index; the source and method are
[`benchmarks/m2-opensearch-e2e.md`](benchmarks/m2-opensearch-e2e.md). Above 1.0
means native is faster.

**The `no (slower_shape)` rows are correct natively; they are routed to Lucene
because they measured slower.** The Rust engine prunes (block-max MAXSCORE) only
for terms and pure disjunctions of terms; every other boolean shape runs its
exhaustive scorer, which loses to Lucene's WAND on dense terms. Setting
`index.lucene_rust.search.native_shapes: all` runs them native anyway — the
verify script does, to prove they stay correct.

## Falls back to Lucene

Each fallback is counted by reason at `GET /_plugins/lucene_rust/stats`.

| Reason | What it means |
|---|---|
| `disabled` | `index.lucene_rust.search.enabled: false` |
| `aggregations`, `post_filter`, `min_score`, `terminate_after`, `collectors` | the request adds a collector to the query phase |
| `sort`, `search_after`, `scroll`, `collapse`, `rescore`, `profile`, `timeout` | the request needs something the native top-hits path does not produce |
| `query_<Class>` | the rewritten query's root is not a supported shape — e.g. `query_PhraseQuery` (`match_phrase`), `query_ApproximateScoreQuery` (`range`, `match_all`), `query_MultiTermQueryConstantScoreBlendedWrapper` (`prefix`, `wildcard`), `query_DisjunctionMaxQuery` (`multi_match`) |
| `clause_<Class>` | the same, for a clause inside a `bool` |
| `field_similarity` | a field scores with anything but default-parameter BM25 |
| `boost_invalid` | a negative or non-finite boost |
| `boolean_empty`, `boolean_pure_negative` | a `bool` Lucene rewrites to match nothing |
| `slower_shape` | a correct native shape routed to Lucene by measurement (above) |
| `postings_format` | some field of the index uses a postings format other than `Lucene104` |
| `reader_*`, `directory_*` | the searcher's reader or directory is not a local, standard one (remote store, a wrapped reader the plugin cannot see through) |
| `native_open_failed`, `native_live_docs_failed`, `native_error` | the native side refused the reader or the query; the node log has the message. Never a failed search: the query re-runs on Lucene |

## Known limits

- **One `QueryPhaseSearcher` per node.** OpenSearch accepts exactly one plugin
  providing it. `neural-search` provides one too, so the two cannot be
  installed together; the test image removes the bundled plugins.
- **Compound segments are copied, not mapped.** The native reader reads a
  `.cfs` segment into memory (OpenSearch's small, freshly flushed segments are
  compound); only non-compound segments are memory-mapped. Released with the
  reader either way.
- **Linux only**, x86_64 and aarch64; the library needs glibc ≥ 2.34 (the
  OpenSearch 3.8.0 image has 2.34, and `verify-opensearch.sh` checks it).
