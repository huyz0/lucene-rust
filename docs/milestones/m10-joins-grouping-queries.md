# M10 — Nested documents, joins, grouping, and the `queries` module

> **Goal:** block joins, query-time joins, grouping, function and interval
> queries, as `lucene-join`, `lucene-grouping` and `lucene-queries` have them,
> so OpenSearch's `nested`, `has_child`, `collapse`, `function_score` and
> `intervals` run natively.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md) |
| **Unblocks** | native `nested`, `function_score`, `intervals`, `combined_fields`, field collapsing |
| **Status** | in progress (T10.0, T10.1 done; T10.2 queries ported) |

---

## Why this milestone exists

None of `lucene-join` (31 files, 5.2k lines) or `lucene-grouping` (32 files,
3.2k lines) is ported, and 9 of `lucene-queries`' concrete queries are not
(its span queries are). These back some of OpenSearch's most used features:
`nested` fields are block joins (`ToParentBlockJoinQuery`), `function_score`
is `FunctionScoreQuery`, `intervals` is `IntervalQuery`, field collapsing is
grouping. Each of them falls back to Lucene today.

---

## Scope

### In scope

- `lucene-join`: `ToParentBlockJoinQuery`, `ToChildBlockJoinQuery`,
  `ParentChildrenBlockJoinQuery`, `CheckJoinIndex`, the diversifying child KNN
  queries, query-time joins (`JoinUtil`, `TermsQuery`,
  `GlobalOrdinalsQuery`), and indexing blocks of documents atomically.
- `lucene-grouping`: `GroupingSearch`, first- and second-pass collectors,
  `AllGroupsCollector`, `BlockGroupingCollector`, `TopGroups`.
- `lucene-queries`: `function` (value sources, `FunctionScoreQuery`,
  `FunctionRangeQuery`), `intervals`, `payloads`, `mlt` (`MoreLikeThis`),
  `CommonTermsQuery`, the remaining spans.
- The `sandbox` queries OpenSearch builds (`CombinedFieldQuery`'s sandbox
  form, `CoveringQuery`, `PhraseWildcardQuery`, the doc-values multi-range
  queries).
- OpenSearch: `nested`, `has_child`/`has_parent`, `function_score`,
  `intervals`, `combined_fields`, `more_like_this` and `collapse` as native
  shapes.

### Out of scope

- The rest of `sandbox` (M14).

---

## Tasks

- **T10.0** — Port inventories for `lucene-join`, `lucene-grouping` and
  `lucene-queries` (`docs/inventory/lucene-{join,grouping,queries}.tsv`,
  `check-port-inventory.py --module join|grouping|queries` in the gate and
  CI). **Done.** `--milestone M10 --module <m>` lists what each still owes.
- **T10.1** — Document blocks in the writer (atomic add and update of a
  parent with its children) and `CheckJoinIndex`. **Done.** Blocks were
  already atomic (one sequence number, never split by a flush, `hasBlocks`);
  this adds `IndexWriterConfig.setParentField` (`set_parent_field`), index
  sorting with blocks at flush and merge (whole blocks moved by their
  parents' keys), index sorting for explicit documents, the document API's
  `softUpdateDocuments`, `CheckIndex.testSort` over parents, and
  `BitSetProducer`/`QueryBitSetProducer`/`CheckJoinIndex`. Proven by
  `VerifyJoin` (real Lucene reads, checks, appends to and force-merges a
  Rust-written block index, sorted and unsorted), the merge stress test
  (`block_join_merge_stress.rs`) and block ops in `op-stream-fuzz.sh`. The
  concurrent writer takes native documents only, so it has no parent field
  and still refuses blocks in a sorted index.
- **T10.2** — Block-join queries and their scoring modes. **Queries
  ported** (`lucene-search/src/join`, `exec/join.rs`):
  `ToParentBlockJoinQuery` (all five score modes, scorer and bulk scorer),
  `ToChildBlockJoinQuery`, `ParentChildrenBlockJoinQuery`,
  `ParentsChildrenBlockJoinQuery`, `ToParentBlockJoinSortField` with
  `BlockJoinSelector`/`ToParentDocValues`, explain and matches,
  `DiversifyingChildrenFloat/ByteKnnVectorQuery`; 1056 differential searches
  (`GenBlockJoin`) bit for bit. Open: benchmarks.
- **T10.3** — Query-time joins.
- **T10.4** — Grouping.
- **T10.5** — Function queries and value sources.
- **T10.6** — Intervals, payload queries, `MoreLikeThis`, `CommonTermsQuery`.
- **T10.7** — Plugin wiring for the OpenSearch shapes above.

## Acceptance criteria

- [ ] Every query matches Lucene's hits and scores bit for bit on generated
      fixtures, including empty and single-child blocks and deleted parents.
- [ ] A Rust-written block index passes Lucene's `CheckJoinIndex`.
- [ ] OpenSearch's `nested` and `function_score` YAML suites fail
      identically with and without native execution.
- [ ] Each new query is no slower than Lucene on its benchmark.

## Risks and unknowns

- **Block integrity under merges.** A merge that splits or reorders a block
  breaks every join silently. It needs a resource-bound-style test: many
  merges, then `CheckJoinIndex`.
- **Value-source breadth.** `function` has many small classes; most are
  one-liners, but each needs a scoring differential.

## Exit artifacts

- Join, grouping and function fixture generators
- `docs/parity.md` rows for `lucene-join`, `lucene-grouping`, `lucene-queries`
  and the ported `sandbox` classes
