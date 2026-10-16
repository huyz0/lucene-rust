# lucene-suggest -- suggesters and the completion postings format (M13 T13.3)

[Index](../parity.md). `org.apache.lucene.search.suggest`: the `Lookup` API every suggester
implements, the `InputIterator` family it is built from, the FST and TST lookups, the analyzing
suggesters, and `suggest.document`'s completion postings format that OpenSearch uses. T13.3 is
planned as sub-tasks, in order: (a) the base layer, (b) TST and FST lookups (the TST is done), (c) analyzing
suggesters, (d) `suggest.document`, (e) spell. T13.3a is the base layer below.

| Java | Rust | Status |
|---|---|---|
| `Lookup` | `lucene-search/src/suggest/lookup.rs::Lookup` | **ported** T13.3a: the trait (`getCount`, `build`, `lookup` with and without contexts, `store`, `load`). Tests: unit tests; the TST fixture drives it. |
| `Lookup.LookupResult` | `lucene-search/src/suggest/lookup.rs::LookupResult` | **ported** T13.3a: key, highlight key, value, payload, contexts; ordered by key in UTF-16 units. Differs: `Set<BytesRef> contexts` is a `BTreeSet`. Tests: unit tests. |
| `Lookup.CHARSEQUENCE_COMPARATOR` | `lucene-search/src/suggest/lookup.rs::charsequence_compare` | **ported** T13.3a: the first differing UTF-16 unit, a prefix first. Differs: returns an `Ordering`, not Java's difference. Tests: unit tests (a supplementary character against U+FFFD). |
| `Lookup.LookupPriorityQueue` | `lucene-search/src/suggest/lookup.rs::LookupPriorityQueue` | **ported** T13.3a: the bounded min-heap on `value`, Java's `PriorityQueue` heap operation for operation (`insertWithOverflow`, `pop`, `getResults`), so ties come out in Java's order. Differs: no `add` (Java's `add` throws when full; callers use `insertWithOverflow`). Tests: unit tests (ties, eviction, zero capacity). |
| `InputIterator` | `lucene-search/src/suggest/input.rs::InputIterator` | **ported** T13.3a: `next`, `weight`, `payload`, `hasPayloads`, `contexts`, `hasContexts`. Differs: `next` returns `Vec<u8>`; contexts are `BTreeSet`. Tests: unit tests. |
| `InputIterator.InputIteratorWrapper`, `InputIterator.EMPTY` | `lucene-search/src/suggest/input.rs::{InputIteratorWrapper, input_empty}` | **ported** T13.3a: weight 1, no payloads, no contexts. Tests: unit tests. |
| `BufferedInputIterator` | `lucene-search/src/suggest/input.rs::BufferedInputIterator` | **ported** T13.3a: drains the source at construction and replays it. Differs: `BytesRefArray` is a `Vec`; a null payload is stored as empty (Java fails later); `weight()` before the first `next` or after the end is 0 (Java throws). Tests: unit tests. |
| `SortedInputIterator` | `lucene-search/src/suggest/input.rs::SortedInputIterator` | **ported** T13.3a, in memory: the same `(term, weight)` order as Java's `tieBreakByCostComparator`, stable. Differs: `OfflineSorter` on disk is replaced by an in-memory sort; no temp files. Tests: unit tests. |
| `tst/TSTLookup`, `tst/TSTAutocomplete`, `tst/TernaryTreeNode` | `lucene-search/src/suggest/tst.rs::TstLookup` | **ported** T13.3b: `build` (sorted by `utf8SortedAsUTF16SortOrder`, then `balancedTree` into the empty NUL root), `add`, `get`, `lookup` (`prefixCompletion`'s depth-first order, high child first; `onlyMorePopular` through `LookupPriorityQueue`), `store`/`load`. Lucene's answers equal ours for every short prefix of a 324-entry corpus (792 queries, two modes, three limits) and `get` for 234 keys; `store()` writes Lucene's bytes. Differs: values are `i64` (Java `Object`); lookup of an empty key and contexts/payloads in `build` are errors, as Java throws; a supplementary character is two nodes and does not survive `store`/`load`, as in Java (each half is written as U+FFFD); `insert` is iterative; `ramBytesUsed` is not ported. Tests: `suggest_fixtures.rs` (`GenSuggest.java`: `terms.tsv`, `tst.tsv`, `tst_get.tsv`, `tst.bin`), `tst.rs` unit tests. |
| `UnsortedInputIterator`, `InMemorySorter`, `BytesRefSorter`, `ExternalRefSorter`, `FileDictionary`, `DocumentDictionary`, `DocumentValueSourceDictionary`, `Dictionary`, `BitsProducer` | not yet ported | **todo** T13.3a: the rest of the base layer. |
| `package-info` | `lucene-search/src/suggest/mod.rs` | not a class: the module's documentation and re-exports. |
