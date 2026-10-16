# lucene-misc -- utilities (M14 T14.3)

[Index](../parity.md). `org.apache.lucene.misc`: the index and search utilities the jar ships
beside core. This file records what is ported; the rest of the jar's 30 classes are each
`todo:M14` in `docs/inventory/lucene-misc.tsv` and fall to the M14.b-i sub-milestones
(`docs/milestones/m14-remaining-modules-and-tooling.md`).

| Java | Rust | Status |
|---|---|---|
| `misc/HighFreqTerms`, `misc/TermStats` | `lucene-search/src/misc/high_freq.rs::{get_high_freq_terms, TermStats, TermOrder}`, `lucene-search/src/java_priority_queue.rs::JavaPriorityQueue` | **ported** M14.b: the top `numTerms` terms of an index by document frequency or total term frequency, over one field or every indexed field, highest first, ties broken by field (`String.compareTo`) then the term's unsigned bytes. The queue is Java's `PriorityQueue` (sift order decides which of equal terms overflow), shared with the classification module. Equal to Lucene's over six configurations of a three-segment index, whole index and one field, more terms asked for than exist -- `misc_fixtures.rs`. Differs: `getTermText()` is lossy UTF-8 as `String` is; a named field with no terms and an index with no indexed fields are `Error::IllegalState`, where Java throws `RuntimeException`. The `main` command-line entry point is M14.i. |

`lucene-search/src/misc/mod.rs` is the module facade and has no Java counterpart of its own.
