# lucene-misc -- utilities (M14 T14.3)

[Index](../parity.md). `org.apache.lucene.misc`: the index and search utilities the jar ships
beside core. This file records what is ported; the rest of the jar's 30 classes are each
`todo:M14` in `docs/inventory/lucene-misc.tsv` and fall to the M14.b-i sub-milestones
(`docs/milestones/m14-remaining-modules-and-tooling.md`).

| Java | Rust | Status |
|---|---|---|
| `misc/HighFreqTerms`, `misc/TermStats` | `lucene-search/src/misc/high_freq.rs::{get_high_freq_terms, TermStats, TermOrder}`, `lucene-search/src/java_priority_queue.rs::JavaPriorityQueue` | **ported** M14.b: the top `numTerms` terms of an index by document frequency or total term frequency, over one field or every indexed field, highest first, ties broken by field (`String.compareTo`) then the term's unsigned bytes. The queue is Java's `PriorityQueue` (sift order decides which of equal terms overflow), shared with the classification module. Equal to Lucene's over six configurations of a three-segment index, whole index and one field, more terms asked for than exist -- `misc_fixtures.rs`. Differs: `getTermText()` is lossy UTF-8 as `String` is; a named field with no terms and an index with no indexed fields are `Error::IllegalState`, where Java throws `RuntimeException`. The `main` command-line entry point is M14.i. |
| `misc/{CollectorMemoryTracker, util/MemoryTracker}` | `lucene-search/src/misc/memory.rs::{CollectorMemoryTracker, MemoryTracker}` | **ported** M14.b: an atomic byte total that refuses an update leaving it above the limit (`Memory limit exceeded for <name>`) or below zero (`Illegal Memory State for <name>`), after applying it, as Java does. Unit tests only (`memory.rs`): there is no Lucene output to compare. Differs: the refusals are `Error::IllegalState`. |
| `misc/SweetSpotSimilarity` | `lucene-search/src/misc/sweet_spot.rs::SweetSpotSimilarity` | **ported** M14.b: `ClassicSimilarity` with the plateaued `lengthNorm`, `baselineTf` as `tf`, `hyperbolicTf` (`StrictMath.pow`), the setters and `toString`; its own `SimScorer` builds the norm table from the overridden `lengthNorm`. Equal to Lucene's, as float bits: both configurations' function values, `toString`, and the top ten of four queries over the `high_freq` index -- `misc_fixtures.rs` (`GenMiscSweetSpot`, run with the `Math.pow`/`Math.log` intrinsics off). |

`lucene-search/src/misc/mod.rs` is the module facade and has no Java counterpart of its own.
