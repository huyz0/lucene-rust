//! `lucene-sandbox` (Lucene 10.5.0's `sandbox` module), ported package by
//! package. The inventory is `docs/inventory/lucene-sandbox.tsv`; the ledger
//! is `docs/parity/sandbox.md`; the milestone is M14.f.
//!
//! - [`document`]: `org.apache.lucene.sandbox.document` -- `HalfFloatPoint`,
//!   `BigIntegerPoint`, `LatLonBoundingBox`, `FloatPointNearestNeighbor` and
//!   the four `*PointMultiRangeBuilder`s.
//! - [`search`]: `org.apache.lucene.sandbox.search`'s queries and collectors.
//! - [`payloads`]: `org.apache.lucene.payloads` -- `PayloadSpanUtil` and
//!   `PayloadSpanCollector`.
//!
//! `sandbox/index/MergeOnFlushMergePolicy` is `lucene-index`'s
//! `merge_policy::merge_on_flush`.
//!
//! `queries/FuzzyLikeThisQuery` lives at [`crate::fuzzy_like_this`], where
//! M13's `CorePlusExtensionsParser` needed it first.

pub mod document;
pub mod payloads;
pub mod search;
