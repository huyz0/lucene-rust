//! `lucene-misc` (Lucene 10.5.0's `misc` module), ported class by class. The
//! inventory is `docs/inventory/lucene-misc.tsv`; the milestone is M14.

pub mod high_freq;

pub use high_freq::{get_high_freq_terms, TermOrder, TermStats, DEFAULT_NUMTERMS};
