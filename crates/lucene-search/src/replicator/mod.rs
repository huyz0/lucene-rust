//! `lucene-replicator` (Lucene 10.5.0's `replicator` module), ported class by
//! class. The inventory is `docs/inventory/lucene-replicator.tsv`; the
//! milestone is M14.e.
//!
//! Lucene 10.5.0's replicator is `org.apache.lucene.replicator.nrt` alone
//! (the revision replicator and its HTTP service are gone): a primary node
//! that turns each near-real-time point of an `IndexWriter` into a
//! [`nrt::CopyState`], and replica nodes that copy the files of that state
//! and open it.

pub mod nrt;
