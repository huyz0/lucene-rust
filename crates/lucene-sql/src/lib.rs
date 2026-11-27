#![forbid(unsafe_code)]
//! `lucene-sql`: read-only SQL over an index.
//!
//! A statement is first offered to the [`router`], which recognises four hot
//! shapes (point read by key, `match_text` top-k, `knn` top-k, and a
//! group-by on a keyword) from the parse tree and answers them with the
//! [`native`] API; everything else runs on DataFusion over the
//! [`provider`]'s table, whose scans call the same native searches and into
//! which [`rule`] pushes `ORDER BY _score DESC LIMIT k`.
//!
//! The table is declared with a [`spec::TableSpec`] and read through a
//! [`snapshot::Snapshot`]; [`engine::SqlEngine`] ties them together.

pub mod columns;
pub mod engine;
pub mod error;
pub mod group;
pub mod group_rule;
pub mod native;
pub mod plan_cache;
pub mod pred;
pub mod provider;
pub mod router;
pub mod rule;
pub mod snapshot;
pub mod spec;
pub use engine::{SqlConfig, SqlEngine};
pub use error::{Error, Result};
pub use snapshot::Snapshot;
pub use spec::TableSpec;
