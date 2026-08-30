#![forbid(unsafe_code)]
//! lucene-index: see /PLAN.md for scope.

// Every module below is audited against the arithmetic gate
// (`clippy::arithmetic_side_effects`, denied crate-wide via
// `[lints] workspace = true`) -- see `docs/arithmetic-gate.md`, where this
// crate's burn-down row reads "none". A new module is gated from its first
// line and needs nothing here; if one ever has to be exempted, the opt-out is
// a `#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)` on its
// `mod` declaration, and `scripts/check-arith-allows.py` cross-checks the
// count against that table.
pub mod buffered_updates;
pub mod check_index;
pub mod checksum_verify;
pub mod deletes;
pub mod field_updates;
pub mod index_file_deleter;
pub mod index_writer;
pub mod indexing_chain;
pub mod merge;
pub mod merge_policy;
pub mod points_delete;
pub mod segment_info;
pub mod segment_infos;
pub mod segment_writer;
pub mod term_delete;
pub mod update_document;
