#![forbid(unsafe_code)]
//! lucene-codecs: see /PLAN.md for scope.

// Every module below is audited against the arithmetic gate
// (`clippy::arithmetic_side_effects`, denied crate-wide via
// `[lints] workspace = true`) -- see `docs/arithmetic-gate.md`, where this
// crate's burn-down row reads "none". A new module is gated from its first
// line and needs nothing here; if one ever has to be exempted, the opt-out is
// a `#[allow(clippy::arithmetic_side_effects)] // TODO(arith-audit)` on its
// `mod` declaration, and `scripts/check-arith-allows.py` cross-checks the
// count against that table.

// `direct_reader`, `for_util` and `lz4` are `pub` for a reason that is
// recorded in each module's own `//!` header, not here: an outer `///` doc
// on a `mod` declaration makes rustdoc resolve *the whole merged doc* --
// including the module file's own `//!` lines -- in the crate-root scope,
// which silently breaks every intra-doc link the module writes to its own
// items. See `docs/rustdoc-gate.md`.
mod block_packed;
pub mod blocktree;
pub mod compound_format;
mod deflate;
pub mod direct_monotonic;
pub mod direct_reader;
pub mod doc_values;
pub mod doc_values_updates;
pub mod field_infos;
pub mod for_util;
pub mod fst;
pub mod fuzzy;
pub mod hnsw;
pub mod hnsw_vectors;
pub mod indexed_disi;
pub mod live_docs;
pub mod lz4;
pub mod norms;
mod packed_ints;
pub mod points;
pub mod postings;
pub mod postings_writer;
pub mod regexp;
pub mod stored_fields;
pub mod suggest;
pub mod term_vectors;
pub mod terms_dict;
pub mod vectors;
pub mod wildcard;
