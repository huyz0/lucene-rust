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
mod block_packed;
pub mod blocktree;
pub mod compound_format;
mod deflate;
pub mod direct_monotonic;
/// `DirectReader`/`DirectWriter` bit-packing. Public for the same reason
/// [`for_util`] is: a per-value decode primitive on the doc-values and
/// monotonic-sequence paths, and `DirectReader.getInstance` is public in
/// Lucene, so the two can be benchmarked directly against each other.
pub mod direct_reader;
pub mod doc_values;
pub mod doc_values_updates;
pub mod field_infos;
/// `ForUtil`/`PForUtil` bit-packing. Public so that the decode kernel can be
/// benchmarked against Lucene's own from outside the crate -- Lucene makes
/// `PostingIndexInput` public for exactly this reason, and a kernel this hot
/// with no external microbenchmark is how a 3x regression hides inside a flat
/// profile. Not otherwise part of this crate's intended surface: callers want
/// `postings`, which owns the framing these primitives sit inside.
pub mod for_util;
pub mod fst;
pub mod fuzzy;
pub mod hnsw;
pub mod hnsw_vectors;
pub mod indexed_disi;
pub mod live_docs;
/// LZ4 block compression/decompression (`org.apache.lucene.util.compress.LZ4`).
/// Public for the same reason [`for_util`] is: `LZ4` is a public class in
/// Lucene, both hash-table strategies are part of its contract, and a
/// compressor this hot deserves a microbenchmark that can call it from
/// outside the crate.
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
