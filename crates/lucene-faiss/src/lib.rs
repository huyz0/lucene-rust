#![forbid(unsafe_code)]
//! Lucene's sandbox Faiss codec, natively: the `.faissm`/`.faissd` files of
//! `FaissKnnVectorsFormat` and the Faiss index serialization they hold, read,
//! written and searched here without linking Faiss.
//!
//! Targets Lucene `main` (11.0), whose sandbox codec has no backward
//! compatibility guarantee.

// ARITH: indices run over `0..n`, `0..nlist`, `0..d` where n, d and the id count are checked equal before any loop; counts are bounded by the input slice length, so none overflows `usize`.
#[allow(clippy::arithmetic_side_effects)]
pub mod build;
pub mod codec;
mod distance;
pub mod error;
pub mod factory;
pub mod field;
// ARITH: every `n * width` is preceded by a check that `n` fits the bytes left in the buffer (`take`), so the products are bounded by the file length; remaining-length subtractions follow `pos <= len` invariants of the reader.
#[allow(clippy::arithmetic_side_effects)]
pub mod format;
// ARITH: heap positions are below the heap's capacity `k` or `ef`, which the caller allocated, so `2 * i + 1` and similar index steps cannot overflow.
#[allow(clippy::arithmetic_side_effects)]
mod heap;
// ARITH: indices run over the training set and `k <= n` (checked by the caller); counts are bounded by the input slice length.
#[allow(clippy::arithmetic_side_effects)]
pub mod kmeans;
// ARITH: ids and list positions come from an index `read_index` validated (sizes against the remaining bytes, ids against `ntotal`); loops run over those bounds.
#[allow(clippy::arithmetic_side_effects)]
pub mod search;
// ARITH: loops run over the `d` components of a quantizer whose vectors have `d` entries (checked on read and in `search`); `c * 8 + i` is below `d`.
#[allow(clippy::arithmetic_side_effects)]
mod sq;
// ARITH: `i * d` indexes a store whose code or float array holds `len * d` entries, with `i < len` from the validated graph.
#[allow(clippy::arithmetic_side_effects)]
mod store;
mod visited;

pub use build::build_index;
pub use error::{Error, Result};
pub use factory::{parse_description, parse_params, Description, IndexParams, Kind};
pub use field::{FaissField, Similarity};
pub use format::{read_index, write_index, Index, Metric, Sq8};
pub use search::{SearchParams, Selector};
