#![forbid(unsafe_code)]
//! Multi-vector (late-interaction, ColBERT-style) fields and `MaxSim`
//! scoring: a document holds one vector per token, and a query of token
//! vectors scores it by the sum, over its tokens, of each one's best dot
//! product with any of the document's.
//!
//! A multi-vector is stored as one `BINARY` doc value ([`encoding`]), so the
//! files stay Lucene's. [`TokenEncoding::Float32`] is byte for byte Lucene's
//! own `LateInteractionField`; [`TokenEncoding::Float16`] and
//! [`TokenEncoding::Int8`] are this crate's, smaller and laid out for the
//! scoring kernel ([`lucene_util::max_sim`]), and Lucene reads them as plain
//! binary doc values.
//!
//! [`MaxSimQuery`] scores one stored value, [`MaxSimValuesSource`] serves a
//! field's values to the search crate's rescorers ([`max_sim_rescorer`]), and
//! [`pooled_mean`] folds a multi-vector into the single vector a `knn_vector`
//! field can index for candidate generation.

pub mod encoding;
pub mod pooled;
pub mod score;
pub mod values_source;

pub use encoding::{blocks_of, decode_into_blocks, encode, TokenEncoding};
pub use pooled::pooled_mean;
pub use score::MaxSimQuery;
pub use values_source::{max_sim_rescorer, MaxSimValuesSource, Missing};

/// What can go wrong reading or writing a multi-vector.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// A multi-vector or query that is empty, ragged, or not what its header
    /// says.
    #[error("{0}")]
    IllegalArgument(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn illegal(msg: impl Into<String>) -> Error {
    Error::IllegalArgument(msg.into())
}
