//! Errors of the Faiss index reader, writer, factory and search.

/// What went wrong reading, writing, describing or searching a Faiss index.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The bytes are not a Faiss index this port reads, or are cut short.
    #[error("corrupt faiss index: {0}")]
    Corrupt(String),
    /// A valid Faiss index of a kind this port does not implement (a PQ or
    /// scalar-quantized storage, a binary index, ...).
    #[error("unsupported faiss index: {0}")]
    Unsupported(String),
    /// An index factory string this port does not implement, or that is
    /// malformed. Only `[IDMap,]HNSW<M>[,Flat]`, `[IDMap,]IVF<nlist>,Flat`
    /// and `[IDMap,]Flat` are built.
    #[error("unsupported faiss index description [{description}]: {reason}")]
    Description {
        /// The factory string as given.
        description: String,
        /// Why it is refused.
        reason: String,
    },
    /// The vectors or the query do not fit the index (dimension, count).
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// A Lucene file framing error (header, footer, checksum).
    #[error(transparent)]
    Store(#[from] lucene_store::error::Error),
}

/// `Result` of this crate.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn corrupt<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Corrupt(msg.into()))
}
