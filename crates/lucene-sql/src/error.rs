//! The errors a SQL query over an index can end in.

#[cfg(feature = "sql")]
use datafusion::error::DataFusionError;

/// What went wrong running a statement or opening a table.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The index could not be read.
    #[error("index: {0}")]
    Index(#[from] lucene_search::Error),
    /// A segment could not be opened or read.
    #[error("index: {0}")]
    Reader(#[from] lucene_search::directory_reader::Error),
    /// Arrow could not build an array or batch.
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// A statement the SQL layer does not run, or a table definition that
    /// does not fit the index.
    #[error("{0}")]
    Unsupported(String),
    /// DataFusion rejected or failed the statement.
    #[cfg(feature = "sql")]
    #[error("sql: {0}")]
    DataFusion(#[from] DataFusionError),
}

/// `Result` of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Marks a failure that only means "the aggregation kernels cannot answer this
/// exactly, run the statement the standard way": the engine catches it and
/// plans the statement again without the kernel rule.
#[derive(Debug, thiserror::Error)]
#[error("the aggregation kernels cannot answer this exactly: {0}")]
pub struct KernelInexact(pub String);

impl From<lucene_util::memory::Exceeded> for Error {
    fn from(e: lucene_util::memory::Exceeded) -> Self {
        Error::Index(e.into())
    }
}

impl Error {
    /// The bound or cancel that stopped the statement, when one did: the
    /// host's request breaker, the process bound, a cancelled task or the
    /// statement's deadline.
    pub fn stopped(&self) -> Option<&lucene_util::memory::Exceeded> {
        match self {
            Error::Index(lucene_search::Error::Memory(e)) => Some(e),
            #[cfg(feature = "sql")]
            Error::DataFusion(e) => match e.find_root() {
                DataFusionError::External(inner) => match inner.downcast_ref::<Error>() {
                    Some(Error::Index(lucene_search::Error::Memory(e))) => Some(e),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}

/// `err` wrapped for a DataFusion stream.
#[cfg(feature = "sql")]
pub(crate) fn external(err: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}
