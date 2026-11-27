//! The errors a SQL query over an index can end in.

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
    /// A statement the SQL layer does not run, or a table definition that
    /// does not fit the index.
    #[error("{0}")]
    Unsupported(String),
    /// DataFusion rejected or failed the statement.
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

impl Error {
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}

/// `err` wrapped for a DataFusion stream.
pub(crate) fn external(err: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}

impl Error {
    pub(crate) fn from_arrow(e: datafusion::arrow::error::ArrowError) -> Self {
        Error::DataFusion(DataFusionError::ArrowError(Box::new(e), None))
    }
}
