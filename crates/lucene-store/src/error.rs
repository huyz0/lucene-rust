use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unexpected end of input at offset {offset}")]
    Eof { offset: usize },
    #[error("malformed vint/vlong: too many bytes")]
    MalformedVarint,
    #[error("corrupted index: {0}")]
    Corrupted(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
