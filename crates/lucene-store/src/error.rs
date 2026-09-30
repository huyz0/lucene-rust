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
    /// Port of `LockObtainFailedException`: a lock (in practice
    /// `write.lock`) is held elsewhere -- by another process, or by another
    /// writer in this one.
    #[error("lock obtain failed: {0}")]
    LockObtainFailed(String),
    /// Port of `LockReleaseFailedException`: a lock could not be released
    /// cleanly, and the directory may need manual intervention.
    #[error("lock release failed: {0}")]
    LockReleaseFailed(String),
    /// Port of `AlreadyClosedException` where it survives ownership: a lock
    /// that was released, or invalidated behind its holder's back.
    #[error("already closed: {0}")]
    AlreadyClosed(String),
    /// Port of `IllegalArgumentException` from the store package's
    /// constructors and `NRTCachingDirectory.rename`.
    #[error("illegal argument: {0}")]
    IllegalArgument(String),
}

impl Error {
    /// `NoSuchFileException`/`FileNotFoundException`: the `catch` clauses
    /// `FileSwitchDirectory.listAll` and `NRTCachingDirectory.slowFileExists`
    /// narrow on.
    pub fn is_no_such_file(&self) -> bool {
        matches!(self, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound)
    }

    /// Java's `NoSuchFileException(message)`.
    pub(crate) fn no_such_file(message: impl Into<String>) -> Self {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            message.into(),
        ))
    }

    /// Java's `FileAlreadyExistsException(message)`.
    pub(crate) fn file_already_exists(message: impl Into<String>) -> Self {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            message.into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_helpers_carry_the_java_exception_kind() {
        assert!(Error::no_such_file("x").is_no_such_file());
        let e = Error::file_already_exists("y");
        assert!(matches!(&e, Error::Io(io) if io.kind() == std::io::ErrorKind::AlreadyExists));
        assert!(!e.is_no_such_file());
        assert!(!Error::Corrupted("z".into()).is_no_such_file());
        assert_eq!(
            Error::LockObtainFailed("held".into()).to_string(),
            "lock obtain failed: held"
        );
        assert_eq!(
            Error::LockReleaseFailed("r".into()).to_string(),
            "lock release failed: r"
        );
        assert_eq!(
            Error::AlreadyClosed("c".into()).to_string(),
            "already closed: c"
        );
        assert_eq!(
            Error::IllegalArgument("a".into()).to_string(),
            "illegal argument: a"
        );
    }
}
