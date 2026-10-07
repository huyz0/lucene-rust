#![forbid(unsafe_code)]
//! lucene-analysis-stempel: Lucene's `analysis-stempel` module (M12 T12.2)
//! -- the Egothor stemmer's tables ([`egothor`], [`diff`]),
//! `StempelStemmer`, `StempelFilter`, `PolishAnalyzer` and
//! `StempelPolishStemFilterFactory` -- with Lucene's default Polish table
//! (`stemmer_20000.tbl`) and stop words vendored.
//!
//! This product includes software developed by the Egothor Project
//! (<http://egothor.sf.net/>); see [`egothor`] for its licence. The
//! table-building half of Egothor (`Compile`, `DiffIt`, `Gener`, `Lift`,
//! `Optimizer`, `Reduce`, `Diff.exec`, `Trie.add`/`reduce`) is not ported:
//! M12 loads tables, it does not build them.

pub mod diff;
pub mod egothor;
pub mod factory;
pub mod filter;
pub mod polish;
pub mod stemmer;

pub use factory::{register_factories, StempelPolishStemFilterFactory};
pub use filter::StempelFilter;
pub use polish::PolishAnalyzer;
pub use stemmer::StempelStemmer;

use std::fmt;

/// A stemmer table that cannot be read (Java's `IOException`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StempelError {
    message: String,
}

impl StempelError {
    /// An error with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        StempelError {
            message: message.into(),
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for StempelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StempelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_text() {
        let e = StempelError::new("bad");
        assert_eq!(e.message(), "bad");
        assert_eq!(e.to_string(), "bad");
    }
}
