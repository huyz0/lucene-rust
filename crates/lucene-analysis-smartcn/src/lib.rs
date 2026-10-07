#![forbid(unsafe_code)]
//! lucene-analysis-smartcn: Lucene's `analysis-smartcn` module (M12 T12.2)
//! -- Chinese word segmentation by a hierarchical hidden Markov model
//! ([`hhmm`]) over the core word dictionary and the word-bigram dictionary
//! Lucene ships (`coredict.mem`, `bigramdict.mem`, vendored zlib-compressed
//! and read through [`serialized`], a reader of the Java serialization they
//! are stored in), `HMMChineseTokenizer` (on `lucene-analysis`'
//! `SegmentingTokenizerBase` and the JDK's sentence iterator),
//! `SmartChineseAnalyzer` and `HMMChineseTokenizerFactory`.
//!
//! The SmartChineseAnalyzer source and dictionaries were provided by
//! Xiaoping Gao, copyright 2009 www.imdict.net, under the Apache License 2.0
//! (`NOTICE`).
//!
//! `AnalyzerProfile` and the `.dct` dictionary loaders are not ported: Java
//! reads `analysis.data.dir`'s `coredict.dct`/`bigramdict.dct` only when the
//! jar's `.mem` resources fail to load, and here they are compiled in.

pub mod analyzer;
pub mod factory;
pub mod hhmm;
pub mod serialized;
pub mod tokenizer;
pub mod utility;
pub mod word_segmenter;

pub use analyzer::SmartChineseAnalyzer;
pub use factory::{register_factories, HMMChineseTokenizerFactory};
pub use tokenizer::HMMChineseTokenizer;

use std::fmt;

/// A dictionary that cannot be read (Java's `IOException` or
/// `ClassNotFoundException` from `ObjectInputStream`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartcnError {
    message: String,
}

impl SmartcnError {
    /// An error with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        SmartcnError {
            message: message.into(),
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SmartcnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SmartcnError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_text() {
        let e = SmartcnError::new("bad");
        assert_eq!(e.message(), "bad");
        assert_eq!(e.to_string(), "bad");
    }
}
