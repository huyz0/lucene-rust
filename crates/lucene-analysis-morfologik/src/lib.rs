#![forbid(unsafe_code)]
//! lucene-analysis-morfologik: Lucene's `analysis-morfologik` module (M12
//! T12.3) -- `MorfologikFilter`, `MorphosyntacticTagsAttribute`,
//! `MorfologikAnalyzer`, `UkrainianMorfologikAnalyzer` and
//! `MorfologikFilterFactory` -- over the parts of Morfologik 2.1.9
//! (`morfologik-fsa`, `morfologik-stemming`; BSD, Copyright (c) 2006 Dawid
//! Weiss, (c) 2007-2016 Dawid Weiss, Marcin Miłkowski) a lookup needs:
//! the automata ([`fsa`]), dictionary metadata ([`metadata`],
//! [`properties`]) and `DictionaryLookup` with its sequence decoders
//! ([`dictionary`]). Building automata and dictionaries is not ported (M12
//! loads dictionaries, it does not build them).

pub mod analyzer;
pub mod dictionary;
pub mod factory;
pub mod filter;
pub mod fsa;
pub mod metadata;
pub mod properties;

pub use analyzer::{MorfologikAnalyzer, UkrainianMorfologikAnalyzer};
pub use dictionary::{Dictionary, DictionaryLookup, WordData};
pub use factory::{register_factories, MorfologikFilterFactory};
pub use filter::{MorfologikFilter, MorphosyntacticTagsAttribute};

use std::fmt;

/// A dictionary that cannot be read or looked up in: the Java exception's
/// simple name and message, as one string (`"IOException: ..."`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorfologikError {
    message: String,
}

impl MorfologikError {
    /// An error with `message`.
    pub fn new(message: impl Into<String>) -> Self {
        MorfologikError {
            message: message.into(),
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for MorfologikError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MorfologikError {}

impl From<MorfologikError> for lucene_analysis::AnalysisError {
    fn from(e: MorfologikError) -> Self {
        lucene_analysis::AnalysisError::Io(e.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_text() {
        let e = MorfologikError::new("IOException: x");
        assert_eq!(e.message(), "IOException: x");
        assert_eq!(e.to_string(), "IOException: x");
        let a: lucene_analysis::AnalysisError = e.into();
        assert!(matches!(a, lucene_analysis::AnalysisError::Io(_)));
    }
}
