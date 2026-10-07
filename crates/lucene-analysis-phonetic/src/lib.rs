#![forbid(unsafe_code)]
//! lucene-analysis-phonetic: Lucene's `analysis-phonetic` module (M12
//! T12.3) -- `PhoneticFilter`, `DoubleMetaphoneFilter`,
//! `BeiderMorseFilter`, `DaitchMokotoffSoundexFilter` and their factories
//! -- with the Apache Commons Codec 1.17.2 encoders they run (the version
//! Lucene 10.5.0's module declares) reimplemented here: `Soundex`,
//! `RefinedSoundex`, `Metaphone`, `DoubleMetaphone`, `Caverphone1`/`2`,
//! `ColognePhonetic`, `Nysiis`, `MatchRatingApproachEncoder`,
//! `DaitchMokotoffSoundex` and Beider-Morse ([`bm`]), with the rule files
//! Commons Codec ships (Apache-2.0, vendored under `src/resources/`).
//!
//! The encoders work on Java `char`s (UTF-16 units): every one of them
//! indexes, slices and compares `char`s, and `String.toUpperCase`'s
//! special casing (`ß` -> `SS`) changes lengths. The filters convert a
//! term at the boundary.
//!
//! [`register_factories`] adds the four factories to `lucene-analysis`'
//! SPI registry, as Java finds them on the module path.

pub mod bm;
pub mod caverphone;
pub mod cologne;
pub mod daitch_mokotoff;
pub mod double_metaphone;
pub mod encoder;
pub mod factory;
pub mod filters;
mod java;
pub mod match_rating;
pub mod metaphone;
pub mod nysiis;
pub mod soundex;

pub use encoder::Encoder;
pub use factory::register_factories;
pub use filters::{
    BeiderMorseFilter, DaitchMokotoffSoundexFilter, DoubleMetaphoneFilter, PhoneticFilter,
};

use std::fmt;

/// An exception an encoder throws: the Java class's simple name and the
/// message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderError {
    class: &'static str,
    message: String,
}

impl EncoderError {
    /// An `IllegalArgumentException`.
    pub fn illegal_argument(message: impl Into<String>) -> Self {
        EncoderError {
            class: "IllegalArgumentException",
            message: message.into(),
        }
    }

    /// A `NullPointerException`: `PhoneticFilter`'s `encode(value).toString()`
    /// on an encoder that answered `null`.
    pub fn null_result() -> Self {
        EncoderError {
            class: "NullPointerException",
            message: String::new(),
        }
    }

    /// The Java exception's simple class name.
    pub fn java_class(&self) -> &'static str {
        self.class
    }

    /// The exception's message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class, self.message)
    }
}

impl std::error::Error for EncoderError {}

impl From<EncoderError> for lucene_analysis::AnalysisError {
    fn from(e: EncoderError) -> Self {
        lucene_analysis::AnalysisError::IllegalArgument(e.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors() {
        let e = EncoderError::illegal_argument("bad");
        assert_eq!(e.java_class(), "IllegalArgumentException");
        assert_eq!(e.to_string(), "IllegalArgumentException: bad");
        assert_eq!(
            EncoderError::null_result().java_class(),
            "NullPointerException"
        );
        let a: lucene_analysis::AnalysisError = e.into();
        assert!(matches!(a, lucene_analysis::AnalysisError::IllegalArgument(m) if m == "bad"));
    }
}
