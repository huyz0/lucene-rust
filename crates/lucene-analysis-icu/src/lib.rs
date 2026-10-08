#![forbid(unsafe_code)]
//! lucene-analysis-icu: Lucene's `analysis-icu` module (M12 T12.4) --
//! Unicode normalization (`ICUNormalizer2Filter`, `ICUNormalizer2CharFilter`),
//! UTR #30 folding (`ICUFoldingFilter`), script-aware segmentation
//! (`ICUTokenizer`), collation keys (`ICUCollationKeyAnalyzer`,
//! `ICUCollationAttributeFactory`, `ICUCollationDocValuesField` over
//! [`Collator`]) and their factories, over a port of
//! the ICU4J 77.1 runtime pieces they call ([`icu4j`]) reading ICU's own
//! binary data.
//!
//! **Why a port of ICU4J and not ICU4X.** Lucene's module ships ICU binary
//! data of its own -- `utr30.nrm` (the folding normalizer) and the compiled
//! break rules `Default.brk` and `MyanmarSyllable.brk` -- in ICU4J's formats,
//! and its API takes a caller's `.nrm` too. ICU4X reads neither format (its
//! data is its own, built by its datagen tool), so it could not run Lucene's
//! data at all; reading ICU's `.nrm`/`.brk`/`.dict` files with ICU4J's
//! algorithms is what reproduces ICU4J byte for byte by construction, which
//! the differential tests then check. ICU4J's code and data are under the
//! Unicode License v3 (`NOTICE`, `docs/licences.md`).

pub mod collation;
pub mod factory;
pub mod folding_filter;
pub mod icu4j;
pub mod normalizer2_char_filter;
pub mod normalizer2_filter;
pub mod segmentation;
pub mod tokenattributes;

pub use collation::{
    ICUCollatedTermFilter, ICUCollationAttributeFactory, ICUCollationDocValuesField,
    ICUCollationKeyAnalyzer,
};
pub use factory::{
    register_factories, ICUFoldingFilterFactory, ICUNormalizer2CharFilterFactory,
    ICUNormalizer2FilterFactory, ICUTokenizerFactory,
};
pub use folding_filter::ICUFoldingFilter;
pub use icu4j::coll::collator::Collator;
pub use icu4j::normalizer2::{Mode, Normalizer2};
pub use icu4j::unicode_set::UnicodeSet;
pub use normalizer2_char_filter::ICUNormalizer2CharFilter;
pub use normalizer2_filter::ICUNormalizer2Filter;
pub use segmentation::{DefaultICUTokenizerConfig, ICUTokenizer, ICUTokenizerConfig};
pub use tokenattributes::ScriptAttribute;

use std::fmt;

/// The Java exception an ICU operation throws, as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcuErrorKind {
    /// `IllegalArgumentException` (a bad pattern, name or argument).
    IllegalArgument,
    /// `IllegalIcuArgumentException`, ICU's subclass of it (an unknown
    /// property or property value name).
    IllegalIcuArgument,
    /// `java.util.MissingResourceException` (no such ICU data item).
    MissingResource,
    /// `ICUUncheckedIOException` (unreadable or corrupt ICU data).
    Io,
    /// `UnsupportedOperationException`: an operation this port refuses by
    /// design (each refusal names what is not ported).
    UnsupportedOperation,
}

/// An ICU error: its Java exception class and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcuError {
    kind: IcuErrorKind,
    message: String,
}

impl IcuError {
    /// An `ICUUncheckedIOException` with `message` (corrupt data).
    pub fn new(message: impl Into<String>) -> Self {
        Self::with_kind(IcuErrorKind::Io, message)
    }

    /// An error of `kind`.
    pub fn with_kind(kind: IcuErrorKind, message: impl Into<String>) -> Self {
        IcuError {
            kind,
            message: message.into(),
        }
    }

    /// An `IllegalArgumentException`.
    pub fn illegal_argument(message: impl Into<String>) -> Self {
        Self::with_kind(IcuErrorKind::IllegalArgument, message)
    }

    /// An `UnsupportedOperationException` (not ported).
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::with_kind(IcuErrorKind::UnsupportedOperation, message)
    }

    /// The exception class.
    pub fn kind(&self) -> IcuErrorKind {
        self.kind
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for IcuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for IcuError {}

impl From<IcuError> for lucene_analysis::AnalysisError {
    fn from(e: IcuError) -> Self {
        match e.kind {
            IcuErrorKind::IllegalArgument | IcuErrorKind::IllegalIcuArgument => {
                lucene_analysis::AnalysisError::IllegalArgument(e.message)
            }
            IcuErrorKind::UnsupportedOperation => {
                lucene_analysis::AnalysisError::UnsupportedOperation(e.message)
            }
            IcuErrorKind::MissingResource | IcuErrorKind::Io => {
                lucene_analysis::AnalysisError::Io(e.message)
            }
        }
    }
}

impl From<IcuError> for lucene_analysis::factory::FactoryError {
    fn from(e: IcuError) -> Self {
        use lucene_analysis::factory::{FactoryError, JavaException};
        if let Some(rest) = e.message.strip_prefix("NumberFormatException: ") {
            return FactoryError::new(JavaException::NumberFormat, rest);
        }
        let kind = match e.kind {
            IcuErrorKind::IllegalArgument => JavaException::IllegalArgument,
            IcuErrorKind::IllegalIcuArgument => JavaException::IllegalIcuArgument,
            IcuErrorKind::MissingResource => JavaException::MissingResource,
            IcuErrorKind::Io => JavaException::Runtime,
            IcuErrorKind::UnsupportedOperation => JavaException::UnsupportedOperation,
        };
        FactoryError::new(kind, e.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::factory::FactoryError;
    use lucene_analysis::AnalysisError;

    #[test]
    fn error_conversions() {
        let e = IcuError::new("bad");
        assert_eq!(e.message(), "bad");
        assert_eq!(e.to_string(), "bad");
        assert_eq!(e.kind(), IcuErrorKind::Io);
        let cases = [
            (IcuError::illegal_argument("a"), "IllegalArgumentException"),
            (IcuError::unsupported("u"), "UnsupportedOperationException"),
            (
                IcuError::with_kind(IcuErrorKind::MissingResource, "m"),
                "MissingResourceException",
            ),
            (IcuError::new("io"), "RuntimeException"),
            (
                IcuError::with_kind(IcuErrorKind::IllegalIcuArgument, "i"),
                "IllegalIcuArgumentException",
            ),
            (
                IcuError::illegal_argument("NumberFormatException: x"),
                "NumberFormatException",
            ),
        ];
        for (e, class) in cases {
            assert_eq!(FactoryError::from(e.clone()).java_class(), class);
            let a = AnalysisError::from(e);
            assert!(!a.to_string().is_empty());
        }
        assert!(matches!(
            AnalysisError::from(IcuError::illegal_argument("x")),
            AnalysisError::IllegalArgument(_)
        ));
        assert!(matches!(
            AnalysisError::from(IcuError::unsupported("x")),
            AnalysisError::UnsupportedOperation(_)
        ));
        assert!(matches!(
            AnalysisError::from(IcuError::new("x")),
            AnalysisError::Io(_)
        ));
    }
}
