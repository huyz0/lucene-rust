//! `org.apache.lucene.analysis.core` (analysis-common's `core` package; the
//! module is named apart from Rust's `core`).
//!
//! `KeywordTokenizer`, `LowerCaseFilter` and `StopFilter` predate M11 and live
//! at the crate root ([`crate::KeywordTokenizer`], [`crate::LowerCaseFilter`],
//! [`crate::StopFilter`]); `WhitespaceTokenizer`, `LetterTokenizer` and
//! `UnicodeWhitespaceTokenizer` are [`crate::util::CharTokenizer`]s.

mod analyzers;
mod filters;
mod flatten_graph_filter;

pub use analyzers::{SimpleAnalyzer, StopAnalyzer, UnicodeWhitespaceAnalyzer, WhitespaceAnalyzer};
pub use filters::{DecimalDigitFilter, TypeTokenFilter, UpperCaseFilter};
pub use flatten_graph_filter::FlattenGraphFilter;
