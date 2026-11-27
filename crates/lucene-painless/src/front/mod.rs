//! The front end: Painless source text to a syntax tree.
//!
//! The grammar is `PainlessLexer.g4` and `PainlessParser.g4` of OpenSearch's
//! `lang-painless`, parsed by hand. Later phases (analysis and lowering to
//! [`crate::ir`]) consume the tree; nothing here resolves a type or a name.

pub mod allowlist;
pub mod analyze;
pub mod ast;
mod lex;
mod parse;
mod sites;

pub use analyze::{analyze, analyze_in, Binding, Refusal, PARAMS, SCORE};
pub use parse::{parse, Limits};

/// Why source does not parse: where, and what was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset into the source.
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at offset {}", self.message, self.offset)
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests;
