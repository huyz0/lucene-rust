#![forbid(unsafe_code)]
//! Log-message fields stored as a shared template plus each message's
//! variables (Elasticsearch's `pattern_text` idea), so a log index keeps one
//! copy of each message shape instead of one per line.
//!
//! A message is split on single spaces. A token holding an ASCII digit is a
//! variable; the rest is the template. On 10.6M real Loghub lines this gives
//! 7,887 templates and rebuilds every line exactly. A field `f` is written as
//! three ordinary Lucene fields ([`fields()`]), so stock Lucene reads the
//! files:
//!
//! - `f`: the message, indexed docs-only without norms (as `match_only_text`),
//! - `f.template`: the template, a `SORTED` doc value,
//! - `f.args`: the variables, a stored binary value ([`format::encode_args`]),
//!   absent when the message has none.
//!
//! A phrase on `f` is the conjunction of its terms, verified per document by
//! [`PatternPhraseSource`]. The verdict is decided once per template where
//! the template alone settles it, so most phrases never read the variables.

pub mod fields;
pub mod format;
pub mod phrase;

pub use fields::{fields, ARGS_SUFFIX, TEMPLATE_SUFFIX};
pub use format::{decode_args, encode_args, join, split, Split, VARIABLE};
pub use phrase::{pattern_phrase_query, PatternPhraseSource, PhraseTokens, Tokenizer};

/// What can go wrong reading a pattern-text value.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// A stored template or variables value that is not this crate's format.
    #[error("corrupt pattern_text value: {0}")]
    Corrupt(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;
