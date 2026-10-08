//! `org.apache.lucene.analysis.icu.segmentation`: `ICUTokenizer` and its
//! script-aware break iteration.
//!
//! Lucene's `CharArrayIterator` (a `CharacterIterator` over a slice of the
//! tokenizer's buffer) is [`crate::icu4j::char_iter::CharIter`].

pub mod break_iterator_wrapper;
pub mod composite_break_iterator;
pub mod config;
pub mod script_iterator;
pub mod tokenizer;

pub use config::{DefaultICUTokenizerConfig, ICUTokenizerConfig};
pub use tokenizer::ICUTokenizer;
