#![forbid(unsafe_code)]
//! lucene-analysis-kuromoji: Lucene's `analysis-kuromoji` module (M12
//! T12.1) -- Japanese morphological analysis over the IPADIC dictionary.
//!
//! - [`JapaneseTokenizer`] over `lucene-analysis`' `morph` Viterbi search
//!   ([`viterbi`]), the dictionaries ([`dict`]), its attributes
//!   ([`attributes`]) and tokens ([`token`]).
//! - The filters ([`base_form`], [`pos_stop`], [`reading_form`],
//!   [`katakana_stem`], [`uppercase`], [`number`], [`completion`]), the
//!   char filter ([`iteration_mark`]), [`JapaneseAnalyzer`],
//!   [`JapaneseCompletionAnalyzer`] and the factories
//!   ([`register_factories`]).

pub mod analyzer;
pub mod attributes;
pub mod base_form;
pub mod completion;
pub mod dict;
pub mod factory;
pub mod iteration_mark;
pub mod katakana_stem;
pub mod number;
pub mod pos_stop;
pub mod reading_form;
pub mod token;
pub mod tokenizer;
pub mod uppercase;
pub mod viterbi;

pub use analyzer::{JapaneseAnalyzer, JapaneseCompletionAnalyzer};
pub use attributes::{
    BaseFormAttribute, InflectionAttribute, PartOfSpeechAttribute, ReadingAttribute,
};
pub use dict::UserDictionary;
pub use factory::register_factories;
pub use token::Token;
pub use tokenizer::{JapaneseTokenizer, Mode};
