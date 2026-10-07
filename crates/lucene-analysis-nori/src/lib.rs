#![forbid(unsafe_code)]
//! lucene-analysis-nori: Lucene's `analysis-nori` module (M12 T12.1) --
//! Korean morphological analysis over mecab-ko-dic.
//!
//! - [`KoreanTokenizer`] over `lucene-analysis`' `morph` Viterbi search
//!   ([`viterbi`]), the dictionaries ([`dict`]), [`pos`], its attributes
//!   ([`attributes`]) and tokens ([`token`]).
//! - The filters ([`pos_stop`], [`reading_form`], [`number`]),
//!   [`KoreanAnalyzer`] and the factories ([`register_factories`]).

pub mod analyzer;
pub mod attributes;
pub mod dict;
pub mod factory;
pub mod number;
pub mod pos;
pub mod pos_stop;
pub mod reading_form;
pub mod token;
pub mod tokenizer;
pub mod viterbi;

pub use analyzer::KoreanAnalyzer;
pub use attributes::{PartOfSpeechAttribute, ReadingAttribute};
pub use dict::UserDictionary;
pub use factory::register_factories;
pub use token::Token;
pub use tokenizer::KoreanTokenizer;
pub use viterbi::DecompoundMode;
