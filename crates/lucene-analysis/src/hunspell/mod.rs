//! `org.apache.lucene.analysis.hunspell`: Hunspell dictionaries
//! ([`Dictionary`]), stemming ([`Stemmer`], [`HunspellStemFilter`]), spell
//! checking and suggestions ([`Hunspell`]), word-form generation
//! ([`WordFormGenerator`]).
//!
//! The port keeps Lucene's text model: words, affixes and flags are Java
//! `char`s (UTF-16 units, flags `u16`), so positions, `String.compareTo`
//! order, `String.hashCode` slots and per-`char` case mapping are Lucene's.
//! The public API takes and returns `&str`/`String`; an unpaired surrogate a
//! stem could carry becomes U+FFFD, as everywhere in this crate.

mod affix_condition;
mod affixed_word;
mod conv_table;
mod dictionary;
mod flags;
mod speller;
mod stemmer;
mod suggester;
mod timeout;
mod word_case;
mod word_form_generator;
mod word_storage;

pub use affixed_word::{Affix, AffixedWord};
pub use dictionary::{DictEntry, Dictionary};
pub use speller::Hunspell;
pub use stemmer::Stemmer;
pub use suggester::Suggester;
pub use timeout::{CheckCanceled, SuggestionTimeout, TimeoutPolicy, SUGGEST_TIME_LIMIT};
pub use word_form_generator::{
    EverythingPossible, FragmentChecker, NGramFragmentChecker, WordFormGenerator,
};

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `Dictionary.FLAG_UNSET`.
pub(crate) const FLAG_UNSET: u16 = 0;
/// `Dictionary.HIDDEN_FLAG` (Hunspell's `ONLYUPCASEFLAG`).
pub(crate) const HIDDEN_FLAG: u16 = 65511;

/// Why a dictionary could not be loaded (Java's `ParseException`,
/// `IllegalArgumentException`, `IllegalStateException`,
/// `NumberFormatException`, `ArrayIndexOutOfBoundsException`,
/// `NegativeArraySizeException`, `UnsupportedOperationException`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HunspellError {
    /// `java.text.ParseException`, with its error offset (the line number).
    #[error("{message} (line {line})")]
    Parse {
        /// The message.
        message: String,
        /// `getErrorOffset()`: the line number.
        line: usize,
    },
    /// `IllegalArgumentException`.
    #[error("illegal argument: {0}")]
    IllegalArgument(String),
    /// `IllegalStateException` (or another runtime exception).
    #[error("illegal state: {0}")]
    IllegalState(String),
    /// `NumberFormatException`.
    #[error("number format: {0}")]
    NumberFormat(String),
    /// `ArrayIndexOutOfBoundsException` (more `AF`/`AM` lines than
    /// announced, a flag directive without a flag).
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(String),
    /// `NegativeArraySizeException` (a negative `AF`/`AM` count).
    #[error("negative array size: {0}")]
    NegativeArraySize(String),
    /// `UnsupportedOperationException`, or a feature this port does not
    /// support (a charset other than UTF-8/ISO8859-1/ISO8859-14).
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// `org.apache.lucene.analysis.hunspell.HunspellStemFilter`: replaces each
/// non-keyword token by its stems (the extra stems at position increment
/// 0); an unknown word passes unchanged.
pub struct HunspellStemFilter<I> {
    input: I,
    dictionary: std::sync::Arc<Dictionary>,
    buffer: Vec<Vec<u16>>,
    /// The current term's units (reused across tokens).
    term: Vec<u16>,
    saved_state: Option<State>,
    dedup: bool,
    longest_only: bool,
}

impl<I: TokenStream> HunspellStemFilter<I> {
    /// `new HunspellStemFilter(input, dictionary, dedup, longestOnly)`
    /// (`dedup` is ignored when `longestOnly` is set, as in Java).
    pub fn new(
        input: I,
        dictionary: std::sync::Arc<Dictionary>,
        dedup: bool,
        longest_only: bool,
    ) -> Self {
        HunspellStemFilter {
            input,
            dictionary,
            buffer: Vec::new(),
            term: Vec::new(),
            saved_state: None,
            dedup: dedup && !longest_only,
            longest_only,
        }
    }

    /// `new HunspellStemFilter(input, dictionary)`: every stem, deduplicated.
    pub fn with_defaults(input: I, dictionary: std::sync::Arc<Dictionary>) -> Self {
        Self::new(input, dictionary, true, false)
    }
}

impl<I: TokenStream> TokenFilter for HunspellStemFilter<I> {
    crate::filter_input!();
    // Java: HunspellStemFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.buffer.is_empty() {
            let next = self.buffer.remove(0);
            let a = self.input.attributes_mut();
            if let Some(state) = &self.saved_state {
                a.restore_state(state);
            }
            a.set_position_increment(0)?;
            a.set_term_utf16(&next);
            return Ok(true);
        }
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if a.is_keyword() {
            return Ok(true);
        }
        self.term.clear();
        self.term.extend(a.term().encode_utf16());
        let stemmer = Stemmer::new(&self.dictionary);
        let mut buffer = if self.dedup {
            stemmer.unique_stems_units(&self.term)
        } else {
            stemmer.stem_units(&self.term)
        };
        if buffer.is_empty() {
            return Ok(true);
        }
        if self.longest_only && buffer.len() > 1 {
            buffer.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| b.cmp(a)));
        }
        let stem = buffer.remove(0);
        a.set_term_utf16(&stem);
        if self.longest_only {
            buffer.clear();
        } else if !buffer.is_empty() {
            self.saved_state = Some(a.capture_state());
        }
        self.buffer = buffer;
        Ok(true)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
