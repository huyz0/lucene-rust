//! `org.apache.lucene.analysis.icu.segmentation.ICUTokenizerConfig` and
//! `DefaultICUTokenizerConfig`: which break rules each script runs and what
//! token type each rule status is.
//!
//! The rules are compiled data: Lucene's `Default.brk` (UAX #29 word rules
//! with dictionary segmentation for Southeast Asian scripts and Hangul) and
//! `MyanmarSyllable.brk` (Apache-2.0, from the `analysis-icu` jar), and
//! ICU's root word rules `word.brk` (`BreakIterator.getWordInstance(ULocale.ROOT)`,
//! with dictionary segmentation for Chinese and Japanese) from the ICU4J
//! 77.1 jar.

use std::sync::{Arc, OnceLock};

use crate::icu4j::rbbi::RuleBasedBreakIterator;
use crate::icu4j::rbbi_data::RbbiData;
use crate::segmentation::script_iterator::{HANGUL, HIRAGANA, JAPANESE, MYANMAR};

/// `ICUTokenizerConfig.EMOJI_SEQUENCE_STATUS`.
pub const EMOJI_SEQUENCE_STATUS: i32 = 299;
/// `RuleBasedBreakIterator.WORD_NUMBER`.
pub const WORD_NUMBER: i32 = 100;
/// `RuleBasedBreakIterator.WORD_LETTER`.
pub const WORD_LETTER: i32 = 200;
/// `RuleBasedBreakIterator.WORD_KANA`.
pub const WORD_KANA: i32 = 300;
/// `RuleBasedBreakIterator.WORD_IDEO`.
pub const WORD_IDEO: i32 = 400;
/// `RuleBasedBreakIterator.WORD_NONE`.
pub const WORD_NONE: i32 = 0;

/// `ICUTokenizerConfig`: Lucene's extension point for per-script rules.
pub trait ICUTokenizerConfig: Send + Sync {
    /// `getBreakIterator(script)`: a fresh iterator (Java clones a
    /// prototype).
    fn get_break_iterator(&self, script: i32) -> RuleBasedBreakIterator;
    /// `getType(script, ruleStatus)`.
    fn get_type(&self, script: i32, rule_status: i32) -> &'static str;
    /// `combineCJ()`.
    fn combine_cj(&self) -> bool;
}

const DEFAULT_BRK: &[u8] = include_bytes!("../resources/Default.brk");
const MYANMAR_SYLLABLE_BRK: &[u8] = include_bytes!("../resources/MyanmarSyllable.brk");
const WORD_BRK: &[u8] = include_bytes!("../resources/word.brk");

fn rules(cell: &'static OnceLock<Arc<RbbiData>>, data: &[u8]) -> Arc<RbbiData> {
    cell.get_or_init(|| Arc::new(RbbiData::get(data).expect("vendored break rules load")))
        .clone()
}

/// The `cjkBreakIterator` prototype: ICU's root word rules.
pub fn cjk_rules() -> Arc<RbbiData> {
    static R: OnceLock<Arc<RbbiData>> = OnceLock::new();
    rules(&R, WORD_BRK)
}

/// The `defaultBreakIterator` prototype: `Default.brk`.
pub fn default_rules() -> Arc<RbbiData> {
    static R: OnceLock<Arc<RbbiData>> = OnceLock::new();
    rules(&R, DEFAULT_BRK)
}

/// The `myanmarSyllableIterator` prototype: `MyanmarSyllable.brk`.
pub fn myanmar_syllable_rules() -> Arc<RbbiData> {
    static R: OnceLock<Arc<RbbiData>> = OnceLock::new();
    rules(&R, MYANMAR_SYLLABLE_BRK)
}

/// `DefaultICUTokenizerConfig`.
#[derive(Debug, Clone, Copy)]
pub struct DefaultICUTokenizerConfig {
    cjk_as_words: bool,
    myanmar_as_words: bool,
}

impl DefaultICUTokenizerConfig {
    /// `WORD_IDEO`: `<IDEOGRAPHIC>`.
    pub const WORD_IDEO: &'static str = "<IDEOGRAPHIC>";
    /// `WORD_HIRAGANA`.
    pub const WORD_HIRAGANA: &'static str = "<HIRAGANA>";
    /// `WORD_KATAKANA`.
    pub const WORD_KATAKANA: &'static str = "<KATAKANA>";
    /// `WORD_HANGUL`.
    pub const WORD_HANGUL: &'static str = "<HANGUL>";
    /// `WORD_LETTER`: `<ALPHANUM>`.
    pub const WORD_LETTER: &'static str = "<ALPHANUM>";
    /// `WORD_NUMBER`: `<NUM>`.
    pub const WORD_NUMBER: &'static str = "<NUM>";
    /// `WORD_EMOJI`.
    pub const WORD_EMOJI: &'static str = "<EMOJI>";

    /// `new DefaultICUTokenizerConfig(cjkAsWords, myanmarAsWords)`.
    pub fn new(cjk_as_words: bool, myanmar_as_words: bool) -> Self {
        DefaultICUTokenizerConfig {
            cjk_as_words,
            myanmar_as_words,
        }
    }

    /// `getType(script, ruleStatus)`, shared with subclasses.
    pub fn default_type(script: i32, rule_status: i32) -> &'static str {
        match rule_status {
            WORD_IDEO => Self::WORD_IDEO,
            WORD_KANA => {
                if script == HIRAGANA {
                    Self::WORD_HIRAGANA
                } else {
                    Self::WORD_KATAKANA
                }
            }
            WORD_LETTER => {
                if script == HANGUL {
                    Self::WORD_HANGUL
                } else {
                    Self::WORD_LETTER
                }
            }
            WORD_NUMBER => Self::WORD_NUMBER,
            EMOJI_SEQUENCE_STATUS => Self::WORD_EMOJI,
            _ => "<OTHER>",
        }
    }
}

impl ICUTokenizerConfig for DefaultICUTokenizerConfig {
    fn get_break_iterator(&self, script: i32) -> RuleBasedBreakIterator {
        let data = match script {
            JAPANESE => cjk_rules(),
            MYANMAR if !self.myanmar_as_words => myanmar_syllable_rules(),
            _ => default_rules(),
        };
        RuleBasedBreakIterator::from_data(data)
    }

    fn get_type(&self, script: i32, rule_status: i32) -> &'static str {
        Self::default_type(script, rule_status)
    }

    fn combine_cj(&self) -> bool {
        self.cjk_as_words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_and_iterators() {
        let c = DefaultICUTokenizerConfig::new(true, false);
        assert!(c.combine_cj());
        assert_eq!(c.get_type(HIRAGANA, WORD_KANA), "<HIRAGANA>");
        assert_eq!(c.get_type(22, WORD_KANA), "<KATAKANA>");
        assert_eq!(c.get_type(HANGUL, WORD_LETTER), "<HANGUL>");
        assert_eq!(c.get_type(25, WORD_LETTER), "<ALPHANUM>");
        assert_eq!(c.get_type(25, WORD_NUMBER), "<NUM>");
        assert_eq!(c.get_type(25, WORD_IDEO), "<IDEOGRAPHIC>");
        assert_eq!(c.get_type(25, EMOJI_SEQUENCE_STATUS), "<EMOJI>");
        assert_eq!(c.get_type(25, 7), "<OTHER>");
        assert_eq!(c.get_type(25, WORD_NONE), "<OTHER>");
        for script in [JAPANESE, MYANMAR, 25] {
            let it = c.get_break_iterator(script);
            assert!(it.data().cat_count > 0);
        }
        let w = DefaultICUTokenizerConfig::new(false, true).get_break_iterator(MYANMAR);
        assert!(Arc::ptr_eq(w.data(), &default_rules()));
    }
}
