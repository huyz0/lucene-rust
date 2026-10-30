//! The Thai word iterator `ThaiTokenizer` runs here
//! ([`lucene_analysis::lang::th`]): ICU's root word rules (`word.brk`),
//! whose dictionary set covers Thai, so a Thai run is split by ICU's
//! `ThaiBreakEngine` over ICU's Thai dictionary -- what
//! `BreakIterator.getWordInstance(new ULocale("th"))` does in ICU4J. The
//! JDK's own Thai iterator uses a different word list, so some Thai runs
//! split differently from Lucene on the JDK (`docs/parity/analysis-lang.md`).

use lucene_analysis::lang::th::{self, ThaiWordBreaker};

use crate::icu4j::rbbi::{RuleBasedBreakIterator, DONE};
use crate::segmentation::config::cjk_rules;

/// ICU's word iterator over one sentence.
struct IcuThaiWordBreaker {
    iterator: RuleBasedBreakIterator,
}

impl ThaiWordBreaker for IcuThaiWordBreaker {
    fn set_text(&mut self, text: &[u16]) {
        self.iterator.set_text(text);
    }

    fn current(&self) -> i32 {
        self.iterator.current()
    }

    fn next(&mut self) -> i32 {
        match self.iterator.next() {
            DONE => th::DONE,
            b => b,
        }
    }
}

fn new_breaker() -> Box<dyn ThaiWordBreaker> {
    Box::new(IcuThaiWordBreaker {
        iterator: RuleBasedBreakIterator::from_data(cjk_rules()),
    })
}

/// Installs ICU's word iterator as `ThaiTokenizer`'s
/// ([`th::install_word_breaker`]); idempotent.
pub fn install_thai_word_breaker() {
    let _ = th::install_word_breaker(new_breaker);
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::token_stream::consume;
    use lucene_analysis::{Analyzer, StrReader, Tokenizer};

    fn tokens(text: &str) -> Vec<(String, i32, i32)> {
        install_thai_word_breaker();
        let mut t = th::thai_tokenizer().unwrap();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = Vec::new();
        consume(&mut t, |a| {
            out.push((a.term().to_string(), a.start_offset(), a.end_offset()))
        })
        .unwrap();
        out
    }

    /// Lucene's `TestThaiAnalyzer.testOffsets` and `testStopWords` inputs:
    /// the JDK and ICU agree on these.
    #[test]
    fn segments_thai_as_lucenes_tests_expect() {
        let t = tokens("การที่ได้ต้องแสดงว่างานดี");
        let terms: Vec<&str> = t.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(terms, ["การ", "ที่", "ได้", "ต้อง", "แสดง", "ว่า", "งาน", "ดี"]);
        let offsets: Vec<(i32, i32)> = t.iter().map(|x| (x.1, x.2)).collect();
        assert_eq!(
            offsets,
            [
                (0, 3),
                (3, 6),
                (6, 9),
                (9, 13),
                (13, 17),
                (17, 20),
                (20, 23),
                (23, 25)
            ]
        );
        install_thai_word_breaker();
        let a = Analyzer::new(th::ThaiAnalyzer::default());
        let mut ts = a.token_stream("f", "การที่ได้ต้องแสดงว่างานดี").unwrap();
        let mut got = Vec::new();
        consume(&mut ts, |a| got.push(a.term().to_string())).unwrap();
        assert_eq!(got, ["แสดง", "งาน", "ดี"]);
    }

    /// Non-Thai text breaks as the word rules break it, and tokens not
    /// starting with a letter or digit are dropped.
    #[test]
    fn mixed_text_and_punctuation() {
        let t = tokens("ภาษาไทย English 123, ok!");
        let terms: Vec<&str> = t.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(terms, ["ภาษา", "ไทย", "English", "123", "ok"]);
        assert!(tokens("").is_empty());
        assert!(tokens(" ,.! ").is_empty());
    }
}
