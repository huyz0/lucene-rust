//! `org.apache.lucene.analysis.cn.smart.HMMChineseTokenizer`: sentences
//! from the JDK's sentence iterator (`SegmentingTokenizerBase`), words from
//! [`WordSegmenter`]; every token's type is `word`.

use lucene_analysis::util::{Segmenter, SegmentingBase, SegmentingTokenizer};
use lucene_analysis::AnalysisError;

use crate::hhmm::SegToken;
use crate::word_segmenter::WordSegmenter;

/// The `HMMChineseTokenizer` half of `SegmentingTokenizerBase`.
#[derive(Debug, Clone, Default)]
pub struct HMMChineseSegmenter {
    word_segmenter: WordSegmenter,
    /// `tokens`: the current sentence's words, consumed from the front.
    tokens: std::vec::IntoIter<SegToken>,
}

impl Segmenter for HMMChineseSegmenter {
    fn set_next_sentence(
        &mut self,
        base: &SegmentingBase,
        sentence_start: usize,
        sentence_end: usize,
    ) -> Result<(), AnalysisError> {
        let sentence = &base.buffer()[sentence_start..sentence_end];
        let offset = base.offset().saturating_add(sentence_start as i32);
        self.tokens = self
            .word_segmenter
            .segment_sentence(sentence, offset)?
            .into_iter();
        Ok(())
    }

    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
        let Some(token) = self.tokens.next() else {
            return Ok(false);
        };
        let start = base.correct_offset(token.start_offset);
        let end = base.correct_offset(token.end_offset);
        let a = base.attributes_mut();
        a.clear_attributes();
        a.set_term_utf16(&token.char_array);
        a.set_offset(start, end)?;
        a.set_token_type("word");
        Ok(true)
    }

    fn reset(&mut self) {
        self.tokens = Vec::new().into_iter();
    }
}

/// `HMMChineseTokenizer`.
pub type HMMChineseTokenizer = SegmentingTokenizer<HMMChineseSegmenter>;

/// `new HMMChineseTokenizer()`.
pub fn hmm_chinese_tokenizer() -> HMMChineseTokenizer {
    SegmentingTokenizer::new(HMMChineseSegmenter::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::token_stream::consume;
    use lucene_analysis::{StrReader, Tokenizer};

    fn tokens(text: &str) -> Vec<(String, i32, i32, String)> {
        let mut t = hmm_chinese_tokenizer();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = Vec::new();
        consume(&mut t, |a| {
            out.push((
                a.term().to_string(),
                a.start_offset(),
                a.end_offset(),
                a.token_type().to_string(),
            ))
        })
        .unwrap();
        out
    }

    #[test]
    fn segments_chinese_and_normalises_the_rest() {
        let t = tokens("我是中国人。Hello ＷＯＲＬＤ１２3");
        let terms: Vec<&str> = t.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(
            terms,
            ["我", "是", "中国", "人", ",", "hello", "world", "123"]
        );
        assert!(t.iter().all(|x| x.3 == "word"));
        assert_eq!((t[2].1, t[2].2), (2, 4));
        assert!(tokens("").is_empty());
        assert!(tokens("   ").is_empty());
    }
}
