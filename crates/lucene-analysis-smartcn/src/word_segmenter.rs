//! `org.apache.lucene.analysis.cn.smart.WordSegmenter`: a sentence's words
//! from the HHMM segmenter, their text restored and normalised
//! (`SegTokenFilter`) and their offsets made absolute.

use lucene_analysis::AnalysisError;

use crate::hhmm::seg_token_filter;
use crate::hhmm::{HHMMSegmenter, SegToken};
use crate::utility::word_type;

/// `WordSegmenter`.
#[derive(Debug, Clone, Default)]
pub struct WordSegmenter {
    hhmm_segmenter: HHMMSegmenter,
}

impl WordSegmenter {
    /// A segmenter over `hhmm_segmenter`'s dictionaries.
    pub fn new(hhmm_segmenter: HHMMSegmenter) -> Self {
        WordSegmenter { hhmm_segmenter }
    }

    /// `segmentSentence(sentence, startOffset)`: the tokens, without the
    /// sentence begin and end tokens, at offsets `start_offset` on.
    pub fn segment_sentence(
        &self,
        sentence: &[u16],
        start_offset: i32,
    ) -> Result<Vec<SegToken>, AnalysisError> {
        let mut seg_token_list = self.hhmm_segmenter.process(sentence)?;
        // tokens from sentence, excluding WordType.SENTENCE_BEGIN and
        // WordType.SENTENCE_END
        if seg_token_list.len() <= 2 {
            return Ok(Vec::new());
        }
        seg_token_list.pop();
        Ok(seg_token_list
            .into_iter()
            .skip(1)
            .map(|st| convert_seg_token(st, sentence, start_offset))
            .collect())
    }
}

/// `convertSegToken(st, sentence, sentenceStartOffset)`: a letter or digit
/// run's text from the sentence, `SegTokenFilter`, absolute offsets.
pub fn convert_seg_token(
    mut st: SegToken,
    sentence: &[u16],
    sentence_start_offset: i32,
) -> SegToken {
    if matches!(
        st.word_type,
        word_type::STRING
            | word_type::NUMBER
            | word_type::FULLWIDTH_NUMBER
            | word_type::FULLWIDTH_STRING
    ) {
        let (s, e) = (st.start_offset as usize, st.end_offset as usize);
        st.char_array = sentence[s..e].to_vec();
    }
    let mut st = seg_token_filter::filter(st);
    st.start_offset = st.start_offset.saturating_add(sentence_start_offset);
    st.end_offset = st.end_offset.saturating_add(sentence_start_offset);
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_with_absolute_offsets() {
        let s = WordSegmenter::new(HHMMSegmenter::default());
        let u: Vec<u16> = "中国ABC".encode_utf16().collect();
        let t = s.segment_sentence(&u, 10).unwrap();
        let got: Vec<(String, i32, i32)> = t
            .iter()
            .map(|t| {
                (
                    String::from_utf16_lossy(&t.char_array),
                    t.start_offset,
                    t.end_offset,
                )
            })
            .collect();
        assert_eq!(got, [("中国".into(), 10, 12), ("abc".into(), 12, 15)]);
        assert!(s.segment_sentence(&[], 0).unwrap().is_empty());
    }
}
