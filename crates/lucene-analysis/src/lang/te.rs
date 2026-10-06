//! `org.apache.lucene.analysis.te`: `TeluguAnalyzer`, `TeluguNormalizer` and
//! `TeluguStemmer`.

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::util::stemmer_util::{delete, ends_with};
use crate::{CharArraySet, StandardTokenizer, StopFilter};

use super::in_::IndicNormalizationFilter;
use super::{comment_set, mark_exclusions, CharStemmer, StemFilter};

/// `TeluguAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/te_stopwords.txt")));

/// `TeluguNormalizer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TeluguNormalizer;

impl CharStemmer for TeluguNormalizer {
    // Java: TeluguNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i = 0;
        while i < len {
            match s[i] {
                0x0C00 | 0x0C01 => s[i] = 0x0C02,
                0x0C03 | 0x200D | 0x200C => {
                    len = delete(s, i, len);
                    continue;
                }
                0x0C14 => s[i] = 0x0C13,
                0x0C10 => s[i] = 0x0C0F,
                0x0C06 => s[i] = 0x0C05,
                0x0C08 => s[i] = 0x0C07,
                0x0C0A => s[i] = 0x0C09,
                0x0C40 => s[i] = 0x0C3F,
                0x0C42 => s[i] = 0x0C41,
                0x0C47 => s[i] = 0x0C46,
                0x0C4B => s[i] = 0x0C4A,
                0x0C46 => {
                    if i + 1 < len && s[i + 1] == 0x0C56 {
                        s[i] = 0x0C48;
                        len = delete(s, i + 1, len);
                    }
                }
                0x0C12 => {
                    if i + 1 < len && s[i + 1] == 0x0C55 {
                        s[i] = 0x0C13;
                        len = delete(s, i + 1, len);
                    } else if i + 1 < len && s[i + 1] == 0x0C4C {
                        s[i] = 0x0C14;
                        len = delete(s, i + 1, len);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        len
    }
}

/// `TeluguStemmer`'s suffix groups (generated from Lucene's source).
const RULES: [(usize, &[&str], usize); 3] = [
    (5, &["ళ్ళు", "డ్లు"], 4),
    (
        3,
        &[
            "డు", "ము", "వు", "లు", "ని", "ను", "చే", "కై", "లో", "డు", "ది", "కి", "సు", "వై", "పై",
        ],
        2,
    ),
    (2, &["ి", "ీ", "ు", "ూ", "ె", "ే", "ొ", "ో", "ా"], 1),
];

/// `TeluguStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TeluguStemmer;

impl CharStemmer for TeluguStemmer {
    // Java: TeluguStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        for (min, xs, cut) in RULES {
            if len > min && xs.iter().any(|x| ends_with(s, len, x)) {
                return len - cut;
            }
        }
        len
    }
}

/// `TeluguNormalizationFilter` (keyword terms untouched).
pub type TeluguNormalizationFilter<I> = StemFilter<I, TeluguNormalizer>;
/// `TeluguStemFilter`.
pub type TeluguStemFilter<I> = StemFilter<I, TeluguStemmer>;

language_analyzer! {
    /// `TeluguAnalyzer`: `StandardTokenizer`, `DecimalDigitFilter`,
    /// exclusions, `IndicNormalizationFilter`, [`TeluguNormalizationFilter`],
    /// `StopFilter`, [`TeluguStemFilter`] (no lowercasing).
    TeluguAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = DecimalDigitFilter::new(StandardTokenizer::new());
        let r = TeluguNormalizationFilter::new(IndicNormalizationFilter::new(mark_exclusions(
            r,
            &s.exclusion,
        )));
        TeluguStemFilter::new(StopFilter::new(r, Arc::clone(&s.stopwords)))
    }
    normalize(s, input) {
        TeluguNormalizationFilter::new(IndicNormalizationFilter::new(DecimalDigitFilter::new(input)))
    }
}
