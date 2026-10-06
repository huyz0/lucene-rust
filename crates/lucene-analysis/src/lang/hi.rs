//! `org.apache.lucene.analysis.hi`: `HindiAnalyzer`, `HindiNormalizer` and
//! `HindiStemmer` (Ramanathan & Rao's light stemmer).

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::util::stemmer_util::{delete, ends_with};
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::in_::IndicNormalizationFilter;
use super::{comment_set, mark_exclusions, CharStemmer, StemFilter};

/// `HindiAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/hi_stopwords.txt")));

/// `HindiNormalizer`: nukta, virama, ZWJ/ZWNJ removed; chandrabindu,
/// dead `n` and the long vowels folded.
#[derive(Debug, Default, Clone, Copy)]
pub struct HindiNormalizer;

impl CharStemmer for HindiNormalizer {
    // Java: HindiNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i = 0;
        while i < len {
            match s[i] {
                0x0928 => {
                    if i + 1 < len && s[i + 1] == 0x094D {
                        s[i] = 0x0902;
                        len = delete(s, i + 1, len);
                    }
                }
                0x093C | 0x200D | 0x200C | 0x094D => {
                    len = delete(s, i, len);
                    continue;
                }
                0x0901 => s[i] = 0x0902,
                0x0929 => s[i] = 0x0928,
                0x0931 => s[i] = 0x0930,
                0x0934 => s[i] = 0x0933,
                0x0958 => s[i] = 0x0915,
                0x0959 => s[i] = 0x0916,
                0x095A => s[i] = 0x0917,
                0x095B => s[i] = 0x091C,
                0x095C => s[i] = 0x0921,
                0x095D => s[i] = 0x0922,
                0x095E => s[i] = 0x092B,
                0x095F => s[i] = 0x092F,
                0x0945 => s[i] = 0x0947,
                0x0946 => s[i] = 0x0947,
                0x0949 => s[i] = 0x094B,
                0x094A => s[i] = 0x094B,
                0x090D => s[i] = 0x090F,
                0x090E => s[i] = 0x090F,
                0x0911 => s[i] = 0x0913,
                0x0912 => s[i] = 0x0913,
                0x0972 => s[i] = 0x0905,
                0x0906 => s[i] = 0x0905,
                0x0908 => s[i] = 0x0907,
                0x090A => s[i] = 0x0909,
                0x0960 => s[i] = 0x090B,
                0x0961 => s[i] = 0x090C,
                0x0910 => s[i] = 0x090F,
                0x0914 => s[i] = 0x0913,
                0x0940 => s[i] = 0x093F,
                0x0942 => s[i] = 0x0941,
                0x0944 => s[i] = 0x0943,
                0x0963 => s[i] = 0x0962,
                0x0948 => s[i] = 0x0947,
                0x094C => s[i] = 0x094B,
                _ => {}
            }
            i += 1;
        }
        len
    }
}

const ENDINGS_5: [&str; 7] = ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"];
const ENDINGS_4: [&str; 18] = [
    "ाएगी",
    "ाएगा",
    "ाओगी",
    "ाओगे",
    "एंगी",
    "ेंगी",
    "एंगे",
    "ेंगे",
    "ूंगी",
    "ूंगा",
    "ातीं",
    "नाओं",
    "नाएं",
    "ताओं",
    "ताएं",
    "ियाँ",
    "ियों",
    "ियां",
];
const ENDINGS_3: [&str; 19] = [
    "ाकर",
    "ाइए",
    "ाईं",
    "ाया",
    "ेगी",
    "ेगा",
    "ोगी",
    "ोगे",
    "ाने",
    "ाना",
    "ाते",
    "ाती",
    "ाता",
    "तीं",
    "ाओं",
    "ाएं",
    "ुओं",
    "ुएं",
    "ुआं",
];
const ENDINGS_2: [&str; 16] = [
    "कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें",
];
const ENDINGS_1: [&str; 7] = ["ो", "े", "ू", "ु", "ी", "ि", "ा"];

/// `HindiStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct HindiStemmer;

impl CharStemmer for HindiStemmer {
    // Java: HindiStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        let any = |xs: &[&str]| xs.iter().any(|x| ends_with(s, len, x));
        for (min, xs, cut) in [
            (6, &ENDINGS_5[..], 5),
            (5, &ENDINGS_4[..], 4),
            (4, &ENDINGS_3[..], 3),
            (3, &ENDINGS_2[..], 2),
            (2, &ENDINGS_1[..], 1),
        ] {
            if len > min && any(xs) {
                return len - cut;
            }
        }
        len
    }
}

/// `HindiNormalizationFilter` (keyword terms untouched).
pub type HindiNormalizationFilter<I> = StemFilter<I, HindiNormalizer>;
/// `HindiStemFilter`.
pub type HindiStemFilter<I> = StemFilter<I, HindiStemmer>;

language_analyzer! {
    /// `HindiAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `DecimalDigitFilter`, exclusions, `IndicNormalizationFilter`,
    /// [`HindiNormalizationFilter`], `StopFilter`, [`HindiStemFilter`].
    HindiAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = DecimalDigitFilter::new(LowerCaseFilter::new(StandardTokenizer::new()));
        let r = HindiNormalizationFilter::new(IndicNormalizationFilter::new(mark_exclusions(
            r,
            &s.exclusion,
        )));
        HindiStemFilter::new(StopFilter::new(r, Arc::clone(&s.stopwords)))
    }
    normalize(s, input) {
        HindiNormalizationFilter::new(IndicNormalizationFilter::new(DecimalDigitFilter::new(
            LowerCaseFilter::new(input),
        )))
    }
}
