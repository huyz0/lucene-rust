//! `org.apache.lucene.analysis.ckb`: `SoraniAnalyzer`, `SoraniNormalizer`
//! and `SoraniStemmer` (Central Kurdish).

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::java_character::{get_type, FORMAT};
use crate::util::stemmer_util::{delete, ends_with};
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::ar::is_tashkeel;
use super::{mark_exclusions, plain_set, CharStemmer, NormalizeFilter, StemFilter};

/// `SoraniAnalyzer.getDefaultStopSet()`: `stopwords.txt` read with
/// `WordlistLoader.getWordSet` (no comment syntax).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| plain_set(include_str!("stopwords/ckb_stopwords.txt")));

const FARSI_YEH: u16 = 0x6CC;
const KEHEH: u16 = 0x6A9;
const HEH: u16 = 0x647;
const AE: u16 = 0x6D5;
const RREH: u16 = 0x695;

/// `SoraniNormalizer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SoraniNormalizer;

impl CharStemmer for SoraniNormalizer {
    // Java: SoraniNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i = 0;
        while i < len {
            match s[i] {
                0x64A | 0x649 => s[i] = FARSI_YEH,
                0x643 => s[i] = KEHEH,
                0x200C => {
                    if i > 0 && s[i - 1] == HEH {
                        s[i - 1] = AE;
                    }
                    len = delete(s, i, len);
                    continue;
                }
                HEH => {
                    if i == len - 1 {
                        s[i] = AE;
                    }
                }
                0x629 => s[i] = AE,
                0x6BE => s[i] = HEH,
                0x631 => {
                    if i == 0 {
                        s[i] = RREH;
                    }
                }
                0x692 => s[i] = RREH,
                c if is_tashkeel(c) || get_type(u32::from(c)) == FORMAT => {
                    len = delete(s, i, len);
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
        len
    }
}

/// `SoraniStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SoraniStemmer;

impl CharStemmer for SoraniStemmer {
    // Java: SoraniStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let e = |s: &[u16], len: usize, x: &str| ends_with(s, len, x);
        // postposition
        if len > 5 && e(s, len, "دا") {
            len -= 2;
        } else if len > 4 && e(s, len, "نا") {
            len -= 1;
        } else if len > 6 && e(s, len, "ەوە") {
            len -= 3;
        }
        // possessive pronoun
        if len > 6 && (e(s, len, "مان") || e(s, len, "یان") || e(s, len, "تان")) {
            len -= 3;
        }
        // indefinite singular ezafe
        if len > 6 && e(s, len, "ێکی") {
            return len - 3;
        } else if len > 7 && e(s, len, "یەکی") {
            return len - 4;
        }
        let rules: [(usize, &str, usize); 16] = [
            (5, "ێک", 2),
            (6, "یەک", 3),
            (6, "ەکە", 3),
            (5, "کە", 2),
            (7, "ەکان", 4),
            (6, "کان", 3),
            (7, "یانی", 4),
            (6, "انی", 3),
            (6, "یان", 3),
            (5, "ان", 2),
            (7, "یانە", 4),
            (6, "انە", 3),
            (5, "ایە", 2),
            (5, "ەیە", 2),
            (4, "ە", 1),
            (4, "ی", 1),
        ];
        for (min, x, cut) in rules {
            if len > min && e(s, len, x) {
                return len - cut;
            }
        }
        len
    }
}

/// `SoraniNormalizationFilter`.
pub type SoraniNormalizationFilter<I> = NormalizeFilter<I, SoraniNormalizer>;
/// `SoraniStemFilter`.
pub type SoraniStemFilter<I> = StemFilter<I, SoraniStemmer>;

language_analyzer! {
    /// `SoraniAnalyzer`: `StandardTokenizer`, [`SoraniNormalizationFilter`],
    /// `LowerCaseFilter`, `DecimalDigitFilter`, `StopFilter`, exclusions,
    /// [`SoraniStemFilter`].
    SoraniAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = DecimalDigitFilter::new(LowerCaseFilter::new(SoraniNormalizationFilter::new(
            StandardTokenizer::new(),
        )));
        let r = StopFilter::new(r, Arc::clone(&s.stopwords));
        SoraniStemFilter::new(mark_exclusions(r, &s.exclusion))
    }
    normalize(s, input) {
        DecimalDigitFilter::new(LowerCaseFilter::new(SoraniNormalizationFilter::new(input)))
    }
}
