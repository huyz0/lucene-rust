//! `org.apache.lucene.analysis.ar`: `ArabicAnalyzer`, `ArabicNormalizer` and
//! the light-10 `ArabicStemmer`.

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::util::stemmer_util::{delete, delete_n};
use crate::{CharArraySet, LowerCaseFilter};

use super::{comment_set, mark_exclusions, CharStemmer, NormalizeFilter, StemFilter};

/// `ArabicAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/ar_stopwords.txt")));

const ALEF: u16 = 0x627;
const BEH: u16 = 0x628;
const TEH_MARBUTA: u16 = 0x629;
const TEH: u16 = 0x62A;
const FEH: u16 = 0x641;
const KAF: u16 = 0x643;
const LAM: u16 = 0x644;
const NOON: u16 = 0x646;
const HEH: u16 = 0x647;
const WAW: u16 = 0x648;
const YEH: u16 = 0x64A;

/// The harakat and tatweel `ArabicNormalizer` deletes (shared with Sorani).
pub(crate) fn is_tashkeel(c: u16) -> bool {
    matches!(c, 0x640 | 0x64B..=0x652)
}

/// `ArabicNormalizer`: alef variants to bare alef, dotless yeh to yeh, teh
/// marbuta to heh; harakat and tatweel removed.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArabicNormalizer;

impl CharStemmer for ArabicNormalizer {
    // Java: ArabicNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i = 0;
        while i < len {
            match s[i] {
                0x622 | 0x623 | 0x625 => s[i] = ALEF,
                0x649 => s[i] = YEH,
                TEH_MARBUTA => s[i] = HEH,
                c if is_tashkeel(c) => {
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

const PREFIXES: [&[u16]; 7] = [
    &[ALEF, LAM],
    &[WAW, ALEF, LAM],
    &[BEH, ALEF, LAM],
    &[KAF, ALEF, LAM],
    &[FEH, ALEF, LAM],
    &[LAM, LAM],
    &[WAW],
];

const SUFFIXES: [&[u16]; 10] = [
    &[HEH, ALEF],
    &[ALEF, NOON],
    &[ALEF, TEH],
    &[WAW, NOON],
    &[YEH, NOON],
    &[YEH, HEH],
    &[YEH, TEH_MARBUTA],
    &[HEH],
    &[TEH_MARBUTA],
    &[YEH],
];

/// `ArabicStemmer` (Larkey's light-10).
#[derive(Debug, Default, Clone, Copy)]
pub struct ArabicStemmer;

impl ArabicStemmer {
    /// `stemPrefix(char[], int)`: the first matching prefix removed.
    pub fn stem_prefix(s: &mut [u16], len: usize) -> usize {
        for p in PREFIXES {
            let ok = if p.len() == 1 && len < 4 {
                false
            } else {
                len >= p.len() + 2 && s[..p.len()] == *p
            };
            if ok {
                return delete_n(s, 0, len, p.len());
            }
        }
        len
    }

    /// `stemSuffix(char[], int)`: every matching suffix, in order, removed.
    pub fn stem_suffix(s: &mut [u16], mut len: usize) -> usize {
        for x in SUFFIXES {
            if len >= x.len() + 2 && s[len - x.len()..len] == *x {
                len = delete_n(s, len - x.len(), len, x.len());
            }
        }
        len
    }
}

impl CharStemmer for ArabicStemmer {
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        let len = Self::stem_prefix(s, len);
        Self::stem_suffix(s, len)
    }
}

/// `ArabicNormalizationFilter`.
pub type ArabicNormalizationFilter<I> = NormalizeFilter<I, ArabicNormalizer>;
/// `ArabicStemFilter`.
pub type ArabicStemFilter<I> = StemFilter<I, ArabicStemmer>;

language_analyzer! {
    /// `ArabicAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `DecimalDigitFilter`, `StopFilter`, [`ArabicNormalizationFilter`],
    /// exclusions, [`ArabicStemFilter`].
    ArabicAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = crate::StopFilter::new(
            DecimalDigitFilter::new(LowerCaseFilter::new(crate::StandardTokenizer::new())),
            Arc::clone(&s.stopwords),
        );
        ArabicStemFilter::new(mark_exclusions(ArabicNormalizationFilter::new(r), &s.exclusion))
    }
    normalize(s, input) {
        ArabicNormalizationFilter::new(DecimalDigitFilter::new(LowerCaseFilter::new(input)))
    }
}
