//! `org.apache.lucene.analysis.fa`: `PersianAnalyzer`, `PersianCharFilter`
//! (ZWNJ to space), `PersianNormalizer` and `PersianStemmer`.

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::reader::{CharFilter, CharReader};
use crate::util::stemmer_util::{delete, delete_n};
use crate::{AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::ar::ArabicNormalizationFilter;
use super::{comment_set, mark_exclusions, CharStemmer, NormalizeFilter, StemFilter};

/// `PersianAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/fa_stopwords.txt")));

const ZWNJ: u16 = 0x200C;

/// `PersianCharFilter`: every ZWNJ (U+200C) read as a space; offsets
/// unchanged.
pub struct PersianCharFilter {
    input: Box<dyn CharReader>,
}

impl PersianCharFilter {
    /// `new PersianCharFilter(Reader)`.
    pub fn new(input: Box<dyn CharReader>) -> Self {
        PersianCharFilter { input }
    }
}

impl CharFilter for PersianCharFilter {
    fn input(&self) -> &dyn CharReader {
        &*self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut *self.input
    }

    // Java: PersianCharFilter.read(char[], int, int)
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let n = self.input.read(buf)?;
        for c in &mut buf[..n] {
            if *c == ZWNJ {
                *c = u16::from(b' ');
            }
        }
        Ok(n)
    }

    fn correct(&self, current_off: i32) -> i32 {
        current_off
    }
}

/// `PersianNormalizer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PersianNormalizer;

impl CharStemmer for PersianNormalizer {
    // Java: PersianNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i = 0;
        while i < len {
            match s[i] {
                0x6CC | 0x6D2 => s[i] = 0x64A,
                0x6A9 => s[i] = 0x643,
                0x6C0 | 0x6C1 => s[i] = 0x647,
                0x654 => {
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

const SUFFIXES: [&[u16]; 8] = [
    &[0x627, 0x62A],
    &[0x627, 0x646],
    &[0x62A, 0x631, 0x64A, 0x646],
    &[0x62A, 0x631],
    &[0x64A, 0x64A],
    &[0x64A],
    &[0x647, 0x627],
    &[ZWNJ],
];

/// `PersianStemmer`: every listed suffix, in order, removed when two
/// chars would remain.
#[derive(Debug, Default, Clone, Copy)]
pub struct PersianStemmer;

impl CharStemmer for PersianStemmer {
    // Java: PersianStemmer.stemSuffix
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        for x in SUFFIXES {
            if len >= x.len() + 2 && s[len - x.len()..len] == *x {
                len = delete_n(s, len - x.len(), len, x.len());
            }
        }
        len
    }
}

/// `PersianNormalizationFilter`.
pub type PersianNormalizationFilter<I> = NormalizeFilter<I, PersianNormalizer>;
/// `PersianStemFilter`.
pub type PersianStemFilter<I> = StemFilter<I, PersianStemmer>;

language_analyzer! {
    /// `PersianAnalyzer`: [`PersianCharFilter`] (`initReader`),
    /// `StandardTokenizer`, `LowerCaseFilter`, `DecimalDigitFilter`,
    /// `ArabicNormalizationFilter`, [`PersianNormalizationFilter`],
    /// `StopFilter`, exclusions, [`PersianStemFilter`].
    PersianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = PersianNormalizationFilter::new(ArabicNormalizationFilter::new(
            DecimalDigitFilter::new(LowerCaseFilter::new(StandardTokenizer::new())),
        ));
        let r = StopFilter::new(r, Arc::clone(&s.stopwords));
        PersianStemFilter::new(mark_exclusions(r, &s.exclusion))
    }
    normalize(s, input) {
        PersianNormalizationFilter::new(ArabicNormalizationFilter::new(DecimalDigitFilter::new(
            LowerCaseFilter::new(input),
        )))
    }
    init_reader(reader) {
        PersianCharFilter::new(reader)
    }
}
