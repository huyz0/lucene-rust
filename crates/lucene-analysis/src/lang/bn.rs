//! `org.apache.lucene.analysis.bn`: `BengaliAnalyzer`, `BengaliNormalizer`
//! and `BengaliStemmer`.

use std::sync::{Arc, LazyLock};

use crate::core_analysis::DecimalDigitFilter;
use crate::util::stemmer_util::{delete, ends_with};
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::in_::IndicNormalizationFilter;
use super::{comment_set, mark_exclusions, CharStemmer, StemFilter};

/// `BengaliAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/bn_stopwords.txt")));

/// `BengaliNormalizer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BengaliNormalizer;

impl CharStemmer for BengaliNormalizer {
    // Java: BengaliNormalizer.normalize; `i` may step back below zero, as
    // in Java, before the loop's increment.
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i: isize = 0;
        while (i as usize) < len {
            let u = i as usize;
            match s[u] {
                0x0981 => {
                    len = delete(s, u, len);
                    i -= 1;
                }
                0x09C0 => s[u] = 0x09BF,
                0x09C2 => s[u] = 0x09C1,
                0x0995 => {
                    if u + 2 < len && s[u + 1] == 0x09CD && s[u + 2] == 0x09BF {
                        if u == 0 {
                            s[u] = 0x0996;
                            len = delete(s, u + 2, len);
                            len = delete(s, u + 1, len);
                        } else {
                            s[u + 1] = 0x0996;
                            len = delete(s, u + 2, len);
                        }
                    }
                }
                0x0999 => s[u] = 0x0982,
                0x09AF => {
                    if u == 2 && s[u - 1] == 0x09CD {
                        s[u - 1] = 0x09C7;
                        if u + 1 < len && s[u + 1] == 0x09BE {
                            len = delete(s, u + 1, len);
                        }
                        len = delete(s, u, len);
                        i -= 1;
                    } else if u >= 1 && s[u - 1] == 0x09CD {
                        len = delete(s, u, len);
                        len = delete(s, u - 1, len);
                        i -= 2;
                    }
                }
                0x09AC => {
                    if !((u >= 1 && s[u - 1] != 0x09CD) || u == 0) {
                        if u == 2 || (u >= 5 && s[u - 3] == 0x09CD) {
                            len = delete(s, u, len);
                            len = delete(s, u - 1, len);
                            i -= 2;
                        } else if u >= 2 {
                            s[u - 1] = s[u - 2];
                            len = delete(s, u, len);
                            i -= 1;
                        }
                    }
                }
                0x0983 => {
                    if u == len - 1 {
                        if len <= 3 {
                            s[u] = 0x09B9;
                        } else {
                            len = delete(s, u, len);
                        }
                    } else {
                        s[u] = s[u + 1];
                    }
                }
                0x09B6 | 0x09B7 => s[u] = 0x09B8,
                0x09A3 => s[u] = 0x09A8,
                0x09DC | 0x09DD => s[u] = 0x09B0,
                0x09CE => s[u] = 0x09A4,
                _ => {}
            }
            i += 1;
        }
        len
    }
}

/// `BengaliStemmer`'s suffix groups (generated from Lucene's source):
/// `(len >, suffixes, chars removed)`, first match wins.
const RULES: [(usize, &[&str], usize); 8] = [
    (
        9,
        &[
            "িয়াছিলাম",
            "িতেছিলাম",
            "িতেছিলেন",
            "ইতেছিলেন",
            "িয়াছিলেন",
            "ইয়াছিলেন",
        ],
        8,
    ),
    (
        8,
        &[
            "িতেছিলি",
            "িতেছিলে",
            "িয়াছিলা",
            "িয়াছিলে",
            "িতেছিলা",
            "িয়াছিলি",
            "য়েদেরকে",
        ],
        7,
    ),
    (
        7,
        &[
            "িতেছিস",
            "িতেছেন",
            "িয়াছিস",
            "িয়াছেন",
            "েছিলাম",
            "েছিলেন",
            "েদেরকে",
        ],
        6,
    ),
    (
        6,
        &[
            "িতেছি",
            "িতেছা",
            "িতেছে",
            "ছিলাম",
            "ছিলেন",
            "িয়াছি",
            "িয়াছা",
            "িয়াছে",
            "েছিলে",
            "েছিলা",
            "য়েদের",
            "দেরকে",
        ],
        5,
    ),
    (
        5,
        &[
            "িলাম",
            "িলেন",
            "িতাম",
            "িতেন",
            "িবেন",
            "ছিলি",
            "ছিলে",
            "ছিলা",
            "তেছে",
            "িতেছ",
            "খানা",
            "খানি",
            "গুলো",
            "গুলি",
            "য়েরা",
            "েদের",
        ],
        4,
    ),
    (
        4,
        &[
            "লাম",
            "িলি",
            "ইলি",
            "িলে",
            "ইলে",
            "লেন",
            "িলা",
            "ইলা",
            "তাম",
            "িতি",
            "ইতি",
            "িতে",
            "ইতে",
            "তেন",
            "িতা",
            "িবা",
            "ইবা",
            "িবি",
            "ইবি",
            "বেন",
            "িবে",
            "ইবে",
            "ছেন",
            "য়োন",
            "য়ের",
            "েরা",
            "দের",
        ],
        3,
    ),
    (
        3,
        &[
            "িস", "েন", "লি", "লে", "লা", "তি", "তে", "তা", "বি", "বে", "বা", "ছি", "ছা", "ছে", "ুন",
            "ুক", "টা", "টি", "নি", "ের", "তে", "রা", "কে",
        ],
        2,
    ),
    (2, &["ি", "ী", "া", "ো", "ে", "ব", "ত"], 1),
];

/// `BengaliStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BengaliStemmer;

impl CharStemmer for BengaliStemmer {
    // Java: BengaliStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        for (min, xs, cut) in RULES {
            if len > min && xs.iter().any(|x| ends_with(s, len, x)) {
                return len - cut;
            }
        }
        len
    }
}

/// `BengaliNormalizationFilter` (keyword terms untouched).
pub type BengaliNormalizationFilter<I> = StemFilter<I, BengaliNormalizer>;
/// `BengaliStemFilter`.
pub type BengaliStemFilter<I> = StemFilter<I, BengaliStemmer>;

language_analyzer! {
    /// `BengaliAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `DecimalDigitFilter`, exclusions, `IndicNormalizationFilter`,
    /// [`BengaliNormalizationFilter`], `StopFilter`, [`BengaliStemFilter`].
    BengaliAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = DecimalDigitFilter::new(LowerCaseFilter::new(StandardTokenizer::new()));
        let r = BengaliNormalizationFilter::new(IndicNormalizationFilter::new(mark_exclusions(
            r,
            &s.exclusion,
        )));
        BengaliStemFilter::new(StopFilter::new(r, Arc::clone(&s.stopwords)))
    }
    normalize(s, input) {
        BengaliNormalizationFilter::new(IndicNormalizationFilter::new(DecimalDigitFilter::new(
            LowerCaseFilter::new(input),
        )))
    }
}
