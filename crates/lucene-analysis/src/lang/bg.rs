//! `org.apache.lucene.analysis.bg`: `BulgarianAnalyzer` and
//! `BulgarianStemmer` (Nakov's light stemmer).

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::{ends, ends_with};
use crate::CharArraySet;

use super::{comment_set, mark_exclusions, std_lower_stop, CharStemmer, StemFilter};

/// `BulgarianAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/bg_stopwords.txt")));

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `BulgarianStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BulgarianStemmer;

impl BulgarianStemmer {
    fn remove_article(s: &[u16], len: usize) -> usize {
        if len > 6 && ends!(s, len, "ият") {
            return len - 3;
        }
        if len > 5
            && ["ът", "то", "те", "та", "ия"]
                .iter()
                .any(|x| ends_with(s, len, x))
        {
            return len - 2;
        }
        if len > 4 && ends!(s, len, "ят") {
            return len - 2;
        }
        len
    }

    fn remove_plural(s: &mut [u16], len: usize) -> usize {
        if len > 6 {
            if ends!(s, len, "овци") || ends!(s, len, "ове") {
                return len - 3;
            }
            if ends!(s, len, "еве") {
                s[len - 3] = c('й');
                return len - 2;
            }
        }
        if len > 5 {
            if ends!(s, len, "ища") {
                return len - 3;
            }
            if ends!(s, len, "та") {
                return len - 2;
            }
            if ends!(s, len, "ци") {
                s[len - 2] = c('к');
                return len - 1;
            }
            if ends!(s, len, "зи") {
                s[len - 2] = c('г');
                return len - 1;
            }
            if s[len - 3] == c('е') && s[len - 1] == c('и') {
                s[len - 3] = c('я');
                return len - 1;
            }
        }
        if len > 4 {
            if ends!(s, len, "си") {
                s[len - 2] = c('х');
                return len - 1;
            }
            if ends!(s, len, "и") {
                return len - 1;
            }
        }
        len
    }
}

impl CharStemmer for BulgarianStemmer {
    // Java: BulgarianStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len < 4 {
            return len;
        }
        if len > 5 && ends!(s, len, "ища") {
            return len - 3;
        }
        len = Self::remove_article(s, len);
        len = Self::remove_plural(s, len);
        if len > 3 {
            if ends!(s, len, "я") {
                len -= 1;
            }
            if ends!(s, len, "а") || ends!(s, len, "о") || ends!(s, len, "е") {
                len -= 1;
            }
        }
        if len > 4 && ends!(s, len, "ен") {
            s[len - 2] = c('н');
            len -= 1;
        }
        if len > 5 && s[len - 2] == c('ъ') {
            s[len - 2] = s[len - 1];
            len -= 1;
        }
        len
    }
}

/// `BulgarianStemFilter`.
pub type BulgarianStemFilter<I> = StemFilter<I, BulgarianStemmer>;

language_analyzer! {
    /// `BulgarianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`BulgarianStemFilter`].
    BulgarianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        BulgarianStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
