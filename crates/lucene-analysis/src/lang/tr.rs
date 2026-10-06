//! `org.apache.lucene.analysis.tr`: `TurkishAnalyzer`, `ApostropheFilter`
//! and `TurkishLowerCaseFilter` (dotted/dotless i).

use std::sync::{Arc, LazyLock};

use crate::java_character::{get_type, to_lower_case, NON_SPACING_MARK};
use crate::{CharArraySet, StandardTokenizer, StopFilter};

use super::{comment_set, mark_exclusions, snowball, CharStemmer, NormalizeFilter};

/// `TurkishAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/tr_stopwords.txt")));

/// `ApostropheFilter`'s transform: everything from the first `'` or U+2019
/// on is dropped.
#[derive(Debug, Default, Clone, Copy)]
pub struct Apostrophe;

impl CharStemmer for Apostrophe {
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        s[..len]
            .iter()
            .position(|&u| u == 0x27 || u == 0x2019)
            .unwrap_or(len)
    }
}

/// `ApostropheFilter`.
pub type ApostropheFilter<I> = NormalizeFilter<I, Apostrophe>;

/// `TurkishLowerCaseFilter`'s transform.
#[derive(Debug, Default, Clone, Copy)]
pub struct TurkishLowerCase;

const COMBINING_DOT_ABOVE: u32 = 0x307;

fn code_point_at(s: &[u16], i: usize, len: usize) -> u32 {
    crate::java_character::code_point_at(s, i, len)
}

fn is_before_dot(s: &[u16], pos: usize, len: usize) -> bool {
    let mut i = pos;
    while i < len {
        let ch = code_point_at(s, i, len);
        if get_type(ch) != NON_SPACING_MARK {
            return false;
        }
        if ch == COMBINING_DOT_ABOVE {
            return true;
        }
        i += if ch >= 0x10000 { 2 } else { 1 };
    }
    false
}

impl CharStemmer for TurkishLowerCase {
    // Java: TurkishLowerCaseFilter.incrementToken
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut i_or_after = false;
        let mut i = 0;
        while i < len {
            let ch = code_point_at(s, i, len);
            i_or_after = ch == 0x49 || (i_or_after && get_type(ch) == NON_SPACING_MARK);
            if i_or_after {
                if ch == COMBINING_DOT_ABOVE {
                    s.remove(i);
                    len -= 1;
                    continue;
                }
                if ch == 0x49 {
                    if is_before_dot(s, i + 1, len) {
                        s[i] = 0x69;
                    } else {
                        s[i] = 0x131;
                        i_or_after = false;
                    }
                    i += 1;
                    continue;
                }
            }
            let lower = to_lower_case(ch);
            if lower >= 0x10000 {
                let c = char::from_u32(lower).unwrap_or(char::REPLACEMENT_CHARACTER);
                let mut b = [0u16; 2];
                let enc = c.encode_utf16(&mut b);
                s[i..i + enc.len()].copy_from_slice(enc);
                i += enc.len();
            } else {
                s[i] = lower as u16;
                i += 1;
            }
        }
        len
    }
}

/// `TurkishLowerCaseFilter`.
pub type TurkishLowerCaseFilter<I> = NormalizeFilter<I, TurkishLowerCase>;

language_analyzer! {
    /// `TurkishAnalyzer`: `StandardTokenizer`, [`ApostropheFilter`],
    /// [`TurkishLowerCaseFilter`], `StopFilter`, exclusions, Snowball
    /// `TurkishStemmer`.
    TurkishAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = TurkishLowerCaseFilter::new(ApostropheFilter::new(StandardTokenizer::new()));
        let r = StopFilter::new(r, Arc::clone(&s.stopwords));
        snowball(mark_exclusions(r, &s.exclusion), "Turkish")
    }
    normalize(s, input) {
        TurkishLowerCaseFilter::new(input)
    }
}
