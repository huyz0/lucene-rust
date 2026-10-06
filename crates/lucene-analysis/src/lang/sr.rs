//! `org.apache.lucene.analysis.sr`: `SerbianAnalyzer` and the two Cyrillic
//! to Latin normalizations: `SerbianNormalizationFilter` ("bald" Latin:
//! diacritics dropped, `đ` -> `dj`) and `SerbianNormalizationRegularFilter`
//! (regular Latin with diacritics).

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{comment_set, mark_exclusions, snowball, std_lower_stop, CharStemmer, NormalizeFilter};

/// `SerbianAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/sr_stopwords.txt")));

/// One char's transliteration: one or two Latin units, or unchanged.
fn transliterate(s: &mut Vec<u16>, mut len: usize, map: fn(char) -> Option<&'static str>) -> usize {
    let mut i = 0;
    while i < len {
        let out = char::from_u32(u32::from(s[i])).and_then(map);
        if let Some(out) = out {
            let o: Vec<u16> = out.encode_utf16().collect();
            s[i] = o[0];
            if o.len() == 2 {
                s.insert(i + 1, o[1]);
                i += 1;
                len += 1;
            }
        }
        i += 1;
    }
    len
}

/// `SerbianNormalizationFilter`'s transform ("bald" Latin).
#[derive(Debug, Default, Clone, Copy)]
pub struct SerbianBaldLatin;

impl CharStemmer for SerbianBaldLatin {
    // Java: SerbianNormalizationFilter.incrementToken
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        transliterate(s, len, |c| {
            Some(match c {
                'а' => "a",
                'б' => "b",
                'в' => "v",
                'г' => "g",
                'д' => "d",
                'ђ' | 'đ' => "dj",
                'е' => "e",
                'ж' | 'з' | 'ž' => "z",
                'и' => "i",
                'ј' => "j",
                'к' => "k",
                'л' => "l",
                'љ' => "lj",
                'м' => "m",
                'н' => "n",
                'њ' => "nj",
                'о' => "o",
                'п' => "p",
                'р' => "r",
                'с' => "s",
                'т' => "t",
                'ћ' | 'ц' | 'ч' | 'č' | 'ć' => "c",
                'у' => "u",
                'ф' => "f",
                'х' => "h",
                'џ' => "dz",
                'ш' | 'š' => "s",
                _ => return None,
            })
        })
    }
}

/// `SerbianNormalizationRegularFilter`'s transform (regular Latin).
#[derive(Debug, Default, Clone, Copy)]
pub struct SerbianRegularLatin;

impl CharStemmer for SerbianRegularLatin {
    // Java: SerbianNormalizationRegularFilter.incrementToken
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        transliterate(s, len, |c| {
            Some(match c {
                'а' => "a",
                'б' => "b",
                'в' => "v",
                'г' => "g",
                'д' => "d",
                'ђ' => "đ",
                'е' => "e",
                'ж' => "ž",
                'з' => "z",
                'и' => "i",
                'ј' => "j",
                'к' => "k",
                'л' => "l",
                'љ' => "lj",
                'м' => "m",
                'н' => "n",
                'њ' => "nj",
                'о' => "o",
                'п' => "p",
                'р' => "r",
                'с' => "s",
                'т' => "t",
                'ћ' => "ć",
                'у' => "u",
                'ф' => "f",
                'х' => "h",
                'ц' => "c",
                'ч' => "č",
                'џ' => "dž",
                'ш' => "š",
                _ => return None,
            })
        })
    }
}

/// `SerbianNormalizationFilter`.
pub type SerbianNormalizationFilter<I> = NormalizeFilter<I, SerbianBaldLatin>;
/// `SerbianNormalizationRegularFilter`.
pub type SerbianNormalizationRegularFilter<I> = NormalizeFilter<I, SerbianRegularLatin>;

language_analyzer! {
    /// `SerbianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, Snowball `SerbianStemmer`, then
    /// [`SerbianNormalizationFilter`].
    SerbianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        SerbianNormalizationFilter::new(snowball(
            mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion),
            "Serbian",
        ))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
