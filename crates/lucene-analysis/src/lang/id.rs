//! `org.apache.lucene.analysis.id`: `IndonesianAnalyzer` and
//! `IndonesianStemmer` (Tala's algorithm: particles, possessives, then
//! derivational prefixes and suffixes).

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::{delete_n, ends, ends_with, starts_with};
use crate::CharArraySet;

use super::{comment_set, mark_exclusions, std_lower_stop, CharStemmer, StemFilter};

/// `IndonesianAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/id_stopwords.txt")));

const REMOVED_KE: u32 = 1;
const REMOVED_PENG: u32 = 2;
const REMOVED_DI: u32 = 4;
const REMOVED_MENG: u32 = 8;
const REMOVED_TER: u32 = 16;
const REMOVED_BER: u32 = 32;
const REMOVED_PE: u32 = 64;

fn is_vowel(ch: u16) -> bool {
    matches!(ch, 0x61 | 0x65 | 0x69 | 0x6F | 0x75)
}

/// `IndonesianStemmer`; `stem_derivational` is the filter's flag (default
/// `true`).
#[derive(Debug, Clone, Copy)]
pub struct IndonesianStemmer {
    /// Also strip derivational prefixes and suffixes.
    pub stem_derivational: bool,
}

impl Default for IndonesianStemmer {
    fn default() -> Self {
        IndonesianStemmer {
            stem_derivational: true,
        }
    }
}

/// The stemmer's per-call state (`numSyllables`, `flags`).
struct Run {
    num_syllables: i32,
    flags: u32,
}

impl Run {
    fn remove_particle(&mut self, t: &[u16], len: usize) -> usize {
        if ["kah", "lah", "pun"].iter().any(|x| ends_with(t, len, x)) {
            self.num_syllables -= 1;
            return len - 3;
        }
        len
    }

    fn remove_possessive_pronoun(&mut self, t: &[u16], len: usize) -> usize {
        if ends!(t, len, "ku") || ends!(t, len, "mu") {
            self.num_syllables -= 1;
            return len - 2;
        }
        if ends!(t, len, "nya") {
            self.num_syllables -= 1;
            return len - 3;
        }
        len
    }

    /// One prefix rule: `flag` set, a syllable removed, `n` units deleted.
    fn strip(&mut self, t: &mut [u16], len: usize, flag: u32, n: usize) -> usize {
        self.flags |= flag;
        self.num_syllables -= 1;
        delete_n(t, 0, len, n)
    }

    // Java: IndonesianStemmer.removeFirstOrderPrefix
    fn remove_first_order_prefix(&mut self, t: &mut [u16], len: usize) -> usize {
        let sw = |t: &[u16], p: &str| starts_with(t, len, p);
        if sw(t, "meng") {
            return self.strip(t, len, REMOVED_MENG, 4);
        }
        if sw(t, "meny") && len > 4 && is_vowel(t[4]) {
            t[3] = u16::from(b's');
            return self.strip(t, len, REMOVED_MENG, 3);
        }
        if sw(t, "men") || sw(t, "mem") {
            return self.strip(t, len, REMOVED_MENG, 3);
        }
        if sw(t, "me") {
            return self.strip(t, len, REMOVED_MENG, 2);
        }
        if sw(t, "peng") {
            return self.strip(t, len, REMOVED_PENG, 4);
        }
        if sw(t, "peny") && len > 4 && is_vowel(t[4]) {
            t[3] = u16::from(b's');
            return self.strip(t, len, REMOVED_PENG, 3);
        }
        if sw(t, "peny") {
            return self.strip(t, len, REMOVED_PENG, 4);
        }
        if sw(t, "pen") && len > 3 && is_vowel(t[3]) {
            t[2] = u16::from(b't');
            return self.strip(t, len, REMOVED_PENG, 2);
        }
        if sw(t, "pen") || sw(t, "pem") {
            return self.strip(t, len, REMOVED_PENG, 3);
        }
        if sw(t, "di") {
            return self.strip(t, len, REMOVED_DI, 2);
        }
        if sw(t, "ter") {
            return self.strip(t, len, REMOVED_TER, 3);
        }
        if sw(t, "ke") {
            return self.strip(t, len, REMOVED_KE, 2);
        }
        len
    }

    // Java: IndonesianStemmer.removeSecondOrderPrefix
    fn remove_second_order_prefix(&mut self, t: &mut [u16], len: usize) -> usize {
        let sw = |t: &[u16], p: &str| starts_with(t, len, p);
        if sw(t, "ber") || (len == 7 && sw(t, "belajar")) {
            return self.strip(t, len, REMOVED_BER, 3);
        }
        if sw(t, "be")
            && len > 4
            && !is_vowel(t[2])
            && t[3] == u16::from(b'e')
            && t[4] == u16::from(b'r')
        {
            return self.strip(t, len, REMOVED_BER, 2);
        }
        if sw(t, "per") || (len == 7 && sw(t, "pelajar")) {
            return self.strip(t, len, 0, 3);
        }
        if sw(t, "pe") {
            return self.strip(t, len, REMOVED_PE, 2);
        }
        len
    }

    // Java: IndonesianStemmer.removeSuffix
    fn remove_suffix(&mut self, t: &[u16], len: usize) -> usize {
        let f = self.flags;
        if ends!(t, len, "kan") && f & (REMOVED_KE | REMOVED_PENG | REMOVED_PE) == 0 {
            self.num_syllables -= 1;
            return len - 3;
        }
        if ends!(t, len, "an") && f & (REMOVED_DI | REMOVED_MENG | REMOVED_TER) == 0 {
            self.num_syllables -= 1;
            return len - 2;
        }
        if ends!(t, len, "i")
            && !ends!(t, len, "si")
            && f & (REMOVED_BER | REMOVED_KE | REMOVED_PENG) == 0
        {
            self.num_syllables -= 1;
            return len - 1;
        }
        len
    }

    // Java: IndonesianStemmer.stemDerivational
    fn stem_derivational(&mut self, t: &mut [u16], mut len: usize) -> usize {
        let old = len;
        if self.num_syllables > 2 {
            len = self.remove_first_order_prefix(t, len);
        }
        if old != len {
            let old = len;
            if self.num_syllables > 2 {
                len = self.remove_suffix(t, len);
            }
            if old != len && self.num_syllables > 2 {
                len = self.remove_second_order_prefix(t, len);
            }
        } else {
            if self.num_syllables > 2 {
                len = self.remove_second_order_prefix(t, len);
            }
            if self.num_syllables > 2 {
                len = self.remove_suffix(t, len);
            }
        }
        len
    }
}

impl CharStemmer for IndonesianStemmer {
    // Java: IndonesianStemmer.stem
    fn stem(&self, t: &mut Vec<u16>, mut len: usize) -> usize {
        let mut r = Run {
            num_syllables: t[..len].iter().filter(|&&c| is_vowel(c)).count() as i32,
            flags: 0,
        };
        if r.num_syllables > 2 {
            len = r.remove_particle(t, len);
        }
        if r.num_syllables > 2 {
            len = r.remove_possessive_pronoun(t, len);
        }
        if self.stem_derivational {
            len = r.stem_derivational(t, len);
        }
        len
    }
}

/// `IndonesianStemFilter` (`stemDerivational` true; see
/// [`StemFilter::with_stemmer`] for the other setting).
pub type IndonesianStemFilter<I> = StemFilter<I, IndonesianStemmer>;

language_analyzer! {
    /// `IndonesianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`IndonesianStemFilter`].
    IndonesianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        IndonesianStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
