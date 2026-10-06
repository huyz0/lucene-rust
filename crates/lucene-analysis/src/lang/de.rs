//! `org.apache.lucene.analysis.de`: `GermanAnalyzer`, the light, minimal and
//! classic (`GermanStemmer`) stemmers and `GermanNormalizationFilter`.

use std::sync::{Arc, LazyLock};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::miscellaneous::SetKeywordMarkerFilter;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::stemmer_util::delete;
use crate::{AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::{copy_set, java_string_to_lower_case, snowball_set, CharStemmer, StemFilter};

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `GermanAnalyzer.getDefaultStopSet()` (`snowball/german_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/german_stop.txt")));

/// `GermanLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GermanLightStemmer;

fn st_ending(ch: u16) -> bool {
    matches!(
        ch,
        0x62 | 0x64 | 0x66 | 0x67 | 0x68 | 0x6B | 0x6C | 0x6D | 0x6E | 0x74
    )
}

impl GermanLightStemmer {
    fn step1(s: &[u16], len: usize) -> usize {
        if len > 5 && s[len - 3] == c('e') && s[len - 2] == c('r') && s[len - 1] == c('n') {
            return len - 3;
        }
        if len > 4 && s[len - 2] == c('e') && matches!(s[len - 1], 0x6D | 0x6E | 0x72 | 0x73) {
            return len - 2;
        }
        if len > 3 && s[len - 1] == c('e') {
            return len - 1;
        }
        if len > 3 && s[len - 1] == c('s') && st_ending(s[len - 2]) {
            return len - 1;
        }
        len
    }

    fn step2(s: &[u16], len: usize) -> usize {
        if len > 5 && s[len - 3] == c('e') && s[len - 2] == c('s') && s[len - 1] == c('t') {
            return len - 3;
        }
        if len > 4 && s[len - 2] == c('e') && (s[len - 1] == c('r') || s[len - 1] == c('n')) {
            return len - 2;
        }
        if len > 4 && s[len - 2] == c('s') && s[len - 1] == c('t') && st_ending(s[len - 3]) {
            return len - 2;
        }
        len
    }
}

impl CharStemmer for GermanLightStemmer {
    // Java: GermanLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE4 | 0xE0 | 0xE1 | 0xE2 => c('a'),
                0xF6 | 0xF2 | 0xF3 | 0xF4 => c('o'),
                0xEF | 0xEC | 0xED | 0xEE => c('i'),
                0xFC | 0xF9 | 0xFA | 0xFB => c('u'),
                o => o,
            };
        }
        let len = Self::step1(s, len);
        Self::step2(s, len)
    }
}

/// `GermanMinimalStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GermanMinimalStemmer;

impl CharStemmer for GermanMinimalStemmer {
    // Java: GermanMinimalStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        if len < 5 {
            return len;
        }
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE4 => c('a'),
                0xF6 => c('o'),
                0xFC => c('u'),
                o => o,
            };
        }
        if len > 6 && s[len - 3] == c('n') && s[len - 2] == c('e') && s[len - 1] == c('n') {
            return len - 3;
        }
        if len > 5 {
            let p = s[len - 2];
            match s[len - 1] {
                0x6E | 0x73 | 0x72 if p == c('e') => return len - 2,
                0x65 if p == c('s') => return len - 2,
                _ => {}
            }
        }
        match s[len - 1] {
            0x6E | 0x65 | 0x73 | 0x72 => len - 1,
            _ => len,
        }
    }
}

/// `GermanLightStemFilter`.
pub type GermanLightStemFilter<I> = StemFilter<I, GermanLightStemmer>;
/// `GermanMinimalStemFilter`.
pub type GermanMinimalStemFilter<I> = StemFilter<I, GermanMinimalStemmer>;

/// `GermanNormalizationFilter`: umlauts and `ß` folded, `ae`/`oe`/`ue`
/// collapsed (German2 snowball's normalisation).
pub struct GermanNormalizationFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> GermanNormalizationFilter<I> {
    /// `new GermanNormalizationFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        GermanNormalizationFilter {
            input,
            buf: Vec::new(),
        }
    }
}

/// `GermanNormalizationFilter.incrementToken`'s loop over one term.
pub(crate) fn german_normalize(buffer: &mut Vec<u16>) {
    const N: u8 = 0;
    const V: u8 = 1;
    const U: u8 = 2;
    let mut state = N;
    let mut length = buffer.len();
    let mut i = 0usize;
    while i < length {
        match buffer[i] {
            0x61 | 0x6F => state = U,
            0x75 => state = if state == N { U } else { V },
            0x65 => {
                if state == U {
                    length = delete(buffer, i, length);
                    buffer.truncate(length);
                    state = V;
                    continue;
                }
                state = V;
            }
            0x69 | 0x71 | 0x79 => state = V,
            0xE4 => {
                buffer[i] = c('a');
                state = V;
            }
            0xF6 => {
                buffer[i] = c('o');
                state = V;
            }
            0xFC => {
                buffer[i] = c('u');
                state = V;
            }
            0xDF => {
                buffer[i] = c('s');
                i += 1;
                buffer.insert(i, c('s'));
                length += 1;
                state = N;
            }
            _ => state = N,
        }
        i += 1;
    }
}

impl<I: TokenStream> TokenFilter for GermanNormalizationFilter<I> {
    crate::filter_input!();

    // Java: GermanNormalizationFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        crate::util::with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            german_normalize(b);
            true
        });
        Ok(true)
    }
}

/// `GermanStemmer` (Jörg Caumanns' algorithm).
#[derive(Debug, Default)]
pub struct GermanStemmer {
    subst_count: usize,
}

impl GermanStemmer {
    /// `stem(String)`.
    pub fn stem(&mut self, term: &[u16]) -> Vec<u16> {
        let term = java_string_to_lower_case(term);
        if !term
            .iter()
            .all(|&u| crate::java_character::is_letter(u32::from(u)))
        {
            return term;
        }
        let mut sb = term;
        self.substitute(&mut sb);
        self.strip(&mut sb);
        self.optimize(&mut sb);
        Self::resubstitute(&mut sb);
        Self::remove_particle_denotion(&mut sb);
        sb
    }

    fn ends(b: &[u16], suffix: &str) -> bool {
        super::super::util::stemmer_util::ends_with(b, b.len(), suffix)
    }

    // Java: GermanStemmer.strip
    fn strip(&self, b: &mut Vec<u16>) {
        let mut do_more = true;
        while do_more && b.len() > 3 {
            let n = b.len();
            if (n + self.subst_count > 5 && Self::ends(b, "nd"))
                || (n + self.subst_count > 4 && (Self::ends(b, "em") || Self::ends(b, "er")))
            {
                b.truncate(n - 2);
            } else if matches!(b[n - 1], 0x65 | 0x73 | 0x6E | 0x74) {
                b.truncate(n - 1);
            } else {
                do_more = false;
            }
        }
    }

    // Java: GermanStemmer.optimize
    fn optimize(&self, b: &mut Vec<u16>) {
        if b.len() > 5 && Self::ends(b, "erin*") {
            b.pop();
            self.strip(b);
        }
        if let Some(last) = b.last_mut() {
            if *last == c('z') {
                *last = c('x');
            }
        }
    }

    // Java: GermanStemmer.removeParticleDenotion
    fn remove_particle_denotion(b: &mut Vec<u16>) {
        if b.len() > 4 {
            let gege: Vec<u16> = "gege".encode_utf16().collect();
            for i in 0..b.len() - 3 {
                if b[i..i + 4] == gege[..] {
                    b.drain(i..i + 2);
                    return;
                }
            }
        }
    }

    // Java: GermanStemmer.substitute
    fn substitute(&mut self, b: &mut Vec<u16>) {
        self.subst_count = 0;
        let mut i = 0;
        while i < b.len() {
            if i > 0 && b[i] == b[i - 1] {
                b[i] = c('*');
            } else if b[i] == 0xE4 {
                b[i] = c('a');
            } else if b[i] == 0xF6 {
                b[i] = c('o');
            } else if b[i] == 0xFC {
                b[i] = c('u');
            } else if b[i] == 0xDF {
                b[i] = c('s');
                b.insert(i + 1, c('s'));
                self.subst_count += 1;
            }
            if i + 1 < b.len() {
                if i + 2 < b.len() && b[i] == c('s') && b[i + 1] == c('c') && b[i + 2] == c('h') {
                    b[i] = c('$');
                    b.drain(i + 1..i + 3);
                    self.subst_count += 2;
                } else {
                    let rep = match (b[i], b[i + 1]) {
                        (0x63, 0x68) => Some(0xA7), // ch -> §
                        (0x65, 0x69) => Some(c('%')),
                        (0x69, 0x65) => Some(c('&')),
                        (0x69, 0x67) => Some(c('#')),
                        (0x73, 0x74) => Some(c('!')),
                        _ => None,
                    };
                    if let Some(r) = rep {
                        b[i] = r;
                        b.remove(i + 1);
                        self.subst_count += 1;
                    }
                }
            }
            i += 1;
        }
    }

    // Java: GermanStemmer.resubstitute
    fn resubstitute(b: &mut Vec<u16>) {
        let mut i = 0;
        while i < b.len() {
            let pair: Option<[u16; 2]> = match b[i] {
                0x2A => {
                    b[i] = b[i - 1];
                    None
                }
                0x24 => {
                    b[i] = c('s');
                    b.splice(i + 1..i + 1, [c('c'), c('h')]);
                    None
                }
                0xA7 => Some([c('c'), c('h')]),
                0x25 => Some([c('e'), c('i')]),
                0x26 => Some([c('i'), c('e')]),
                0x23 => Some([c('i'), c('g')]),
                0x21 => Some([c('s'), c('t')]),
                _ => None,
            };
            if let Some([x, y]) = pair {
                b[i] = x;
                b.insert(i + 1, y);
            }
            i += 1;
        }
    }
}

/// `GermanStemFilter`: [`GermanStemmer`] on non-keyword terms.
pub struct GermanStemFilter<I> {
    input: I,
    stemmer: GermanStemmer,
    buf: Vec<u16>,
}

impl<I: TokenStream> GermanStemFilter<I> {
    /// `new GermanStemFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        GermanStemFilter {
            input,
            stemmer: GermanStemmer::default(),
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for GermanStemFilter<I> {
    crate::filter_input!();

    // Java: GermanStemFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.is_keyword() {
            let stemmer = &mut self.stemmer;
            crate::util::with_utf16_term(a, &mut self.buf, |b| {
                let s = stemmer.stem(b);
                if s != *b {
                    *b = s;
                    true
                } else {
                    false
                }
            });
        }
        Ok(true)
    }
}

/// `GermanAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
/// `SetKeywordMarkerFilter`, [`GermanNormalizationFilter`],
/// [`GermanLightStemFilter`].
#[derive(Debug, Clone)]
pub struct GermanAnalyzer {
    stopwords: Arc<CharArraySet>,
    exclusion: Arc<CharArraySet>,
}

impl Default for GermanAnalyzer {
    fn default() -> Self {
        Self::new(&DEFAULT_STOP_SET)
    }
}

impl GermanAnalyzer {
    /// `new GermanAnalyzer(CharArraySet stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        Self::with_exclusions(stopwords, &CharArraySet::empty())
    }

    /// `new GermanAnalyzer(stopwords, stemExclusionSet)`.
    pub fn with_exclusions(stopwords: &CharArraySet, exclusions: &CharArraySet) -> Self {
        GermanAnalyzer {
            stopwords: copy_set(stopwords),
            exclusion: copy_set(exclusions),
        }
    }
}

impl AnalyzerDefinition for GermanAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let r = StopFilter::new(
            LowerCaseFilter::new(StandardTokenizer::new()),
            Arc::clone(&self.stopwords),
        );
        let r = SetKeywordMarkerFilter::new(r, Arc::clone(&self.exclusion));
        let r = GermanLightStemFilter::new(GermanNormalizationFilter::new(r));
        Ok(TokenStreamComponents::new(r))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(GermanNormalizationFilter::new(LowerCaseFilter::new(input)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn st(s: &impl CharStemmer, w: &str) -> String {
        let mut b = u(w);
        let len = b.len();
        let n = s.stem(&mut b, len);
        String::from_utf16_lossy(&b[..n])
    }

    // Expected stems are Lucene 10.5.0's.
    #[test]
    fn stemmers_match_lucene() {
        assert_eq!(st(&GermanLightStemmer, "häusern"), "haus");
        assert_eq!(st(&GermanLightStemmer, "schönsten"), "schon");
        assert_eq!(st(&GermanMinimalStemmer, "häusern"), "hauser");
        assert_eq!(st(&GermanMinimalStemmer, "haus"), "haus");
        let mut g = GermanStemmer::default();
        let s = |g: &mut GermanStemmer, w: &str| String::from_utf16_lossy(&g.stem(&u(w)));
        assert_eq!(s(&mut g, "Häuser"), "hau");
        assert_eq!(s(&mut g, "abschließen"), "abschliess");
        assert_eq!(s(&mut g, "a1"), "a1");
        let mut b = u("Straße schön aue");
        german_normalize(&mut b);
        assert_eq!(String::from_utf16_lossy(&b), "Strasse schon aue");
    }
}
