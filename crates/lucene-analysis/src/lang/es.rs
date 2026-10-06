//! `org.apache.lucene.analysis.es`: `SpanishAnalyzer` and the light,
//! minimal and plural stemmers.

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{mark_exclusions, snowball_set, std_lower_stop, CharStemmer, StemFilter};

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `SpanishAnalyzer.getDefaultStopSet()` (`snowball/spanish_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/spanish_stop.txt")));

/// The accent folding the Spanish (and Italian) light stemmers share.
pub(crate) fn fold_vowels(s: &mut [u16]) {
    for ch in s {
        *ch = match *ch {
            0xE0 | 0xE1 | 0xE2 | 0xE4 => c('a'),
            0xF2 | 0xF3 | 0xF4 | 0xF6 => c('o'),
            0xE8 | 0xE9 | 0xEA | 0xEB => c('e'),
            0xF9 | 0xFA | 0xFB | 0xFC => c('u'),
            0xEC | 0xED | 0xEE | 0xEF => c('i'),
            o => o,
        };
    }
}

/// `SpanishLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpanishLightStemmer;

impl CharStemmer for SpanishLightStemmer {
    // Java: SpanishLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        if len < 5 {
            return len;
        }
        fold_vowels(&mut s[..len]);
        match s[len - 1] {
            0x6F | 0x61 | 0x65 => len - 1,
            0x73 => {
                if s[len - 2] == c('e') && s[len - 3] == c('s') && s[len - 4] == c('e') {
                    return len - 2;
                }
                if s[len - 2] == c('e') && s[len - 3] == c('c') {
                    s[len - 3] = c('z');
                    return len - 2;
                }
                if matches!(s[len - 2], 0x6F | 0x61 | 0x65) {
                    return len - 2;
                }
                len
            }
            _ => len,
        }
    }
}

/// `SpanishMinimalStemmer` (deprecated in Lucene for the plural stemmer).
#[derive(Debug, Default, Clone, Copy)]
pub struct SpanishMinimalStemmer;

impl CharStemmer for SpanishMinimalStemmer {
    // Java: SpanishMinimalStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        if len < 4 || s[len - 1] != c('s') {
            return len;
        }
        fold_vowels(&mut s[..len]);
        for ch in s[..len].iter_mut() {
            if *ch == 0xF1 {
                *ch = c('n');
            }
        }
        if s[len - 2] == c('a') || s[len - 2] == c('o') {
            return len - 1;
        }
        if s[len - 2] == c('e') {
            if s[len - 3] == c('s') && s[len - 4] == c('e') {
                return len - 2;
            }
            if s[len - 3] == c('c') {
                s[len - 3] = c('z');
            }
            return len - 2;
        }
        len - 1
    }
}

/// `SpanishPluralStemmer`'s invariant plurals (case-insensitive).
static INVARIANTS: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "abrebotellas",
            "abrecartas",
            "abrelatas",
            "afueras",
            "albatros",
            "albricias",
            "aledaños",
            "alexis",
            "alicates",
            "analisis",
            "andurriales",
            "antitesis",
            "añicos",
            "apendicitis",
            "apocalipsis",
            "arcoiris",
            "aries",
            "bilis",
            "boletus",
            "boris",
            "brindis",
            "cactus",
            "canutas",
            "caries",
            "cascanueces",
            "cascarrabias",
            "ciempies",
            "cifosis",
            "cortaplumas",
            "corpus",
            "cosmos",
            "cosquillas",
            "creces",
            "crisis",
            "cuatrocientas",
            "cuatrocientos",
            "cuelgacapas",
            "cuentacuentos",
            "cuentapasos",
            "cumpleaños",
            "doscientas",
            "doscientos",
            "dosis",
            "enseres",
            "entonces",
            "esponsales",
            "estatus",
            "exequias",
            "fauces",
            "forceps",
            "fotosintesis",
            "gafas",
            "gafotas",
            "gargaras",
            "gris",
            "honorarios",
            "ictus",
            "jueves",
            "lapsus",
            "lavacoches",
            "lavaplatos",
            "limpiabotas",
            "lunes",
            "maitines",
            "martes",
            "mondadientes",
            "novecientas",
            "novecientos",
            "nupcias",
            "ochocientas",
            "ochocientos",
            "pais",
            "paris",
            "parabrisas",
            "paracaidas",
            "parachoques",
            "paraguas",
            "pararrayos",
            "pisapapeles",
            "piscis",
            "portaaviones",
            "portamaletas",
            "portamantas",
            "quinientas",
            "quinientos",
            "quitamanchas",
            "recogepelotas",
            "rictus",
            "rompeolas",
            "sacacorchos",
            "sacapuntas",
            "saltamontes",
            "salvavidas",
            "seis",
            "seiscientas",
            "seiscientos",
            "setecientas",
            "setecientos",
            "sintesis",
            "tenis",
            "tifus",
            "trabalenguas",
            "vacaciones",
            "venus",
            "versus",
            "viacrucis",
            "virus",
            "viveres",
            "volandas",
        ],
        true,
    )
});

/// `SpanishPluralStemmer`'s special cases: plurals that lose two chars.
static SPECIAL_CASES: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "yoes",
            "noes",
            "sies",
            "clubes",
            "faralaes",
            "albalaes",
            "itemes",
            "albumes",
            "sandwiches",
            "relojes",
            "bojes",
            "contrarreloj",
            "carcajes",
        ],
        true,
    )
});

fn is_vowel(ch: u16) -> bool {
    matches!(ch, 0x61 | 0x65 | 0x69 | 0x6F | 0x75)
}

/// `SpanishPluralStemmer`: plural to singular (the analyzer's alternative to
/// the light stemmer).
#[derive(Debug, Default, Clone, Copy)]
pub struct SpanishPluralStemmer;

impl CharStemmer for SpanishPluralStemmer {
    // Java: SpanishPluralStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        if len < 4 {
            return len;
        }
        fold_vowels(&mut s[..len]);
        let word = String::from_utf16_lossy(&s[..len]);
        if INVARIANTS.contains(&word) {
            return len;
        }
        if SPECIAL_CASES.contains(&word) {
            return len - 2;
        }
        if s[len - 1] != c('s') {
            return len;
        }
        let (a, b, d) = (s[len - 4], s[len - 3], s[len - 2]);
        if !is_vowel(d) {
            return len - 1;
        }
        // Java's precedence: `q || (g && u && (i || e))`.
        if a == c('q') || (a == c('g') && b == c('u') && (d == c('i') || d == c('e'))) {
            return len - 1;
        }
        if is_vowel(a) && b == c('r') && d == c('e') {
            return len - 2;
        }
        if is_vowel(a) && [c('d'), c('l'), c('n'), c('x')].contains(&b) && d == c('e') {
            return len - 2;
        }
        if (b == c('y') || b == c('u')) && d == c('e') {
            return len - 2;
        }
        if [c('u'), c('l'), c('r'), c('t'), c('n')].contains(&a) && b == c('i') && d == c('e') {
            return len - 2;
        }
        if b == c('s') && d == c('e') {
            return len - 2;
        }
        if (is_vowel(b) || b == c('d')) && d == c('i') {
            s[len - 2] = c('y');
            return len - 1;
        }
        if d == c('e') && b == c('c') {
            s[len - 3] = c('z');
            return len - 2;
        }
        // `isVowel(s[len - 2])` holds here.
        len - 1
    }
}

/// `SpanishPluralStemFilter`.
pub type SpanishPluralStemFilter<I> = StemFilter<I, SpanishPluralStemmer>;

/// `SpanishLightStemFilter`.
pub type SpanishLightStemFilter<I> = StemFilter<I, SpanishLightStemmer>;
/// `SpanishMinimalStemFilter`.
pub type SpanishMinimalStemFilter<I> = StemFilter<I, SpanishMinimalStemmer>;

language_analyzer! {
    /// `SpanishAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`SpanishLightStemFilter`].
    SpanishAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        SpanishLightStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
