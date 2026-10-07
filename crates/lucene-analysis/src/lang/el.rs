//! `org.apache.lucene.analysis.el`: `GreekAnalyzer`, `GreekLowerCaseFilter`
//! (accents and final sigma folded) and `GreekStemmer` (Ntais' algorithm).
//!
//! `GreekStemmer`'s 23 rules and their exception sets are transliterated
//! mechanically from Lucene's source (each rule keeps Java's condition
//! structure, quirks included -- e.g. `rule0`'s `len > 6` guarding only the
//! first of its `||` alternatives).

// The GreekStemmer rules are transliterated mechanically and keep Java's
// statement shape (a trailing `return`, nested `if`s) for line-by-line
// comparison with Lucene's source.
#![allow(clippy::needless_return, clippy::collapsible_if)]

use std::sync::{Arc, LazyLock};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::java_character::to_lower_case;
use crate::token_stream::TokenStream;
use crate::util::stemmer_util::ends;
use crate::{AnalysisError, CharArraySet, StandardTokenizer, StopFilter};

use super::{comment_set, CharStemmer, NormalizeFilter, StemFilter};

/// `GreekAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/el_stopwords.txt")));

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `GreekLowerCaseFilter.lowerCase(int)`.
fn greek_lower(cp: u32) -> u32 {
    match cp {
        0x3C2 => 0x3C3,
        0x386 | 0x3AC => 0x3B1,
        0x388 | 0x3AD => 0x3B5,
        0x389 | 0x3AE => 0x3B7,
        0x38A | 0x3AA | 0x3AF | 0x3CA | 0x390 => 0x3B9,
        0x38E | 0x3AB | 0x3CD | 0x3CB | 0x3B0 => 0x3C5,
        0x38C | 0x3CC => 0x3BF,
        0x38F | 0x3CE => 0x3C9,
        0x3A2 => 0x3C2,
        _ => to_lower_case(cp),
    }
}

/// `GreekLowerCaseFilter`'s transform: [`greek_lower`] per code point.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreekLowerCase;

impl CharStemmer for GreekLowerCase {
    // Java: GreekLowerCaseFilter.incrementToken
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        let mut i = 0;
        while i < len {
            if s[i] < 0x80 {
                // Rust-only fast path: `greek_lower` of ASCII is ASCII's.
                s[i] = u16::from((s[i] as u8).to_ascii_lowercase());
                i += 1;
                continue;
            }
            let cp = crate::java_character::code_point_at(s, i, len);
            let lower = greek_lower(cp);
            let mut b = [0u16; 2];
            let enc = match char::from_u32(lower) {
                Some(ch) => ch.encode_utf16(&mut b),
                None => {
                    b[0] = lower as u16;
                    &mut b[..1]
                }
            };
            s[i..i + enc.len()].copy_from_slice(enc);
            i += enc.len();
        }
        len
    }
}

/// `GreekLowerCaseFilter`.
pub type GreekLowerCaseFilter<I> = NormalizeFilter<I, GreekLowerCase>;

/// `CharArraySet.contains(char[], 0, len)` of an exception set.
/// The word is decoded on the stack (the exception words are short); a
/// longer one takes a `String`.
fn contains(set: &CharArraySet, s: &[u16], len: usize) -> bool {
    let mut buf = [0u8; 64];
    let mut at = 0;
    for r in char::decode_utf16(s[..len].iter().copied()) {
        let c = r.unwrap_or(char::REPLACEMENT_CHARACTER);
        if at + c.len_utf8() > buf.len() {
            return set.contains(&String::from_utf16_lossy(&s[..len]));
        }
        at += c.encode_utf8(&mut buf[at..]).len();
    }
    std::str::from_utf8(&buf[..at]).is_ok_and(|w| set.contains(w))
}

/// `GreekStemmer.endsWithVowel`.
fn ends_with_vowel(s: &[u16], len: usize) -> bool {
    len != 0 && [c('α'), c('ε'), c('η'), c('ι'), c('ο'), c('υ'), c('ω')].contains(&s[len - 1])
}

/// `GreekStemmer.endsWithVowelNoY`.
fn ends_with_vowel_no_y(s: &[u16], len: usize) -> bool {
    len != 0 && [c('α'), c('ε'), c('η'), c('ι'), c('ο'), c('ω')].contains(&s[len - 1])
}

/// `GreekStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreekStemmer;

impl CharStemmer for GreekStemmer {
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        stem_greek(s, len)
    }
}

/// `GreekStemFilter`.
pub type GreekStemFilter<I> = StemFilter<I, GreekStemmer>;

/// `GreekAnalyzer`: `StandardTokenizer`, [`GreekLowerCaseFilter`],
/// `StopFilter`, [`GreekStemFilter`] (Lucene's has no stem exclusion set).
#[derive(Debug, Clone)]
pub struct GreekAnalyzer {
    stopwords: Arc<CharArraySet>,
}

impl Default for GreekAnalyzer {
    fn default() -> Self {
        Self::new(&DEFAULT_STOP_SET)
    }
}

impl GreekAnalyzer {
    /// `new GreekAnalyzer(CharArraySet stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        GreekAnalyzer {
            stopwords: super::copy_set(stopwords),
        }
    }
}

impl AnalyzerDefinition for GreekAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let r = GreekLowerCaseFilter::new(StandardTokenizer::new());
        Ok(TokenStreamComponents::new(GreekStemFilter::new(
            StopFilter::new(r, Arc::clone(&self.stopwords)),
        )))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(GreekLowerCaseFilter::new(input))
    }
}

// ---- generated from GreekStemmer.java (see the module docs)

static EXC4: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(["θ", "δ", "ελ", "γαλ", "ν", "π", "ιδ", "παρ"], false)
});

static EXC6: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "αλ",
            "αδ",
            "ενδ",
            "αμαν",
            "αμμοχαλ",
            "ηθ",
            "ανηθ",
            "αντιδ",
            "φυσ",
            "βρωμ",
            "γερ",
            "εξωδ",
            "καλπ",
            "καλλιν",
            "καταδ",
            "μουλ",
            "μπαν",
            "μπαγιατ",
            "μπολ",
            "μποσ",
            "νιτ",
            "ξικ",
            "συνομηλ",
            "πετσ",
            "πιτσ",
            "πικαντ",
            "πλιατσ",
            "ποστελν",
            "πρωτοδ",
            "σερτ",
            "συναδ",
            "τσαμ",
            "υποδ",
            "φιλον",
            "φυλοδ",
            "χασ",
        ],
        false,
    )
});

static EXC7: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "αναπ",
            "αποθ",
            "αποκ",
            "αποστ",
            "βουβ",
            "ξεθ",
            "ουλ",
            "πεθ",
            "πικρ",
            "ποτ",
            "σιχ",
            "χ",
        ],
        false,
    )
});

static EXC8A: LazyLock<CharArraySet> =
    LazyLock::new(|| CharArraySet::from_words(["τρ", "τσ"], false));

static EXC8B: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "βετερ",
            "βουλκ",
            "βραχμ",
            "γ",
            "δραδουμ",
            "θ",
            "καλπουζ",
            "καστελ",
            "κορμορ",
            "λαοπλ",
            "μωαμεθ",
            "μ",
            "μουσουλμ",
            "ν",
            "ουλ",
            "π",
            "πελεκ",
            "πλ",
            "πολισ",
            "πορτολ",
            "σαρακατσ",
            "σουλτ",
            "τσαρλατ",
            "ορφ",
            "τσιγγ",
            "τσοπ",
            "φωτοστεφ",
            "χ",
            "ψυχοπλ",
            "αγ",
            "ορφ",
            "γαλ",
            "γερ",
            "δεκ",
            "διπλ",
            "αμερικαν",
            "ουρ",
            "πιθ",
            "πουριτ",
            "σ",
            "ζωντ",
            "ικ",
            "καστ",
            "κοπ",
            "λιχ",
            "λουθηρ",
            "μαιντ",
            "μελ",
            "σιγ",
            "σπ",
            "στεγ",
            "τραγ",
            "τσαγ",
            "φ",
            "ερ",
            "αδαπ",
            "αθιγγ",
            "αμηχ",
            "ανικ",
            "ανοργ",
            "απηγ",
            "απιθ",
            "ατσιγγ",
            "βασ",
            "βασκ",
            "βαθυγαλ",
            "βιομηχ",
            "βραχυκ",
            "διατ",
            "διαφ",
            "ενοργ",
            "θυσ",
            "καπνοβιομηχ",
            "καταγαλ",
            "κλιβ",
            "κοιλαρφ",
            "λιβ",
            "μεγλοβιομηχ",
            "μικροβιομηχ",
            "νταβ",
            "ξηροκλιβ",
            "ολιγοδαμ",
            "ολογαλ",
            "πενταρφ",
            "περηφ",
            "περιτρ",
            "πλατ",
            "πολυδαπ",
            "πολυμηχ",
            "στεφ",
            "ταβ",
            "τετ",
            "υπερηφ",
            "υποκοπ",
            "χαμηλοδαπ",
            "ψηλοταβ",
        ],
        false,
    )
});

static EXC9: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "αβαρ",
            "βεν",
            "εναρ",
            "αβρ",
            "αδ",
            "αθ",
            "αν",
            "απλ",
            "βαρον",
            "ντρ",
            "σκ",
            "κοπ",
            "μπορ",
            "νιφ",
            "παγ",
            "παρακαλ",
            "σερπ",
            "σκελ",
            "συρφ",
            "τοκ",
            "υ",
            "δ",
            "εμ",
            "θαρρ",
            "θ",
        ],
        false,
    )
});

static EXC12A: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(["π", "απ", "συμπ", "ασυμπ", "ακαταπ", "αμεταμφ"], false)
});

static EXC12B: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "αλ",
            "αρ",
            "εκτελ",
            "ζ",
            "μ",
            "ξ",
            "παρακαλ",
            "αρ",
            "προ",
            "νισ",
        ],
        false,
    )
});

static EXC13: LazyLock<CharArraySet> =
    LazyLock::new(|| CharArraySet::from_words(["διαθ", "θ", "παρακαταθ", "προσθ", "συνθ"], false));

static EXC14: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "φαρμακ",
            "χαδ",
            "αγκ",
            "αναρρ",
            "βρομ",
            "εκλιπ",
            "λαμπιδ",
            "λεχ",
            "μ",
            "πατ",
            "ρ",
            "λ",
            "μεδ",
            "μεσαζ",
            "υποτειν",
            "αμ",
            "αιθ",
            "ανηκ",
            "δεσποζ",
            "ενδιαφερ",
            "δε",
            "δευτερευ",
            "καθαρευ",
            "πλε",
            "τσα",
        ],
        false,
    )
});

static EXC15A: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "αβαστ",
            "πολυφ",
            "αδηφ",
            "παμφ",
            "ρ",
            "ασπ",
            "αφ",
            "αμαλ",
            "αμαλλι",
            "ανυστ",
            "απερ",
            "ασπαρ",
            "αχαρ",
            "δερβεν",
            "δροσοπ",
            "ξεφ",
            "νεοπ",
            "νομοτ",
            "ολοπ",
            "ομοτ",
            "προστ",
            "προσωποπ",
            "συμπ",
            "συντ",
            "τ",
            "υποτ",
            "χαρ",
            "αειπ",
            "αιμοστ",
            "ανυπ",
            "αποτ",
            "αρτιπ",
            "διατ",
            "εν",
            "επιτ",
            "κροκαλοπ",
            "σιδηροπ",
            "λ",
            "ναυ",
            "ουλαμ",
            "ουρ",
            "π",
            "τρ",
            "μ",
        ],
        false,
    )
});

static EXC15B: LazyLock<CharArraySet> =
    LazyLock::new(|| CharArraySet::from_words(["ψοφ", "ναυλοχ"], false));

static EXC16: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        ["ν", "χερσον", "δωδεκαν", "ερημον", "μεγαλον", "επταν"],
        false,
    )
});

static EXC17: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        [
            "ασβ",
            "σβ",
            "αχρ",
            "χρ",
            "απλ",
            "αειμν",
            "δυσχρ",
            "ευχρ",
            "κοινοχρ",
            "παλιμψ",
        ],
        false,
    )
});

static EXC18: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(["ν", "ρ", "σπι", "στραβομουτσ", "κακομουτσ", "εξων"], false)
});

static EXC19: LazyLock<CharArraySet> = LazyLock::new(|| {
    CharArraySet::from_words(
        ["παρασουσ", "φ", "χ", "ωριοπλ", "αζ", "αλλοσουσ", "ασουσ"],
        false,
    )
});

// Java: GreekStemmer.stem
fn stem_greek(s: &mut [u16], len: usize) -> usize {
    let mut len = len;

    if len < 4 {
        return len;
    }
    // Rust-only fast path: every rule first needs the term to end in a
    // literal suffix, an exception word or a vowel, all of them lowercase
    // unaccented Greek (U+03B1..U+03C9), so a term ending in anything else
    // passes through all 23 unchanged.
    if !(0x3B1..=0x3C9).contains(&s[len - 1]) {
        return len;
    }

    let orig_len = len;

    len = rule0(s, len);
    len = rule1(s, len);
    len = rule2(s, len);
    len = rule3(s, len);
    len = rule4(s, len);
    len = rule5(s, len);
    len = rule6(s, len);
    len = rule7(s, len);
    len = rule8(s, len);
    len = rule9(s, len);
    len = rule10(s, len);
    len = rule11(s, len);
    len = rule12(s, len);
    len = rule13(s, len);
    len = rule14(s, len);
    len = rule15(s, len);
    len = rule16(s, len);
    len = rule17(s, len);
    len = rule18(s, len);
    len = rule19(s, len);
    len = rule20(s, len);

    if len == orig_len {
        len = rule21(s, len);
    }

    return rule22(s, len);
}

// Java: GreekStemmer.rule0
#[allow(unused_mut)]
fn rule0(s: &mut [u16], mut len: usize) -> usize {
    if len > 9 && (ends!(s, len, "καθεστωτοσ") || ends!(s, len, "καθεστωτων")) {
        return len - 4;
    }

    if len > 8 && (ends!(s, len, "γεγονοτοσ") || ends!(s, len, "γεγονοτων")) {
        return len - 4;
    }

    if len > 8 && ends!(s, len, "καθεστωτα") {
        return len - 3;
    }

    if len > 7 && (ends!(s, len, "τατογιου") || ends!(s, len, "τατογιων")) {
        return len - 4;
    }

    if len > 7 && ends!(s, len, "γεγονοτα") {
        return len - 3;
    }

    if len > 7 && ends!(s, len, "καθεστωσ") {
        return len - 2;
    }

    if len > 6 && (ends!(s, len, "σκαγιου"))
        || ends!(s, len, "σκαγιων")
        || ends!(s, len, "ολογιου")
        || ends!(s, len, "ολογιων")
        || ends!(s, len, "κρεατοσ")
        || ends!(s, len, "κρεατων")
        || ends!(s, len, "περατοσ")
        || ends!(s, len, "περατων")
        || ends!(s, len, "τερατοσ")
        || ends!(s, len, "τερατων")
    {
        return len - 4;
    }

    if len > 6 && ends!(s, len, "τατογια") {
        return len - 3;
    }

    if len > 6 && ends!(s, len, "γεγονοσ") {
        return len - 2;
    }

    if len > 5
        && (ends!(s, len, "φαγιου")
            || ends!(s, len, "φαγιων")
            || ends!(s, len, "σογιου")
            || ends!(s, len, "σογιων"))
    {
        return len - 4;
    }

    if len > 5
        && (ends!(s, len, "σκαγια")
            || ends!(s, len, "ολογια")
            || ends!(s, len, "κρεατα")
            || ends!(s, len, "περατα")
            || ends!(s, len, "τερατα"))
    {
        return len - 3;
    }

    if len > 4
        && (ends!(s, len, "φαγια")
            || ends!(s, len, "σογια")
            || ends!(s, len, "φωτοσ")
            || ends!(s, len, "φωτων"))
    {
        return len - 3;
    }

    if len > 4 && (ends!(s, len, "κρεασ") || ends!(s, len, "περασ") || ends!(s, len, "τερασ"))
    {
        return len - 2;
    }

    if len > 3 && ends!(s, len, "φωτα") {
        return len - 2;
    }

    if len > 2 && ends!(s, len, "φωσ") {
        return len - 1;
    }

    return len;
}

// Java: GreekStemmer.rule1
#[allow(unused_mut)]
fn rule1(s: &mut [u16], mut len: usize) -> usize {
    if len > 4 && (ends!(s, len, "αδεσ") || ends!(s, len, "αδων")) {
        len -= 4;
        if !(ends!(s, len, "οκ")
            || ends!(s, len, "μαμ")
            || ends!(s, len, "μαν")
            || ends!(s, len, "μπαμπ")
            || ends!(s, len, "πατερ")
            || ends!(s, len, "γιαγι")
            || ends!(s, len, "νταντ")
            || ends!(s, len, "κυρ")
            || ends!(s, len, "θει")
            || ends!(s, len, "πεθερ"))
        {
            len += 2;
        }
    }
    return len;
}

// Java: GreekStemmer.rule2
#[allow(unused_mut)]
fn rule2(s: &mut [u16], mut len: usize) -> usize {
    if len > 4 && (ends!(s, len, "εδεσ") || ends!(s, len, "εδων")) {
        len -= 4;
        if ends!(s, len, "οπ")
            || ends!(s, len, "ιπ")
            || ends!(s, len, "εμπ")
            || ends!(s, len, "υπ")
            || ends!(s, len, "γηπ")
            || ends!(s, len, "δαπ")
            || ends!(s, len, "κρασπ")
            || ends!(s, len, "μιλ")
        {
            len += 2;
        }
    }
    return len;
}

// Java: GreekStemmer.rule3
#[allow(unused_mut)]
fn rule3(s: &mut [u16], mut len: usize) -> usize {
    if len > 5 && (ends!(s, len, "ουδεσ") || ends!(s, len, "ουδων")) {
        len -= 5;
        if ends!(s, len, "αρκ")
            || ends!(s, len, "καλιακ")
            || ends!(s, len, "πεταλ")
            || ends!(s, len, "λιχ")
            || ends!(s, len, "πλεξ")
            || ends!(s, len, "σκ")
            || ends!(s, len, "σ")
            || ends!(s, len, "φλ")
            || ends!(s, len, "φρ")
            || ends!(s, len, "βελ")
            || ends!(s, len, "λουλ")
            || ends!(s, len, "χν")
            || ends!(s, len, "σπ")
            || ends!(s, len, "τραγ")
            || ends!(s, len, "φε")
        {
            len += 3;
        }
    }
    return len;
}

// Java: GreekStemmer.rule4
#[allow(unused_mut)]
fn rule4(s: &mut [u16], mut len: usize) -> usize {
    if len > 3 && (ends!(s, len, "εωσ") || ends!(s, len, "εων")) {
        len -= 3;
        if contains(&EXC4, s, len) {
            len += 1;
        }
    }
    return len;
}

// Java: GreekStemmer.rule5
#[allow(unused_mut)]
fn rule5(s: &mut [u16], mut len: usize) -> usize {
    if len > 2 && ends!(s, len, "ια") {
        len -= 2;
        if ends_with_vowel(s, len) {
            len += 1;
        }
    } else if len > 3 && (ends!(s, len, "ιου") || ends!(s, len, "ιων")) {
        len -= 3;
        if ends_with_vowel(s, len) {
            len += 1;
        }
    }
    return len;
}

// Java: GreekStemmer.rule6
#[allow(unused_mut)]
fn rule6(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;
    if len > 3 && (ends!(s, len, "ικα") || ends!(s, len, "ικο")) {
        len -= 3;
        removed = true;
    } else if len > 4 && (ends!(s, len, "ικου") || ends!(s, len, "ικων")) {
        len -= 4;
        removed = true;
    }

    if removed {
        if ends_with_vowel(s, len) || contains(&EXC6, s, len) {
            len += 2;
        }
    }
    return len;
}

// Java: GreekStemmer.rule7
#[allow(unused_mut)]
fn rule7(s: &mut [u16], mut len: usize) -> usize {
    if len == 5 && ends!(s, len, "αγαμε") {
        return len - 1;
    }

    if len > 7 && ends!(s, len, "ηθηκαμε") {
        len -= 7;
    } else if len > 6 && ends!(s, len, "ουσαμε") {
        len -= 6;
    } else if len > 5
        && (ends!(s, len, "αγαμε") || ends!(s, len, "ησαμε") || ends!(s, len, "ηκαμε"))
    {
        len -= 5;
    }

    if len > 3 && ends!(s, len, "αμε") {
        len -= 3;
        if contains(&EXC7, s, len) {
            len += 2;
        }
    }

    return len;
}

// Java: GreekStemmer.rule8
#[allow(unused_mut)]
fn rule8(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;

    if len > 8 && ends!(s, len, "ιουντανε") {
        len -= 8;
        removed = true;
    } else if len > 7 && ends!(s, len, "ιοντανε")
        || ends!(s, len, "ουντανε")
        || ends!(s, len, "ηθηκανε")
    {
        len -= 7;
        removed = true;
    } else if len > 6 && ends!(s, len, "ιοτανε")
        || ends!(s, len, "οντανε")
        || ends!(s, len, "ουσανε")
    {
        len -= 6;
        removed = true;
    } else if len > 5 && ends!(s, len, "αγανε")
        || ends!(s, len, "ησανε")
        || ends!(s, len, "οτανε")
        || ends!(s, len, "ηκανε")
    {
        len -= 5;
        removed = true;
    }

    if removed && contains(&EXC8A, s, len) {
        len += 4;
        s[len - 4] = c('α');
        s[len - 3] = c('γ');
        s[len - 2] = c('α');
        s[len - 1] = c('ν');
    }

    if len > 3 && ends!(s, len, "ανε") {
        len -= 3;
        if ends_with_vowel_no_y(s, len) || contains(&EXC8B, s, len) {
            len += 2;
        }
    }

    return len;
}

// Java: GreekStemmer.rule9
#[allow(unused_mut)]
fn rule9(s: &mut [u16], mut len: usize) -> usize {
    if len > 5 && ends!(s, len, "ησετε") {
        len -= 5;
    }

    if len > 3 && ends!(s, len, "ετε") {
        len -= 3;
        if contains(&EXC9, s, len)
            || ends_with_vowel_no_y(s, len)
            || ends!(s, len, "οδ")
            || ends!(s, len, "αιρ")
            || ends!(s, len, "φορ")
            || ends!(s, len, "ταθ")
            || ends!(s, len, "διαθ")
            || ends!(s, len, "σχ")
            || ends!(s, len, "ενδ")
            || ends!(s, len, "ευρ")
            || ends!(s, len, "τιθ")
            || ends!(s, len, "υπερθ")
            || ends!(s, len, "ραθ")
            || ends!(s, len, "ενθ")
            || ends!(s, len, "ροθ")
            || ends!(s, len, "σθ")
            || ends!(s, len, "πυρ")
            || ends!(s, len, "αιν")
            || ends!(s, len, "συνδ")
            || ends!(s, len, "συν")
            || ends!(s, len, "συνθ")
            || ends!(s, len, "χωρ")
            || ends!(s, len, "πον")
            || ends!(s, len, "βρ")
            || ends!(s, len, "καθ")
            || ends!(s, len, "ευθ")
            || ends!(s, len, "εκθ")
            || ends!(s, len, "νετ")
            || ends!(s, len, "ρον")
            || ends!(s, len, "αρκ")
            || ends!(s, len, "βαρ")
            || ends!(s, len, "βολ")
            || ends!(s, len, "ωφελ")
        {
            len += 2;
        }
    }

    return len;
}

// Java: GreekStemmer.rule10
#[allow(unused_mut)]
fn rule10(s: &mut [u16], mut len: usize) -> usize {
    if len > 5 && (ends!(s, len, "οντασ") || ends!(s, len, "ωντασ")) {
        len -= 5;
        if len == 3 && ends!(s, len, "αρχ") {
            len += 3;
            s[len - 3] = c('ο');
        }
        if ends!(s, len, "κρε") {
            len += 3;
            s[len - 3] = c('ω');
        }
    }

    return len;
}

// Java: GreekStemmer.rule11
#[allow(unused_mut)]
fn rule11(s: &mut [u16], mut len: usize) -> usize {
    if len > 6 && ends!(s, len, "ομαστε") {
        len -= 6;
        if len == 2 && ends!(s, len, "ον") {
            len += 5;
        }
    } else if len > 7 && ends!(s, len, "ιομαστε") {
        len -= 7;
        if len == 2 && ends!(s, len, "ον") {
            len += 5;
            s[len - 5] = c('ο');
            s[len - 4] = c('μ');
            s[len - 3] = c('α');
            s[len - 2] = c('σ');
            s[len - 1] = c('τ');
        }
    }
    return len;
}

// Java: GreekStemmer.rule12
#[allow(unused_mut)]
fn rule12(s: &mut [u16], mut len: usize) -> usize {
    if len > 5 && ends!(s, len, "ιεστε") {
        len -= 5;
        if contains(&EXC12A, s, len) {
            len += 4;
        }
    }

    if len > 4 && ends!(s, len, "εστε") {
        len -= 4;
        if contains(&EXC12B, s, len) {
            len += 3;
        }
    }

    return len;
}

// Java: GreekStemmer.rule13
#[allow(unused_mut)]
fn rule13(s: &mut [u16], mut len: usize) -> usize {
    if len > 6 && ends!(s, len, "ηθηκεσ") {
        len -= 6;
    } else if len > 5 && (ends!(s, len, "ηθηκα") || ends!(s, len, "ηθηκε")) {
        len -= 5;
    }

    let mut removed = false;

    if len > 4 && ends!(s, len, "ηκεσ") {
        len -= 4;
        removed = true;
    } else if len > 3 && (ends!(s, len, "ηκα") || ends!(s, len, "ηκε")) {
        len -= 3;
        removed = true;
    }

    if removed
        && (contains(&EXC13, s, len)
            || ends!(s, len, "σκωλ")
            || ends!(s, len, "σκουλ")
            || ends!(s, len, "ναρθ")
            || ends!(s, len, "σφ")
            || ends!(s, len, "οθ")
            || ends!(s, len, "πιθ"))
    {
        len += 2;
    }

    return len;
}

// Java: GreekStemmer.rule14
#[allow(unused_mut)]
fn rule14(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;

    if len > 5 && ends!(s, len, "ουσεσ") {
        len -= 5;
        removed = true;
    } else if len > 4 && (ends!(s, len, "ουσα") || ends!(s, len, "ουσε")) {
        len -= 4;
        removed = true;
    }

    if removed
        && (contains(&EXC14, s, len)
            || ends_with_vowel(s, len)
            || ends!(s, len, "ποδαρ")
            || ends!(s, len, "βλεπ")
            || ends!(s, len, "πανταχ")
            || ends!(s, len, "φρυδ")
            || ends!(s, len, "μαντιλ")
            || ends!(s, len, "μαλλ")
            || ends!(s, len, "κυματ")
            || ends!(s, len, "λαχ")
            || ends!(s, len, "ληγ")
            || ends!(s, len, "φαγ")
            || ends!(s, len, "ομ")
            || ends!(s, len, "πρωτ"))
    {
        len += 3;
    }

    return len;
}

// Java: GreekStemmer.rule15
#[allow(unused_mut)]
fn rule15(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;
    if len > 4 && ends!(s, len, "αγεσ") {
        len -= 4;
        removed = true;
    } else if len > 3 && (ends!(s, len, "αγα") || ends!(s, len, "αγε")) {
        len -= 3;
        removed = true;
    }

    if removed {
        let cond1 = contains(&EXC15A, s, len)
            || ends!(s, len, "οφ")
            || ends!(s, len, "πελ")
            || ends!(s, len, "χορτ")
            || ends!(s, len, "λλ")
            || ends!(s, len, "σφ")
            || ends!(s, len, "ρπ")
            || ends!(s, len, "φρ")
            || ends!(s, len, "πρ")
            || ends!(s, len, "λοχ")
            || ends!(s, len, "σμην");

        let cond2 = contains(&EXC15B, s, len) || ends!(s, len, "κολλ");

        if cond1 && !cond2 {
            len += 2;
        }
    }

    return len;
}

// Java: GreekStemmer.rule16
#[allow(unused_mut)]
fn rule16(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;
    if len > 4 && ends!(s, len, "ησου") {
        len -= 4;
        removed = true;
    } else if len > 3 && (ends!(s, len, "ησε") || ends!(s, len, "ησα")) {
        len -= 3;
        removed = true;
    }

    if removed && contains(&EXC16, s, len) {
        len += 2;
    }

    return len;
}

// Java: GreekStemmer.rule17
#[allow(unused_mut)]
fn rule17(s: &mut [u16], mut len: usize) -> usize {
    if len > 4 && ends!(s, len, "ηστε") {
        len -= 4;
        if contains(&EXC17, s, len) {
            len += 3;
        }
    }

    return len;
}

// Java: GreekStemmer.rule18
#[allow(unused_mut)]
fn rule18(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;

    if len > 6 && (ends!(s, len, "ησουνε") || ends!(s, len, "ηθουνε")) {
        len -= 6;
        removed = true;
    } else if len > 4 && ends!(s, len, "ουνε") {
        len -= 4;
        removed = true;
    }

    if removed && contains(&EXC18, s, len) {
        len += 3;
        s[len - 3] = c('ο');
        s[len - 2] = c('υ');
        s[len - 1] = c('ν');
    }
    return len;
}

// Java: GreekStemmer.rule19
#[allow(unused_mut)]
fn rule19(s: &mut [u16], mut len: usize) -> usize {
    let mut removed = false;

    if len > 6 && (ends!(s, len, "ησουμε") || ends!(s, len, "ηθουμε")) {
        len -= 6;
        removed = true;
    } else if len > 4 && ends!(s, len, "ουμε") {
        len -= 4;
        removed = true;
    }

    if removed && contains(&EXC19, s, len) {
        len += 3;
        s[len - 3] = c('ο');
        s[len - 2] = c('υ');
        s[len - 1] = c('μ');
    }
    return len;
}

// Java: GreekStemmer.rule20
#[allow(unused_mut)]
fn rule20(s: &mut [u16], mut len: usize) -> usize {
    if len > 5 && (ends!(s, len, "ματων") || ends!(s, len, "ματοσ")) {
        len -= 3;
    } else if len > 4 && ends!(s, len, "ματα") {
        len -= 2;
    }
    return len;
}

// Java: GreekStemmer.rule21
#[allow(unused_mut)]
fn rule21(s: &mut [u16], mut len: usize) -> usize {
    if len > 9 && ends!(s, len, "ιοντουσαν") {
        return len - 9;
    }

    if len > 8
        && (ends!(s, len, "ιομασταν")
            || ends!(s, len, "ιοσασταν")
            || ends!(s, len, "ιουμαστε")
            || ends!(s, len, "οντουσαν"))
    {
        return len - 8;
    }

    if len > 7
        && (ends!(s, len, "ιεμαστε")
            || ends!(s, len, "ιεσαστε")
            || ends!(s, len, "ιομουνα")
            || ends!(s, len, "ιοσαστε")
            || ends!(s, len, "ιοσουνα")
            || ends!(s, len, "ιουνται")
            || ends!(s, len, "ιουνταν")
            || ends!(s, len, "ηθηκατε")
            || ends!(s, len, "ομασταν")
            || ends!(s, len, "οσασταν")
            || ends!(s, len, "ουμαστε"))
    {
        return len - 7;
    }

    if len > 6
        && (ends!(s, len, "ιομουν")
            || ends!(s, len, "ιονταν")
            || ends!(s, len, "ιοσουν")
            || ends!(s, len, "ηθειτε")
            || ends!(s, len, "ηθηκαν")
            || ends!(s, len, "ομουνα")
            || ends!(s, len, "οσαστε")
            || ends!(s, len, "οσουνα")
            || ends!(s, len, "ουνται")
            || ends!(s, len, "ουνταν")
            || ends!(s, len, "ουσατε"))
    {
        return len - 6;
    }

    if len > 5
        && (ends!(s, len, "αγατε")
            || ends!(s, len, "ιεμαι")
            || ends!(s, len, "ιεται")
            || ends!(s, len, "ιεσαι")
            || ends!(s, len, "ιοταν")
            || ends!(s, len, "ιουμα")
            || ends!(s, len, "ηθεισ")
            || ends!(s, len, "ηθουν")
            || ends!(s, len, "ηκατε")
            || ends!(s, len, "ησατε")
            || ends!(s, len, "ησουν")
            || ends!(s, len, "ομουν")
            || ends!(s, len, "ονται")
            || ends!(s, len, "ονταν")
            || ends!(s, len, "οσουν")
            || ends!(s, len, "ουμαι")
            || ends!(s, len, "ουσαν"))
    {
        return len - 5;
    }

    if len > 4
        && (ends!(s, len, "αγαν")
            || ends!(s, len, "αμαι")
            || ends!(s, len, "ασαι")
            || ends!(s, len, "αται")
            || ends!(s, len, "ειτε")
            || ends!(s, len, "εσαι")
            || ends!(s, len, "εται")
            || ends!(s, len, "ηδεσ")
            || ends!(s, len, "ηδων")
            || ends!(s, len, "ηθει")
            || ends!(s, len, "ηκαν")
            || ends!(s, len, "ησαν")
            || ends!(s, len, "ησει")
            || ends!(s, len, "ησεσ")
            || ends!(s, len, "ομαι")
            || ends!(s, len, "οταν"))
    {
        return len - 4;
    }

    if len > 3
        && (ends!(s, len, "αει")
            || ends!(s, len, "εισ")
            || ends!(s, len, "ηθω")
            || ends!(s, len, "ησω")
            || ends!(s, len, "ουν")
            || ends!(s, len, "ουσ"))
    {
        return len - 3;
    }

    if len > 2
        && (ends!(s, len, "αν")
            || ends!(s, len, "ασ")
            || ends!(s, len, "αω")
            || ends!(s, len, "ει")
            || ends!(s, len, "εσ")
            || ends!(s, len, "ησ")
            || ends!(s, len, "οι")
            || ends!(s, len, "οσ")
            || ends!(s, len, "ου")
            || ends!(s, len, "υσ")
            || ends!(s, len, "ων"))
    {
        return len - 2;
    }

    if len > 1 && ends_with_vowel(s, len) {
        return len - 1;
    }

    return len;
}

// Java: GreekStemmer.rule22
#[allow(unused_mut)]
fn rule22(s: &mut [u16], mut len: usize) -> usize {
    if ends!(s, len, "εστερ") || ends!(s, len, "εστατ") {
        return len - 5;
    }

    if ends!(s, len, "οτερ")
        || ends!(s, len, "οτατ")
        || ends!(s, len, "υτερ")
        || ends!(s, len, "υτατ")
        || ends!(s, len, "ωτερ")
        || ends!(s, len, "ωτατ")
    {
        return len - 4;
    }

    return len;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exception_lookup_spills_long_words() {
        let long = "α".repeat(40);
        let set = CharArraySet::from_words([long.as_str(), "αβ"], false);
        let u = |w: &str| w.encode_utf16().collect::<Vec<u16>>();
        assert!(contains(&set, &u(&long), 40));
        assert!(contains(&set, &u("αβγ"), 2));
        assert!(!contains(&set, &u("αβγ"), 3));
    }
}
