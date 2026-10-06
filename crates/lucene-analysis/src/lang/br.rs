//! `org.apache.lucene.analysis.br`: `BrazilianAnalyzer` and
//! `BrazilianStemmer` (a Snowball-style Portuguese stemmer over regions
//! R1, R2 and RV).
//!
//! Strings are UTF-16 units, as Java's; a Java `null` field is `None`.
//! Steps 1-5 are transliterated mechanically from Lucene's source, so their
//! long suffix lists keep Java's order (and step 1's discarded
//! `replaceSuffix(CT, "logias", "log")`, which leaves `CT` unchanged).

// The BrazilianStemmer rules are transliterated mechanically and keep Java's
// statement shape (a trailing `return`, nested `if`s) for line-by-line
// comparison with Lucene's source.
#![allow(clippy::needless_return, clippy::collapsible_if)]

use std::sync::{Arc, LazyLock};

use crate::java_character::is_letter;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

use super::{comment_set, java_string_to_lower_case, mark_exclusions, std_lower_stop};

/// `BrazilianAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/br_stopwords.txt")));

type Str = Option<Vec<u16>>;

fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

fn len(v: &Str) -> usize {
    v.as_ref().map_or(0, Vec::len)
}

fn is_vowel(c: u16) -> bool {
    matches!(c, 0x61 | 0x65 | 0x69 | 0x6F | 0x75)
}

/// `suffix(String value, String suffix)`.
fn suffix(value: &Str, suffix: &str) -> bool {
    match value {
        None => false,
        Some(v) => {
            let s = units(suffix);
            s.len() <= v.len() && v.ends_with(&s)
        }
    }
}

/// `removeSuffix(String, String)`.
fn remove_suffix(value: &Str, to_remove: &str) -> Str {
    if !suffix(value, to_remove) {
        return value.clone();
    }
    let v = value.as_ref().expect("suffix matched");
    Some(v[..v.len() - units(to_remove).len()].to_vec())
}

/// `replaceSuffix(String, String, String)`.
fn replace_suffix(value: &Str, to_replace: &str, change_to: &str) -> Str {
    let vvalue = remove_suffix(value, to_replace);
    if *value == vvalue {
        return value.clone();
    }
    vvalue.map(|mut v| {
        v.extend(units(change_to));
        v
    })
}

/// `suffixPreceded(String value, String suffix, String preceded)`.
fn suffix_preceded(value: &Str, s: &str, preceded: &str) -> bool {
    suffix(value, s) && suffix(&remove_suffix(value, s), preceded)
}

/// `getR1(String)`.
fn get_r1(value: &Str) -> Str {
    let v = value.as_ref()?;
    let i = v.len().checked_sub(1)?;
    let mut j = 0;
    while j < i && !is_vowel(v[j]) {
        j += 1;
    }
    if j >= i {
        return None;
    }
    while j < i && is_vowel(v[j]) {
        j += 1;
    }
    if j >= i {
        return None;
    }
    Some(v[j + 1..].to_vec())
}

/// `getRV(String)`.
fn get_rv(value: &Str) -> Str {
    let v = value.as_ref()?;
    let i = v.len() as isize - 1;
    if i > 0 && !is_vowel(v[1]) {
        let mut j = 2;
        while (j as isize) < i && !is_vowel(v[j]) {
            j += 1;
        }
        if (j as isize) < i {
            return Some(v[j + 1..].to_vec());
        }
    }
    if i > 1 && is_vowel(v[0]) && is_vowel(v[1]) {
        let mut j = 2;
        while (j as isize) < i && is_vowel(v[j]) {
            j += 1;
        }
        if (j as isize) < i {
            return Some(v[j + 1..].to_vec());
        }
    }
    if i > 2 {
        return Some(v[3..].to_vec());
    }
    None
}

/// `changeTerm(String)`: `toLowerCase(pt-BR)`, then accents folded.
fn change_term(value: &[u16]) -> Vec<u16> {
    java_string_to_lower_case(value)
        .into_iter()
        .map(|c| match c {
            0xE1 | 0xE2 | 0xE3 => 0x61,
            0xE9 | 0xEA => 0x65,
            0xED => 0x69,
            0xF3 | 0xF4 | 0xF5 => 0x6F,
            0xFA | 0xFC => 0x75,
            0xE7 => 0x63,
            0xF1 => 0x6E,
            o => o,
        })
        .collect()
}

/// `BrazilianStemmer`.
#[derive(Debug, Default)]
pub struct BrazilianStemmer {
    ct: Str,
    r1: Str,
    r2: Str,
    rv: Str,
}

impl BrazilianStemmer {
    /// `stem(String)`: `None` for a term too short or too long to index
    /// (Java's `null`).
    pub fn stem(&mut self, term: &[u16]) -> Option<Vec<u16>> {
        self.create_ct(term);
        let ct = self.ct.as_ref().expect("createCT sets CT");
        if !(ct.len() < 30 && ct.len() > 2) {
            return None;
        }
        if !ct.iter().all(|&u| is_letter(u32::from(u))) {
            return self.ct.clone();
        }
        self.r1 = get_r1(&self.ct);
        self.r2 = get_r1(&self.r1);
        self.rv = get_rv(&self.ct);
        let mut altered = self.step1();
        if !altered {
            altered = self.step2();
        }
        if altered {
            self.step3();
        } else {
            self.step4();
        }
        self.step5();
        self.ct.clone()
    }

    // Java: BrazilianStemmer.createCT
    fn create_ct(&mut self, term: &[u16]) {
        let mut ct = change_term(term);
        if ct.len() >= 2 {
            if [0x22, 0x27, 0x2D, 0x2C, 0x3B, 0x2E, 0x3F, 0x21].contains(&ct[0]) {
                ct.remove(0);
            }
            if ct.len() >= 2
                && [0x2D, 0x2C, 0x3B, 0x2E, 0x3F, 0x21, 0x27, 0x22].contains(&ct[ct.len() - 1])
            {
                ct.pop();
            }
        }
        self.ct = Some(ct);
    }

    // Java: BrazilianStemmer.step1
    fn step1(&mut self) -> bool {
        if self.ct.is_none() {
            return false;
        }

        if suffix(&self.ct, "uciones") && suffix(&self.r2, "uciones") {
            self.ct = replace_suffix(&self.ct, "uciones", "u");
            return true;
        }

        if len(&self.ct) >= 6 {
            if suffix(&self.ct, "imentos") && suffix(&self.r2, "imentos") {
                self.ct = remove_suffix(&self.ct, "imentos");
                return true;
            }
            if suffix(&self.ct, "amentos") && suffix(&self.r2, "amentos") {
                self.ct = remove_suffix(&self.ct, "amentos");
                return true;
            }
            if suffix(&self.ct, "adores") && suffix(&self.r2, "adores") {
                self.ct = remove_suffix(&self.ct, "adores");
                return true;
            }
            if suffix(&self.ct, "adoras") && suffix(&self.r2, "adoras") {
                self.ct = remove_suffix(&self.ct, "adoras");
                return true;
            }
            if suffix(&self.ct, "logias") && suffix(&self.r2, "logias") {
                // Java discards this result.
                let _ = replace_suffix(&self.ct, "logias", "log");
                return true;
            }
            if suffix(&self.ct, "encias") && suffix(&self.r2, "encias") {
                self.ct = replace_suffix(&self.ct, "encias", "ente");
                return true;
            }
            if suffix(&self.ct, "amente") && suffix(&self.r1, "amente") {
                self.ct = remove_suffix(&self.ct, "amente");
                return true;
            }
            if suffix(&self.ct, "idades") && suffix(&self.r2, "idades") {
                self.ct = remove_suffix(&self.ct, "idades");
                return true;
            }
        }

        if len(&self.ct) >= 5 {
            if suffix(&self.ct, "acoes") && suffix(&self.r2, "acoes") {
                self.ct = remove_suffix(&self.ct, "acoes");
                return true;
            }
            if suffix(&self.ct, "imento") && suffix(&self.r2, "imento") {
                self.ct = remove_suffix(&self.ct, "imento");
                return true;
            }
            if suffix(&self.ct, "amento") && suffix(&self.r2, "amento") {
                self.ct = remove_suffix(&self.ct, "amento");
                return true;
            }
            if suffix(&self.ct, "adora") && suffix(&self.r2, "adora") {
                self.ct = remove_suffix(&self.ct, "adora");
                return true;
            }
            if suffix(&self.ct, "ismos") && suffix(&self.r2, "ismos") {
                self.ct = remove_suffix(&self.ct, "ismos");
                return true;
            }
            if suffix(&self.ct, "istas") && suffix(&self.r2, "istas") {
                self.ct = remove_suffix(&self.ct, "istas");
                return true;
            }
            if suffix(&self.ct, "logia") && suffix(&self.r2, "logia") {
                self.ct = replace_suffix(&self.ct, "logia", "log");
                return true;
            }
            if suffix(&self.ct, "ucion") && suffix(&self.r2, "ucion") {
                self.ct = replace_suffix(&self.ct, "ucion", "u");
                return true;
            }
            if suffix(&self.ct, "encia") && suffix(&self.r2, "encia") {
                self.ct = replace_suffix(&self.ct, "encia", "ente");
                return true;
            }
            if suffix(&self.ct, "mente") && suffix(&self.r2, "mente") {
                self.ct = remove_suffix(&self.ct, "mente");
                return true;
            }
            if suffix(&self.ct, "idade") && suffix(&self.r2, "idade") {
                self.ct = remove_suffix(&self.ct, "idade");
                return true;
            }
        }

        if len(&self.ct) >= 4 {
            if suffix(&self.ct, "acao") && suffix(&self.r2, "acao") {
                self.ct = remove_suffix(&self.ct, "acao");
                return true;
            }
            if suffix(&self.ct, "ezas") && suffix(&self.r2, "ezas") {
                self.ct = remove_suffix(&self.ct, "ezas");
                return true;
            }
            if suffix(&self.ct, "icos") && suffix(&self.r2, "icos") {
                self.ct = remove_suffix(&self.ct, "icos");
                return true;
            }
            if suffix(&self.ct, "icas") && suffix(&self.r2, "icas") {
                self.ct = remove_suffix(&self.ct, "icas");
                return true;
            }
            if suffix(&self.ct, "ismo") && suffix(&self.r2, "ismo") {
                self.ct = remove_suffix(&self.ct, "ismo");
                return true;
            }
            if suffix(&self.ct, "avel") && suffix(&self.r2, "avel") {
                self.ct = remove_suffix(&self.ct, "avel");
                return true;
            }
            if suffix(&self.ct, "ivel") && suffix(&self.r2, "ivel") {
                self.ct = remove_suffix(&self.ct, "ivel");
                return true;
            }
            if suffix(&self.ct, "ista") && suffix(&self.r2, "ista") {
                self.ct = remove_suffix(&self.ct, "ista");
                return true;
            }
            if suffix(&self.ct, "osos") && suffix(&self.r2, "osos") {
                self.ct = remove_suffix(&self.ct, "osos");
                return true;
            }
            if suffix(&self.ct, "osas") && suffix(&self.r2, "osas") {
                self.ct = remove_suffix(&self.ct, "osas");
                return true;
            }
            if suffix(&self.ct, "ador") && suffix(&self.r2, "ador") {
                self.ct = remove_suffix(&self.ct, "ador");
                return true;
            }
            if suffix(&self.ct, "ivas") && suffix(&self.r2, "ivas") {
                self.ct = remove_suffix(&self.ct, "ivas");
                return true;
            }
            if suffix(&self.ct, "ivos") && suffix(&self.r2, "ivos") {
                self.ct = remove_suffix(&self.ct, "ivos");
                return true;
            }
            if suffix(&self.ct, "iras")
                && suffix(&self.rv, "iras")
                && suffix_preceded(&self.ct, "iras", "e")
            {
                self.ct = replace_suffix(&self.ct, "iras", "ir");
                return true;
            }
        }

        if len(&self.ct) >= 3 {
            if suffix(&self.ct, "eza") && suffix(&self.r2, "eza") {
                self.ct = remove_suffix(&self.ct, "eza");
                return true;
            }
            if suffix(&self.ct, "ico") && suffix(&self.r2, "ico") {
                self.ct = remove_suffix(&self.ct, "ico");
                return true;
            }
            if suffix(&self.ct, "ica") && suffix(&self.r2, "ica") {
                self.ct = remove_suffix(&self.ct, "ica");
                return true;
            }
            if suffix(&self.ct, "oso") && suffix(&self.r2, "oso") {
                self.ct = remove_suffix(&self.ct, "oso");
                return true;
            }
            if suffix(&self.ct, "osa") && suffix(&self.r2, "osa") {
                self.ct = remove_suffix(&self.ct, "osa");
                return true;
            }
            if suffix(&self.ct, "iva") && suffix(&self.r2, "iva") {
                self.ct = remove_suffix(&self.ct, "iva");
                return true;
            }
            if suffix(&self.ct, "ivo") && suffix(&self.r2, "ivo") {
                self.ct = remove_suffix(&self.ct, "ivo");
                return true;
            }
            if suffix(&self.ct, "ira")
                && suffix(&self.rv, "ira")
                && suffix_preceded(&self.ct, "ira", "e")
            {
                self.ct = replace_suffix(&self.ct, "ira", "ir");
                return true;
            }
        }

        return false;
    }

    // Java: BrazilianStemmer.step2
    fn step2(&mut self) -> bool {
        if self.rv.is_none() {
            return false;
        }

        if len(&self.rv) >= 7 {
            if suffix(&self.rv, "issemos") {
                self.ct = remove_suffix(&self.ct, "issemos");
                return true;
            }
            if suffix(&self.rv, "essemos") {
                self.ct = remove_suffix(&self.ct, "essemos");
                return true;
            }
            if suffix(&self.rv, "assemos") {
                self.ct = remove_suffix(&self.ct, "assemos");
                return true;
            }
            if suffix(&self.rv, "ariamos") {
                self.ct = remove_suffix(&self.ct, "ariamos");
                return true;
            }
            if suffix(&self.rv, "eriamos") {
                self.ct = remove_suffix(&self.ct, "eriamos");
                return true;
            }
            if suffix(&self.rv, "iriamos") {
                self.ct = remove_suffix(&self.ct, "iriamos");
                return true;
            }
        }

        if len(&self.rv) >= 6 {
            if suffix(&self.rv, "iremos") {
                self.ct = remove_suffix(&self.ct, "iremos");
                return true;
            }
            if suffix(&self.rv, "eremos") {
                self.ct = remove_suffix(&self.ct, "eremos");
                return true;
            }
            if suffix(&self.rv, "aremos") {
                self.ct = remove_suffix(&self.ct, "aremos");
                return true;
            }
            if suffix(&self.rv, "avamos") {
                self.ct = remove_suffix(&self.ct, "avamos");
                return true;
            }
            if suffix(&self.rv, "iramos") {
                self.ct = remove_suffix(&self.ct, "iramos");
                return true;
            }
            if suffix(&self.rv, "eramos") {
                self.ct = remove_suffix(&self.ct, "eramos");
                return true;
            }
            if suffix(&self.rv, "aramos") {
                self.ct = remove_suffix(&self.ct, "aramos");
                return true;
            }
            if suffix(&self.rv, "asseis") {
                self.ct = remove_suffix(&self.ct, "asseis");
                return true;
            }
            if suffix(&self.rv, "esseis") {
                self.ct = remove_suffix(&self.ct, "esseis");
                return true;
            }
            if suffix(&self.rv, "isseis") {
                self.ct = remove_suffix(&self.ct, "isseis");
                return true;
            }
            if suffix(&self.rv, "arieis") {
                self.ct = remove_suffix(&self.ct, "arieis");
                return true;
            }
            if suffix(&self.rv, "erieis") {
                self.ct = remove_suffix(&self.ct, "erieis");
                return true;
            }
            if suffix(&self.rv, "irieis") {
                self.ct = remove_suffix(&self.ct, "irieis");
                return true;
            }
        }

        if len(&self.rv) >= 5 {
            if suffix(&self.rv, "irmos") {
                self.ct = remove_suffix(&self.ct, "irmos");
                return true;
            }
            if suffix(&self.rv, "iamos") {
                self.ct = remove_suffix(&self.ct, "iamos");
                return true;
            }
            if suffix(&self.rv, "armos") {
                self.ct = remove_suffix(&self.ct, "armos");
                return true;
            }
            if suffix(&self.rv, "ermos") {
                self.ct = remove_suffix(&self.ct, "ermos");
                return true;
            }
            if suffix(&self.rv, "areis") {
                self.ct = remove_suffix(&self.ct, "areis");
                return true;
            }
            if suffix(&self.rv, "ereis") {
                self.ct = remove_suffix(&self.ct, "ereis");
                return true;
            }
            if suffix(&self.rv, "ireis") {
                self.ct = remove_suffix(&self.ct, "ireis");
                return true;
            }
            if suffix(&self.rv, "asses") {
                self.ct = remove_suffix(&self.ct, "asses");
                return true;
            }
            if suffix(&self.rv, "esses") {
                self.ct = remove_suffix(&self.ct, "esses");
                return true;
            }
            if suffix(&self.rv, "isses") {
                self.ct = remove_suffix(&self.ct, "isses");
                return true;
            }
            if suffix(&self.rv, "astes") {
                self.ct = remove_suffix(&self.ct, "astes");
                return true;
            }
            if suffix(&self.rv, "assem") {
                self.ct = remove_suffix(&self.ct, "assem");
                return true;
            }
            if suffix(&self.rv, "essem") {
                self.ct = remove_suffix(&self.ct, "essem");
                return true;
            }
            if suffix(&self.rv, "issem") {
                self.ct = remove_suffix(&self.ct, "issem");
                return true;
            }
            if suffix(&self.rv, "ardes") {
                self.ct = remove_suffix(&self.ct, "ardes");
                return true;
            }
            if suffix(&self.rv, "erdes") {
                self.ct = remove_suffix(&self.ct, "erdes");
                return true;
            }
            if suffix(&self.rv, "irdes") {
                self.ct = remove_suffix(&self.ct, "irdes");
                return true;
            }
            if suffix(&self.rv, "ariam") {
                self.ct = remove_suffix(&self.ct, "ariam");
                return true;
            }
            if suffix(&self.rv, "eriam") {
                self.ct = remove_suffix(&self.ct, "eriam");
                return true;
            }
            if suffix(&self.rv, "iriam") {
                self.ct = remove_suffix(&self.ct, "iriam");
                return true;
            }
            if suffix(&self.rv, "arias") {
                self.ct = remove_suffix(&self.ct, "arias");
                return true;
            }
            if suffix(&self.rv, "erias") {
                self.ct = remove_suffix(&self.ct, "erias");
                return true;
            }
            if suffix(&self.rv, "irias") {
                self.ct = remove_suffix(&self.ct, "irias");
                return true;
            }
            if suffix(&self.rv, "estes") {
                self.ct = remove_suffix(&self.ct, "estes");
                return true;
            }
            if suffix(&self.rv, "istes") {
                self.ct = remove_suffix(&self.ct, "istes");
                return true;
            }
            if suffix(&self.rv, "areis") {
                self.ct = remove_suffix(&self.ct, "areis");
                return true;
            }
            if suffix(&self.rv, "aveis") {
                self.ct = remove_suffix(&self.ct, "aveis");
                return true;
            }
        }

        if len(&self.rv) >= 4 {
            if suffix(&self.rv, "aria") {
                self.ct = remove_suffix(&self.ct, "aria");
                return true;
            }
            if suffix(&self.rv, "eria") {
                self.ct = remove_suffix(&self.ct, "eria");
                return true;
            }
            if suffix(&self.rv, "iria") {
                self.ct = remove_suffix(&self.ct, "iria");
                return true;
            }
            if suffix(&self.rv, "asse") {
                self.ct = remove_suffix(&self.ct, "asse");
                return true;
            }
            if suffix(&self.rv, "esse") {
                self.ct = remove_suffix(&self.ct, "esse");
                return true;
            }
            if suffix(&self.rv, "isse") {
                self.ct = remove_suffix(&self.ct, "isse");
                return true;
            }
            if suffix(&self.rv, "aste") {
                self.ct = remove_suffix(&self.ct, "aste");
                return true;
            }
            if suffix(&self.rv, "este") {
                self.ct = remove_suffix(&self.ct, "este");
                return true;
            }
            if suffix(&self.rv, "iste") {
                self.ct = remove_suffix(&self.ct, "iste");
                return true;
            }
            if suffix(&self.rv, "arei") {
                self.ct = remove_suffix(&self.ct, "arei");
                return true;
            }
            if suffix(&self.rv, "erei") {
                self.ct = remove_suffix(&self.ct, "erei");
                return true;
            }
            if suffix(&self.rv, "irei") {
                self.ct = remove_suffix(&self.ct, "irei");
                return true;
            }
            if suffix(&self.rv, "aram") {
                self.ct = remove_suffix(&self.ct, "aram");
                return true;
            }
            if suffix(&self.rv, "eram") {
                self.ct = remove_suffix(&self.ct, "eram");
                return true;
            }
            if suffix(&self.rv, "iram") {
                self.ct = remove_suffix(&self.ct, "iram");
                return true;
            }
            if suffix(&self.rv, "avam") {
                self.ct = remove_suffix(&self.ct, "avam");
                return true;
            }
            if suffix(&self.rv, "arem") {
                self.ct = remove_suffix(&self.ct, "arem");
                return true;
            }
            if suffix(&self.rv, "erem") {
                self.ct = remove_suffix(&self.ct, "erem");
                return true;
            }
            if suffix(&self.rv, "irem") {
                self.ct = remove_suffix(&self.ct, "irem");
                return true;
            }
            if suffix(&self.rv, "ando") {
                self.ct = remove_suffix(&self.ct, "ando");
                return true;
            }
            if suffix(&self.rv, "endo") {
                self.ct = remove_suffix(&self.ct, "endo");
                return true;
            }
            if suffix(&self.rv, "indo") {
                self.ct = remove_suffix(&self.ct, "indo");
                return true;
            }
            if suffix(&self.rv, "arao") {
                self.ct = remove_suffix(&self.ct, "arao");
                return true;
            }
            if suffix(&self.rv, "erao") {
                self.ct = remove_suffix(&self.ct, "erao");
                return true;
            }
            if suffix(&self.rv, "irao") {
                self.ct = remove_suffix(&self.ct, "irao");
                return true;
            }
            if suffix(&self.rv, "adas") {
                self.ct = remove_suffix(&self.ct, "adas");
                return true;
            }
            if suffix(&self.rv, "idas") {
                self.ct = remove_suffix(&self.ct, "idas");
                return true;
            }
            if suffix(&self.rv, "aras") {
                self.ct = remove_suffix(&self.ct, "aras");
                return true;
            }
            if suffix(&self.rv, "eras") {
                self.ct = remove_suffix(&self.ct, "eras");
                return true;
            }
            if suffix(&self.rv, "iras") {
                self.ct = remove_suffix(&self.ct, "iras");
                return true;
            }
            if suffix(&self.rv, "avas") {
                self.ct = remove_suffix(&self.ct, "avas");
                return true;
            }
            if suffix(&self.rv, "ares") {
                self.ct = remove_suffix(&self.ct, "ares");
                return true;
            }
            if suffix(&self.rv, "eres") {
                self.ct = remove_suffix(&self.ct, "eres");
                return true;
            }
            if suffix(&self.rv, "ires") {
                self.ct = remove_suffix(&self.ct, "ires");
                return true;
            }
            if suffix(&self.rv, "ados") {
                self.ct = remove_suffix(&self.ct, "ados");
                return true;
            }
            if suffix(&self.rv, "idos") {
                self.ct = remove_suffix(&self.ct, "idos");
                return true;
            }
            if suffix(&self.rv, "amos") {
                self.ct = remove_suffix(&self.ct, "amos");
                return true;
            }
            if suffix(&self.rv, "emos") {
                self.ct = remove_suffix(&self.ct, "emos");
                return true;
            }
            if suffix(&self.rv, "imos") {
                self.ct = remove_suffix(&self.ct, "imos");
                return true;
            }
            if suffix(&self.rv, "iras") {
                self.ct = remove_suffix(&self.ct, "iras");
                return true;
            }
            if suffix(&self.rv, "ieis") {
                self.ct = remove_suffix(&self.ct, "ieis");
                return true;
            }
        }

        if len(&self.rv) >= 3 {
            if suffix(&self.rv, "ada") {
                self.ct = remove_suffix(&self.ct, "ada");
                return true;
            }
            if suffix(&self.rv, "ida") {
                self.ct = remove_suffix(&self.ct, "ida");
                return true;
            }
            if suffix(&self.rv, "ara") {
                self.ct = remove_suffix(&self.ct, "ara");
                return true;
            }
            if suffix(&self.rv, "era") {
                self.ct = remove_suffix(&self.ct, "era");
                return true;
            }
            if suffix(&self.rv, "ira") {
                self.ct = remove_suffix(&self.ct, "ava");
                return true;
            }
            if suffix(&self.rv, "iam") {
                self.ct = remove_suffix(&self.ct, "iam");
                return true;
            }
            if suffix(&self.rv, "ado") {
                self.ct = remove_suffix(&self.ct, "ado");
                return true;
            }
            if suffix(&self.rv, "ido") {
                self.ct = remove_suffix(&self.ct, "ido");
                return true;
            }
            if suffix(&self.rv, "ias") {
                self.ct = remove_suffix(&self.ct, "ias");
                return true;
            }
            if suffix(&self.rv, "ais") {
                self.ct = remove_suffix(&self.ct, "ais");
                return true;
            }
            if suffix(&self.rv, "eis") {
                self.ct = remove_suffix(&self.ct, "eis");
                return true;
            }
            if suffix(&self.rv, "ira") {
                self.ct = remove_suffix(&self.ct, "ira");
                return true;
            }
            if suffix(&self.rv, "ear") {
                self.ct = remove_suffix(&self.ct, "ear");
                return true;
            }
        }

        if len(&self.rv) >= 2 {
            if suffix(&self.rv, "ia") {
                self.ct = remove_suffix(&self.ct, "ia");
                return true;
            }
            if suffix(&self.rv, "ei") {
                self.ct = remove_suffix(&self.ct, "ei");
                return true;
            }
            if suffix(&self.rv, "am") {
                self.ct = remove_suffix(&self.ct, "am");
                return true;
            }
            if suffix(&self.rv, "em") {
                self.ct = remove_suffix(&self.ct, "em");
                return true;
            }
            if suffix(&self.rv, "ar") {
                self.ct = remove_suffix(&self.ct, "ar");
                return true;
            }
            if suffix(&self.rv, "er") {
                self.ct = remove_suffix(&self.ct, "er");
                return true;
            }
            if suffix(&self.rv, "ir") {
                self.ct = remove_suffix(&self.ct, "ir");
                return true;
            }
            if suffix(&self.rv, "as") {
                self.ct = remove_suffix(&self.ct, "as");
                return true;
            }
            if suffix(&self.rv, "es") {
                self.ct = remove_suffix(&self.ct, "es");
                return true;
            }
            if suffix(&self.rv, "is") {
                self.ct = remove_suffix(&self.ct, "is");
                return true;
            }
            if suffix(&self.rv, "eu") {
                self.ct = remove_suffix(&self.ct, "eu");
                return true;
            }
            if suffix(&self.rv, "iu") {
                self.ct = remove_suffix(&self.ct, "iu");
                return true;
            }
            if suffix(&self.rv, "iu") {
                self.ct = remove_suffix(&self.ct, "iu");
                return true;
            }
            if suffix(&self.rv, "ou") {
                self.ct = remove_suffix(&self.ct, "ou");
                return true;
            }
        }

        return false;
    }

    // Java: BrazilianStemmer.step3
    fn step3(&mut self) {
        if self.rv.is_none() {
            return;
        }

        if suffix(&self.rv, "i") && suffix_preceded(&self.rv, "i", "c") {
            self.ct = remove_suffix(&self.ct, "i");
        }
    }

    // Java: BrazilianStemmer.step4
    fn step4(&mut self) {
        if self.rv.is_none() {
            return;
        }

        if suffix(&self.rv, "os") {
            self.ct = remove_suffix(&self.ct, "os");
            return;
        }
        if suffix(&self.rv, "a") {
            self.ct = remove_suffix(&self.ct, "a");
            return;
        }
        if suffix(&self.rv, "i") {
            self.ct = remove_suffix(&self.ct, "i");
            return;
        }
        if suffix(&self.rv, "o") {
            self.ct = remove_suffix(&self.ct, "o");
            return;
        }
    }

    // Java: BrazilianStemmer.step5
    fn step5(&mut self) {
        if self.rv.is_none() {
            return;
        }

        if suffix(&self.rv, "e") {
            if suffix_preceded(&self.rv, "e", "gu") {
                self.ct = remove_suffix(&self.ct, "e");
                self.ct = remove_suffix(&self.ct, "u");
                return;
            }

            if suffix_preceded(&self.rv, "e", "ci") {
                self.ct = remove_suffix(&self.ct, "e");
                self.ct = remove_suffix(&self.ct, "i");
                return;
            }

            self.ct = remove_suffix(&self.ct, "e");
            return;
        }
    }
}

/// `BrazilianStemFilter`: [`BrazilianStemmer`] on non-keyword terms.
pub struct BrazilianStemFilter<I> {
    input: I,
    stemmer: BrazilianStemmer,
    buf: Vec<u16>,
}

impl<I: TokenStream> BrazilianStemFilter<I> {
    /// `new BrazilianStemFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        BrazilianStemFilter {
            input,
            stemmer: BrazilianStemmer::default(),
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for BrazilianStemFilter<I> {
    crate::filter_input!();

    // Java: BrazilianStemFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.is_keyword() {
            let stemmer = &mut self.stemmer;
            crate::util::with_utf16_term(a, &mut self.buf, |b| match stemmer.stem(b) {
                Some(s) if s != *b => {
                    *b = s;
                    true
                }
                _ => false,
            });
        }
        Ok(true)
    }
}

language_analyzer! {
    /// `BrazilianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`BrazilianStemFilter`].
    BrazilianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        BrazilianStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
