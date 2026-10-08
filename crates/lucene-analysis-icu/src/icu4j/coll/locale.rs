//! `com.ibm.icu.util.ULocale`'s ID canonicalization (`getName`) and its
//! parts, through a port of `com.ibm.icu.impl.LocaleIDParser`: ICU locale
//! IDs (`de_DE`, `zh_Hant_TW`, `de@collation=phonebook`), which is how a
//! collator is asked for.
//!
//! Ported: language (lower case, three-letter codes mapped to two letters
//! as `LocaleIDs` does), script (four letters, title case), country (two or
//! three, upper case, three-letter codes mapped), variants (upper case,
//! `-`/`,` as `_`), `@key=value;...` keywords (keys lower case and sorted,
//! values trimmed), `root`, and a leading `und`. Not ported: IDs `getName`
//! reads as BCP 47 language tags (those with a one-character subtag, e.g.
//! `en-u-co-phonebk`), which are refused with `UnsupportedOperation`.

use std::collections::BTreeMap;

use crate::icu4j::coll::res::pack_text;
use crate::{IcuError, IcuErrorKind};

const DONE: char = '\u{ffff}';

/// `LocaleIDParser`.
struct Parser {
    id: Vec<char>,
    index: usize,
    buffer: String,
    had_country: bool,
}

fn is_terminator(c: char) -> bool {
    c == '@' || c == DONE || c == '.'
}

fn is_terminator_or_id_separator(c: char) -> bool {
    c == '_' || c == '-' || is_terminator(c)
}

fn mapped(table: &str, code: &str) -> Option<String> {
    pack_text(table).lines().find_map(|l| {
        let (k, v) = l.split_once('=')?;
        (k == code).then(|| v.to_string())
    })
}

// ARITH: indexes within the ID's characters.
#[allow(clippy::arithmetic_side_effects)]
impl Parser {
    fn new(id: &str) -> Self {
        Parser {
            id: id.chars().collect(),
            index: 0,
            buffer: String::new(),
            had_country: false,
        }
    }

    fn next(&mut self) -> char {
        if self.index >= self.id.len() {
            self.index += 1;
            return DONE;
        }
        let c = self.id[self.index];
        self.index += 1;
        c
    }

    fn at_terminator(&self) -> bool {
        self.index >= self.id.len() || is_terminator(self.id[self.index])
    }

    fn have_experimental_language_prefix(&self) -> bool {
        if self.id.len() > 2 {
            let c = self.id[1];
            if c == '-' || c == '_' {
                let c = self.id[0];
                return matches!(c, 'x' | 'X' | 'i' | 'I');
            }
        }
        false
    }

    fn have_keyword_assign(&self) -> bool {
        self.id.iter().skip(self.index).any(|&c| c == '=')
    }

    fn parse_language(&mut self) {
        let start = self.buffer.len();
        if self.have_experimental_language_prefix() {
            self.buffer.push(self.id[0].to_ascii_lowercase());
            self.buffer.push('-');
            self.index = 2;
        }
        loop {
            let c = self.next();
            if is_terminator_or_id_separator(c) {
                break;
            }
            self.buffer.push(c.to_ascii_lowercase());
        }
        self.index -= 1;
        if self.buffer.len() - start == 3 {
            if let Some(two) = mapped("lang3.txt", &self.buffer) {
                self.buffer = two;
            }
        }
    }

    fn parse_script(&mut self) {
        if self.at_terminator() {
            return;
        }
        let old_index = self.index;
        self.index += 1;
        let old_blen = self.buffer.len();
        let mut first = true;
        loop {
            let c = self.next();
            if is_terminator_or_id_separator(c) || !c.is_ascii_alphabetic() {
                break;
            }
            if first {
                self.buffer.push('_');
                self.buffer.push(c.to_ascii_uppercase());
                first = false;
            } else {
                self.buffer.push(c.to_ascii_lowercase());
            }
        }
        self.index -= 1;
        if self.index - old_index != 5 {
            self.index = old_index;
            self.buffer.truncate(old_blen);
        }
    }

    fn parse_country(&mut self) {
        if self.at_terminator() {
            return;
        }
        let old_index = self.index;
        self.index += 1;
        let mut old_blen = self.buffer.len();
        let mut first = true;
        loop {
            let c = self.next();
            if is_terminator_or_id_separator(c) {
                break;
            }
            if first {
                self.had_country = true;
                self.buffer.push('_');
                old_blen += 1;
                first = false;
            }
            self.buffer.push(c.to_ascii_uppercase());
        }
        self.index -= 1;
        let appended = self.buffer.len() - old_blen;
        if appended == 0 {
            // Empty country: "de__PHONEBOOK".
        } else if !(2..=3).contains(&appended) {
            self.index = old_index;
            old_blen -= 1;
            self.buffer.truncate(old_blen);
            self.had_country = false;
        } else if appended == 3 {
            if let Some(two) = mapped("region3.txt", &self.buffer[old_blen..]) {
                self.buffer.truncate(old_blen);
                self.buffer.push_str(&two);
            }
        }
    }

    fn parse_variant(&mut self) -> Result<(), IcuError> {
        let mut old_blen = self.buffer.len();
        let mut start = true;
        let mut need_separator = true;
        let mut skipping = false;
        let mut first = true;
        loop {
            let c = self.next();
            if c == DONE {
                break;
            }
            if c == '.' {
                start = false;
                skipping = true;
            } else if c == '@' {
                if self.have_keyword_assign() {
                    break;
                }
                skipping = false;
                start = false;
                need_separator = true;
            } else if start {
                start = false;
                if c != '_' && c != '-' {
                    self.index -= 1;
                }
            } else if !skipping {
                if need_separator {
                    need_separator = false;
                    if first && !self.had_country {
                        self.buffer.push('_');
                        old_blen += 1;
                    }
                    self.buffer.push('_');
                    if first {
                        old_blen += 1;
                        first = false;
                    }
                }
                let mut c = c.to_ascii_uppercase();
                if c == '-' || c == ',' {
                    c = '_';
                }
                self.buffer.push(c);
                if self.buffer.len() - old_blen > 179 {
                    return Err(IcuError::with_kind(
                        IcuErrorKind::IllegalArgument,
                        "variants is too long",
                    ));
                }
            }
        }
        self.index -= 1;
        Ok(())
    }

    /// `getKeywordMap()` (the parser's position is past the base name).
    fn keyword_map(&mut self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        // setToKeywordStart (canonicalize == false)
        let Some(at) = self.id.iter().skip(self.index).position(|&c| c == '@') else {
            return m;
        };
        let i = self.index + at + 1;
        if i >= self.id.len() {
            return m;
        }
        self.index = i;
        loop {
            let start = self.index;
            loop {
                let c = self.next();
                if c == DONE || c == '=' {
                    break;
                }
            }
            self.index -= 1;
            let end = self.index.min(self.id.len());
            let key: String = self.id[start.min(end)..end]
                .iter()
                .collect::<String>()
                .trim()
                .to_ascii_lowercase();
            if key.is_empty() {
                break;
            }
            let c = self.next();
            if c != '=' {
                if c == DONE {
                    break;
                }
                if self.next() != ';' {
                    break;
                }
                continue;
            }
            let vstart = self.index;
            loop {
                let c = self.next();
                if c == DONE || c == ';' {
                    break;
                }
            }
            self.index -= 1;
            let vend = self.index.min(self.id.len());
            let value: String = self.id[vstart.min(vend)..vend]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
            if !value.is_empty() {
                m.entry(key).or_insert(value);
            }
            if self.next() != ';' {
                break;
            }
        }
        m
    }
}

/// `getShortestSubtagLength(localeID)` (the last subtag is not counted, as
/// in Java).
fn shortest_subtag_length(id: &str) -> usize {
    let mut length = id.chars().count();
    let mut reset = true;
    let mut tmp = 0usize;
    for c in id.chars() {
        if c != '_' && c != '-' {
            if reset {
                reset = false;
                tmp = 0;
            }
            tmp = tmp.saturating_add(1);
        } else {
            if tmp != 0 && tmp < length {
                length = tmp;
            }
            reset = true;
        }
    }
    length
}

/// A parsed `ULocale`: its canonical name and parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Locale {
    /// `getBaseName()`.
    pub base_name: String,
    /// The keywords, keys lower case.
    pub keywords: BTreeMap<String, String>,
}

impl Locale {
    /// `new ULocale(localeID)`.
    pub fn new(id: &str) -> Result<Locale, IcuError> {
        let id = if !id.contains('@') && shortest_subtag_length(id) == 1 {
            return Err(IcuError::with_kind(
                IcuErrorKind::UnsupportedOperation,
                format!("BCP 47 language tag locale IDs are not supported: {id}"),
            ));
        } else if id.eq_ignore_ascii_case("root") {
            ""
        } else if id.len() >= 3 && id.as_bytes()[..3].eq_ignore_ascii_case(b"und") {
            match id.as_bytes().get(3) {
                None => "",
                Some(b'-' | b'_') => &id[3..],
                Some(_) => id,
            }
        } else {
            id
        };
        let mut p = Parser::new(id);
        p.parse_language();
        p.parse_script();
        p.parse_country();
        p.parse_variant()?;
        if p.buffer.ends_with('_') {
            p.buffer.pop();
        }
        let base_name = std::mem::take(&mut p.buffer);
        let keywords = p.keyword_map();
        Ok(Locale {
            base_name,
            keywords,
        })
    }

    /// The root locale.
    pub fn root() -> Locale {
        Locale {
            base_name: String::new(),
            keywords: BTreeMap::new(),
        }
    }

    /// `getName()`.
    pub fn name(&self) -> String {
        let mut s = self.base_name.clone();
        for (i, (k, v)) in self.keywords.iter().enumerate() {
            s.push(if i == 0 { '@' } else { ';' });
            s.push_str(k);
            s.push('=');
            s.push_str(v);
        }
        s
    }

    /// `getKeywordValue(name)`.
    pub fn keyword(&self, name: &str) -> Option<&str> {
        self.keywords
            .get(&name.trim().to_ascii_lowercase())
            .map(String::as_str)
    }

    /// The keywords part of `getName()` (`@...`, or empty).
    pub fn keyword_suffix(&self) -> String {
        let n = self.name();
        n[self.base_name.len()..].to_string()
    }
}

/// `ULocale.getLanguage/getScript/getCountry/getVariant(name)` of a base
/// name: its four parts.
pub fn parts(name: &str) -> (String, String, String, String) {
    let mut p = Parser::new(name);
    p.parse_language();
    let lang = std::mem::take(&mut p.buffer);
    p.parse_script();
    let script = std::mem::take(&mut p.buffer)
        .trim_start_matches('_')
        .to_string();
    p.parse_country();
    let country = std::mem::take(&mut p.buffer)
        .trim_start_matches('_')
        .to_string();
    // getVariant re-parses with skips; after language/script/country the
    // remaining text is the variant.
    let _ = p.parse_variant();
    let variant = std::mem::take(&mut p.buffer)
        .trim_start_matches('_')
        .to_string();
    (lang, script, country, variant)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(id: &str) -> String {
        Locale::new(id).unwrap().name()
    }

    #[test]
    fn canonical_names() {
        assert_eq!(name("de_DE"), "de_DE");
        assert_eq!(name("DE-de"), "de_DE");
        assert_eq!(name("zh_hant_tw"), "zh_Hant_TW");
        assert_eq!(name("root"), "");
        assert_eq!(name("und"), "");
        assert_eq!(name("und_DE"), "_DE");
        assert_eq!(name("undx"), "undx");
        assert_eq!(name("deu_DEU"), "de_DE");
        assert_eq!(name("de__phonebook"), "de__PHONEBOOK");
        assert_eq!(name("de@collation=phonebook"), "de@collation=phonebook");
        assert_eq!(
            name("de@ CoLLation = phonebook ;colStrength=primary"),
            "de@collation=phonebook;colstrength=primary"
        );
        assert_eq!(name("en_US_POSIX"), "en_US_POSIX");
        assert_eq!(name("en_USA1"), "en__USA1");
        assert_eq!(name("es__traditional"), "es__TRADITIONAL");
        assert_eq!(name("en.utf8@x=1"), "en@x=1");
        assert_eq!(name("en@=1;b=2"), "en");
        assert_eq!(name("en@a;b=2"), "en@a;b=2");
        assert_eq!(name("en@a=;b=2"), "en@b=2");
        assert_eq!(name("en@a=1;a=2"), "en@a=1");
        assert_eq!(name("en@"), "en");
        assert!(Locale::new("en-u-co-phonebk").is_err());
        assert!(Locale::new(&format!("en__{}", "A".repeat(200))).is_err());
        let l = Locale::new("de@collation=phonebook").unwrap();
        assert_eq!(l.keyword("Collation"), Some("phonebook"));
        assert_eq!(l.keyword_suffix(), "@collation=phonebook");
        assert_eq!(Locale::root().name(), "");
    }

    #[test]
    fn locale_parts() {
        assert_eq!(
            parts("sr_Latn_RS"),
            ("sr".into(), "Latn".into(), "RS".into(), String::new())
        );
        assert_eq!(parts("de__PHONEBOOK").3, "PHONEBOOK");
        assert_eq!(parts("de").1, "");
        assert_eq!(shortest_subtag_length("en-u-co"), 1);
        assert_eq!(shortest_subtag_length("en_US"), 2);
    }
}
