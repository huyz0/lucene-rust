//! ICU4J's character names (`UCharacter.getCharFromExtendedName`), what
//! `\N{name}` and `[:name=...:]` in a `UnicodeSet` look up.
//!
//! The data is `resources/unames.bin.z`, which `tools/GenIcuNames.java` writes
//! from ICU4J 77.1's answers (see there for the format): every name ICU keeps
//! in its tables, plus the ranges of ideographs whose name is a prefix and
//! the code point in hex. A Hangul syllable's name is the Unicode algorithm.
//!
//! A name is looked up as ICU does: `<type-HEX>` (the name of a code point
//! that has none) first, then the algorithmic names, then the table, on the
//! name in upper case. Differs: only the names ICU4J reports through
//! `getName` and `getNameAlias` are in the table, so an alias ICU accepts
//! without reporting it (a name alias beyond the first of a code point) is
//! unknown here.

use std::sync::OnceLock;

use crate::icu4j::uprops;

static DATA: &[u8] = include_bytes!("../resources/unames.bin.z");

/// `UCharacterName`'s tables.
struct Names {
    /// An ideograph range: the name's prefix, the first and last code point.
    ranges: Vec<(String, u32, u32)>,
    /// The names in ascending order.
    names: Vec<String>,
    code_points: Vec<u32>,
}

/// A reader over the decompressed table; it stops at the first truncation,
/// which the empty tables of a corrupt resource then answer as "no name".
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.p.checked_add(n)?;
        let s = self.b.get(self.p..end)?;
        self.p = end;
        Some(s)
    }

    fn u8(&mut self) -> Option<usize> {
        self.take(1).map(|s| usize::from(s[0]))
    }

    fn u16(&mut self) -> Option<usize> {
        self.take(2)
            .map(|s| usize::from(u16::from_be_bytes([s[0], s[1]])))
    }

    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
}

fn load() -> Option<Names> {
    let bytes = miniz_oxide::inflate::decompress_to_vec_zlib(DATA).ok()?;
    let mut r = Reader { b: &bytes, p: 0 };
    if r.take(4)? != b"LIN1" {
        return None;
    }
    let mut ranges = Vec::new();
    for _ in 0..r.u16()? {
        let n = r.u8()?;
        let prefix = String::from_utf8(r.take(n)?.to_vec()).ok()?;
        ranges.push((prefix, r.u32()?, r.u32()?));
    }
    let count = usize::try_from(r.u32()?).ok()?;
    let mut names: Vec<String> = Vec::new();
    let mut code_points = Vec::new();
    let mut prev = String::new();
    for _ in 0..count.min(1 << 20) {
        let common = r.u8()?;
        let n = r.u8()?;
        let rest = String::from_utf8(r.take(n)?.to_vec()).ok()?;
        let mut name = prev.get(..common)?.to_string();
        name.push_str(&rest);
        code_points.push(r.u32()?);
        prev.clone_from(&name);
        names.push(name);
    }
    Some(Names {
        ranges,
        names,
        code_points,
    })
}

fn names() -> &'static Names {
    static N: OnceLock<Names> = OnceLock::new();
    N.get_or_init(|| {
        load().unwrap_or(Names {
            ranges: Vec::new(),
            names: Vec::new(),
            code_points: Vec::new(),
        })
    })
}

/// `UCharacterName.TYPE_NAMES`, by `UCharacter.getType` (30 is a
/// noncharacter, 31 and 32 the lead and trail surrogates).
const TYPE_NAMES: [&str; 33] = [
    "unassigned",
    "uppercase letter",
    "lowercase letter",
    "titlecase letter",
    "modifier letter",
    "other letter",
    "non spacing mark",
    "enclosing mark",
    "combining spacing mark",
    "decimal digit number",
    "letter number",
    "other number",
    "space separator",
    "line separator",
    "paragraph separator",
    "control",
    "format",
    "private use area",
    "surrogate",
    "dash punctuation",
    "start punctuation",
    "end punctuation",
    "connector punctuation",
    "other punctuation",
    "math symbol",
    "currency symbol",
    "modifier symbol",
    "other symbol",
    "initial punctuation",
    "final punctuation",
    "noncharacter",
    "lead surrogate",
    "trail surrogate",
];

/// `UCharacterUtil.isNonCharacter`.
fn is_non_character(c: u32) -> bool {
    (0xfdd0..=0xfdef).contains(&c) || (c & 0xfffe) == 0xfffe
}

/// `getExtendedChar`: `<type-HEX>` for the code point HEX of that type.
fn extended_char(lower: &str) -> Option<u32> {
    let inner = lower.strip_prefix('<')?.strip_suffix('>')?;
    let (kind_name, hex) = inner.rsplit_once('-')?;
    let cp = u32::from_str_radix(hex, 16).ok()?;
    if cp > 0x10ffff || hex.starts_with('-') {
        return None;
    }
    let kind = if is_non_character(cp) {
        30
    } else {
        match usize::try_from(uprops::char_type(i32::try_from(cp).ok()?)).ok()? {
            18 if cp <= 0xdbff => 31,
            18 => 32,
            other => other,
        }
    };
    (TYPE_NAMES.get(kind) == Some(&kind_name)).then_some(cp)
}

/// The Hangul syllable named `rest` (after `HANGUL SYLLABLE `): the
/// Unicode algorithm's lead, vowel and tail jamo short names.
fn hangul_syllable(rest: &str) -> Option<u32> {
    const LEADS: [&str; 19] = [
        "G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "", "J", "JJ", "C", "K", "T",
        "P", "H",
    ];
    const VOWELS: [&str; 21] = [
        "A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO",
        "WE", "WI", "YU", "EU", "YI", "I",
    ];
    const TAILS: [&str; 28] = [
        "", "G", "GG", "GS", "N", "NJ", "NH", "D", "L", "LG", "LM", "LB", "LS", "LT", "LP", "LH",
        "M", "B", "BS", "S", "SS", "NG", "J", "C", "K", "T", "P", "H",
    ];
    for (l, lead) in LEADS.iter().enumerate() {
        let Some(after_lead) = rest.strip_prefix(lead) else {
            continue;
        };
        for (v, vowel) in VOWELS.iter().enumerate() {
            let Some(tail) = after_lead.strip_prefix(vowel) else {
                continue;
            };
            if let Some(t) = TAILS.iter().position(|x| *x == tail) {
                // SBase + (LIndex * VCount + VIndex) * TCount + TIndex
                let syllable = l.saturating_mul(21).saturating_add(v);
                let cp = syllable.saturating_mul(28).saturating_add(t);
                return u32::try_from(cp.saturating_add(0xac00)).ok();
            }
        }
    }
    None
}

/// `UCharacter.getCharFromExtendedName(name)`: the code point the name
/// names, or `None` (Java's `-1`).
pub fn char_from_extended_name(name: &str) -> Option<u32> {
    if name.is_empty() {
        return None;
    }
    if name.starts_with('<') {
        return extended_char(&name.to_lowercase());
    }
    let upper = name.to_uppercase();
    let n = names();
    for (prefix, first, last) in &n.ranges {
        if let Some(hex) = upper.strip_prefix(prefix.as_str()) {
            if let Ok(cp) = u32::from_str_radix(hex, 16) {
                if (*first..=*last).contains(&cp) {
                    return Some(cp);
                }
            }
        }
    }
    if let Some(rest) = upper.strip_prefix("HANGUL SYLLABLE ") {
        return hangul_syllable(rest);
    }
    let i = n.names.binary_search(&upper).ok()?;
    n.code_points.get(i).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_of_every_kind() {
        let c = char_from_extended_name;
        assert_eq!(c("LATIN SMALL LETTER A"), Some(0x61));
        assert_eq!(c("latin small letter a"), Some(0x61));
        assert_eq!(c("HYPHEN-MINUS"), Some(0x2d));
        assert_eq!(c("CJK UNIFIED IDEOGRAPH-4E00"), Some(0x4e00));
        assert_eq!(c("CJK UNIFIED IDEOGRAPH-4e01"), Some(0x4e01));
        assert_eq!(c("CJK UNIFIED IDEOGRAPH-20000"), Some(0x20000));
        assert_eq!(c("CJK UNIFIED IDEOGRAPH-04E00"), Some(0x4e00));
        assert_eq!(c("CJK UNIFIED IDEOGRAPH-0"), None);
        assert_eq!(c("HANGUL SYLLABLE GA"), Some(0xac00));
        assert_eq!(c("HANGUL SYLLABLE HIH"), Some(0xd7a3));
        assert_eq!(c("HANGUL SYLLABLE GAX"), None);
        assert_eq!(c("<control-0007>"), Some(7));
        assert_eq!(c("<control-0041>"), None);
        assert_eq!(c("<noncharacter-FFFE>"), Some(0xfffe));
        assert_eq!(c("<private use area-E000>"), Some(0xe000));
        assert_eq!(c("<private use-E000>"), None);
        assert_eq!(c("<lead surrogate-D800>"), Some(0xd800));
        assert_eq!(c("<trail surrogate-DC00>"), Some(0xdc00));
        assert_eq!(c("<unassigned-0378>"), Some(0x378));
        assert_eq!(c("<control-zz>"), None);
        assert_eq!(c("NO SUCH CHARACTER"), None);
        assert_eq!(c(""), None);
    }
}
