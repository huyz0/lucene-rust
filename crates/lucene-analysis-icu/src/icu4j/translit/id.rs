//! `com.ibm.icu.text.TransliteratorIDParser`: transliterator IDs --
//! `Source-Target/Variant`, a filter `[...]` before each, `(Reverse)`
//! parts, `;`-separated compounds with a global filter first (or, in
//! parentheses, last) -- into single IDs, their canonical forms, and the
//! basic IDs the registry looks up; and the special inverses
//! (`Lower` <-> `Upper`, `NFC` <-> `NFD`, `Any-Remove` -> `Null`, ...).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::icu4j::translit::{FORWARD, REVERSE};
use crate::icu4j::unicode_set::{is_pattern_white_space, UnicodeSet};
use crate::icu4j::uprops;
use crate::icu4j::utf16;
use crate::IcuError;

const ANY: &str = "Any";

/// `SingleID`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleId {
    pub canon_id: String,
    pub basic_id: String,
    pub filter: Option<String>,
}

/// `Specs`.
#[derive(Debug, Clone)]
struct Specs {
    source: String,
    target: String,
    variant: Option<String>,
    filter: Option<String>,
    saw_source: bool,
}

/// `SPECIAL_INVERSES` (keys lower-cased: `CaseInsensitiveString`).
fn special_inverses() -> &'static Mutex<HashMap<String, String>> {
    static M: Mutex<Option<()>> = Mutex::new(None);
    let _ = &M;
    static INVERSES: std::sync::OnceLock<Mutex<HashMap<String, String>>> =
        std::sync::OnceLock::new();
    INVERSES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `registerSpecialInverse(target, inverseTarget, bidirectional)`.
pub fn register_special_inverse(target: &str, inverse: &str, bidirectional: bool) {
    let mut m = special_inverses().lock().unwrap_or_else(|e| e.into_inner());
    m.insert(target.to_lowercase(), inverse.to_string());
    if bidirectional && !target.eq_ignore_ascii_case(inverse) {
        m.insert(inverse.to_lowercase(), target.to_string());
    }
}

/// `PatternProps.skipWhiteSpace(s, i)`.
pub(crate) fn skip_white_space(s: &[u16], mut i: usize) -> usize {
    while i < s.len() && is_pattern_white_space(i32::from(s[i])) {
        i = i.saturating_add(1);
    }
    i
}

/// `Utility.parseChar(id, pos, ch)`.
pub(crate) fn parse_char(id: &[u16], pos: &mut usize, ch: char) -> bool {
    let start = *pos;
    *pos = skip_white_space(id, *pos);
    if *pos == id.len() || id[*pos] != ch as u16 {
        *pos = start;
        return false;
    }
    *pos = pos.saturating_add(1);
    true
}

/// `UCharacter.isUnicodeIdentifierStart(c)`: `ID_Start`.
fn is_id_start(c: i32) -> bool {
    uprops::uprops().int_value(16, c) != 0
}

/// `UCharacter.isUnicodeIdentifierPart(c)`: `ID_Continue`.
pub(crate) fn is_id_part(c: i32) -> bool {
    uprops::uprops().int_value(15, c) != 0
}

/// `UCharacter.isUnicodeIdentifierStart` for a stand-in check.
pub(crate) fn is_id_start_char(c: i32) -> bool {
    is_id_start(c)
}

/// `Utility.parseUnicodeIdentifier(str, pos)`.
fn parse_unicode_identifier(s: &[u16], pos: &mut usize) -> Option<String> {
    let mut buf: Vec<u16> = Vec::new();
    let mut p = *pos;
    while p < s.len() {
        let ch = utf16::code_point_at(s, p);
        if buf.is_empty() {
            if is_id_start(ch) {
                utf16::push_code_point(&mut buf, ch);
            } else {
                return None;
            }
        } else if is_id_part(ch) {
            utf16::push_code_point(&mut buf, ch);
        } else {
            break;
        }
        p = p.saturating_add(if ch > 0xffff { 2 } else { 1 });
    }
    *pos = p;
    Some(String::from_utf16_lossy(&buf))
}

fn sub(s: &[u16], a: usize, b: usize) -> String {
    String::from_utf16_lossy(s.get(a..b).unwrap_or(&[]))
}

/// `parseFilterID(id, pos, allowFilter)`.
fn parse_filter_id_specs(
    id: &[u16],
    pos: &mut usize,
    allow_filter: bool,
) -> Result<Option<Specs>, IcuError> {
    let mut first: Option<String> = None;
    let mut target: Option<String> = None;
    let mut variant: Option<String> = None;
    let mut filter: Option<String> = None;
    let mut delimiter: u16 = 0;
    let mut spec_count = 0u32;
    let start = *pos;
    loop {
        *pos = skip_white_space(id, *pos);
        if *pos == id.len() {
            break;
        }
        if allow_filter && filter.is_none() && UnicodeSet::resembles_pattern(id, *pos) {
            let mut p = *pos;
            UnicodeSet::parse_at(id, &mut p, None)?;
            filter = Some(sub(id, *pos, p));
            *pos = p;
            continue;
        }
        if delimiter == 0 {
            let c = id[*pos];
            if (c == '-' as u16 && target.is_none()) || (c == '/' as u16 && variant.is_none()) {
                delimiter = c;
                *pos = pos.saturating_add(1);
                continue;
            }
        }
        if delimiter == 0 && spec_count > 0 {
            break;
        }
        let Some(spec) = parse_unicode_identifier(id, pos) else {
            break;
        };
        match delimiter {
            0 => first = Some(spec),
            0x2d => target = Some(spec),
            _ => variant = Some(spec),
        }
        spec_count = spec_count.saturating_add(1);
        delimiter = 0;
    }
    let mut source: Option<String> = None;
    if let Some(f) = first {
        if target.is_none() {
            target = Some(f);
        } else {
            source = Some(f);
        }
    }
    if source.is_none() && target.is_none() {
        *pos = start;
        return Ok(None);
    }
    let saw_source = source.is_some();
    Ok(Some(Specs {
        source: source.unwrap_or_else(|| ANY.to_string()),
        target: target.unwrap_or_else(|| ANY.to_string()),
        variant,
        filter,
        saw_source,
    }))
}

/// `specsToID(specs, dir)`.
fn specs_to_id(specs: Option<&Specs>, dir: i32) -> SingleId {
    let mut canon_id = String::new();
    let mut basic_id = String::new();
    let mut basic_prefix = String::new();
    if let Some(specs) = specs {
        let mut buf = String::new();
        if dir == FORWARD {
            if specs.saw_source {
                buf.push_str(&specs.source);
                buf.push('-');
            } else {
                basic_prefix = format!("{}-", specs.source);
            }
            buf.push_str(&specs.target);
        } else {
            buf.push_str(&specs.target);
            buf.push('-');
            buf.push_str(&specs.source);
        }
        if let Some(v) = &specs.variant {
            buf.push('/');
            buf.push_str(v);
        }
        basic_id = format!("{basic_prefix}{buf}");
        if let Some(f) = &specs.filter {
            buf.insert_str(0, f);
        }
        canon_id = buf;
    }
    SingleId {
        canon_id,
        basic_id,
        filter: None,
    }
}

/// `specsToSpecialInverse(specs)`.
fn specs_to_special_inverse(specs: &Specs) -> Option<SingleId> {
    if !specs.source.eq_ignore_ascii_case(ANY) {
        return None;
    }
    let inverse = special_inverses()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&specs.target.to_lowercase())
        .cloned()?;
    let mut buf = String::new();
    if let Some(f) = &specs.filter {
        buf.push_str(f);
    }
    if specs.saw_source {
        buf.push_str("Any-");
    }
    buf.push_str(&inverse);
    let mut basic_id = format!("Any-{inverse}");
    if let Some(v) = &specs.variant {
        buf.push('/');
        buf.push_str(v);
        basic_id = format!("{basic_id}/{v}");
    }
    Some(SingleId {
        canon_id: buf,
        basic_id,
        filter: None,
    })
}

/// `parseFilterID(id, pos)`: a single ID without `(...)`, for a rule's
/// `&Translit(` function call.
pub fn parse_filter_id(id: &[u16], pos: &mut usize) -> Result<Option<SingleId>, IcuError> {
    let start = *pos;
    let Some(specs) = parse_filter_id_specs(id, pos, true)? else {
        *pos = start;
        return Ok(None);
    };
    let mut single = specs_to_id(Some(&specs), FORWARD);
    single.filter.clone_from(&specs.filter);
    Ok(Some(single))
}

/// `parseSingleID(id, pos, dir)`.
pub fn parse_single_id(
    id: &[u16],
    pos: &mut usize,
    dir: i32,
) -> Result<Option<SingleId>, IcuError> {
    let start = *pos;
    let mut specs_a: Option<Specs> = None;
    let mut specs_b: Option<Specs> = None;
    let mut saw_paren = false;
    for pass in 1..=2 {
        if pass == 2 {
            specs_a = parse_filter_id_specs(id, pos, true)?;
            if specs_a.is_none() {
                *pos = start;
                return Ok(None);
            }
        }
        if parse_char(id, pos, '(') {
            saw_paren = true;
            if !parse_char(id, pos, ')') {
                specs_b = parse_filter_id_specs(id, pos, true)?;
                if specs_b.is_none() || !parse_char(id, pos, ')') {
                    *pos = start;
                    return Ok(None);
                }
            }
            break;
        }
    }
    let single = if saw_paren {
        if dir == FORWARD {
            let mut single = specs_to_id(specs_a.as_ref(), FORWARD);
            single.canon_id = format!(
                "{}({})",
                single.canon_id,
                specs_to_id(specs_b.as_ref(), FORWARD).canon_id
            );
            if let Some(a) = &specs_a {
                single.filter.clone_from(&a.filter);
            }
            single
        } else {
            let mut single = specs_to_id(specs_b.as_ref(), FORWARD);
            single.canon_id = format!(
                "{}({})",
                single.canon_id,
                specs_to_id(specs_a.as_ref(), FORWARD).canon_id
            );
            if let Some(b) = &specs_b {
                single.filter.clone_from(&b.filter);
            }
            single
        }
    } else {
        let Some(a) = specs_a else {
            *pos = start;
            return Ok(None);
        };
        let mut single = if dir == FORWARD {
            specs_to_id(Some(&a), FORWARD)
        } else {
            specs_to_special_inverse(&a).unwrap_or_else(|| specs_to_id(Some(&a), REVERSE))
        };
        single.filter.clone_from(&a.filter);
        single
    };
    Ok(Some(single))
}

/// `parseGlobalFilter(id, pos, dir, withParens, canonID)`.
pub fn parse_global_filter(
    id: &[u16],
    pos: &mut usize,
    dir: i32,
    with_parens: &mut i32,
    canon_id: Option<&mut String>,
) -> Option<UnicodeSet> {
    let start = *pos;
    if *with_parens == -1 {
        *with_parens = i32::from(parse_char(id, pos, '('));
    } else if *with_parens == 1 && !parse_char(id, pos, '(') {
        *pos = start;
        return None;
    }
    *pos = skip_white_space(id, *pos);
    if !UnicodeSet::resembles_pattern(id, *pos) {
        return None;
    }
    let mut p = *pos;
    let Ok(filter) = UnicodeSet::parse_at(id, &mut p, None) else {
        *pos = start;
        return None;
    };
    let mut pattern = sub(id, *pos, p);
    *pos = p;
    if *with_parens == 1 && !parse_char(id, pos, ')') {
        *pos = start;
        return None;
    }
    if let Some(canon) = canon_id {
        if dir == FORWARD {
            if *with_parens == 1 {
                pattern = format!("({pattern})");
            }
            canon.push_str(&pattern);
            canon.push(';');
        } else {
            if *with_parens == 0 {
                pattern = format!("({pattern})");
            }
            canon.insert_str(0, &format!("{pattern};"));
        }
    }
    Some(filter)
}

/// A parsed compound ID: the canonical ID, the single IDs and the global
/// filter.
pub type CompoundId = (String, Vec<SingleId>, Option<UnicodeSet>);

/// `parseCompoundID(id, dir, canonID, list, globalFilter)`: `None` for an
/// invalid ID.
pub fn parse_compound_id(id: &str, dir: i32) -> Result<Option<CompoundId>, IcuError> {
    let id: Vec<u16> = id.encode_utf16().collect();
    let mut pos = 0usize;
    let mut with_parens = 0;
    let mut list: Vec<SingleId> = Vec::new();
    let mut canon_id = String::new();
    let mut global_filter = None;
    if let Some(filter) =
        parse_global_filter(&id, &mut pos, dir, &mut with_parens, Some(&mut canon_id))
    {
        if !parse_char(&id, &mut pos, ';') {
            canon_id.clear();
            pos = 0;
        }
        if dir == FORWARD {
            global_filter = Some(filter);
        }
    }
    let mut saw_delimiter = true;
    while let Some(single) = parse_single_id(&id, &mut pos, dir)? {
        if dir == FORWARD {
            list.push(single);
        } else {
            list.insert(0, single);
        }
        if !parse_char(&id, &mut pos, ';') {
            saw_delimiter = false;
            break;
        }
    }
    if list.is_empty() {
        return Ok(None);
    }
    let canon_ids: Vec<&str> = list.iter().map(|s| s.canon_id.as_str()).collect();
    canon_id.push_str(&canon_ids.join(";"));
    if saw_delimiter {
        with_parens = 1;
        if let Some(filter) =
            parse_global_filter(&id, &mut pos, dir, &mut with_parens, Some(&mut canon_id))
        {
            parse_char(&id, &mut pos, ';');
            if dir == REVERSE {
                global_filter = Some(filter);
            }
        }
    }
    pos = skip_white_space(&id, pos);
    if pos != id.len() {
        return Ok(None);
    }
    Ok(Some((canon_id, list, global_filter)))
}

/// `IDtoSTV(id)`: source, target, variant, and whether the source was
/// given.
pub fn id_to_stv(id: &str) -> (String, String, String, bool) {
    let mut source = ANY.to_string();
    let target;
    let mut variant;
    let sep = id.find('-');
    let var = id.find('/').unwrap_or(id.len());
    let mut is_source_present = false;
    match sep {
        None => {
            target = id[..var].to_string();
            variant = id[var..].to_string();
        }
        Some(sep) if sep < var => {
            if sep > 0 {
                source = id[..sep].to_string();
                is_source_present = true;
            }
            target = id[sep..var].trim_start_matches('-').to_string();
            variant = id[var..].to_string();
        }
        Some(sep) => {
            if var > 0 {
                source = id[..var].to_string();
                is_source_present = true;
            }
            variant = id[var..sep].to_string();
            target = id[sep..].trim_start_matches('-').to_string();
        }
    }
    if !variant.is_empty() {
        variant.remove(0);
    }
    (source, target, variant, is_source_present)
}

/// `STVtoID(source, target, variant)`.
pub fn stv_to_id(source: &str, target: &str, variant: &str) -> String {
    let mut id = if source.is_empty() {
        ANY.to_string()
    } else {
        source.to_string()
    };
    id.push('-');
    id.push_str(target);
    if !variant.is_empty() {
        id.push('/');
        id.push_str(variant);
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert_eq!(
            id_to_stv("Latin-Greek/UNGEGN"),
            ("Latin".into(), "Greek".into(), "UNGEGN".into(), true)
        );
        assert_eq!(
            id_to_stv("Null"),
            ("Any".into(), "Null".into(), String::new(), false)
        );
        assert_eq!(
            id_to_stv("Any/Hex-Foo"),
            ("Any".into(), "Foo".into(), "Hex".into(), true)
        );
        assert_eq!(stv_to_id("", "Latin", ""), "Any-Latin");
        assert_eq!(stv_to_id("Hex", "Any", "Java"), "Hex-Any/Java");
        let (canon, list, filter) = parse_compound_id("[:Latin:] NFD; Lower", FORWARD)
            .unwrap()
            .unwrap();
        // A leading set is a global filter in the canonical ID only before
        // a ';', but Java returns it as the global filter either way.
        assert_eq!(list.len(), 2);
        assert!(filter.is_some());
        assert_eq!(canon, "[:Latin:] NFD;Lower");
        assert_eq!(list[0].basic_id, "Any-NFD");
        assert!(parse_compound_id("Latin-Greek x y", FORWARD)
            .unwrap()
            .is_none());
        assert!(parse_compound_id("", FORWARD).unwrap().is_none());
        let (canon, list, _) = parse_compound_id("Latin-Greek(Greek-Latin)", REVERSE)
            .unwrap()
            .unwrap();
        assert_eq!(canon, "Greek-Latin(Latin-Greek)");
        assert_eq!(list[0].basic_id, "Greek-Latin");
        let (_, _, filter) = parse_compound_id("Latin-Greek; ([a-z])", REVERSE)
            .unwrap()
            .unwrap();
        assert!(filter.is_some());
        register_special_inverse("TestA", "TestB", true);
        let (canon, _, _) = parse_compound_id("TestA", REVERSE).unwrap().unwrap();
        assert_eq!(canon, "TestB");
        let mut pos = 0;
        let s: Vec<u16> = "[a] Lower(".encode_utf16().collect();
        let single = parse_filter_id(&s, &mut pos).unwrap().unwrap();
        // The set pattern takes the white space after it, as in Java.
        assert_eq!(single.filter.as_deref(), Some("[a] "));
        assert!(parse_char(&s, &mut pos, '('));
        assert!(!is_id_start_char('1' as i32));
    }
}
