//! ICU4J's character properties (`UCharacter.getIntPropertyValue`,
//! `UCharacter.getType`, `UScript.getScript`/`hasScript`/`getName`, the
//! property and value aliases of `UPropertyAliases`), read from
//! `resources/uprops.bin.z`, which `tools/GenIcuProperties.java` writes from
//! ICU4J 77.1's answers (Unicode 16.0 as ICU implements it, Unicode-3.0
//! licensed data).
//!
//! Format (big-endian, zlib-compressed): `"LIP1"`, `u16` property count;
//! per property `i32` UProperty id, `u8` kind (0 binary, 1 enumerated, 2
//! `General_Category_Mask`, 3 `Script_Extensions`, 4 any other
//! property, names only, 5 `Age`, its runs the version packed into an
//! `i32`, 6 `Numeric_Value`, each distinct double a value named by its
//! bits in hex), names (`u8` count, each
//! `u8` length and ASCII), `u16` value count with each value's `i32` and
//! names, `u32` run count and the runs (`u32` first code point, `i32`
//! value; a run lasts until the next starts), and for a binary property
//! the strings its set holds (`u32` count, each `u16` length and UTF-16
//! units: the emoji properties of strings). Then the script lists
//! `Script_Extensions` runs index (`u16` count, each `u8` length and `u16`
//! script codes), and every script code's short and long name.
//!
//! This replaces ICU's own `uprops.icu`/`pnames.icu` readers: the same
//! facts through ICU's public API, in a layout this crate looks up with a
//! binary search per property.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::icu4j::binary::ByteReader;
use crate::IcuError;

/// `UProperty.GENERAL_CATEGORY`.
pub const GENERAL_CATEGORY: i32 = 0x1005;
/// `UProperty.SCRIPT`.
pub const SCRIPT: i32 = 0x100a;
/// `UProperty.LINE_BREAK`.
pub const LINE_BREAK: i32 = 0x1008;
/// `UProperty.GENERAL_CATEGORY_MASK`.
pub const GENERAL_CATEGORY_MASK: i32 = 0x2000;
/// `UProperty.SCRIPT_EXTENSIONS`.
pub const SCRIPT_EXTENSIONS: i32 = 0x7000;
/// `UProperty.CANONICAL_COMBINING_CLASS`.
pub const CANONICAL_COMBINING_CLASS: i32 = 0x1002;
/// `UProperty.LEAD_CANONICAL_COMBINING_CLASS`.
pub const LEAD_CANONICAL_COMBINING_CLASS: i32 = 0x1010;
/// `UProperty.TRAIL_CANONICAL_COMBINING_CLASS`.
pub const TRAIL_CANONICAL_COMBINING_CLASS: i32 = 0x1011;
/// `UProperty.BINARY_START`..`BINARY_LIMIT` lies below this.
pub const INT_START: i32 = 0x1000;

/// `UCharacter.UNASSIGNED` (`Cn`).
pub const UNASSIGNED: i32 = 0;
/// `UCharacterCategory.NON_SPACING_MARK` (`Mn`).
pub const NON_SPACING_MARK: i32 = 6;
/// `ENCLOSING_MARK` (`Me`).
pub const ENCLOSING_MARK: i32 = 7;
/// `COMBINING_SPACING_MARK` (`Mc`).
pub const COMBINING_SPACING_MARK: i32 = 8;
/// `SPACE_SEPARATOR` (`Zs`).
pub const SPACE_SEPARATOR: i32 = 12;
/// `LINE_SEPARATOR` (`Zl`).
pub const LINE_SEPARATOR: i32 = 13;
/// `PARAGRAPH_SEPARATOR` (`Zp`).
pub const PARAGRAPH_SEPARATOR: i32 = 14;

/// What a property's values are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropKind {
    /// A binary property (values 0 and 1).
    Binary,
    /// An enumerated (integer) property.
    Enumerated,
    /// `General_Category_Mask`: values are category bit sets.
    GeneralCategoryMask,
    /// `Script_Extensions`: run values index the script lists.
    ScriptExtensions,
    /// A double, string or other property: only its names are known.
    Other,
    /// `Age`: run values are versions, `major << 24 | minor << 16 | milli
    /// << 8 | micro` (0 for an unassigned code point).
    Age,
    /// `Numeric_Value`: each value names the bits of one double in hex.
    NumericValue,
}

/// One property.
#[derive(Debug, Clone)]
pub struct Property {
    pub id: i32,
    pub kind: PropKind,
    pub names: Vec<String>,
    pub values: Vec<(i32, Vec<String>)>,
    starts: Vec<u32>,
    run_values: Vec<i32>,
    /// A binary property's strings (`RGI_Emoji` and the other emoji
    /// properties of strings).
    pub strings: Vec<Vec<u16>>,
}

impl Property {
    /// The value of code point `c` (0 outside `0..=0x10ffff`).
    #[inline]
    pub fn value(&self, c: i32) -> i32 {
        let Ok(c) = u32::try_from(c) else { return 0 };
        match self.starts.partition_point(|&s| s <= c).checked_sub(1) {
            Some(i) => self.run_values.get(i).copied().unwrap_or(0),
            None => 0,
        }
    }

    /// The runs: `(first, last, value)` for every run.
    pub fn runs(&self) -> impl Iterator<Item = (u32, u32, i32)> + '_ {
        self.starts.iter().enumerate().map(move |(i, &s)| {
            let end = self
                .starts
                .get(i.wrapping_add(1))
                .map_or(0x10ffff, |&n| n.saturating_sub(1));
            (s, end, self.run_values.get(i).copied().unwrap_or(0))
        })
    }
}

/// All properties.
#[derive(Debug)]
pub struct UProps {
    props: Vec<Property>,
    by_id: HashMap<i32, usize>,
    scx_lists: Vec<Vec<u16>>,
    script_names: Vec<(String, String)>,
}

const DATA: &[u8] = include_bytes!("../resources/uprops.bin.z");

fn names(r: &mut ByteReader<'_>) -> Result<Vec<String>, IcuError> {
    let n = r.u8()?;
    (0..n)
        .map(|_| {
            let len = usize::from(r.u8()?);
            Ok(String::from_utf8_lossy(r.take(len)?).into_owned())
        })
        .collect()
}

impl UProps {
    fn parse(bytes: &[u8]) -> Result<UProps, IcuError> {
        let mut r = ByteReader::new(bytes);
        if r.take(4)? != b"LIP1" {
            return Err(IcuError::new("uprops: bad magic"));
        }
        let count = r.u16()?;
        let mut props = Vec::new();
        let mut by_id = HashMap::new();
        for i in 0..count {
            let id = r.i32()?;
            let kind = match r.u8()? {
                0 => PropKind::Binary,
                1 => PropKind::Enumerated,
                2 => PropKind::GeneralCategoryMask,
                3 => PropKind::ScriptExtensions,
                5 => PropKind::Age,
                6 => PropKind::NumericValue,
                _ => PropKind::Other,
            };
            let pnames = names(&mut r)?;
            let nvalues = r.u16()?;
            let mut values = Vec::new();
            for _ in 0..nvalues {
                let v = r.i32()?;
                values.push((v, names(&mut r)?));
            }
            let nruns = r.i32()?;
            let mut starts = Vec::new();
            let mut run_values = Vec::new();
            for _ in 0..nruns {
                starts.push(r.i32()? as u32);
                run_values.push(r.i32()?);
            }
            let mut strings = Vec::new();
            if kind == PropKind::Binary {
                let n = r.i32()?;
                for _ in 0..n {
                    let len = usize::from(r.u16()?);
                    strings.push(r.u16s(len)?);
                }
            }
            by_id.insert(id, usize::from(i));
            props.push(Property {
                id,
                kind,
                names: pnames,
                values,
                starts,
                run_values,
                strings,
            });
        }
        let nlists = r.u16()?;
        let mut scx_lists = Vec::new();
        for _ in 0..nlists {
            let n = usize::from(r.u8()?);
            scx_lists.push(r.u16s(n)?);
        }
        let nscripts = r.u16()?;
        let mut script_names = Vec::new();
        for _ in 0..nscripts {
            let mut n = names(&mut r)?.into_iter();
            let short = n.next().unwrap_or_default();
            let long = n.next().unwrap_or_default();
            script_names.push((short, long));
        }
        Ok(UProps {
            props,
            by_id,
            scx_lists,
            script_names,
        })
    }

    /// The property with UProperty id `id`.
    pub fn property(&self, id: i32) -> Option<&Property> {
        self.by_id.get(&id).and_then(|&i| self.props.get(i))
    }

    /// `UPropertyAliases.getPropertyEnum(alias)`: the property a loosely
    /// matched alias names.
    pub fn property_by_alias(&self, alias: &str) -> Option<&Property> {
        self.props
            .iter()
            .find(|p| p.names.iter().any(|n| compare_names(n, alias)))
    }

    /// `UPropertyAliases.getPropertyValueEnum(prop, alias)`.
    pub fn value_by_alias(&self, prop: &Property, alias: &str) -> Option<i32> {
        prop.values
            .iter()
            .find(|(_, names)| names.iter().any(|n| compare_names(n, alias)))
            .map(|&(v, _)| v)
    }

    /// `UCharacter.getIntPropertyValue(c, prop)`.
    pub fn int_value(&self, prop: i32, c: i32) -> i32 {
        self.property(prop).map_or(0, |p| p.value(c))
    }

    /// The script codes in `c`'s `Script_Extensions` (`UScript.getScriptExtensions`).
    pub fn script_extensions(&self, c: i32) -> &[u16] {
        let i = self.int_value(SCRIPT_EXTENSIONS, c);
        usize::try_from(i)
            .ok()
            .and_then(|i| self.scx_lists.get(i))
            .map_or(&[], |l| l.as_slice())
    }

    /// The script lists `Script_Extensions` runs index.
    pub fn script_lists(&self) -> &[Vec<u16>] {
        &self.scx_lists
    }

    /// `(UScript.getShortName(code), UScript.getName(code))`.
    pub fn script_name(&self, code: i32) -> Option<(&str, &str)> {
        usize::try_from(code)
            .ok()
            .and_then(|i| self.script_names.get(i))
            .map(|(s, l)| (s.as_str(), l.as_str()))
    }
}

/// The properties, parsed on first use.
pub fn uprops() -> &'static UProps {
    static P: OnceLock<UProps> = OnceLock::new();
    P.get_or_init(|| {
        let bytes = miniz_oxide::inflate::decompress_to_vec_zlib(DATA)
            .expect("the vendored uprops.bin.z inflates");
        UProps::parse(&bytes).expect("the vendored uprops.bin.z parses")
    })
}

/// `UPropertyAliases.compare(a, b) == 0`: ASCII case-insensitive, ignoring
/// `-`, `_`, space and the ASCII white space controls.
pub fn compare_names(a: &str, b: &str) -> bool {
    fn skip(c: &u8) -> bool {
        matches!(c, b'-' | b'_' | b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
    }
    let mut x = a.bytes().filter(|c| !skip(c));
    let mut y = b.bytes().filter(|c| !skip(c));
    loop {
        match (x.next(), y.next()) {
            (None, None) => return true,
            (Some(p), Some(q)) if p.eq_ignore_ascii_case(&q) => {}
            _ => return false,
        }
    }
}

/// `UCharacter.getType(c)`.
#[inline]
pub fn char_type(c: i32) -> i32 {
    uprops().int_value(GENERAL_CATEGORY, c)
}

/// `UScript.getScript(c)`.
#[inline]
pub fn script(c: i32) -> i32 {
    uprops().int_value(SCRIPT, c)
}

/// `UScript.hasScript(c, sc)`.
pub fn has_script(c: i32, sc: i32) -> bool {
    uprops()
        .script_extensions(c)
        .iter()
        .any(|&s| i32::from(s) == sc)
}

/// `UCharacter.isWhitespace(c)`: a space separator, line or paragraph
/// separator other than the no-break spaces, or one of U+0009..U+000D and
/// U+001C..U+001F.
pub fn is_whitespace(c: i32) -> bool {
    let t = char_type(c);
    ((t == SPACE_SEPARATOR || t == LINE_SEPARATOR || t == PARAGRAPH_SEPARATOR)
        && c != 0x00a0
        && c != 0x2007
        && c != 0x202f)
        || (0x9..=0xd).contains(&c)
        || (0x1c..=0x1f).contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let p = uprops();
        assert_eq!(char_type('A' as i32), 1); // Lu
        assert_eq!(char_type(0x0301), NON_SPACING_MARK);
        assert_eq!(char_type(0x10ffff), UNASSIGNED);
        assert_eq!(char_type(-1), 0);
        assert_eq!(script('a' as i32), 25); // Latn
        assert_eq!(script(0x0e01), 38); // Thai
        assert!(has_script(0x3001, 17)); // U+3001 has Hani among its extensions
        assert!(!has_script('a' as i32, 38));
        assert_eq!(p.script_name(25), Some(("Latn", "Latin")));
        assert_eq!(p.script_name(105), Some(("Jpan", "Jpan")));
        assert_eq!(p.script_name(-1), None);
        let gc = p.property_by_alias("general category").unwrap();
        assert_eq!(gc.id, GENERAL_CATEGORY);
        assert_eq!(p.value_by_alias(gc, "uppercase_letter"), Some(1));
        assert_eq!(p.value_by_alias(gc, "nope"), None);
        assert!(p.property_by_alias("no such property").is_none());
        let lb = p.property(LINE_BREAK).unwrap();
        assert_eq!(lb.kind, PropKind::Enumerated);
        assert!(lb.runs().count() > 100);
        assert!(!p.script_lists().is_empty());
        assert!(is_whitespace(' ' as i32) && is_whitespace(0x2028) && is_whitespace(0x1f));
        assert!(!is_whitespace(0xa0) && !is_whitespace('a' as i32));
        assert!(compare_names("Line_Break", "linebreak"));
        assert!(!compare_names("lb", "lbx"));
    }

    #[test]
    fn bad_data() {
        assert!(UProps::parse(b"XXXX").is_err());
        assert!(UProps::parse(b"LIP1").is_err());
    }
}
