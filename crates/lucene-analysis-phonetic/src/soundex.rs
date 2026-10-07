//! Commons Codec's Soundex family: `SoundexUtils`, `Soundex` and
//! `RefinedSoundex` (`org.apache.commons.codec.language`, 1.17.2).

use crate::java::{is_letter, to_upper, to_upper_char};
use crate::EncoderError;

/// `SoundexUtils.clean`: the letters (`Character.isLetter(char)`) of `s`,
/// upper-cased in `Locale.ENGLISH` (which may lengthen them: `ß` -> `SS`).
pub fn clean(s: &[u16]) -> Vec<u16> {
    if s.is_empty() {
        return Vec::new();
    }
    // ASCII: the letters are A-Z/a-z and upper-case to A-Z, one pass.
    if s.iter().all(|&c| c < 0x80) {
        return s
            .iter()
            .filter(|&&c| (c as u8).is_ascii_alphabetic())
            .map(|&c| u16::from((c as u8).to_ascii_uppercase()))
            .collect();
    }
    let letters: Vec<u16> = s.iter().copied().filter(|&c| is_letter(c)).collect();
    to_upper(&letters)
}

/// `Soundex.SILENT_MARKER`.
pub const SILENT_MARKER: u16 = b'-' as u16;
/// `Soundex.US_ENGLISH_MAPPING_STRING`.
pub const US_ENGLISH_MAPPING: &str = "01230120022455012623010202";
/// The mapping of `Soundex.US_ENGLISH_GENEALOGY`.
pub const US_ENGLISH_GENEALOGY_MAPPING: &str = "-123-12--22455-12623-1-2-2";

/// `org.apache.commons.codec.language.Soundex`.
#[derive(Debug, Clone)]
pub struct Soundex {
    mapping: Vec<u16>,
    special_case_hw: bool,
}

impl Default for Soundex {
    /// `new Soundex()`: the US English mapping, `H`/`W` special-cased.
    fn default() -> Self {
        Soundex {
            mapping: US_ENGLISH_MAPPING.encode_utf16().collect(),
            special_case_hw: true,
        }
    }
}

impl Soundex {
    /// `new Soundex(String mapping)`: `H`/`W` are special-cased unless the
    /// mapping holds a [`SILENT_MARKER`].
    pub fn with_mapping(mapping: &str) -> Self {
        let mapping: Vec<u16> = mapping.encode_utf16().collect();
        let special_case_hw = !mapping.contains(&SILENT_MARKER);
        Soundex {
            mapping,
            special_case_hw,
        }
    }

    /// `new Soundex(String mapping, boolean specialCaseHW)`.
    pub fn with_mapping_hw(mapping: &str, special_case_hw: bool) -> Self {
        Soundex {
            mapping: mapping.encode_utf16().collect(),
            special_case_hw,
        }
    }

    /// `Soundex.US_ENGLISH_SIMPLIFIED`.
    pub fn us_english_simplified() -> Self {
        Self::with_mapping_hw(US_ENGLISH_MAPPING, false)
    }

    /// `Soundex.US_ENGLISH_GENEALOGY`.
    pub fn us_english_genealogy() -> Self {
        Self::with_mapping(US_ENGLISH_GENEALOGY_MAPPING)
    }

    // Java: Soundex.map
    fn map(&self, ch: u16) -> Result<u16, EncoderError> {
        let index = i32::from(ch) - i32::from(b'A');
        match usize::try_from(index)
            .ok()
            .and_then(|i| self.mapping.get(i))
        {
            Some(&d) => Ok(d),
            None => Err(EncoderError::illegal_argument(format!(
                "The character is not mapped: {} (index={index})",
                String::from_utf16_lossy(&[ch])
            ))),
        }
    }

    /// `Soundex.soundex(String)`: the first letter and three digits; throws
    /// (`IllegalArgumentException`) on a letter outside `A`-`Z` after
    /// cleaning.
    pub fn soundex(&self, s: &[u16]) -> Result<Vec<u16>, EncoderError> {
        let s = clean(s);
        if s.is_empty() {
            return Ok(s);
        }
        let mut out = [b'0' as u16; 4];
        let mut count = 0;
        let first = s[0];
        out[count] = first;
        count += 1;
        let mut last_digit = self.map(first)?;
        let mut i = 1;
        while i < s.len() && count < out.len() {
            let ch = s[i];
            i += 1;
            if self.special_case_hw && (ch == u16::from(b'H') || ch == u16::from(b'W')) {
                continue;
            }
            let digit = self.map(ch)?;
            if digit == SILENT_MARKER {
                continue;
            }
            if digit != u16::from(b'0') && digit != last_digit {
                out[count] = digit;
                count += 1;
            }
            last_digit = digit;
        }
        Ok(out.to_vec())
    }
}

/// `RefinedSoundex.US_ENGLISH_MAPPING_STRING`.
pub const REFINED_US_ENGLISH_MAPPING: &str = "01360240043788015936020505";

/// `org.apache.commons.codec.language.RefinedSoundex`.
#[derive(Debug, Clone)]
pub struct RefinedSoundex {
    mapping: Vec<u16>,
}

impl Default for RefinedSoundex {
    fn default() -> Self {
        Self::with_mapping(REFINED_US_ENGLISH_MAPPING)
    }
}

impl RefinedSoundex {
    /// `new RefinedSoundex(String mapping)`.
    pub fn with_mapping(mapping: &str) -> Self {
        RefinedSoundex {
            mapping: mapping.encode_utf16().collect(),
        }
    }

    // Java: RefinedSoundex.getMappingCode; 0 for "no code".
    fn mapping_code(&self, c: u16) -> u16 {
        // ASCII letters (what clean() leaves of most words): no table lookups.
        if let Some(i) = u8::try_from(c).ok().filter(u8::is_ascii_alphabetic) {
            let index = usize::from(i.to_ascii_uppercase() - b'A');
            return self.mapping.get(index).copied().unwrap_or(0);
        }
        if !is_letter(c) {
            return 0;
        }
        let index = i32::from(to_upper_char(c)) - i32::from(b'A');
        usize::try_from(index)
            .ok()
            .and_then(|i| self.mapping.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// `RefinedSoundex.soundex(String)`.
    pub fn soundex(&self, s: &[u16]) -> Vec<u16> {
        let s = clean(s);
        if s.is_empty() {
            return s;
        }
        let mut out = vec![s[0]];
        let mut last = u16::from(b'*');
        for &c in &s {
            let current = self.mapping_code(c);
            if current == last {
                continue;
            }
            if current != 0 {
                out.push(current);
            }
            last = current;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    fn sx(e: &Soundex, s: &str) -> String {
        string(&e.soundex(&units(s)).unwrap())
    }

    #[test]
    fn soundex_codes() {
        let e = Soundex::default();
        assert_eq!(sx(&e, "Robert"), "R163");
        assert_eq!(sx(&e, "Ashcraft"), "A261");
        assert_eq!(sx(&e, "Tymczak"), "T522");
        assert_eq!(sx(&e, "Pfister"), "P236");
        assert_eq!(sx(&e, "a"), "A000");
        assert_eq!(sx(&e, "12 !"), "");
        assert_eq!(sx(&e, ""), "");
        assert_eq!(sx(&e, "straße"), "S362");
        let err = e.soundex(&units("Émile")).unwrap_err();
        assert_eq!(err.java_class(), "IllegalArgumentException");
        assert!(err
            .message()
            .starts_with("The character is not mapped: É (index=136)"));
        assert_eq!(sx(&Soundex::us_english_simplified(), "Ashcraft"), "A226");
        assert_eq!(sx(&Soundex::us_english_genealogy(), "Heywood"), "H300");
        assert!(!Soundex::us_english_genealogy().special_case_hw);
        assert!(Soundex::with_mapping(US_ENGLISH_MAPPING).special_case_hw);
        // A mapping too short for the letter.
        assert!(Soundex::with_mapping("01").soundex(&units("az")).is_err());
    }

    #[test]
    fn refined_codes() {
        let e = RefinedSoundex::default();
        let r = |s: &str| string(&e.soundex(&units(s)));
        assert_eq!(r("testing"), "T6036084");
        assert_eq!(r("TESTING"), "T6036084");
        assert_eq!(r("The"), "T60");
        assert_eq!(r("Éa"), "É0");
        assert_eq!(r(""), "");
        assert_eq!(r("123"), "");
        assert_eq!(
            string(&RefinedSoundex::with_mapping("0").soundex(&units("ab"))),
            "A0"
        );
    }
}
