//! Commons Codec's `Nysiis` (1.17.2): the New York State Identification and
//! Intelligence System code, strict (six characters) or not.

use crate::soundex::clean;

const SPACE: u16 = b' ' as u16;
const TRUE_LENGTH: usize = 6;

fn c(b: u8) -> u16 {
    u16::from(b)
}

fn is_vowel(ch: u16) -> bool {
    b"AEIOU".iter().any(|&v| ch == c(v))
}

/// `Nysiis.transcodeRemaining`: the units replacing `curr` (and, for a
/// two- or three-unit result, the ones after it).
fn transcode_remaining(prev: u16, curr: u16, next: u16, a_next: u16) -> Vec<u16> {
    if curr == c(b'E') && next == c(b'V') {
        return vec![c(b'A'), c(b'F')];
    }
    if is_vowel(curr) {
        return vec![c(b'A')];
    }
    match u8::try_from(curr).unwrap_or(0) {
        b'Q' => return vec![c(b'G')],
        b'Z' => return vec![c(b'S')],
        b'M' => return vec![c(b'N')],
        b'K' => {
            return if next == c(b'N') {
                vec![c(b'N'), c(b'N')]
            } else {
                vec![c(b'C')]
            }
        }
        _ => {}
    }
    if curr == c(b'S') && next == c(b'C') && a_next == c(b'H') {
        return vec![c(b'S'), c(b'S'), c(b'S')];
    }
    if curr == c(b'P') && next == c(b'H') {
        return vec![c(b'F'), c(b'F')];
    }
    if curr == c(b'H') && (!is_vowel(prev) || !is_vowel(next)) {
        return vec![prev];
    }
    if curr == c(b'W') && is_vowel(prev) {
        return vec![prev];
    }
    vec![curr]
}

/// `PAT.matcher(str).replaceFirst(to)` for an anchored literal (`^MAC`) or
/// an alternation of them (`^(PH|PF)`, `(EE|IE)$`): the first alternative
/// that matches is replaced.
fn replace_first(s: &mut Vec<u16>, at_start: bool, alternatives: &[&str], to: &str) {
    for alt in alternatives {
        let a: Vec<u16> = alt.encode_utf16().collect();
        if at_start && s.starts_with(&a) {
            s.splice(..a.len(), to.encode_utf16());
            return;
        }
        if !at_start && s.ends_with(&a) {
            let at = s.len() - a.len();
            s.splice(at.., to.encode_utf16());
            return;
        }
    }
}

/// `org.apache.commons.codec.language.Nysiis`.
#[derive(Debug, Clone)]
pub struct Nysiis {
    strict: bool,
}

impl Default for Nysiis {
    /// `new Nysiis()`: strict.
    fn default() -> Self {
        Nysiis { strict: true }
    }
}

impl Nysiis {
    /// `new Nysiis(boolean strict)`.
    pub fn new(strict: bool) -> Self {
        Nysiis { strict }
    }

    /// `Nysiis.isStrict()`.
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// `Nysiis.nysiis(String)`.
    pub fn nysiis(&self, s: &[u16]) -> Vec<u16> {
        let mut s = clean(s);
        if s.is_empty() {
            return s;
        }
        replace_first(&mut s, true, &["MAC"], "MCC");
        replace_first(&mut s, true, &["KN"], "NN");
        replace_first(&mut s, true, &["K"], "C");
        replace_first(&mut s, true, &["PH", "PF"], "FF");
        replace_first(&mut s, true, &["SCH"], "SSS");
        replace_first(&mut s, false, &["EE", "IE"], "Y");
        replace_first(&mut s, false, &["DT", "RT", "RD", "NT", "ND"], "D");
        let mut key = vec![s[0]];
        let mut chars = s;
        let len = chars.len();
        for i in 1..len {
            let next = if i + 1 < len { chars[i + 1] } else { SPACE };
            let a_next = if i + 2 < len { chars[i + 2] } else { SPACE };
            let transcoded = transcode_remaining(chars[i - 1], chars[i], next, a_next);
            chars[i..i + transcoded.len()].copy_from_slice(&transcoded);
            if chars[i] != chars[i - 1] {
                key.push(chars[i]);
            }
        }
        if key.len() > 1 {
            let mut last_char = key[key.len() - 1];
            if last_char == c(b'S') {
                key.pop();
                last_char = key[key.len() - 1];
            }
            if key.len() > 2 {
                let last2 = key[key.len() - 2];
                if last2 == c(b'A') && last_char == c(b'Y') {
                    key.remove(key.len() - 2);
                }
            }
            if last_char == c(b'A') {
                key.pop();
            }
        }
        if self.strict {
            key.truncate(TRUE_LENGTH);
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    #[test]
    fn flags_and_empty() {
        assert!(Nysiis::default().is_strict());
        assert!(!Nysiis::new(false).is_strict());
        assert_eq!(string(&Nysiis::default().nysiis(&units("12"))), "");
    }
}
