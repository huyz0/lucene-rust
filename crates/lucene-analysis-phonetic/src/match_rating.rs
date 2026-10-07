//! Commons Codec's `MatchRatingApproachEncoder` (1.17.2): the Western
//! Airlines Match Rating Approach name code.

use crate::java::{is_regex_space, to_upper};

/// `PLAIN_ASCII`, index for index with [`UNICODE`].
const PLAIN_ASCII: &str = "AaEeIiOoUuAaEeIiOoUuYyAaEeIiOoUuYyAaOoNnAaEeIiOoUuYyAaCcOoUu";
/// `UNICODE`: the accented letters `removeAccents` replaces.
const UNICODE: &str = "\u{C0}\u{E0}\u{C8}\u{E8}\u{CC}\u{EC}\u{D2}\u{F2}\u{D9}\u{F9}\
\u{C1}\u{E1}\u{C9}\u{E9}\u{CD}\u{ED}\u{D3}\u{F3}\u{DA}\u{FA}\u{DD}\u{FD}\
\u{C2}\u{E2}\u{CA}\u{EA}\u{CE}\u{EE}\u{D4}\u{F4}\u{DB}\u{FB}\u{176}\u{177}\
\u{C3}\u{E3}\u{D5}\u{F5}\u{D1}\u{F1}\
\u{C4}\u{E4}\u{CB}\u{EB}\u{CF}\u{EF}\u{D6}\u{F6}\u{DC}\u{FC}\u{178}\u{FF}\
\u{C5}\u{E5}\u{C7}\u{E7}\u{150}\u{151}\u{170}\u{171}";

fn c(b: u8) -> u16 {
    u16::from(b)
}

/// `MatchRatingApproachEncoder.cleanName`: upper case, `-&'.,` removed,
/// accents removed, whitespace (`\s`) removed.
fn clean_name(name: &[u16]) -> Vec<u16> {
    let upper = to_upper(name);
    let plain: Vec<u16> = PLAIN_ASCII.encode_utf16().collect();
    let accented: Vec<u16> = UNICODE.encode_utf16().collect();
    upper
        .into_iter()
        .filter(|&ch| !b"-&'.,".iter().any(|&b| ch == c(b)))
        .map(|ch| match accented.iter().position(|&a| a == ch) {
            Some(pos) => plain[pos],
            None => ch,
        })
        .filter(|&ch| !is_regex_space(ch))
        .collect()
}

/// `MatchRatingApproachEncoder.removeVowels`: the upper-case vowels removed
/// except a leading one (`\s{2,}\b` has nothing to match after `cleanName`).
fn remove_vowels(name: &[u16]) -> Vec<u16> {
    let first = name[0];
    let mut out: Vec<u16> = name
        .iter()
        .copied()
        .filter(|&ch| !b"AEIOU".iter().any(|&v| ch == c(v)))
        .collect();
    // Java: isVowel(firstLetter), `equalsIgnoreCase` against A E I O U:
    // equal after `toUpperCase`, or after `toLowerCase(toUpperCase(c))`.
    let upper = crate::java::to_upper_char(first);
    let lower = crate::java::to_lower_char(upper);
    let is_vowel = b"AEIOU"
        .iter()
        .any(|&v| upper == c(v) || lower == c(v.to_ascii_lowercase()));
    if is_vowel {
        out.insert(0, first);
    }
    out
}

/// `MatchRatingApproachEncoder.removeDoubleConsonants`.
fn remove_double_consonants(name: &[u16]) -> Vec<u16> {
    let mut replaced = to_upper(name);
    for b in b"BCDFGHJKLMNPQRSTVWXYZ" {
        let dc = [c(*b), c(*b)];
        if replaced.windows(2).any(|w| w == dc) {
            replaced = crate::java::replace(&replaced, &dc, &dc[..1]);
        }
    }
    replaced
}

/// `MatchRatingApproachEncoder.getFirst3Last3`.
fn first3_last3(name: Vec<u16>) -> Vec<u16> {
    let n = name.len();
    if n > 6 {
        let mut out = name[..3].to_vec();
        out.extend_from_slice(&name[n - 3..]);
        out
    } else {
        name
    }
}

/// `MatchRatingApproachEncoder.encode(String)`.
pub fn match_rating_encode(name: &[u16]) -> Vec<u16> {
    let space = [c(b' ')];
    // Java: EMPTY.equalsIgnoreCase(name) || SPACE.equalsIgnoreCase(name) || length 1.
    if name.is_empty() || name == space || name.len() == 1 {
        return Vec::new();
    }
    let name = clean_name(name);
    if name == space || name.is_empty() {
        return Vec::new();
    }
    let name = remove_vowels(&name);
    if name == space || name.is_empty() {
        return Vec::new();
    }
    first3_last3(remove_double_consonants(&name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    #[test]
    fn short_names() {
        let e = |s: &str| string(&match_rating_encode(&units(s)));
        assert_eq!(e(""), "");
        assert_eq!(e(" "), "");
        assert_eq!(e("a"), "");
        assert_eq!(e("--"), "");
        assert_eq!(e("ea"), "E");
    }
}
