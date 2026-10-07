//! Commons Codec's `ColognePhonetic` (1.17.2): the Kölner Phonetik digit
//! code, over Java `char`s.

use crate::java::to_upper;

const CHAR_IGNORE: u16 = b'-' as u16;

fn in_set(set: &[u8], c: u16) -> bool {
    set.iter().any(|&b| u16::from(b) == c)
}

/// `ColognePhonetic.CologneOutputBuffer`.
struct Output {
    data: Vec<u16>,
    last_code: u16,
}

impl Output {
    // Java: CologneOutputBuffer.put
    fn put(&mut self, code: u8) {
        let code = u16::from(code);
        if code != CHAR_IGNORE
            && self.last_code != code
            && (code != u16::from(b'0') || self.data.is_empty())
        {
            self.data.push(code);
        }
        self.last_code = code;
    }
}

/// `ColognePhonetic.colognePhonetic(String)`.
pub fn cologne_phonetic(text: &[u16]) -> Vec<u16> {
    // Java: preprocess -- upper case (Locale.GERMAN: no tailoring), umlauts
    // to their base letters.
    let input: Vec<u16> = to_upper(text)
        .into_iter()
        .map(|c| match c {
            0xC4 => u16::from(b'A'),
            0xDC => u16::from(b'U'),
            0xD6 => u16::from(b'O'),
            c => c,
        })
        .collect();
    let mut output = Output {
        data: Vec::with_capacity(input.len() * 2),
        last_code: u16::from(b'/'),
    };
    let mut last_char = CHAR_IGNORE;
    for (i, &chr) in input.iter().enumerate() {
        let next_char = input.get(i + 1).copied().unwrap_or(CHAR_IGNORE);
        if !(u16::from(b'A')..=u16::from(b'Z')).contains(&chr) {
            continue;
        }
        let is = |b: u8| chr == u16::from(b);
        if in_set(b"AEIJOUY", chr) {
            output.put(b'0');
        } else if is(b'B') || is(b'P') && next_char != u16::from(b'H') {
            output.put(b'1');
        } else if (is(b'D') || is(b'T')) && !in_set(b"CSZ", next_char) {
            output.put(b'2');
        } else if in_set(b"FPVW", chr) {
            output.put(b'3');
        } else if in_set(b"GKQ", chr) {
            output.put(b'4');
        } else if is(b'X') && !in_set(b"CKQ", last_char) {
            output.put(b'4');
            output.put(b'8');
        } else if is(b'S') || is(b'Z') {
            output.put(b'8');
        } else if is(b'C') {
            if output.data.is_empty() {
                if in_set(b"AHKLOQRUX", next_char) {
                    output.put(b'4');
                } else {
                    output.put(b'8');
                }
            } else if in_set(b"SZ", last_char) || !in_set(b"AHKOQUX", next_char) {
                output.put(b'8');
            } else {
                output.put(b'4');
            }
        } else if in_set(b"DTX", chr) {
            output.put(b'8');
        } else if is(b'R') {
            output.put(b'7');
        } else if is(b'L') {
            output.put(b'5');
        } else if is(b'M') || is(b'N') {
            output.put(b'6');
        } else if is(b'H') {
            output.put(b'-');
        }
        last_char = chr;
    }
    output.data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    #[test]
    fn empty_and_ignored() {
        assert_eq!(string(&cologne_phonetic(&units(""))), "");
        assert_eq!(string(&cologne_phonetic(&units("123 ?"))), "");
    }
}
