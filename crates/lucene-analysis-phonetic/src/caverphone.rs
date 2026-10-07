//! Commons Codec's `Caverphone1` and `Caverphone2` (1.17.2; the deprecated
//! `Caverphone` is `Caverphone2`): David Hood's sequence of rewrites.
//!
//! Java writes each step as a `String.replace` (literal, every occurrence,
//! left to right) or a `replaceAll` of an anchored or run regex (`^cough`,
//! `mb$`, `s+`). After the first two steps the text is lower-case ASCII
//! (`[^a-z]` removed), so the port runs the same steps over bytes with
//! one helper per regex shape; no step needs a regex engine.

use crate::java::{to_lower, units};

/// `String.replace(from, to)` over ASCII bytes.
fn replace(s: &mut Vec<u8>, from: &str, to: &str) {
    let (from, to) = (from.as_bytes(), to.as_bytes());
    if !s.windows(from.len()).any(|w| w == from) {
        return;
    }
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    *s = out;
}

/// `replaceAll("^from", to)`.
fn replace_start(s: &mut Vec<u8>, from: &str, to: &str) {
    if s.starts_with(from.as_bytes()) {
        s.splice(..from.len(), to.bytes());
    }
}

/// `replaceAll("from$", to)`.
fn replace_end(s: &mut Vec<u8>, from: &str, to: &str) {
    if s.ends_with(from.as_bytes()) {
        let at = s.len() - from.len();
        s.splice(at.., to.bytes());
    }
}

/// `replaceAll("c+", to)`: every run of `c` becomes `to`.
fn replace_runs(s: &mut Vec<u8>, c: u8, to: u8) {
    let mut out = Vec::with_capacity(s.len());
    let mut prev_was_c = false;
    for &b in s.iter() {
        if b == c {
            if !prev_was_c {
                out.push(to);
            }
            prev_was_c = true;
        } else {
            out.push(b);
            prev_was_c = false;
        }
    }
    *s = out;
}

fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// `replaceAll("^[aeiou]", "A")`.
fn replace_start_vowel(s: &mut [u8]) {
    if s.first().copied().is_some_and(is_vowel) {
        s[0] = b'A';
    }
}

/// `replaceAll("[aeiou]", "3")`.
fn replace_vowels(s: &mut [u8]) {
    for b in s.iter_mut().filter(|b| is_vowel(**b)) {
        *b = b'3';
    }
}

/// The common start of both versions: `toLowerCase(Locale.ENGLISH)`, then
/// `replaceAll("[^a-z]", "")`.
fn lower_letters(source: &[u16]) -> Vec<u8> {
    to_lower(source)
        .into_iter()
        .filter_map(|c| u8::try_from(c).ok().filter(u8::is_ascii_lowercase))
        .collect()
}

/// `txt + ONES` cut to `ONES.length()`.
fn pad(mut txt: Vec<u8>, len: usize) -> Vec<u16> {
    txt.resize(txt.len().max(len), b'1');
    txt.truncate(len);
    txt.into_iter().map(u16::from).collect()
}

/// `Caverphone1.encode(String)`: six characters.
pub fn caverphone1(source: &[u16]) -> Vec<u16> {
    if source.is_empty() {
        return units("111111");
    }
    let mut t = lower_letters(source);
    let t = &mut t;
    replace_start(t, "cough", "cou2f");
    replace_start(t, "rough", "rou2f");
    replace_start(t, "tough", "tou2f");
    replace_start(t, "enough", "enou2f");
    replace_start(t, "gn", "2n");
    replace_end(t, "mb", "m2");
    for (f, r) in [
        ("cq", "2q"),
        ("ci", "si"),
        ("ce", "se"),
        ("cy", "sy"),
        ("tch", "2ch"),
        ("c", "k"),
        ("q", "k"),
        ("x", "k"),
        ("v", "f"),
        ("dg", "2g"),
        ("tio", "sio"),
        ("tia", "sia"),
        ("d", "t"),
        ("ph", "fh"),
        ("b", "p"),
        ("sh", "s2"),
        ("z", "s"),
    ] {
        replace(t, f, r);
    }
    replace_start_vowel(t);
    replace_vowels(t);
    replace(t, "3gh3", "3kh3");
    replace(t, "gh", "22");
    replace(t, "g", "k");
    for (c, to) in [
        (b's', b'S'),
        (b't', b'T'),
        (b'p', b'P'),
        (b'k', b'K'),
        (b'f', b'F'),
        (b'm', b'M'),
        (b'n', b'N'),
    ] {
        replace_runs(t, c, to);
    }
    replace(t, "w3", "W3");
    replace(t, "wy", "Wy");
    replace(t, "wh3", "Wh3");
    replace(t, "why", "Why");
    replace(t, "w", "2");
    replace_start(t, "h", "A");
    for (f, r) in [
        ("h", "2"),
        ("r3", "R3"),
        ("ry", "Ry"),
        ("r", "2"),
        ("l3", "L3"),
        ("ly", "Ly"),
        ("l", "2"),
        ("j", "y"),
        ("y3", "Y3"),
        ("y", "2"),
        ("2", ""),
        ("3", ""),
    ] {
        replace(t, f, r);
    }
    pad(std::mem::take(t), 6)
}

/// `Caverphone2.encode(String)`: ten characters.
pub fn caverphone2(source: &[u16]) -> Vec<u16> {
    if source.is_empty() {
        return units("1111111111");
    }
    let mut t = lower_letters(source);
    let t = &mut t;
    replace_end(t, "e", "");
    replace_start(t, "cough", "cou2f");
    replace_start(t, "rough", "rou2f");
    replace_start(t, "tough", "tou2f");
    replace_start(t, "enough", "enou2f");
    replace_start(t, "trough", "trou2f");
    replace_start(t, "gn", "2n");
    replace_end(t, "mb", "m2");
    for (f, r) in [
        ("cq", "2q"),
        ("ci", "si"),
        ("ce", "se"),
        ("cy", "sy"),
        ("tch", "2ch"),
        ("c", "k"),
        ("q", "k"),
        ("x", "k"),
        ("v", "f"),
        ("dg", "2g"),
        ("tio", "sio"),
        ("tia", "sia"),
        ("d", "t"),
        ("ph", "fh"),
        ("b", "p"),
        ("sh", "s2"),
        ("z", "s"),
    ] {
        replace(t, f, r);
    }
    replace_start_vowel(t);
    replace_vowels(t);
    replace(t, "j", "y");
    replace_start(t, "y3", "Y3");
    replace_start(t, "y", "A");
    replace(t, "y", "3");
    replace(t, "3gh3", "3kh3");
    replace(t, "gh", "22");
    replace(t, "g", "k");
    for (c, to) in [
        (b's', b'S'),
        (b't', b'T'),
        (b'p', b'P'),
        (b'k', b'K'),
        (b'f', b'F'),
        (b'm', b'M'),
        (b'n', b'N'),
    ] {
        replace_runs(t, c, to);
    }
    replace(t, "w3", "W3");
    replace(t, "wh3", "Wh3");
    replace_end(t, "w", "3");
    replace(t, "w", "2");
    replace_start(t, "h", "A");
    replace(t, "h", "2");
    replace(t, "r3", "R3");
    replace_end(t, "r", "3");
    replace(t, "r", "2");
    replace(t, "l3", "L3");
    replace_end(t, "l", "3");
    replace(t, "l", "2");
    replace(t, "2", "");
    replace_end(t, "3", "A");
    replace(t, "3", "");
    pad(std::mem::take(t), 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::string;

    #[test]
    fn helpers() {
        let mut s = b"aaab".to_vec();
        replace_runs(&mut s, b'a', b'A');
        assert_eq!(s, b"Ab");
        let mut s = b"mbmb".to_vec();
        replace_end(&mut s, "mb", "m2");
        assert_eq!(s, b"mbm2");
        replace_start(&mut s, "x", "y");
        assert_eq!(s, b"mbm2");
        assert_eq!(string(&caverphone1(&[])), "111111");
        assert_eq!(string(&caverphone2(&[])), "1111111111");
    }
}
