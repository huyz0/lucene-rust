//! Commons Codec's `Metaphone` (1.17.2): Lawrence Philips' original
//! algorithm, rule for rule over Java `char`s.

use crate::java::to_upper;

const VOWELS: &[u8] = b"AEIOU";
const FRONTV: &[u8] = b"EIY";
const VARSON: &[u8] = b"CSPTG";

fn is_in(set: &[u8], c: u16) -> bool {
    set.iter().any(|&b| u16::from(b) == c)
}

/// `org.apache.commons.codec.language.Metaphone`.
#[derive(Debug, Clone)]
pub struct Metaphone {
    max_code_len: i32,
}

impl Default for Metaphone {
    fn default() -> Self {
        Metaphone { max_code_len: 4 }
    }
}

/// `local.charAt(i)` where Java would throw only if the algorithm were
/// wrong: every call site is bounds-checked by the rule before it.
fn at(s: &[u16], i: usize) -> u16 {
    s[i]
}

impl Metaphone {
    /// `Metaphone.getMaxCodeLen()`.
    pub fn max_code_len(&self) -> i32 {
        self.max_code_len
    }

    /// `Metaphone.setMaxCodeLen(int)`: any value; zero or less yields `""`.
    pub fn set_max_code_len(&mut self, max_code_len: i32) {
        self.max_code_len = max_code_len;
    }

    // Java: Metaphone.isPreviousChar
    fn is_previous_char(s: &[u16], index: usize, c: u8) -> bool {
        index > 0 && index < s.len() && s[index - 1] == u16::from(c)
    }

    // Java: Metaphone.isNextChar
    fn is_next_char(s: &[u16], index: usize, c: u8) -> bool {
        index + 1 < s.len() && s[index + 1] == u16::from(c)
    }

    // Java: Metaphone.regionMatch
    fn region_match(s: &[u16], index: usize, test: &str) -> bool {
        let t: Vec<u16> = test.encode_utf16().collect();
        s.get(index..index + t.len()) == Some(&t[..])
    }

    // Java: Metaphone.isLastChar
    fn is_last_char(wdsz: usize, n: usize) -> bool {
        n + 1 == wdsz
    }

    // Java: Metaphone.isVowel
    fn is_vowel(s: &[u16], index: usize) -> bool {
        is_in(VOWELS, at(s, index))
    }

    /// `Metaphone.metaphone(String)`.
    // Java's rules that drop a letter are kept as separate empty branches.
    #[allow(clippy::if_same_then_else)]
    pub fn metaphone(&self, txt: &[u16]) -> Vec<u16> {
        if txt.is_empty() {
            return Vec::new();
        }
        if txt.len() == 1 {
            return to_upper(txt);
        }
        let mut inwd = to_upper(txt);
        let c = |b: u8| u16::from(b);
        // Java: the switch on the first letter (KN/GN/PN, AE, WR/WH, X).
        let mut local: Vec<u16> = match inwd[0] {
            x if x == c(b'K') || x == c(b'G') || x == c(b'P') => {
                if inwd[1] == c(b'N') {
                    inwd[1..].to_vec()
                } else {
                    inwd
                }
            }
            x if x == c(b'A') => {
                if inwd[1] == c(b'E') {
                    inwd[1..].to_vec()
                } else {
                    inwd
                }
            }
            x if x == c(b'W') => {
                if inwd[1] == c(b'R') {
                    inwd[1..].to_vec()
                } else if inwd[1] == c(b'H') {
                    let mut l = inwd[1..].to_vec();
                    l[0] = c(b'W');
                    l
                } else {
                    inwd
                }
            }
            x if x == c(b'X') => {
                inwd[0] = c(b'S');
                inwd
            }
            _ => inwd,
        };
        let local = &mut local[..];
        let wdsz = local.len();
        let mut code: Vec<u16> = Vec::new();
        let max = self.max_code_len;
        let below_max =
            |code: &Vec<u16>| i64::try_from(code.len()).unwrap_or(i64::MAX) < i64::from(max);
        let mut n = 0usize;
        while below_max(&code) && n < wdsz {
            let symb = local[n];
            if symb != c(b'C') && n > 0 && local[n - 1] == symb {
                // A doubled letter but C is skipped.
            } else {
                match symb as u8 {
                    _ if symb > 0x7F => {}
                    b'A' | b'E' | b'I' | b'O' | b'U' => {
                        if n == 0 {
                            code.push(symb);
                        }
                    }
                    b'B' => {
                        if !(Self::is_previous_char(local, n, b'M') && Self::is_last_char(wdsz, n))
                        {
                            code.push(symb);
                        }
                    }
                    b'C' => {
                        if Self::is_previous_char(local, n, b'S')
                            && !Self::is_last_char(wdsz, n)
                            && is_in(FRONTV, local[n + 1])
                        {
                            // SCI, SCE, SCY: dropped.
                        } else if Self::region_match(local, n, "CIA") {
                            code.push(c(b'X'));
                        } else if !Self::is_last_char(wdsz, n) && is_in(FRONTV, local[n + 1]) {
                            code.push(c(b'S'));
                        } else if Self::is_previous_char(local, n, b'S')
                            && Self::is_next_char(local, n, b'H')
                        {
                            code.push(c(b'K'));
                        } else if !Self::is_next_char(local, n, b'H')
                            || (n == 0 && wdsz >= 3 && Self::is_vowel(local, 2))
                        {
                            code.push(c(b'K'));
                        } else {
                            code.push(c(b'X'));
                        }
                    }
                    b'D' => {
                        if !Self::is_last_char(wdsz, n + 1)
                            && Self::is_next_char(local, n, b'G')
                            && is_in(FRONTV, local[n + 2])
                        {
                            code.push(c(b'J'));
                            n += 2;
                        } else {
                            code.push(c(b'T'));
                        }
                    }
                    b'G' => {
                        if Self::is_last_char(wdsz, n + 1) && Self::is_next_char(local, n, b'H') {
                        } else if !Self::is_last_char(wdsz, n + 1)
                            && Self::is_next_char(local, n, b'H')
                            && !Self::is_vowel(local, n + 2)
                        {
                        } else if n > 0
                            && (Self::region_match(local, n, "GN")
                                || Self::region_match(local, n, "GNED"))
                        {
                        } else {
                            let hard = Self::is_previous_char(local, n, b'G');
                            if !Self::is_last_char(wdsz, n) && is_in(FRONTV, local[n + 1]) && !hard
                            {
                                code.push(c(b'J'));
                            } else {
                                code.push(c(b'K'));
                            }
                        }
                    }
                    b'H' => {
                        if Self::is_last_char(wdsz, n) || (n > 0 && is_in(VARSON, local[n - 1])) {
                        } else if Self::is_vowel(local, n + 1) {
                            code.push(c(b'H'));
                        }
                    }
                    b'F' | b'J' | b'L' | b'M' | b'N' | b'R' => code.push(symb),
                    b'K' => {
                        if n == 0 || !Self::is_previous_char(local, n, b'C') {
                            code.push(symb);
                        }
                    }
                    b'P' => {
                        if Self::is_next_char(local, n, b'H') {
                            code.push(c(b'F'));
                        } else {
                            code.push(symb);
                        }
                    }
                    b'Q' => code.push(c(b'K')),
                    b'S' => {
                        if Self::region_match(local, n, "SH")
                            || Self::region_match(local, n, "SIO")
                            || Self::region_match(local, n, "SIA")
                        {
                            code.push(c(b'X'));
                        } else {
                            code.push(c(b'S'));
                        }
                    }
                    b'T' => {
                        if Self::region_match(local, n, "TIA")
                            || Self::region_match(local, n, "TIO")
                        {
                            code.push(c(b'X'));
                        } else if Self::region_match(local, n, "TCH") {
                        } else if Self::region_match(local, n, "TH") {
                            code.push(c(b'0'));
                        } else {
                            code.push(c(b'T'));
                        }
                    }
                    b'V' => code.push(c(b'F')),
                    b'W' | b'Y' => {
                        if !Self::is_last_char(wdsz, n) && Self::is_vowel(local, n + 1) {
                            code.push(symb);
                        }
                    }
                    b'X' => {
                        code.push(c(b'K'));
                        code.push(c(b'S'));
                    }
                    b'Z' => code.push(c(b'S')),
                    _ => {}
                }
            }
            n += 1;
            // Java: `if (code.length() > getMaxCodeLen()) code.setLength(...)`;
            // only reachable for a positive maximum (the loop runs only below it).
            if let Ok(m) = usize::try_from(max) {
                code.truncate(m);
            }
        }
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    fn m(s: &str) -> String {
        string(&Metaphone::default().metaphone(&units(s)))
    }

    #[test]
    fn metaphone_rules() {
        assert_eq!(m(""), "");
        assert_eq!(m("a"), "A");
        assert_eq!(m("ß"), "SS");
        assert_eq!(m("Knight"), "NT");
        assert_eq!(m("aero"), "ER");
        assert_eq!(m("wright"), "RT");
        assert_eq!(m("white"), "WT");
        assert_eq!(m("xavier"), "SFR");
        assert_eq!(m("science"), "SNS");
        assert_eq!(m("dumb"), "TM");
        assert_eq!(m("judge"), "JJ");
        assert_eq!(m("school"), "SKL");
        assert_eq!(m("character"), "KRKT");
        assert_eq!(m("church"), "KRX");
        assert_eq!(m("Thomas"), "0MS");
        assert_eq!(m("nation"), "NXN");
        assert_eq!(m("phone"), "FN");
        assert_eq!(m("queen"), "KN");
        assert_eq!(m("box"), "BKS");
        assert_eq!(m("zebra"), "SBR");
        assert_eq!(m("gnome"), "NM");
        assert_eq!(m("signed"), "SNT");
        assert_eq!(m("ghost"), "KST");
        assert_eq!(m("high"), "H");
        assert_eq!(m("agh"), "A");
        assert_eq!(m("bigger"), "BKR");
        assert_eq!(m("ginger"), "JNJR");
        assert_eq!(m("yes"), "YS");
        assert_eq!(m("dogs"), "TKS");
        assert_eq!(m("watch"), "WX");
        assert_eq!(m("kick"), "KK");
        assert_eq!(m("Ätna"), "TN");
        let mut long = Metaphone::default();
        long.set_max_code_len(8);
        assert_eq!(long.max_code_len(), 8);
        assert_eq!(string(&long.metaphone(&units("character"))), "KRKTR");
        long.set_max_code_len(1);
        assert_eq!(string(&long.metaphone(&units("xavier"))), "S");
        long.set_max_code_len(0);
        assert_eq!(string(&long.metaphone(&units("xavier"))), "");
        long.set_max_code_len(-3);
        assert_eq!(string(&long.metaphone(&units("xavier"))), "");
    }
}
