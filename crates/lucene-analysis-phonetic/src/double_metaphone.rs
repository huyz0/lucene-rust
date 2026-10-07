//! Commons Codec's `DoubleMetaphone` (1.17.2): Lawrence Philips' second
//! algorithm, a primary and an alternate code, handler for handler.
//!
//! Indices are `isize` because the rules probe left of the current position
//! (`contains(value, index - 2, ...)`), which Java answers `false` for a
//! negative start and `charAt` answers `'\0'` for any out-of-range index.

use crate::java::{to_upper, trim};

const VOWELS: &[u8] = b"AEIOUY";
const SILENT_START: [&str; 5] = ["GN", "KN", "PN", "WR", "PS"];
const L_R_N_M_B_H_F_V_W_SPACE: &[&str] = &["L", "R", "N", "M", "B", "H", "F", "V", "W", " "];
const ES_EP_EB_EL_EY_IB_IL_IN_IE_EI_ER: &[&str] = &[
    "ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN", "IE", "EI", "ER",
];
const L_T_K_S_N_M_B_Z: &[&str] = &["L", "T", "K", "S", "N", "M", "B", "Z"];

/// `DoubleMetaphone.DoubleMetaphoneResult`: the two codes, each capped at
/// the maximum length.
struct DmResult {
    primary: Vec<u16>,
    alternate: Vec<u16>,
    max_length: usize,
}

impl DmResult {
    fn append_primary(&mut self, v: &str) {
        let add = self.max_length.saturating_sub(self.primary.len());
        self.primary.extend(v.encode_utf16().take(add));
    }
    fn append_alternate(&mut self, v: &str) {
        let add = self.max_length.saturating_sub(self.alternate.len());
        self.alternate.extend(v.encode_utf16().take(add));
    }
    fn append(&mut self, v: &str) {
        self.append_primary(v);
        self.append_alternate(v);
    }
    fn append2(&mut self, primary: &str, alternate: &str) {
        self.append_primary(primary);
        self.append_alternate(alternate);
    }
    fn is_complete(&self) -> bool {
        self.primary.len() >= self.max_length && self.alternate.len() >= self.max_length
    }
}

/// `org.apache.commons.codec.language.DoubleMetaphone`.
#[derive(Debug, Clone)]
pub struct DoubleMetaphone {
    max_code_len: i32,
}

impl Default for DoubleMetaphone {
    fn default() -> Self {
        DoubleMetaphone { max_code_len: 4 }
    }
}

/// The upper-cased word the handlers read.
struct Word<'a>(&'a [u16]);

impl Word<'_> {
    fn len(&self) -> isize {
        self.0.len() as isize
    }

    /// `DoubleMetaphone.charAt`: `'\0'` out of range.
    fn at(&self, index: isize) -> u8 {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.0.get(i))
            .map_or(0, |&c| if c < 0x80 { c as u8 } else { 0xFF })
    }

    /// The unit at `index` as Java sees it (for the non-ASCII letters).
    fn unit(&self, index: isize) -> u16 {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.0.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// `DoubleMetaphone.contains(value, start, length, criteria...)`.
    fn contains(&self, start: isize, length: isize, criteria: &[&str]) -> bool {
        let (Ok(s), Ok(l)) = (usize::try_from(start), usize::try_from(length)) else {
            return false;
        };
        let Some(target) = self.0.get(s..s + l) else {
            return false;
        };
        criteria
            .iter()
            .any(|c| c.len() == l && c.bytes().zip(target).all(|(b, &t)| u16::from(b) == t))
    }

    fn is_vowel(c: u8) -> bool {
        VOWELS.contains(&c)
    }
}

impl DoubleMetaphone {
    /// `DoubleMetaphone.getMaxCodeLen()`.
    pub fn max_code_len(&self) -> i32 {
        self.max_code_len
    }

    /// `DoubleMetaphone.setMaxCodeLen(int)`.
    pub fn set_max_code_len(&mut self, max_code_len: i32) {
        self.max_code_len = max_code_len;
    }

    /// `DoubleMetaphone.doubleMetaphone(String, boolean)`: `None` for a word
    /// that is empty after `trim()` (Java's `null`), and `None` too for a
    /// negative maximum length, where Java's `new StringBuilder(max)` throws
    /// `NegativeArraySizeException` (both make `PhoneticFilter` keep the
    /// token).
    pub fn double_metaphone(&self, value: &[u16], alternate: bool) -> Option<Vec<u16>> {
        // Java: cleanInput
        let value = trim(value);
        if value.is_empty() {
            return None;
        }
        let max_length = usize::try_from(self.max_code_len).ok()?;
        let upper = to_upper(value);
        let v = Word(&upper);
        let slavo_germanic = Self::is_slavo_germanic(&upper);
        let mut index: isize = if Self::is_silent_start(&upper) { 1 } else { 0 };
        let mut result = DmResult {
            primary: Vec::new(),
            alternate: Vec::new(),
            max_length,
        };
        while !result.is_complete() && index < v.len() {
            let unit = v.unit(index);
            index = match unit {
                0xC7 => {
                    // Ç
                    result.append("S");
                    index + 1
                }
                0xD1 => {
                    // Ñ
                    result.append("N");
                    index + 1
                }
                _ => match v.at(index) {
                    b'A' | b'E' | b'I' | b'O' | b'U' | b'Y' => {
                        if index == 0 {
                            result.append("A");
                        }
                        index + 1
                    }
                    b'B' => {
                        result.append("P");
                        Self::skip_double(&v, index, b'B')
                    }
                    b'C' => Self::handle_c(&v, &mut result, index),
                    b'D' => Self::handle_d(&v, &mut result, index),
                    b'F' => {
                        result.append("F");
                        Self::skip_double(&v, index, b'F')
                    }
                    b'G' => Self::handle_g(&v, &mut result, index, slavo_germanic),
                    b'H' => Self::handle_h(&v, &mut result, index),
                    b'J' => Self::handle_j(&v, &mut result, index, slavo_germanic),
                    b'K' => {
                        result.append("K");
                        Self::skip_double(&v, index, b'K')
                    }
                    b'L' => Self::handle_l(&v, &mut result, index),
                    b'M' => {
                        result.append("M");
                        if Self::condition_m0(&v, index) {
                            index + 2
                        } else {
                            index + 1
                        }
                    }
                    b'N' => {
                        result.append("N");
                        Self::skip_double(&v, index, b'N')
                    }
                    b'P' => Self::handle_p(&v, &mut result, index),
                    b'Q' => {
                        result.append("K");
                        Self::skip_double(&v, index, b'Q')
                    }
                    b'R' => Self::handle_r(&v, &mut result, index, slavo_germanic),
                    b'S' => Self::handle_s(&v, &mut result, index, slavo_germanic),
                    b'T' => Self::handle_t(&v, &mut result, index),
                    b'V' => {
                        result.append("F");
                        Self::skip_double(&v, index, b'V')
                    }
                    b'W' => Self::handle_w(&v, &mut result, index),
                    b'X' => Self::handle_x(&v, &mut result, index),
                    b'Z' => Self::handle_z(&v, &mut result, index, slavo_germanic),
                    _ => index + 1,
                },
            };
        }
        Some(if alternate {
            result.alternate
        } else {
            result.primary
        })
    }

    /// `index + 2` when the next letter repeats `c`, else `index + 1`.
    fn skip_double(v: &Word, index: isize, c: u8) -> isize {
        if v.at(index + 1) == c {
            index + 2
        } else {
            index + 1
        }
    }

    // Java: DoubleMetaphone.conditionC0
    fn condition_c0(v: &Word, index: isize) -> bool {
        if v.contains(index, 4, &["CHIA"]) {
            return true;
        }
        if index <= 1 {
            return false;
        }
        if Word::is_vowel(v.at(index - 2)) {
            return false;
        }
        if !v.contains(index - 1, 3, &["ACH"]) {
            return false;
        }
        let c = v.at(index + 2);
        c != b'I' && c != b'E' || v.contains(index - 2, 6, &["BACHER", "MACHER"])
    }

    // Java: DoubleMetaphone.conditionCH0
    fn condition_ch0(v: &Word, index: isize) -> bool {
        if index != 0 {
            return false;
        }
        if !v.contains(index + 1, 5, &["HARAC", "HARIS"])
            && !v.contains(index + 1, 3, &["HOR", "HYM", "HIA", "HEM"])
        {
            return false;
        }
        !v.contains(0, 5, &["CHORE"])
    }

    // Java: DoubleMetaphone.conditionCH1
    fn condition_ch1(v: &Word, index: isize) -> bool {
        v.contains(0, 4, &["VAN ", "VON "])
            || v.contains(0, 3, &["SCH"])
            || v.contains(index - 2, 6, &["ORCHES", "ARCHIT", "ORCHID"])
            || v.contains(index + 2, 1, &["T", "S"])
            || (v.contains(index - 1, 1, &["A", "O", "U", "E"]) || index == 0)
                && (v.contains(index + 2, 1, L_R_N_M_B_H_F_V_W_SPACE) || index + 1 == v.len() - 1)
    }

    // Java: DoubleMetaphone.conditionL0
    fn condition_l0(v: &Word, index: isize) -> bool {
        if index == v.len() - 3 && v.contains(index - 1, 4, &["ILLO", "ILLA", "ALLE"]) {
            return true;
        }
        (v.contains(v.len() - 2, 2, &["AS", "OS"]) || v.contains(v.len() - 1, 1, &["A", "O"]))
            && v.contains(index - 1, 4, &["ALLE"])
    }

    // Java: DoubleMetaphone.conditionM0
    fn condition_m0(v: &Word, index: isize) -> bool {
        if v.at(index + 1) == b'M' {
            return true;
        }
        v.contains(index - 1, 3, &["UMB"])
            && (index + 1 == v.len() - 1 || v.contains(index + 2, 2, &["ER"]))
    }

    // Java: DoubleMetaphone.handleC
    fn handle_c(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if Self::condition_c0(v, index) {
            result.append("K");
            index += 2;
        } else if index == 0 && v.contains(index, 6, &["CAESAR"]) {
            result.append("S");
            index += 2;
        } else if v.contains(index, 2, &["CH"]) {
            index = Self::handle_ch(v, result, index);
        } else if v.contains(index, 2, &["CZ"]) && !v.contains(index - 2, 4, &["WICZ"]) {
            result.append2("S", "X");
            index += 2;
        } else if v.contains(index + 1, 3, &["CIA"]) {
            result.append("X");
            index += 3;
        } else if v.contains(index, 2, &["CC"]) && !(index == 1 && v.at(0) == b'M') {
            return Self::handle_cc(v, result, index);
        } else if v.contains(index, 2, &["CK", "CG", "CQ"]) {
            result.append("K");
            index += 2;
        } else if v.contains(index, 2, &["CI", "CE", "CY"]) {
            if v.contains(index, 3, &["CIO", "CIE", "CIA"]) {
                result.append2("S", "X");
            } else {
                result.append("S");
            }
            index += 2;
        } else {
            result.append("K");
            if v.contains(index + 1, 2, &[" C", " Q", " G"]) {
                index += 3;
            } else if v.contains(index + 1, 1, &["C", "K", "Q"])
                && !v.contains(index + 1, 2, &["CE", "CI"])
            {
                index += 2;
            } else {
                index += 1;
            }
        }
        index
    }

    // Java: DoubleMetaphone.handleCC
    fn handle_cc(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if v.contains(index + 2, 1, &["I", "E", "H"]) && !v.contains(index + 2, 2, &["HU"]) {
            if index == 1 && v.at(index - 1) == b'A'
                || v.contains(index - 1, 5, &["UCCEE", "UCCES"])
            {
                result.append("KS");
            } else {
                result.append("X");
            }
            index += 3;
        } else {
            result.append("K");
            index += 2;
        }
        index
    }

    // Java: DoubleMetaphone.handleCH
    fn handle_ch(v: &Word, result: &mut DmResult, index: isize) -> isize {
        if index > 0 && v.contains(index, 4, &["CHAE"]) {
            result.append2("K", "X");
            return index + 2;
        }
        if Self::condition_ch0(v, index) || Self::condition_ch1(v, index) {
            result.append("K");
            return index + 2;
        }
        if index > 0 {
            if v.contains(0, 2, &["MC"]) {
                result.append("K");
            } else {
                result.append2("X", "K");
            }
        } else {
            result.append("X");
        }
        index + 2
    }

    // Java: DoubleMetaphone.handleD
    fn handle_d(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if v.contains(index, 2, &["DG"]) {
            if v.contains(index + 2, 1, &["I", "E", "Y"]) {
                result.append("J");
                index += 3;
            } else {
                result.append("TK");
                index += 2;
            }
        } else if v.contains(index, 2, &["DT", "DD"]) {
            result.append("T");
            index += 2;
        } else {
            result.append("T");
            index += 1;
        }
        index
    }

    // Java: DoubleMetaphone.handleG -- two of its rules emit the same codes
    // and are kept apart, as written.
    #[allow(clippy::if_same_then_else)]
    fn handle_g(v: &Word, result: &mut DmResult, mut index: isize, slavo_germanic: bool) -> isize {
        if v.at(index + 1) == b'H' {
            index = Self::handle_gh(v, result, index);
        } else if v.at(index + 1) == b'N' {
            if index == 1 && Word::is_vowel(v.at(0)) && !slavo_germanic {
                result.append2("KN", "N");
            } else if !v.contains(index + 2, 2, &["EY"])
                && v.at(index + 1) != b'Y'
                && !slavo_germanic
            {
                result.append2("N", "KN");
            } else {
                result.append("KN");
            }
            index += 2;
        } else if v.contains(index + 1, 2, &["LI"]) && !slavo_germanic {
            result.append2("KL", "L");
            index += 2;
        } else if index == 0
            && (v.at(index + 1) == b'Y'
                || v.contains(index + 1, 2, ES_EP_EB_EL_EY_IB_IL_IN_IE_EI_ER))
        {
            result.append2("K", "J");
            index += 2;
        } else if (v.contains(index + 1, 2, &["ER"]) || v.at(index + 1) == b'Y')
            && !v.contains(0, 6, &["DANGER", "RANGER", "MANGER"])
            && !v.contains(index - 1, 1, &["E", "I"])
            && !v.contains(index - 1, 3, &["RGY", "OGY"])
        {
            result.append2("K", "J");
            index += 2;
        } else if v.contains(index + 1, 1, &["E", "I", "Y"])
            || v.contains(index - 1, 4, &["AGGI", "OGGI"])
        {
            if v.contains(0, 4, &["VAN ", "VON "])
                || v.contains(0, 3, &["SCH"])
                || v.contains(index + 1, 2, &["ET"])
            {
                result.append("K");
            } else if v.contains(index + 1, 3, &["IER"]) {
                result.append("J");
            } else {
                result.append2("J", "K");
            }
            index += 2;
        } else {
            index = Self::skip_double(v, index, b'G');
            result.append("K");
        }
        index
    }

    // Java: DoubleMetaphone.handleGH
    fn handle_gh(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if index > 0 && !Word::is_vowel(v.at(index - 1)) {
            result.append("K");
            index += 2;
        } else if index == 0 {
            if v.at(index + 2) == b'I' {
                result.append("J");
            } else {
                result.append("K");
            }
            index += 2;
        } else if index > 1 && v.contains(index - 2, 1, &["B", "H", "D"])
            || index > 2 && v.contains(index - 3, 1, &["B", "H", "D"])
            || index > 3 && v.contains(index - 4, 1, &["B", "H"])
        {
            index += 2;
        } else {
            if index > 2
                && v.at(index - 1) == b'U'
                && v.contains(index - 3, 1, &["C", "G", "L", "R", "T"])
            {
                result.append("F");
            } else if index > 0 && v.at(index - 1) != b'I' {
                result.append("K");
            }
            index += 2;
        }
        index
    }

    // Java: DoubleMetaphone.handleH
    fn handle_h(v: &Word, result: &mut DmResult, index: isize) -> isize {
        if (index == 0 || Word::is_vowel(v.at(index - 1))) && Word::is_vowel(v.at(index + 1)) {
            result.append("H");
            index + 2
        } else {
            index + 1
        }
    }

    // Java: DoubleMetaphone.handleJ
    fn handle_j(v: &Word, result: &mut DmResult, mut index: isize, slavo_germanic: bool) -> isize {
        if v.contains(index, 4, &["JOSE"]) || v.contains(0, 4, &["SAN "]) {
            if index == 0 && v.at(index + 4) == b' ' || v.len() == 4 || v.contains(0, 4, &["SAN "])
            {
                result.append("H");
            } else {
                result.append2("J", "H");
            }
            index += 1;
        } else {
            if index == 0 && !v.contains(index, 4, &["JOSE"]) {
                result.append2("J", "A");
            } else if Word::is_vowel(v.at(index - 1))
                && !slavo_germanic
                && (v.at(index + 1) == b'A' || v.at(index + 1) == b'O')
            {
                result.append2("J", "H");
            } else if index == v.len() - 1 {
                result.append2("J", " ");
            } else if !v.contains(index + 1, 1, L_T_K_S_N_M_B_Z)
                && !v.contains(index - 1, 1, &["S", "K", "L"])
            {
                result.append("J");
            }
            index = Self::skip_double(v, index, b'J');
        }
        index
    }

    // Java: DoubleMetaphone.handleL
    fn handle_l(v: &Word, result: &mut DmResult, index: isize) -> isize {
        if v.at(index + 1) == b'L' {
            if Self::condition_l0(v, index) {
                result.append_primary("L");
            } else {
                result.append("L");
            }
            index + 2
        } else {
            result.append("L");
            index + 1
        }
    }

    // Java: DoubleMetaphone.handleP
    fn handle_p(v: &Word, result: &mut DmResult, index: isize) -> isize {
        if v.at(index + 1) == b'H' {
            result.append("F");
            index + 2
        } else {
            result.append("P");
            if v.contains(index + 1, 1, &["P", "B"]) {
                index + 2
            } else {
                index + 1
            }
        }
    }

    // Java: DoubleMetaphone.handleR
    fn handle_r(v: &Word, result: &mut DmResult, index: isize, slavo_germanic: bool) -> isize {
        if index == v.len() - 1
            && !slavo_germanic
            && v.contains(index - 2, 2, &["IE"])
            && !v.contains(index - 4, 2, &["ME", "MA"])
        {
            result.append_alternate("R");
        } else {
            result.append("R");
        }
        Self::skip_double(v, index, b'R')
    }

    // Java: DoubleMetaphone.handleS
    fn handle_s(v: &Word, result: &mut DmResult, mut index: isize, slavo_germanic: bool) -> isize {
        if v.contains(index - 1, 3, &["ISL", "YSL"]) {
            index += 1;
        } else if index == 0 && v.contains(index, 5, &["SUGAR"]) {
            result.append2("X", "S");
            index += 1;
        } else if v.contains(index, 2, &["SH"]) {
            if v.contains(index + 1, 4, &["HEIM", "HOEK", "HOLM", "HOLZ"]) {
                result.append("S");
            } else {
                result.append("X");
            }
            index += 2;
        } else if v.contains(index, 3, &["SIO", "SIA"]) || v.contains(index, 4, &["SIAN"]) {
            if slavo_germanic {
                result.append("S");
            } else {
                result.append2("S", "X");
            }
            index += 3;
        } else if index == 0 && v.contains(index + 1, 1, &["M", "N", "L", "W"])
            || v.contains(index + 1, 1, &["Z"])
        {
            result.append2("S", "X");
            index = if v.contains(index + 1, 1, &["Z"]) {
                index + 2
            } else {
                index + 1
            };
        } else if v.contains(index, 2, &["SC"]) {
            index = Self::handle_sc(v, result, index);
        } else {
            if index == v.len() - 1 && v.contains(index - 2, 2, &["AI", "OI"]) {
                result.append_alternate("S");
            } else {
                result.append("S");
            }
            index = if v.contains(index + 1, 1, &["S", "Z"]) {
                index + 2
            } else {
                index + 1
            };
        }
        index
    }

    // Java: DoubleMetaphone.handleSC
    fn handle_sc(v: &Word, result: &mut DmResult, index: isize) -> isize {
        if v.at(index + 2) == b'H' {
            if v.contains(index + 3, 2, &["OO", "ER", "EN", "UY", "ED", "EM"]) {
                if v.contains(index + 3, 2, &["ER", "EN"]) {
                    result.append2("X", "SK");
                } else {
                    result.append("SK");
                }
            } else if index == 0 && !Word::is_vowel(v.at(3)) && v.at(3) != b'W' {
                result.append2("X", "S");
            } else {
                result.append("X");
            }
        } else if v.contains(index + 2, 1, &["I", "E", "Y"]) {
            result.append("S");
        } else {
            result.append("SK");
        }
        index + 3
    }

    // Java: DoubleMetaphone.handleT
    fn handle_t(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if v.contains(index, 4, &["TION"]) || v.contains(index, 3, &["TIA", "TCH"]) {
            result.append("X");
            index += 3;
        } else if v.contains(index, 2, &["TH"]) || v.contains(index, 3, &["TTH"]) {
            if v.contains(index + 2, 2, &["OM", "AM"])
                || v.contains(0, 4, &["VAN ", "VON "])
                || v.contains(0, 3, &["SCH"])
            {
                result.append("T");
            } else {
                result.append2("0", "T");
            }
            index += 2;
        } else {
            result.append("T");
            index = if v.contains(index + 1, 1, &["T", "D"]) {
                index + 2
            } else {
                index + 1
            };
        }
        index
    }

    // Java: DoubleMetaphone.handleW
    fn handle_w(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if v.contains(index, 2, &["WR"]) {
            result.append("R");
            index += 2;
        } else if index == 0 && (Word::is_vowel(v.at(index + 1)) || v.contains(index, 2, &["WH"])) {
            if Word::is_vowel(v.at(index + 1)) {
                result.append2("A", "F");
            } else {
                result.append("A");
            }
            index += 1;
        } else if index == v.len() - 1 && Word::is_vowel(v.at(index - 1))
            || v.contains(index - 1, 5, &["EWSKI", "EWSKY", "OWSKI", "OWSKY"])
            || v.contains(0, 3, &["SCH"])
        {
            result.append_alternate("F");
            index += 1;
        } else if v.contains(index, 4, &["WICZ", "WITZ"]) {
            result.append2("TS", "FX");
            index += 4;
        } else {
            index += 1;
        }
        index
    }

    // Java: DoubleMetaphone.handleX
    fn handle_x(v: &Word, result: &mut DmResult, mut index: isize) -> isize {
        if index == 0 {
            result.append("S");
            index += 1;
        } else {
            if !(index == v.len() - 1
                && (v.contains(index - 3, 3, &["IAU", "EAU"])
                    || v.contains(index - 2, 2, &["AU", "OU"])))
            {
                result.append("KS");
            }
            index = if v.contains(index + 1, 1, &["C", "X"]) {
                index + 2
            } else {
                index + 1
            };
        }
        index
    }

    // Java: DoubleMetaphone.handleZ
    fn handle_z(v: &Word, result: &mut DmResult, index: isize, slavo_germanic: bool) -> isize {
        if v.at(index + 1) == b'H' {
            result.append("J");
            index + 2
        } else {
            if v.contains(index + 1, 2, &["ZO", "ZI", "ZA"])
                || slavo_germanic && index > 0 && v.at(index - 1) != b'T'
            {
                result.append2("S", "TS");
            } else {
                result.append("S");
            }
            Self::skip_double(v, index, b'Z')
        }
    }

    // Java: DoubleMetaphone.isSilentStart
    fn is_silent_start(value: &[u16]) -> bool {
        SILENT_START
            .iter()
            .any(|s| crate::java::starts_with(value, s))
    }

    // Java: DoubleMetaphone.isSlavoGermanic
    fn is_slavo_germanic(value: &[u16]) -> bool {
        let has = |p: &[u8]| {
            value
                .windows(p.len())
                .any(|w| w.iter().zip(p).all(|(&a, &b)| a == u16::from(b)))
        };
        has(b"W") || has(b"K") || has(b"CZ") || has(b"WITZ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java::{string, units};

    fn dm(s: &str, alt: bool) -> String {
        DoubleMetaphone::default()
            .double_metaphone(&units(s), alt)
            .map_or("null".into(), |v| string(&v))
    }

    #[test]
    fn empty_and_negative() {
        assert_eq!(dm("  ", false), "null");
        let mut e = DoubleMetaphone::default();
        e.set_max_code_len(-1);
        assert_eq!(e.max_code_len(), -1);
        assert!(e.double_metaphone(&units("x"), false).is_none());
        e.set_max_code_len(0);
        assert_eq!(e.double_metaphone(&units("smith"), false), Some(vec![]));
    }
}
