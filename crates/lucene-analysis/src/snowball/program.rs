//! `org.tartarus.snowball.SnowballProgram` and `Among`: the runtime the
//! generated stemmers in [`super::algorithms`] call.
//!
//! The generated code is the Snowball compiler's Rust backend, so this runtime
//! keeps that backend's interface (`SnowballEnv`, `i32` positions, `Among`
//! carrying an optional routine -- the API of Snowball's own
//! `rust/src/snowball/snowball_env.rs` at the same commit). Its *state and
//! semantics* are Lucene's `SnowballProgram`'s, method for method:
//!
//! - **The string is UTF-16** (`current: Vec<u16>` with a logical `length`, as
//!   Java's `char[]` and `length`), and every position counts UTF-16 units. The
//!   Rust backend assumes a UTF-8 string with byte positions; the two disagree
//!   wherever an algorithm keeps a position across an edit that changes a
//!   character's width (`yiddish.sbl` replaces a two-unit prefix by the
//!   three-unit `TSU` after `$p1` is set), compares a position with a number
//!   (`dutch.sbl`'s `$p1 < 3`), or counts a surrogate pair. Java's units make
//!   all of them agree with Lucene.
//! - **The buffer is edited in place** (`replace_s` is Java's: the tail
//!   shifted within the buffer, the buffer grown only when the result is
//!   longer), so a `ket` an earlier deletion left past the end behaves as in
//!   Java (`greek.sbl`'s `steps3`).
//! - **`eq_s` honours `limit`**, where Snowball's Rust runtime compares
//!   against the whole string.
//!
//! `tools/gen_snowball.sh` makes the backend's output fit: its string
//! literals become `&[u16]` arrays, `len` is read as `length`, and the one
//! byte-size constant it folds (`yiddish.sbl`'s `sizeof`) is the unit count
//! Java's backend emits.

/// `org.tartarus.snowball.SnowballProgram`'s state: the string being stemmed
/// and the cursor, limits and slice markers of the Snowball language. The
/// generated code reads and writes the fields directly, as the Java
/// subclasses read their `protected` fields.
#[derive(Debug, Default, Clone)]
pub struct SnowballEnv {
    /// Java's `char[] current`: the string in `[0, length)`, then spare room.
    pub current: Vec<u16>,
    /// `length`.
    pub length: i32,
    /// `cursor`.
    pub cursor: i32,
    /// `limit`.
    pub limit: i32,
    /// `limit_backward`.
    pub limit_backward: i32,
    /// `bra`.
    pub bra: i32,
    /// `ket`.
    pub ket: i32,
    /// Whether `replace_s` ran since `set_current` (no Java counterpart: an
    /// unedited string need not be copied back).
    edited: bool,
}

/// `org.tartarus.snowball.Among`: one entry of a sorted `among` table -- the
/// string, the index of the longest entry it extends (`substring_i`, `-1` for
/// none), the result, and the optional routine that must also succeed
/// (Java's `MethodHandle`).
pub struct Among<T: 'static>(pub &'static [u16], pub i32, pub i32, pub Option<Routine<T>>);

/// An `Among` entry's routine: a generated `r_*` function over the
/// language's context.
pub type Routine<T> = &'static (dyn Fn(&mut SnowballEnv, &mut T) -> bool + Sync);

/// A `usize` index from an `i32` position the generated code computed (always
/// a non-negative offset into `current`).
#[inline]
fn at(i: i32) -> usize {
    i as usize
}

/// UTF-16 units as a string, an unpaired surrogate (a stemmer can cut a pair)
/// as U+FFFD.
fn push_utf16(out: &mut String, units: &[u16]) {
    if units.iter().all(|&u| u < 0x80) {
        out.extend(units.iter().map(|&u| char::from(u as u8)));
    } else {
        out.extend(char::decode_utf16(units.iter().copied()).map(|c| c.unwrap_or('\u{FFFD}')));
    }
}

impl SnowballEnv {
    /// Java: `SnowballProgram.setCurrent(String)`. A string longer than
    /// Java's `int` length can hold is refused (`false`, the program left
    /// empty).
    pub fn set_current(&mut self, value: &str) -> bool {
        self.current.clear();
        if value.is_ascii() {
            self.current.extend(value.bytes().map(u16::from));
        } else {
            self.current.extend(value.encode_utf16());
        }
        self.edited = false;
        let fits = i32::try_from(self.current.len()).is_ok();
        if !fits {
            self.current.clear();
        }
        self.length = self.current.len() as i32;
        self.cursor = 0;
        self.limit = self.length;
        self.limit_backward = 0;
        self.bra = 0;
        self.ket = self.length;
        fits
    }

    /// The current string's UTF-16 units (Java's `getCurrentBuffer()` up to
    /// `getCurrentBufferLength()`).
    pub fn current_units(&self) -> &[u16] {
        &self.current[..at(self.length)]
    }

    /// Whether the string was edited since [`Self::set_current`].
    pub fn edited(&self) -> bool {
        self.edited
    }

    /// Java: `SnowballProgram.getCurrent()`, into `out`.
    pub fn get_current_into(&self, out: &mut String) {
        out.clear();
        push_utf16(out, self.current_units());
    }

    /// Whether unit `ch` is in the grouping bitmap `s` over `min..=max`.
    #[inline]
    fn in_set(s: &[u8], min: u32, max: u32, ch: u16) -> bool {
        let ch = u32::from(ch);
        if ch > max || ch < min {
            return false;
        }
        let ch = ch - min;
        s[(ch >> 3) as usize] & (1 << (ch & 7)) != 0
    }

    /// Java: `SnowballProgram.in_grouping`.
    pub fn in_grouping(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor >= self.limit || !Self::in_set(s, min, max, self.current[at(self.cursor)]) {
            return false;
        }
        self.cursor += 1;
        true
    }

    /// Java: `SnowballProgram.in_grouping_b`.
    pub fn in_grouping_b(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor <= self.limit_backward
            || !Self::in_set(s, min, max, self.current[at(self.cursor - 1)])
        {
            return false;
        }
        self.cursor -= 1;
        true
    }

    /// Java: `SnowballProgram.out_grouping`.
    pub fn out_grouping(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor >= self.limit || Self::in_set(s, min, max, self.current[at(self.cursor)]) {
            return false;
        }
        self.cursor += 1;
        true
    }

    /// Java: `SnowballProgram.out_grouping_b`.
    pub fn out_grouping_b(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor <= self.limit_backward
            || Self::in_set(s, min, max, self.current[at(self.cursor - 1)])
        {
            return false;
        }
        self.cursor -= 1;
        true
    }

    /// Java: `SnowballProgram.eq_s`.
    pub fn eq_s(&mut self, s: &[u16]) -> bool {
        let n = s.len() as i32;
        if self.limit - self.cursor < n {
            return false;
        }
        let c = at(self.cursor);
        if self.current[c..c + s.len()] != *s {
            return false;
        }
        self.cursor += n;
        true
    }

    /// Java: `SnowballProgram.eq_s_b`.
    pub fn eq_s_b(&mut self, s: &[u16]) -> bool {
        let n = s.len() as i32;
        if self.cursor - self.limit_backward < n {
            return false;
        }
        let c = at(self.cursor);
        if self.current[c - s.len()..c] != *s {
            return false;
        }
        self.cursor -= n;
        true
    }

    /// The `next` command: Java's `cursor++` (the generated code checks the
    /// limit first).
    pub fn next_char(&mut self) {
        self.cursor += 1;
    }

    /// The backward `next` command: `cursor--`.
    pub fn previous_char(&mut self) {
        self.cursor -= 1;
    }

    /// The `hop n` command: Java's `c = cursor + n; if (c > limit) fail`.
    pub fn hop(&mut self, delta: i32) -> bool {
        let c = self.cursor + delta;
        if c > self.limit {
            return false;
        }
        self.cursor = c;
        true
    }

    /// The backward `hop n` command: `c = cursor - n; if (c < limit_backward)
    /// fail`.
    pub fn hop_back(&mut self, delta: i32) -> bool {
        let c = self.cursor - delta;
        if c < self.limit_backward {
            return false;
        }
        self.cursor = c;
        true
    }

    /// Java: `SnowballProgram.replace_s` -- replaces `c_bra..c_ket` with `s`
    /// in the buffer and moves `length`, `limit` and the cursor by the length
    /// change. As Java, a `c_ket` past `length` skips the tail shift.
    fn replace_s(&mut self, c_bra: i32, c_ket: i32, s: &[u16]) -> i32 {
        let n = s.len() as i32;
        self.edited = true;
        let adjustment = n - (c_ket - c_bra);
        let new_length = self.length + adjustment;
        // Java grows the array to `newLength`; writing `s` needs `c_bra + n`,
        // which only exceeds it when `c_ket` is past `length`.
        let need = at(new_length.max(c_bra + n));
        if need > self.current.len() {
            self.current.resize(need, 0);
        }
        if adjustment != 0 && c_ket < self.length {
            self.current
                .copy_within(at(c_ket)..at(self.length), at(c_bra + n));
        }
        self.current[at(c_bra)..at(c_bra + n)].copy_from_slice(s);
        self.length = new_length;
        self.limit += adjustment;
        if self.cursor >= c_ket {
            self.cursor += adjustment;
        } else if self.cursor > c_bra {
            self.cursor = c_bra;
        }
        adjustment
    }

    /// Java: `SnowballProgram.slice_from` (`void`; Java's `slice_check` is
    /// assertions only).
    pub fn slice_from(&mut self, s: &[u16]) {
        self.replace_s(self.bra, self.ket, s);
    }

    /// Java: `SnowballProgram.slice_del`.
    pub fn slice_del(&mut self) {
        self.slice_from(&[])
    }

    /// Java: `SnowballProgram.insert`.
    pub fn insert(&mut self, c_bra: i32, c_ket: i32, s: &[u16]) {
        let adjustment = self.replace_s(c_bra, c_ket, s);
        if c_bra <= self.bra {
            self.bra += adjustment;
        }
        if c_bra <= self.ket {
            self.ket += adjustment;
        }
    }

    /// Java: `SnowballProgram.slice_to`.
    pub fn slice_to(&mut self) -> Vec<u16> {
        self.current[at(self.bra)..at(self.ket)].to_vec()
    }

    /// Java: `SnowballProgram.find_among`: the binary search over a sorted
    /// table for the longest entry matching at the cursor (whose routine, if
    /// any, succeeds); its result, or 0.
    pub fn find_among<T>(&mut self, v: &[Among<T>], context: &mut T) -> i32 {
        let mut i: i32 = 0;
        let mut j: i32 = v.len() as i32;
        let c = self.cursor;
        let l = self.limit;
        let mut common_i = 0i32;
        let mut common_j = 0i32;
        let mut first_key_inspected = false;
        loop {
            let k = i + ((j - i) >> 1);
            let mut diff: i32 = 0;
            let mut common = common_i.min(common_j);
            for &u in &v[at(k)].0[at(common)..] {
                if c + common == l {
                    diff = -1;
                    break;
                }
                diff = i32::from(self.current[at(c + common)]) - i32::from(u);
                if diff != 0 {
                    break;
                }
                common += 1;
            }
            if diff < 0 {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if j - i <= 1 {
                if i > 0 || j == i || first_key_inspected {
                    break;
                }
                first_key_inspected = true;
            }
        }
        loop {
            let w = &v[at(i)];
            let wlen = w.0.len() as i32;
            if common_i >= wlen {
                self.cursor = c + wlen;
                match w.3 {
                    None => return w.2,
                    Some(method) => {
                        let res = method(self, context);
                        self.cursor = c + wlen;
                        if res {
                            return w.2;
                        }
                    }
                }
            }
            i = w.1;
            if i < 0 {
                return 0;
            }
        }
    }

    /// Java: `SnowballProgram.find_among_b`, the backward search.
    pub fn find_among_b<T>(&mut self, v: &[Among<T>], context: &mut T) -> i32 {
        let mut i: i32 = 0;
        let mut j: i32 = v.len() as i32;
        let c = self.cursor;
        let lb = self.limit_backward;
        let mut common_i = 0i32;
        let mut common_j = 0i32;
        let mut first_key_inspected = false;
        loop {
            let k = i + ((j - i) >> 1);
            let mut diff: i32 = 0;
            let mut common = common_i.min(common_j);
            let w = v[at(k)].0;
            for &u in w[..w.len() - at(common)].iter().rev() {
                if c - common == lb {
                    diff = -1;
                    break;
                }
                diff = i32::from(self.current[at(c - common - 1)]) - i32::from(u);
                if diff != 0 {
                    break;
                }
                common += 1;
            }
            if diff < 0 {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if j - i <= 1 {
                if i > 0 || j == i || first_key_inspected {
                    break;
                }
                first_key_inspected = true;
            }
        }
        loop {
            let w = &v[at(i)];
            let wlen = w.0.len() as i32;
            if common_i >= wlen {
                self.cursor = c - wlen;
                match w.3 {
                    None => return w.2,
                    Some(method) => {
                        let res = method(self, context);
                        self.cursor = c - wlen;
                        if res {
                            return w.2;
                        }
                    }
                }
            }
            i = w.1;
            if i < 0 {
                return 0;
            }
        }
    }
}
