//! The Egothor stemmer's tables (`org.egothor.stemmer`): `Trie`, `Row`,
//! `Cell`, `MultiTrie` and `MultiTrie2` as `StempelStemmer.load` reads them
//! from a `DataInputStream`, and their `getLastOnPath` lookups.
//!
//! Derived from the Egothor project's `org.egothor.stemmer` as Lucene 10.5.0
//! ships it, under the Egothor Software License 1.00 (BSD-style):
//!
//! ```text
//!                    Egothor Software License version 1.00
//!                    Copyright (C) 1997-2004 Leo Galambos.
//!                 Copyright (C) 2002-2004 "Egothor developers"
//!                      on behalf of the Egothor Project.
//!                             All rights reserved.
//!
//!   Redistribution  and  use  in  source and binary forms, with or without
//!   modification, are permitted provided that the following conditions are
//!   met:
//!    1. Redistributions  of  source  code  must retain the above copyright
//!       notice, the list of contributors, this list of conditions, and the
//!       following disclaimer.
//!    2. Redistributions  in binary form must reproduce the above copyright
//!       notice, the list of contributors, this list of conditions, and the
//!       disclaimer  that  follows  these  conditions  in the documentation
//!       and/or other materials provided with the distribution.
//!    3. The name "Egothor" must not be used to endorse or promote products
//!       derived  from  this software without prior written permission. For
//!       written permission, please contact Leo.G@seznam.cz
//!    4. Products  derived  from this software may not be called "Egothor",
//!       nor  may  "Egothor"  appear  in  their name, without prior written
//!       permission from Leo.G@seznam.cz.
//!
//!   In addition, we request that you include in the end-user documentation
//!   provided  with  the  redistribution  and/or  in the software itself an
//!   acknowledgement equivalent to the following:
//!   "This product includes software developed by the Egothor Project.
//!    http://egothor.sf.net/"
//!
//!   THIS  SOFTWARE  IS  PROVIDED  ``AS  IS''  AND ANY EXPRESSED OR IMPLIED
//!   WARRANTIES,  INCLUDING,  BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//!   MERCHANTABILITY  AND  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//!   IN  NO  EVENT  SHALL THE EGOTHOR PROJECT OR ITS CONTRIBUTORS BE LIABLE
//!   FOR   ANY   DIRECT,   INDIRECT,  INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR
//!   CONSEQUENTIAL  DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//!   SUBSTITUTE  GOODS  OR  SERVICES;  LOSS  OF  USE,  DATA, OR PROFITS; OR
//!   BUSINESS  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
//!   WHETHER  IN  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
//!   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
//!   IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//!
//!   This  software  consists  of  voluntary  contributions  made  by  many
//!   individuals  on  behalf  of  the  Egothor  Project  and was originally
//!   created by Leo Galambos (Leo.G@seznam.cz).
//! ```
//!
//! # Format
//!
//! Java's `DataOutput`, big-endian: a table is `writeUTF(method)` (a name
//! holding `M` selects a `MultiTrie2`), then for a `Trie`: `boolean
//! forward`, `int root`, `int` command count and that many `writeUTF`
//! commands, `int` row count and the rows; a row is an `int` cell count and
//! per cell `char` (the key), `int cmd`, `int cnt`, `int ref`, `int skip`.
//! A `MultiTrie2` is `boolean forward`, `int BY`, `int` trie count and the
//! tries. Commands are edit scripts for [`crate::diff`].
//!
//! Differs: a table whose root, a cell's `ref` or a cell's `cmd` points
//! outside its rows or commands is refused when read; Java reads it and
//! throws `NullPointerException`/`IndexOutOfBoundsException` (or, inside a
//! `MultiTrie2`, silently stops) on the first word that reaches the bad cell.

use crate::StempelError;

/// `MultiTrie.EOM`: the command that ends a `MultiTrie`'s chain.
const EOM: u16 = b'*' as u16;

/// Java's `DataInputStream` over a byte slice.
pub struct DataInput<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn eof() -> StempelError {
    StempelError::new("EOFException: the stemmer table ends early")
}

impl<'a> DataInput<'a> {
    /// A reader at the start of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        DataInput { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StempelError> {
        let end = self.pos.checked_add(n).ok_or_else(eof)?;
        let bytes = self.buf.get(self.pos..end).ok_or_else(eof)?;
        self.pos = end;
        Ok(bytes)
    }

    /// `readBoolean()`.
    pub fn read_boolean(&mut self) -> Result<bool, StempelError> {
        Ok(self.take(1)?[0] != 0)
    }

    /// `readInt()`.
    pub fn read_int(&mut self) -> Result<i32, StempelError> {
        let b = self.take(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// `readChar()`.
    pub fn read_char(&mut self) -> Result<u16, StempelError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    /// `readUTF()`: a `u16` byte length, then modified UTF-8, as UTF-16 units.
    pub fn read_utf(&mut self) -> Result<Vec<u16>, StempelError> {
        let len = usize::from(self.read_char()?);
        let bytes = self.take(len)?;
        let bad =
            || StempelError::new("UTFDataFormatException: malformed input in the stemmer table");
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while let Some(&b0) = bytes.get(i) {
            let b0 = u16::from(b0);
            let cont = |k: usize| -> Result<u16, StempelError> {
                let b = bytes.get(k).copied().ok_or_else(bad)?;
                if b & 0xC0 != 0x80 {
                    return Err(bad());
                }
                Ok(u16::from(b & 0x3F))
            };
            // ARITH: i < len <= 65535, so i + 3 cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            let (unit, width) = match b0 >> 4 {
                0..=7 => (b0, 1),
                12 | 13 => (((b0 & 0x1F) << 6) | cont(i + 1)?, 2),
                14 => (((b0 & 0x0F) << 12) | (cont(i + 1)? << 6) | cont(i + 2)?, 3),
                _ => return Err(bad()),
            };
            out.push(unit);
            // ARITH: as above.
            #[allow(clippy::arithmetic_side_effects)]
            {
                i += width;
            }
        }
        Ok(out)
    }
}

/// `org.egothor.stemmer.Cell`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    /// The row a further character continues in, or negative.
    pub ref_: i32,
    /// The command for a word ending here, or negative.
    pub cmd: i32,
    /// How many words chose `cmd` (table statistics; unused by lookups).
    pub cnt: i32,
    /// Characters to skip after this one (`getFully` only).
    pub skip: i32,
}

/// `org.egothor.stemmer.Row`: cells by character, sorted for lookup
/// (Java's `CharObjectHashMap`; a later cell for the same character
/// replaces an earlier one, as `put` does).
#[derive(Debug, Clone, Default)]
pub struct Row {
    cells: Vec<(u16, Cell)>,
}

impl Row {
    // Java: Row(DataInput)
    fn read(input: &mut DataInput) -> Result<Row, StempelError> {
        let count = input.read_int()?;
        let mut cells: Vec<(u16, Cell)> = Vec::new();
        for _ in 0..count.max(0) {
            let ch = input.read_char()?;
            let cell = Cell {
                cmd: input.read_int()?,
                cnt: input.read_int()?,
                ref_: input.read_int()?,
                skip: input.read_int()?,
            };
            cells.push((ch, cell));
        }
        // Stable, so the last of equal characters is last; keep it.
        cells.sort_by_key(|&(c, _)| c);
        cells.reverse();
        cells.dedup_by_key(|&mut (c, _)| c);
        cells.reverse();
        Ok(Row { cells })
    }

    /// `Row.at(char)`.
    pub fn at(&self, ch: u16) -> Option<&Cell> {
        self.cells
            .binary_search_by_key(&ch, |&(c, _)| c)
            .ok()
            .map(|i| &self.cells[i].1)
    }

    /// `Row.getCmd(char)`: `-1` without a cell.
    pub fn cmd(&self, ch: u16) -> i32 {
        self.at(ch).map_or(-1, |c| c.cmd)
    }

    /// `Row.getRef(char)`: `-1` without a cell.
    pub fn ref_(&self, ch: u16) -> i32 {
        self.at(ch).map_or(-1, |c| c.ref_)
    }
}

/// A lookup that, in Java, indexes past a string's end
/// (`StringIndexOutOfBoundsException`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfBounds;

/// `org.egothor.stemmer.Trie`.
#[derive(Debug, Clone)]
pub struct Trie {
    rows: Vec<Row>,
    cmds: Vec<Vec<u16>>,
    root: usize,
    forward: bool,
}

impl Trie {
    /// `new Trie(DataInput)`, checked: every row and command a cell names
    /// exists.
    pub fn read(input: &mut DataInput) -> Result<Trie, StempelError> {
        let forward = input.read_boolean()?;
        let root = input.read_int()?;
        let mut cmds = Vec::new();
        for _ in 0..input.read_int()?.max(0) {
            cmds.push(input.read_utf()?);
        }
        let mut rows = Vec::new();
        for _ in 0..input.read_int()?.max(0) {
            rows.push(Row::read(input)?);
        }
        let in_range = |v: i32, len: usize| usize::try_from(v).map_or(true, |v| v < len);
        let root = usize::try_from(root)
            .ok()
            .filter(|&r| r < rows.len())
            .ok_or_else(|| StempelError::new(format!("stemmer table root {root} is not a row")))?;
        for row in &rows {
            for (_, c) in &row.cells {
                if !in_range(c.ref_, rows.len()) || !in_range(c.cmd, cmds.len()) {
                    return Err(StempelError::new(format!(
                        "stemmer table cell names row {} / command {} of {} / {}",
                        c.ref_,
                        c.cmd,
                        rows.len(),
                        cmds.len()
                    )));
                }
            }
        }
        Ok(Trie {
            rows,
            cmds,
            root,
            forward,
        })
    }

    /// Whether keys are walked first character first.
    pub fn is_forward(&self) -> bool {
        self.forward
    }

    /// The rows, root first by index [`Self::root`].
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// The root row's index.
    pub fn root(&self) -> usize {
        self.root
    }

    /// The commands.
    pub fn cmds(&self) -> &[Vec<u16>] {
        &self.cmds
    }

    /// `Trie.getLastOnPath(key)`: the command of the longest prefix of the
    /// key (in the trie's direction) that has one. `Err` for an empty key,
    /// where Java's `StrEnum.next()` indexes past it.
    pub fn get_last_on_path(&self, key: &[u16]) -> Result<Option<&[u16]>, OutOfBounds> {
        let n = key.len();
        let Some(last_index) = n.checked_sub(1) else {
            return Err(OutOfBounds);
        };
        // ARITH: i < n, so n - 1 - i >= 0.
        #[allow(clippy::arithmetic_side_effects)]
        let at = |i: usize| if self.forward { key[i] } else { key[n - 1 - i] };
        let cmd = |w: i32| usize::try_from(w).ok().map(|w| self.cmds[w].as_slice());
        let mut now = &self.rows[self.root];
        let mut last = None;
        for i in 0..last_index {
            let ch = at(i);
            if let Some(c) = cmd(now.cmd(ch)) {
                last = Some(c);
            }
            match usize::try_from(now.ref_(ch)) {
                Ok(r) => now = &self.rows[r],
                Err(_) => return Ok(last),
            }
        }
        Ok(cmd(now.cmd(at(last_index))).or(last))
    }
}

/// `MultiTrie2.lengthPP`: how many key characters a command's `-` and `D`
/// (by their count letter) and `R` (one each) consume.
fn length_pp(cmd: &[u16]) -> Result<i64, OutOfBounds> {
    let mut len: i64 = 0;
    let mut i = 0;
    while i < cmd.len() {
        let c = cmd[i];
        // ARITH: i < cmd.len() <= 65535.
        #[allow(clippy::arithmetic_side_effects)]
        {
            i += 1;
        }
        if c == u16::from(b'-') || c == u16::from(b'D') {
            let p = *cmd.get(i).ok_or(OutOfBounds)?;
            // ARITH: at most 32768 terms of at most 65536 each.
            #[allow(clippy::arithmetic_side_effects)]
            {
                len += i64::from(p) - i64::from(b'a') + 1;
            }
        } else if c == u16::from(b'R') {
            // ARITH: as above.
            #[allow(clippy::arithmetic_side_effects)]
            {
                len += 1;
            }
        }
        // ARITH: as above.
        #[allow(clippy::arithmetic_side_effects)]
        {
            i += 1;
        }
    }
    Ok(len)
}

/// `org.egothor.stemmer.MultiTrie2` (read as `MultiTrie` reads it): one
/// trie per command part, each part's lookup key shortened by what the
/// previous parts consumed.
#[derive(Debug, Clone)]
pub struct MultiTrie2 {
    forward: bool,
    by: i32,
    tries: Vec<Trie>,
}

impl MultiTrie2 {
    /// `new MultiTrie2(DataInput)` (`MultiTrie(DataInput)`).
    pub fn read(input: &mut DataInput) -> Result<MultiTrie2, StempelError> {
        let forward = input.read_boolean()?;
        let by = input.read_int()?;
        let mut tries = Vec::new();
        for _ in 0..input.read_int()?.max(0) {
            tries.push(Trie::read(input)?);
        }
        Ok(MultiTrie2 { forward, by, tries })
    }

    /// The tries.
    pub fn tries(&self) -> &[Trie] {
        &self.tries
    }

    /// `MultiTrie.BY`.
    pub fn by(&self) -> i32 {
        self.by
    }

    /// `MultiTrie2.skip`: the key without `count` characters at its front
    /// (forward) or back.
    fn skip(&self, key: &[u16], count: i64) -> Result<std::ops::Range<usize>, OutOfBounds> {
        let n = key.len();
        let count = usize::try_from(count).map_err(|_| OutOfBounds)?;
        let rest = n.checked_sub(count).ok_or(OutOfBounds)?;
        Ok(if self.forward { count..n } else { 0..rest })
    }

    /// `MultiTrie2.getLastOnPath(key)`: the commands of successive tries,
    /// concatenated until one has none, ends the chain (`*`), cannot follow
    /// the previous (`--`, `DD`), or Java would index out of bounds (the
    /// result so far is kept, as Java's `catch` keeps it).
    pub fn get_last_on_path(&self, key: &[u16]) -> Vec<u16> {
        let mut result = Vec::new();
        let mut key = key;
        let mut last_key = key;
        let mut prev: Option<&[u16]> = None;
        let mut last_ch = u16::from(b' ');
        for (i, trie) in self.tries.iter().enumerate() {
            let r = match trie.get_last_on_path(last_key) {
                Err(OutOfBounds) | Ok(None) => return result,
                Ok(Some(r)) => r,
            };
            if r == [EOM] {
                return result;
            }
            let Some(&first) = r.first() else {
                return result;
            };
            // Java: cannotFollow(lastch, r.charAt(0)).
            if (last_ch == u16::from(b'-') || last_ch == u16::from(b'D')) && last_ch == first {
                return result;
            }
            let Some(second_last) = r.len().checked_sub(2).map(|k| r[k]) else {
                return result;
            };
            last_ch = second_last;
            if first == u16::from(b'-') {
                if i > 0 {
                    let Some(p) = prev else { return result };
                    let Ok(range) = length_pp(p).and_then(|l| self.skip(key, l)) else {
                        return result;
                    };
                    key = &key[range];
                }
                let Ok(range) = length_pp(r).and_then(|l| self.skip(key, l)) else {
                    return result;
                };
                key = &key[range];
            }
            prev = Some(r);
            result.extend_from_slice(r);
            if !key.is_empty() {
                last_key = key;
            }
        }
        result
    }
}

/// A stemmer table as `StempelStemmer.load` returns it.
#[derive(Debug, Clone)]
pub enum Table {
    /// A `Trie` (the method name has no `M`).
    Trie(Trie),
    /// A `MultiTrie2`.
    Multi(MultiTrie2),
}

impl Table {
    /// `StempelStemmer.load(InputStream)`: `readUTF()` names the method
    /// (upper-cased; holding `M` means a `MultiTrie2`), the table follows.
    pub fn load(bytes: &[u8]) -> Result<Table, StempelError> {
        let mut input = DataInput::new(bytes);
        let method = input.read_utf()?;
        let upper = lucene_analysis::lang::java_string_to_upper_case(&method);
        if upper.contains(&u16::from(b'M')) {
            Ok(Table::Multi(MultiTrie2::read(&mut input)?))
        } else {
            Ok(Table::Trie(Trie::read(&mut input)?))
        }
    }

    /// `Trie.getLastOnPath` of either kind; `None` where Java answers
    /// `null`, `Err` where a plain `Trie` throws.
    pub fn get_last_on_path(&self, key: &[u16]) -> Result<Option<Vec<u16>>, OutOfBounds> {
        match self {
            Table::Trie(t) => Ok(t.get_last_on_path(key)?.map(<[u16]>::to_vec)),
            Table::Multi(m) => Ok(Some(m.get_last_on_path(key))),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk
    use super::*;

    /// A test-only `DataOutput`.
    #[derive(Default)]
    pub(crate) struct Out(pub Vec<u8>);

    impl Out {
        pub fn boolean(&mut self, b: bool) -> &mut Self {
            self.0.push(u8::from(b));
            self
        }
        pub fn int(&mut self, v: i32) -> &mut Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        pub fn char(&mut self, c: char) -> &mut Self {
            self.0.extend_from_slice(&(c as u16).to_be_bytes());
            self
        }
        pub fn utf(&mut self, s: &str) -> &mut Self {
            let b = s.as_bytes();
            self.0.extend_from_slice(&(b.len() as u16).to_be_bytes());
            self.0.extend_from_slice(b);
            self
        }
        /// A trie: `forward`, `root`, `cmds`, rows of `(char, cmd, ref)`.
        pub fn trie(
            &mut self,
            forward: bool,
            root: i32,
            cmds: &[&str],
            rows: &[&[(char, i32, i32)]],
        ) -> &mut Self {
            self.boolean(forward).int(root).int(cmds.len() as i32);
            for c in cmds {
                self.utf(c);
            }
            self.int(rows.len() as i32);
            for row in rows {
                self.int(row.len() as i32);
                for &(ch, cmd, r) in *row {
                    self.char(ch).int(cmd).int(1).int(r).int(0);
                }
            }
            self
        }
    }

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn reads_data_input() {
        let mut o = Out::default();
        o.boolean(true).int(-2).char('ż').utf("aż\u{0}");
        let mut i = DataInput::new(&o.0);
        assert!(i.read_boolean().unwrap());
        assert_eq!(i.read_int().unwrap(), -2);
        assert_eq!(i.read_char().unwrap(), 'ż' as u16);
        // Rust's UTF-8 for U+0000 is one byte, which modified UTF-8 also reads.
        assert_eq!(i.read_utf().unwrap(), u("aż\u{0}"));
        assert!(i.read_boolean().is_err());
        for bad in [
            &[0u8, 1, 0x80][..],
            &[0, 2, 0xC4, 0x41],
            &[0, 1, 0xF0],
            &[0, 2, 0xE0, 0x80],
            &[0, 5, b'a'],
        ] {
            assert!(DataInput::new(bad).read_utf().is_err(), "{bad:?}");
        }
        assert_eq!(
            DataInput::new(&[0, 3, 0xE2, 0x82, 0xAC])
                .read_utf()
                .unwrap(),
            u("€")
        );
    }

    #[test]
    fn trie_lookups() {
        // Backward trie: "ab" -> row1 via 'b', then 'a' has cmd 1; 'b' alone cmd 0.
        let mut o = Out::default();
        o.trie(
            false,
            0,
            &["Da", "Rx"],
            &[&[('b', 0, 1), ('c', 1, -1)], &[('a', 1, -1)]],
        );
        let t = Trie::read(&mut DataInput::new(&o.0)).unwrap();
        assert!(!t.is_forward());
        assert_eq!(t.root(), 0);
        assert_eq!(t.rows().len(), 2);
        assert_eq!(t.cmds().len(), 2);
        assert_eq!(
            t.get_last_on_path(&u("ab")).unwrap(),
            Some(u("Rx").as_slice())
        );
        assert_eq!(
            t.get_last_on_path(&u("zb")).unwrap(),
            Some(u("Da").as_slice())
        );
        assert_eq!(
            t.get_last_on_path(&u("zc")).unwrap(),
            Some(u("Rx").as_slice())
        );
        assert_eq!(
            t.get_last_on_path(&u("zzc")).unwrap(),
            Some(u("Rx").as_slice())
        );
        assert_eq!(t.get_last_on_path(&u("zzz")).unwrap(), None);
        assert_eq!(
            t.get_last_on_path(&u("b")).unwrap(),
            Some(u("Da").as_slice())
        );
        assert_eq!(t.get_last_on_path(&[]), Err(OutOfBounds));
        assert_eq!(t.rows()[0].at('q' as u16), None);
        // Duplicate characters in a row: the last one wins.
        let mut d = Out::default();
        d.trie(true, 0, &["Da", "Db"], &[&[('a', 0, -1), ('a', 1, -1)]]);
        let t = Trie::read(&mut DataInput::new(&d.0)).unwrap();
        assert_eq!(
            t.get_last_on_path(&u("a")).unwrap(),
            Some(u("Db").as_slice())
        );
    }

    #[test]
    fn corrupt_tries_are_refused() {
        for (root, rows) in [
            (1, &[&[('a', 0, -1)][..]][..]),
            (-1, &[&[('a', 0, -1)][..]][..]),
            (0, &[&[('a', 5, -1)][..]][..]),
            (0, &[&[('a', 0, 3)][..]][..]),
        ] {
            let mut o = Out::default();
            o.trie(true, root, &["Da"], rows);
            assert!(Trie::read(&mut DataInput::new(&o.0)).is_err());
        }
        let mut o = Out::default();
        o.boolean(true).int(0).int(1).utf("x");
        assert!(Trie::read(&mut DataInput::new(&o.0)).is_err());
    }

    #[test]
    fn length_pp_counts_consumed_chars() {
        assert_eq!(length_pp(&u("-cDbRxIy")), Ok(3 + 2 + 1));
        assert_eq!(length_pp(&u("-")), Err(OutOfBounds));
        assert_eq!(length_pp(&u("Ia")), Ok(0));
    }

    /// A test trie: its commands and its rows of `(char, cmd, ref)`.
    type TestTrie<'a> = (&'a [&'a str], &'a [&'a [(char, i32, i32)]]);

    fn multi(forward: bool, tries: &[TestTrie]) -> MultiTrie2 {
        let mut o = Out::default();
        o.utf("-0ME2")
            .boolean(forward)
            .int(1)
            .int(tries.len() as i32);
        for (cmds, rows) in tries {
            o.trie(forward, 0, cmds, rows);
        }
        match Table::load(&o.0).unwrap() {
            Table::Multi(m) => m,
            Table::Trie(_) => panic!("a MultiTrie2"),
        }
    }

    #[test]
    fn multi_trie2_chains() {
        // Two tries: the first answers "-b" (skip 2), the second, looked up
        // on the shortened key, "Rx".
        let m = multi(
            false,
            &[
                (&["-b"], &[&[('c', 0, -1)]]),
                (&["Rx", "*"], &[&[('a', 0, -1), ('c', 1, -1)]]),
            ],
        );
        assert_eq!(m.by(), 1);
        assert_eq!(m.tries().len(), 2);
        assert_eq!(m.get_last_on_path(&u("aac")), u("-bRx"));
        // The second trie ends the chain.
        assert_eq!(m.get_last_on_path(&u("ccc")), u("-b"));
        // No command at all.
        assert_eq!(m.get_last_on_path(&u("zzz")), Vec::<u16>::new());
        // Empty key: the first lookup throws in Java; nothing is kept.
        assert_eq!(m.get_last_on_path(&[]), Vec::<u16>::new());
        // The key cannot be shortened by more than it holds.
        assert_eq!(m.get_last_on_path(&u("c")), Vec::<u16>::new());
        // "--" cannot follow "-": the chain stops after the first part.
        let m = multi(
            true,
            &[(&["-a"], &[&[('a', 0, -1)]]), (&["-a"], &[&[('a', 0, -1)]])],
        );
        assert_eq!(m.get_last_on_path(&u("aaaa")), u("-a"));
        // A one-character command has no second-to-last character.
        let m = multi(true, &[(&["R"], &[&[('a', 0, -1)]])]);
        assert_eq!(m.get_last_on_path(&u("a")), Vec::<u16>::new());
        // An empty command.
        let m = multi(true, &[(&[""], &[&[('a', 0, -1)]])]);
        assert_eq!(m.get_last_on_path(&u("a")), Vec::<u16>::new());
        // A previous part whose length cannot be read.
        let m = multi(
            true,
            &[(&["Ia"], &[&[('a', 0, -1)]]), (&["-"], &[&[('a', 0, -1)]])],
        );
        assert_eq!(m.get_last_on_path(&u("aa")), u("Ia"));
        let m = multi(true, &[(&["-a-"], &[&[('a', 0, -1)]])]);
        assert_eq!(m.get_last_on_path(&u("aa")), Vec::<u16>::new());
        let m = multi(
            true,
            &[
                (&["Ix-", "-a"], &[&[('a', 0, -1)]]),
                (&["-a"], &[&[('a', 0, -1)]]),
            ],
        );
        assert_eq!(m.get_last_on_path(&u("aa")), u("Ix-"));
    }

    #[test]
    fn plain_tables_and_errors() {
        let mut o = Out::default();
        o.utf("-0E2").trie(true, 0, &["Rx"], &[&[('a', 0, -1)]]);
        let t = Table::load(&o.0).unwrap();
        assert!(matches!(t, Table::Trie(_)));
        assert_eq!(t.get_last_on_path(&u("a")).unwrap(), Some(u("Rx")));
        assert_eq!(t.get_last_on_path(&[]), Err(OutOfBounds));
        assert!(Table::load(&[0, 1]).is_err());
        let mut lower = Out::default();
        lower.utf("m").boolean(true).int(1).int(0);
        assert!(matches!(Table::load(&lower.0).unwrap(), Table::Multi(_)));
        assert_eq!(
            Table::load(&lower.0)
                .unwrap()
                .get_last_on_path(&u("a"))
                .unwrap(),
            Some(vec![])
        );
    }
}
