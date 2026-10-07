//! The JFlex 1.8.2 scanner skeleton Lucene's generated analysis-common
//! scanners share (`UAX29URLEmailTokenizerImpl`, `HTMLStripCharFilter`),
//! over tables read back from the compiled Lucene classes.
//!
//! # Tables
//!
//! The generated classes carry run-length-packed `String` tables that their
//! static initialisers unpack into `int[]`s. `tools/ExtractJFlexTables.java`
//! reads those unpacked arrays by reflection and writes them, zlib-compressed,
//! to a binary file per scanner (the email scanner's transition table alone
//! has 1,095,920 entries -- 5 MB as Rust source). [`JFlexTables::load`]
//! inflates it once.
//!
//! # Skeleton
//!
//! [`JFlexScanner::scan`] is the part of `getNextToken()` before the action
//! `switch`: the longest match from `zzMarkedPos`, `zzRefill` included. The
//! scanner that owns it runs the `switch` on the action it returns, as the
//! generated code does inline.

use crate::java_character::{char_count, code_point_at, is_high_surrogate};
use crate::reader::CharReader;
use crate::AnalysisError;

/// `YYEOF`.
pub const YYEOF: i32 = -1;

/// The unpacked tables of one JFlex scanner.
#[derive(Debug)]
pub struct JFlexTables {
    /// `ZZ_LEXSTATE`.
    pub lexstate: Vec<i32>,
    cmap_top: Vec<i32>,
    cmap_blocks: Vec<i32>,
    /// `ZZ_ACTION`.
    pub action: Vec<i32>,
    rowmap: Vec<i32>,
    trans: Vec<i32>,
    attribute: Vec<i32>,
    /// `zzCMap` of every BMP `char`, unpacked from the two tables above at
    /// load, so the scanner's step for a `char` is one lookup.
    cmap_bmp: Box<[i32; 0x10000]>,
}

const TABLE_NAMES: [&str; 7] = [
    "ZZ_LEXSTATE",
    "ZZ_CMAP_TOP",
    "ZZ_CMAP_BLOCKS",
    "ZZ_ACTION",
    "ZZ_ROWMAP",
    "ZZ_TRANS",
    "ZZ_ATTRIBUTE",
];

fn corrupt(what: &str) -> AnalysisError {
    AnalysisError::IllegalState(format!("corrupt JFlex tables: {what}"))
}

impl JFlexTables {
    /// Inflates and parses a file `ExtractJFlexTables` wrote: `"JFLX"`, then
    /// per table its name (`u8` length + ASCII), its element count (`u32`
    /// LE) and its elements (`i32` LE), the whole zlib-compressed.
    pub fn load(compressed: &[u8]) -> Result<Self, AnalysisError> {
        let raw = miniz_oxide::inflate::decompress_to_vec_zlib(compressed)
            .map_err(|_| corrupt("zlib"))?;
        let mut p = raw.strip_prefix(b"JFLX").ok_or_else(|| corrupt("magic"))?;
        let mut take = |n: usize| -> Result<&[u8], AnalysisError> {
            if p.len() < n {
                return Err(corrupt("truncated"));
            }
            let (a, b) = p.split_at(n);
            p = b;
            Ok(a)
        };
        let mut tables: Vec<Vec<i32>> = Vec::with_capacity(TABLE_NAMES.len());
        for name in TABLE_NAMES {
            let len = usize::from(take(1)?[0]);
            if take(len)? != name.as_bytes() {
                return Err(corrupt(name));
            }
            let count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
            let bytes = take(count.checked_mul(4).ok_or_else(|| corrupt("count"))?)?;
            tables.push(
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect(),
            );
        }
        let mut it = tables.into_iter();
        let mut next = || it.next().expect("seven tables");
        let mut t = JFlexTables {
            cmap_bmp: Box::new([0; 0x10000]),
            lexstate: next(),
            cmap_top: next(),
            cmap_blocks: next(),
            action: next(),
            rowmap: next(),
            trans: next(),
            attribute: next(),
        };
        let states = t.rowmap.len();
        if t.action.len() != states
            || t.attribute.len() != states
            || t.cmap_top.len() != 0x110000 >> 8
        {
            return Err(corrupt("shapes"));
        }
        for c in 0..0x10000usize {
            let block = if c < 256 { 0 } else { t.cmap_top[c >> 8] };
            t.cmap_bmp[c] = usize::try_from(block | (c & 255) as i32)
                .ok()
                .and_then(|i| t.cmap_blocks.get(i))
                .copied()
                .ok_or_else(|| corrupt("ZZ_CMAP_BLOCKS"))?;
        }
        Ok(t)
    }

    /// `zzCMap(int)`.
    fn cmap(&self, input: i32) -> i32 {
        let offset = input & 255;
        if offset == input {
            self.cmap_blocks[offset as usize]
        } else {
            self.cmap_blocks[(self.cmap_top[(input >> 8) as usize] | offset) as usize]
        }
    }
}

/// What [`JFlexScanner::scan`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    /// `zzInput == YYEOF && zzStartRead == zzCurrentPos`: the end, in the
    /// given lexical state.
    Eof,
    /// `ZZ_ACTION[zzAction]`, or `-1` for no match.
    Action(i32),
}

/// The state of a JFlex 1.8.2 scanner: either Lucene's tokenizer skeleton
/// (`ZZ_BUFFERSIZE` an instance field, the buffer never grows, so a full
/// buffer ends the match) or JFlex's standard one (the buffer doubles), as
/// `HTMLStripCharFilter` uses.
#[derive(Debug)]
pub struct JFlexScanner {
    tables: &'static JFlexTables,
    /// `ZZ_BUFFERSIZE`.
    pub buffer_size: usize,
    /// `zzBuffer`.
    pub buffer: Vec<u16>,
    /// `zzLexicalState`.
    pub lexical_state: usize,
    /// `zzMarkedPos`.
    pub marked_pos: usize,
    current_pos: usize,
    /// `zzStartRead`.
    pub start_read: usize,
    /// `zzEndRead`.
    pub end_read: usize,
    at_eof: bool,
    final_high_surrogate: usize,
    /// `yychar`.
    pub yychar: i64,
    growable: bool,
    one: [u16; 1],
}

impl JFlexScanner {
    /// A scanner over `tables` with a fixed `buffer_size`-unit buffer.
    pub fn new(tables: &'static JFlexTables, buffer_size: usize) -> Self {
        Self::with_growth(tables, buffer_size, false)
    }

    /// A scanner whose buffer starts at `buffer_size` units and doubles when
    /// a match outgrows it (`growable`), or stays fixed.
    pub fn with_growth(tables: &'static JFlexTables, buffer_size: usize, growable: bool) -> Self {
        JFlexScanner {
            tables,
            buffer_size,
            buffer: vec![0; buffer_size],
            lexical_state: 0,
            marked_pos: 0,
            current_pos: 0,
            start_read: 0,
            end_read: 0,
            at_eof: false,
            final_high_surrogate: 0,
            yychar: 0,
            growable,
            one: [0],
        }
    }

    /// `yyreset(Reader)` (the reader itself is the caller's).
    pub fn yyreset(&mut self) {
        self.at_eof = false;
        self.current_pos = 0;
        self.marked_pos = 0;
        self.start_read = 0;
        self.end_read = 0;
        self.final_high_surrogate = 0;
        self.yychar = 0;
        self.lexical_state = 0;
        if self.buffer.len() > self.buffer_size {
            self.buffer = vec![0; self.buffer_size];
        }
    }

    /// `setBufferSize(int)` (Lucene's addition to the skeleton).
    pub fn set_buffer_size(&mut self, num_chars: usize) {
        self.buffer_size = num_chars;
        let mut nb = vec![0u16; num_chars];
        let n = self.buffer.len().min(num_chars);
        nb[..n].copy_from_slice(&self.buffer[..n]);
        self.buffer = nb;
    }

    /// `yylength()`.
    pub fn yylength(&self) -> usize {
        self.marked_pos - self.start_read
    }

    /// `yytext()` as UTF-16 units.
    pub fn text(&self) -> &[u16] {
        &self.buffer[self.start_read..self.marked_pos]
    }

    /// `yycharat(int)`.
    pub fn yycharat(&self, pos: usize) -> u16 {
        self.buffer[self.start_read + pos]
    }

    /// `yypushback(int)`.
    pub fn yypushback(&mut self, number: usize) -> Result<(), AnalysisError> {
        if number > self.yylength() {
            return Err(AnalysisError::IllegalState(
                "Error: pushback value was too large".into(),
            ));
        }
        self.marked_pos -= number;
        Ok(())
    }

    /// `zzAtEOF`.
    pub fn at_eof(&self) -> bool {
        self.at_eof
    }

    /// `yyclose()`'s scanner half: at EOF, buffer invalidated.
    pub fn yyclose(&mut self) {
        self.at_eof = true;
        self.end_read = self.start_read;
    }

    /// `Character.offsetByCodePoints(zzBuffer, zzStartRead, zzEndRead -
    /// zzStartRead, index, offset)`, the lookahead actions' adjustment.
    pub fn offset_by_code_points(&self, index: usize, offset: i32) -> usize {
        let (start, limit) = (self.start_read, self.end_read);
        let mut x = index;
        if offset >= 0 {
            for _ in 0..offset {
                if x >= limit {
                    break;
                }
                let hi = self.buffer[x];
                x += 1;
                if is_high_surrogate(hi)
                    && x < limit
                    && crate::java_character::is_low_surrogate(self.buffer[x])
                {
                    x += 1;
                }
            }
        } else {
            for _ in 0..-offset {
                if x <= start {
                    break;
                }
                x -= 1;
                let lo = self.buffer[x];
                if crate::java_character::is_low_surrogate(lo)
                    && x > start
                    && is_high_surrogate(self.buffer[x - 1])
                {
                    x -= 1;
                }
            }
        }
        x
    }

    // Java: zzRefill
    fn refill(&mut self, reader: &mut dyn CharReader) -> Result<bool, AnalysisError> {
        if self.start_read > 0 {
            self.end_read += self.final_high_surrogate;
            self.final_high_surrogate = 0;
            self.buffer.copy_within(self.start_read..self.end_read, 0);
            self.end_read -= self.start_read;
            self.current_pos -= self.start_read;
            self.marked_pos -= self.start_read;
            self.start_read = 0;
        }
        if self.growable && self.current_pos >= self.buffer.len() - self.final_high_surrogate {
            // JFlex's standard skeleton: blow the buffer up.
            let n = self.buffer.len() * 2;
            self.buffer.resize(n, 0);
            self.end_read += self.final_high_surrogate;
            self.final_high_surrogate = 0;
        }
        let requested = if self.growable {
            // The standard skeleton asks for the whole rest of the buffer.
            self.buffer.len() - self.end_read
        } else {
            self.buffer.len() - self.end_read - self.final_high_surrogate
        };
        // (Unreachable for the growing skeleton, which grew the buffer above.)
        if requested == 0 {
            return Ok(true);
        }
        let end = self.end_read;
        let num_read = reader.read(&mut self.buffer[end..end + requested])?;
        if num_read == 0 {
            // Java: read() returning -1, the end of the stream.
            return Ok(true);
        }
        self.end_read += num_read;
        if is_high_surrogate(self.buffer[self.end_read - 1]) {
            if num_read == requested {
                self.end_read -= 1;
                self.final_high_surrogate = 1;
                if num_read == 1 && !self.growable {
                    return Ok(true);
                }
            } else {
                let n = reader.read(&mut self.one)?;
                if n == 0 {
                    return Ok(true);
                }
                self.buffer[self.end_read] = self.one[0];
                self.end_read += 1;
            }
        }
        Ok(false)
    }

    /// `getNextToken()` up to its action `switch`: matches from
    /// `zzMarkedPos`, updating `yychar`, `zzStartRead` and `zzMarkedPos`.
    pub fn scan(&mut self, reader: &mut dyn CharReader) -> Result<Scan, AnalysisError> {
        let t = self.tables;
        let mut marked = self.marked_pos;
        self.yychar += (marked - self.start_read) as i64;
        let mut action: i32 = -1;
        let mut current = marked;
        self.current_pos = marked;
        self.start_read = marked;
        let mut state = t.lexstate[self.lexical_state] as usize;
        let attributes = t.attribute[state];
        if attributes & 1 == 1 {
            action = state as i32;
        }
        let input: i32 = 'for_action: loop {
            let input;
            if current < self.end_read {
                let unit = self.buffer[current];
                if is_high_surrogate(unit) {
                    input = code_point_at(&self.buffer, current, self.end_read) as i32;
                    current += char_count(input as u32);
                } else {
                    input = i32::from(unit);
                    current += 1;
                    // The BMP fast path of the step below.
                    let next = t.trans[(t.rowmap[state] + t.cmap_bmp[usize::from(unit)]) as usize];
                    if next == -1 {
                        break 'for_action input;
                    }
                    state = next as usize;
                    let attributes = t.attribute[state];
                    if attributes & 1 == 1 {
                        action = state as i32;
                        marked = current;
                        if attributes & 8 == 8 {
                            break 'for_action input;
                        }
                    }
                    continue;
                }
            } else if self.at_eof {
                break 'for_action YYEOF;
            } else {
                self.current_pos = current;
                self.marked_pos = marked;
                let eof = self.refill(reader)?;
                current = self.current_pos;
                marked = self.marked_pos;
                if eof {
                    break 'for_action YYEOF;
                }
                input = code_point_at(&self.buffer, current, self.end_read) as i32;
                current += char_count(input as u32);
            }
            let next = t.trans[(t.rowmap[state] + t.cmap(input)) as usize];
            if next == -1 {
                break 'for_action input;
            }
            state = next as usize;
            let attributes = t.attribute[state];
            if attributes & 1 == 1 {
                action = state as i32;
                marked = current;
                if attributes & 8 == 8 {
                    break 'for_action input;
                }
            }
        };
        self.marked_pos = marked;
        if input == YYEOF && self.start_read == self.current_pos {
            self.at_eof = true;
            return Ok(Scan::Eof);
        }
        Ok(Scan::Action(if action < 0 {
            action
        } else {
            t.action[action as usize]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;

    fn zlib(raw: &[u8]) -> Vec<u8> {
        miniz_oxide::deflate::compress_to_vec_zlib(raw, 6)
    }

    fn table(out: &mut Vec<u8>, name: &str, values: &[i32]) {
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// A one-state scanner whose every code point is a one-unit match of
    /// action 1, and whose class map sends everything to class 0.
    fn tiny() -> Vec<u8> {
        let mut raw = b"JFLX".to_vec();
        table(&mut raw, "ZZ_LEXSTATE", &[0, 0]);
        table(&mut raw, "ZZ_CMAP_TOP", &vec![0; 0x110000 >> 8]);
        table(&mut raw, "ZZ_CMAP_BLOCKS", &[0; 256]);
        table(&mut raw, "ZZ_ACTION", &[0, 1]);
        table(&mut raw, "ZZ_ROWMAP", &[0, 1]);
        table(&mut raw, "ZZ_TRANS", &[1, -1]);
        table(&mut raw, "ZZ_ATTRIBUTE", &[0, 9]);
        zlib(&raw)
    }

    #[test]
    fn corrupt_tables_are_errors() {
        assert!(JFlexTables::load(b"not zlib").is_err());
        assert!(JFlexTables::load(&zlib(b"XXXX")).is_err());
        assert!(JFlexTables::load(&zlib(b"JFLX\x02")).is_err());
        let mut raw = b"JFLX".to_vec();
        table(&mut raw, "ZZ_WRONG", &[0]);
        assert!(JFlexTables::load(&zlib(&raw)).is_err());
        let mut raw = b"JFLX".to_vec();
        for name in TABLE_NAMES {
            table(&mut raw, name, &[0]);
        }
        assert!(JFlexTables::load(&zlib(&raw)).is_err(), "bad shapes");
        assert!(JFlexTables::load(&tiny()).is_ok());
        // A `ZZ_CMAP_TOP` entry past `ZZ_CMAP_BLOCKS`.
        let mut raw = b"JFLX".to_vec();
        table(&mut raw, "ZZ_LEXSTATE", &[0, 0]);
        let mut top = vec![0; 0x110000 >> 8];
        top[1] = 256;
        table(&mut raw, "ZZ_CMAP_TOP", &top);
        table(&mut raw, "ZZ_CMAP_BLOCKS", &[0; 256]);
        table(&mut raw, "ZZ_ACTION", &[0, 1]);
        table(&mut raw, "ZZ_ROWMAP", &[0, 1]);
        table(&mut raw, "ZZ_TRANS", &[1, -1]);
        table(&mut raw, "ZZ_ATTRIBUTE", &[0, 9]);
        let e = JFlexTables::load(&zlib(&raw)).unwrap_err().to_string();
        assert!(e.contains("ZZ_CMAP_BLOCKS"), "{e}");
    }

    #[test]
    fn scanner_reads_across_refills_and_surrogates() {
        let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
        // A fixed 3-unit buffer: pairs straddle refills.
        let mut s = JFlexScanner::new(tables, 3);
        let mut r = StrReader::new("a😀b😀");
        let mut units = Vec::new();
        loop {
            match s.scan(&mut r).unwrap() {
                Scan::Eof => break,
                Scan::Action(a) => {
                    assert_eq!(a, 1);
                    units.push(s.text().to_vec());
                    assert_eq!(s.yycharat(0), s.text()[0]);
                }
            }
        }
        assert_eq!(units.len(), 4);
        assert_eq!(String::from_utf16(&units.concat()).unwrap(), "a😀b😀");
        assert!(s.at_eof());
        assert_eq!(
            s.yychar, 6,
            "at the end, yychar is the input length in UTF-16 units"
        );
        assert!(s.yypushback(5).is_err());
        s.yypushback(0).unwrap();
        s.yyclose();
        s.set_buffer_size(8);
        s.yyreset();
        assert_eq!(s.buffer.len(), 8);
        s.set_buffer_size(2);
        s.buffer = vec![0; 9];
        s.yyreset();
        assert_eq!(s.buffer.len(), 2, "yyreset shrinks a grown buffer");
        // The growing skeleton doubles its buffer.
        let mut g = JFlexScanner::with_growth(tables, 2, true);
        let mut r = StrReader::new("😀😀😀");
        let mut n = 0;
        while let Scan::Action(_) = g.scan(&mut r).unwrap() {
            n += 1;
        }
        assert_eq!(n, 3);
    }

    /// A reader that hands out one unit per `read`.
    struct OneAtATime(StrReader);

    impl CharReader for OneAtATime {
        fn read(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
            let n = buf.len().min(1);
            self.0.read(&mut buf[..n])
        }
    }

    fn all_units(s: &mut JFlexScanner, r: &mut dyn CharReader) -> String {
        let mut units = Vec::new();
        while let Scan::Action(_) = s.scan(r).unwrap() {
            units.extend_from_slice(s.text());
        }
        String::from_utf16(&units).unwrap()
    }

    #[test]
    fn refill_edge_cases() {
        let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
        // A high surrogate read last is held back for the next refill.
        let mut s = JFlexScanner::new(tables, 2);
        assert_eq!(all_units(&mut s, &mut StrReader::new("a😀b")), "a😀b");
        // A one-unit refill that reads a lone high surrogate holds it back.
        let mut s = JFlexScanner::new(tables, 2);
        assert_eq!(all_units(&mut s, &mut StrReader::new("ab😀c")), "ab😀c");
        // A short read ending in a high surrogate reads its low half.
        let mut s = JFlexScanner::new(tables, 8);
        assert_eq!(
            all_units(&mut s, &mut OneAtATime(StrReader::new("😀x"))),
            "😀x"
        );
        let mut s = JFlexScanner::with_growth(tables, 8, true);
        assert_eq!(
            all_units(&mut s, &mut OneAtATime(StrReader::new("x😀"))),
            "x😀"
        );
    }

    #[test]
    fn offsets_by_code_points() {
        let tables: &'static JFlexTables = Box::leak(Box::new(JFlexTables::load(&tiny()).unwrap()));
        let mut s = JFlexScanner::new(tables, 16);
        s.buffer[..5].copy_from_slice(&"a😀b😀".encode_utf16().collect::<Vec<_>>()[..5]);
        s.end_read = 5;
        assert_eq!(s.offset_by_code_points(0, 2), 3);
        assert_eq!(s.offset_by_code_points(0, 9), 5);
        assert_eq!(s.offset_by_code_points(3, -1), 1);
        assert_eq!(s.offset_by_code_points(4, -9), 0);
        assert_eq!(s.yylength(), 0);
    }
}
