//! `StandardTokenizerImpl`: Lucene 10.5.0's JFlex 1.8.2-generated UAX#29
//! word-break scanner, ported line for line.
//!
//! The DFA is not re-derived from the grammar: [`super::tables`] holds the
//! very `int[]` tables the Java class unpacks at class-init time, extracted
//! from the 10.5.0 jar by reflection
//! (`crates/lucene-analysis/tools/ExtractStandardTokenizerTables.java`). What
//! is ported here is the scanner loop that walks them -- `getNextToken`,
//! `zzRefill`, `zzCMap`, `yyreset`, `setBufferSize` -- with Java's control
//! flow and field names.
//!
//! # Units
//!
//! Like Java, the scanner reads **UTF-16 code units** from its reader into a
//! `u16` buffer and decodes code points with `Character.codePointAt(char[],
//! int, int)` semantics (an unpaired surrogate is its own code point). Every
//! position here -- `yychar`, `yylength`, the buffer indices -- is in code
//! units, so a token's offsets are Java's `char` offsets exactly; see
//! [`crate::reader::StrReader`] for how a Rust `&str` becomes that input.
//!
//! # Rust-forced changes
//!
//! - The reader is passed into [`StandardTokenizerImpl::get_next_token`]
//!   rather than stored: the owning [`super::StandardTokenizer`] owns it (as
//!   `Tokenizer.input`), and Java's scanner only ever holds an alias of that
//!   same field.
//! - `Reader.read` returning `-1` at end of input is
//!   [`crate::CharReader::read`] returning `0`, so JFlex's "Reader returned 0
//!   characters" error has no counterpart.
//! - `zzScanError(ZZ_NO_MATCH)` (an `Error` in Java, "can't possibly happen"
//!   with this grammar's catch-all rule) is an `Err`.

use super::tables::{
    ZZ_ACTION, ZZ_ATTRIBUTE, ZZ_CMAP_BLOCKS, ZZ_CMAP_TOP, ZZ_LEXSTATE, ZZ_ROWMAP, ZZ_TRANS,
};
use crate::{AnalysisError, CharReader};

/// `YYEOF`.
pub(crate) const YYEOF: i32 = -1;

/// `StandardTokenizer.ALPHANUM` .. `EMOJI`, as the scanner's actions return them.
pub(crate) const WORD_TYPE: i32 = 0;
pub(crate) const NUMERIC_TYPE: i32 = 1;
pub(crate) const SOUTH_EAST_ASIAN_TYPE: i32 = 2;
pub(crate) const IDEOGRAPHIC_TYPE: i32 = 3;
pub(crate) const HIRAGANA_TYPE: i32 = 4;
pub(crate) const KATAKANA_TYPE: i32 = 5;
pub(crate) const HANGUL_TYPE: i32 = 6;
pub(crate) const EMOJI_TYPE: i32 = 7;

/// `ZZ_BUFFERSIZE`'s initial value.
const INITIAL_BUFFER_SIZE: usize = 255;

/// `Character.isHighSurrogate`.
#[inline]
fn is_high_surrogate(c: u16) -> bool {
    (0xD800..=0xDBFF).contains(&c)
}

/// `Character.codePointAt(char[] a, int index, int limit)`.
#[inline]
fn code_point_at(a: &[u16], index: usize, limit: usize) -> i32 {
    let c1 = a[index];
    if is_high_surrogate(c1) && index + 1 < limit {
        let c2 = a[index + 1];
        if (0xDC00..=0xDFFF).contains(&c2) {
            return (((c1 as i32) - 0xD800) << 10) + ((c2 as i32) - 0xDC00) + 0x10000;
        }
    }
    c1 as i32
}

/// `Character.charCount(int)`.
#[inline]
fn char_count(cp: i32) -> usize {
    if cp >= 0x10000 {
        2
    } else {
        1
    }
}

/// No transition: `ZZ_TRANS`' `-1` (state 63 does not exist).
const DEAD: u8 = 0x3F;
/// Packed into a transition: the target state is accepting
/// (`ZZ_ATTRIBUTE & 1`).
const ACCEPTING: u8 = 0x40;
/// Packed into a transition: the target state ends the match
/// (`ZZ_ATTRIBUTE & 8`).
const FINAL: u8 = 0x80;

/// `ZZ_TRANS` read through `ZZ_ROWMAP` and `ZZ_ATTRIBUTE`, laid out as one
/// row of 32 classes per state (the scanner has 59 states and 29 classes),
/// so the scan loop's lookup per character indexes a fixed-size array by
/// masked values: no row offset to add and no bounds to check. Each entry is
/// the target state with its `ZZ_ATTRIBUTE` bits packed above it
/// ([`ACCEPTING`], [`FINAL`]), saving the dependent attribute load per
/// character. The same transitions; only the layout differs.
struct Dfa {
    next: [[u8; 32]; 64],
    attribute: [u8; 64],
}

fn dfa() -> &'static Dfa {
    static DFA: std::sync::OnceLock<Dfa> = std::sync::OnceLock::new();
    DFA.get_or_init(|| {
        let mut d = Dfa {
            next: [[DEAD; 32]; 64],
            attribute: [0; 64],
        };
        for (state, &row) in ZZ_ROWMAP.iter().enumerate() {
            d.attribute[state] = ZZ_ATTRIBUTE[state];
            for class in 0..32usize {
                if let Some(&next) = ZZ_TRANS.get(usize::from(row) + class) {
                    if class <= MAX_CLASS {
                        d.next[state][class] = match u8::try_from(next) {
                            Ok(t) => {
                                let attr = ZZ_ATTRIBUTE[usize::from(t)];
                                let mut packed = t;
                                if attr & 1 == 1 {
                                    packed |= ACCEPTING;
                                }
                                if attr & 8 == 8 {
                                    packed |= FINAL;
                                }
                                packed
                            }
                            Err(_) => DEAD,
                        };
                    }
                }
            }
        }
        d
    })
}

/// The largest character class `zzCMap` yields.
const MAX_CLASS: usize = 28;

/// `zzCMap`: raw input code point to DFA character class.
#[inline]
fn zz_cmap(input: i32) -> usize {
    let offset = (input & 255) as usize;
    if offset as i32 == input {
        ZZ_CMAP_BLOCKS[offset] as usize
    } else {
        ZZ_CMAP_BLOCKS[ZZ_CMAP_TOP[(input >> 8) as usize] as usize | offset] as usize
    }
}

/// Java's `StandardTokenizerImpl` scanner state.
pub(crate) struct StandardTokenizerImpl {
    /// `ZZ_BUFFERSIZE` (an instance field in Lucene's skeleton, so
    /// `setBufferSize` can change it).
    zz_buffersize: usize,
    zz_state: usize,
    zz_lexical_state: usize,
    zz_buffer: Vec<u16>,
    zz_marked_pos: usize,
    zz_current_pos: usize,
    zz_start_read: usize,
    zz_end_read: usize,
    zz_at_eof: bool,
    zz_final_high_surrogate: usize,
    yychar: i64,
}

impl StandardTokenizerImpl {
    pub(crate) fn new() -> Self {
        StandardTokenizerImpl {
            zz_buffersize: INITIAL_BUFFER_SIZE,
            zz_state: 0,
            zz_lexical_state: 0,
            zz_buffer: vec![0; INITIAL_BUFFER_SIZE],
            zz_marked_pos: 0,
            zz_current_pos: 0,
            zz_start_read: 0,
            zz_end_read: 0,
            zz_at_eof: false,
            zz_final_high_surrogate: 0,
            yychar: 0,
        }
    }

    /// `yychar()`: code units processed before the current match.
    pub(crate) fn yychar(&self) -> i32 {
        // Java: `(int) yychar` -- "jflex supports > 2GB docs but not lucene".
        self.yychar as i32
    }

    /// `yylength()`.
    pub(crate) fn yylength(&self) -> usize {
        self.zz_marked_pos - self.zz_start_read
    }

    /// The matched text, `getText(CharTermAttribute)`'s source.
    pub(crate) fn text(&self) -> &[u16] {
        &self.zz_buffer[self.zz_start_read..self.zz_marked_pos]
    }

    /// `setBufferSize(int numChars)`.
    pub(crate) fn set_buffer_size(&mut self, num_chars: usize) {
        self.zz_buffersize = num_chars;
        let mut new_buffer = vec![0u16; num_chars];
        let keep = self.zz_buffer.len().min(num_chars);
        new_buffer[..keep].copy_from_slice(&self.zz_buffer[..keep]);
        self.zz_buffer = new_buffer;
    }

    /// `yyreset(Reader)`: every position back to the start, the lexical
    /// state to `YYINITIAL`, the buffer shrunk back if it outgrew
    /// `ZZ_BUFFERSIZE`. (The reader itself is the tokenizer's.)
    pub(crate) fn yyreset(&mut self) {
        self.zz_at_eof = false;
        self.zz_current_pos = 0;
        self.zz_marked_pos = 0;
        self.zz_start_read = 0;
        self.zz_end_read = 0;
        self.zz_final_high_surrogate = 0;
        self.yychar = 0;
        self.zz_lexical_state = 0;
        if self.zz_buffer.len() > self.zz_buffersize {
            self.zz_buffer = vec![0; self.zz_buffersize];
        }
    }

    /// `zzRefill()`: returns `true` iff there was no new input.
    fn zz_refill(&mut self, reader: &mut dyn CharReader) -> Result<bool, AnalysisError> {
        // first: make room (if you can)
        if self.zz_start_read > 0 {
            self.zz_end_read += self.zz_final_high_surrogate;
            self.zz_final_high_surrogate = 0;
            self.zz_buffer
                .copy_within(self.zz_start_read..self.zz_end_read, 0);
            // translate stored positions
            self.zz_end_read -= self.zz_start_read;
            self.zz_current_pos -= self.zz_start_read;
            self.zz_marked_pos -= self.zz_start_read;
            self.zz_start_read = 0;
        }

        // fill the buffer with new input
        let requested = self.zz_buffer.len() - self.zz_end_read - self.zz_final_high_surrogate;
        if requested == 0 {
            return Ok(true);
        }
        let num_read = reader.read(&mut self.zz_buffer[self.zz_end_read..][..requested])?;
        if num_read > 0 {
            self.zz_end_read += num_read;
            if is_high_surrogate(self.zz_buffer[self.zz_end_read - 1]) {
                if num_read == requested {
                    // We requested too few chars to encode a full Unicode character
                    self.zz_end_read -= 1;
                    self.zz_final_high_surrogate = 1;
                    if num_read == 1 {
                        return Ok(true);
                    }
                } else {
                    // There is room in the buffer for at least one more char
                    let mut one = [0u16; 1];
                    // Expecting to read a paired low surrogate char
                    if reader.read(&mut one)? == 0 {
                        return Ok(true);
                    }
                    self.zz_buffer[self.zz_end_read] = one[0];
                    self.zz_end_read += 1;
                }
            }
            // potentially more input available
            return Ok(false);
        }
        // end of stream
        Ok(true)
    }

    /// `getNextToken()`: the next token type (`WORD_TYPE` .. `EMOJI_TYPE`),
    /// or [`YYEOF`].
    pub(crate) fn get_next_token(
        &mut self,
        reader: &mut dyn CharReader,
    ) -> Result<i32, AnalysisError> {
        let mut zz_input: i32;
        let mut zz_action: i32;

        // cached fields:
        let mut zz_current_pos_l: usize;
        let mut zz_marked_pos_l: usize;
        let mut zz_end_read_l = self.zz_end_read;
        let dfa = dfa();

        loop {
            zz_marked_pos_l = self.zz_marked_pos;

            self.yychar += (zz_marked_pos_l - self.zz_start_read) as i64;

            zz_action = -1;

            zz_current_pos_l = zz_marked_pos_l;
            self.zz_current_pos = zz_marked_pos_l;
            self.zz_start_read = zz_marked_pos_l;

            // `zzState`, kept in a local for the scan and stored back after it.
            let mut zz_state = ZZ_LEXSTATE[self.zz_lexical_state] as usize;

            // set up zzAction for empty match case:
            let zz_attributes = dfa.attribute[zz_state & 63];
            if (zz_attributes & 1) == 1 {
                zz_action = zz_state as i32;
            }

            // zzForAction:
            loop {
                if zz_current_pos_l < zz_end_read_l {
                    zz_input = code_point_at(&self.zz_buffer, zz_current_pos_l, zz_end_read_l);
                    zz_current_pos_l += char_count(zz_input);
                } else if self.zz_at_eof {
                    zz_input = YYEOF;
                    break;
                } else {
                    // store back cached positions
                    self.zz_current_pos = zz_current_pos_l;
                    self.zz_marked_pos = zz_marked_pos_l;
                    let eof = self.zz_refill(reader)?;
                    // get translated positions and possibly new buffer
                    zz_current_pos_l = self.zz_current_pos;
                    zz_marked_pos_l = self.zz_marked_pos;
                    zz_end_read_l = self.zz_end_read;
                    if eof {
                        zz_input = YYEOF;
                        break;
                    }
                    zz_input = code_point_at(&self.zz_buffer, zz_current_pos_l, zz_end_read_l);
                    zz_current_pos_l += char_count(zz_input);
                }
                let zz_next = dfa.next[zz_state & 63][zz_cmap(zz_input) & 31];
                if zz_next == DEAD {
                    break;
                }
                zz_state = usize::from(zz_next & 0x3F);

                if zz_next & ACCEPTING != 0 {
                    zz_action = zz_state as i32;
                    zz_marked_pos_l = zz_current_pos_l;
                    if zz_next & FINAL != 0 {
                        break;
                    }
                }
            }

            // store back cached position
            self.zz_marked_pos = zz_marked_pos_l;
            self.zz_state = zz_state;

            if zz_input == YYEOF && self.zz_start_read == self.zz_current_pos {
                self.zz_at_eof = true;
                return Ok(YYEOF);
            }
            let action = if zz_action < 0 {
                zz_action
            } else {
                ZZ_ACTION[zz_action as usize] as i32
            };
            match action {
                // Not numeric, word, ideographic, hiragana, emoji or SE Asian -- ignore it.
                1 => {}
                2 => return Ok(NUMERIC_TYPE),
                3 => return Ok(WORD_TYPE),
                4 => return Ok(EMOJI_TYPE),
                5 => return Ok(SOUTH_EAST_ASIAN_TYPE),
                6 => return Ok(HANGUL_TYPE),
                7 => return Ok(IDEOGRAPHIC_TYPE),
                8 => return Ok(KATAKANA_TYPE),
                9 => return Ok(HIRAGANA_TYPE),
                _ => {
                    return Err(AnalysisError::IllegalState(
                        "Error: could not match input".to_string(),
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_point_at_follows_java() {
        let a = [0xD83Du16, 0xDE00, 0xD800, b'x' as u16, 0xDC00];
        assert_eq!(code_point_at(&a, 0, 5), 0x1F600);
        // limit cuts the pair: the high surrogate alone.
        assert_eq!(code_point_at(&a, 0, 1), 0xD83D);
        // unpaired high, then a lone low surrogate.
        assert_eq!(code_point_at(&a, 2, 5), 0xD800);
        assert_eq!(code_point_at(&a, 4, 5), 0xDC00);
        assert_eq!(char_count(0x1F600), 2);
        assert_eq!(char_count(0xD800), 1);
    }

    /// `zzCMap` over every code point 0..=0x10FFFF against Java's own
    /// (`GenStandardTokenizer`'s `cmap_runs.txt`, run-length encoded).
    #[test]
    fn cmap_matches_java_for_every_code_point() {
        let runs = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/standard_tokenizer/cmap_runs.txt"
        ))
        .expect("run GenStandardTokenizer");
        let runs: Vec<(i32, usize)> = runs
            .lines()
            .map(|l| {
                let (cp, class) = l.split_once(' ').unwrap();
                (i32::from_str_radix(cp, 16).unwrap(), class.parse().unwrap())
            })
            .collect();
        assert_eq!(runs[0].0, 0);
        for (i, &(start, class)) in runs.iter().enumerate() {
            let end = runs.get(i + 1).map_or(0x110000, |r| r.0);
            for cp in start..end {
                assert_eq!(zz_cmap(cp), class, "code point {cp:#x}");
            }
        }
    }

    #[test]
    fn tables_have_the_jflex_shapes() {
        assert_eq!(ZZ_CMAP_TOP.len(), 0x110000 >> 8);
        assert_eq!(ZZ_ACTION.len(), ZZ_ROWMAP.len());
        assert_eq!(ZZ_ATTRIBUTE.len(), ZZ_ROWMAP.len());
        // `Dfa` packs a state into six bits, with `DEAD` the one left over.
        assert!(ZZ_ROWMAP.len() < usize::from(DEAD));
        assert!(ZZ_ATTRIBUTE.iter().all(|&a| a & !9 == 0));
        // every transition target is a state or -1
        assert!(ZZ_TRANS
            .iter()
            .all(|&t| t == -1 || (t as usize) < ZZ_ROWMAP.len()));
        // ASCII letters and digits are not in the same class
        assert_ne!(zz_cmap('a' as i32), zz_cmap('1' as i32));
        assert_eq!(zz_cmap('a' as i32), zz_cmap('b' as i32));
    }
}
