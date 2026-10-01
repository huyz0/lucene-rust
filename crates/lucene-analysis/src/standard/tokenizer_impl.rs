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
    /// Scanning a whole UTF-8 text ([`Self::get_next_token_utf8`]) rather
    /// than the reader's UTF-16 buffer: `zz_start_read`/`zz_marked_pos` are
    /// then absolute code-unit positions in the text, and these their byte
    /// positions.
    utf8: bool,
    utf8_start: usize,
    utf8_marked: usize,
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
            utf8: false,
            utf8_start: 0,
            utf8_marked: 0,
        }
    }

    /// Scan the whole text handed to [`Self::get_next_token_utf8`] from now
    /// until the next [`Self::yyreset`].
    pub(crate) fn set_utf8(&mut self) {
        self.utf8 = true;
    }

    /// Whether this scan reads a UTF-8 text ([`Self::set_utf8`]).
    pub(crate) fn is_utf8(&self) -> bool {
        self.utf8
    }

    /// The matched text of a UTF-8 scan over `text`.
    pub(crate) fn text_utf8<'t>(&self, text: &'t str) -> &'t str {
        &text[self.utf8_start..self.utf8_marked]
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
        self.utf8 = false;
        self.utf8_start = 0;
        self.utf8_marked = 0;
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
            if let Some(token_type) = action_type(zz_action)? {
                return Ok(token_type);
            }
        }
    }
}

impl StandardTokenizerImpl {
    /// [`Self::get_next_token`] over a whole text held as UTF-8, without
    /// reading it as UTF-16 code units: the same DFA walked over the same
    /// code points, with every position kept in code units (Java's `char`s)
    /// beside its byte position, so the token types, `yychar`, `yylength`
    /// and the matched text are exactly what reading the text through
    /// `zzRefill` would have produced.
    ///
    /// The one thing `zzRefill` adds over a plain walk of the text is its
    /// buffer of `ZZ_BUFFERSIZE` code units: a match is shifted to the start
    /// of the buffer when the scanner runs off its end, and once the match
    /// fills the whole buffer the refill reports end of input (`requested ==
    /// 0`), so a run longer than the buffer is cut. A supplementary
    /// character whose high surrogate would be the buffer's last unit is
    /// held back (`zzFinalHighSurrogate`), so the buffer then ends before
    /// it. Both are the same rule from the match's start: the scanner sees
    /// the code points that fit whole in `ZZ_BUFFERSIZE` units from where the
    /// match began, and anything past them reads as end of input -- which
    /// is what this applies. (A `&str` has no unpaired surrogates, so
    /// `zzRefill`'s read of a trailing low surrogate always succeeds.)
    pub(crate) fn get_next_token_utf8(&mut self, text: &str) -> Result<i32, AnalysisError> {
        let bytes = text.as_bytes();
        let dfa = dfa();
        loop {
            self.yychar += (self.zz_marked_pos - self.zz_start_read) as i64;
            let start_u = self.zz_marked_pos;
            let start_b = self.utf8_marked;
            self.zz_start_read = start_u;
            self.utf8_start = start_b;
            // ARITH: a buffer size is at most `MAX_TOKEN_LENGTH_LIMIT`, and a
            // position at most the text's length.
            let limit_u = start_u + self.zz_buffersize;

            let mut zz_state = ZZ_LEXSTATE[self.zz_lexical_state] as usize;
            let mut zz_action: i32 = -1;
            if (dfa.attribute[zz_state & 63] & 1) == 1 {
                zz_action = zz_state as i32;
            }
            let (mut cur_u, mut cur_b) = (start_u, start_b);
            let (mut marked_u, mut marked_b) = (start_u, start_b);
            let mut eof = self.zz_at_eof;
            while !eof {
                let Some(&b0) = bytes.get(cur_b) else {
                    eof = true;
                    break;
                };
                // `text` is valid UTF-8: the lead byte gives the length and
                // every continuation byte is present.
                let (cp, len_b, len_u) = if b0 < 0x80 {
                    (i32::from(b0), 1, 1)
                } else {
                    let cont = |k: usize| i32::from(bytes[cur_b + k] & 0x3F);
                    let b0 = i32::from(b0);
                    if b0 < 0xE0 {
                        (((b0 & 0x1F) << 6) | cont(1), 2, 1)
                    } else if b0 < 0xF0 {
                        (((b0 & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3, 1)
                    } else {
                        (
                            ((b0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
                            4,
                            2,
                        )
                    }
                };
                if cur_u + len_u > limit_u {
                    // The buffer is full: `zzRefill` reports end of input.
                    eof = true;
                    break;
                }
                cur_u += len_u;
                cur_b += len_b;
                let zz_next = dfa.next[zz_state & 63][zz_cmap(cp) & 31];
                if zz_next == DEAD {
                    break;
                }
                zz_state = usize::from(zz_next & 0x3F);
                if zz_next & ACCEPTING != 0 {
                    zz_action = zz_state as i32;
                    marked_u = cur_u;
                    marked_b = cur_b;
                    if zz_next & FINAL != 0 {
                        break;
                    }
                }
            }

            self.zz_marked_pos = marked_u;
            self.utf8_marked = marked_b;
            self.zz_current_pos = cur_u;
            self.zz_state = zz_state;

            if eof && cur_u == start_u {
                self.zz_at_eof = true;
                return Ok(YYEOF);
            }
            if let Some(token_type) = action_type(zz_action)? {
                return Ok(token_type);
            }
        }
    }
}

/// `getNextToken`'s action switch: the token type `zzAction` returns, or
/// `None` for the rule that ignores what it matched.
#[inline]
fn action_type(zz_action: i32) -> Result<Option<i32>, AnalysisError> {
    let action = if zz_action < 0 {
        zz_action
    } else {
        ZZ_ACTION[zz_action as usize] as i32
    };
    Ok(Some(match action {
        // Not numeric, word, ideographic, hiragana, emoji or SE Asian -- ignore it.
        1 => return Ok(None),
        2 => NUMERIC_TYPE,
        3 => WORD_TYPE,
        4 => EMOJI_TYPE,
        5 => SOUTH_EAST_ASIAN_TYPE,
        6 => HANGUL_TYPE,
        7 => IDEOGRAPHIC_TYPE,
        8 => KATAKANA_TYPE,
        9 => HIRAGANA_TYPE,
        _ => {
            return Err(AnalysisError::IllegalState(
                "Error: could not match input".to_string(),
            ))
        }
    }))
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
