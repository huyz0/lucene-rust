//! Port of `org.apache.lucene.util.compress.LZ4` — both halves.
//!
//! `pub` for the same reason [`crate::for_util`] is: `LZ4` is a public class
//! in Lucene, both hash-table strategies are part of its contract, and a
//! compressor this hot deserves a microbenchmark that can call it from
//! outside the crate.
//!
//! [`decompress`] handles the standard LZ4 block format (token byte,
//! optional extended literal/match lengths, 16-bit little-endian match
//! offset), self-terminating once `decompressed_len` bytes have been
//! produced — the caller never needs to know the *compressed* length
//! up front.
//!
//! `dest`/`d_off` mirror Java's signature exactly (rather than taking a
//! plain `&mut [u8]` starting at 0): [`crate::stored_fields`]'s preset-dictionary
//! scheme decompresses into a buffer that already has dictionary bytes sitting
//! before `d_off`, and match back-references are allowed to reach into that
//! region.
//!
//! [`compress_with_dictionary`] is the full port of Java's
//! `LZ4.compressWithDictionary`, including the preset-dictionary window that
//! `LZ4WithPresetDictCompressionMode` (stored fields' `Mode.BEST_SPEED`)
//! relies on, and both match-finding strategies:
//! [`FastCompressionHashTable`] (one last-occurrence per hash; what
//! `CompressionMode.FAST` and `LZ4WithPresetDictCompressionMode` use) and
//! [`HighCompressionHashTable`] (a 256-deep hash chain over the last 64kB;
//! what `Lucene103BlockTreeTermsWriter` uses for term-suffix compression).
//! Both are *reusable* across calls exactly as Java's are — the table is
//! never cleared between inputs, `get` re-verifies every candidate
//! byte-for-byte instead, which is what keeps compressing many small
//! chunks cheap.

use lucene_store::data_input::DataInput;
use lucene_store::{Error, Result};

pub(crate) const MIN_MATCH: usize = 4;
const LAST_LITERALS: usize = 5;
pub const MAX_DISTANCE: usize = 1 << 16;
/// `LZ4.MEMORY_USAGE`: the fast table is sized so it always costs 16kB,
/// whatever the per-slot width works out to.
const MEMORY_USAGE: u32 = 14;
/// `LZ4.HASH_LOG_HC` / `HASH_TABLE_SIZE_HC`.
const HASH_LOG_HC: u32 = 15;
const HASH_TABLE_SIZE_HC: usize = 1 << HASH_LOG_HC;
/// `HighCompressionHashTable.MAX_ATTEMPTS`.
const MAX_ATTEMPTS: usize = 256;
/// `HighCompressionHashTable.MASK`.
const HC_MASK: usize = MAX_DISTANCE - 1;

/// [`decompress`] over an in-memory input: the same block format and the same
/// result, several times faster on the short sequences real stored fields and
/// term suffixes are made of.
///
/// [`decompress`] moves every literal run and every match with a
/// variable-length `copy_from_slice`/`copy_within` -- a `memmove` call per
/// sequence, whose setup dwarfs the few bytes an LZ4 sequence usually
/// carries. With the whole compressed block visible as a slice, the common
/// sequence (literal run under 15 bytes, match under 19 bytes at a distance of
/// at least 8) is instead moved as fixed-size 16- and 8-byte blocks (two
/// 16-byte moves for a match at a distance of 16 or more), which
/// compile to plain vector/word loads and stores -- the "wild copy" of the
/// reference LZ4 decoder. A fixed-size copy may write a few bytes past the
/// sequence's end; it is only taken while those bytes are still inside
/// `dest[..d_off + decompressed_len]`, which the rest of the block overwrites
/// before anyone can observe them. Near the end of the block, and for any
/// long or overlapping sequence, it falls back to the exact copies.
///
/// All of it is bounds-checked safe code: each fixed-size copy costs one
/// slice check, not a check per byte. The loop itself ([`decode_block`])
/// reports failures as a small [`Bad`] and holds no reference to `input`, so
/// the common path keeps its cursors in registers; the `Error` is built once,
/// on the way out.
pub fn decompress_slice(
    input: &mut lucene_store::data_input::SliceInput<'_>,
    decompressed_len: usize,
    dest: &mut [u8],
    d_off: usize,
) -> Result<usize> {
    let dest_end = d_off
        .checked_add(decompressed_len)
        .ok_or(Error::Eof { offset: d_off })?;
    if dest_end > dest.len() {
        return Err(Error::Eof { offset: dest_end });
    }
    let at = input.position();
    match decode_block(input.as_slice(), dest, d_off, dest_end) {
        Ok((s, d)) => {
            input.skip(s)?;
            Ok(d)
        }
        Err(Bad::Eof(s)) => Err(Error::Eof {
            offset: at.saturating_add(s),
        }),
        Err(Bad::Corrupt(m)) => Err(Error::Corrupted(m.into())),
    }
}

/// Why [`decode_block`] stopped short: the input ended at this offset, or
/// the block is not valid LZ4.
enum Bad {
    Eof(usize),
    Corrupt(&'static str),
}

/// `dest[to..to + 16] = src[from..from + 16]`, as one 16-byte load and store.
// ARITH: callers keep `from + 16 <= src.len()` and `to + 16 <= dest.len()`.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn copy16(dest: &mut [u8], to: usize, src: &[u8], from: usize) {
    let mut a = [0u8; 16];
    a.copy_from_slice(&src[from..from + 16]);
    dest[to..to + 16].copy_from_slice(&a);
}

/// `N` bytes of `dest` from `from` to `to`, read whole before any is written:
/// a plain move when `to - from >= N`, as every caller keeps.
// ARITH: callers keep `from + N <= to` and `to + N <= dest.len()`.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn move_block<const N: usize>(dest: &mut [u8], from: usize, to: usize) {
    let mut a = [0u8; N];
    a.copy_from_slice(&dest[from..from + N]);
    dest[to..to + N].copy_from_slice(&a);
}

/// [`read_length_extension`] over a slice cursor.
// ARITH: `*s < src.len()` is proved by the `get` before the increment.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn extension(src: &[u8], s: &mut usize, mut len: usize) -> std::result::Result<usize, Bad> {
    loop {
        let b = *src.get(*s).ok_or(Bad::Eof(*s))?;
        *s += 1;
        len = len
            .checked_add(b as usize)
            .ok_or(Bad::Corrupt("LZ4 length extension overflows"))?;
        if b != 0xFF {
            return Ok(len);
        }
    }
}

/// [`decompress_slice`]'s loop over `src`, the compressed block: decodes into
/// `dest[d..dest_end]` and returns the bytes of `src` it read and the end it
/// wrote to.
// ARITH: for the whole loop -- `s` only advances past bytes `src.get` just
// proved exist, or by a `lit` that `s + lit <= src.len()` was checked for, so
// `s <= src.len()` and `s + k` for a small constant `k` is far from overflow.
// `d` only advances to an end checked against `dest.len()` (or under
// `d + 32 <= dest_end <= dest.len()`), so `d + 16`/`d + 32` are small sums of
// in-bounds indices. `d - dec` is taken only under `dec <= d`, and each
// fixed-size move is taken only at a distance its length allows.
#[allow(clippy::arithmetic_side_effects)]
fn decode_block(
    src: &[u8],
    dest: &mut [u8],
    mut d: usize,
    dest_end: usize,
) -> std::result::Result<(usize, usize), Bad> {
    let mut s = 0usize;
    loop {
        let token = *src.get(s).ok_or(Bad::Eof(s))? as usize;
        s += 1;
        let mut lit = token >> 4;
        if lit < 0x0F && s + 16 <= src.len() && d + 16 <= dest_end {
            copy16(dest, d, src, s);
        } else {
            if lit == 0x0F {
                lit = extension(src, &mut s, lit)?;
            }
            let end = d.checked_add(lit).ok_or(Bad::Eof(s))?;
            let s_end = s.checked_add(lit).ok_or(Bad::Eof(s))?;
            let run = src.get(s..s_end).ok_or(Bad::Eof(s))?;
            dest.get_mut(d..end)
                .ok_or(Bad::Eof(s))?
                .copy_from_slice(run);
        }
        s += lit;
        d += lit;
        if d >= dest_end {
            return Ok((s, d));
        }
        let (Some(&lo), Some(&hi)) = (src.get(s), src.get(s + 1)) else {
            return Err(Bad::Eof(s));
        };
        let dec = u16::from_le_bytes([lo, hi]) as usize;
        s += 2;
        if dec == 0 {
            return Err(Bad::Corrupt("LZ4 match offset 0 is invalid"));
        }
        if dec > d {
            return Err(Bad::Corrupt(
                "LZ4 match references before the start of the buffer",
            ));
        }
        let from = d - dec;
        let mut ml = token & 0x0F;
        if ml != 0x0F && d + 32 <= dest_end {
            if dec >= 16 {
                move_block::<16>(dest, from, d);
                move_block::<16>(dest, from + 16, d + 16);
                d += ml + MIN_MATCH;
                continue;
            }
            if dec >= 8 {
                move_block::<8>(dest, from, d);
                move_block::<8>(dest, from + 8, d + 8);
                move_block::<8>(dest, from + 16, d + 16);
                d += ml + MIN_MATCH;
                continue;
            }
        }
        if ml == 0x0F {
            ml = extension(src, &mut s, ml)?;
        }
        let ml = ml
            .checked_add(MIN_MATCH)
            .ok_or(Bad::Corrupt("LZ4 match length overflows"))?;
        let end = d.checked_add(ml).ok_or(Bad::Eof(s))?;
        if end > dest.len() {
            return Err(Bad::Eof(s));
        }
        copy_match(dest, from, d, ml);
        d = end;
        if d >= dest_end {
            return Ok((s, d));
        }
    }
}

/// Copies an LZ4 match: `len` bytes to `dest[to..]` from `dest[from..]`,
/// `from < to`, where the ranges may overlap (`to - from < len`: the way
/// LZ4 encodes a run, each output byte may repeat one just written).
///
/// Bulk `copy_within` rather than Java's `System.arraycopy`-or-byte-loop
/// split. Without overlap that is one copy. With it, the bytes from `from`
/// up to the write position are periodic with period `to - from`, so the
/// copy can take as source everything already written that keeps the phase
/// -- a whole number of periods -- and double the run each step: `log2(len /
/// period)` memmoves instead of `len / period` (one per *byte* for a period
/// of 1, which is a plain `fill`).
///
/// The caller has checked `to + len <= dest.len()` and `from < to`.
// ARITH: `from < to` (so `period >= 1`) and `to + len <= dest.len()` are the
// caller's preconditions. `written < len` guards each step, so `len -
// written` cannot underflow; `n <= period + written` keeps the source range
// `from..from + n` inside the bytes already final (`from + period + written
// == to + written`), and `to + written + n <= to + len`.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn copy_match(dest: &mut [u8], from: usize, to: usize, len: usize) {
    let period = to - from;
    if period >= len {
        dest.copy_within(from..from + len, to);
        return;
    }
    if period == 1 {
        let b = dest[from];
        dest[to..to + len].fill(b);
        return;
    }
    let mut written = 0usize;
    while written < len {
        // `period + written` is a whole number of periods whenever
        // `written` is, which every step but the last keeps.
        let n = (period + written).min(len - written);
        dest.copy_within(from..from + n, to + written);
        written += n;
    }
}

/// Decompresses into `dest[d_off..d_off+decompressed_len]`, reading a
/// self-terminating LZ4 block from `input`. Back-references may reach
/// earlier into `dest` than `d_off` (a preset dictionary). Returns
/// `d_off + decompressed_len` (mirroring Java's return value, the new
/// write position) on success.
pub fn decompress(
    input: &mut impl DataInput,
    decompressed_len: usize,
    dest: &mut [u8],
    d_off: usize,
) -> Result<usize> {
    let dest_end = d_off
        .checked_add(decompressed_len)
        .ok_or(Error::Eof { offset: d_off })?;
    if dest_end > dest.len() {
        return Err(Error::Eof { offset: dest_end });
    }
    let mut d_off = d_off;

    loop {
        let token = input.read_byte()? as usize;
        let mut literal_len = token >> 4;
        if literal_len == 0x0F {
            literal_len = read_length_extension(input, literal_len)?;
        }
        if literal_len != 0 {
            let end = d_off
                .checked_add(literal_len)
                .ok_or(Error::Eof { offset: d_off })?;
            if end > dest.len() {
                return Err(Error::Eof { offset: end });
            }
            input.read_bytes(&mut dest[d_off..end])?;
            d_off = end;
        }

        if d_off >= dest_end {
            break;
        }

        let match_dec = input.read_u16()? as usize;
        if match_dec == 0 {
            return Err(Error::Corrupted("LZ4 match offset 0 is invalid".into()));
        }

        let mut match_len = token & 0x0F;
        if match_len == 0x0F {
            match_len = read_length_extension(input, match_len)?;
        }
        let Some(match_len) = match_len.checked_add(MIN_MATCH) else {
            return Err(Error::Corrupted("LZ4 match length overflows".into()));
        };

        if match_dec > d_off {
            return Err(Error::Corrupted(
                "LZ4 match references before the start of the buffer".into(),
            ));
        }
        // ARITH: `match_dec <= d_off` was just checked.
        #[allow(clippy::arithmetic_side_effects)]
        let src_start = d_off - match_dec;
        let end = d_off
            .checked_add(match_len)
            .ok_or(Error::Eof { offset: d_off })?;
        if end > dest.len() {
            return Err(Error::Eof { offset: end });
        }
        copy_match(dest, src_start, d_off, match_len);
        d_off = end;

        if d_off >= dest_end {
            break;
        }
    }

    Ok(d_off)
}

/// Reads an LZ4 length extension: 0xFF bytes accumulate 255 each until a byte
/// below 0xFF ends the run. Every increment consumes a byte of input, so a
/// well-formed block cannot overflow -- but "every increment consumes a byte"
/// is a statement about the *file*, not about `usize`, and a long enough
/// stream of 0xFF would wrap the accumulator to a small length and decode a
/// silently truncated block. Reported as corruption instead.
fn read_length_extension(input: &mut impl DataInput, base: usize) -> Result<usize> {
    let mut len = base;
    loop {
        let b = input.read_byte()?;
        let Some(next) = len.checked_add(b as usize) else {
            return Err(Error::Corrupted("LZ4 length extension overflows".into()));
        };
        len = next;
        if b != 0xFF {
            return Ok(len);
        }
    }
}

/// Reads 4 bytes at `buf[i..i+4]` as a native-endian `u32` for hashing/
/// comparison purposes only (mirrors Java's comment on `readInt`: LZ4's
/// algorithm doesn't care about endianness here since these bytes are never
/// written to the output -- only compared to each other and hashed).
// ARITH: every caller indexes a buffer it has already bounded (`get`/
// `previous`/`add_hash` only ever pass an offset below `match_limit`, which is
// `end - LAST_LITERALS - MIN_MATCH`), so `i + 4` is in range; the slice index
// would panic before the addition could overflow anyway, since a slice length
// is at most `isize::MAX`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn read4(buf: &[u8], i: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[i..i + 4]);
    u32::from_ne_bytes(b)
}

/// Port of `LZ4.hash` (the multiplicative hash used by both hash-table
/// variants): `(i * -1640531535) >>> (32 - hashBits)` in Java's `int` math,
/// i.e. wrapping `u32` multiplication by the same constant reinterpreted as
/// unsigned (`0x9E3779B1`).
// ARITH: `hash_bits` is either `HASH_LOG_HC` (15) or
// `FastCompressionHashTable::hash_log`, which `reset` sets to
// `MEMORY_USAGE + 3 - {4,5}` = 12 or 13. Both are well under 32, so
// `32 - hash_bits` is a legal shift.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn hash(v: u32, hash_bits: u32) -> u32 {
    v.wrapping_mul(0x9E3779B1) >> (32 - hash_bits)
}

/// Port of `Arrays.mismatch(b, o1, limit, b, o2, limit)`: the number of
/// leading bytes that agree between `b[o1..limit]` and `b[o2..limit]`. Java's
/// `commonBytes` asserts this is never -1 (i.e. never "fully equal" up to
/// `limit`) because the two regions always end up with differing lengths
/// available before `limit`; this port doesn't rely on that invariant, it
/// just stops at whichever bound is reached first.
///
/// `Arrays.mismatch` is a JIT intrinsic that compares a vector register at a
/// time; the equivalent here is the 8-byte word loop below (`u64::from_le_bytes`
/// pins the interpretation so `trailing_zeros` names the first differing byte
/// on either endianness), with the byte loop kept only for the tail.
// ARITH: this is the encode side, over a buffer the compressor owns. Both
// callers pass `o1`/`o2` below `limit` (they are `match_ref + MIN_MATCH` and
// `off + MIN_MATCH` with `off < match_limit == limit - MIN_MATCH`), so
// `limit - o1` and `limit - o2` cannot underflow; `n` is bounded by `max`,
// which is bounded by `limit`, so every `o + n` index is inside `b`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn common_bytes(b: &[u8], o1: usize, o2: usize, limit: usize) -> usize {
    let max = (limit - o1).min(limit - o2);
    let mut n = 0;
    while n + 8 <= max {
        let x = u64::from_le_bytes(b[o1 + n..o1 + n + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[o2 + n..o2 + n + 8].try_into().unwrap());
        if x != y {
            return n + ((x ^ y).trailing_zeros() / 8) as usize;
        }
        n += 8;
    }
    while n < max && b[o1 + n] == b[o2 + n] {
        n += 1;
    }
    n
}

// ARITH: `l` only decreases, by 255 while it is at least 255.
#[allow(clippy::arithmetic_side_effects)]
fn encode_len(mut l: usize, out: &mut Vec<u8>) {
    while l >= 0xFF {
        out.push(0xFF);
        l -= 0xFF;
    }
    out.push(l as u8);
}

// ARITH: encode side. `literal_len >= 0x0F` guards the subtraction, and
// `anchor + literal_len` is the match offset (or `end`) the compressor
// computed from `bytes`'s own length.
#[allow(clippy::arithmetic_side_effects)]
fn encode_literals(bytes: &[u8], token: u8, anchor: usize, literal_len: usize, out: &mut Vec<u8>) {
    out.push(token);
    if literal_len >= 0x0F {
        encode_len(literal_len - 0x0F, out);
    }
    out.extend_from_slice(&bytes[anchor..anchor + literal_len]);
}

pub(crate) fn encode_last_literals(
    bytes: &[u8],
    anchor: usize,
    literal_len: usize,
    out: &mut Vec<u8>,
) {
    let token = (literal_len.min(0x0F) as u8) << 4;
    encode_literals(bytes, token, anchor, literal_len, out);
}

// ARITH: encode side. `match_off >= anchor` and `match_off > match_ref` are
// both compressor invariants (`anchor` trails `off`, and `HashTable::get`
// only returns a candidate strictly below `off`); `match_len >= MIN_MATCH` by
// construction, and the `>= MIN_MATCH + 0x0F` guard covers the second
// subtraction.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn encode_sequence(
    bytes: &[u8],
    anchor: usize,
    match_ref: usize,
    match_off: usize,
    match_len: usize,
    out: &mut Vec<u8>,
) {
    let literal_len = match_off - anchor;
    debug_assert!(match_len >= MIN_MATCH);
    let token = ((literal_len.min(0x0F) as u8) << 4) | (match_len - MIN_MATCH).min(0x0F) as u8;
    encode_literals(bytes, token, anchor, literal_len, out);

    let match_dec = match_off - match_ref;
    debug_assert!(match_dec > 0 && match_dec < (1 << 16));
    out.extend_from_slice(&(match_dec as u16).to_le_bytes());

    if match_len >= MIN_MATCH + 0x0F {
        encode_len(match_len - 0x0F - MIN_MATCH, out);
    }
}

/// Port of `LZ4.HashTable`: a record of previous occurrences of 4-byte
/// sequences, reusable across compression calls (`reset` never clears the
/// table -- `get` verifies candidates instead).
pub trait HashTable {
    /// `HashTable.reset`: prepare to compress `bytes[off..off+len]`.
    fn reset(&mut self, bytes: &[u8], off: usize, len: usize);
    /// `HashTable.initDictionary`: index the first `dict_len` bytes.
    fn init_dictionary(&mut self, bytes: &[u8], dict_len: usize);
    /// `HashTable.get`: an index storing the same 4 bytes as `bytes[off..off+4]`,
    /// or `None`. Only ever called on strictly increasing `off`.
    fn get(&mut self, bytes: &[u8], off: usize) -> Option<usize>;
    /// `HashTable.previous`: an earlier index storing the same 4 bytes as
    /// `bytes[off..off+4]`, or `None`. Unlike [`Self::get`] it needn't be
    /// called on increasing offsets.
    fn previous(&mut self, bytes: &[u8], off: usize) -> Option<usize>;
}

/// Port of `LZ4.FastCompressionHashTable`: one last occurrence per hash, in
/// 16kB of memory regardless of input size.
///
/// Java keeps two physical shapes (`Table16`/`Table32`) purely to hold that
/// 16kB budget while widening the stored offset for inputs above 64kB; the
/// offset is stored relative to `base` either way, so a single `u32` slot
/// here is never *less* precise than Java's and reproduces the same
/// candidate at every step. What is preserved exactly is the **hash log**
/// (`MEMORY_USAGE + 3 - bitsPerOffsetLog`: 13 below 64kB, 12 above), since
/// that decides which offsets collide and therefore which matches get found.
///
/// The table is deliberately *not* cleared by [`HashTable::reset`], matching
/// Java: a stale entry is either rejected by the `ref < off` /
/// `off - ref < MAX_DISTANCE` guards or by the byte-for-byte `read4`
/// re-check, so clearing would only make compressing many short chunks
/// (exactly what stored fields does) needlessly expensive. A freshly
/// allocated all-zero table is likewise fine -- slot value 0 means "offset
/// `base`", a legitimate candidate that gets verified like any other.
#[derive(Default)]
pub struct FastCompressionHashTable {
    table: Vec<u32>,
    hash_log: u32,
    /// The offset width (16 or 32) Java's packed table would have been
    /// allocated with; see [`HashTable::reset`].
    bits_per_offset: u32,
    base: usize,
}

impl FastCompressionHashTable {
    pub fn new() -> Self {
        Self::default()
    }
}

impl HashTable for FastCompressionHashTable {
    // ARITH: `bits_per_offset_log` is 4 or 5 and `MEMORY_USAGE` is 14, so
    // `hash_log` is 12 or 13 and `1usize << hash_log` is well in range.
    #[allow(clippy::arithmetic_side_effects)]
    fn reset(&mut self, _bytes: &[u8], off: usize, len: usize) {
        // Java: bitsPerOffset = 16 below 64kB else 32; bitsPerOffsetLog =
        // ceil(log2(bitsPerOffset)) = 4 or 5; hashLog = MEMORY_USAGE + 3 - that.
        let (bits_per_offset, bits_per_offset_log) =
            if len.saturating_sub(LAST_LITERALS) < (1 << 16) {
                (16, 4)
            } else {
                (32, 5)
            };
        self.hash_log = MEMORY_USAGE + 3 - bits_per_offset_log;
        // Java reallocates -- zeroed -- whenever its packed table is too
        // small *or too narrow*: `hashTable.size() < 1 << hashLog ||
        // hashTable.getBitsPerValue() < bitsPerOffset`. A switch from 16- to
        // 32-bit offsets therefore drops every stale entry even though the
        // smaller hash log would fit. Reallocating at the same points keeps
        // the table's contents Java's at every step rather than relying on
        // verification to reject whatever stale entry the wider `u32` slots
        // would otherwise have kept.
        if self.table.len() < 1usize << self.hash_log || self.bits_per_offset < bits_per_offset {
            self.table = vec![0u32; 1usize << self.hash_log];
            self.bits_per_offset = bits_per_offset;
        }
        self.base = off;
    }

    // ARITH: `base + i` for `i < dict_len` addresses the dictionary window the
    // caller already proved is inside `bytes` (`compress_with_dictionary`
    // computes `end = dict_off + dict_len + len` against `bytes`'s length).
    #[allow(clippy::arithmetic_side_effects)]
    fn init_dictionary(&mut self, bytes: &[u8], dict_len: usize) {
        for i in 0..dict_len {
            let v = read4(bytes, self.base + i);
            let h = hash(v, self.hash_log) as usize;
            self.table[h] = i as u32;
        }
    }

    // ARITH: `off >= base` (the compressor scans forward from `dict_off`,
    // which `reset` stored as `base`), so `off - base` cannot underflow;
    // `base + prev` is a previously stored offset, and `candidate < off` is
    // checked before `off - candidate`.
    #[allow(clippy::arithmetic_side_effects)]
    fn get(&mut self, bytes: &[u8], off: usize) -> Option<usize> {
        let v = read4(bytes, off);
        let h = hash(v, self.hash_log) as usize;
        let prev = self.table[h] as usize;
        self.table[h] = (off - self.base) as u32;
        let candidate = self.base + prev;
        // `candidate < off` is checked first, so `read4` never reads past the
        // region already scanned (and never past `bytes`).
        if candidate < off && off - candidate < MAX_DISTANCE && read4(bytes, candidate) == v {
            Some(candidate)
        } else {
            None
        }
    }

    fn previous(&mut self, _bytes: &[u8], _off: usize) -> Option<usize> {
        None
    }
}

/// Port of `LZ4.HighCompressionHashTable`: up to 256 occurrences of each
/// 4-byte sequence within the last 64kB, chained through `chain_table`, which
/// makes it far likelier to find (and, via [`HashTable::previous`], to keep
/// searching for a *longer*) match than [`FastCompressionHashTable`].
///
/// This is what `Lucene103BlockTreeTermsWriter` uses to compress a terms
/// block's suffix bytes, and what `CompressionMode.FAST_DECOMPRESSION` uses.
pub struct HighCompressionHashTable {
    hash_table: Vec<i32>,
    chain_table: Vec<u16>,
    base: usize,
    next: usize,
    end: usize,
    attempts: usize,
}

impl Default for HighCompressionHashTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HighCompressionHashTable {
    pub fn new() -> Self {
        Self {
            hash_table: vec![-1i32; HASH_TABLE_SIZE_HC],
            chain_table: vec![0xFFFFu16; MAX_DISTANCE],
            base: 0,
            next: 0,
            end: 0,
            attempts: 0,
        }
    }

    // ARITH: the deltas are computed in `i64` from `usize` offsets bounded by
    // the input buffer, and the result is clamped into `1..MAX_DISTANCE`
    // before the `as u16`.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_hash(&mut self, bytes: &[u8], off: usize) {
        let v = read4(bytes, off);
        let h = hash(v, HASH_LOG_HC) as usize;
        let prev = self.hash_table[h];
        let mut delta = off as i64 - prev as i64;
        if delta <= 0 || delta >= MAX_DISTANCE as i64 {
            delta = MAX_DISTANCE as i64 - 1;
        }
        self.chain_table[off & HC_MASK] = delta as u16;
        self.hash_table[h] = off as i32;
    }
}

impl HashTable for HighCompressionHashTable {
    // ARITH: `end >= base` always (both are set together at the end of this
    // function, and `end = off + len` with `off = base`), so `end - base` and
    // `end - 1` (guarded by `end == 0`) cannot underflow. `off + len` is the
    // caller's own buffer bound. This is the encode side throughout.
    #[allow(clippy::arithmetic_side_effects)]
    fn reset(&mut self, _bytes: &[u8], off: usize, len: usize) {
        if self.end - self.base < self.chain_table.len() {
            // The last call to compress was on less than 64kB: only the
            // touched part of the chain table needs resetting, and the hash
            // table can stay (its entries are range-checked in `get`). This
            // is what keeps compressing many short inputs cheap.
            let start_offset = self.base & HC_MASK;
            let end_offset = if self.end == 0 {
                0
            } else {
                ((self.end - 1) & HC_MASK) + 1
            };
            if start_offset < end_offset {
                self.chain_table[start_offset..end_offset].fill(0xFFFF);
            } else {
                self.chain_table[..end_offset].fill(0xFFFF);
                self.chain_table[start_offset..].fill(0xFFFF);
            }
        } else {
            self.hash_table.fill(-1);
            self.chain_table.fill(0xFFFF);
        }
        self.base = off;
        self.next = off;
        self.end = off + len;
    }

    // ARITH: `base + i` for `i < dict_len` is inside the caller's buffer, and
    // `next` advances to `base + dict_len`, which `reset` already proved is at
    // most `end`.
    #[allow(clippy::arithmetic_side_effects)]
    fn init_dictionary(&mut self, bytes: &[u8], dict_len: usize) {
        debug_assert_eq!(self.next, self.base);
        for i in 0..dict_len {
            self.add_hash(bytes, self.base + i);
        }
        self.next += dict_len;
    }

    // ARITH: `next < off` bounds the catch-up loop and `next += 1` lands at
    // `off`. `off + 1 - MAX_DISTANCE.min(off + 1)` cannot underflow because
    // the subtrahend is clamped to `off + 1`. `candidate -= chain_table[..]`
    // works in `i32` on values bounded by the input length, and the loop stops
    // as soon as `candidate < min >= 0`. `attempts` is compared against
    // `MAX_ATTEMPTS` before every increment.
    #[allow(clippy::arithmetic_side_effects)]
    fn get(&mut self, bytes: &[u8], off: usize) -> Option<usize> {
        while self.next < off {
            self.add_hash(bytes, self.next);
            self.next += 1;
        }

        let v = read4(bytes, off);
        let h = hash(v, HASH_LOG_HC) as usize;

        self.attempts = 0;
        let mut candidate = self.hash_table[h];
        if candidate >= off as i32 {
            // remainder from a previous call to compress()
            return None;
        }
        let min = self.base.max(off + 1 - MAX_DISTANCE.min(off + 1)) as i32;
        while candidate >= min && self.attempts < MAX_ATTEMPTS {
            let r = candidate as usize;
            if read4(bytes, r) == v {
                return Some(r);
            }
            candidate -= (self.chain_table[r & HC_MASK] as u32) as i32;
            self.attempts += 1;
        }
        None
    }

    // ARITH: the walk is in `i64` over offsets bounded by the input length,
    // and stops as soon as `candidate < base`. `attempts` is compared against
    // `MAX_ATTEMPTS` before every increment.
    #[allow(clippy::arithmetic_side_effects)]
    fn previous(&mut self, bytes: &[u8], off: usize) -> Option<usize> {
        let v = read4(bytes, off);
        let mut candidate = off as i64 - self.chain_table[off & HC_MASK] as i64;
        while candidate >= self.base as i64 && self.attempts < MAX_ATTEMPTS {
            let r = candidate as usize;
            if read4(bytes, r) == v {
                return Some(r);
            }
            candidate -= self.chain_table[r & HC_MASK] as i64;
            self.attempts += 1;
        }
        None
    }
}

/// Port of `LZ4.compress(bytes, off, len, out, ht)` -- no preset dictionary.
pub fn compress_into(bytes: &[u8], out: &mut Vec<u8>, ht: &mut impl HashTable) {
    compress_with_dictionary(bytes, 0, 0, bytes.len(), out, ht);
}

/// Convenience wrapper for one-shot callers with no dictionary and no table
/// to reuse: allocates a [`FastCompressionHashTable`] and returns the block.
/// Prefer [`compress_into`] with a caller-owned table when compressing many
/// buffers in a row -- that is exactly what Java's compressors do, and what
/// keeps the 16kB table out of the per-chunk allocation path.
pub fn compress(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut ht = FastCompressionHashTable::new();
    compress_into(bytes, &mut out, &mut ht);
    out
}

/// Port of `LZ4.compressWithDictionary`: compresses
/// `bytes[dict_off+dict_len .. dict_off+dict_len+len]`, allowing matches to
/// reach back into `bytes[dict_off..dict_off+dict_len]` (the preset
/// dictionary, which is *not* itself emitted). Produces a single
/// self-terminating LZ4 block, the same wire format [`decompress`] reads.
///
/// Panics (debug) if `dict_len > MAX_DISTANCE`, the same precondition Java
/// rejects with `IllegalArgumentException`; every caller in this crate
/// derives `dict_len` from `min(MAX_DISTANCE, ...)` the way
/// `LZ4WithPresetDictCompressionMode` does.
// ARITH: encode side, over a buffer the caller owns. `end = dict_off +
// dict_len + len` is the caller's own bound on `bytes` (both in-crate callers
// pass a window of a `Vec` they just built). The `len > LAST_LITERALS +
// MIN_MATCH` guard makes `end - LAST_LITERALS` and `limit - MIN_MATCH`
// non-negative; `off + 1 - MAX_DISTANCE.min(off + 1)` cannot underflow
// because the subtrahend is clamped; `off += match_len` and `off += 1` stay
// at or below `limit` because the loop re-tests `off <= limit`; and
// `end - anchor` is non-negative because `anchor` only ever takes the value
// of an `off` that was at or below `end`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compress_with_dictionary(
    bytes: &[u8],
    dict_off: usize,
    dict_len: usize,
    len: usize,
    out: &mut Vec<u8>,
    ht: &mut impl HashTable,
) {
    debug_assert!(dict_len <= MAX_DISTANCE);
    let end = dict_off + dict_len + len;
    let mut off = dict_off + dict_len;
    let mut anchor = off;

    if len > LAST_LITERALS + MIN_MATCH {
        let limit = end - LAST_LITERALS;
        let match_limit = limit - MIN_MATCH;
        ht.reset(bytes, dict_off, dict_len + len);
        ht.init_dictionary(bytes, dict_len);

        'main: while off <= limit {
            // find a match
            let mut match_ref;
            loop {
                if off >= match_limit {
                    break 'main;
                }
                match ht.get(bytes, off) {
                    Some(r) => {
                        debug_assert!(r >= dict_off && r < off);
                        match_ref = r;
                        break;
                    }
                    None => off += 1,
                }
            }

            // compute match length
            let mut match_len =
                MIN_MATCH + common_bytes(bytes, match_ref + MIN_MATCH, off + MIN_MATCH, limit);

            // try to find a better match (a no-op for FastCompressionHashTable,
            // whose `previous` always returns None)
            let min = dict_off.max(off + 1 - MAX_DISTANCE.min(off + 1));
            let mut r = ht.previous(bytes, match_ref);
            while let Some(candidate) = r {
                if candidate < min {
                    break;
                }
                let candidate_len =
                    MIN_MATCH + common_bytes(bytes, candidate + MIN_MATCH, off + MIN_MATCH, limit);
                if candidate_len > match_len {
                    match_ref = candidate;
                    match_len = candidate_len;
                }
                r = ht.previous(bytes, candidate);
            }

            encode_sequence(bytes, anchor, match_ref, off, match_len, out);
            off += match_len;
            anchor = off;
        }
    }

    // last literals
    encode_last_literals(bytes, anchor, end - anchor, out);
}

#[cfg(test)]
mod tests;
