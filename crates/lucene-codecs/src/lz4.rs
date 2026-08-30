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

const MIN_MATCH: usize = 4;
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
        // Bulk `copy_within` in runs of at most `match_dec` bytes rather than
        // Java's `System.arraycopy`-or-byte-loop split. One run is the
        // non-overlapping case (Java's `arraycopy` branch); when
        // `match_dec < match_len` the ranges overlap and each output byte can
        // depend on one just written (this is how LZ4 encodes short runs), so
        // the copy is cut at `match_dec` and repeated -- every run's source is
        // then wholly inside already-written bytes, which keeps the memmove
        // semantics correct while still moving whole words at a time instead
        // of one bounds-checked byte at a time.
        let mut written = 0usize;
        // ARITH: `written < match_len` is the loop condition, so
        // `match_len - written` cannot underflow and `n` is in
        // `1..=match_len - written`, which makes `written += n` terminate at
        // exactly `match_len`. `src_start + written + n <= src_start +
        // match_len = d_off - match_dec + match_len <= end <= dest.len()`, and
        // `d_off + written < end` likewise -- both were established by the
        // `checked_add`/`end > dest.len()` pair above. No per-byte check is
        // added here: this is the inner copy loop of the hottest decode path
        // in the crate.
        #[allow(clippy::arithmetic_side_effects)]
        while written < match_len {
            let n = match_dec.min(match_len - written);
            dest.copy_within(
                src_start + written..src_start + written + n,
                d_off + written,
            );
            written += n;
        }
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

fn encode_last_literals(bytes: &[u8], anchor: usize, literal_len: usize, out: &mut Vec<u8>) {
    let token = (literal_len.min(0x0F) as u8) << 4;
    encode_literals(bytes, token, anchor, literal_len, out);
}

// ARITH: encode side. `match_off >= anchor` and `match_off > match_ref` are
// both compressor invariants (`anchor` trails `off`, and `HashTable::get`
// only returns a candidate strictly below `off`); `match_len >= MIN_MATCH` by
// construction, and the `>= MIN_MATCH + 0x0F` guard covers the second
// subtraction.
#[allow(clippy::arithmetic_side_effects)]
fn encode_sequence(
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
        let bits_per_offset_log = if len.saturating_sub(LAST_LITERALS) < (1 << 16) {
            4
        } else {
            5
        };
        self.hash_log = MEMORY_USAGE + 3 - bits_per_offset_log;
        if self.table.len() < 1usize << self.hash_log {
            self.table = vec![0u32; 1usize << self.hash_log];
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
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::data_input::SliceInput;

    /// literal-only block: token=0x30 (literalLen=3, matchLen=0), then "abc".
    /// Since matchLen would be read next but dOff reaches destEnd right after
    /// the literals, no match bytes follow.
    /// An LZ4 length extension is a run of `0xFF` bytes, each adding 255 to
    /// the accumulator. The accumulator used to be a bare `usize +=`, so a
    /// long enough run wrapped it to a small length and decoded a silently
    /// truncated block (release) or panicked (debug). It has to come back as
    /// corruption instead, for both the literal and the match halves of the
    /// token.
    #[test]
    fn length_extension_run_is_bounded() {
        // A short, legal extension still decodes: literalLen 15 + 3 = 18.
        let mut compressed = vec![0xF0u8, 3];
        compressed.extend_from_slice(&[b'z'; 18]);
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 18];
        assert_eq!(decompress(&mut input, 18, &mut dest, 0).unwrap(), 18);

        // An extension that runs off the end of the block is an EOF, not a
        // wrapped length.
        let compressed = vec![0xFFu8; 4096];
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 64];
        assert!(decompress(&mut input, 64, &mut dest, 0).is_err());

        // The accumulator itself must be checked: `read_length_extension` is
        // the only place it grows, and it must saturate into an error rather
        // than wrap.
        let huge = vec![0xFFu8; 64];
        let mut input = SliceInput::new(&huge);
        assert!(read_length_extension(&mut input, usize::MAX - 1).is_err());
    }

    #[test]
    fn literal_only_block() {
        let compressed = [0x30u8, b'a', b'b', b'c'];
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 3];
        let end = decompress(&mut input, 3, &mut dest, 0).unwrap();
        assert_eq!(end, 3);
        assert_eq!(&dest, b"abc");
    }

    /// "aaaa" then a match copying the first 'a' 4 more times: literals="a"
    /// (len1), match_dec=1, match_len_field=0 -> matchLen=4 -> copies "aaaa".
    #[test]
    fn overlapping_match_copy() {
        // token: literalLen=1 (upper nibble), matchLen field=0 (lower nibble)
        let token = 1u8 << 4;
        let mut compressed = vec![token, b'a'];
        compressed.extend_from_slice(&1u16.to_le_bytes()); // matchDec = 1
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 5]; // 1 literal + 4 match bytes
        let end = decompress(&mut input, 5, &mut dest, 0).unwrap();
        assert_eq!(end, 5);
        assert_eq!(&dest, b"aaaaa");
    }

    #[test]
    fn extended_literal_length_encoding() {
        // literalLen = 0x0F + 0xFF + 5 = 15 + 255 + 5 = 275... too big for a
        // small test; use a smaller extension: 0x0F then one length byte 3
        // -> literalLen = 15 + 3 = 18.
        let token = 0xF0u8; // literalLen nibble = 0x0F (extended), matchLen nibble = 0
        let mut compressed = vec![token, 3u8]; // extension byte: +3 -> 18 literal bytes
        let literal: Vec<u8> = (0..18).map(|i| i as u8).collect();
        compressed.extend_from_slice(&literal);
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 18];
        let end = decompress(&mut input, 18, &mut dest, 0).unwrap();
        assert_eq!(end, 18);
        assert_eq!(dest.as_slice(), literal.as_slice());
    }

    #[test]
    fn preset_dictionary_reference_before_d_off() {
        // dest already has "hello" at [0..5] (the "dictionary"); decompress
        // a match-only block at d_off=5 that copies "hello" via a back-reference.
        let mut dest = *b"hello\0\0\0\0\0";
        // token: literalLen=0, matchLen field = 1 (since MIN_MATCH=4, +1=5 matches "hello"'s length)
        let token = 1u8;
        let mut compressed = vec![token];
        compressed.extend_from_slice(&5u16.to_le_bytes()); // matchDec=5, refers to dest[0]
        let mut input = SliceInput::new(&compressed);
        let end = decompress(&mut input, 5, &mut dest, 5).unwrap();
        assert_eq!(end, 10);
        assert_eq!(&dest, b"hellohello");
    }

    #[test]
    fn zero_match_offset_is_error() {
        let token = 1u8; // matchLen field=1, literalLen=0
        let mut compressed = vec![token];
        compressed.extend_from_slice(&0u16.to_le_bytes());
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 5];
        assert!(decompress(&mut input, 5, &mut dest, 0).is_err());
    }

    #[test]
    fn match_before_buffer_start_is_error() {
        let token = 1u8;
        let mut compressed = vec![token];
        compressed.extend_from_slice(&100u16.to_le_bytes()); // matchDec > dOff
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 5];
        assert!(decompress(&mut input, 5, &mut dest, 0).is_err());
    }

    #[test]
    fn extended_match_length_encoding() {
        // 1 literal byte 'z', then an extended-length match (matchLen field
        // = 0x0F + one extension byte 3 -> matchLen = 15+3+MIN_MATCH(4) = 22)
        // referencing match_dec=1 (run-length-encodes 'z' for the rest).
        let token = (1u8 << 4) | 0x0F;
        let mut compressed = vec![token, b'z'];
        compressed.extend_from_slice(&1u16.to_le_bytes()); // matchDec = 1
        compressed.push(3); // extension byte -> +3
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 23]; // 1 literal + 22 match bytes
        let end = decompress(&mut input, 23, &mut dest, 0).unwrap();
        assert_eq!(end, 23);
        assert_eq!(&dest, &[b'z'; 23]);
    }

    #[test]
    fn truncated_input_is_error() {
        let compressed = [0x30u8, b'a']; // claims 3 literal bytes, only 1 present
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 3];
        assert!(decompress(&mut input, 3, &mut dest, 0).is_err());
    }

    #[test]
    fn dest_too_small_for_decompressed_len_is_error() {
        let compressed = [0x30u8, b'a', b'b', b'c'];
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 2];
        assert!(decompress(&mut input, 3, &mut dest, 0).is_err());
    }

    fn assert_compress_round_trips(payload: &[u8]) {
        let compressed = compress(payload);
        let mut input = SliceInput::new(&compressed);
        let mut dest = vec![0u8; payload.len()];
        let end = decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
        assert_eq!(end, payload.len());
        assert_eq!(
            dest,
            payload,
            "round trip mismatch for len {}",
            payload.len()
        );
    }

    #[test]
    fn compress_empty_input_round_trips() {
        assert_compress_round_trips(&[]);
    }

    #[test]
    fn compress_single_byte_round_trips() {
        assert_compress_round_trips(b"x");
    }

    #[test]
    fn compress_short_input_below_match_threshold_round_trips() {
        // len <= LAST_LITERALS + MIN_MATCH, so the whole main-loop match
        // search is skipped entirely and this is pure last-literals.
        assert_compress_round_trips(b"abcdefghi");
    }

    #[test]
    fn compress_input_one_byte_above_match_threshold_round_trips() {
        // len == LAST_LITERALS + MIN_MATCH + 1 == 10: the smallest input for
        // which the main match-search loop actually runs at least once
        // (`match_limit`/`off <= limit` at the narrowest possible active
        // window) -- distinct from the len<=9 case above, which skips the
        // loop entirely.
        assert_compress_round_trips(b"abcdefghij");
    }

    #[test]
    fn compress_highly_repetitive_input_actually_finds_matches() {
        // A phrase repeated many times has abundant 4+ byte back-references
        // available; assert the compressor actually uses them (output much
        // smaller than input), not just that it round-trips.
        let payload = "the quick brown fox jumps over the lazy dog ".repeat(200);
        let payload = payload.as_bytes();
        let compressed = compress(payload);
        assert!(
            compressed.len() < payload.len() / 4,
            "expected real back-reference compression, got {} bytes from {} bytes of input",
            compressed.len(),
            payload.len()
        );
        assert_compress_round_trips(payload);
    }

    #[test]
    fn compress_incompressible_input_round_trips() {
        // Pseudo-random bytes (a simple xorshift-like sequence, no external
        // RNG dependency): no meaningful matches expected, exercising the
        // literal-heavy path of a real compressor rather than the stub's
        // always-one-literal-run shape.
        let mut state: u32 = 0x1234_5678;
        let payload: Vec<u8> = (0..2000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xFF) as u8
            })
            .collect();
        assert_compress_round_trips(&payload);
    }

    #[test]
    fn compress_match_at_very_start_round_trips() {
        // First 8 bytes are a repeated 4-byte pattern (an immediate match),
        // followed by non-repeating tail bytes.
        let mut payload = b"abcdabcd".to_vec();
        payload.extend((0..50u8).map(|i| i.wrapping_mul(7).wrapping_add(3)));
        assert_compress_round_trips(&payload);
    }

    #[test]
    fn compress_match_at_very_end_round_trips() {
        // Non-repeating head, then a repeated 4-byte pattern right at the
        // tail (within LAST_LITERALS of the end, exercising the boundary
        // where the main loop must stop and fall back to last-literals).
        let mut payload: Vec<u8> = (0..50u8)
            .map(|i| i.wrapping_mul(11).wrapping_add(5))
            .collect();
        payload.extend_from_slice(b"wxyzwxyz");
        assert_compress_round_trips(&payload);
    }

    #[test]
    fn compress_long_input_forces_extended_length_encoding() {
        // `encode_len`'s `while l >= 0xFF { push 0xFF; l -= 0xFF }` loop
        // must run at least twice (i.e. emit >=2 continuation bytes, not
        // just one) for BOTH the literal-length and match-length nibbles --
        // an off-by-one in the loop's second iteration wouldn't be caught by
        // a length needing only one continuation byte. `encode_literals`
        // calls `encode_len(literal_len - 0x0F, ...)`, so `literal_len` must
        // be >= 0x0F + 2*0xFF = 525 to force two iterations there; match
        // length is encoded as `match_len_total - MIN_MATCH - 0x0F`, so the
        // match run must be >= 4 + 0x0F + 2*0xFF = 529 bytes.
        //
        // Head: a long pseudo-random run (same xorshift generator as
        // `compress_incompressible_input_round_trips`) long enough that no
        // accidental 4-byte repeat gives the compressor an early match,
        // unlike a short-period `i % 251` sequence which repeats within
        // ~250 bytes and would (as an earlier version of this test did)
        // trigger a match well before the literal run reached the length
        // needed to actually exercise the chaining loop.
        let mut state: u32 = 0x9E37_79B9;
        let mut payload: Vec<u8> = (0..600)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xFF) as u8
            })
            .collect();
        // Tail: one long repeated-byte run, giving a single match of
        // total length >= 600 (well past the 529-byte threshold derived
        // above).
        payload.extend(std::iter::repeat_n(0x7Au8, 600));
        assert_compress_round_trips(&payload);

        // Confirm the chaining loop was actually exercised (not just that
        // the round trip happened to still work): re-derive the token
        // stream's first sequence and check its literal length is large
        // enough to have required >=2 continuation bytes.
        let compressed = compress(&payload);
        let token = compressed[0];
        let literal_len_nibble = (token >> 4) & 0x0F;
        assert_eq!(
            literal_len_nibble, 0x0F,
            "expected the first token's literal-length nibble to be maxed out (extended encoding)"
        );
        // Walk the 0xFF continuation bytes right after the token byte and
        // confirm there are at least 2 of them (proving the loop ran more
        // than once), then confirm the terminating byte plus the two 0xFF
        // bytes reconstruct a literal_len >= 525.
        let mut i = 1usize;
        let mut continuation_bytes = 0usize;
        let mut extra = 0usize;
        while compressed[i] == 0xFF {
            continuation_bytes += 1;
            extra += 0xFF;
            i += 1;
        }
        extra += compressed[i] as usize;
        assert!(
            continuation_bytes >= 2,
            "expected >=2 0xFF continuation bytes in the literal-length encoding, got {continuation_bytes}"
        );
        assert!(0x0F + extra >= 525);
    }

    #[test]
    fn overlapping_match_with_a_multi_byte_period_repeats_the_whole_run() {
        // matchDec=3 with matchLen=8: the copy has to be cut into 3-byte
        // runs (3+3+2), each reading bytes the previous run just wrote.
        // A single memmove of 8 bytes would produce "abcabcab" only by luck
        // of direction; a wrong split produces garbage. Literals "abc",
        // then matchLen field = 8-4 = 4.
        let token = (3u8 << 4) | 4;
        let mut compressed = vec![token, b'a', b'b', b'c'];
        compressed.extend_from_slice(&3u16.to_le_bytes());
        let mut input = SliceInput::new(&compressed);
        let mut dest = [0u8; 11];
        let end = decompress(&mut input, 11, &mut dest, 0).unwrap();
        assert_eq!(end, 11);
        assert_eq!(&dest, b"abcabcabcab");
    }

    #[test]
    fn common_bytes_counts_across_the_word_loop_boundary() {
        // The 8-byte word loop plus byte tail must agree with a naive count
        // for prefixes on either side of, and exactly at, a word boundary.
        let mut b = vec![0u8; 64];
        for shared in [0usize, 1, 7, 8, 9, 16, 23] {
            for (i, slot) in b.iter_mut().enumerate() {
                *slot = (i % 251) as u8;
            }
            // b[0..] and b[32..] agree for `shared` bytes then differ.
            b.copy_within(0..shared, 32);
            b[32 + shared] = b[shared] ^ 0xFF;
            assert_eq!(common_bytes(&b, 0, 32, 64), shared, "shared={shared}");
        }
    }

    #[test]
    fn compress_with_dictionary_round_trips_against_the_preset_dictionary() {
        // The dictionary is bytes the decoder already has sitting before
        // `d_off`; the compressed block must be able to back-reference into
        // it, and must NOT re-emit it.
        let dict = "the quick brown fox jumps over the lazy dog ".repeat(4);
        let block = format!("{dict}and then the quick brown fox went home");
        let mut buffer = dict.as_bytes().to_vec();
        buffer.extend_from_slice(block.as_bytes());

        let mut out = Vec::new();
        let mut ht = FastCompressionHashTable::new();
        compress_with_dictionary(&buffer, 0, dict.len(), block.len(), &mut out, &mut ht);

        // The dictionary makes this compress far better than the same block
        // compressed standalone would.
        let standalone = compress(block.as_bytes());
        assert!(
            out.len() < standalone.len(),
            "preset dictionary should help: {} vs {}",
            out.len(),
            standalone.len()
        );

        let mut dest = vec![0u8; dict.len() + block.len()];
        dest[..dict.len()].copy_from_slice(dict.as_bytes());
        let mut input = SliceInput::new(&out);
        let end = decompress(&mut input, block.len(), &mut dest, dict.len()).unwrap();
        assert_eq!(end, dict.len() + block.len());
        assert_eq!(&dest[dict.len()..], block.as_bytes());
    }

    #[test]
    fn a_reused_fast_hash_table_stays_correct_across_many_inputs() {
        // Java never clears the table between `reset` calls -- stale entries
        // are rejected by `get`'s range + byte-for-byte re-check instead.
        // Compressing a series of *different* buffers through one table has
        // to keep round-tripping, or that re-check is wrong.
        let mut ht = FastCompressionHashTable::new();
        for i in 0..40u32 {
            let payload: Vec<u8> = (0..500u32)
                .map(|j| (j.wrapping_mul(i.wrapping_add(1)) % 97) as u8)
                .collect();
            let mut out = Vec::new();
            compress_into(&payload, &mut out, &mut ht);
            let mut dest = vec![0u8; payload.len()];
            let mut input = SliceInput::new(&out);
            decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
            assert_eq!(dest, payload, "round {i}");
        }
    }

    #[test]
    fn fast_hash_table_switches_hash_log_above_64kb_and_still_round_trips() {
        // `reset` picks hashLog 13 below 64kB and 12 at or above it (Java's
        // `MEMORY_USAGE + 3 - bitsPerOffsetLog`); exercise both, in that
        // order, through one reused table so the smaller-hashLog run also
        // reads a table sized for the larger one.
        let mut ht = FastCompressionHashTable::new();
        for len in [1_000usize, 200_000] {
            let payload: Vec<u8> = (0..len).map(|i| (i % 211) as u8).collect();
            let mut out = Vec::new();
            compress_into(&payload, &mut out, &mut ht);
            let mut dest = vec![0u8; len];
            let mut input = SliceInput::new(&out);
            decompress(&mut input, len, &mut dest, 0).unwrap();
            assert_eq!(dest, payload, "len {len}");
        }
    }

    #[test]
    fn high_compression_hash_table_beats_the_fast_one_and_round_trips() {
        // A payload whose 4-byte hooks recur many times with only the *last*
        // occurrence extendable into a long match: the fast table keeps only
        // that last occurrence per hash, the HC table walks the chain and
        // finds the longer one.
        let mut payload = Vec::new();
        for i in 0..200u32 {
            payload.extend_from_slice(b"HOOK");
            payload.extend_from_slice(&i.to_le_bytes());
            payload.extend_from_slice(b"filler-that-does-not-repeat");
            payload.extend_from_slice(&(i * 7919).to_le_bytes());
        }
        payload.extend_from_slice(b"HOOKfiller-that-does-not-repeat");

        let fast = compress(&payload);
        let mut high = Vec::new();
        let mut hc = HighCompressionHashTable::new();
        compress_into(&payload, &mut high, &mut hc);

        assert!(
            high.len() <= fast.len(),
            "HighCompressionHashTable should never lose to the fast one: {} vs {}",
            high.len(),
            fast.len()
        );

        let mut dest = vec![0u8; payload.len()];
        let mut input = SliceInput::new(&high);
        decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
        assert_eq!(dest, payload);
    }

    #[test]
    fn high_compression_hash_table_is_reusable_across_short_and_long_inputs() {
        // `reset`'s two branches: the cheap partial chain-table wipe after a
        // sub-64kB input, and the full wipe after a larger one. Alternate
        // between them through one table, including a wrap-around case where
        // the touched chain-table window straddles the mask boundary.
        let mut hc = HighCompressionHashTable::new();
        for len in [300usize, 100_000, 300, 70_000, 50] {
            let payload: Vec<u8> = (0..len).map(|i| (i % 37) as u8).collect();
            let mut out = Vec::new();
            compress_into(&payload, &mut out, &mut hc);
            let mut dest = vec![0u8; len];
            let mut input = SliceInput::new(&out);
            decompress(&mut input, len, &mut dest, 0).unwrap();
            assert_eq!(dest, payload, "len {len}");
        }
    }

    #[test]
    fn high_compression_with_dictionary_round_trips() {
        let dict: Vec<u8> = (0..2_000u32).map(|i| (i % 61) as u8).collect();
        let block: Vec<u8> = (0..3_000u32).map(|i| ((i + 13) % 61) as u8).collect();
        let mut buffer = dict.clone();
        buffer.extend_from_slice(&block);

        let mut out = Vec::new();
        let mut hc = HighCompressionHashTable::new();
        compress_with_dictionary(&buffer, 0, dict.len(), block.len(), &mut out, &mut hc);

        let mut dest = vec![0u8; dict.len() + block.len()];
        dest[..dict.len()].copy_from_slice(&dict);
        let mut input = SliceInput::new(&out);
        decompress(&mut input, block.len(), &mut dest, dict.len()).unwrap();
        assert_eq!(&dest[dict.len()..], &block[..]);
    }

    #[test]
    fn zero_length_decompress_still_consumes_one_token_byte() {
        // Even an empty unit is encoded as a single zero token, matching
        // Java's do-while loop shape (it always reads at least one byte).
        let compressed = [0x00u8];
        let mut input = SliceInput::new(&compressed);
        let mut dest: [u8; 0] = [];
        let end = decompress(&mut input, 0, &mut dest, 0).unwrap();
        assert_eq!(end, 0);
    }
}
