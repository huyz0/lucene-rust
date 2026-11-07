//! A raw DEFLATE (RFC 1951) decoder for whole in-memory units, built the way
//! libdeflate's is: the compressed bytes and the whole output range are known
//! up front, so the decoder needs no sliding window or resumable state.
//!
//! - A 64-bit bit buffer refilled eight bytes at a time, without a branch per
//!   byte.
//! - One table lookup per literal/length symbol and per distance symbol (11
//!   and 8 index bits, with second-level tables for longer codes). Each
//!   entry also carries the symbol's base value and extra-bit count, so a
//!   length or distance is decoded with one more shift and mask.
//! - A fast loop that runs while the input and output have room for a whole
//!   sequence, decodes up to three literals per refill, and copies matches
//!   eight bytes at a time. A careful loop finishes the last bytes.
//!
//! Matches may refer back past the start of the output into whatever is in
//! `dest` before it, which is how a preset dictionary works: the caller puts
//! the dictionary bytes there first.
//!
//! The decoder never reads or writes outside its slices: a corrupt stream is
//! an [`InflateError`], never a panic. The fast loop's stores and match
//! copies skip bounds checks the loop condition has already made.

/// Why a DEFLATE stream could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("deflate: {0}")]
pub struct InflateError(pub &'static str);

type Result<T> = std::result::Result<T, InflateError>;

/// Index bits of the literal/length main table.
const LITLEN_TABLE_BITS: u32 = 11;
/// Index bits of the distance main table.
const DIST_TABLE_BITS: u32 = 8;
/// Index bits of the code-length (precode) table; precode lengths are at
/// most 7 bits, so it has no second level.
const PRECODE_TABLE_BITS: u32 = 7;

const MAX_CODE_LEN: usize = 15;
const NUM_LITLEN: usize = 288;
const NUM_DIST: usize = 32;
const NUM_PRECODE: usize = 19;

/// Entry layout: bits 0..8 the bits the symbol takes in all (its codeword
/// plus its extra bits, or for a pointer the main table's index bits), 8..12
/// the codeword's own length (for a pointer, the second-level table's index
/// bits), 12..16 the kind, 16..32 the base value (a literal byte, a length or
/// distance base, or a second-level table's start). So a symbol is consumed
/// with one shift, and its extra bits are the bits between the two lengths.
const KIND_LITERAL: u32 = 1 << 12;
const KIND_END: u32 = 2 << 12;
const KIND_POINTER: u32 = 4 << 12;
const KIND_INVALID: u32 = 8 << 12;
/// A length or distance: no kind bit.
const KIND_MASK: u32 = 0xF << 12;
const INVALID: u32 = KIND_INVALID;

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order the precode lengths are stored in.
const PRECODE_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// The fast loop runs while this many output bytes are free: the longest
/// match (258) plus the 16-byte overshoot of its last word copy, plus the
/// three literals decoded before it.
const FAST_OUT_MARGIN: usize = 258 + 16 + 3;
/// ... and while this many input bytes are left: one eight-byte refill.
const FAST_IN_MARGIN: usize = 8;

#[inline]
fn corrupt(what: &'static str) -> InflateError {
    InflateError(what)
}

/// A decode table: a main level indexed by the low bits of the bit buffer,
/// then the second-level tables the pointers in it name.
struct Table<const MAIN: usize> {
    entries: Vec<u32>,
}

impl<const MAIN: usize> Table<MAIN> {
    fn new() -> Self {
        Self {
            entries: vec![INVALID; MAIN],
        }
    }

    /// Builds the table for the code with these `lengths` (0 = unused),
    /// with `value(symbol)` giving each symbol's kind and base, and its
    /// extra-bit count in the low byte. Rejects over-subscribed codes, and incomplete ones other than
    /// the two RFC 1951 allows in practice (no codes at all, or one code
    /// of one bit), which zlib accepts too.
    // ARITH: code lengths are at most 15, so every shift is below 16 bits
    // and every count fits; `next_code` is built from the counts and only
    // grows within a 15-bit space a complete code fills exactly; table
    // indices are below the sizes computed for them.
    #[allow(clippy::arithmetic_side_effects)]
    fn build(
        &mut self,
        lengths: &[u8],
        table_bits: u32,
        allow_empty: bool,
        value: impl Fn(usize) -> u32,
    ) -> Result<()> {
        debug_assert_eq!(MAIN, 1 << table_bits);
        let mut count = [0u16; MAX_CODE_LEN + 1];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let used: u32 = count.iter().map(|&c| c as u32).sum();
        let mut left: i32 = 1;
        let mut max_len = 0;
        for (len, &c) in count.iter().enumerate().skip(1) {
            left = (left << 1) - c as i32;
            if left < 0 {
                return Err(corrupt("over-subscribed Huffman code"));
            }
            if c > 0 {
                max_len = len;
            }
        }
        if used == 0 || left > 0 {
            self.entries.clear();
            self.entries.resize(MAIN, INVALID);
            if used == 0 {
                return if allow_empty {
                    Ok(())
                } else {
                    Err(corrupt("empty Huffman code"))
                };
            }
            if !(used == 1 && max_len == 1) {
                return Err(corrupt("incomplete Huffman code"));
            }
            // One one-bit code: code 0, every even index; odd ones invalid.
            if let Some(sym) = lengths.iter().position(|&l| l != 0) {
                let entry = with_len(value(sym), 1);
                for e in self.entries.iter_mut().step_by(2) {
                    *e = entry;
                }
            }
            return Ok(());
        }

        // Symbols sorted by code length (stable, so by symbol within a
        // length): the order canonical codes are assigned in.
        let mut offset = [0u16; MAX_CODE_LEN + 2];
        for len in 1..=MAX_CODE_LEN {
            offset[len + 1] = offset[len] + count[len];
        }
        let mut sorted = [0u16; NUM_LITLEN];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                sorted[offset[l as usize] as usize] = sym as u16;
                offset[l as usize] += 1;
            }
        }

        // The main table, a complete code filling it exactly: the entries of
        // each length's codes are written into the first `2^len` slots, then
        // that prefix is doubled, which replicates every shorter code into
        // all the slots its low bits select. Slots left over are the prefixes
        // of longer codes, which get pointers below.
        let tb = table_bits as usize;
        self.entries.truncate(MAIN);
        self.entries.resize(MAIN, INVALID);
        let mut code = 0u32;
        let mut k = 0usize;
        for (len, &n) in count.iter().enumerate().take(tb + 1).skip(1) {
            if len > 1 {
                let half = 1usize << (len - 1);
                self.entries.copy_within(0..half, half);
            }
            for _ in 0..n {
                let sym = sorted[k] as usize;
                let rev = (code.reverse_bits() >> (32 - len)) as usize;
                self.entries[rev] = with_len(value(sym), len as u32);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        if max_len <= tb {
            return Ok(());
        }

        // Longer codes, still in canonical order: the codes sharing their
        // top `tb` bits are consecutive, and the last of them is the longest,
        // which sizes their second-level table.
        let mut long = [(0u16, 0u8, 0u32); NUM_LITLEN];
        let mut n = 0;
        for (len, &c) in count.iter().enumerate().take(max_len + 1).skip(tb + 1) {
            for _ in 0..c {
                long[n] = (sorted[k], len as u8, code);
                code += 1;
                k += 1;
                n += 1;
            }
            code <<= 1;
        }
        let mut i = 0;
        while i < n {
            let top = |(_, l, c): (u16, u8, u32)| c >> (l as usize - tb);
            let group = top(long[i]);
            let mut j = i + 1;
            while j < n && top(long[j]) == group {
                j += 1;
            }
            let bits = long[j - 1].1 as usize - tb;
            let start = self.entries.len();
            self.entries.resize(start + (1 << bits), INVALID);
            let prefix = (group.reverse_bits() >> (32 - tb)) as usize;
            self.entries[prefix] =
                KIND_POINTER | ((start as u32) << 16) | ((bits as u32) << 8) | tb as u32;
            for &(sym, l, c) in &long[i..j] {
                let len = l as usize;
                let rev = (c.reverse_bits() >> (32 - len)) as usize;
                let entry = with_len(value(sym as usize), (len - tb) as u32);
                let mut slot = rev >> tb;
                while slot < 1 << bits {
                    self.entries[start + slot] = entry;
                    slot += 1 << (len - tb);
                }
            }
            i = j;
        }
        Ok(())
    }
}

/// `value` (kind, base, extra-bit count) with a codeword of `len` bits.
// ARITH: `len <= 15` and the extra count is at most 13.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn with_len(value: u32, len: u32) -> u32 {
    let extra = value & 0xFF;
    (value & !0x0FFF) | (len << 8) | (len + extra)
}

fn litlen_value(sym: usize) -> u32 {
    match sym {
        0..=255 => KIND_LITERAL | ((sym as u32) << 16),
        256 => KIND_END,
        257..=285 => {
            // ARITH: `sym - 257` is in `0..29`.
            #[allow(clippy::arithmetic_side_effects)]
            let i = sym - 257;
            ((LEN_BASE[i] as u32) << 16) | LEN_EXTRA[i] as u32
        }
        _ => INVALID,
    }
}

fn dist_value(sym: usize) -> u32 {
    match (DIST_BASE.get(sym), DIST_EXTRA.get(sym)) {
        (Some(&base), Some(&extra)) => ((base as u32) << 16) | extra as u32,
        _ => INVALID,
    }
}

fn precode_value(sym: usize) -> u32 {
    (sym as u32) << 16
}

/// The bit reader. Bits past `valid` in `buf` are the following input bytes
/// already loaded (or zeros past the end); every refill ORs the same bytes
/// into the same places, so they stay correct.
#[derive(Clone, Copy)]
struct Bits<'a> {
    input: &'a [u8],
    /// The next input byte not yet in `buf`; past `input.len()` once the
    /// end has been padded with zeros.
    pos: usize,
    buf: u64,
    valid: u32,
}

impl<'a> Bits<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            buf: 0,
            valid: 0,
        }
    }

    /// Fills `buf` to at least 56 valid bits.
    // ARITH: `valid < 64` always, so the shift is legal and `valid / 8 <= 7`;
    // `pos + 8` was checked against the input length.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn refill(&mut self) {
        if let Some(w) = self.input.get(self.pos..self.pos + 8) {
            let w = u64::from_le_bytes(w.try_into().unwrap());
            self.buf |= w << self.valid;
            // Whole bytes fit in the free bits: `7 - valid / 8` of them.
            self.pos += 7 - (self.valid >> 3) as usize;
            self.valid |= 56;
        } else {
            self.refill_slow();
        }
    }

    // ARITH: as `refill`; `pos` only grows by one per byte, past the end by
    // at most eight bytes per refill and the decoder stops long before it
    // could approach `usize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    #[cold]
    fn refill_slow(&mut self) {
        while self.valid <= 56 {
            let b = self.input.get(self.pos).copied().unwrap_or(0);
            self.buf |= (b as u64) << self.valid;
            self.pos += 1;
            self.valid += 8;
        }
    }

    #[inline(always)]
    fn peek(&self) -> u64 {
        self.buf
    }

    /// Drops `n <= valid` bits.
    // ARITH: callers consume at most what a refill guaranteed.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(always)]
    fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.valid);
        self.buf >>= n;
        self.valid -= n;
    }

    /// Reads `n <= 16` bits, refilling first if needed.
    #[inline]
    fn read(&mut self, n: u32) -> u32 {
        if self.valid < n {
            self.refill();
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.consume(n);
        v
    }

    /// Whether more bits were consumed than the input holds (the decoder
    /// ran into the zero padding).
    // ARITH: `pos * 8` cannot overflow: `pos` is at most the input length
    // plus a few padding bytes.
    #[allow(clippy::arithmetic_side_effects)]
    fn overran(&self) -> bool {
        (self.pos as u64) * 8 - self.valid as u64 > (self.input.len() as u64) * 8
    }

    /// Drops the bits up to the next byte boundary and returns the byte
    /// offset of the next unread byte, emptying the buffer.
    // ARITH: `valid / 8` whole bytes were loaded past the boundary, so
    // `pos - valid / 8` is the boundary's byte offset.
    #[allow(clippy::arithmetic_side_effects)]
    fn align_to_byte(&mut self) -> usize {
        let drop = self.valid & 7;
        self.consume(drop);
        let at = self.pos - (self.valid >> 3) as usize;
        self.buf = 0;
        self.valid = 0;
        self.pos = at;
        at
    }
}

/// Decoder tables, reusable across calls.
pub struct Inflater {
    litlen: Table<{ 1 << LITLEN_TABLE_BITS }>,
    dist: Table<{ 1 << DIST_TABLE_BITS }>,
    precode: Table<{ 1 << PRECODE_TABLE_BITS }>,
    /// Whether `litlen`/`dist` hold the fixed (BTYPE 01) codes, so a run of
    /// fixed blocks builds them once.
    fixed: bool,
    lengths: [u8; NUM_LITLEN + NUM_DIST],
}

impl Default for Inflater {
    fn default() -> Self {
        Self {
            litlen: Table::new(),
            dist: Table::new(),
            precode: Table::new(),
            fixed: false,
            lengths: [0; NUM_LITLEN + NUM_DIST],
        }
    }
}

impl Inflater {
    /// Decodes all of `input` (one raw DEFLATE stream, ending with a final
    /// block) into `dest[d_off..d_off + len]`. Back-references may reach
    /// any byte of `dest` before `d_off`. Fails unless the stream produces
    /// exactly `len` bytes and fits in `input`.
    pub fn inflate(
        &mut self,
        input: &[u8],
        dest: &mut [u8],
        d_off: usize,
        len: usize,
    ) -> Result<()> {
        let end = d_off
            .checked_add(len)
            .filter(|&e| e <= dest.len())
            .ok_or_else(|| corrupt("output range outside the buffer"))?;
        // Only the output range is writable; the bytes before it are history.
        let dest = &mut dest[..end];
        let mut bits = Bits::new(input);
        let mut out = d_off;
        loop {
            let header = bits.read(3);
            let last = header & 1 != 0;
            match header >> 1 {
                0 => out = stored_block(&mut bits, dest, out)?,
                1 => {
                    if !self.fixed {
                        self.build_fixed()?;
                    }
                    out = self.huffman_block(&mut bits, dest, out)?;
                }
                2 => {
                    self.fixed = false;
                    self.read_dynamic_tables(&mut bits)?;
                    out = self.huffman_block(&mut bits, dest, out)?;
                }
                _ => return Err(corrupt("reserved block type")),
            }
            if bits.overran() {
                return Err(corrupt("stream runs past its compressed length"));
            }
            if last {
                break;
            }
        }
        if out != end {
            return Err(corrupt("stream ends before the expected length"));
        }
        Ok(())
    }

    fn build_fixed(&mut self) -> Result<()> {
        let mut l = [0u8; NUM_LITLEN];
        l[..144].fill(8);
        l[144..256].fill(9);
        l[256..280].fill(7);
        l[280..].fill(8);
        self.litlen
            .build(&l, LITLEN_TABLE_BITS, false, litlen_value)?;
        self.dist
            .build(&[5u8; NUM_DIST], DIST_TABLE_BITS, false, dist_value)?;
        self.fixed = true;
        Ok(())
    }

    // ARITH: `hlit <= 286 + 2` and `hdist <= 32`, so every index into
    // `lengths` (at most `NUM_LITLEN + NUM_DIST`) is checked against
    // `total`; repeat counts are at most 138 and checked before use.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_dynamic_tables(&mut self, bits: &mut Bits<'_>) -> Result<()> {
        let hlit = bits.read(5) as usize + 257;
        let hdist = bits.read(5) as usize + 1;
        let hclen = bits.read(4) as usize + 4;
        if hlit > 286 || hdist > 30 {
            return Err(corrupt("too many length or distance codes"));
        }
        let mut pre = [0u8; NUM_PRECODE];
        for &i in PRECODE_ORDER.iter().take(hclen) {
            pre[i] = bits.read(3) as u8;
        }
        self.precode
            .build(&pre, PRECODE_TABLE_BITS, false, precode_value)?;
        let total = hlit + hdist;
        let mut i = 0;
        while i < total {
            if bits.valid < 7 + 7 {
                bits.refill();
            }
            let entry = self.precode.entries[(bits.peek() & 0x7F) as usize];
            if entry & KIND_INVALID != 0 {
                return Err(corrupt("bad code length code"));
            }
            bits.consume(entry & 0xFF);
            let sym = (entry >> 16) as u8;
            let (value, repeat) = match sym {
                0..=15 => (sym, 1),
                16 => {
                    let prev = match i.checked_sub(1) {
                        Some(p) => self.lengths[p],
                        None => return Err(corrupt("repeat with no previous length")),
                    };
                    (prev, 3 + bits.read(2) as usize)
                }
                17 => (0, 3 + bits.read(3) as usize),
                _ => (0, 11 + bits.read(7) as usize),
            };
            if i + repeat > total {
                return Err(corrupt("code lengths overflow"));
            }
            self.lengths[i..i + repeat].fill(value);
            i += repeat;
        }
        if self.lengths[256] == 0 {
            return Err(corrupt("no end-of-block code"));
        }
        let (lit, dist) = self.lengths[..total].split_at(hlit);
        self.litlen
            .build(lit, LITLEN_TABLE_BITS, false, litlen_value)?;
        self.dist.build(dist, DIST_TABLE_BITS, true, dist_value)?;
        Ok(())
    }

    /// Decodes one Huffman-coded block into `dest[out..]`; returns the new
    /// output position.
    // ARITH: in the fast loop `out + FAST_OUT_MARGIN <= dest.len()` and a
    // refill left at least 56 bits, which covers one literal/length code with
    // its extra bits (15 + 5) and one distance code with its extra bits
    // (15 + 13). Lengths are at most 258, distances are checked against
    // `out` before the copy, and every index below is under `dest.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline(never)]
    fn huffman_block(&self, bits: &mut Bits<'_>, dest: &mut [u8], mut out: usize) -> Result<usize> {
        let litlen = &self.litlen.entries;
        let dist = &self.dist.entries;
        let lit_main: &[u32; 1 << LITLEN_TABLE_BITS] =
            litlen[..1 << LITLEN_TABLE_BITS].try_into().unwrap();
        let dist_main: &[u32; 1 << DIST_TABLE_BITS] =
            dist[..1 << DIST_TABLE_BITS].try_into().unwrap();
        const LIT_MASK: u64 = (1 << LITLEN_TABLE_BITS) - 1;
        const DIST_MASK: u64 = (1 << DIST_TABLE_BITS) - 1;
        let end = dest.len();

        // The fast loop. At its top at least 56 bits are buffered and `entry`
        // is the table entry for them; every path ends by refilling and
        // looking the next entry up, so a lookup overlaps the previous
        // symbol's output.
        let mut b = *bits;
        b.refill();
        let mut entry = lit_main[(b.peek() & LIT_MASK) as usize];
        while out + FAST_OUT_MARGIN <= end && b.pos + FAST_IN_MARGIN <= b.input.len() {
            if entry & KIND_LITERAL != 0 {
                // Up to three literals per refill: 3 * 15 bits <= 56.
                b.consume(entry & 0xFF);
                let lit1 = (entry >> 16) as u8;
                entry = lit_main[(b.peek() & LIT_MASK) as usize];
                if entry & KIND_LITERAL != 0 {
                    b.consume(entry & 0xFF);
                    let lit2 = (entry >> 16) as u8;
                    entry = lit_main[(b.peek() & LIT_MASK) as usize];
                    if entry & KIND_LITERAL != 0 {
                        b.consume(entry & 0xFF);
                        let lit3 = (entry >> 16) as u8;
                        b.refill();
                        entry = lit_main[(b.peek() & LIT_MASK) as usize];
                        store(dest, out, lit1);
                        store(dest, out + 1, lit2);
                        store(dest, out + 2, lit3);
                        out += 3;
                        continue;
                    }
                    b.refill();
                    store(dest, out, lit1);
                    store(dest, out + 1, lit2);
                    out += 2;
                    continue;
                }
                b.refill();
                store(dest, out, lit1);
                out += 1;
                continue;
            }
            entry = second_level(litlen, entry, &mut b);
            if entry & KIND_MASK != 0 {
                match entry & KIND_MASK {
                    KIND_LITERAL => {
                        b.consume(entry & 0xFF);
                        dest[out] = (entry >> 16) as u8;
                        out += 1;
                        b.refill();
                        entry = lit_main[(b.peek() & LIT_MASK) as usize];
                        continue;
                    }
                    KIND_END => {
                        b.consume(entry & 0xFF);
                        *bits = b;
                        return Ok(out);
                    }
                    _ => return Err(corrupt("invalid literal/length code")),
                }
            }
            let length = take_value(entry, &mut b);
            let d = second_level(dist, dist_main[(b.peek() & DIST_MASK) as usize], &mut b);
            if d & KIND_MASK != 0 {
                return Err(corrupt("invalid distance code"));
            }
            let distance = take_value(d, &mut b);
            b.refill();
            entry = lit_main[(b.peek() & LIT_MASK) as usize];
            if distance > out {
                return Err(corrupt("distance before the start of the buffer"));
            }
            copy_match_fast(dest, out, distance, length);
            out += length;
        }
        *bits = b;

        // The careful loop.
        loop {
            bits.refill();
            let entry = second_level(litlen, lit_main[(bits.peek() & LIT_MASK) as usize], bits);
            match entry & KIND_MASK {
                KIND_LITERAL => {
                    bits.consume(entry & 0xFF);
                    *dest
                        .get_mut(out)
                        .ok_or_else(|| corrupt("stream produces too many bytes"))? =
                        (entry >> 16) as u8;
                    out += 1;
                    if bits.overran() {
                        return Err(corrupt("stream runs past its compressed length"));
                    }
                    continue;
                }
                KIND_END => {
                    bits.consume(entry & 0xFF);
                    return Ok(out);
                }
                0 => {}
                _ => return Err(corrupt("invalid literal/length code")),
            }
            let length = take_value(entry, bits);
            let d = second_level(dist, dist_main[(bits.peek() & DIST_MASK) as usize], bits);
            if d & KIND_MASK != 0 {
                return Err(corrupt("invalid distance code"));
            }
            let distance = take_value(d, bits);
            if bits.overran() {
                return Err(corrupt("stream runs past its compressed length"));
            }
            if distance > out {
                return Err(corrupt("distance before the start of the buffer"));
            }
            if length > end - out {
                return Err(corrupt("stream produces too many bytes"));
            }
            for i in out..out + length {
                dest[i] = dest[i - distance];
            }
            out += length;
        }
    }
}

/// Follows a main-table pointer `entry` into its second-level table,
/// consuming the main table's bits; any other entry is returned as it is.
// ARITH: a pointer's start plus an index below its table's size is inside
// `table`; a corrupt one finds no entry and decodes as invalid.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn second_level(table: &[u32], entry: u32, b: &mut Bits<'_>) -> u32 {
    if entry & KIND_POINTER == 0 {
        return entry;
    }
    b.consume(entry & 0xFF);
    let sub_bits = (entry >> 8) & 0xF;
    let idx = (entry >> 16) as usize + (b.peek() & ((1 << sub_bits) - 1)) as usize;
    table.get(idx).copied().unwrap_or(INVALID)
}

/// Consumes a length or distance symbol with its extra bits and returns its
/// value: the base plus the extra bits that follow the codeword.
// ARITH: the total is at most 15 + 13 bits, below the buffer's 64; the value
// is at most 258 or 32768.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn take_value(entry: u32, b: &mut Bits<'_>) -> usize {
    let saved = b.peek();
    let total = entry & 0xFF;
    b.consume(total);
    let codeword = (entry >> 8) & 0xF;
    (entry >> 16) as usize + ((saved & ((1u64 << total) - 1)) >> codeword) as usize
}

/// `dest[at] = byte` for an `at` the fast loop has proved in range.
#[inline(always)]
fn store(dest: &mut [u8], at: usize, byte: u8) {
    debug_assert!(at < dest.len());
    // SAFETY: the fast loop runs only while `out + FAST_OUT_MARGIN <=
    // dest.len()`, and stores at most three literals past `out`.
    unsafe { *dest.get_unchecked_mut(at) = byte }
}

/// Copies `N` bytes from `dest[src..]` to `dest[dst..]`, two ranges that do
/// not overlap and are both inside `dest`.
#[inline(always)]
fn copy_word<const N: usize>(dest: &mut [u8], src: usize, dst: usize) {
    debug_assert!(src + N <= dst && dst + N <= dest.len());
    // SAFETY: `copy_match_fast`'s callers keep `src + N <= dst` (the match
    // distance, or the replication step, is at least `N`) and
    // `dst + N <= dest.len()` (the fast loop's margin).
    unsafe {
        let p = dest.as_mut_ptr();
        std::ptr::copy_nonoverlapping(p.add(src), p.add(dst), N);
    }
}

/// Copies a `length`-byte match from `distance` bytes back to `dest[out..]`,
/// writing up to 16 bytes past its end (the fast loop's margin covers them).
// ARITH: `1 <= distance <= out` and `out + length + 16 <= dest.len()` (the
// fast loop's margin), so every index is in range.
#[allow(clippy::arithmetic_side_effects)]
#[inline(always)]
fn copy_match_fast(dest: &mut [u8], out: usize, distance: usize, length: usize) {
    let stop = out + length;
    let mut dst = out;
    if distance >= 16 {
        let mut src = out - distance;
        while dst < stop {
            copy_word::<16>(dest, src, dst);
            src += 16;
            dst += 16;
        }
    } else if distance >= 8 {
        let mut src = out - distance;
        while dst < stop {
            copy_word::<8>(dest, src, dst);
            src += 8;
            dst += 8;
        }
    } else if distance == 1 {
        let b = dest[out - 1];
        dest[out..stop].fill(b);
    } else {
        // A short period: write the first `step` bytes one at a time, where
        // `step` is the smallest multiple of the period that is at least 8;
        // the rest repeats with period `step`, so eight-byte words copied
        // from `step` back never overlap what they write.
        let step = distance * 8usize.div_ceil(distance);
        let first = stop.min(out + step);
        while dst < first {
            dest[dst] = dest[dst - distance];
            dst += 1;
        }
        while dst < stop {
            copy_word::<8>(dest, dst - step, dst);
            dst += 8;
        }
    }
}

/// Copies a stored (BTYPE 00) block's bytes to `dest[out..]`.
// ARITH: `at + 4 + len` is checked against the input before any use.
#[allow(clippy::arithmetic_side_effects)]
fn stored_block(bits: &mut Bits<'_>, dest: &mut [u8], out: usize) -> Result<usize> {
    let at = bits.align_to_byte();
    let header = bits
        .input
        .get(at..at + 4)
        .ok_or_else(|| corrupt("truncated stored block"))?;
    let len = u16::from_le_bytes([header[0], header[1]]);
    let nlen = u16::from_le_bytes([header[2], header[3]]);
    if len != !nlen {
        return Err(corrupt("stored block length check"));
    }
    let len = len as usize;
    let bytes = bits
        .input
        .get(at + 4..at + 4 + len)
        .ok_or_else(|| corrupt("truncated stored block"))?;
    let target = dest
        .get_mut(out..out + len)
        .ok_or_else(|| corrupt("stream produces too many bytes"))?;
    target.copy_from_slice(bytes);
    bits.pos = at + 4 + len;
    Ok(out + len)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn corpus(n: usize, seed: u64) -> Vec<u8> {
        let words = [
            "lucene",
            "segment",
            "merge",
            "the",
            "of",
            "{\"id\":",
            "\"body\":",
            "1234",
            "GET /a",
            "HTTP/1.1",
            "200",
            "stored",
            "fields",
        ];
        let mut x = seed | 1;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x.is_multiple_of(9) {
                out.push((x >> 24) as u8);
            } else {
                out.extend_from_slice(words[(x >> 33) as usize % words.len()].as_bytes());
                out.push(b' ');
            }
        }
        out.truncate(n);
        out
    }

    fn miniz(raw: &[u8], level: u8) -> Vec<u8> {
        miniz_oxide::deflate::compress_to_vec(raw, level)
    }

    fn check(compressed: &[u8], raw: &[u8], dict: &[u8]) {
        let mut dest = dict.to_vec();
        dest.resize(dict.len() + raw.len(), 0xAA);
        Inflater::default()
            .inflate(compressed, &mut dest, dict.len(), raw.len())
            .unwrap();
        assert_eq!(&dest[dict.len()..], raw);
    }

    #[test]
    fn decodes_every_block_type_and_level() {
        for n in [0, 1, 7, 100, 300, 5000, 70_000, 300_000] {
            let raw = corpus(n, n as u64 + 3);
            for level in [0, 1, 6, 9] {
                check(&miniz(&raw, level), &raw, &[]);
            }
        }
        let runs = vec![b'z'; 100_000];
        check(&miniz(&runs, 6), &runs, &[]);
        // Every short match period, inside text so the fast loop sees them.
        for period in 2..20 {
            let mut raw = corpus(3000, period as u64);
            for i in 0..2000 {
                raw.push(b'a' + (i % period) as u8);
            }
            raw.extend_from_slice(&corpus(3000, 99));
            check(&miniz(&raw, 6), &raw, &[]);
        }
    }

    #[test]
    fn rejects_corruption_without_panicking() {
        let raw = corpus(50_000, 9);
        let good = miniz(&raw, 6);
        let mut x = 1u64;
        for _ in 0..3000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut bad = good.clone();
            let i = (x >> 33) as usize % bad.len();
            bad[i] ^= 1 << ((x >> 20) % 8);
            if x.is_multiple_of(3) {
                bad.truncate(i);
            }
            let mut dest = vec![0u8; raw.len()];
            let r = Inflater::default().inflate(&bad, &mut dest, 0, raw.len());
            if r.is_ok() {
                // A flip in a literal can still decode to the right length.
                assert_eq!(dest.len(), raw.len());
            }
        }
        // Wrong expected lengths.
        let mut dest = vec![0u8; raw.len() + 1];
        assert!(Inflater::default()
            .inflate(&good, &mut dest, 0, raw.len() + 1)
            .is_err());
        let mut dest = vec![0u8; raw.len() - 1];
        assert!(Inflater::default()
            .inflate(&good, &mut dest, 0, raw.len() - 1)
            .is_err());
        // Random garbage.
        for seed in 0..500u64 {
            let junk = corpus(64, seed);
            let mut dest = vec![0u8; 1000];
            let _ = Inflater::default().inflate(&junk, &mut dest, 0, 1000);
        }
    }

    #[test]
    fn a_distance_past_the_start_is_corruption() {
        // A fixed block: literal 'a', then length 3 at distance 2 (only one
        // byte of history exists), then end.
        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(1, 2);
        w.put_rev(0x30 + b'a' as u32, 8);
        w.put_rev(1, 7); // length symbol 257 = 3
        w.put_rev(1, 5); // distance symbol 1 = 2
        w.put_rev(0, 7); // end of block
        let bytes = w.finish();
        let mut dest = vec![0u8; 4];
        assert!(Inflater::default()
            .inflate(&bytes, &mut dest, 0, 4)
            .is_err());
        // With one dictionary byte before the output, the same stream is fine.
        let mut dest = vec![b'x', 0, 0, 0, 0];
        Inflater::default()
            .inflate(&bytes, &mut dest, 1, 4)
            .unwrap();
        assert_eq!(&dest, b"xaxax");
    }

    #[derive(Default)]
    struct BitWriter {
        out: Vec<u8>,
        acc: u64,
        n: u32,
    }

    impl BitWriter {
        fn put(&mut self, v: u32, n: u32) {
            self.acc |= (v as u64) << self.n;
            self.n += n;
            while self.n >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.n -= 8;
            }
        }
        /// A Huffman code, most significant bit first.
        fn put_rev(&mut self, code: u32, len: u32) {
            self.put(code.reverse_bits() >> (32 - len), len);
        }
        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.out.push(self.acc as u8);
            }
            self.out
        }
    }
}

#[cfg(test)]
mod timing {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    /// `INFLATE_CORPUS=file cargo test --release -p lucene-util --lib
    /// inflate_timing -- --ignored --nocapture`
    #[test]
    #[ignore = "timing harness"]
    fn inflate_timing() {
        let Ok(path) = std::env::var("INFLATE_CORPUS") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        for block in [8usize << 10, 48 << 10] {
            let blocks: Vec<&[u8]> = data.chunks(block).collect();
            let comp: Vec<Vec<u8>> = blocks
                .iter()
                .map(|b| miniz_oxide::deflate::compress_to_vec(b, 6))
                .collect();
            let mut dest = vec![0u8; block];
            let mut inf = Inflater::default();
            let mut best_new = f64::MAX;
            let mut best_old = f64::MAX;
            let mut st = Box::<miniz_oxide::inflate::core::DecompressorOxide>::default();
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for (b, c) in blocks.iter().zip(&comp) {
                    inf.inflate(c, &mut dest, 0, b.len()).unwrap();
                }
                best_new = best_new.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                for (b, c) in blocks.iter().zip(&comp) {
                    st.init();
                    let (s, _, n) = miniz_oxide::inflate::core::decompress(
                        &mut st,
                        c,
                        &mut dest[..b.len()],
                        0,
                        miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF,
                    );
                    assert_eq!(n, b.len(), "{s:?}");
                }
                best_old = best_old.min(t.elapsed().as_secs_f64());
            }
            let mb = data.len() as f64 / 1e6;
            eprintln!(
                "block {} KiB: new {:.0} MB/s, miniz {:.0} MB/s ({:.2}x)",
                block >> 10,
                mb / best_new,
                mb / best_old,
                best_old / best_new
            );
        }
    }
}
