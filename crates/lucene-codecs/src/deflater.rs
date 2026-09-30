//! Port of the compressing half of `java.util.zip.Deflater` -- that is, of
//! zlib's `deflate` (1.2.12 through 1.3.x, the versions every supported JDK
//! runs on), for exactly the configuration Lucene uses:
//! `new Deflater(6, true)` in `DeflateWithPresetDictCompressionMode`: raw
//! DEFLATE (`windowBits = -15`), `memLevel` 8, `Z_DEFAULT_STRATEGY`, level 6
//! (`deflate_slow`: lazy matching, `good_length` 8, `max_lazy` 16,
//! `nice_length` 128, `max_chain` 128), driven as that class drives it:
//! `reset()`, optionally `setDictionary(...)`, `setInput(...)`, `finish()`,
//! then `deflate(...)` until `finished()`.
//!
//! Why port zlib rather than call a DEFLATE crate: a DEFLATE stream is not
//! unique, and Lucene's `.fdt` bytes in `BEST_COMPRESSION` are whatever
//! zlib's match finder and Huffman builder chose. Reproducing them means
//! reproducing zlib's `longest_match`, `fill_window` (including the stale
//! window bytes past the input it may compare against), `deflate_slow`'s lazy
//! evaluation and `trees.c`'s tree construction, tie for tie. The names below
//! are zlib's.
//!
//! State persists across [`Deflater::reset`] exactly as zlib's does: the
//! window buffer (and its `high_water` mark) and the `prev` chain array are
//! never cleared, because `deflateReset` does not clear them -- and
//! `longest_match` can read window bytes past the current input when it
//! compares a candidate's tail. One `Deflater` per stored-fields writer, as
//! Java keeps one `DeflateWithPresetDictCompressor` per writer.

// --- deflate.h / deflate.c constants --------------------------------------

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;
const W_BITS: usize = 15;
const W_SIZE: usize = 1 << W_BITS;
const W_MASK: usize = W_SIZE - 1;
const WINDOW_SIZE: usize = 2 * W_SIZE;
const MAX_DIST: usize = W_SIZE - MIN_LOOKAHEAD;
/// `memLevel` 8: `hash_bits = memLevel + 7`.
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const HASH_SHIFT: usize = HASH_BITS.div_ceil(MIN_MATCH);
/// `lit_bufsize = 1 << (memLevel + 6)`; a block holds `lit_bufsize - 1`
/// symbols (`sym_end`).
const LIT_BUFSIZE: usize = 1 << 14;
const WIN_INIT: usize = MAX_MATCH;
const TOO_FAR: usize = 4096;
const NIL: usize = 0;

// Level 6 in `configuration_table`.
const GOOD_MATCH: usize = 8;
const MAX_LAZY_MATCH: usize = 16;
const NICE_MATCH: usize = 128;
const MAX_CHAIN: usize = 128;

// --- trees.c constants -----------------------------------------------------

const LENGTH_CODES: usize = 29;
const LITERALS: usize = 256;
const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;
const STORED_BLOCK: u32 = 0;
const STATIC_TREES: u32 = 1;
const DYN_TREES: u32 = 2;

const EXTRA_LBITS: [u8; LENGTH_CODES] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const EXTRA_DBITS: [u8; D_CODES] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [usize; BL_CODES] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// `tr_static_init`'s tables.
struct StaticTables {
    /// `static_ltree`: (code, len) for `L_CODES + 2` entries.
    ltree_code: [u16; L_CODES + 2],
    ltree_len: [u16; L_CODES + 2],
    dtree_code: [u16; D_CODES],
    dtree_len: [u16; D_CODES],
    dist_code: [u8; 512],
    length_code: [u8; MAX_MATCH - MIN_MATCH + 1],
    base_length: [u16; LENGTH_CODES],
    base_dist: [u16; D_CODES],
}

// ARITH: an encoder over in-memory buffers of known size, following zlib's
// own unsigned arithmetic: window positions are below `2 * W_SIZE`, chain
// entries are below `W_SIZE` after `slide_hash`, lengths are bounded by
// `MAX_MATCH`, code lengths by `MAX_BITS`, and the bit accumulator never
// holds more than 7 + 16 bits. The two `ulg` counters zlib lets wrap
// (`opt_len`, `static_len`) use `wrapping_*` explicitly.
#[allow(clippy::arithmetic_side_effects)]
/// `bi_reverse`.
fn bi_reverse(mut code: u32, mut len: u32) -> u32 {
    let mut res = 0u32;
    loop {
        res |= code & 1;
        code >>= 1;
        res <<= 1;
        len -= 1;
        if len == 0 {
            break;
        }
    }
    res >> 1
}

// ARITH: an encoder over in-memory buffers of known size, following zlib's
// own unsigned arithmetic: window positions are below `2 * W_SIZE`, chain
// entries are below `W_SIZE` after `slide_hash`, lengths are bounded by
// `MAX_MATCH`, code lengths by `MAX_BITS`, and the bit accumulator never
// holds more than 7 + 16 bits. The two `ulg` counters zlib lets wrap
// (`opt_len`, `static_len`) use `wrapping_*` explicitly.
#[allow(clippy::arithmetic_side_effects)]
/// `gen_codes`: canonical codes from bit lengths.
fn gen_codes(code: &mut [u16], len: &[u16], max_code: usize, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut c = 0u32;
    for bits in 1..=MAX_BITS {
        c = (c + u32::from(bl_count[bits - 1])) << 1;
        next_code[bits] = c as u16;
    }
    for n in 0..=max_code {
        let l = len[n] as usize;
        if l == 0 {
            continue;
        }
        code[n] = bi_reverse(u32::from(next_code[l]), l as u32) as u16;
        next_code[l] = next_code[l].wrapping_add(1);
    }
}

// ARITH: an encoder over in-memory buffers of known size, following zlib's
// own unsigned arithmetic: window positions are below `2 * W_SIZE`, chain
// entries are below `W_SIZE` after `slide_hash`, lengths are bounded by
// `MAX_MATCH`, code lengths by `MAX_BITS`, and the bit accumulator never
// holds more than 7 + 16 bits. The two `ulg` counters zlib lets wrap
// (`opt_len`, `static_len`) use `wrapping_*` explicitly.
#[allow(clippy::arithmetic_side_effects)]
impl StaticTables {
    fn new() -> Self {
        let mut t = StaticTables {
            ltree_code: [0; L_CODES + 2],
            ltree_len: [0; L_CODES + 2],
            dtree_code: [0; D_CODES],
            dtree_len: [0; D_CODES],
            dist_code: [0; 512],
            length_code: [0; MAX_MATCH - MIN_MATCH + 1],
            base_length: [0; LENGTH_CODES],
            base_dist: [0; D_CODES],
        };
        let mut length = 0usize;
        let mut code = 0usize;
        while code < LENGTH_CODES - 1 {
            t.base_length[code] = length as u16;
            for _ in 0..(1usize << EXTRA_LBITS[code]) {
                t.length_code[length] = code as u8;
                length += 1;
            }
            code += 1;
        }
        // The length 258 (`length == 256` here) takes the last code.
        t.length_code[length - 1] = code as u8;
        let mut dist = 0usize;
        code = 0;
        while code < 16 {
            t.base_dist[code] = dist as u16;
            for _ in 0..(1usize << EXTRA_DBITS[code]) {
                t.dist_code[dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        dist >>= 7;
        while code < D_CODES {
            t.base_dist[code] = (dist << 7) as u16;
            for _ in 0..(1usize << (EXTRA_DBITS[code] - 7)) {
                t.dist_code[256 + dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        let mut bl_count = [0u16; MAX_BITS + 1];
        for n in 0..=L_CODES + 1 {
            let l = match n {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            };
            t.ltree_len[n] = l;
            bl_count[l as usize] += 1;
        }
        let lens = t.ltree_len;
        gen_codes(&mut t.ltree_code, &lens, L_CODES + 1, &bl_count);
        for n in 0..D_CODES {
            t.dtree_len[n] = 5;
            t.dtree_code[n] = bi_reverse(n as u32, 5) as u16;
        }
        t
    }

    /// `d_code(dist)`.
    fn d_code(&self, dist: usize) -> usize {
        if dist < 256 {
            self.dist_code[dist] as usize
        } else {
            self.dist_code[256 + (dist >> 7)] as usize
        }
    }
}

/// Which of the three trees a `tree_desc` describes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TreeKind {
    Literal,
    Distance,
    BitLength,
}

/// A dynamic tree: zlib's `ct_data` array split into its two unions --
/// `freq`/`code` share one field, `dad`/`len` the other, and zlib relies on
/// the overwrites (`gen_bitlen` writes `len` over `dad`, `gen_codes` writes
/// `code` over `freq`), so they are one array each here too.
struct Tree {
    fc: Vec<u16>,
    dl: Vec<u16>,
    max_code: usize,
}

impl Tree {
    fn new(size: usize) -> Self {
        Tree {
            fc: vec![0; size],
            dl: vec![0; size],
            max_code: 0,
        }
    }
}

/// `deflate_state` for one `Deflater`.
pub(crate) struct Deflater {
    tables: StaticTables,
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    high_water: usize,
    ins_h: usize,
    strstart: usize,
    /// Signed: `fill_window` may slide it below zero.
    block_start: i64,
    lookahead: usize,
    insert: usize,
    match_start: usize,
    match_length: usize,
    prev_match: usize,
    prev_length: usize,
    match_available: bool,
    /// The input `setInput` gave, and how much of it `read_buf` consumed.
    input: Vec<u8>,
    input_pos: usize,
    // trees.c
    ltree: Tree,
    dtree: Tree,
    bltree: Tree,
    heap: [usize; HEAP_SIZE],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; HEAP_SIZE],
    bl_count: [u16; MAX_BITS + 1],
    /// `sym_buf`: (dist, lc) per symbol, `dist == 0` for a literal.
    syms: Vec<(u16, u8)>,
    opt_len: u64,
    static_len: u64,
    bi_buf: u64,
    bi_valid: u32,
}

// ARITH: an encoder over in-memory buffers of known size, following zlib's
// own unsigned arithmetic: window positions are below `2 * W_SIZE`, chain
// entries are below `W_SIZE` after `slide_hash`, lengths are bounded by
// `MAX_MATCH`, code lengths by `MAX_BITS`, and the bit accumulator never
// holds more than 7 + 16 bits. The two `ulg` counters zlib lets wrap
// (`opt_len`, `static_len`) use `wrapping_*` explicitly.
#[allow(clippy::arithmetic_side_effects)]
impl Deflater {
    /// `new Deflater(6, true)`: `deflateInit2(level 6, -15, memLevel 8)`.
    pub(crate) fn new() -> Self {
        let mut d = Deflater {
            tables: StaticTables::new(),
            window: vec![0; WINDOW_SIZE],
            prev: vec![0; W_SIZE],
            head: vec![0; HASH_SIZE],
            high_water: 0,
            ins_h: 0,
            strstart: 0,
            block_start: 0,
            lookahead: 0,
            insert: 0,
            match_start: 0,
            match_length: 0,
            prev_match: 0,
            prev_length: 0,
            match_available: false,
            input: Vec::new(),
            input_pos: 0,
            ltree: Tree::new(HEAP_SIZE),
            dtree: Tree::new(2 * D_CODES + 1),
            bltree: Tree::new(2 * BL_CODES + 1),
            heap: [0; HEAP_SIZE],
            heap_len: 0,
            heap_max: 0,
            depth: [0; HEAP_SIZE],
            bl_count: [0; MAX_BITS + 1],
            syms: Vec::with_capacity(LIT_BUFSIZE),
            opt_len: 0,
            static_len: 0,
            bi_buf: 0,
            bi_valid: 0,
        };
        d.reset();
        d
    }

    /// `Deflater.reset()` -> `deflateReset`: `_tr_init` and `lm_init`. The
    /// window, `prev` and `high_water` are kept, as zlib keeps them.
    pub(crate) fn reset(&mut self) {
        self.input.clear();
        self.input_pos = 0;
        // `_tr_init`.
        self.bi_buf = 0;
        self.bi_valid = 0;
        self.init_block();
        // `lm_init`.
        self.head.fill(0);
        self.strstart = 0;
        self.block_start = 0;
        self.lookahead = 0;
        self.insert = 0;
        self.match_length = MIN_MATCH - 1;
        self.prev_length = MIN_MATCH - 1;
        self.match_available = false;
        self.ins_h = 0;
    }

    /// `Deflater.setDictionary(b, off, len)` -> `deflateSetDictionary` on a
    /// raw stream.
    pub(crate) fn set_dictionary(&mut self, dictionary: &[u8]) {
        let mut dictionary = dictionary;
        if dictionary.len() >= W_SIZE {
            self.head.fill(0);
            self.strstart = 0;
            self.block_start = 0;
            self.insert = 0;
            dictionary = &dictionary[dictionary.len() - W_SIZE..];
        }
        let saved_input = std::mem::replace(&mut self.input, dictionary.to_vec());
        let saved_pos = std::mem::replace(&mut self.input_pos, 0);
        self.fill_window();
        while self.lookahead >= MIN_MATCH {
            let mut s = self.strstart;
            let mut n = self.lookahead - (MIN_MATCH - 1);
            loop {
                self.update_hash(self.window[s + MIN_MATCH - 1]);
                self.prev[s & W_MASK] = self.head[self.ins_h];
                self.head[self.ins_h] = s as u16;
                s += 1;
                n -= 1;
                if n == 0 {
                    break;
                }
            }
            self.strstart = s;
            self.lookahead = MIN_MATCH - 1;
            self.fill_window();
        }
        self.strstart += self.lookahead;
        self.block_start = self.strstart as i64;
        self.insert = self.lookahead;
        self.lookahead = 0;
        self.match_length = MIN_MATCH - 1;
        self.prev_length = MIN_MATCH - 1;
        self.match_available = false;
        self.input = saved_input;
        self.input_pos = saved_pos;
    }

    /// `setInput(input)`, `finish()`, then `deflate` until `finished()`:
    /// appends the whole raw DEFLATE stream for `input` to `out`. (Java's
    /// output buffer size never changes the bytes: zlib cuts blocks by
    /// symbol count, not by output space.)
    pub(crate) fn compress(&mut self, input: &[u8], out: &mut Vec<u8>) {
        self.input.clear();
        self.input.extend_from_slice(input);
        self.input_pos = 0;
        self.deflate_slow(out);
        self.input.clear();
        self.input_pos = 0;
    }

    #[inline]
    fn update_hash(&mut self, c: u8) {
        self.ins_h = ((self.ins_h << HASH_SHIFT) ^ usize::from(c)) & HASH_MASK;
    }

    /// `INSERT_STRING`: returns the previous head of the chain.
    #[inline]
    fn insert_string(&mut self, s: usize) -> usize {
        self.update_hash(self.window[s + MIN_MATCH - 1]);
        let head = self.head[self.ins_h];
        self.prev[s & W_MASK] = head;
        self.head[self.ins_h] = s as u16;
        usize::from(head)
    }

    /// `read_buf`.
    fn read_buf(&mut self, at: usize, size: usize) -> usize {
        let len = (self.input.len() - self.input_pos).min(size);
        if len == 0 {
            return 0;
        }
        self.window[at..at + len]
            .copy_from_slice(&self.input[self.input_pos..self.input_pos + len]);
        self.input_pos += len;
        len
    }

    fn avail_in(&self) -> usize {
        self.input.len() - self.input_pos
    }

    /// `slide_hash`.
    fn slide_hash(&mut self) {
        for p in self.head.iter_mut().chain(self.prev.iter_mut()) {
            let m = usize::from(*p);
            *p = if m >= W_SIZE {
                (m - W_SIZE) as u16
            } else {
                NIL as u16
            };
        }
    }

    /// `fill_window`.
    fn fill_window(&mut self) {
        loop {
            let mut more = WINDOW_SIZE - self.lookahead - self.strstart;
            if self.strstart >= W_SIZE + MAX_DIST {
                self.window.copy_within(W_SIZE..W_SIZE + (W_SIZE - more), 0);
                self.match_start = self.match_start.wrapping_sub(W_SIZE);
                self.strstart -= W_SIZE;
                self.block_start -= W_SIZE as i64;
                if self.insert > self.strstart {
                    self.insert = self.strstart;
                }
                self.slide_hash();
                more += W_SIZE;
            }
            if self.avail_in() == 0 {
                break;
            }
            let n = self.read_buf(self.strstart + self.lookahead, more);
            self.lookahead += n;
            if self.lookahead + self.insert >= MIN_MATCH {
                let mut s = self.strstart - self.insert;
                self.ins_h = usize::from(self.window[s]);
                self.update_hash(self.window[s + 1]);
                while self.insert > 0 {
                    self.update_hash(self.window[s + MIN_MATCH - 1]);
                    self.prev[s & W_MASK] = self.head[self.ins_h];
                    self.head[self.ins_h] = s as u16;
                    s += 1;
                    self.insert -= 1;
                    if self.lookahead + self.insert < MIN_MATCH {
                        break;
                    }
                }
            }
            if !(self.lookahead < MIN_LOOKAHEAD && self.avail_in() != 0) {
                break;
            }
        }
        // Zero `WIN_INIT` bytes past the data, once, so `longest_match` never
        // compares uninitialised memory.
        if self.high_water < WINDOW_SIZE {
            let curr = self.strstart + self.lookahead;
            if self.high_water < curr {
                let init = (WINDOW_SIZE - curr).min(WIN_INIT);
                self.window[curr..curr + init].fill(0);
                self.high_water = curr + init;
            } else if self.high_water < curr + WIN_INIT {
                let init = (curr + WIN_INIT - self.high_water).min(WINDOW_SIZE - self.high_water);
                self.window[self.high_water..self.high_water + init].fill(0);
                self.high_water += init;
            }
        }
    }

    /// `longest_match` (the portable, not `UNALIGNED_OK`, variant). Byte 2
    /// of a candidate is never compared, as zlib does not: equal hashes with
    /// equal first two bytes imply it.
    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let mut chain_length = MAX_CHAIN;
        let scan = self.strstart;
        let mut best_len = self.prev_length;
        let mut nice_match = NICE_MATCH;
        let limit = if self.strstart > MAX_DIST {
            self.strstart - MAX_DIST
        } else {
            NIL
        };
        let strend = self.strstart + MAX_MATCH;
        let w = &self.window;
        let mut scan_end1 = w[scan + best_len - 1];
        let mut scan_end = w[scan + best_len];
        if self.prev_length >= GOOD_MATCH {
            chain_length >>= 2;
        }
        if nice_match > self.lookahead {
            nice_match = self.lookahead;
        }
        loop {
            let m = cur_match;
            let skip = w[m + best_len] != scan_end
                || w[m + best_len - 1] != scan_end1
                || w[m] != w[scan]
                || w[m + 1] != w[scan + 1];
            if !skip {
                let mut s = scan + 2;
                let mut mm = m + 2;
                'outer: loop {
                    for _ in 0..8 {
                        s += 1;
                        mm += 1;
                        if w[s] != w[mm] {
                            break 'outer;
                        }
                    }
                    if s >= strend {
                        break;
                    }
                }
                let len = s - scan;
                if len > best_len {
                    self.match_start = cur_match;
                    best_len = len;
                    if len >= nice_match {
                        break;
                    }
                    scan_end1 = w[scan + best_len - 1];
                    scan_end = w[scan + best_len];
                }
            }
            cur_match = usize::from(self.prev[cur_match & W_MASK]);
            if cur_match <= limit {
                break;
            }
            chain_length -= 1;
            if chain_length == 0 {
                break;
            }
        }
        best_len.min(self.lookahead)
    }

    /// `deflate_slow(s, Z_FINISH)`, run to completion.
    fn deflate_slow(&mut self, out: &mut Vec<u8>) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = NIL;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != NIL
                && self.prev_length < MAX_LAZY_MATCH
                && self.strstart - hash_head <= MAX_DIST
            {
                self.match_length = self.longest_match(hash_head);
                if self.match_length <= 5
                    && self.match_length == MIN_MATCH
                    && self.strstart - self.match_start > TOO_FAR
                {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                let bflush = self.tally_dist(
                    self.strstart - 1 - self.prev_match,
                    self.prev_length - MIN_MATCH,
                );
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block(out, false);
                }
            } else if self.match_available {
                let bflush = self.tally_lit(self.window[self.strstart - 1]);
                if bflush {
                    self.flush_block(out, false);
                }
                self.strstart += 1;
                self.lookahead -= 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            self.tally_lit(self.window[self.strstart - 1]);
            self.match_available = false;
        }
        self.insert = self.strstart.min(MIN_MATCH - 1);
        self.flush_block(out, true);
    }

    /// `FLUSH_BLOCK_ONLY`.
    fn flush_block(&mut self, out: &mut Vec<u8>, last: bool) {
        let stored = if self.block_start >= 0 {
            let start = self.block_start as usize;
            Some((start, self.strstart - start))
        } else {
            None
        };
        let stored_len = (self.strstart as i64 - self.block_start) as usize;
        self.tr_flush_block(out, stored, stored_len, last);
        self.block_start = self.strstart as i64;
    }

    // --- trees.c ---------------------------------------------------------

    /// `init_block`.
    fn init_block(&mut self) {
        self.ltree.fc[..L_CODES].fill(0);
        self.dtree.fc[..D_CODES].fill(0);
        self.bltree.fc[..BL_CODES].fill(0);
        self.ltree.fc[END_BLOCK] = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.syms.clear();
    }

    /// `_tr_tally_lit`: returns whether the block is full.
    fn tally_lit(&mut self, c: u8) -> bool {
        self.syms.push((0, c));
        self.ltree.fc[usize::from(c)] += 1;
        self.syms.len() == LIT_BUFSIZE - 1
    }

    /// `_tr_tally_dist`.
    fn tally_dist(&mut self, dist: usize, len: usize) -> bool {
        self.syms.push((dist as u16, len as u8));
        let d = dist - 1;
        self.ltree.fc[usize::from(self.tables.length_code[len]) + LITERALS + 1] += 1;
        let dc = self.tables.d_code(d);
        self.dtree.fc[dc] += 1;
        self.syms.len() == LIT_BUFSIZE - 1
    }

    fn send_bits(&mut self, out: &mut Vec<u8>, value: u32, length: u32) {
        self.bi_buf |= u64::from(value) << self.bi_valid;
        self.bi_valid += length;
        while self.bi_valid >= 8 {
            out.push(self.bi_buf as u8);
            self.bi_buf >>= 8;
            self.bi_valid -= 8;
        }
    }

    /// `bi_windup`.
    fn bi_windup(&mut self, out: &mut Vec<u8>) {
        if self.bi_valid > 0 {
            out.push(self.bi_buf as u8);
        }
        self.bi_buf = 0;
        self.bi_valid = 0;
    }

    fn tree(&self, kind: TreeKind) -> &Tree {
        match kind {
            TreeKind::Literal => &self.ltree,
            TreeKind::Distance => &self.dtree,
            TreeKind::BitLength => &self.bltree,
        }
    }

    fn tree_mut(&mut self, kind: TreeKind) -> &mut Tree {
        match kind {
            TreeKind::Literal => &mut self.ltree,
            TreeKind::Distance => &mut self.dtree,
            TreeKind::BitLength => &mut self.bltree,
        }
    }

    /// `smaller(tree, n, m, depth)`.
    fn smaller(&self, kind: TreeKind, n: usize, m: usize) -> bool {
        let t = self.tree(kind);
        t.fc[n] < t.fc[m] || (t.fc[n] == t.fc[m] && self.depth[n] <= self.depth[m])
    }

    /// `pqdownheap`.
    fn pqdownheap(&mut self, kind: TreeKind, mut k: usize) {
        let v = self.heap[k];
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len && self.smaller(kind, self.heap[j + 1], self.heap[j]) {
                j += 1;
            }
            if self.smaller(kind, v, self.heap[j]) {
                break;
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    /// `(static_tree len, extra_bits, extra_base, elems, max_length)` of a
    /// `static_tree_desc`.
    fn stat_desc(kind: TreeKind) -> (&'static [u8], usize, usize, usize) {
        match kind {
            TreeKind::Literal => (&EXTRA_LBITS, LITERALS + 1, L_CODES, MAX_BITS),
            TreeKind::Distance => (&EXTRA_DBITS, 0, D_CODES, MAX_BITS),
            TreeKind::BitLength => (&EXTRA_BLBITS, 0, BL_CODES, MAX_BL_BITS),
        }
    }

    fn static_len_of(&self, kind: TreeKind, n: usize) -> Option<u16> {
        match kind {
            TreeKind::Literal => Some(self.tables.ltree_len[n]),
            TreeKind::Distance => Some(self.tables.dtree_len[n]),
            TreeKind::BitLength => None,
        }
    }

    /// `build_tree`.
    fn build_tree(&mut self, kind: TreeKind) {
        let (_, _, elems, _) = Self::stat_desc(kind);
        let mut max_code: i64 = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for n in 0..elems {
            if self.tree(kind).fc[n] != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n;
                max_code = n as i64;
                self.depth[n] = 0;
            } else {
                self.tree_mut(kind).dl[n] = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node;
            self.tree_mut(kind).fc[node] = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if let Some(l) = self.static_len_of(kind, node) {
                self.static_len = self.static_len.wrapping_sub(u64::from(l));
            }
        }
        let max_code = max_code as usize;
        self.tree_mut(kind).max_code = max_code;
        let mut n = self.heap_len / 2;
        while n >= 1 {
            self.pqdownheap(kind, n);
            n -= 1;
        }
        let mut node = elems;
        loop {
            // `pqremove`.
            let n = self.heap[1];
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(kind, 1);
            let m = self.heap[1];
            self.heap_max -= 1;
            self.heap[self.heap_max] = n;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m;
            let t = self.tree_mut(kind);
            t.fc[node] = t.fc[n].wrapping_add(t.fc[m]);
            t.dl[n] = node as u16;
            t.dl[m] = node as u16;
            self.depth[node] = self.depth[n].max(self.depth[m]) + 1;
            self.heap[1] = node;
            node += 1;
            self.pqdownheap(kind, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(kind);
        let bl_count = self.bl_count;
        let t = self.tree_mut(kind);
        let len = t.dl.clone();
        gen_codes(&mut t.fc, &len, max_code, &bl_count);
    }

    /// `gen_bitlen`.
    fn gen_bitlen(&mut self, kind: TreeKind) {
        let (extra, base, _, max_length) = Self::stat_desc(kind);
        let max_code = self.tree(kind).max_code;
        self.bl_count = [0; MAX_BITS + 1];
        let root = self.heap[self.heap_max];
        self.tree_mut(kind).dl[root] = 0;
        let mut overflow = 0i32;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h];
            let dad = usize::from(self.tree(kind).dl[n]);
            let mut bits = usize::from(self.tree(kind).dl[dad]) + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            self.tree_mut(kind).dl[n] = bits as u16;
            h += 1;
            if n > max_code {
                continue;
            }
            self.bl_count[bits] += 1;
            let xbits = if n >= base {
                usize::from(extra[n - base])
            } else {
                0
            };
            let f = u64::from(self.tree(kind).fc[n]);
            self.opt_len = self
                .opt_len
                .wrapping_add(f.wrapping_mul((bits + xbits) as u64));
            if let Some(sl) = self.static_len_of(kind, n) {
                self.static_len = self
                    .static_len
                    .wrapping_add(f.wrapping_mul(u64::from(sl) + xbits as u64));
            }
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length - 1;
            while self.bl_count[bits] == 0 {
                bits -= 1;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        let mut bits = max_length;
        while bits != 0 {
            let mut n = self.bl_count[bits];
            while n != 0 {
                h -= 1;
                let m = self.heap[h];
                if m > max_code {
                    continue;
                }
                let t = self.tree(kind);
                if usize::from(t.dl[m]) != bits {
                    let delta = (bits as u64).wrapping_sub(u64::from(t.dl[m]));
                    let f = u64::from(t.fc[m]);
                    self.opt_len = self.opt_len.wrapping_add(delta.wrapping_mul(f));
                    self.tree_mut(kind).dl[m] = bits as u16;
                }
                n -= 1;
            }
            bits -= 1;
        }
    }

    /// `scan_tree`: bit-length-code frequencies for one tree.
    fn scan_tree(&mut self, kind: TreeKind) {
        let max_code = self.tree(kind).max_code;
        let mut prevlen: i32 = -1;
        let mut nextlen = i32::from(self.tree(kind).dl[0]);
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        self.tree_mut(kind).dl[max_code + 1] = 0xffff; // guard
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = i32::from(self.tree(kind).dl[n + 1]);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                self.bltree.fc[curlen as usize] += count as u16;
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.bltree.fc[curlen as usize] += 1;
                }
                self.bltree.fc[REP_3_6] += 1;
            } else if count <= 10 {
                self.bltree.fc[REPZ_3_10] += 1;
            } else {
                self.bltree.fc[REPZ_11_138] += 1;
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 {
                (138, 3)
            } else if curlen == nextlen {
                (6, 3)
            } else {
                (7, 4)
            };
        }
    }

    /// `send_code` for the bit-length tree.
    fn send_bl(&mut self, out: &mut Vec<u8>, c: usize) {
        let (code, len) = (self.bltree.fc[c], self.bltree.dl[c]);
        self.send_bits(out, u32::from(code), u32::from(len));
    }

    /// `send_tree`.
    fn send_tree(&mut self, out: &mut Vec<u8>, kind: TreeKind, max_code: usize) {
        let mut prevlen: i32 = -1;
        let mut nextlen = i32::from(self.tree(kind).dl[0]);
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = i32::from(self.tree(kind).dl[n + 1]);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                loop {
                    self.send_bl(out, curlen as usize);
                    count -= 1;
                    if count == 0 {
                        break;
                    }
                }
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.send_bl(out, curlen as usize);
                    count -= 1;
                }
                self.send_bl(out, REP_3_6);
                self.send_bits(out, (count - 3) as u32, 2);
            } else if count <= 10 {
                self.send_bl(out, REPZ_3_10);
                self.send_bits(out, (count - 3) as u32, 3);
            } else {
                self.send_bl(out, REPZ_11_138);
                self.send_bits(out, (count - 11) as u32, 7);
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 {
                (138, 3)
            } else if curlen == nextlen {
                (6, 3)
            } else {
                (7, 4)
            };
        }
    }

    /// `build_bl_tree`: returns `max_blindex`.
    fn build_bl_tree(&mut self) -> usize {
        self.scan_tree(TreeKind::Literal);
        self.scan_tree(TreeKind::Distance);
        self.build_tree(TreeKind::BitLength);
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bltree.dl[BL_ORDER[max_blindex]] != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.opt_len = self
            .opt_len
            .wrapping_add(3 * (max_blindex as u64 + 1) + 5 + 5 + 4);
        max_blindex
    }

    /// `send_all_trees`.
    fn send_all_trees(&mut self, out: &mut Vec<u8>, lcodes: usize, dcodes: usize, blcodes: usize) {
        self.send_bits(out, (lcodes - 257) as u32, 5);
        self.send_bits(out, (dcodes - 1) as u32, 5);
        self.send_bits(out, (blcodes - 4) as u32, 4);
        for &o in &BL_ORDER[..blcodes] {
            let len = self.bltree.dl[o];
            self.send_bits(out, u32::from(len), 3);
        }
        self.send_tree(out, TreeKind::Literal, lcodes - 1);
        self.send_tree(out, TreeKind::Distance, dcodes - 1);
    }

    /// `compress_block` with the static (`dynamic == false`) or dynamic
    /// trees.
    fn compress_block(&mut self, out: &mut Vec<u8>, dynamic: bool) {
        let syms = std::mem::take(&mut self.syms);
        let lcode = |d: &Self, c: usize| {
            if dynamic {
                (d.ltree.fc[c], d.ltree.dl[c])
            } else {
                (d.tables.ltree_code[c], d.tables.ltree_len[c])
            }
        };
        let dcode = |d: &Self, c: usize| {
            if dynamic {
                (d.dtree.fc[c], d.dtree.dl[c])
            } else {
                (d.tables.dtree_code[c], d.tables.dtree_len[c])
            }
        };
        for &(dist, lc) in &syms {
            if dist == 0 {
                let (c, l) = lcode(self, usize::from(lc));
                self.send_bits(out, u32::from(c), u32::from(l));
            } else {
                let lc = usize::from(lc);
                let code = usize::from(self.tables.length_code[lc]);
                let (c, l) = lcode(self, code + LITERALS + 1);
                self.send_bits(out, u32::from(c), u32::from(l));
                let extra = u32::from(EXTRA_LBITS[code]);
                if extra != 0 {
                    let v = lc - usize::from(self.tables.base_length[code]);
                    self.send_bits(out, v as u32, extra);
                }
                let d = usize::from(dist) - 1;
                let code = self.tables.d_code(d);
                let (c, l) = dcode(self, code);
                self.send_bits(out, u32::from(c), u32::from(l));
                let extra = u32::from(EXTRA_DBITS[code]);
                if extra != 0 {
                    let v = d - usize::from(self.tables.base_dist[code]);
                    self.send_bits(out, v as u32, extra);
                }
            }
        }
        let (c, l) = lcode(self, END_BLOCK);
        self.send_bits(out, u32::from(c), u32::from(l));
        self.syms = syms;
    }

    /// `_tr_stored_block`.
    fn stored_block(&mut self, out: &mut Vec<u8>, start: usize, len: usize, last: bool) {
        self.send_bits(out, (STORED_BLOCK << 1) | u32::from(last), 3);
        self.bi_windup(out);
        out.extend_from_slice(&(len as u16).to_le_bytes());
        out.extend_from_slice(&(!(len as u16)).to_le_bytes());
        out.extend_from_slice(&self.window[start..start + len]);
    }

    /// `_tr_flush_block` at a level above 0.
    fn tr_flush_block(
        &mut self,
        out: &mut Vec<u8>,
        stored: Option<(usize, usize)>,
        stored_len: usize,
        last: bool,
    ) {
        self.build_tree(TreeKind::Literal);
        self.build_tree(TreeKind::Distance);
        let max_blindex = self.build_bl_tree();
        let mut opt_lenb = self.opt_len.wrapping_add(3 + 7) >> 3;
        let static_lenb = self.static_len.wrapping_add(3 + 7) >> 3;
        if static_lenb <= opt_lenb {
            opt_lenb = static_lenb;
        }
        match stored {
            Some((start, len)) if stored_len as u64 + 4 <= opt_lenb => {
                self.stored_block(out, start, len, last);
            }
            _ if static_lenb == opt_lenb => {
                self.send_bits(out, (STATIC_TREES << 1) | u32::from(last), 3);
                self.compress_block(out, false);
            }
            _ => {
                self.send_bits(out, (DYN_TREES << 1) | u32::from(last), 3);
                let (lcodes, dcodes) = (self.ltree.max_code + 1, self.dtree.max_code + 1);
                self.send_all_trees(out, lcodes, dcodes, max_blindex + 1);
                self.compress_block(out, true);
            }
        }
        self.init_block();
        if last {
            self.bi_windup(out);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::deflate;
    use lucene_store::data_input::SliceInput;

    fn inflate(dict: &[u8], compressed: &[u8], len: usize) -> Vec<u8> {
        let mut dest = dict.to_vec();
        dest.resize(dict.len() + len, 0);
        let mut input = SliceInput::new(compressed);
        deflate::decompress(&mut input, compressed.len(), len, &mut dest, dict.len()).unwrap();
        dest[dict.len()..].to_vec()
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed >> 33
    }

    /// Every shape round-trips through the inflater, with and without a
    /// dictionary, across resets that leave stale window bytes behind:
    /// text (dynamic trees), random bytes (stored blocks), a run (long
    /// matches), tiny inputs (static trees), inputs past the window
    /// (sliding) and past a block's symbol budget.
    #[test]
    fn round_trips_through_inflate() {
        let mut seed = 42u64;
        let text: Vec<u8> = (0..200_000)
            .map(|i| b"the quick brown fox jumps over the lazy dog "[i % 41 + (i / 997) % 3])
            .collect();
        let random: Vec<u8> = (0..70_000).map(|_| lcg(&mut seed) as u8).collect();
        let run = vec![b'a'; 100_000];
        let words: Vec<u8> = (0..150_000)
            .map(|_| b'a' + (lcg(&mut seed) % 6) as u8)
            .collect();
        let mut d = Deflater::new();
        for input in [
            &text[..],
            &random,
            &run,
            &words,
            b"x",
            b"abcabcabc",
            &[][..],
        ] {
            for dict_len in [0usize, 5, 1000, 40_000] {
                let dict = &text[text.len() - dict_len..];
                d.reset();
                if dict_len > 0 {
                    d.set_dictionary(dict);
                }
                let mut out = Vec::new();
                d.compress(input, &mut out);
                assert_eq!(inflate(dict, &out, input.len()), input, "dict {dict_len}");
            }
        }
    }

    #[test]
    fn static_tables_match_rfc1951() {
        let t = StaticTables::new();
        // Literal 0: 8 bits, code 00110000 reversed.
        assert_eq!(t.ltree_len[0], 8);
        assert_eq!(t.ltree_code[0], bi_reverse(0x30, 8) as u16);
        assert_eq!(t.ltree_len[256], 7);
        assert_eq!(t.ltree_code[256], 0);
        assert_eq!(t.length_code[255], 28);
        assert_eq!(t.base_dist[29], 24576);
        assert_eq!(t.d_code(32767), 29);
    }
}
