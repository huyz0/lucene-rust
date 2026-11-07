//! A high-compression LZ4 block compressor: the hash-chain match finder of the
//! reference `lz4hc` library (levels 3 to 9), writing the plain LZ4 block
//! format that [`crate::lz4::decompress`] and Java's `LZ4.decompress` read.
//!
//! It finds longer matches than Lucene's own `HighCompressionHashTable`
//! because each match may be widened backwards into the literals before it,
//! and because up to three overlapping candidates are weighed against each
//! other before the first is emitted (`lz4hc.c`'s `ML1`/`ML2`/`ML3`
//! search). Its output is a different, smaller, byte stream for the same
//! input; any LZ4 decoder reads it.
//!
//! Like [`crate::lz4::compress_with_dictionary`], it compresses
//! `bytes[dict_off + dict_len..][..len]` against the preset dictionary
//! `bytes[dict_off..][..dict_len]` that sits right before it, which is the
//! layout of a `LZ4WithPresetDictCompressionMode` sub-block.
//!
//! Positions are stored as `u32` offsets from a base that moves forward on
//! every call, so neither table is cleared between calls: an entry left by an
//! earlier call lies below the current window and is never followed.

use crate::lz4::{encode_last_literals, encode_sequence, MIN_MATCH};

/// `LZ4HC_HASH_LOG`.
const HASH_LOG: u32 = 15;
/// `LZ4HC_MAXD`: the chain table covers the whole 64 KiB window.
const CHAIN_SIZE: usize = 1 << 16;
const CHAIN_MASK: usize = CHAIN_SIZE - 1;
/// The longest distance an LZ4 offset can encode.
const MAX_DISTANCE: usize = (1 << 16) - 1;
/// `MFLIMIT`: no match starts in the last 12 bytes of a block.
const MF_LIMIT: usize = 12;
/// `LASTLITERALS`: the last 5 bytes of a block are always literals.
const LAST_LITERALS: usize = 5;
/// `OPTIMAL_ML`: the longest match whose length still fits in the token.
const OPTIMAL_ML: usize = 0x0F - 1 + MIN_MATCH;
/// Positions the base may advance to before both tables are cleared.
const REBASE_LIMIT: usize = 1 << 30;
/// The `lz4hc` level 9 search depth.
pub const DEFAULT_MAX_ATTEMPTS: usize = 256;

/// The match finder's state, reusable across any number of blocks.
pub struct Lz4HcCompressor {
    /// The last position (plus `base`) of each 4-byte hash, or 0.
    hash_table: Box<[u32]>,
    /// Distance from a position to the previous one with the same hash.
    chain_table: Box<[u16]>,
    /// Added to a buffer offset to make a table position. Every position of
    /// the current call is above every position of an earlier call.
    base: usize,
    /// The first buffer offset not yet inserted into the tables.
    next_to_update: usize,
    max_attempts: usize,
}

impl Default for Lz4HcCompressor {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ATTEMPTS)
    }
}

/// A match: `len` bytes at buffer offset `start` repeat those at `reference`.
#[derive(Clone, Copy)]
struct Match {
    start: usize,
    reference: usize,
    len: usize,
}

#[inline]
fn read4(b: &[u8], i: usize) -> u32 {
    // ARITH: every caller reads below `match_limit + 4 <= b.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    let w: [u8; 4] = b[i..i + 4].try_into().unwrap();
    u32::from_le_bytes(w)
}

#[inline]
fn read2(b: &[u8], i: usize) -> Option<u16> {
    let w: [u8; 2] = b.get(i..i.checked_add(2)?)?.try_into().ok()?;
    Some(u16::from_le_bytes(w))
}

#[inline]
fn hash(v: u32) -> usize {
    // ARITH: a shift below 32.
    #[allow(clippy::arithmetic_side_effects)]
    let h = v.wrapping_mul(2_654_435_761) >> (32 - HASH_LOG);
    h as usize
}

/// How many bytes `b[a..]` and `b[r..]` share before `limit`, where `r < a`.
// ARITH: `r < a < limit <= b.len()`, so every index stays below `limit`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn count_forward(b: &[u8], mut a: usize, mut r: usize, limit: usize) -> usize {
    let start = a;
    while a + 8 <= limit {
        let x = u64::from_le_bytes(b[a..a + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[r..r + 8].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return a - start + (diff.trailing_zeros() / 8) as usize;
        }
        a += 8;
        r += 8;
    }
    while a < limit && b[a] == b[r] {
        a += 1;
        r += 1;
    }
    a - start
}

/// How many bytes before `a` and before `r` agree, going backwards no lower
/// than `a_low` on the `a` side and `r_low` on the `r` side.
// ARITH: the loop stops before either offset goes below its floor.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn count_back(b: &[u8], a: usize, r: usize, a_low: usize, r_low: usize) -> usize {
    let max = (a - a_low).min(r - r_low);
    let mut n = 0;
    while n < max && b[a - n - 1] == b[r - n - 1] {
        n += 1;
    }
    n
}

impl Lz4HcCompressor {
    /// A compressor that follows at most `max_attempts` chain links per
    /// search (`lz4hc`'s level 9 uses 256).
    pub fn new(max_attempts: usize) -> Self {
        Self {
            hash_table: vec![0u32; 1 << HASH_LOG].into_boxed_slice(),
            chain_table: vec![0u16; CHAIN_SIZE].into_boxed_slice(),
            base: 1,
            next_to_update: 0,
            max_attempts: max_attempts.max(1),
        }
    }

    /// Prepares the tables for a call over a buffer of `buffer_len` bytes
    /// whose window starts at buffer offset `window_start`.
    // ARITH: `base` stays below `REBASE_LIMIT + buffer_len + CHAIN_SIZE`, far
    // under `u32::MAX` for any buffer this format writes (at most 2 GiB).
    #[allow(clippy::arithmetic_side_effects)]
    fn start(&mut self, buffer_len: usize, window_start: usize) {
        if self.base > REBASE_LIMIT || buffer_len >= REBASE_LIMIT {
            self.hash_table.fill(0);
            self.chain_table.fill(0);
            self.base = 1;
        }
        self.next_to_update = window_start;
    }

    /// Ends a call: the next call's positions start past every one stored.
    // ARITH: see `start`.
    #[allow(clippy::arithmetic_side_effects)]
    fn finish(&mut self, buffer_len: usize) {
        self.base += buffer_len + CHAIN_SIZE;
    }

    /// Inserts every position from `next_to_update` up to (not including)
    /// `target`.
    // ARITH: positions are `base + offset`, kept below `u32::MAX` by
    // `start`; the chain delta is clamped to `MAX_DISTANCE` before the cast.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn insert(&mut self, b: &[u8], target: usize) {
        while self.next_to_update < target {
            let i = self.next_to_update;
            let pos = self.base + i;
            let h = hash(read4(b, i));
            let prev = self.hash_table[h] as usize;
            let delta = if prev == 0 || prev >= pos {
                MAX_DISTANCE
            } else {
                (pos - prev).min(MAX_DISTANCE)
            };
            self.chain_table[pos & CHAIN_MASK] = delta as u16;
            self.hash_table[h] = pos as u32;
            self.next_to_update += 1;
        }
    }

    /// `LZ4HC_InsertAndGetWiderMatch`: the longest match for the bytes at
    /// `ip` that may also extend backwards down to `low_limit`, if it is
    /// longer than `longest`. Its length counts from its (possibly earlier)
    /// start; it ends before `high_limit`. `window_start` is the lowest
    /// buffer offset a match may refer to.
    // ARITH: buffer offsets are below `b.len()`; table positions are
    // `base + offset` with `base >= 1`, so `pos - base` is an offset. The
    // walk stops as soon as a position is below `lowest`, and every length
    // is bounded by `high_limit - low_limit`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn wider_match(
        &mut self,
        b: &[u8],
        ip: usize,
        low_limit: usize,
        high_limit: usize,
        longest: usize,
        window_start: usize,
    ) -> Option<Match> {
        self.insert(b, ip);
        let look_back = ip - low_limit;
        let lowest = self.base + window_start.max(ip.saturating_sub(MAX_DISTANCE));
        let pattern = read4(b, ip);
        let mut best: Option<Match> = None;
        let mut longest = longest;
        let mut pos = self.hash_table[hash(pattern)] as usize;
        let mut attempts = self.max_attempts;
        while pos >= lowest && attempts > 0 {
            attempts -= 1;
            let r = pos - self.base;
            // A cheap filter first: the two bytes that would end a match one
            // longer than the best so far must agree.
            let probe = low_limit + longest - 1;
            let quick = match (r + longest).checked_sub(look_back + 1) {
                Some(rp) => read2(b, probe).is_some() && read2(b, probe) == read2(b, rp),
                None => true,
            };
            if quick && read4(b, r) == pattern {
                let forward = MIN_MATCH + count_forward(b, ip + MIN_MATCH, r + MIN_MATCH, high_limit);
                let back = if look_back > 0 {
                    count_back(b, ip, r, low_limit, window_start)
                } else {
                    0
                };
                let len = forward + back;
                if len > longest {
                    longest = len;
                    best = Some(Match {
                        start: ip - back,
                        reference: r - back,
                        len,
                    });
                }
            }
            let delta = self.chain_table[pos & CHAIN_MASK] as usize;
            if delta == 0 || delta > pos {
                break;
            }
            pos -= delta;
        }
        best
    }

    /// Compresses `bytes[dict_off + dict_len..][..len]` into one LZ4 block
    /// appended to `out`. Matches may refer back into the `dict_len` bytes
    /// before it, which are not themselves written.
    ///
    /// # Panics
    /// When `dict_off + dict_len + len` is past the end of `bytes`.
    // ARITH: `end = dict_off + dict_len + len <= bytes.len()` (checked by the
    // slice below). Inside the loop every offset is between `src` and
    // `match_limit`, every match ends at or before `match_limit`, and the
    // `OPTIMAL_ML` corrections only shorten a match while keeping it at
    // least `MIN_MATCH` long, as in `lz4hc.c`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn compress_with_dictionary(
        &mut self,
        bytes: &[u8],
        dict_off: usize,
        dict_len: usize,
        len: usize,
        out: &mut Vec<u8>,
    ) {
        let src = dict_off + dict_len;
        let end = src + len;
        let b = &bytes[..end];
        let mut anchor = src;
        if len < MF_LIMIT + 1 {
            encode_last_literals(b, anchor, end - anchor, out);
            return;
        }
        self.start(end, dict_off);
        let mf_limit = end - MF_LIMIT;
        let match_limit = end - LAST_LITERALS;
        // The dictionary's last 3 bytes start no 4-byte pattern of their own
        // inside it; `insert` picks them up once the block is reached.
        self.insert(b, src.saturating_sub(3).max(dict_off));

        let mut ip = src;
        let emit = |out: &mut Vec<u8>, anchor: &mut usize, m: Match| {
            encode_sequence(b, *anchor, m.reference, m.start, m.len, out);
            *anchor = m.start + m.len;
        };

        'outer: while ip <= mf_limit {
            let Some(mut m1) = self.wider_match(b, ip, ip, match_limit, MIN_MATCH - 1, dict_off)
            else {
                ip += 1;
                continue;
            };
            let mut m0 = m1;

            // `_Search2`
            'search2: loop {
                let m2 = if m1.start + m1.len <= mf_limit {
                    self.wider_match(
                        b,
                        m1.start + m1.len - 2,
                        m1.start,
                        match_limit,
                        m1.len,
                        dict_off,
                    )
                } else {
                    None
                };
                let Some(mut m2) = m2 else {
                    emit(out, &mut anchor, m1);
                    ip = anchor;
                    continue 'outer;
                };
                if m0.start < m1.start && m2.start < m1.start + m0.len {
                    // The first match was skipped forward; restore it.
                    m1 = m0;
                }
                if m2.start - m1.start < 3 {
                    // The first match is too short to keep: the second
                    // replaces it.
                    m1 = m2;
                    continue 'search2;
                }

                // `_Search3`
                loop {
                    if m2.start - m1.start < OPTIMAL_ML {
                        let mut new_ml = m1.len.min(OPTIMAL_ML);
                        if m1.start + new_ml > m2.start + m2.len - MIN_MATCH {
                            new_ml = m2.start - m1.start + m2.len - MIN_MATCH;
                        }
                        if new_ml > m2.start - m1.start {
                            let correction = new_ml - (m2.start - m1.start);
                            m2.start += correction;
                            m2.reference += correction;
                            m2.len -= correction;
                        }
                    }
                    let m3 = if m2.start + m2.len <= mf_limit {
                        self.wider_match(
                            b,
                            m2.start + m2.len - 3,
                            m2.start,
                            match_limit,
                            m2.len,
                            dict_off,
                        )
                    } else {
                        None
                    };
                    let Some(m3) = m3 else {
                        // No better third match: write the first two.
                        if m2.start < m1.start + m1.len {
                            m1.len = m2.start - m1.start;
                        }
                        emit(out, &mut anchor, m1);
                        emit(out, &mut anchor, m2);
                        ip = anchor;
                        continue 'outer;
                    };
                    if m3.start < m1.start + m1.len + 3 {
                        // No room for the second match: drop it.
                        if m3.start >= m1.start + m1.len {
                            // The first can be written now, and the third
                            // becomes the first.
                            if m2.start < m1.start + m1.len {
                                let correction = m1.start + m1.len - m2.start;
                                m2.start += correction;
                                m2.reference += correction;
                                m2.len = m2.len.saturating_sub(correction);
                                if m2.len < MIN_MATCH {
                                    m2 = m3;
                                }
                            }
                            emit(out, &mut anchor, m1);
                            m1 = m3;
                            m0 = m2;
                            continue 'search2;
                        }
                        m2 = m3;
                        continue;
                    }

                    // Three ascending matches: write the first, trimmed to
                    // end where the second starts.
                    if m2.start < m1.start + m1.len {
                        if m2.start - m1.start < OPTIMAL_ML {
                            if m1.len > OPTIMAL_ML {
                                m1.len = OPTIMAL_ML;
                            }
                            if m1.start + m1.len > m2.start + m2.len - MIN_MATCH {
                                m1.len = m2.start - m1.start + m2.len - MIN_MATCH;
                            }
                            if m1.len > m2.start - m1.start {
                                let correction = m1.len - (m2.start - m1.start);
                                m2.start += correction;
                                m2.reference += correction;
                                m2.len -= correction;
                            }
                        } else {
                            m1.len = m2.start - m1.start;
                        }
                    }
                    emit(out, &mut anchor, m1);
                    m1 = m2;
                    m2 = m3;
                }
            }
        }
        encode_last_literals(b, anchor, end - anchor, out);
        self.finish(end);
    }

    /// Compresses all of `bytes` into one LZ4 block with no dictionary.
    pub fn compress_into(&mut self, bytes: &[u8], out: &mut Vec<u8>) {
        self.compress_with_dictionary(bytes, 0, 0, bytes.len(), out);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::lz4;
    use lucene_store::data_input::SliceInput;

    fn corpus(n: usize, seed: u64) -> Vec<u8> {
        let words = [
            "lucene", "segment", "merge", "posting", "the", "of", "and", "{\"id\":", "\"body\":",
            "stored", "fields", "1234", "2026-11-07T", "GET /index.html", "HTTP/1.1", "200",
        ];
        let mut x = seed | 1;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x % 11 == 0 {
                out.push((x >> 24) as u8);
            } else {
                out.extend_from_slice(words[(x >> 33) as usize % words.len()].as_bytes());
                out.push(b' ');
            }
        }
        out.truncate(n);
        out
    }

    fn round_trip(c: &mut Lz4HcCompressor, bytes: &[u8], dict_len: usize) -> usize {
        let len = bytes.len() - dict_len;
        let mut out = Vec::new();
        c.compress_with_dictionary(bytes, 0, dict_len, len, &mut out);
        let mut dest = bytes[..dict_len].to_vec();
        dest.resize(bytes.len(), 0);
        let mut input = SliceInput::new(&out);
        assert_eq!(lz4::decompress(&mut input, len, &mut dest, dict_len).unwrap(), bytes.len());
        assert_eq!(dest, bytes);
        let mut dest2 = bytes[..dict_len].to_vec();
        dest2.resize(bytes.len(), 0);
        let mut input = SliceInput::new(&out);
        lz4::decompress_slice(&mut input, len, &mut dest2, dict_len).unwrap();
        assert_eq!(dest2, bytes);
        out.len()
    }

    #[test]
    fn round_trips_and_beats_the_fast_table() {
        let mut c = Lz4HcCompressor::default();
        for (n, seed) in [(0, 1), (5, 2), (13, 3), (100, 4), (8 << 10, 5), (200 << 10, 6)] {
            let bytes = corpus(n, seed);
            let hc = round_trip(&mut c, &bytes, 0);
            if n >= 8 << 10 {
                let fast = lz4::compress(&bytes).len();
                let mut lucene_hc = Vec::new();
                lz4::compress_into(&bytes, &mut lucene_hc, &mut lz4::HighCompressionHashTable::new());
                assert!(hc < fast, "{hc} vs fast {fast}");
                assert!(hc <= lucene_hc.len(), "{hc} vs Lucene HC {}", lucene_hc.len());
            }
        }
    }

    #[test]
    fn dictionary_is_used_and_tables_are_reused() {
        let mut c = Lz4HcCompressor::default();
        let bytes = corpus(40 << 10, 9);
        let mut with_dict = bytes[..4096].to_vec();
        with_dict.extend_from_slice(&bytes[..8192]);
        let dict = round_trip(&mut c, &with_dict, 4096);
        let plain = round_trip(&mut c, &bytes[..8192], 0);
        assert!(dict < plain / 2, "the block repeats its dictionary: {dict} vs {plain}");
        // Many calls in a row, with every shape, stay correct.
        for i in 0..300 {
            let n = (i * 977) % (20 << 10);
            let d = (i * 131) % 4096;
            let b = corpus(n + d, i as u64);
            round_trip(&mut c, &b, d.min(b.len()));
        }
    }

    #[test]
    fn runs_and_incompressible_bytes() {
        let mut c = Lz4HcCompressor::default();
        round_trip(&mut c, &vec![b'a'; 100_000], 0);
        let mut x = 7u64;
        let noise: Vec<u8> = (0..70_000)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                (x >> 56) as u8
            })
            .collect();
        round_trip(&mut c, &noise, 0);
        round_trip(&mut c, &noise, 60_000);
    }

    #[test]
    fn rebase_clears_the_tables() {
        let mut c = Lz4HcCompressor::default();
        c.base = REBASE_LIMIT + 1;
        let b = corpus(10_000, 3);
        round_trip(&mut c, &b, 0);
        assert!(c.base < REBASE_LIMIT);
    }
}
