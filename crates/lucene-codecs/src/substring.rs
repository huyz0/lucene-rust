//! The `RustSubstring` doc-values format's BINARY entries
//! ([`crate::doc_values::Lucene90DocValuesFormat::rust_substring`]): a
//! field whose values can be searched by substring, wildcard and regexp
//! without a term per value -- what Elasticsearch's and OpenSearch's
//! `wildcard` field does with a trigram postings field beside a doc-values
//! copy of every value, here inside the doc-values entry itself.
//!
//! ## Layout
//!
//! Documents are grouped in blocks of `2^block_shift` consecutive doc ids.
//! Each block's values are LZ4-compressed together (a header of doc offsets
//! and lengths, then the values end to end), and the block is indexed by
//! the trigrams its values contain: one posting per (trigram, block) rather
//! than per (trigram, document). Log lines in neighbouring documents repeat
//! most of their trigrams, so a 32-document block holds about a quarter of
//! the postings its documents would hold one by one.
//!
//! - `.dvm`, after the usual BINARY prefix (data region, docs with a value,
//!   value count) with [`MIN_LENGTH_MARKER`] for `minLength`: the block
//!   shift, the block count, the regions below and the block-address
//!   monotonic array's metadata ([`write_data`], [`read_meta`]).
//! - `.dvd`: the compressed blocks, their start addresses
//!   ([`crate::direct_monotonic`], one per block plus the end), the gram
//!   table (keys, block counts and postings offsets as fixed-width arrays,
//!   binary-searched in place) and the postings (block ids, as a bitset or
//!   as 128-delta frame-of-reference chunks, whichever is smaller).
//!
//! Grams are three symbols over the value's bytes with ASCII letters folded
//! to lower case, and a boundary symbol ([`crate::term_ngram::BOUNDARY`]) at
//! each end of a value -- `ab` is `^ab`, `ab$`; one index serves case
//! sensitive and insensitive searches alike, as the values settle every
//! candidate. A value with a character whose lower case is ASCII (`İ`,
//! the Kelvin sign) also indexes the grams of its lower-cased form, so a
//! case-insensitive search never misses it.
//!
//! ## Searching
//!
//! A [`SubstringMatcher`] holds the grams every match contains and the exact
//! test. [`SubstringIndex::candidate_blocks`] intersects the grams' postings
//! rarest first; each candidate block is decompressed once and its values
//! tested, a plain substring by one scan over the block's bytes.
//!
//! Stock Lucene has no format by this name and refuses the field rather
//! than misreading it.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::DataOutput;
use lucene_util::fx_hash::FxHashMap;

use crate::direct_monotonic;
use crate::doc_values::{Error, Result};
use crate::lz4;
use crate::regexp::{RegexpError, RegexpPattern};
use crate::term_ngram::{gram_key, BOUNDARY};
use crate::wildcard::WildcardPattern;

use std::sync::Arc;

use crate::automaton::{CompiledDfa, DEAD};

/// The `minLength` a BINARY entry of this format records, where Lucene90
/// writes the shortest value's length: below any length, it marks the
/// entry as blocks rather than one byte run.
pub const MIN_LENGTH_MARKER: i32 = -1001;
/// Documents per block, as a shift: 32.
pub const DEFAULT_BLOCK_SHIFT: u8 = 5;
/// The largest block shift a format accepts: 1024 documents per block.
pub const MAX_BLOCK_SHIFT: u8 = 10;

/// Postings entries decoded in the time a candidate block takes to read.
const DECODE_PER_BLOCK_READ: f64 = 700.0;
const ADDRESS_BLOCK_SHIFT: u32 = 16;
const POSTINGS_LIST: u8 = 0;
const POSTINGS_BITSET: u8 = 1;
const CHUNK: usize = 128;
/// Bytes of one gram-table row: key and block count (`u32`s) and the
/// postings offset (`u64`).
const GRAM_ROW: usize = 16;

/// A substring entry's `.dvm` metadata.
#[derive(Debug, Clone)]
pub struct SubstringEntry {
    pub block_shift: u8,
    pub num_blocks: i64,
    pub blocks_offset: i64,
    pub blocks_length: i64,
    pub addresses_offset: i64,
    pub addresses_length: i64,
    pub addresses_meta: direct_monotonic::Meta,
    pub grams_offset: i64,
    pub gram_count: i64,
    pub postings_offset: i64,
    pub postings_length: i64,
}

fn corrupt(what: &str) -> Error {
    Error::Store(lucene_store::Error::Corrupted(format!(
        "substring doc values: {what}"
    )))
}

// ---------------------------------------------------------------------------
// Grams
// ---------------------------------------------------------------------------

#[inline]
fn fold(b: u8) -> u32 {
    u32::from(b.to_ascii_lowercase())
}

/// Pushes the gram keys of `value` (folded, bounded) to `out`.
fn value_grams(value: &[u8], symbols: &mut Vec<u32>, out: &mut Vec<u32>) {
    symbols.clear();
    symbols.push(BOUNDARY);
    symbols.extend(value.iter().map(|&b| fold(b)));
    symbols.push(BOUNDARY);
    for w in symbols.windows(3) {
        out.push(gram_key(w));
    }
    // `İ` (U+0130) lower-cases to `i` + U+0307 and the Kelvin sign (U+212A)
    // to `k`: a case-insensitive search reads them as ASCII.
    if value.windows(2).any(|w| w == [0xC4, 0xB0])
        || value.windows(3).any(|w| w == [0xE2, 0x84, 0xAA])
    {
        let lowered = String::from_utf8_lossy(value).to_lowercase();
        symbols.clear();
        symbols.push(BOUNDARY);
        symbols.extend(lowered.bytes().map(fold));
        symbols.push(BOUNDARY);
        for w in symbols.windows(3) {
            out.push(gram_key(w));
        }
    }
}

/// The three symbols of a three-symbol gram key.
// ARITH: shifts and masks of a `u32`.
#[allow(clippy::arithmetic_side_effects)]
fn gram_symbols(key: u32) -> Option<[u32; 3]> {
    if key >> 30 != 2 {
        return None;
    }
    Some([(key >> 18) & 0x1FF, (key >> 9) & 0x1FF, key & 0x1FF])
}

/// The keys of the three-symbol windows of `run`; with `ascii_only`, only
/// windows whose symbols are ASCII or boundaries.
fn run_grams(run: &[u32], ascii_only: bool, out: &mut Vec<u32>) {
    for w in run.windows(3) {
        if ascii_only && w.iter().any(|&s| s != BOUNDARY && s >= 0x80) {
            continue;
        }
        out.push(gram_key(w));
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Postings of one gram while a field is written: the last block id and
/// the deltas since, as vints.
#[derive(Default)]
struct GramBuilder {
    last: Option<u32>,
    count: u32,
    deltas: Vec<u8>,
}

impl GramBuilder {
    // ARITH: block ids are pushed ascending and distinct, so `block - last
    // - 1` is non-negative; `count` is at most the block count, an `i32`
    // range value.
    #[allow(clippy::arithmetic_side_effects)]
    fn push(&mut self, block: u32) {
        let delta = match self.last {
            Some(last) => block - last - 1,
            None => block,
        };
        self.last = Some(block);
        self.count += 1;
        write_vu32(&mut self.deltas, delta);
    }
}

// ARITH: shifting a `u32` right by 7 until it is below 0x80.
#[allow(clippy::arithmetic_side_effects)]
fn write_vu32(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

// ARITH: `shift` stops at 28, so it stays below 32.
#[allow(clippy::arithmetic_side_effects)]
fn read_vu32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let b = *bytes.get(*pos)?;
        *pos += 1;
        v |= u32::from(b & 0x7F) << shift;
        if b < 0x80 {
            return Some(v);
        }
        shift += 7;
        if shift > 28 {
            return None;
        }
    }
}

/// Bits to hold `v`.
// ARITH: `leading_zeros` is at most 32.
#[allow(clippy::arithmetic_side_effects)]
fn bits(v: u32) -> u8 {
    (32 - v.leading_zeros()) as u8
}

/// Packs `values` in `width` bits each, least significant first.
// ARITH: `acc_bits` stays below 64: at most 7 bits are left over before
// each push of at most 32.
#[allow(clippy::arithmetic_side_effects)]
fn pack(values: &[u32], width: u8, out: &mut Vec<u8>) {
    if width == 0 {
        return;
    }
    let mut acc: u64 = 0;
    let mut acc_bits = 0u32;
    for &v in values {
        acc |= u64::from(v) << acc_bits;
        acc_bits += u32::from(width);
        while acc_bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        out.push(acc as u8);
    }
}

/// One gram's postings: a bitset over `num_blocks` or chunked deltas,
/// whichever is smaller.
// ARITH: counts and widths of in-memory buffers; `num_blocks` is at most
// `i32::MAX >> 0` blocks.
#[allow(clippy::arithmetic_side_effects)]
fn encode_postings(
    builder: &GramBuilder,
    num_blocks: u32,
    out: &mut Vec<u8>,
    scratch: &mut Vec<u32>,
) {
    scratch.clear();
    let mut pos = 0;
    while let Some(d) = read_vu32(&builder.deltas, &mut pos) {
        scratch.push(d);
    }
    let bitset_len = (num_blocks as usize).div_ceil(64) * 8;
    let start = out.len();
    out.push(POSTINGS_LIST);
    for chunk in scratch.chunks(CHUNK) {
        let width = chunk.iter().map(|&d| bits(d)).max().unwrap_or(0);
        out.push(width);
        pack(chunk, width, out);
    }
    if out.len() - start > 1 + bitset_len {
        out.truncate(start);
        out.push(POSTINGS_BITSET);
        let mut words = vec![0u64; bitset_len / 8];
        let mut block = 0u32;
        for (k, &d) in scratch.iter().enumerate() {
            block = if k == 0 { d } else { block + d + 1 };
            words[(block >> 6) as usize] |= 1u64 << (block & 63);
        }
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
}

/// Writes a substring field's data: blocks, block addresses, gram table and
/// postings, appended to `data`. `docs` ascend; `value(i)` is `docs[i]`'s
/// value. Returns the field's `.dvm` part, which follows the BINARY prefix
/// ([`read_meta`] reads it back).
// ARITH: offsets into in-memory buffers, doc ids below `max_doc` (an
// `i32`) shifted right, and per-block counts no larger than the block size.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_data<'v>(
    data: &mut Vec<u8>,
    docs: &[i32],
    value: impl Fn(usize) -> &'v [u8],
    max_doc: i32,
    block_shift: u8,
) -> Vec<u8> {
    let shift = u32::from(block_shift.min(MAX_BLOCK_SHIFT));
    let num_blocks = if max_doc <= 0 {
        0u32
    } else {
        ((max_doc as u32 - 1) >> shift) + 1
    };
    let mut grams: FxHashMap<u32, GramBuilder> = FxHashMap::default();
    let mut addresses: Vec<i64> = Vec::with_capacity(num_blocks as usize + 1);
    let blocks_offset = data.len() as i64;
    let mut raw: Vec<u8> = Vec::new();
    let mut header: Vec<u8> = Vec::new();
    let mut keys: Vec<u32> = Vec::new();
    let mut symbols: Vec<u32> = Vec::new();
    let mut ht = lz4::FastCompressionHashTable::new();
    let mut i = 0usize;
    for block in 0..num_blocks {
        addresses.push(data.len() as i64 - blocks_offset);
        let base = (block << shift) as i32;
        let end_doc = base.saturating_add(1 << shift);
        let first = i;
        while i < docs.len() && docs[i] < end_doc {
            i += 1;
        }
        if first == i {
            continue;
        }
        header.clear();
        raw.clear();
        keys.clear();
        write_vu32(&mut header, (i - first) as u32);
        for (k, &doc) in docs.iter().enumerate().take(i).skip(first) {
            let v = value(k);
            write_vu32(&mut header, (doc - base) as u32);
            write_vu32(&mut header, v.len() as u32);
            value_grams(v, &mut symbols, &mut keys);
        }
        raw.extend_from_slice(&header);
        for k in first..i {
            raw.extend_from_slice(value(k));
        }
        data.write_vint(raw.len() as i32);
        lz4::compress_into(&raw, data, &mut ht);
        keys.sort_unstable();
        keys.dedup();
        for &key in &keys {
            grams.entry(key).or_default().push(block);
        }
    }
    addresses.push(data.len() as i64 - blocks_offset);
    let blocks_length = data.len() as i64 - blocks_offset;

    let (addr_meta_bytes, addr_data) = direct_monotonic::write(&addresses, ADDRESS_BLOCK_SHIFT);
    let addresses_offset = data.len() as i64;
    data.extend_from_slice(&addr_data);
    let addresses_length = data.len() as i64 - addresses_offset;

    let mut sorted: Vec<(u32, GramBuilder)> = grams.into_iter().collect();
    sorted.sort_unstable_by_key(|(k, _)| *k);
    let mut postings: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(sorted.len());
    let mut scratch = Vec::new();
    for (_, b) in &sorted {
        offsets.push(postings.len() as u64);
        encode_postings(b, num_blocks, &mut postings, &mut scratch);
    }
    let grams_offset = data.len() as i64;
    for ((key, b), off) in sorted.iter().zip(&offsets) {
        data.extend_from_slice(&key.to_le_bytes());
        data.extend_from_slice(&b.count.to_le_bytes());
        data.extend_from_slice(&off.to_le_bytes());
    }
    let postings_offset = data.len() as i64;
    data.extend_from_slice(&postings);
    // Room for the 8-byte loads the chunk decoder takes.
    data.extend_from_slice(&[0u8; 8]);
    let mut meta = Vec::new();
    meta.push(shift as u8);
    meta.write_vlong(i64::from(num_blocks));
    meta.write_i64(blocks_offset);
    meta.write_i64(blocks_length);
    meta.write_i64(addresses_offset);
    meta.write_i64(addresses_length);
    meta.extend_from_slice(&addr_meta_bytes);
    meta.write_i64(grams_offset);
    meta.write_vlong(sorted.len() as i64);
    meta.write_i64(postings_offset);
    meta.write_i64(data.len() as i64 - postings_offset);
    meta
}

/// Reads the `.dvm` part [`write_data`] returned.
// ARITH: `num_blocks + 1` after `num_blocks` is checked against `i32::MAX`.
#[allow(clippy::arithmetic_side_effects)]
pub fn read_meta(input: &mut SliceInput) -> Result<SubstringEntry> {
    let block_shift = input.read_byte()?;
    if block_shift > MAX_BLOCK_SHIFT {
        return Err(corrupt("block shift out of range"));
    }
    let num_blocks = input.read_vlong()?;
    // Every block's first doc id is an `i32`.
    if !(0..=(i64::from(i32::MAX) >> block_shift) + 1).contains(&num_blocks) {
        return Err(corrupt("block count out of range"));
    }
    let blocks_offset = input.read_i64()?;
    let blocks_length = input.read_i64()?;
    let addresses_offset = input.read_i64()?;
    let addresses_length = input.read_i64()?;
    let addresses_meta = direct_monotonic::load_meta(input, num_blocks + 1, ADDRESS_BLOCK_SHIFT)?;
    let grams_offset = input.read_i64()?;
    let gram_count = input.read_vlong()?;
    if !(0..=1i64 << 27).contains(&gram_count) {
        return Err(corrupt("gram count out of range"));
    }
    let postings_offset = input.read_i64()?;
    let postings_length = input.read_i64()?;
    Ok(SubstringEntry {
        block_shift,
        num_blocks,
        blocks_offset,
        blocks_length,
        addresses_offset,
        addresses_length,
        addresses_meta,
        grams_offset,
        gram_count,
        postings_offset,
        postings_length,
    })
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

fn region(data: &[u8], offset: i64, length: i64) -> Result<&[u8]> {
    offset
        .checked_add(length)
        .and_then(|end| Some(usize::try_from(offset).ok()?..usize::try_from(end).ok()?))
        .and_then(|r| data.get(r))
        .ok_or_else(|| corrupt("region outside the data file"))
}

/// One decompressed block: its documents and their values.
#[derive(Debug, Default, Clone)]
pub struct Block {
    id: i64,
    docs: Vec<i32>,
    /// Value `k` is `raw[ends[k] - len_k .. ends[k]]`; `starts[k]` is its
    /// start.
    starts: Vec<usize>,
    ends: Vec<usize>,
    values_start: usize,
    raw: Vec<u8>,
}

impl Block {
    /// An empty block buffer.
    pub fn new() -> Self {
        Block {
            id: -1,
            ..Block::default()
        }
    }

    /// The block's documents with a value, ascending.
    pub fn docs(&self) -> &[i32] {
        &self.docs
    }

    /// The `k`th document's value.
    pub fn value(&self, k: usize) -> &[u8] {
        &self.raw[self.starts[k]..self.ends[k]]
    }

    /// `doc`'s value, when the block holds one.
    pub fn value_of(&self, doc: i32) -> Option<&[u8]> {
        let k = self.docs.binary_search(&doc).ok()?;
        Some(self.value(k))
    }

    /// The values end to end.
    fn values(&self) -> &[u8] {
        &self.raw[self.values_start..]
    }
}

/// A substring entry over its segment's `.dvd` bytes.
#[derive(Debug)]
pub struct SubstringIndex<'a> {
    entry: &'a SubstringEntry,
    blocks: &'a [u8],
    addresses: direct_monotonic::Reader<'a>,
    grams: &'a [u8],
    postings: &'a [u8],
}

/// Candidate blocks: every block with a value, or the listed ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Candidates {
    All,
    Blocks(Vec<u32>),
}

impl<'a> SubstringIndex<'a> {
    /// Opens `entry` over `data`, the whole `.dvd` file.
    // ARITH: `gram_count` is at most 2^27 (checked by `read_meta`), times
    // a 16-byte row.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(data: &'a [u8], entry: &'a SubstringEntry) -> Result<Self> {
        let blocks = region(data, entry.blocks_offset, entry.blocks_length)?;
        let addr = region(data, entry.addresses_offset, entry.addresses_length)?;
        let grams = region(data, entry.grams_offset, entry.gram_count * GRAM_ROW as i64)?;
        let postings = region(data, entry.postings_offset, entry.postings_length)?;
        Ok(SubstringIndex {
            entry,
            blocks,
            addresses: direct_monotonic::Reader::new(addr, &entry.addresses_meta),
            grams,
            postings,
        })
    }

    /// Documents per block, as a shift.
    pub fn block_shift(&self) -> u8 {
        self.entry.block_shift
    }

    /// The number of blocks.
    pub fn num_blocks(&self) -> i64 {
        self.entry.num_blocks
    }

    /// The number of distinct grams.
    pub fn gram_count(&self) -> i64 {
        self.entry.gram_count
    }

    // ARITH: `i < gram_count`, so the row lies inside `grams`, whose length
    // `new` checked.
    #[allow(clippy::arithmetic_side_effects)]
    fn row(&self, i: usize) -> (u32, u32, u64) {
        let r = &self.grams[i * GRAM_ROW..(i + 1) * GRAM_ROW];
        (
            u32::from_le_bytes([r[0], r[1], r[2], r[3]]),
            u32::from_le_bytes([r[4], r[5], r[6], r[7]]),
            u64::from_le_bytes([r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]]),
        )
    }

    /// The gram table row of `key`: its index and block count.
    // ARITH: binary search bounds within `0..=gram_count`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn find_gram(&self, key: u32) -> Option<(usize, u32)> {
        let (mut lo, mut hi) = (0usize, self.entry.gram_count as usize);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (k, df, _) = self.row(mid);
            match k.cmp(&key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some((mid, df)),
            }
        }
        None
    }

    fn postings_bytes(&self, i: usize) -> Result<&'a [u8]> {
        let (_, _, off) = self.row(i);
        usize::try_from(off)
            .ok()
            .and_then(|o| self.postings.get(o..))
            .ok_or_else(|| corrupt("postings offset outside the postings"))
    }

    /// Decodes gram `i`'s block ids, ascending, into `out`.
    // ARITH: chunk widths are checked against 32; ids grow by `delta + 1`
    // with wrapping, and a decoded id past the block count is corruption.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn decode(&self, i: usize, out: &mut Vec<u32>) -> Result<()> {
        out.clear();
        let (_, df, _) = self.row(i);
        let bytes = self.postings_bytes(i)?;
        let Some((&tag, body)) = bytes.split_first() else {
            return Err(corrupt("empty postings"));
        };
        let nb = self.entry.num_blocks as u32;
        if tag == POSTINGS_BITSET {
            let words = (nb as usize).div_ceil(64);
            let body = body
                .get(..words * 8)
                .ok_or_else(|| corrupt("short bitset"))?;
            for (w, chunk) in body.chunks_exact(8).enumerate() {
                let mut word = u64::from_le_bytes(chunk.try_into().unwrap_or_default());
                while word != 0 {
                    out.push((w as u32) << 6 | word.trailing_zeros());
                    word &= word - 1;
                }
            }
            if out.last().is_some_and(|&b| b >= nb) {
                return Err(corrupt("block id past the block count"));
            }
            return Ok(());
        }
        if tag != POSTINGS_LIST {
            return Err(corrupt("unknown postings encoding"));
        }
        let mut pos = 0usize;
        let mut prev: i64 = -1;
        let mut left = df as usize;
        while left > 0 {
            let n = left.min(CHUNK);
            let width = u32::from(*body.get(pos).ok_or_else(|| corrupt("short postings"))?);
            pos += 1;
            if width > 32 {
                return Err(corrupt("chunk width above 32"));
            }
            let len = (n * width as usize).div_ceil(8);
            let packed = body
                .get(pos..pos + len)
                .ok_or_else(|| corrupt("short postings"))?;
            pos += len;
            let mask: u64 = if width == 32 {
                u64::from(u32::MAX)
            } else {
                (1u64 << width) - 1
            };
            let mut bit = 0usize;
            for _ in 0..n {
                let d = if width == 0 {
                    0
                } else {
                    let byte = bit >> 3;
                    let mut word = [0u8; 8];
                    let avail = packed.len().saturating_sub(byte).min(8);
                    word[..avail].copy_from_slice(&packed[byte..byte + avail]);
                    ((u64::from_le_bytes(word) >> (bit & 7)) & mask) as u32
                };
                bit += width as usize;
                prev += i64::from(d) + 1;
                // Ascending by construction; past `u32` is corruption.
                out.push(u32::try_from(prev).map_err(|_| corrupt("block id overflows"))?);
            }
            left -= n;
        }
        if out.last().is_some_and(|&b| b >= nb) {
            return Err(corrupt("block id past the block count"));
        }
        Ok(())
    }

    /// Whether gram `i`'s postings are a bitset, and the bitset's words.
    // ARITH: `words` is the block count over 64, far below `usize::MAX / 8`.
    #[allow(clippy::arithmetic_side_effects)]
    fn bitset(&self, i: usize) -> Result<Option<&'a [u8]>> {
        let bytes = self.postings_bytes(i)?;
        if bytes.first() == Some(&POSTINGS_BITSET) {
            let words = (self.entry.num_blocks as usize).div_ceil(64);
            return bytes
                .get(1..1 + words * 8)
                .map(Some)
                .ok_or_else(|| corrupt("short bitset"));
        }
        Ok(None)
    }

    /// The blocks holding every gram of `grams`: [`Candidates::All`] when
    /// `grams` is empty, no blocks when one gram is absent. Rarest gram
    /// first; a gram whose postings cost more to decode than reading the
    /// candidates it would rule out is not decoded.
    // ARITH: candidate counts below the block count.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn candidate_blocks(&self, grams: &[u32]) -> Result<Candidates> {
        if grams.is_empty() {
            return Ok(Candidates::All);
        }
        let mut rows = Vec::with_capacity(grams.len());
        for &g in grams {
            match self.find_gram(g) {
                Some(row) => rows.push(row),
                None => return Ok(Candidates::Blocks(Vec::new())),
            }
        }
        rows.sort_unstable_by_key(|&(_, df)| df);
        rows.dedup_by_key(|r| r.0);
        let mut cands = Vec::new();
        self.decode(rows[0].0, &mut cands)?;
        let mut other = Vec::new();
        for &(i, df) in &rows[1..] {
            if cands.is_empty() {
                break;
            }
            // Decoding costs about 1/700 of reading a block per entry, and
            // a gram in a share `p` of the blocks leaves `p` of the
            // candidates.
            let p = f64::from(df) / self.entry.num_blocks.max(1) as f64;
            if f64::from(df) > cands.len() as f64 * DECODE_PER_BLOCK_READ * (1.0 - p) {
                break;
            }
            if let Some(words) = self.bitset(i)? {
                cands.retain(|&b| {
                    let w = (b >> 6) as usize * 8;
                    words.get(w..w + 8).is_some_and(|c| {
                        u64::from_le_bytes(c.try_into().unwrap_or_default()) >> (b & 63) & 1 == 1
                    })
                });
                continue;
            }
            self.decode(i, &mut other)?;
            intersect(&mut cands, &other);
        }
        Ok(Candidates::Blocks(cands))
    }

    /// Decompresses block `block` into `buf` (kept when it already holds
    /// it).
    // ARITH: header counts and lengths are checked against the
    // decompressed block's length before use.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_block(&self, block: i64, buf: &mut Block) -> Result<()> {
        if buf.id == block {
            return Ok(());
        }
        buf.id = -1;
        buf.docs.clear();
        buf.starts.clear();
        buf.ends.clear();
        buf.raw.clear();
        buf.values_start = 0;
        if block < 0 || block >= self.entry.num_blocks {
            return Err(corrupt("block out of range"));
        }
        let start = self.addresses.get(block)?;
        let end = self.addresses.get(block + 1)?;
        if end < start {
            return Err(corrupt("block addresses not ascending"));
        }
        if end > start {
            let bytes = region(self.blocks, start, end - start)?;
            let mut input = SliceInput::new(bytes);
            let len = input.read_vint()?;
            let len = usize::try_from(len).map_err(|_| corrupt("negative block length"))?;
            // LZ4 expands a byte to at most 255.
            if len > bytes.len().saturating_mul(255).saturating_add(16) {
                return Err(corrupt("block length past what its bytes can hold"));
            }
            buf.raw.resize(len, 0);
            lz4::decompress_slice(&mut input, len, &mut buf.raw, 0)?;
            let raw = &buf.raw;
            let mut pos = 0usize;
            let n = read_vu32(raw, &mut pos).ok_or_else(|| corrupt("block header"))? as usize;
            if n > (1usize << self.entry.block_shift) {
                return Err(corrupt("block holds more documents than its size"));
            }
            let base = (block << self.entry.block_shift) as i32;
            let mut lens = Vec::with_capacity(n);
            for _ in 0..n {
                let d = read_vu32(raw, &mut pos).ok_or_else(|| corrupt("block header"))?;
                let l = read_vu32(raw, &mut pos).ok_or_else(|| corrupt("block header"))?;
                if d >= 1 << self.entry.block_shift {
                    return Err(corrupt("doc offset past the block"));
                }
                buf.docs.push(base + d as i32);
                lens.push(l as usize);
            }
            buf.values_start = pos;
            for l in lens {
                buf.starts.push(pos);
                pos = pos
                    .checked_add(l)
                    .filter(|&p| p <= raw.len())
                    .ok_or_else(|| corrupt("value past the block"))?;
                buf.ends.push(pos);
            }
        }
        buf.id = block;
        Ok(())
    }

    /// `doc`'s value, read into `buf`'s block.
    // ARITH: `doc >= 0`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn value<'b>(&self, doc: i32, buf: &'b mut Block) -> Result<Option<&'b [u8]>> {
        if doc < 0 {
            return Ok(None);
        }
        self.read_block(i64::from(doc) >> self.entry.block_shift, buf)?;
        Ok(buf.value_of(doc))
    }
}

/// `a` ∩ `b`, both ascending, into `a`.
// ARITH: indexes below the slices' lengths.
#[allow(clippy::arithmetic_side_effects)]
fn intersect(a: &mut Vec<u32>, b: &[u32]) {
    let mut j = 0usize;
    a.retain(|&x| {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        j < b.len() && b[j] == x
    });
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Test {
    /// Any value.
    Any,
    Exact(Vec<u8>),
    Prefix(Vec<u8>),
    Suffix(Vec<u8>),
    Contains(Box<memchr::memmem::Finder<'static>>),
    Wildcard(Box<Glob>),
    /// The pattern, and its automaton when it is small enough to build.
    Regexp(Box<RegexpPattern>, Option<Arc<CompiledDfa>>),
}

/// What a search on a substring field asks: the grams every match holds
/// and the exact test of a value.
#[derive(Debug, Clone)]
pub struct SubstringMatcher {
    grams: Vec<u32>,
    test: Test,
    /// OpenSearch's `case_insensitive` for a wildcard: the value is lower
    /// cased (`toLowerCase(Locale.ROOT)`) before the test, whose pattern
    /// already is.
    lowercase: bool,
    /// A byte run every matching value holds, looked for over a whole
    /// block before the test runs on the values holding it (ASCII lower
    /// cased on both sides when `fold_prefilter`).
    prefilter: Option<memchr::memmem::Finder<'static>>,
    fold_prefilter: bool,
}

/// Whether `dfa` accepts the whole of `v`, stopping at a dead state or at
/// one accepting every well-formed rest (`.*` after the needle).
fn run_dfa(c: &CompiledDfa, v: &[u8]) -> bool {
    let dfa = &c.dfa;
    let mut state = dfa.start();
    for (i, &b) in v.iter().enumerate() {
        if state == DEAD {
            return false;
        }
        if c.utf8_total.get(state as usize) == Some(&true) {
            return std::str::from_utf8(v.get(i..).unwrap_or_default()).is_ok();
        }
        state = dfa.step(state, b);
    }
    dfa.is_accept(state)
}

/// A wildcard as its `*`-separated segments of bytes and `?`s, matched
/// leftmost first without backtracking: a `?` is one byte in an ASCII
/// value. Other values (a `?` may be several bytes) take the backtracking
/// [`WildcardPattern`].
#[derive(Debug, Clone)]
struct Glob {
    /// `None` is `?`.
    head: Vec<Option<u8>>,
    middle: Vec<Segment>,
    /// `None` without a `*`: the head is the whole pattern.
    tail: Option<Vec<Option<u8>>>,
    has_one: bool,
    slow: WildcardPattern,
}

#[derive(Debug, Clone)]
struct Segment {
    bytes: Vec<Option<u8>>,
    /// The segment's longest literal run and its offset in the segment.
    run: Option<(usize, memchr::memmem::Finder<'static>)>,
}

fn segment_at(seg: &[Option<u8>], v: &[u8], at: usize) -> bool {
    v.get(at..)
        .and_then(|v| v.get(..seg.len()))
        .is_some_and(|w| seg.iter().zip(w).all(|(p, b)| p.is_none_or(|p| p == *b)))
}

impl Glob {
    // ARITH: `start <= i` and `s <= e` are kept by the run scan.
    #[allow(clippy::arithmetic_side_effects)]
    fn new(toks: &[Tok], pattern: &[u8]) -> Self {
        let mut segments: Vec<Vec<Option<u8>>> = vec![Vec::new()];
        let mut has_one = false;
        for t in toks {
            match t {
                Tok::Lit(l) => segments
                    .last_mut()
                    .into_iter()
                    .for_each(|s| s.extend(l.iter().map(|&b| Some(b)))),
                Tok::One => {
                    has_one = true;
                    segments.last_mut().into_iter().for_each(|s| s.push(None));
                }
                Tok::Star => segments.push(Vec::new()),
            }
        }
        let head = segments.remove(0);
        let tail = segments.pop();
        let middle = segments
            .into_iter()
            .map(|bytes| {
                let mut best: Option<(usize, usize)> = None;
                let mut start = 0usize;
                for (i, b) in bytes.iter().chain([&None]).enumerate() {
                    if b.is_none() {
                        if i > start && best.is_none_or(|(s, e)| e - s < i - start) {
                            best = Some((start, i));
                        }
                        start = i + 1;
                    }
                }
                let run = best.map(|(s, e)| {
                    let lit: Vec<u8> = bytes[s..e].iter().flatten().copied().collect();
                    (s, memchr::memmem::Finder::new(&lit).into_owned())
                });
                Segment { bytes, run }
            })
            .collect();
        Glob {
            head,
            middle,
            tail,
            has_one,
            slow: WildcardPattern::new(pattern),
        }
    }

    // ARITH: positions stay within `v`, checked before each subtraction.
    #[allow(clippy::arithmetic_side_effects)]
    fn matches(&self, v: &[u8]) -> bool {
        if self.has_one && !v.is_ascii() {
            return self.slow.matches(v);
        }
        let Some(tail) = &self.tail else {
            return v.len() == self.head.len() && segment_at(&self.head, v, 0);
        };
        if v.len() < self.head.len() + tail.len()
            || !segment_at(&self.head, v, 0)
            || !segment_at(tail, v, v.len() - tail.len())
        {
            return false;
        }
        let end = v.len() - tail.len();
        let mut pos = self.head.len();
        for seg in &self.middle {
            let Some(at) = Self::find(seg, &v[..end], pos) else {
                return false;
            };
            pos = at + seg.bytes.len();
        }
        true
    }

    /// The leftmost start at or after `from` where `seg` matches in `v`.
    // ARITH: `off <= seg.bytes.len()`, and a hit lies inside `v`.
    #[allow(clippy::arithmetic_side_effects)]
    fn find(seg: &Segment, v: &[u8], from: usize) -> Option<usize> {
        let len = seg.bytes.len();
        if from + len > v.len() {
            return None;
        }
        let Some((off, finder)) = &seg.run else {
            return Some(from);
        };
        let mut at = from + off;
        while let Some(hit) = finder.find(v.get(at..)?) {
            let start = at + hit - off;
            if start + len > v.len() {
                return None;
            }
            if segment_at(&seg.bytes, v, start) {
                return Some(start);
            }
            at += hit + 1;
        }
        None
    }
}

enum Tok {
    Lit(Vec<u8>),
    Star,
    One,
}

/// `WildcardQuery`'s syntax: `*`, `?`, and `\` escaping the next character.
fn wildcard_tokens(pattern: &str) -> Vec<Tok> {
    let mut toks: Vec<Tok> = Vec::new();
    let mut chars = pattern.chars();
    let mut lit: Vec<u8> = Vec::new();
    let mut buf = [0u8; 4];
    while let Some(c) = chars.next() {
        match c {
            '*' | '?' => {
                if !lit.is_empty() {
                    toks.push(Tok::Lit(std::mem::take(&mut lit)));
                }
                if c == '?' {
                    toks.push(Tok::One);
                } else if !matches!(toks.last(), Some(Tok::Star)) {
                    toks.push(Tok::Star);
                }
            }
            '\\' => match chars.next() {
                Some(n) => lit.extend_from_slice(n.encode_utf8(&mut buf).as_bytes()),
                None => lit.push(b'\\'),
            },
            _ => lit.extend_from_slice(c.encode_utf8(&mut buf).as_bytes()),
        }
    }
    if !lit.is_empty() {
        toks.push(Tok::Lit(lit));
    }
    toks
}

/// `WildcardQuery`'s escape of every special character in `s`.
pub fn escape_wildcard(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '*' | '?' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn lowercase_bytes(v: &[u8], scratch: &mut Vec<u8>) {
    scratch.clear();
    if v.is_ascii() {
        scratch.extend(v.iter().map(u8::to_ascii_lowercase));
    } else {
        scratch.extend_from_slice(String::from_utf8_lossy(v).to_lowercase().as_bytes());
    }
}

impl SubstringMatcher {
    /// A `WildcardQuery` pattern over the whole value; with
    /// `case_insensitive`, OpenSearch's lower-cased pattern against the
    /// lower-cased value.
    // ARITH: token indexes below the token count.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn wildcard(pattern: &str, case_insensitive: bool) -> Self {
        let pattern = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };
        let toks = wildcard_tokens(&pattern);
        let mut grams = Vec::new();
        let n = toks.len();
        for (k, t) in toks.iter().enumerate() {
            if let Tok::Lit(bytes) = t {
                let mut run: Vec<u32> = Vec::with_capacity(bytes.len() + 2);
                if k == 0 {
                    run.push(BOUNDARY);
                }
                run.extend(bytes.iter().map(|&b| fold(b)));
                if k + 1 == n {
                    run.push(BOUNDARY);
                }
                run_grams(&run, case_insensitive, &mut grams);
            }
        }
        grams.sort_unstable();
        grams.dedup();
        let test = match toks.as_slice() {
            [] => Test::Exact(Vec::new()),
            [Tok::Star] => Test::Any,
            [Tok::Lit(l)] => Test::Exact(l.clone()),
            [Tok::Lit(l), Tok::Star] => Test::Prefix(l.clone()),
            [Tok::Star, Tok::Lit(l)] => Test::Suffix(l.clone()),
            [Tok::Star, Tok::Lit(l), Tok::Star] => {
                Test::Contains(Box::new(memchr::memmem::Finder::new(l).into_owned()))
            }
            _ => Test::Wildcard(Box::new(Glob::new(&toks, pattern.as_bytes()))),
        };
        let prefilter = match test {
            Test::Wildcard(_) => toks
                .iter()
                .filter_map(|t| match t {
                    Tok::Lit(l) => Some(l),
                    _ => None,
                })
                .max_by_key(|l| l.len())
                .map(|l| memchr::memmem::Finder::new(l).into_owned()),
            _ => None,
        };
        SubstringMatcher {
            grams,
            test,
            lowercase: case_insensitive,
            prefilter,
            fold_prefilter: false,
        }
    }

    /// Values holding `needle` anywhere: the wildcard `*needle*`.
    pub fn contains(needle: &str, case_insensitive: bool) -> Self {
        Self::wildcard(&format!("*{}*", escape_wildcard(needle)), case_insensitive)
    }

    /// Values equal to `value`: OpenSearch's `term` on a `wildcard` field.
    pub fn exact(value: &str, case_insensitive: bool) -> Self {
        Self::wildcard(&escape_wildcard(value), case_insensitive)
    }

    /// A Lucene `RegExp` over the whole value; with `case_insensitive`,
    /// `RegExp.ASCII_CASE_INSENSITIVE`, which Lucene 10.5 folds like
    /// `CASE_INSENSITIVE` (see [`RegexpPattern::ascii_case_insensitive`]).
    pub fn regexp(pattern: &str, case_insensitive: bool) -> std::result::Result<Self, RegexpError> {
        let parsed = RegexpPattern::parse(pattern)?;
        // Case insensitive, a non-ASCII byte or an `s` (long s, U+017F, is
        // its equivalent and indexed as itself) may stand for another
        // character in the value: such grams are not required.
        let free = |x: u32| {
            !case_insensitive || x == BOUNDARY || (x < 0x80 && fold(x as u8) != u32::from(b's'))
        };
        let mut grams: Vec<u32> = parsed
            .required_grams()
            .into_iter()
            .filter_map(gram_symbols)
            .filter(|s| s.iter().all(|&x| free(x)))
            .map(|s| {
                let f = |x: u32| if x < 0x80 { fold(x as u8) } else { x };
                gram_key(&[f(s[0]), f(s[1]), f(s[2])])
            })
            .collect();
        grams.sort_unstable();
        grams.dedup();
        let mut literal = parsed.required_literal();
        if case_insensitive {
            // The longest piece an ASCII-lower-cased value holds as is: no
            // non-ASCII byte, no `k` (the Kelvin sign) and no `s`.
            literal.make_ascii_lowercase();
            literal = literal
                .split(|&b| b >= 0x80 || b == b'k' || b == b's')
                .max_by_key(|p| p.len())
                .unwrap_or_default()
                .to_vec();
        }
        let prefilter =
            (!literal.is_empty()).then(|| memchr::memmem::Finder::new(&literal).into_owned());
        let compiled = if case_insensitive {
            parsed.ascii_case_insensitive()
        } else {
            parsed
        };
        Ok(SubstringMatcher {
            grams,
            test: {
                let dfa = compiled.compiled();
                Test::Regexp(Box::new(compiled), dfa)
            },
            lowercase: false,
            prefilter,
            fold_prefilter: case_insensitive,
        })
    }

    /// The gram keys every matching value holds.
    pub fn grams(&self) -> &[u32] {
        &self.grams
    }

    /// Whether every value matches.
    pub fn matches_any_value(&self) -> bool {
        matches!(self.test, Test::Any)
    }

    /// Whether `value` matches. `scratch` holds a lower-cased copy.
    pub fn matches(&self, value: &[u8], scratch: &mut Vec<u8>) -> bool {
        if self.lowercase {
            lowercase_bytes(value, scratch);
            self.test(scratch)
        } else {
            self.test(value)
        }
    }

    /// The test of a value, already lower cased for a case-insensitive
    /// wildcard.
    fn test(&self, v: &[u8]) -> bool {
        match &self.test {
            Test::Any => true,
            Test::Exact(l) => v == &l[..],
            Test::Prefix(l) => v.starts_with(l),
            Test::Suffix(l) => v.ends_with(l),
            Test::Contains(f) => f.find(v).is_some(),
            Test::Wildcard(w) => w.matches(v),
            Test::Regexp(_, Some(dfa)) => run_dfa(dfa, v),
            Test::Regexp(r, None) => r.matches(v),
        }
    }

    /// Appends the documents of `block` whose value matches to `out`: the
    /// block's values searched end to end for the needle (or prefilter),
    /// and only the values holding it tested.
    // ARITH: positions inside the block's values, whose ends ascend.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn collect(&self, block: &Block, out: &mut Vec<i32>, scratch: &mut Vec<u8>) {
        let (finder, verify) = match (&self.test, &self.prefilter) {
            (Test::Contains(f), _) => (&**f, false),
            (_, Some(f)) => (f, true),
            _ => return self.collect_each(block, out, scratch),
        };
        let values = block.values();
        let hay: &[u8] = if self.lowercase || self.fold_prefilter {
            // A case-insensitive wildcard lower cases values beyond ASCII.
            if self.lowercase && !values.is_ascii() {
                return self.collect_each(block, out, scratch);
            }
            scratch.clear();
            scratch.extend(values.iter().map(u8::to_ascii_lowercase));
            scratch
        } else {
            values
        };
        // The value a wildcard tests is lower cased; a regexp's is not.
        let tested = if self.lowercase { hay } else { values };
        let base = block.values_start;
        let needle = finder.needle().len();
        let mut pos = 0usize;
        let mut k = 0usize;
        while k < block.docs.len() {
            let Some(hit) = finder.find(&hay[pos..]) else {
                break;
            };
            let at = pos + hit;
            while k < block.docs.len()
                && block.ends[k] - base <= at
                && !(needle == 0 && block.ends[k] - base == at)
            {
                k += 1;
            }
            if k == block.docs.len() {
                break;
            }
            let (s, e) = (block.starts[k] - base, block.ends[k] - base);
            // A hit straddling into the next value is looked for again
            // from there: any later one in this value straddles too.
            if at >= s && at + needle <= e && (!verify || self.test(&tested[s..e])) {
                out.push(block.docs[k]);
            }
            pos = e;
            k += 1;
        }
    }

    fn collect_each(&self, block: &Block, out: &mut Vec<i32>, scratch: &mut Vec<u8>) {
        for (k, &doc) in block.docs.iter().enumerate() {
            if self.matches(block.value(k), scratch) {
                out.push(doc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn build(values: &[Option<&str>], shift: u8) -> (Vec<u8>, SubstringEntry) {
        let docs: Vec<i32> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(d, _)| d as i32)
            .collect();
        let present: Vec<&str> = values.iter().flatten().copied().collect();
        let mut data = vec![0u8; 3];
        let meta = write_data(
            &mut data,
            &docs,
            |i| present[i].as_bytes(),
            values.len() as i32,
            shift,
        );
        let back = read_meta(&mut SliceInput::new(&meta)).unwrap();
        (data, back)
    }

    fn search(values: &[Option<&str>], shift: u8, m: &SubstringMatcher) -> Vec<i32> {
        let (data, entry) = build(values, shift);
        let index = SubstringIndex::new(&data, &entry).unwrap();
        let blocks: Vec<u32> = match index.candidate_blocks(m.grams()).unwrap() {
            Candidates::All => (0..index.num_blocks() as u32).collect(),
            Candidates::Blocks(b) => b,
        };
        let mut out = Vec::new();
        let mut buf = Block::new();
        let mut scratch = Vec::new();
        for b in blocks {
            index.read_block(i64::from(b), &mut buf).unwrap();
            m.collect(&buf, &mut out, &mut scratch);
        }
        out
    }

    fn brute(values: &[Option<&str>], m: &SubstringMatcher) -> Vec<i32> {
        let mut scratch = Vec::new();
        values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some_and(|v| m.matches(v.as_bytes(), &mut scratch)))
            .map(|(d, _)| d as i32)
            .collect()
    }

    const VALUES: &[Option<&str>] = &[
        Some("Connection timeout on 10.0.0.1"),
        None,
        Some("java.net.SocketTimeoutException: read timed out"),
        Some("ok"),
        Some(""),
        Some("ERROR disk full"),
        Some("error: retry"),
        None,
        Some("abcabcabc"),
        Some("İstanbul KELVIN \u{212A}"),
        Some("tail ab"),
        Some("ab"),
        Some("x"),
    ];

    #[test]
    fn every_pattern_finds_what_a_scan_finds() {
        let patterns: Vec<SubstringMatcher> = vec![
            SubstringMatcher::contains("timeout", false),
            SubstringMatcher::contains("timeout", true),
            SubstringMatcher::contains("Timeout", false),
            SubstringMatcher::contains("error", false),
            SubstringMatcher::contains("error", true),
            SubstringMatcher::contains("ab", false),
            SubstringMatcher::contains("", false),
            SubstringMatcher::contains("istanbul", true),
            SubstringMatcher::contains("kelvin k", true),
            SubstringMatcher::contains("cabca", false),
            SubstringMatcher::exact("ok", false),
            SubstringMatcher::exact("OK", true),
            SubstringMatcher::exact("", false),
            SubstringMatcher::wildcard("*", false),
            SubstringMatcher::wildcard("ab*", false),
            SubstringMatcher::wildcard("*ab", false),
            SubstringMatcher::wildcard("?b", false),
            SubstringMatcher::wildcard("e*r*", true),
            SubstringMatcher::wildcard("*time?out*", false),
            SubstringMatcher::wildcard("*10.0.*.1", false),
            SubstringMatcher::regexp(".*[Tt]imeout.*", false).unwrap(),
            SubstringMatcher::regexp("error.*", true).unwrap(),
            SubstringMatcher::regexp("(ok|x)", false).unwrap(),
            SubstringMatcher::regexp(".*abc(abc)+", false).unwrap(),
            SubstringMatcher::regexp(".*TIME(d|out).*", true).unwrap(),
            SubstringMatcher::regexp(".*ab(c|x)a.*", false).unwrap(),
            SubstringMatcher::wildcard("*a?c*a*", false),
            SubstringMatcher::wildcard("*KELVIN*", true),
            SubstringMatcher::wildcard("*?l?i*", false),
        ];
        for shift in [0u8, 1, 2, 5] {
            for m in &patterns {
                assert_eq!(
                    search(VALUES, shift, m),
                    brute(VALUES, m),
                    "{m:?} shift {shift}"
                );
            }
        }
    }

    /// The leftmost-first glob and the backtracking matcher agree on every
    /// pattern over a small alphabet, ASCII and not.
    #[test]
    fn glob_agrees_with_backtracking() {
        let alphabet = ["a", "b", "ab", "é"];
        let mut x = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for _ in 0..3000 {
            let mut p = String::new();
            for _ in 0..1 + next(6) {
                p.push_str(["*", "?", "a", "b", "ab", "é"][next(6)]);
            }
            let mut v = String::new();
            for _ in 0..next(9) {
                v.push_str(alphabet[next(alphabet.len())]);
            }
            let m = SubstringMatcher::wildcard(&p, false);
            let slow = WildcardPattern::new(p.as_bytes()).matches(v.as_bytes());
            assert_eq!(
                m.matches(v.as_bytes(), &mut Vec::new()),
                slow,
                "{p:?} {v:?}"
            );
        }
    }

    /// The regexp's prefilter run and early-stopping automaton change no
    /// answer of `RegexpPattern::matches`.
    #[test]
    fn regexp_prefilter_and_automaton_agree_with_the_pattern() {
        for (p, ci) in [
            (".*[Tt]imeout.*", false),
            (".*FAILED password.*", true),
            ("(a|b)*abc.*", false),
            (".*blk_-?[0-9]+ term", false),
        ] {
            let m = SubstringMatcher::regexp(p, ci).unwrap();
            let parsed = RegexpPattern::parse(p).unwrap();
            let r = if ci {
                parsed.ascii_case_insensitive()
            } else {
                parsed
            };
            for &v in VALUES.iter().flatten() {
                let v = v.as_bytes();
                assert_eq!(m.matches(v, &mut Vec::new()), r.matches(v), "{p} {v:?}");
            }
        }
        let lit = |p: &str| RegexpPattern::parse(p).unwrap().required_literal();
        assert_eq!(lit(".*blk_-?[0-9]+ terminating"), b" terminating");
        assert_eq!(lit("(ab)+cde.*"), b"cde");
        assert_eq!(lit(".*(x|y)"), b"");
        assert_eq!(lit("ab&abc"), b"abc");
    }

    /// A case-insensitive regexp folds like Lucene 10.5's `CaseFolding`:
    /// class characters, non-ASCII letters and the Kelvin and long-s signs,
    /// through the index as through a scan.
    #[test]
    fn case_insensitive_regexp_folds_beyond_ascii() {
        let values: &[Option<&str>] = &[
            Some("TIMEOUT now"),
            Some("\u{212A}elvin scale"),
            Some("caf\u{C9} CAF\u{C9} open"),
            Some("STRA\u{1E9E}E 5"),
            Some("pa\u{17F}\u{17F}word failed"),
            Some("nothing here"),
        ];
        let cases = [
            (".*[t]imeout.*", vec![0]),
            ("kelvin.*", vec![1]),
            (".*caf\u{E9}.*", vec![2]),
            ("stra\u{DF}e.*", vec![3]),
            (".*password.*", vec![4]),
        ];
        for (p, want) in cases {
            let m = SubstringMatcher::regexp(p, true).unwrap();
            for shift in [0u8, 2] {
                assert_eq!(search(values, shift, &m), want, "{p} shift {shift}");
            }
            assert_eq!(brute(values, &m), want, "{p}");
        }
    }

    #[test]
    fn case_insensitive_wildcard_lowers_value_and_pattern() {
        let m = SubstringMatcher::contains("TIMEOUT", true);
        let mut s = Vec::new();
        assert!(m.matches(b"read Timeout", &mut s));
        assert!(!SubstringMatcher::contains("TIMEOUT", false).matches(b"read Timeout", &mut s));
        // The Kelvin sign lower-cases to `k`.
        assert!(SubstringMatcher::contains("k", true).matches("\u{212A}".as_bytes(), &mut s));
    }

    #[test]
    fn grams_fold_and_anchor() {
        let m = SubstringMatcher::wildcard("Ab*", false);
        assert_eq!(
            m.grams(),
            &[gram_key(&[BOUNDARY, u32::from(b'a'), u32::from(b'b')])]
        );
        assert!(SubstringMatcher::wildcard("*ab*", false).grams().is_empty());
        // Case-insensitive grams leave out non-ASCII bytes.
        assert!(SubstringMatcher::contains("éé", true).grams().is_empty());
        assert_eq!(SubstringMatcher::contains("éé", false).grams().len(), 2);
    }

    #[test]
    fn postings_round_trip_as_list_and_bitset() {
        // 300 documents, one per block: "common" in five of every seven
        // blocks (a bitset), "rare" in a few (a list).
        let values: Vec<String> = (0..300)
            .map(|i| match i {
                _ if i % 97 == 0 => format!("rare common {i}"),
                _ if i % 7 == 4 || i % 7 == 5 => format!("other {i}"),
                _ => format!("common {i}"),
            })
            .collect();
        let opts: Vec<Option<&str>> = values.iter().map(|s| Some(s.as_str())).collect();
        let (data, entry) = build(&opts, 0);
        let index = SubstringIndex::new(&data, &entry).unwrap();
        let common = gram_key(&[u32::from(b'c'), u32::from(b'o'), u32::from(b'm')]);
        let rare = gram_key(&[u32::from(b'r'), u32::from(b'a'), u32::from(b'r')]);
        let (ci, cdf) = index.find_gram(common).unwrap();
        let (ri, rdf) = index.find_gram(rare).unwrap();
        let commons: Vec<u32> = (0..300)
            .filter(|i| i % 97 == 0 || !(i % 7 == 4 || i % 7 == 5))
            .collect();
        assert_eq!((cdf as usize, rdf), (commons.len(), 4));
        assert!(index.bitset(ci).unwrap().is_some());
        assert!(index.bitset(ri).unwrap().is_none());
        let mut out = Vec::new();
        index.decode(ri, &mut out).unwrap();
        assert_eq!(out, vec![0, 97, 194, 291]);
        index.decode(ci, &mut out).unwrap();
        assert_eq!(out, commons);
        assert_eq!(
            index.candidate_blocks(&[common, rare]).unwrap(),
            Candidates::Blocks(vec![0, 97, 194, 291])
        );
        assert_eq!(
            index
                .candidate_blocks(&[common, gram_key(&[1, 2, 3])])
                .unwrap(),
            Candidates::Blocks(vec![])
        );
        let mut buf = Block::new();
        assert_eq!(
            index.value(97, &mut buf).unwrap(),
            Some(&b"rare common 97"[..])
        );
        assert_eq!(index.value(6, &mut buf).unwrap(), Some(&b"common 6"[..]));
    }

    #[test]
    fn empty_and_sparse_fields() {
        let (data, entry) = build(&[None, None, Some("a"), None], 1);
        let index = SubstringIndex::new(&data, &entry).unwrap();
        let mut buf = Block::new();
        assert_eq!(index.value(0, &mut buf).unwrap(), None);
        assert_eq!(index.value(2, &mut buf).unwrap(), Some(&b"a"[..]));
        assert_eq!(index.value(3, &mut buf).unwrap(), None);
        assert!(index.read_block(9, &mut buf).is_err());
        let (data, entry) = build(&[], 5);
        let index = SubstringIndex::new(&data, &entry).unwrap();
        assert_eq!(index.num_blocks(), 0);
    }

    #[test]
    fn corrupt_meta_is_refused() {
        let mut meta = vec![MAX_BLOCK_SHIFT + 1];
        meta.extend_from_slice(&[0; 64]);
        assert!(read_meta(&mut SliceInput::new(&meta)).is_err());
    }
}
