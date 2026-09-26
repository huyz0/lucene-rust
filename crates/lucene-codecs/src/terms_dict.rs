//! Port of the terms-dictionary half of `Lucene90DocValuesProducer.TermsDict`
//! (used by SORTED/SORTED_SET doc values to map an ordinal to its term
//! bytes) — read-only, **decode-once, not lazy**.
//!
//! On disk, terms are grouped into 64-term blocks: each block's first term
//! is stored uncompressed, and the remaining 63 are prefix-compressed
//! against their immediate predecessor and then LZ4-compressed together
//! using the first term as a preset dictionary. Two auxiliary
//! `DirectMonotonicReader` arrays exist purely to support random access
//! without a full scan: a block-address array (seek straight to any
//! 64-term block) and a coarser "reverse index" over rarer sample terms
//! (binary-search which block a target term might be in). Since this port
//! materializes the whole dictionary in one pass ([`decode_all_terms`])
//! rather than exposing a lazy seekable `TermsEnum`, neither array is
//! needed for lookups — they're still parsed structurally (to keep the
//! `.dvm` cursor aligned for whatever field comes next) but their values
//! are discarded. See the `rust-performance` skill: this is the same
//! decode-fully trade-off already made for `IndexedDISI` and stored fields.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::Result;

use crate::direct_monotonic;
use crate::lz4;

/// `Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_SHIFT`.
const TERMS_DICT_BLOCK_LZ4_SHIFT: u32 = 6;

/// 64 terms per LZ4 block (`1 << TERMS_DICT_BLOCK_LZ4_SHIFT`).
const BLOCK_SIZE: i64 = 1 << TERMS_DICT_BLOCK_LZ4_SHIFT;

#[derive(Debug, Clone)]
pub struct TermsDictEntry {
    pub terms_dict_size: i64,
    pub max_term_length: i32,
    /// `TermsDictEntry.maxBlockLength`: the longest decompressed block body
    /// the writer produced. Java sizes `TermsDict`'s `blockBuffer` from it;
    /// this port uses it as the bound on the per-block length vint, which
    /// otherwise sizes an unbounded allocation.
    pub max_block_length: i32,
    pub terms_data_offset: i64,
    pub terms_data_length: i64,
    /// The block-address array and the reverse terms index, for random
    /// access ([`TermsDict`]); `None` for an entry built by hand.
    pub index: Option<Box<TermsDictIndex>>,
}

/// `TermsDictEntry`'s random-access half: where each 64-term block starts
/// (`termsAddresses`), and every `2^index_shift`-th term's bytes
/// (`termsIndex`, found through `termsIndexAddresses`). All offsets are into
/// the whole `.dvd`.
#[derive(Debug, Clone)]
pub struct TermsDictIndex {
    pub addresses_meta: direct_monotonic::Meta,
    pub addresses_offset: i64,
    pub addresses_length: i64,
    pub index_shift: u32,
    pub index_addresses_meta: direct_monotonic::Meta,
    pub index_offset: i64,
    pub index_length: i64,
    pub index_addresses_offset: i64,
    pub index_addresses_length: i64,
}

/// Parses a `TermsDictEntry` from the `.dvm` metadata stream. Must be called
/// at the exact position `Lucene90DocValuesProducer.readTermDict` would
/// read from (right after a SORTED/SORTED_SET field's ords entry).
pub fn read_term_dict_entry(input: &mut SliceInput) -> Result<TermsDictEntry> {
    let terms_dict_size = input.read_vlong()?;
    let block_shift = input.read_i32()? as u32;
    // `terms_dict_size` is a vlong off the `.dvm` and so is unbounded in both
    // directions. Java's `(termsDictSize + (1L << SHIFT) - 1) >>> SHIFT` wraps
    // where this port's `+` panics, and a negative size then reaches
    // `load_meta` as a negative block count. Both are corruption.
    let Some(addresses_size) = terms_dict_size
        .checked_add(BLOCK_SIZE - 1)
        .filter(|_| terms_dict_size >= 0)
        .map(|n| n >> TERMS_DICT_BLOCK_LZ4_SHIFT)
    else {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict size out of range: {terms_dict_size}"
        )));
    };
    let addresses_meta = direct_monotonic::load_meta(input, addresses_size, block_shift)?;
    let max_term_length = input.read_i32()?;
    let max_block_length = input.read_i32()?;
    if max_block_length < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict maxBlockLength must be non-negative, got {max_block_length}"
        )));
    }
    let terms_data_offset = input.read_i64()?;
    let terms_data_length = input.read_i64()?;
    let addresses_offset = input.read_i64()?;
    let addresses_length = input.read_i64()?;
    let terms_dict_index_shift = input.read_i32()? as u32;
    // `termsDictIndexShift` is a raw `int` off the `.dvm`. Java's `1L <<
    // shift` masks the shift to its low six bits, so a corrupt value there is
    // merely a wrong answer; in Rust it is a panic in a debug build and a
    // masked shift in a release one. The writer only ever emits
    // `TERMS_DICT_REVERSE_INDEX_SHIFT`, so anything that cannot even be a
    // legal `i64` shift is corruption.
    if terms_dict_index_shift >= 63 {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict reverse-index shift out of range: {terms_dict_index_shift}"
        )));
    }
    // ARITH: `terms_dict_index_shift < 63` so `1i64 << shift` is in range and
    // positive; `terms_dict_size >= 0` was established above, so the
    // `checked_add` is the only place this can go wrong.
    #[allow(clippy::arithmetic_side_effects)]
    let index_size = terms_dict_size.checked_add((1i64 << terms_dict_index_shift) - 1);
    let Some(num_index_values) = index_size
        .map(|n| n >> terms_dict_index_shift)
        .and_then(|n| n.checked_add(1))
    else {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict reverse-index size out of range: termsDictSize={terms_dict_size} \
             shift={terms_dict_index_shift}"
        )));
    };
    let index_addresses_meta = direct_monotonic::load_meta(input, num_index_values, block_shift)?;
    let index_offset = input.read_i64()?;
    let index_length = input.read_i64()?;
    let index_addresses_offset = input.read_i64()?;
    let index_addresses_length = input.read_i64()?;

    Ok(TermsDictEntry {
        terms_dict_size,
        max_term_length,
        max_block_length,
        terms_data_offset,
        terms_data_length,
        index: Some(Box::new(TermsDictIndex {
            addresses_meta,
            addresses_offset,
            addresses_length,
            index_shift: terms_dict_index_shift,
            index_addresses_meta,
            index_offset,
            index_length,
            index_addresses_offset,
            index_addresses_length,
        })),
    })
}

/// A one-term-at-a-time cursor over a SORTED/SORTED_SET doc-values terms
/// dictionary — this port's `TermsDict`, which is what
/// `Lucene90DocValuesProducer.TermsDict` is: an enumerator, not a
/// materializer.
///
/// [`decode_all_terms`] is this cursor collected, and stays the right call
/// for a caller that genuinely needs the whole dictionary in memory
/// (resolving a facet label by ordinal, comparing two dictionaries during a
/// merge). What it is *not* right for is a caller that only walks the
/// dictionary once in order: `OrdinalMap::build` merge-sorts every segment's
/// terms and keeps none of them, and materializing its input cost **267 MB of
/// a 319 MB peak** on a 5-segment x 1 M-term shape where the map itself is
/// 51 MB (`docs/sweep/m2/c29-search-carryovers.md`). Java never pays it
/// because `OrdinalMap.build` takes `TermsEnum[]`.
///
/// The current term is held in one buffer that is reused across calls, so a
/// full walk allocates only as the longest term grows — the prefix-compressed
/// format hands each term its predecessor's prefix, so the prefix never has
/// to be copied at all. That is what Java does too: `TermsDict.next` keeps
/// one `BytesRef term` sized at `maxTermLength`, sets `term.length =
/// prefixLength + suffixLength` and reads the suffix in **at offset
/// `prefixLength`**, leaving the prefix bytes exactly where the previous term
/// left them.
pub struct TermsCursor<'a> {
    input: SliceInput<'a>,
    terms_dict_size: i64,
    max_block_length: i32,
    /// The ordinal of the term [`Self::next_term`] will produce, i.e. how many it
    /// has already produced.
    ord: i64,
    /// Decompressed body of the current block (everything after its first,
    /// uncompressed term), plus a manual read cursor into it — not a
    /// `SliceInput`, since that would borrow `block_body` across the call
    /// that reassigns it.
    block_body: Vec<u8>,
    block_pos: usize,
    /// The current term, and the previous one until it is overwritten: the
    /// prefix a prefix-compressed term keeps is its own predecessor's, so
    /// truncate-and-extend is both the cheapest and the most direct
    /// expression of the format.
    term: Vec<u8>,
}

impl<'a> TermsCursor<'a> {
    /// Opens a cursor positioned before the first term. `data` is the whole
    /// `.dvd` file's bytes; the entry names the region inside it.
    pub fn open(data: &'a [u8], entry: &TermsDictEntry) -> Result<Self> {
        // Both halves are `i64`s read straight off the `.dvm`, so their sum is
        // as untrusted as either one: `offset + length` overflows before
        // `data.get` ever sees a range, and a negative offset becomes a huge
        // `usize` through the `as` cast. Same shape, same fix as
        // `norms::sparse_region`.
        let start = usize::try_from(entry.terms_data_offset)
            .map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
        let end = entry
            .terms_data_offset
            .checked_add(entry.terms_data_length)
            .and_then(|e| usize::try_from(e).ok())
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        let region = data
            .get(start..end)
            .ok_or(lucene_store::Error::Eof { offset: 0 })?;
        Ok(TermsCursor {
            input: SliceInput::new(region),
            terms_dict_size: entry.terms_dict_size,
            max_block_length: entry.max_block_length,
            ord: 0,
            block_body: Vec::new(),
            block_pos: 0,
            term: Vec::new(),
        })
    }

    /// `TermsDict.next()`: the next term in ordinal order, or `None` at the
    /// end of the dictionary. The returned slice is valid until the next
    /// call.
    ///
    /// Not spelled `next`, and not an [`Iterator`]: the returned slice
    /// borrows the cursor's own reused buffer, which is exactly the shape
    /// `Iterator` cannot express (a lending iterator) and exactly the shape
    /// that makes a full walk allocation-free.
    // ARITH: `terms_dict_size` is non-negative (`read_term_dict_entry`
    // rejects a negative one), `ord` starts at 0 and advances by exactly 1 per
    // call while `ord < terms_dict_size`, so `ord` stays in
    // `0..=terms_dict_size` and `ord + 1` is at most `terms_dict_size <=
    // i64::MAX`. `BLOCK_SIZE` is a non-zero constant.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn next_term(&mut self) -> Result<Option<&[u8]>> {
        if self.ord >= self.terms_dict_size {
            return Ok(None);
        }
        if self.ord % BLOCK_SIZE == 0 {
            self.read_block_first_term()?;
        } else {
            self.read_prefix_compressed_term()?;
        }
        self.ord += 1;
        Ok(Some(&self.term))
    }

    /// A block's first term: stored uncompressed, and the preset dictionary
    /// the rest of the block's LZ4 body decompresses against.
    // ARITH: see [`Self::next_term`]; `ord + 1` cannot overflow.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_block_first_term(&mut self) -> Result<()> {
        let first_len = self.input.read_vint()?;
        // Java allocates `term.bytes` at `maxTermLength` once and would
        // throw on an over-long length; `first_len as usize` on a
        // negative vint is ~2^64 here, and `resize` aborts rather than
        // erroring, which the FFI boundary cannot catch.
        if first_len < 0 || first_len as usize > self.input.remaining() {
            return Err(lucene_store::Error::Corrupted(format!(
                "invalid terms dict first-term length {first_len}"
            )));
        }
        let first_len = first_len as usize;
        self.term.clear();
        self.term.resize(first_len, 0);
        let mut term = std::mem::take(&mut self.term);
        let read = self.input.read_bytes(&mut term);
        self.term = term;
        read?;

        // Only decompress a block body if more terms remain after this
        // block's first term (mirrors Java's `decompressBlock`, which
        // skips this when the first term is the very last one written).
        if self.ord + 1 < self.terms_dict_size {
            // `block_len` is the *decompressed* body length, a vint off
            // disk that sizes an allocation directly. A negative one
            // becomes ~2^64 through `as usize`, and `vec![0u8; n]`
            // *aborts* on a failed allocation rather than erroring --
            // `catch_unwind` cannot intercept that, so it takes the JVM
            // down through the FFI. It cannot be bounded by the bytes left
            // in the region (LZ4 expands), so bound it the way Java does:
            // `TermsDict` sizes its `blockBuffer` from `maxBlockLength`,
            // which the writer set to the longest body it emitted.
            let block_len = self.input.read_vint()?;
            if block_len < 0 || block_len > self.max_block_length {
                return Err(lucene_store::Error::Corrupted(format!(
                    "invalid terms dict block length {block_len} \
                     (maxBlockLength={})",
                    self.max_block_length
                )));
            }
            // ARITH: `term.len()` is bounded by the terms region (the
            // first-term length check above, itself at most `isize::MAX`)
            // and `block_len` by `max_block_length`, an `i32`, so the sum
            // cannot overflow `usize`.
            let buffer_len = self.term.len() + block_len as usize;
            let block_len = block_len as usize;
            let mut buffer = vec![0u8; buffer_len];
            buffer[..self.term.len()].copy_from_slice(&self.term);
            lz4::decompress_slice(&mut self.input, block_len, &mut buffer, self.term.len())?;
            buffer.drain(..self.term.len());
            self.block_body = buffer;
        } else {
            self.block_body.clear();
        }
        self.block_pos = 0;
        Ok(())
    }

    /// Any term but a block's first: a prefix taken from its immediate
    /// predecessor (already in `self.term`) plus a suffix out of the block
    /// body.
    fn read_prefix_compressed_term(&mut self) -> Result<()> {
        let token = read_u8(&self.block_body, &mut self.block_pos)? as usize;
        // ARITH: `token` is a single byte, so `token & 0x0F` is `0..=15`
        // and `1 + (token >> 4)` is `1..=16`.
        #[allow(clippy::arithmetic_side_effects)]
        let (mut prefix_len, mut suffix_len) = (token & 0x0F, 1 + (token >> 4));
        let previous_len = self.term.len();
        // The two continuation vints are read off disk and size the term
        // this call builds. A negative vint becomes ~2^64 through
        // `as usize`, which overflows the `+=` (a debug-build panic) and
        // otherwise reaches `Vec::with_capacity` as an aborting
        // allocation. The prefix is copied out of the previous term and
        // the suffix out of the block body, so each has its own limit --
        // the first term of a block is stored outside the body and can be
        // longer than it, so bounding the prefix by the body would reject
        // files Lucene wrote.
        if prefix_len == 15 {
            prefix_len = bounded_extension(
                prefix_len,
                previous_len,
                &self.block_body,
                &mut self.block_pos,
            )?;
        }
        if suffix_len == 16 {
            suffix_len = bounded_extension(
                suffix_len,
                self.block_body.len(),
                &self.block_body,
                &mut self.block_pos,
            )?;
        }
        // Java copies into a `term.bytes` buffer sized `maxTermLength` and
        // lets the JVM throw on an over-long prefix; slicing here would
        // panic instead, and a panic cannot cross the FFI boundary.
        if prefix_len > previous_len {
            return Err(lucene_store::Error::Corrupted(format!(
                "terms dict prefix length {prefix_len} exceeds previous term length {previous_len}"
            )));
        }
        // ARITH: `block_pos <= block_body.len()` is
        // `read_u8`/`read_vint`'s invariant and `suffix_len <=
        // block_body.len()` comes out of `bounded_extension`, so
        // `block_pos + suffix_len` cannot overflow -- it is merely allowed
        // to run past the end, which `get` reports.
        #[allow(clippy::arithmetic_side_effects)]
        let suffix_end = self.block_pos + suffix_len;
        if suffix_end > self.block_body.len() {
            return Err(lucene_store::Error::Eof {
                offset: self.block_pos,
            });
        }
        // The prefix this term keeps is its predecessor's, which is what
        // `self.term` already holds -- so truncating is the copy.
        self.term.truncate(prefix_len);
        self.term
            .extend_from_slice(&self.block_body[self.block_pos..suffix_end]);
        self.block_pos = suffix_end;
        Ok(())
    }
}

impl TermsCursor<'_> {
    /// Moves to the start of block `block`, `address` bytes into the terms
    /// region: the next [`Self::next_term`] reads ordinal `block * 64`.
    fn seek_block(&mut self, block: i64, address: usize) -> Result<()> {
        self.input.seek(address)?;
        self.ord = block
            .checked_mul(BLOCK_SIZE)
            .ok_or_else(|| lucene_store::Error::Corrupted(format!("terms dict block {block}")))?;
        Ok(())
    }
}

/// Where [`TermsDict::seek_ceil`] landed: `TermsEnum.SeekStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekStatus {
    Found,
    /// On the first term past the target.
    NotFound,
    /// Past the last term.
    End,
}

fn region(data: &[u8], offset: i64, length: i64) -> Result<&[u8]> {
    let start = usize::try_from(offset).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let end = offset
        .checked_add(length)
        .and_then(|e| usize::try_from(e).ok())
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    data.get(start..end)
        .ok_or(lucene_store::Error::Eof { offset: start })
}

fn usize_of(v: i64, what: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| lucene_store::Error::Corrupted(format!("negative {what} {v}")))
}

/// `Lucene90DocValuesProducer.TermsDict` with its random access: a term by
/// ordinal (`seekExact(long)`, a block-address lookup and a scan inside the
/// block) and an ordinal by term (`seekCeil`, a binary search of the reverse
/// index, then of the blocks' first terms, then a scan) -- what a keyword sort
/// needs to carry a value from one segment to the next.
pub struct TermsDict<'a> {
    cursor: TermsCursor<'a>,
    size: i64,
    index: &'a TermsDictIndex,
    addresses: &'a [u8],
    index_addresses: &'a [u8],
    index_bytes: &'a [u8],
    terms: &'a [u8],
    /// The ordinal of the term the cursor holds; `-1` before any.
    ord: i64,
}

impl<'a> TermsDict<'a> {
    /// Opens the dictionary of `entry` in `data` (the whole `.dvd`).
    pub fn open(data: &'a [u8], entry: &'a TermsDictEntry) -> Result<Self> {
        let index = entry.index.as_deref().ok_or_else(|| {
            lucene_store::Error::Corrupted("terms dict entry without its index".into())
        })?;
        Ok(TermsDict {
            cursor: TermsCursor::open(data, entry)?,
            size: entry.terms_dict_size,
            index,
            addresses: region(data, index.addresses_offset, index.addresses_length)?,
            index_addresses: region(
                data,
                index.index_addresses_offset,
                index.index_addresses_length,
            )?,
            index_bytes: region(data, index.index_offset, index.index_length)?,
            terms: region(data, entry.terms_data_offset, entry.terms_data_length)?,
            ord: -1,
        })
    }

    /// `termsDictSize`.
    pub fn size(&self) -> i64 {
        self.size
    }

    /// The current term.
    pub fn term(&self) -> &[u8] {
        &self.cursor.term
    }

    /// The current term's ordinal (`-1` before any seek).
    pub fn ord(&self) -> i64 {
        self.ord
    }

    /// `seekExact(ord)`: the term of ordinal `ord`.
    // ARITH: `0 <= ord < size <= i64::MAX` (checked first), so `ord >> 6`
    // and `(block << 6) - 1 >= -1` are in range, and `self.ord + 1 <= ord`
    // inside the loop.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_ord(&mut self, ord: i64) -> Result<&[u8]> {
        if ord < 0 || ord >= self.size {
            return Err(lucene_store::Error::Corrupted(format!(
                "terms dict ordinal {ord} outside 0..{}",
                self.size
            )));
        }
        let block = ord >> TERMS_DICT_BLOCK_LZ4_SHIFT;
        if ord < self.ord || block != self.ord >> TERMS_DICT_BLOCK_LZ4_SHIFT {
            let address = direct_monotonic::get(self.addresses, &self.index.addresses_meta, block)?;
            self.cursor
                .seek_block(block, usize_of(address, "terms block address")?)?;
            self.ord = (block << TERMS_DICT_BLOCK_LZ4_SHIFT) - 1;
        }
        while self.ord < ord {
            if self.cursor.next_term()?.is_none() {
                return Err(lucene_store::Error::Corrupted(format!(
                    "terms dict ended before ordinal {ord}"
                )));
            }
            self.ord += 1;
        }
        Ok(&self.cursor.term)
    }

    /// `getTermFromIndex`: the reverse index's `i`-th sample.
    // ARITH: `i` is a reverse-index position found by `seek_terms_index`'s
    // binary search, so `i + 1` is at most its (non-negative) upper bound + 1.
    #[allow(clippy::arithmetic_side_effects)]
    fn term_from_index(&self, i: i64) -> Result<&'a [u8]> {
        let start =
            direct_monotonic::get(self.index_addresses, &self.index.index_addresses_meta, i)?;
        let end = direct_monotonic::get(
            self.index_addresses,
            &self.index.index_addresses_meta,
            i + 1,
        )?;
        if end < start {
            return Err(lucene_store::Error::Corrupted(format!(
                "terms index addresses decrease: {start} then {end}"
            )));
        }
        region(self.index_bytes, start, end - start)
    }

    /// `getFirstTermFromBlock`: a block's uncompressed first term.
    fn first_term_of_block(&self, block: i64) -> Result<&'a [u8]> {
        let address = direct_monotonic::get(self.addresses, &self.index.addresses_meta, block)?;
        let mut input = SliceInput::new(self.terms);
        input.seek(usize_of(address, "terms block address")?)?;
        let len = usize_of(i64::from(input.read_vint()?), "first term length")?;
        let at = input.position();
        let end = at
            .checked_add(len)
            .ok_or(lucene_store::Error::Eof { offset: at })?;
        self.terms
            .get(at..end)
            .ok_or(lucene_store::Error::Eof { offset: at })
    }

    /// `seekTermsIndex`: the last reverse-index sample at or before `key`
    /// (`-1` only for an empty dictionary: sample 0 is the empty term).
    // ARITH: `lo <= hi + 1` and both are within `-1..=(size - 1) >> shift`,
    // with `size >= 1` here; `(lo + hi) >> 1` cannot overflow for these.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_terms_index(&self, key: &[u8]) -> Result<i64> {
        let (mut lo, mut hi) = (0i64, (self.size - 1) >> self.index.index_shift);
        while lo <= hi {
            let mid = (lo + hi) >> 1;
            if self.term_from_index(mid)? <= key {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        Ok(hi)
    }

    /// `seekCeil`: moves to `key`, or to the first term after it.
    // ARITH: `index >= 0` and `index <= (size - 1) >> shift`, so `index <<
    // shift < size`; `ord_lo + 2^shift` is at most `size + 2^shift`, which the
    // writer's shift (at most 62, checked on read) keeps within `i64`. The
    // block search mirrors `seek_terms_index`'s.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_ceil(&mut self, key: &[u8]) -> Result<SeekStatus> {
        if self.size <= 0 {
            self.ord = 0;
            return Ok(SeekStatus::End);
        }
        let index = self.seek_terms_index(key)?;
        if index < 0 {
            self.ord = 0;
            return Ok(SeekStatus::End);
        }
        let shift = self.index.index_shift;
        let ord_lo = index << shift;
        let ord_hi = self.size.min(ord_lo.saturating_add(1i64 << shift)) - 1;
        let (mut lo, mut hi) = (
            ord_lo >> TERMS_DICT_BLOCK_LZ4_SHIFT,
            ord_hi >> TERMS_DICT_BLOCK_LZ4_SHIFT,
        );
        while lo <= hi {
            let mid = (lo + hi) >> 1;
            if self.first_term_of_block(mid)? <= key {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        let block = hi.max(0);
        let address = direct_monotonic::get(self.addresses, &self.index.addresses_meta, block)?;
        self.cursor
            .seek_block(block, usize_of(address, "terms block address")?)?;
        if self.cursor.next_term()?.is_none() {
            return Ok(SeekStatus::End);
        }
        self.ord = block << TERMS_DICT_BLOCK_LZ4_SHIFT;
        if hi < 0 {
            return Ok(SeekStatus::NotFound);
        }
        loop {
            match self.cursor.term.as_slice().cmp(key) {
                std::cmp::Ordering::Equal => return Ok(SeekStatus::Found),
                std::cmp::Ordering::Greater => return Ok(SeekStatus::NotFound),
                std::cmp::Ordering::Less => {}
            }
            if self.cursor.next_term()?.is_none() {
                return Ok(SeekStatus::End);
            }
            self.ord += 1;
        }
    }

    /// `SortedDocValues.lookupTerm`: `key`'s ordinal, or `-insertionPoint - 1`.
    // ARITH: `ord` and `size` are in `0..=i64::MAX`, so their negation minus
    // one is in range.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        Ok(match self.seek_ceil(key)? {
            SeekStatus::Found => self.ord,
            SeekStatus::NotFound => -self.ord - 1,
            SeekStatus::End => -self.size - 1,
        })
    }
}

/// Decodes every term in the dictionary, in ordinal order — [`TermsCursor`]
/// collected. `data` is the whole `.dvd` file's bytes.
///
/// Right for a caller that needs random access to the dictionary by ordinal;
/// see [`TermsCursor`] for what it costs a caller that only walks it once.
pub fn decode_all_terms(data: &[u8], entry: &TermsDictEntry) -> Result<Vec<Vec<u8>>> {
    let mut cursor = TermsCursor::open(data, entry)?;
    // `termsDictSize` comes straight off the wire, so it cannot be trusted to
    // size an allocation on its own: every term costs at least one byte in the
    // terms region, so the region's own length is a hard upper bound on how
    // many there can be. Java never faces this -- it does not preallocate.
    let capacity = entry.terms_dict_size.min(entry.terms_data_length).max(0) as usize;
    let mut terms: Vec<Vec<u8>> = Vec::with_capacity(capacity);
    while let Some(term) = cursor.next_term()? {
        terms.push(term.to_vec());
    }
    Ok(terms)
}

/// Adds a term-length continuation vint (read from `buf` at `pos`) to `base`,
/// rejecting anything past `limit`. Shared by the prefix and suffix halves of
/// a prefix-compressed term, which have the same hazard -- the vint is off
/// disk, `as usize` on a negative one is ~2^64, and the sum sizes both an
/// allocation and a slice range -- but different limits: a prefix is copied
/// out of the previous term, a suffix out of the block body.
fn bounded_extension(base: usize, limit: usize, buf: &[u8], pos: &mut usize) -> Result<usize> {
    let extra = read_vint(buf, pos)?;
    let sum = usize::try_from(extra)
        .ok()
        .and_then(|extra| base.checked_add(extra))
        .filter(|&sum| sum <= limit);
    sum.ok_or_else(|| {
        lucene_store::Error::Corrupted(format!(
            "terms dict term-length extension {extra} is out of range (base {base}, limit {limit})"
        ))
    })
}

// ARITH: `get` returned `Some`, so `*pos < buf.len() <= isize::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8> {
    let b = *buf
        .get(*pos)
        .ok_or(lucene_store::Error::Eof { offset: *pos })?;
    *pos += 1;
    Ok(b)
}

/// Plain vint decode over a `&[u8]` cursor (the block body isn't a
/// `DataInput` -- see the comment on `block_body`/`block_pos` above).
///
/// Bounded at five bytes, which is the most `DataOutput.writeVInt` emits
/// ("between one and five bytes" -- its loop is `i >>>= 7` on an `int`), so
/// the bound cannot reject a `vint` any writer produced. It is deliberately
/// stricter than 10.5.0's *reader*: `DataInput.readVInt` there is a bare
/// unchecked loop, and the "Invalid vInt detected (too many bits)" exception
/// this comment used to cite is a later addition that is **not** in the pinned
/// tree. Unbounded, the shift itself overflows on the sixth byte -- a panic in
/// debug builds, a silently wrong value in release ones.
// ARITH: `shift` starts at 7, is compared against 28 before every use, and
// advances by 7, so it never leaves `7..=35` and `<<` on an `i32` stays in
// range for the shifts that actually run (the loop returns before shifting by
// 35).
#[allow(clippy::arithmetic_side_effects)]
fn read_vint(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut b = read_u8(buf, pos)?;
    let mut v = (b & 0x7f) as i32;
    let mut shift = 7;
    while b & 0x80 != 0 {
        if shift > 28 {
            return Err(lucene_store::Error::Corrupted(
                "invalid vInt detected (too many bits)".into(),
            ));
        }
        b = read_u8(buf, pos)?;
        v |= ((b & 0x7f) as i32) << shift;
        shift += 7;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// A "stored, not compressed" literal-only LZ4 encoding of `bytes`
    /// (mirrors `stored_fields.rs`'s test helper of the same shape).
    fn encode_literal_lz4(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let len = bytes.len();
        assert!(len < 0x0F, "test helper only supports short literals");
        out.push((len as u8) << 4);
        out.extend_from_slice(bytes);
        out
    }

    #[test]
    fn decode_all_terms_empty_dict() {
        let entry = TermsDictEntry {
            terms_dict_size: 0,
            max_term_length: 0,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: 0,
            index: None,
        };
        assert_eq!(
            decode_all_terms(&[], &entry).unwrap(),
            Vec::<Vec<u8>>::new()
        );
    }

    #[test]
    fn decode_all_terms_single_term_no_block_body() {
        // Only 1 term: no block body is ever written (ord+1 < size is
        // false immediately after the first, uncompressed term).
        let mut data = Vec::new();
        write_vint(&mut data, 5);
        data.extend_from_slice(b"apple");

        let entry = TermsDictEntry {
            terms_dict_size: 1,
            max_term_length: 5,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
        };
        assert_eq!(
            decode_all_terms(&data, &entry).unwrap(),
            vec![b"apple".to_vec()]
        );
    }

    /// 3 terms sharing prefixes: "apple", "application", "apply".
    /// "application" vs "apple": common prefix "appl" (4), suffix "ication" (7).
    /// "apply" vs "application": common prefix "appl" (4), suffix "y" (1).
    fn prefix_compressed_block_fixture() -> (Vec<u8>, TermsDictEntry) {
        let mut block_body = Vec::new();
        block_body.push((6u8 << 4) | 4); // suffixLen-1=6, prefixLen=4
        block_body.extend_from_slice(b"ication");
        block_body.push(4); // suffixLen-1=0, prefixLen=4
        block_body.push(b'y');

        let compressed_body = encode_literal_lz4(&block_body);

        let mut data = Vec::new();
        write_vint(&mut data, 5);
        data.extend_from_slice(b"apple");
        write_vint(&mut data, block_body.len() as i32); // decompressed block length
        data.extend_from_slice(&compressed_body);

        let entry = TermsDictEntry {
            terms_dict_size: 3,
            max_term_length: 11,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
        };
        (data, entry)
    }

    #[test]
    fn decode_all_terms_prefix_compressed_block() {
        let (data, entry) = prefix_compressed_block_fixture();
        assert_eq!(
            decode_all_terms(&data, &entry).unwrap(),
            vec![
                b"apple".to_vec(),
                b"application".to_vec(),
                b"apply".to_vec(),
            ]
        );
    }

    /// An exhausted cursor keeps saying so.
    ///
    /// This is the one thing about the cursor `decode_all_terms`' own tests
    /// cannot cover: `decode_all_terms` *is* the cursor collected
    /// (`while let Some(term) = cursor.next_term()?`), so comparing the two
    /// would compare a function to itself. What actually pins the two
    /// decoders together is that there is only one, and what pins the cursor
    /// against Java is `tests/sorted_doc_values_fixtures.rs`, which walks it
    /// over a real `.dvd` and compares against the manifest real Lucene
    /// wrote. A caller merging several cursors stops calling the exhausted
    /// ones, but must not have to know that.
    #[test]
    fn an_exhausted_terms_cursor_keeps_returning_none() {
        let (data, entry) = prefix_compressed_block_fixture();
        let mut cursor = TermsCursor::open(&data, &entry).unwrap();
        let mut count = 0;
        while cursor.next_term().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, entry.terms_dict_size, "the whole dictionary");
        assert!(cursor.next_term().unwrap().is_none());
        assert!(cursor.next_term().unwrap().is_none());
    }

    #[test]
    fn decode_all_terms_extended_prefix_and_suffix_lengths() {
        // prefixLen=15 (needs a vint extension: +2 -> 17) and
        // suffixLen=16 (needs a vint extension: +3 -> 19).
        let previous = vec![b'a'; 20];
        let suffix = vec![b'b'; 19];
        let mut block_body = Vec::new();
        block_body.push((15u8 << 4) | 15); // suffixLen field=15 (extended), prefixLen field=15 (extended)
        write_vint(&mut block_body, 2); // prefixLen = 15 + 2 = 17
        write_vint(&mut block_body, 3); // suffixLen = 16 + 3 = 19
        block_body.append(&mut suffix.clone());

        let mut compressed_body = Vec::new();
        // literal length 25 (>= 0x0F) needs the extended-literal-length LZ4
        // encoding: token nibble 0x0F, then a length-extension byte.
        compressed_body.push(0xF0);
        compressed_body.push((block_body.len() - 0x0F) as u8);
        compressed_body.extend_from_slice(&block_body);

        let mut data = Vec::new();
        write_vint(&mut data, previous.len() as i32);
        data.extend_from_slice(&previous);
        write_vint(&mut data, block_body.len() as i32);
        data.extend_from_slice(&compressed_body);

        let entry = TermsDictEntry {
            terms_dict_size: 2,
            max_term_length: 39,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
        };
        let terms = decode_all_terms(&data, &entry).unwrap();
        assert_eq!(terms[0], previous);
        let mut expected_second = previous[..17].to_vec();
        expected_second.extend_from_slice(&suffix);
        assert_eq!(terms[1], expected_second);
    }

    /// A prefix length longer than the previous term is corrupt input. Java
    /// copies into a `maxTermLength`-sized array and lets the JVM throw;
    /// slicing `previous[..prefix_len]` here would panic, and a panic cannot
    /// cross the FFI boundary.
    #[test]
    fn decode_all_terms_rejects_prefix_longer_than_the_previous_term() {
        // suffixLen-1=0, prefixLen=9 -- but the previous term is 5 bytes.
        let block_body = vec![9u8, b'y'];
        let compressed_body = encode_literal_lz4(&block_body);

        let mut data = Vec::new();
        write_vint(&mut data, 5);
        data.extend_from_slice(b"apple");
        write_vint(&mut data, block_body.len() as i32);
        data.extend_from_slice(&compressed_body);

        let entry = TermsDictEntry {
            terms_dict_size: 2,
            max_term_length: 14,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
        };
        assert!(matches!(
            decode_all_terms(&data, &entry),
            Err(lucene_store::Error::Corrupted(_))
        ));
    }

    /// A first-term length past the end of the terms region must be an error,
    /// not a `vec![0u8; ~2^64]` that aborts the process.
    #[test]
    fn decode_all_terms_rejects_first_term_length_past_the_region() {
        for first_len in [i32::MAX, -1] {
            let mut data = Vec::new();
            write_vint(&mut data, first_len);
            data.extend_from_slice(b"apple");
            let entry = TermsDictEntry {
                terms_dict_size: 1,
                max_term_length: 5,
                max_block_length: 8192,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
                index: None,
            };
            assert!(
                matches!(
                    decode_all_terms(&data, &entry),
                    Err(lucene_store::Error::Corrupted(_))
                ),
                "first_len={first_len}"
            );
        }
    }

    /// `DataInput.readVInt` throws past five bytes; unbounded, the shift here
    /// overflows on the sixth (a debug-build panic, a wrong value in release).
    #[test]
    fn block_body_vint_is_bounded_at_five_bytes_like_java() {
        let mut pos = 0usize;
        let too_long = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01];
        assert!(matches!(
            read_vint(&too_long, &mut pos),
            Err(lucene_store::Error::Corrupted(_))
        ));

        // Five bytes is still legal, and decodes the same value Java's
        // readVInt does.
        let mut pos = 0usize;
        let max = [0xFFu8, 0xFF, 0xFF, 0xFF, 0x07];
        assert_eq!(read_vint(&max, &mut pos).unwrap(), i32::MAX);
        assert_eq!(pos, 5);
    }

    #[test]
    fn read_term_dict_entry_round_trips_fixed_fields() {
        let mut out = Vec::new();
        write_vint_vlong(&mut out, 100); // termsDictSize
        out.extend_from_slice(&2i32.to_le_bytes()); // blockShift
        let addresses_size = (100 + BLOCK_SIZE - 1) >> 6;
        write_direct_monotonic_meta(&mut out, addresses_size);
        out.extend_from_slice(&50i32.to_le_bytes()); // maxTermLength
        out.extend_from_slice(&8192i32.to_le_bytes()); // maxBlockLength
        out.extend_from_slice(&1000i64.to_le_bytes()); // termsDataOffset
        out.extend_from_slice(&2000i64.to_le_bytes()); // termsDataLength
        out.extend_from_slice(&0i64.to_le_bytes()); // termsAddressesOffset
        out.extend_from_slice(&0i64.to_le_bytes()); // termsAddressesLength
        out.extend_from_slice(&4i32.to_le_bytes()); // termsDictIndexShift
        let index_size = (100 + (1i64 << 4) - 1) >> 4;
        write_direct_monotonic_meta(&mut out, 1 + index_size);
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());

        let mut input = SliceInput::new(&out);
        let entry = read_term_dict_entry(&mut input).unwrap();
        assert_eq!(entry.terms_dict_size, 100);
        assert_eq!(entry.max_term_length, 50);
        assert_eq!(entry.max_block_length, 8192);
        assert_eq!(entry.terms_data_offset, 1000);
        assert_eq!(entry.terms_data_length, 2000);
        assert_eq!(input.remaining(), 0);
    }

    /// `termsDictIndexShift` is a raw `i32` off the `.dvm`. `1i64 << shift`
    /// panics for a shift of 64 or more in a debug build (Java's `1L <<`
    /// merely masks it), and the `termsDictSize + (1 << shift) - 1` above it
    /// overflows for a large size.
    #[test]
    fn corrupt_reverse_index_shift_is_a_decode_error_not_a_shift_panic() {
        for shift in [63i32, 64, 100, -1] {
            let mut out = Vec::new();
            write_vint_vlong(&mut out, 100);
            out.extend_from_slice(&2i32.to_le_bytes());
            write_direct_monotonic_meta(&mut out, (100 + BLOCK_SIZE - 1) >> 6);
            out.extend_from_slice(&50i32.to_le_bytes());
            out.extend_from_slice(&8192i32.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&shift.to_le_bytes());
            out.extend_from_slice(&[0u8; 64]);

            let mut input = SliceInput::new(&out);
            assert!(read_term_dict_entry(&mut input).is_err(), "shift={shift}");
        }
    }

    /// A `termsDictSize` near `i64::MAX` overflowed the `+ (BLOCK_SIZE - 1)`
    /// rounding before it ever reached `DirectMonotonicReader.loadMeta`, and a
    /// negative one reached it as a negative block count.
    #[test]
    fn corrupt_terms_dict_size_is_a_decode_error() {
        for size in [i64::MAX, i64::MAX - 10, -1] {
            let mut out = Vec::new();
            write_vint_vlong(&mut out, size);
            out.extend_from_slice(&2i32.to_le_bytes());
            out.extend_from_slice(&[0u8; 64]);
            let mut input = SliceInput::new(&out);
            assert!(read_term_dict_entry(&mut input).is_err(), "size={size}");
        }
    }

    /// `termsDataOffset`/`termsDataLength` are two independent `i64`s off the
    /// `.dvm`, so their sum is as untrusted as either: it overflowed before
    /// `data.get` saw a range, and a negative offset became a huge `usize`.
    #[test]
    fn corrupt_terms_data_region_is_a_decode_error() {
        let data = [0u8; 64];
        for (offset, length) in [(i64::MAX, 1i64), (0, i64::MAX), (-1, 4), (4, -1)] {
            let entry = TermsDictEntry {
                terms_dict_size: 1,
                max_term_length: 4,
                max_block_length: 8192,
                terms_data_offset: offset,
                terms_data_length: length,
                index: None,
            };
            assert!(
                decode_all_terms(&data, &entry).is_err(),
                "offset={offset} length={length}"
            );
        }
    }

    /// The compressed block length is a vint off disk that sizes
    /// `vec![0u8; term.len() + block_len]`. A negative vint is ~2^64 through
    /// `as usize`, and a failed `vec!` allocation *aborts* -- `catch_unwind`
    /// cannot intercept it.
    #[test]
    fn corrupt_block_length_is_a_decode_error_not_an_allocation() {
        // A block length that is positive, plausible and still past the
        // `.dvm`'s own `maxBlockLength` -- the case neither the negative check
        // nor a later EOF would catch.
        {
            let mut data = Vec::new();
            write_vint(&mut data, 5);
            data.extend_from_slice(b"apple");
            write_vint(&mut data, 100);
            data.extend_from_slice(&[0u8; 64]);
            let entry = TermsDictEntry {
                terms_dict_size: 3,
                max_term_length: 11,
                max_block_length: 8,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
                index: None,
            };
            let got = decode_all_terms(&data, &entry);
            assert!(
                matches!(&got, Err(lucene_store::Error::Corrupted(m)) if m.contains("block length")),
                "{got:?}"
            );
        }
        for block_len in [-1i32, i32::MIN, i32::MAX] {
            let mut data = Vec::new();
            write_vint(&mut data, 5);
            data.extend_from_slice(b"apple");
            write_vint(&mut data, block_len);
            data.extend_from_slice(&[0u8; 8]);

            let entry = TermsDictEntry {
                terms_dict_size: 3,
                max_term_length: 11,
                max_block_length: 8192,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
                index: None,
            };
            let got = decode_all_terms(&data, &entry);
            assert!(got.is_err(), "block_len={block_len}: {got:?}");
        }
    }

    /// The prefix/suffix continuation vints size the term being built. A
    /// negative one overflowed the `+=` in a debug build and reached
    /// `Vec::with_capacity` as an aborting allocation otherwise.
    #[test]
    fn corrupt_term_length_extension_is_a_decode_error() {
        for extension in [-1i32, i32::MIN, i32::MAX] {
            for token in [0x0Fu8, 0xF0 | 0x04] {
                let mut block_body = Vec::new();
                block_body.push(token);
                write_vint(&mut block_body, extension);
                block_body.extend_from_slice(b"xyz");

                let mut data = Vec::new();
                write_vint(&mut data, 5);
                data.extend_from_slice(b"apple");
                write_vint(&mut data, block_body.len() as i32);
                data.extend_from_slice(&encode_literal_lz4(&block_body));

                let entry = TermsDictEntry {
                    terms_dict_size: 2,
                    max_term_length: 11,
                    max_block_length: 8192,
                    terms_data_offset: 0,
                    terms_data_length: data.len() as i64,
                    index: None,
                };
                let got = decode_all_terms(&data, &entry);
                assert!(
                    got.is_err(),
                    "extension={extension} token={token:#x}: {got:?}"
                );
            }
        }
    }

    fn write_vint_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_direct_monotonic_meta(out: &mut Vec<u8>, num_values: i64) {
        let block_shift = 2u32;
        let mut num_blocks = num_values >> block_shift;
        if (num_blocks << block_shift) < num_values {
            num_blocks += 1;
        }
        for _ in 0..num_blocks {
            out.extend_from_slice(&0i64.to_le_bytes()); // min
            out.extend_from_slice(&0i32.to_le_bytes()); // avg
            out.extend_from_slice(&0i64.to_le_bytes()); // offset
            out.push(0); // bpv
        }
    }
}
