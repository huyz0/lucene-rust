//! `.doc` decoding for the retired postings formats: ports of the block
//! layouts `backward_codecs.lucene90.Lucene90PostingsReader`,
//! `lucene99.Lucene99PostingsReader`, `lucene912.Lucene912PostingsReader` and
//! `lucene101.Lucene101PostingsReader` (with `lucene103`, which reads the
//! same `.doc` as `Lucene101` version 1) walk.
//!
//! Every generation writes a term's documents as zero or more full 128-doc
//! blocks followed by a variable-length tail for the `docFreq % 128`
//! remainder, and a singleton (`docFreq == 1`) not at all -- its document is
//! pulsed into the term dictionary. They differ in how a block is framed:
//!
//! | format | doc deltas of a full block | skip data | tail |
//! |---|---|---|---|
//! | `Lucene90` | `PForUtil` (with exceptions), then prefix sum | after the term's postings (`skipOffset`) | vint `docDelta << 1 \| freq == 1`, freq vint inline |
//! | `Lucene99` | `ForDeltaUtil`: a width byte (0 = all deltas 1), `ForUtil` | after the term's postings | group-varint deltas, then the freqs != 1 |
//! | `Lucene912` | `ForDeltaUtil` with its own primitive thresholds | inline: a level-1 entry every 32 blocks, a level-0 header per block | group-varint |
//! | `Lucene101` | a signed width byte: `> 0` `ForDeltaUtil` on `int`s, `0` consecutive, `< 0` a bit set of that many `long`s | inline, as `Lucene912` | group-varint |
//!
//! Frequencies of a full block are always one `PForUtil` block right after
//! the doc deltas, when the field indexes them.
//!
//! This module decodes a term's blocks **front to back**, the way
//! `BlockDocsEnum.nextDoc`/`BlockPostingsEnum.nextDoc` visit them: here the
//! skip data is stepped over (inline) or not touched (trailing). Jumping is
//! the lazy cursors' job: `crate::postings::LazyDocsCursor` reads these
//! formats block by block and advances through each one's own skip data --
//! the trailing multi-level list of `Lucene90`/`Lucene99`
//! ([`super::skip_list`]) or the inline levels of `Lucene912`/`Lucene101` --
//! and serves its impacts. See that module's `PostingsFormat` for the seam.

use lucene_store::data_input::{DataInput, SliceInput};

use super::for_util::{self, Primitives, Word, BLOCK_SIZE};
use crate::field_infos::IndexOptions;
use crate::postings::{Error, Postings, PostingsFormat, Result, TermMetadata};

/// `LEVEL1_FACTOR * BLOCK_SIZE` for the inline-skip generations
/// (`Lucene912PostingsFormat.LEVEL1_NUM_DOCS`,
/// `Lucene101PostingsFormat.LEVEL1_NUM_DOCS`).
const LEVEL1_NUM_DOCS: usize = 32 * BLOCK_SIZE;

fn corrupted(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// Decodes every `(doc, freq)` of a term with `doc_freq > 1` from `doc`
/// (the whole `.doc` file) in `format`. Frequencies are all `1` for a field
/// without them, like `BlockDocsEnum.freq()`.
// ARITH: `consumed = n - left` with `left <= n`; the modulo and the
// `left -= BLOCK_SIZE` are by and under the loop's own constant bounds.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn read_postings(
    doc: &[u8],
    format: PostingsFormat,
    meta: TermMetadata,
    doc_freq: i32,
    index_options: IndexOptions,
) -> Result<Postings> {
    if doc_freq <= 1 {
        return Err(Error::Unsupported(
            "docFreq <= 1: use singleton_postings instead (no .doc bytes are written)",
        ));
    }
    let index_has_freq = index_options != IndexOptions::Docs;
    let n = doc_freq as usize;
    let mut r = SliceInput::new(doc);
    let start = usize::try_from(meta.doc_start_fp)
        .map_err(|_| corrupted(format!("docStartFP {} out of range", meta.doc_start_fp)))?;
    r.seek(start)?;
    // Every document costs at least one byte of `.doc`: a corrupt docFreq
    // must not size the allocation beyond what the file could hold.
    let reserve = n.min(doc.len());
    let mut docs: Vec<i32> = Vec::with_capacity(reserve);
    let mut freqs: Vec<i32> = Vec::with_capacity(reserve);

    // The delta base before the first document: `Lucene90`/`Lucene99`
    // (`BlockDocsEnum.accum = 0`, the writer's `lastDocID = 0`) code the
    // first document as its own id; `Lucene912` on (`prevDocID = -1`) as one
    // more than it.
    let mut prev_doc: i64 = match format {
        PostingsFormat::Lucene90 | PostingsFormat::Lucene99 => 0,
        _ => -1,
    };
    let mut left = n;
    let mut block = [0u64; BLOCK_SIZE];
    let inline_skips = matches!(
        format,
        PostingsFormat::Lucene912 | PostingsFormat::Lucene101 | PostingsFormat::Lucene103
    );
    while left >= BLOCK_SIZE {
        if inline_skips {
            let consumed = n - left;
            if n >= LEVEL1_NUM_DOCS
                && consumed.is_multiple_of(LEVEL1_NUM_DOCS)
                && left >= LEVEL1_NUM_DOCS
            {
                // `skipLevel1To`: vint docDelta, vlong byte length of the
                // span, then (with freqs) a short-prefixed run of impacts and
                // `.pos`/`.pay` pointers.
                r.read_vint()?;
                r.read_vlong()?;
                if index_has_freq {
                    let len = i64::from(r.read_u16()? as i16);
                    let len = usize::try_from(len)
                        .map_err(|_| corrupted(format!("level-1 skip length {len}")))?;
                    r.skip(len)?;
                }
            }
            // `moveToNextLevel0Block`'s docs-and-freqs path: the level-0
            // header's own byte length, and everything it covers.
            let num_bytes = r.read_vlong()?;
            let num_bytes = usize::try_from(num_bytes)
                .map_err(|_| corrupted(format!("level-0 skip length {num_bytes}")))?;
            r.skip(num_bytes)?;
        }
        decode_full_doc_block(&mut r, format, prev_doc, &mut block)?;
        for &d in block.iter() {
            docs.push(d as i32);
        }
        prev_doc = i64::from(block[BLOCK_SIZE - 1] as i32);
        if index_has_freq {
            let word = format.word();
            for_util::pfor_decode(&mut r, word, &mut block)?;
            freqs.extend(block.iter().map(|&f| f as i32));
        } else {
            freqs.extend(std::iter::repeat_n(1, BLOCK_SIZE));
        }
        left -= BLOCK_SIZE;
    }

    if left > 0 {
        read_tail(
            &mut r,
            format,
            left,
            index_has_freq,
            prev_doc,
            &mut docs,
            &mut freqs,
        )?;
    }
    Ok(Postings {
        docs,
        freqs,
        level0_impacts: Vec::new(),
        level1_impacts: Vec::new(),
    })
}

/// One full block of an inline-skip generation (`Lucene912`, `Lucene101`,
/// `Lucene103`) or a trailing-skip one (`Lucene90`, `Lucene99`), decoded into
/// the first [`BLOCK_SIZE`] slots of a [`crate::postings::LazyDocsCursor`]'s
/// block arrays: `refillFullBlock`'s doc deltas (the generation's encoding,
/// bit-set blocks expanded), then the frequency block -- decoded when
/// `needs_freq`, stepped over otherwise (`PForUtil.skip`), and all ones for a
/// field without frequencies.
pub(crate) fn decode_block_body(
    r: &mut SliceInput,
    format: PostingsFormat,
    prev_doc: i32,
    index_has_freq: bool,
    needs_freq: bool,
    docs: &mut [i32],
    freqs: &mut [i32],
) -> Result<()> {
    let mut block = [0u64; BLOCK_SIZE];
    // `Lucene90`/`Lucene99` delta-code from the previous block's last document
    // as well; only the first document's base differs, and the cursor passes
    // that in.
    decode_full_doc_block(r, format, i64::from(prev_doc), &mut block)?;
    for (d, &v) in docs.iter_mut().zip(block.iter()) {
        *d = v as i32;
    }
    if index_has_freq && needs_freq {
        for_util::pfor_decode(r, format.word(), &mut block)?;
        for (f, &v) in freqs.iter_mut().zip(block.iter()) {
            *f = v as i32;
        }
    } else {
        if index_has_freq {
            for_util::pfor_skip(r, format.word())?;
        }
        for f in freqs.iter_mut().take(BLOCK_SIZE) {
            *f = 1;
        }
    }
    Ok(())
}

/// `Lucene90PostingsReader.readVIntBlock` into a lazy cursor's block: the
/// `docFreq % 128` remainder of a `Lucene90` term, one vint per document
/// (`docDelta << 1 | freq == 1` when the field has frequencies, then the
/// frequency itself when it is not 1).
// ARITH: `code >> 1` of an unsigned code cannot overflow; doc ids accumulate
// with `wrapping_add`, as Java's `int` sum wraps on a corrupt file.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn read_tail_block_90(
    r: &mut SliceInput,
    prev_doc: i32,
    index_has_freq: bool,
    docs: &mut [i32],
    freqs: &mut [i32],
) -> Result<()> {
    let mut acc = prev_doc;
    for (d, f) in docs.iter_mut().zip(freqs.iter_mut()) {
        let code = r.read_vint()? as u32;
        let (delta, freq) = if index_has_freq {
            let freq = if code & 1 != 0 { 1 } else { r.read_vint()? };
            (code >> 1, freq)
        } else {
            (code, 1)
        };
        acc = acc.wrapping_add(delta as i32);
        *d = acc;
        *f = freq;
    }
    Ok(())
}

/// One full block's 128 doc ids, absolute, from the previous block's last
/// id `prev_doc`.
fn decode_full_doc_block(
    r: &mut SliceInput,
    format: PostingsFormat,
    prev_doc: i64,
    out: &mut [u64; BLOCK_SIZE],
) -> Result<()> {
    match format {
        PostingsFormat::Lucene90 => {
            // `PForUtil.decodeAndPrefixSum`: the deltas are an ordinary
            // `PForUtil` block (exceptions included).
            for_util::pfor_decode(r, Word::Long, out)?;
            for_util::prefix_sum(out, prev_doc);
        }
        PostingsFormat::Lucene99 | PostingsFormat::Lucene912 => {
            // `ForDeltaUtil.decodeAndPrefixSum`: width byte, 0 = all ones.
            let bpv = u32::from(r.read_byte()?);
            if bpv == 0 {
                out.fill(1);
            } else {
                let prims: Primitives = if format == PostingsFormat::Lucene99 {
                    for_util::FOR_UTIL
                } else {
                    for_util::LUCENE912_DELTAS
                };
                for_util::for_decode(r, bpv, Word::Long, prims, out)?;
            }
            for_util::prefix_sum(out, prev_doc);
        }
        PostingsFormat::Lucene101 | PostingsFormat::Lucene103 => {
            // `refillFullBlock`: a signed width byte.
            let bpv = r.read_byte()? as i8;
            if bpv > 0 {
                for_util::for_decode(r, bpv as u32, Word::Int, for_util::LUCENE101_DELTAS, out)?;
                for_util::prefix_sum(out, prev_doc);
            } else {
                // A bit set of the block's documents, based at `prev + 1`:
                // `0` means all 128 are consecutive (two all-ones words).
                let num_longs = if bpv == 0 {
                    BLOCK_SIZE / 64
                } else {
                    usize::from(bpv.unsigned_abs())
                };
                let base = prev_doc.wrapping_add(1);
                let mut n = 0usize;
                for w in 0..num_longs {
                    let word = if bpv == 0 {
                        u64::MAX
                    } else {
                        r.read_i64()? as u64
                    };
                    let mut bits = word;
                    while bits != 0 {
                        let b = bits.trailing_zeros() as i64;
                        if n >= BLOCK_SIZE {
                            return Err(corrupted("bit-set doc block holds more than 128 docs"));
                        }
                        // ARITH: `w < 128` and `b < 64`.
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            out[n] = base.wrapping_add(w as i64 * 64 + b) as u64;
                            n += 1;
                        }
                        bits &= bits.wrapping_sub(1);
                    }
                }
                if n != BLOCK_SIZE {
                    return Err(corrupted(format!(
                        "bit-set doc block holds {n} docs, not 128"
                    )));
                }
            }
        }
        PostingsFormat::Lucene104 => {
            return Err(Error::Unsupported(
                "Lucene104 blocks are read by crate::postings",
            ))
        }
    }
    Ok(())
}

/// The `docFreq % 128` remainder (`readVIntBlock`).
// ARITH: `code >> 1` of an unsigned code cannot overflow; doc ids accumulate
// with `wrapping_add`, as Java's `int` sum wraps on a corrupt file.
#[allow(clippy::arithmetic_side_effects)]
fn read_tail(
    r: &mut SliceInput,
    format: PostingsFormat,
    count: usize,
    index_has_freq: bool,
    prev_doc: i64,
    docs: &mut Vec<i32>,
    freqs: &mut Vec<i32>,
) -> Result<()> {
    let mut acc = prev_doc;
    if format == PostingsFormat::Lucene90 {
        // `Lucene90PostingsReader.readVIntBlock`: interleaved vints.
        for _ in 0..count {
            let code = r.read_vint()? as u32;
            let (delta, freq) = if index_has_freq {
                let freq = if code & 1 != 0 { 1 } else { r.read_vint()? };
                (code >> 1, freq)
            } else {
                (code, 1)
            };
            acc = acc.wrapping_add(i64::from(delta));
            docs.push(acc as i32);
            freqs.push(freq);
        }
        return Ok(());
    }
    // `PostingsUtil.readVIntBlock`: group-varint codes, then the freqs that
    // are not 1, in order.
    let mut codes = vec![0u64; count];
    r.read_group_vints(&mut codes)?;
    for code in codes {
        let (delta, freq) = if index_has_freq {
            let freq = if code & 1 != 0 { 1 } else { r.read_vint()? };
            (code >> 1, freq)
        } else {
            (code, 1)
        };
        acc = acc.wrapping_add(delta as i64);
        docs.push(acc as i32);
        freqs.push(freq);
    }
    Ok(())
}

/// Every position of a term, as the per-occurrence deltas `.pos` stores
/// (each document's first delta is from 0), without touching `.pay`: the
/// full 128-occurrence blocks are `PForUtil` blocks of position deltas only
/// -- their payloads and offsets live in `.pay` -- and the vint tail's
/// inline payload bytes and offsets are stepped over. This is what a
/// positions-only reader of a retired format needs
/// (`crate::postings::PositionsCursor`, a phrase's positions).
// ARITH: `n / BLOCK_SIZE` and `n % BLOCK_SIZE` by a non-zero constant, and
// `code >> 1` of an unsigned code.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn read_position_deltas(
    pos: &[u8],
    format: PostingsFormat,
    meta: TermMetadata,
    total_term_freq: i64,
    has_offsets: bool,
    has_payloads: bool,
) -> Result<Vec<i32>> {
    let n = usize::try_from(total_term_freq)
        .map_err(|_| corrupted(format!("negative totalTermFreq {total_term_freq}")))?;
    let mut r = SliceInput::new(pos);
    let start = usize::try_from(meta.pos_start_fp)
        .map_err(|_| corrupted(format!("posStartFP {} out of range", meta.pos_start_fp)))?;
    r.seek(start)?;
    let mut out = Vec::with_capacity(n.min(pos.len()));
    let (full, tail) = (n / BLOCK_SIZE, n % BLOCK_SIZE);
    let mut block = [0u64; BLOCK_SIZE];
    for _ in 0..full {
        for_util::pfor_decode(&mut r, format.word(), &mut block)?;
        out.extend(block.iter().map(|&d| d as i32));
    }
    // `refillLastPositionBlock` / `refillPositions`' vint branch.
    let mut payload_length = 0i32;
    for _ in 0..tail {
        let code = r.read_vint()?;
        if has_payloads {
            if code & 1 != 0 {
                payload_length = r.read_vint()?;
            }
            out.push(((code as u32) >> 1) as i32);
            if payload_length != 0 {
                let len = usize::try_from(payload_length)
                    .map_err(|_| corrupted(format!("payload length {payload_length}")))?;
                r.skip(len)?;
            }
        } else {
            out.push(code);
        }
        if has_offsets {
            let delta_code = r.read_vint()?;
            if delta_code & 1 != 0 {
                r.read_vint()?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn vint(out: &mut Vec<u8>, mut v: u32) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    fn meta() -> TermMetadata {
        TermMetadata::EMPTY
    }

    #[test]
    fn lucene90_tail_interleaves_freqs() {
        // docs 3 (freq 1), 5 (freq 4): the first delta is from 0.
        let mut b = Vec::new();
        vint(&mut b, (3 << 1) | 1);
        vint(&mut b, 2 << 1);
        vint(&mut b, 4);
        let p = read_postings(
            &b,
            PostingsFormat::Lucene90,
            meta(),
            2,
            IndexOptions::DocsAndFreqs,
        )
        .unwrap();
        assert_eq!(p.docs, vec![3, 5]);
        assert_eq!(p.freqs, vec![1, 4]);
        // Docs only: plain deltas, freqs 1.
        let b = [3u8, 2];
        let p = read_postings(&b, PostingsFormat::Lucene90, meta(), 2, IndexOptions::Docs).unwrap();
        assert_eq!(p.docs, vec![3, 5]);
        assert_eq!(p.freqs, vec![1, 1]);
    }

    #[test]
    fn group_varint_tail_puts_freqs_after_the_codes() {
        // Two codes with a plain vint tail (fewer than 4 -> no groups):
        // deltas 1 (freq 3) and 7 (freq 1). `Lucene99` codes the first
        // document from 0, the later formats from -1.
        let mut b = Vec::new();
        vint(&mut b, 1 << 1);
        vint(&mut b, (7 << 1) | 1);
        vint(&mut b, 3);
        for (f, first) in [
            (PostingsFormat::Lucene99, 1),
            (PostingsFormat::Lucene912, 0),
            (PostingsFormat::Lucene101, 0),
        ] {
            let p = read_postings(&b, f, meta(), 2, IndexOptions::DocsAndFreqs).unwrap();
            assert_eq!(p.docs, vec![first, first + 7]);
            assert_eq!(p.freqs, vec![3, 1]);
        }
    }

    #[test]
    fn lucene101_consecutive_and_bitset_blocks() {
        // A 128-doc consecutive block (width 0) with a constant-freq PFor
        // (token 0, vint 2), then nothing: docFreq 128.
        let mut b = Vec::new();
        vint(&mut b, 0); // level-0 header byte length: empty
        b.push(0); // consecutive
        b.push(0); // PFor token: bpv 0, no exceptions
        vint(&mut b, 2);
        let p = read_postings(
            &b,
            PostingsFormat::Lucene101,
            meta(),
            128,
            IndexOptions::DocsAndFreqs,
        )
        .unwrap();
        assert_eq!(p.docs, (0..128).collect::<Vec<_>>());
        assert!(p.freqs.iter().all(|&f| f == 2));

        // A bit set of 3 longs: bits 0..128 of words 0,2 and none of 1 --
        // docs 0..64 and 128..192.
        let mut b = Vec::new();
        vint(&mut b, 0);
        b.push((-3i8) as u8);
        b.extend(u64::MAX.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(u64::MAX.to_le_bytes());
        let p = read_postings(
            &b,
            PostingsFormat::Lucene101,
            meta(),
            128,
            IndexOptions::Docs,
        )
        .unwrap();
        assert_eq!(p.docs[63], 63);
        assert_eq!(p.docs[64], 128);
        assert_eq!(p.docs[127], 191);

        // A bit set with the wrong number of documents is corrupt.
        let mut b = Vec::new();
        vint(&mut b, 0);
        b.push((-1i8) as u8);
        b.extend(u64::MAX.to_le_bytes());
        assert!(read_postings(
            &b,
            PostingsFormat::Lucene101,
            meta(),
            128,
            IndexOptions::Docs
        )
        .is_err());
    }

    #[test]
    fn singletons_and_current_format_are_not_decoded_here() {
        assert!(
            read_postings(&[], PostingsFormat::Lucene90, meta(), 1, IndexOptions::Docs).is_err()
        );
        let b = vec![0u8; 400];
        assert!(read_postings(
            &b,
            PostingsFormat::Lucene104,
            meta(),
            300,
            IndexOptions::Docs
        )
        .is_err());
    }
}
