//! Port of `org.apache.lucene.codecs.lucene104.Lucene104PostingsReader`'s `.doc`
//! file decode — read-only, scoped to **a single postings block** (fewer than
//! `ForUtil.BLOCK_SIZE` = 256 postings for the term, so no skip data / no PFOR
//! bulk block ever applies) and **`IndexOptions.DOCS`/`DOCS_AND_FREQS`** (no
//! positions/offsets/payloads).
//!
//! ## Why this scope decodes cleanly
//!
//! `Lucene104PostingsWriter.finishTerm` special-cases `docFreq == 1` by pulsing
//! the single doc ID into the term dictionary itself (see
//! `Lucene104PostingsWriter.java:568-577`): no bytes are written to `.doc` at
//! all for a singleton term. For `1 < docFreq < BLOCK_SIZE`,
//! `flushDocBlock(true)` never reaches the packed-int/bit-set branch (that
//! path only runs when `docBufferUpto == BLOCK_SIZE`,
//! `Lucene104PostingsWriter.java:392-461`) — instead it takes the
//! `PostingsUtil.writeVIntBlock` branch (`Lucene104PostingsWriter.java:394-395`),
//! a much simpler group-varint + trailing-vint-freq-exceptions encoding with no
//! skip data, no impacts, and no `ForUtil`/`PForUtil` bit-packing at all. Both
//! `ForUtil`/`PForUtil` (256-wide PFOR bulk (de)coding) and the level-0/level-1
//! skip-list machinery only ever engage once a term has >= `BLOCK_SIZE` (256)
//! postings — entirely out of scope here, see `docs/parity.md`.
//!
//! ## Wire format (docFreq > 1 case)
//!
//! At `TermMetadata::doc_start_fp` in the `.doc` file
//! (`Lucene104PostingsFormat.DOC_EXTENSION`, `IndexHeader(codec="Lucene104PostingsWriterDoc")`):
//! `docFreq` group-varint-encoded values (`GroupVIntUtil`/`DataInput::read_group_vints`,
//! already ported in `lucene-store`), each packing `(docDelta << 1) |
//! (freq == 1 ? 1 : 0)` when the field has freqs (`PostingsUtil.java:39-52`), or
//! plain `docDelta` when it doesn't (`IndexOptions::Docs`). Immediately after,
//! in doc order, one plain vint per doc whose packed bit was 0 (i.e. freq != 1)
//! carries that doc's actual freq. Doc IDs are delta-coded from a base of `-1`
//! (`Lucene104PostingsReader.prefixSum`, `Lucene104PostingsReader.java:194-200`,
//! called with `prevDocID == -1` for a term's first block,
//! `Lucene104PostingsReader.java:555`).
//!
//! ## Per-term metadata (`decodeTerm`)
//!
//! The blocktree term dictionary's per-term metadata bytes (previously skipped
//! by `blocktree.rs`, see its module doc) encode `Lucene104PostingsReader.decodeTerm`
//! (`Lucene104PostingsReader.java:213-251`), scoped here to the no-positions
//! case: one vlong whose low bit selects between an absolute-ish `docStartFP`
//! delta (bit clear — `termState.docStartFP += l >>> 1`, plus a raw vint
//! `singletonDocID` when `docFreq == 1`) or a zigzag `singletonDocID` delta
//! relative to the *previous term in the same block* (bit set — only legal for
//! a non-absolute decode, i.e. not the first term after a block load; see
//! `SegmentTermsEnumFrame.java:471,506,509`: `absolute = metaDataUpto == 0`).
//!
//! ## Deferred (all rejected with [`Error::Unsupported`])
//!
//! Multi-block terms (`docFreq >= BLOCK_SIZE`, needs `ForUtil`/`PForUtil` PFOR
//! bulk decode plus level-0/level-1 skip-list traversal), positions/offsets/payloads
//! (`.pos`/`.pay`, `IndexOptions::DocsAndFreqsAndPositions` and up), and impacts
//! (`ImpactsEnum`, competitive-scoring metadata) — see `docs/parity.md`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::field_infos::IndexOptions;

/// `Lucene104PostingsFormat.DOC_CODEC`.
const DOC_CODEC: &str = "Lucene104PostingsWriterDoc";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;
/// `ForUtil.BLOCK_SIZE` (== `Lucene104PostingsFormat.BLOCK_SIZE`).
pub const BLOCK_SIZE: i32 = 256;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("decodeTerm: singleton-delta bit set on an absolute (first-in-block) decode")]
    AbsoluteSingletonDelta,
    #[error("decodeTerm: singleton-delta bit set but no previous singleton to delta from")]
    NoPreviousSingleton,
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Per-term postings location, decoded from the blocktree's per-term metadata
/// bytes (`Lucene104PostingsReader.decodeTerm`, no-positions subset). `-1` for
/// `singleton_doc_id` means "not a singleton" (`docFreq > 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TermMetadata {
    pub doc_start_fp: u64,
    pub singleton_doc_id: i32,
}

impl TermMetadata {
    /// `IntBlockTermState`'s "empty" starting state (`EMPTY_STATE` /
    /// `absolute == true` semantics): zero `docStartFP`, no singleton yet.
    pub const EMPTY: TermMetadata = TermMetadata {
        doc_start_fp: 0,
        singleton_doc_id: -1,
    };
}

/// `Lucene104PostingsReader.decodeTerm`, restricted to fields with no
/// positions (`IndexOptions::Docs`/`DocsAndFreqs`) — the `posStartFP`/
/// `payStartFP`/`lastPosBlockOffset` fields never appear on the wire for
/// those. `absolute` mirrors `SegmentTermsEnumFrame`'s `metaDataUpto == 0`:
/// true only for the first term decoded after loading a `.tim` block, false
/// for every subsequent term in that same block (deltas are relative to the
/// previous term's decoded state, `prev`).
pub fn decode_term_metadata(
    r: &mut SliceInput,
    doc_freq: i32,
    absolute: bool,
    prev: TermMetadata,
) -> Result<TermMetadata> {
    let l = r.read_vlong()? as u64;
    if l & 1 == 0 {
        let doc_start_fp = prev.doc_start_fp.wrapping_add(l >> 1);
        let singleton_doc_id = if doc_freq == 1 { r.read_vint()? } else { -1 };
        Ok(TermMetadata {
            doc_start_fp,
            singleton_doc_id,
        })
    } else {
        if absolute {
            return Err(Error::AbsoluteSingletonDelta);
        }
        if prev.singleton_doc_id == -1 {
            return Err(Error::NoPreviousSingleton);
        }
        let delta = lucene_util::zigzag::decode(l >> 1);
        let singleton_doc_id = (prev.singleton_doc_id as i64 + delta) as i32;
        Ok(TermMetadata {
            doc_start_fp: prev.doc_start_fp,
            singleton_doc_id,
        })
    }
}

/// One term's decoded `(docID, freq)` pairs, in ascending doc-ID order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Postings {
    pub docs: Vec<i32>,
    pub freqs: Vec<i32>,
}

/// An opened `.doc` file (header/footer validated once), ready for
/// per-term seeks. Mirrors `Lucene104PostingsReader`'s `docIn`, minus
/// everything this slice doesn't support (positions, skip data, impacts).
pub struct DocInput<'a> {
    buf: &'a [u8],
}

impl<'a> DocInput<'a> {
    /// Validates the `.doc` file's index header and footer checksum framing
    /// (`Lucene104PostingsReader`'s constructor, `Lucene104PostingsReader.java:134-140`).
    pub fn open(doc: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        let mut r = SliceInput::new(doc);
        codec_util::check_index_header(
            &mut r,
            DOC_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        codec_util::retrieve_checksum(doc)?;
        Ok(DocInput { buf: doc })
    }

    /// Decodes a term's `(docID, freq)` pairs for `1 < docFreq < BLOCK_SIZE`
    /// (`PostingsUtil.readVIntBlock` + `Lucene104PostingsReader.prefixSum`,
    /// via `BlockPostingsEnum.refillRemainder`'s non-singleton branch,
    /// `Lucene104PostingsReader.java:647-656`). Singletons (`docFreq == 1`)
    /// need no file access at all — see [`singleton_postings`] — and
    /// `docFreq >= BLOCK_SIZE` needs skip data/PFOR this slice doesn't
    /// implement.
    pub fn read_postings(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
    ) -> Result<Postings> {
        if doc_freq <= 1 {
            return Err(Error::Unsupported(
                "docFreq <= 1: use singleton_postings instead (no .doc bytes are written)",
            ));
        }
        if doc_freq >= BLOCK_SIZE {
            return Err(Error::Unsupported(
                "docFreq >= BLOCK_SIZE: multi-block postings (skip data/PFOR) not supported in this slice",
            ));
        }
        if !matches!(
            index_options,
            IndexOptions::Docs | IndexOptions::DocsAndFreqs
        ) {
            return Err(Error::Unsupported(
                "only IndexOptions::Docs/DocsAndFreqs are supported in this slice",
            ));
        }
        let index_has_freq = index_options == IndexOptions::DocsAndFreqs;

        let mut r = SliceInput::new(self.buf);
        r.seek(meta.doc_start_fp as usize)?;

        let n = doc_freq as usize;
        let mut raw = vec![0u64; n];
        r.read_group_vints(&mut raw)?;

        let mut docs = vec![0i32; n];
        let mut freqs = vec![1i32; n];
        if index_has_freq {
            for ((d, f), &v) in docs.iter_mut().zip(freqs.iter_mut()).zip(raw.iter()) {
                *f = (v & 1) as i32;
                *d = (v >> 1) as i32;
            }
            for f in freqs.iter_mut() {
                if *f == 0 {
                    *f = r.read_vint()?;
                }
            }
        } else {
            for (d, &v) in docs.iter_mut().zip(raw.iter()) {
                *d = v as i32;
            }
        }

        // prefixSum(docBuffer, docCountLeft, prevDocID = -1): a term's first
        // block always starts from prevDocID == -1 in this slice (no earlier
        // block ever precedes it, since docFreq < BLOCK_SIZE).
        let mut sum: i64 = -1;
        for d in docs.iter_mut() {
            sum += *d as i64;
            *d = sum as i32;
        }

        Ok(Postings { docs, freqs })
    }
}

/// `docFreq == 1`: the single doc/freq is reconstructed entirely from the
/// term dictionary's metadata (`termState.singletonDocID`) and
/// `totalTermFreq` (implicitly the one doc's freq) — no `.doc` file access,
/// matching `BlockPostingsEnum.refillRemainder`'s singleton branch
/// (`Lucene104PostingsReader.java:640-646`).
pub fn singleton_postings(meta: TermMetadata, total_term_freq: i64) -> Result<Postings> {
    if meta.singleton_doc_id < 0 {
        return Err(Error::NoPreviousSingleton);
    }
    Ok(Postings {
        docs: vec![meta.singleton_doc_id],
        freqs: vec![total_term_freq as i32],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_output::DataOutput;

    /// Test-only encoder for `GroupVIntUtil.writeGroupVInts`'s wire format
    /// (groups of 4 values, 1 flag byte packing each value's byte-length minus
    /// one, then that many little-endian bytes per value; a final partial
    /// group of fewer than 4 falls back to plain vints) — mirrors this
    /// project's pattern of small test-only encoders (see `data_input.rs`'s
    /// own tests) rather than adding a writer this port doesn't otherwise need
    /// yet.
    fn write_group_vints(out: &mut Vec<u8>, values: &[u32]) {
        let mut i = 0;
        while i + 4 <= values.len() {
            let chunk = &values[i..i + 4];
            let lens: Vec<u8> = chunk
                .iter()
                .map(|&v| {
                    let bytes = if v == 0 {
                        1
                    } else {
                        4 - (v.leading_zeros() / 8)
                    };
                    (bytes - 1) as u8
                })
                .collect();
            let flag = (lens[0] << 6) | (lens[1] << 4) | (lens[2] << 2) | lens[3];
            out.push(flag);
            for (j, &v) in chunk.iter().enumerate() {
                let n = lens[j] as usize + 1;
                out.extend_from_slice(&v.to_le_bytes()[..n]);
            }
            i += 4;
        }
        while i < values.len() {
            out.write_vint(values[i] as i32);
            i += 1;
        }
    }

    fn header_and_footer(codec: &str, id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
        let mut before = Vec::new();
        codec_util::write_index_header(&mut before, codec, VERSION_CURRENT, id, "");
        let mut after = Vec::new();
        codec_util::write_footer(&mut after);
        (before, after)
    }

    #[test]
    fn open_rejects_bad_header() {
        let id = [1u8; ID_LENGTH];
        let mut doc = Vec::new();
        codec_util::write_index_header(&mut doc, "WrongCodec", VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut doc);
        assert!(DocInput::open(&doc, &id, "").is_err());
    }

    #[test]
    fn read_postings_two_docs_with_freqs() {
        let id = [2u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // docFreq=2: deltas [3, 2] (docIDs 2 and 4), freqs [2, 1].
        // group-varint packing: (delta<<1)|(freq==1?1:0)
        write_group_vints(&mut doc, &[3 << 1, (2 << 1) | 1]);
        doc.write_vint(2); // explicit freq for the first doc (freq != 1)
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
        };
        let postings = input
            .read_postings(meta, 2, IndexOptions::DocsAndFreqs)
            .unwrap();
        assert_eq!(postings.docs, vec![2, 4]);
        assert_eq!(postings.freqs, vec![2, 1]);
    }

    #[test]
    fn read_postings_docs_only_no_freqs() {
        let id = [3u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // docFreq=3, plain deltas (no freq bit-packing): docIDs 0,1,5 -> deltas 1,1,4
        write_group_vints(&mut doc, &[1, 1, 4]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
        };
        let postings = input.read_postings(meta, 3, IndexOptions::Docs).unwrap();
        assert_eq!(postings.docs, vec![0, 1, 5]);
        assert_eq!(postings.freqs, vec![1, 1, 1]);
    }

    #[test]
    fn read_postings_all_freq_one_docs_only_bit_path() {
        // Every doc has freq==1 (bit set), so no trailing freq vints at all --
        // exercises the branch where the second (freq-exception) loop in
        // `read_postings` never fires.
        let id = [6u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // docIDs 0, 3, 4 (deltas 1, 3, 1), freq==1 for all -> bit always set.
        write_group_vints(&mut doc, &[(1 << 1) | 1, (3 << 1) | 1, (1 << 1) | 1]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
        };
        let postings = input
            .read_postings(meta, 3, IndexOptions::DocsAndFreqs)
            .unwrap();
        assert_eq!(postings.docs, vec![0, 3, 4]);
        assert_eq!(postings.freqs, vec![1, 1, 1]);
    }

    #[test]
    fn read_postings_block_size_minus_one_docs() {
        // docFreq == BLOCK_SIZE - 1 (255): the largest docFreq this slice's
        // group-varint (non-PFOR) path supports -- one below the boundary
        // where `read_postings` rejects with Unsupported.
        let id = [7u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let n = (BLOCK_SIZE - 1) as usize;
        // Consecutive doc IDs 0..n, delta=1 each, freq==2 for every doc (bit
        // clear) so every doc also needs a trailing freq vint.
        let deltas: Vec<u32> = (0..n).map(|_| 1u32 << 1).collect();
        write_group_vints(&mut doc, &deltas);
        for _ in 0..n {
            doc.write_vint(2);
        }
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
        };
        let postings = input
            .read_postings(meta, n as i32, IndexOptions::DocsAndFreqs)
            .unwrap();
        assert_eq!(postings.docs, (0..n as i32).collect::<Vec<_>>());
        assert!(postings.freqs.iter().all(|&f| f == 2));
        assert_eq!(postings.freqs.len(), n);
    }

    #[test]
    fn read_postings_rejects_singleton_doc_freq() {
        let id = [4u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        doc.extend_from_slice(&footer);
        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: 7,
        };
        let err = input
            .read_postings(meta, 1, IndexOptions::Docs)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn read_postings_rejects_multi_block_doc_freq() {
        let id = [5u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        doc.extend_from_slice(&footer);
        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: -1,
        };
        let err = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::DocsAndFreqs)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn singleton_postings_reconstructs_from_metadata() {
        let meta = TermMetadata {
            doc_start_fp: 123,
            singleton_doc_id: 9,
        };
        let postings = singleton_postings(meta, 4).unwrap();
        assert_eq!(postings.docs, vec![9]);
        assert_eq!(postings.freqs, vec![4]);
    }

    #[test]
    fn singleton_postings_rejects_non_singleton_metadata() {
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: -1,
        };
        assert!(singleton_postings(meta, 1).is_err());
    }

    #[test]
    fn decode_term_metadata_absolute_then_delta_docstart() {
        let mut bytes = Vec::new();
        // absolute: docStartFP delta=10 (l = 10<<1 = 20), docFreq>1 so no singleton vint
        bytes.write_vlong(20);
        // second term in same block: docStartFP delta=5 (l = 5<<1 = 10)
        bytes.write_vlong(10);
        let mut r = SliceInput::new(&bytes);

        let first = decode_term_metadata(&mut r, 2, true, TermMetadata::EMPTY).unwrap();
        assert_eq!(first.doc_start_fp, 10);
        assert_eq!(first.singleton_doc_id, -1);

        let second = decode_term_metadata(&mut r, 2, false, first).unwrap();
        assert_eq!(second.doc_start_fp, 15);
    }

    #[test]
    fn decode_term_metadata_singleton_absolute_then_zigzag_delta() {
        let mut bytes = Vec::new();
        // absolute singleton: docStartFP delta=0 (l=0), then raw vint singletonDocID=7
        bytes.write_vlong(0);
        bytes.write_vint(7);
        // next term: singleton delta of +3 via zigzag, flag bit set
        let zz = lucene_util::zigzag::encode(3);
        bytes.write_vlong(((zz as i64) << 1) | 1);
        let mut r = SliceInput::new(&bytes);

        let first = decode_term_metadata(&mut r, 1, true, TermMetadata::EMPTY).unwrap();
        assert_eq!(first.singleton_doc_id, 7);

        let second = decode_term_metadata(&mut r, 1, false, first).unwrap();
        assert_eq!(second.singleton_doc_id, 10);
        assert_eq!(second.doc_start_fp, first.doc_start_fp);
    }

    #[test]
    fn decode_term_metadata_rejects_absolute_singleton_delta() {
        let mut bytes = Vec::new();
        bytes.write_vlong(1); // flag bit set on what must be an absolute decode
        let mut r = SliceInput::new(&bytes);
        let err = decode_term_metadata(&mut r, 1, true, TermMetadata::EMPTY).unwrap_err();
        assert!(matches!(err, Error::AbsoluteSingletonDelta));
    }

    #[test]
    fn decode_term_metadata_rejects_delta_with_no_previous_singleton() {
        let mut bytes = Vec::new();
        bytes.write_vlong(1); // flag bit set, non-absolute
        let mut r = SliceInput::new(&bytes);
        let err = decode_term_metadata(&mut r, 1, false, TermMetadata::EMPTY).unwrap_err();
        assert!(matches!(err, Error::NoPreviousSingleton));
    }
}
