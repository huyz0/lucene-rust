//! Port of `backward_codecs.lucene90.blocktree.Lucene90BlockTreeTermsReader`
//! (with its `FieldReader`): the term dictionary of every postings format
//! from `Lucene90` to `Lucene101`, versions 0 (`VERSION_START`), 1
//! (`VERSION_MSB_VLONG_OUTPUT`) and 2 (`VERSION_FST_CONTINUOUS_ARCS`).
//!
//! # What differs from `Lucene103BlockTreeTermsReader`
//!
//! Only the **terms index**. The `.tim` blocks -- entry counts, suffixes and
//! their compression, stats with singleton runs, per-term postings metadata,
//! floor blocks and sub-block pointers -- are byte for byte the format
//! [`crate::blocktree`] reads (`SegmentTermsEnumFrame` of both packages differ
//! only in how they receive floor data). The index is an FST instead of a
//! trie: one key per block prefix, whose `ByteSequenceOutputs` output is
//!
//! ```text
//! vLong(fp << 2 | hasTerms << 1 | isFloor)   -- MSB-first vlong from version 1
//! [floor data]                               -- when isFloor: vInt(count - 1),
//!                                               then per follow-on block its
//!                                               lead byte and vLong(fpDelta << 1 | hasTerms)
//! ```
//!
//! -- exactly what `Lucene103`'s `TrieBuilder.Output` records per trie node,
//! floor data included, in the same encoding. The `.tmd` record carries the
//! root block's output (`rootCode`) and the FST's metadata; the FST body
//! lives in `.tip` at `indexStartFP`.
//!
//! # How this port reads it
//!
//! In place, as Lucene does. [`open`] reads each field's `.tmd` record and
//! FST metadata, as `FieldReader`'s constructor does, and nothing of the
//! FST's body; the body stays in `.tip` (`OffHeapFSTStore`). Every lookup,
//! enumeration and intersection is [`crate::blocktree`]'s `SegmentTermsEnum`
//! over the untouched `.tim`, stepping through the FST instead of a trie
//! ([`FstTermsIndex`]): it follows an arc per target byte
//! (`FST.findTargetArc`) exactly where it follows a trie child, accumulates
//! the arcs' outputs, and pushes a frame wherever the arc is final -- with
//! `output + nextFinalOutput` decoded into the block's fp, `hasTerms` and
//! floor data (`Lucene90`'s `pushFrame(arc, frameData, length)`), which the
//! frame keeps in its own buffer where a trie frame points into `.tip`.
//! Both indexes send a seek to the same block. A corrupt FST body is found
//! by the first lookup that reaches it, as Lucene's off-heap FST finds it.
//!
//! Until the M8 close-out the FST was converted into this crate's trie --
//! first at open, then at each field's first use -- which cost 3.3 ms per
//! 1M-document segment against Lucene's tens of microseconds. No path needs
//! the conversion: the dictionary, `CheckIndex` and merging read an old
//! field through the same enum. Recorded in `docs/parity.md`.

use std::sync::Arc;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::blocktree::{
    self, BlockTreeFields, Error, FieldTerms, Result, SharedBytes, MIN_FIELD_RECORD_BYTES,
};
use crate::field_infos::FieldInfos;
use crate::fst::{self, Fst, FstMetadata};
use crate::postings::PostingsFormat;

/// `Lucene90BlockTreeTermsReader.TERMS_CODEC_NAME` (and the index/meta
/// names, shared with `Lucene103`).
const TERMS_CODEC_NAME: &str = "BlockTreeTermsDict";
const TERMS_INDEX_CODEC_NAME: &str = "BlockTreeTermsIndex";
const TERMS_META_CODEC_NAME: &str = "BlockTreeTermsMeta";
/// `Lucene90BlockTreeTermsReader.VERSION_START`.
const VERSION_START: i32 = 0;
/// `Lucene90BlockTreeTermsReader.VERSION_MSB_VLONG_OUTPUT`.
const VERSION_MSB_VLONG_OUTPUT: i32 = 1;
/// `Lucene90BlockTreeTermsReader.VERSION_CURRENT`
/// (`VERSION_FST_CONTINUOUS_ARCS`).
pub(crate) const VERSION_CURRENT: i32 = 2;

/// `TERMS_CODEC` of `Lucene90PostingsFormat`, `Lucene99PostingsFormat`,
/// `Lucene912PostingsFormat` and `Lucene101PostingsFormat` alike: the
/// postings reader's header inside `.tmd`.
pub(crate) const POSTINGS_TERMS_CODEC: &str = "Lucene90PostingsWriterTerms";
/// The versions of [`POSTINGS_TERMS_CODEC`] those formats write
/// (`Lucene90`/`Lucene101`: up to 1).
const POSTINGS_TERMS_VERSION_CURRENT: i32 = 1;
/// Their `BLOCK_SIZE`, which `init` checks against `.tmd`.
const POSTINGS_BLOCK_SIZE: i32 = 128;

const OUTPUT_FLAGS_NUM_BITS: u32 = 2;
const OUTPUT_FLAG_IS_FLOOR: i64 = 0x1;
const OUTPUT_FLAG_HAS_TERMS: i64 = 0x2;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// `FieldReader.readMSBVLong`: a vlong written most significant group first.
// ARITH: at most ten 7-bit groups are shifted in; like Java's `long`, the
// high bits of an over-long encoding fall off the top rather than trap.
#[allow(clippy::arithmetic_side_effects)]
fn read_msb_vlong(r: &mut SliceInput) -> Result<i64> {
    let mut l: i64 = 0;
    for _ in 0..10 {
        let b = r.read_byte()?;
        l = l.wrapping_shl(7) | i64::from(b & 0x7f);
        if b & 0x80 == 0 {
            return Ok(l);
        }
    }
    Err(corrupt("MSB vlong longer than 10 bytes"))
}

/// What a final arc's output says about the block its prefix names
/// (`SegmentTermsEnum.pushFrame(arc, frameData, length)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FstBlock {
    /// The block's file pointer in `.tim` (`code >>> 2`).
    pub(crate) fp: u64,
    /// `code & OUTPUT_FLAG_HAS_TERMS`.
    pub(crate) has_terms: bool,
    /// Where the floor data starts in the frame data, when
    /// `code & OUTPUT_FLAG_IS_FLOOR` (`setFloorData`'s
    /// `in.getPosition()`).
    pub(crate) floor_start: Option<usize>,
}

/// Decodes the head of one frame's data: `readVLongOutput`, then the
/// flags `pushFrame` reads off it.
fn decode_frame_data(data: &[u8], version: i32) -> Result<FstBlock> {
    let mut r = SliceInput::new(data);
    let code = if version >= VERSION_MSB_VLONG_OUTPUT {
        read_msb_vlong(&mut r)?
    } else {
        r.read_vlong()?
    };
    if code < 0 {
        return Err(corrupt(format!("negative block pointer code {code}")));
    }
    // ARITH: `code` is non-negative (checked above); a right shift of it
    // cannot overflow.
    #[allow(clippy::arithmetic_side_effects)]
    let fp = (code >> OUTPUT_FLAGS_NUM_BITS) as u64;
    Ok(FstBlock {
        fp,
        has_terms: code & OUTPUT_FLAG_HAS_TERMS != 0,
        floor_start: (code & OUTPUT_FLAG_IS_FLOOR != 0).then(|| r.position()),
    })
}

/// One field's FST terms index, read in place: `FieldReader.index` (its
/// metadata from `.tmd`, its body a region of `.tip`) plus `rootCode`.
pub(crate) struct FstTermsIndex {
    field: String,
    tip: SharedBytes,
    /// The FST body's `[start, end)` in `.tip`.
    body: std::ops::Range<usize>,
    /// The FST's metadata without its empty output: the root frame's data
    /// is [`Self::root_code`] (`FieldReader`: "rootCode.equals(emptyOutput)",
    /// checked at open), so a per-walk [`Fst`] never clones it.
    metadata: FstMetadata,
    /// `rootCode`, the root block's frame data.
    root_code: Vec<u8>,
    version: i32,
}

impl std::fmt::Debug for FstTermsIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "field {:?}, .tip {}..{}, version {}",
            self.field, self.body.start, self.body.end, self.version
        )
    }
}

/// A position in an [`FstTermsIndex`]: the arc reaching a prefix and the
/// outputs of every arc on the way (`SegmentTermsEnum`'s `arcs[]` and
/// `outputAccumulator`, one step's worth).
#[derive(Debug, Clone)]
pub(crate) struct FstNode {
    arc: fst::Arc,
    /// The concatenated outputs of the arcs from the root to `arc`
    /// (`ByteSequenceOutputs.add`), without `arc.nextFinalOutput()`.
    output: Vec<u8>,
    /// The root (`getFirstArc`), whose frame data is `rootCode`.
    root: bool,
}

impl FstNode {
    /// The prefix names a block: the root always does
    /// (`assert arc.isFinal()` on the first arc), any other node when its
    /// arc is final.
    #[inline]
    pub(crate) fn has_block(&self) -> bool {
        self.root || self.arc.is_final()
    }
}

impl FstTermsIndex {
    fn corrupt(&self, e: impl std::fmt::Display) -> Error {
        corrupt(format!("field {:?} terms index: {e}", self.field))
    }

    /// `FieldReader.index` over the mapped `.tip`.
    fn fst(&self) -> Fst<'_> {
        let tip: &[u8] = self.tip.as_ref().as_ref();
        // `open` checked the range against this very `.tip`.
        Fst::from_parts(self.metadata.clone(), &tip[self.body.clone()])
    }

    /// `FST.getFirstArc`.
    pub(crate) fn root(&self) -> FstNode {
        FstNode {
            arc: self.fst().first_arc(),
            output: Vec::new(),
            root: true,
        }
    }

    /// `FST.findTargetArc(label, node.arc, ...)`, accumulating the arc's
    /// output (`outputAccumulator.push(arc.output())`).
    pub(crate) fn child(&self, node: &FstNode, label: u8) -> Result<Option<FstNode>> {
        let Some(arc) = self
            .fst()
            .find_target_arc_byte(label, &node.arc)
            .map_err(|e| self.corrupt(e))?
        else {
            return Ok(None);
        };
        let mut output = node.output.clone();
        output.extend_from_slice(arc.output());
        Ok(Some(FstNode {
            arc,
            output,
            root: false,
        }))
    }

    /// Writes `node`'s frame data -- `rootCode` for the root, else the
    /// accumulated output plus the arc's final output -- into `buf` and
    /// decodes its head. `buf` then holds the floor data the frame reads.
    pub(crate) fn block(&self, node: &FstNode, buf: &mut Vec<u8>) -> Result<FstBlock> {
        buf.clear();
        if node.root {
            buf.extend_from_slice(&self.root_code);
        } else if node.arc.is_final() {
            buf.extend_from_slice(&node.output);
            buf.extend_from_slice(node.arc.next_final_output());
        } else {
            return Err(self.corrupt("a frame pushed on a prefix that names no block"));
        }
        decode_frame_data(buf, self.version).map_err(|e| self.corrupt(e))
    }

    /// The block `node` names, if it names one.
    pub(crate) fn block_fp(&self, node: &FstNode) -> Result<Option<u64>> {
        if !node.has_block() {
            return Ok(None);
        }
        let mut buf = Vec::new();
        Ok(Some(self.block(node, &mut buf)?.fp))
    }
}

/// `Lucene90BlockTreeTermsReader`'s constructor plus a `FieldReader` per
/// field; see the module doc for the index conversion.
pub(crate) fn open(
    tim: SharedBytes,
    tip: SharedBytes,
    tmd: &[u8],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
) -> Result<BlockTreeFields> {
    let tim_bytes: &[u8] = tim.as_ref().as_ref();
    let tip_bytes: &[u8] = tip.as_ref().as_ref();
    let mut tim_input = SliceInput::new(tim_bytes);
    let version = codec_util::check_index_header(
        &mut tim_input,
        TERMS_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?
    .version;
    let mut tip_input = SliceInput::new(tip_bytes);
    codec_util::check_index_header(
        &mut tip_input,
        TERMS_INDEX_CODEC_NAME,
        version,
        version,
        segment_id,
        segment_suffix,
    )?;
    let mut tmd_input = SliceInput::new(tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        version,
        version,
        segment_id,
        segment_suffix,
    )?;
    // `postingsReader.init`.
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        0,
        POSTINGS_TERMS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    let index_block_size = tmd_input.read_vint()?;
    if index_block_size != POSTINGS_BLOCK_SIZE {
        return Err(Error::UnexpectedBlockSize {
            found: index_block_size,
        });
    }

    let num_fields = tmd_input.read_vint()?;
    if num_fields < 0 || num_fields as usize > tmd_input.remaining() / MIN_FIELD_RECORD_BYTES {
        return Err(Error::InvalidNumFields(num_fields));
    }
    let mut fields: Vec<(String, FieldTerms)> = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint()?;
        let num_terms = tmd_input.read_vlong()?;
        if num_terms <= 0 {
            return Err(Error::IllegalNumTerms(field_number));
        }
        let root_code = blocktree::read_bytes_ref(&mut tmd_input)?;
        let field_info = field_infos
            .field_by_number(field_number)
            .ok_or(Error::InvalidFieldNumber(field_number))?;
        let postings_format = PostingsFormat::of_field(field_info)
            .filter(|f| f.uses_fst_terms_index())
            .ok_or_else(|| {
                corrupt(format!(
                    "field {:?} is in a Lucene90 block tree but names no postings format that \
                     writes one",
                    field_info.name
                ))
            })?;
        let (sum_total_term_freq, sum_doc_freq) =
            blocktree::read_freq_pair(&mut tmd_input, field_info.index_options)?;
        let doc_count = tmd_input.read_vint()?;
        let min_term = blocktree::read_bytes_ref(&mut tmd_input)?;
        let mut max_term = blocktree::read_bytes_ref(&mut tmd_input)?;
        if num_terms == 1 {
            max_term = min_term.clone();
        }
        if !(0..=max_doc).contains(&doc_count) {
            return Err(Error::InvalidDocCount { doc_count, max_doc });
        }
        if sum_doc_freq < i64::from(doc_count) {
            return Err(Error::InvalidSumDocFreq {
                sum_doc_freq,
                doc_count,
            });
        }
        if sum_total_term_freq < sum_doc_freq {
            return Err(Error::InvalidSumTotalTermFreq {
                sum_total_term_freq,
                sum_doc_freq,
            });
        }
        let index_start_fp = tmd_input.read_vlong()?;
        let index_start_fp = u64::try_from(index_start_fp)
            .map_err(|_| corrupt(format!("negative indexStartFP {index_start_fp}")))?;
        // The FST's metadata is read, and its body's extent in `.tip`
        // checked, as `FieldReader`'s constructor reads them; the body is
        // walked in place by every lookup.
        let fst = Fst::read_split(&mut tmd_input, tip_bytes, index_start_fp)
            .map_err(|e| corrupt(format!("field {:?} terms index: {e}", field_info.name)))?;
        let mut metadata = fst.metadata().clone();
        // The root block's code is recorded twice, as `rootCode` and as the
        // FST's empty output; `FieldReader` asserts they are equal, and the
        // root frame reads `rootCode` (`pushFrame(arc, fr.rootCode, 0)`).
        metadata.empty_output = None;
        decode_frame_data(&root_code, version)?;
        let start = usize::try_from(index_start_fp)
            .map_err(|_| corrupt(format!("indexStartFP {index_start_fp} out of range")))?;
        let end = start
            .checked_add(metadata.num_bytes as usize)
            .ok_or_else(|| corrupt(format!("FST body at {start} overflows")))?;
        let body = start..end;
        if fields.iter().any(|(n, _)| n == &field_info.name) {
            return Err(Error::DuplicateField(field_info.name.clone()));
        }
        let index = FstTermsIndex {
            field: field_info.name.clone(),
            tip: Arc::clone(&tip),
            body,
            metadata,
            root_code,
            version,
        };
        fields.push((
            field_info.name.clone(),
            FieldTerms::from_parts(
                (num_terms, sum_total_term_freq, sum_doc_freq, doc_count),
                min_term,
                max_term,
                field_info,
                postings_format,
                Arc::clone(&tim),
                index,
            ),
        ));
    }
    let index_length = tmd_input.read_i64()?;
    let terms_length = tmd_input.read_i64()?;
    codec_util::check_footer(&mut tmd_input, tmd.len())?;
    if index_length < 0 || terms_length < 0 {
        return Err(corrupt(format!(
            "negative recorded .tip/.tim length: {index_length}/{terms_length}"
        )));
    }
    codec_util::retrieve_checksum_with_expected_length(tip_bytes, index_length as usize)?;
    codec_util::retrieve_checksum_with_expected_length(tim_bytes, terms_length as usize)?;
    Ok(BlockTreeFields::from_fields(fields))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    #[test]
    fn msb_vlong_reads_most_significant_group_first() {
        // 300 = 0b10_0101100: groups [0b10, 0b0101100] -> 0x82, 0x2c.
        let mut r = SliceInput::new(&[0x82, 0x2c]);
        assert_eq!(read_msb_vlong(&mut r).unwrap(), 300);
        let mut r = SliceInput::new(&[0xff; 11]);
        assert!(read_msb_vlong(&mut r).is_err());
    }

    #[test]
    fn frame_data_decodes_in_both_encodings() {
        // fp 5, has terms, floor, then two floor bytes.
        let code = (5 << 2) | 0x3;
        let v0 = decode_frame_data(&[code as u8, 0x01, 0x61], VERSION_START).unwrap();
        assert_eq!(
            v0,
            FstBlock {
                fp: 5,
                has_terms: true,
                floor_start: Some(1),
            }
        );
        let v1 = decode_frame_data(&[code as u8], VERSION_MSB_VLONG_OUTPUT).unwrap();
        assert_eq!(v1.floor_start, Some(1));
        let plain = decode_frame_data(&[5 << 2], VERSION_CURRENT).unwrap();
        assert!(!plain.has_terms);
        assert!(plain.floor_start.is_none());
        // The two encodings of a two-group code differ: 300 is LSB-first
        // [0xac, 0x02] and MSB-first [0x82, 0x2c].
        assert_eq!(
            decode_frame_data(&[0xac, 0x02], VERSION_START).unwrap().fp,
            75
        );
        assert_eq!(
            decode_frame_data(&[0x82, 0x2c], VERSION_CURRENT)
                .unwrap()
                .fp,
            75
        );
        // A vlong that decodes negative is not a block pointer.
        let neg = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert!(decode_frame_data(&neg, VERSION_START).is_err());
    }

    /// The 9.0.0 fixture's first segment (`Lucene90` postings, block tree
    /// version 0): `(tim, tip, tmd, fnm, id, suffix)`.
    fn fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, FieldInfos, [u8; 16], String) {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/data/bwc/9.0.0");
        let read = |n: &str| std::fs::read(dir.join(n)).unwrap();
        let tim = read("_0_Lucene90_0.tim");
        // The id every index header carries, after magic, name and version.
        let name_len = tim[4] as usize;
        let mut id = [0u8; 16];
        id.copy_from_slice(&tim[9 + name_len..25 + name_len]);
        let fnm = crate::field_infos::parse(&read("_0.fnm"), &id, "").unwrap();
        (
            tim,
            read("_0_Lucene90_0.tip"),
            read("_0_Lucene90_0.tmd"),
            fnm,
            id,
            "Lucene90_0".to_string(),
        )
    }

    fn open_fixture(
        tim: &[u8],
        tip: &[u8],
        tmd: &[u8],
        fnm: &FieldInfos,
        id: &[u8; 16],
        suffix: &str,
    ) -> Result<BlockTreeFields> {
        blocktree::open(tim, tip, tmd, fnm, id, suffix, 3000)
    }

    #[test]
    fn a_real_lucene90_dictionary_opens_through_the_common_entry_point() {
        let (tim, tip, tmd, fnm, id, suffix) = fixture();
        let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
        let body = fields.field("body").unwrap();
        assert_eq!(body.postings_format(), PostingsFormat::Lucene90);
        assert_eq!(body.num_terms, 20);
        assert!(body.try_seek_exact(b"zeta").unwrap().is_some());
        assert!(body.try_seek_exact(b"zzz").unwrap().is_none());
    }

    /// The index is read in place: every field keeps its FST in `.tip`,
    /// and a clone shares it.
    #[test]
    fn the_fst_index_is_read_in_place() {
        let (tim, tip, tmd, fnm, id, suffix) = fixture();
        let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
        for (name, field) in fields.iter_fields() {
            let debug = format!("{field:?}");
            assert!(debug.contains("index: Fst(field "), "{name}: {debug}");
        }
        let body = fields.field("body").unwrap();
        let copy = body.clone();
        assert!(copy.try_seek_exact(b"zeta").unwrap().is_some());
        assert!(copy.try_seek_exact(b"zzz").unwrap().is_none());
    }

    /// Every term of every field is found by `seek_exact` and by
    /// `seek_ceil` on itself, and a key just past each one ceils to the
    /// next term -- the FST walk sends every seek to the block the full
    /// enumeration found it in.
    #[test]
    fn every_term_is_found_by_walking_the_fst() {
        let (tim, tip, tmd, fnm, id, suffix) = fixture();
        let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
        let mut seen = 0;
        for (name, field) in fields.iter_fields() {
            let mut all = Vec::new();
            let mut it = field.iter();
            while let Some(t) = it.try_next_term().unwrap() {
                all.push(t.to_vec());
            }
            assert_eq!(all.len() as i64, field.num_terms, "{name}");
            for (i, t) in all.iter().enumerate() {
                assert!(field.try_seek_exact(t).unwrap().is_some(), "{name} {t:?}");
                let mut e = field.iter();
                assert_eq!(
                    e.try_seek_ceil(t).unwrap(),
                    blocktree::SeekStatus::Found,
                    "{name}"
                );
                let mut past = t.clone();
                past.push(0);
                let mut e = field.iter();
                let status = e.try_seek_ceil(&past).unwrap();
                match all.get(i + 1) {
                    Some(next) => {
                        assert_eq!(status, blocktree::SeekStatus::NotFound, "{name}");
                        assert_eq!(e.term(), Some(next.as_slice()), "{name}");
                    }
                    None => assert_eq!(status, blocktree::SeekStatus::End, "{name}"),
                }
                seen += 1;
            }
        }
        assert!(seen > 100, "{seen}");
    }

    /// A corrupt FST body is met by the lookups that reach it, as
    /// Lucene's off-heap FST meets it at a seek (the checksum is
    /// `CheckIndex`'s to verify): each one either fails -- an FST error
    /// naming the field's terms index -- or misses, and none panics.
    #[test]
    fn a_corrupt_fst_body_is_met_by_the_lookups_that_reach_it() {
        let (tim, clean_tip, tmd, fnm, id, suffix) = fixture();
        let clean = open_fixture(&tim, &clean_tip, &tmd, &fnm, &id, &suffix).unwrap();
        let mut terms: Vec<(String, Vec<u8>)> = Vec::new();
        for (name, field) in clean.iter_fields() {
            let mut it = field.iter();
            while let Some(t) = it.try_next_term().unwrap() {
                terms.push((name.to_string(), t.to_vec()));
            }
        }
        let header = codec_util::index_header_length(TERMS_INDEX_CODEC_NAME, &suffix);
        let footer = clean_tip.len() - 16;
        for fill in [0x00u8, 0xff, 0x5a] {
            let mut tip = clean_tip.clone();
            for (i, b) in tip[header..footer].iter_mut().enumerate() {
                *b = if fill == 0x5a {
                    (i * 31 % 251) as u8
                } else {
                    fill
                };
            }
            let fields = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap();
            let (mut failed, mut missed) = (0, 0);
            for (name, term) in &terms {
                let field = fields.field(name).unwrap();
                match field.try_seek_exact(term) {
                    Ok(Some(_)) => {}
                    Ok(None) => missed += 1,
                    Err(e) => {
                        let msg = e.to_string();
                        if msg.contains("FST") {
                            assert!(msg.contains("terms index"), "{name}: {msg}");
                        }
                        failed += 1;
                    }
                }
            }
            assert!(
                failed + missed > 0,
                "fill {fill:#x}: no lookup met the corrupt bytes"
            );
        }
    }

    /// A node's frame data, pushed where no block is, is an error rather
    /// than a frame at a made-up fp; the root's is `rootCode`.
    #[test]
    fn a_prefix_without_a_block_has_no_frame_data() {
        let index = FstTermsIndex {
            field: "f".to_string(),
            tip: Arc::new(vec![0u8; 4]),
            body: 0..0,
            metadata: FstMetadata {
                input_type: fst::InputType::Byte1,
                empty_output: None,
                start_node: 0,
                version: 8,
                num_bytes: 0,
            },
            root_code: vec![5 << 2],
            version: VERSION_CURRENT,
        };
        let root = index.root();
        assert!(root.has_block());
        assert_eq!(index.block_fp(&root).unwrap(), Some(5));
        // The start node has no arcs.
        assert!(index.child(&root, b'a').unwrap().is_none());
        let inner = FstNode {
            arc: fst::Arc::default(),
            output: Vec::new(),
            root: false,
        };
        assert!(!inner.has_block());
        assert_eq!(index.block_fp(&inner).unwrap(), None);
        let err = index.block(&inner, &mut Vec::new()).unwrap_err();
        assert!(err.to_string().contains("names no block"), "{err}");
        assert!(format!("{index:?}").contains("field \"f\""));
    }

    #[test]
    fn a_field_naming_a_trie_format_is_rejected() {
        let (tim, tip, tmd, mut fnm, id, suffix) = fixture();
        for f in &mut fnm.fields {
            for (k, v) in &mut f.attributes {
                if k == "PerFieldPostingsFormat.format" {
                    *v = "Lucene104".to_string();
                }
            }
        }
        let err = open_fixture(&tim, &tip, &tmd, &fnm, &id, &suffix).unwrap_err();
        assert!(
            err.to_string().contains("names no postings format"),
            "{err}"
        );
    }

    #[test]
    fn truncated_or_mismatched_files_are_errors() {
        let (tim, tip, tmd, fnm, id, suffix) = fixture();
        // `.tmd` cut short: the field records run out.
        assert!(open_fixture(&tim, &tip, &tmd[..tmd.len() / 2], &fnm, &id, &suffix).is_err());
        // `.tip` of the wrong length fails `retrieveChecksum`'s length check.
        let mut long_tip = tip.clone();
        long_tip.insert(40, 0);
        assert!(open_fixture(&tim, &long_tip, &tmd, &fnm, &id, &suffix).is_err());
        // A different segment's id fails every header.
        let other = [7u8; 16];
        assert!(open_fixture(&tim, &tip, &tmd, &fnm, &other, &suffix).is_err());
    }
}
