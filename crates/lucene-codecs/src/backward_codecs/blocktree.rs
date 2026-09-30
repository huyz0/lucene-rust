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
//! [`open`] enumerates each field's FST once -- every `(block prefix, output)`
//! pair, `BytesRefFSTEnum`'s walk -- decodes each output into a
//! `TrieBuilder.Output`, and saves the result as a trie with this crate's own
//! `TrieBuilder` port. From there every lookup, enumeration and intersection
//! is [`crate::blocktree`]'s, over the untouched `.tim`: `SegmentTermsEnum`
//! follows an FST arc per target byte exactly where it follows a trie child,
//! and pushes a frame wherever the arc is final exactly where a trie node has
//! an output, so both indexes send a seek to the same block.
//!
//! This is the one place M8 converts rather than reads in place: Lucene walks
//! the FST off-heap on every seek. The conversion costs one pass over the
//! index at open -- the index has an entry per *block* (on the order of one
//! per 25-48 terms), not per term -- and buys the whole trie-based search
//! path, intersections included, unchanged. Recorded in `docs/parity.md`.

use std::sync::Arc;

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::blocktree::{
    self, BlockTreeFields, Error, FieldTerms, Result, SharedBytes, MIN_FIELD_RECORD_BYTES,
};
use crate::blocktree_writer::{TrieBuilder, TrieOutput};
use crate::field_infos::FieldInfos;
use crate::fst::Fst;
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

/// Decodes one FST output (`SegmentTermsEnum.pushFrame`'s `code`, then
/// `setFloorData` on the rest) into the trie's output for the same block.
fn decode_output(output: &[u8], version: i32) -> Result<TrieOutput> {
    let mut r = SliceInput::new(output);
    let code = if version >= VERSION_MSB_VLONG_OUTPUT {
        read_msb_vlong(&mut r)?
    } else {
        r.read_vlong()?
    };
    if code < 0 {
        return Err(corrupt(format!("negative block pointer code {code}")));
    }
    let floor = code & OUTPUT_FLAG_IS_FLOOR != 0;
    // ARITH: `code` is non-negative (checked above); a right shift of it
    // cannot overflow.
    #[allow(clippy::arithmetic_side_effects)]
    let fp = (code >> OUTPUT_FLAGS_NUM_BITS) as u64;
    Ok(TrieOutput {
        fp,
        has_terms: code & OUTPUT_FLAG_HAS_TERMS != 0,
        floor_data: floor.then(|| r.as_slice().to_vec()),
    })
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
        let fst = Fst::read_split(&mut tmd_input, tip_bytes, index_start_fp)
            .map_err(|e| corrupt(format!("field {:?} terms index: {e}", field_info.name)))?;

        // The FST's keys are the block prefixes; the empty prefix is the
        // root block, whose output `.tmd` also records as `rootCode`
        // (`FieldReader`: "rootCode.equals(emptyOutput)").
        let mut builder = TrieBuilder {
            empty_output: Some(decode_output(&root_code, version)?),
            entries: Vec::new(),
        };
        let walk = fst
            .iter()
            .map_err(|e| corrupt(format!("field {:?} terms index: {e}", field_info.name)))?;
        for entry in walk {
            let (key, output) = entry
                .map_err(|e| corrupt(format!("field {:?} terms index: {e}", field_info.name)))?;
            if key.is_empty() {
                continue;
            }
            builder
                .entries
                .push((key, decode_output(&output, version)?));
        }
        let mut trie = Vec::new();
        let location = builder.save(&mut trie);
        let (index_start, root_fp, index_end) = (
            location.index_start as usize,
            location.root_fp as usize,
            location.index_end as usize,
        );
        blocktree::load_node(&trie[index_start..index_end], root_fp)?;
        if fields.iter().any(|(n, _)| n == &field_info.name) {
            return Err(Error::DuplicateField(field_info.name.clone()));
        }
        let trie: SharedBytes = Arc::new(trie);
        fields.push((
            field_info.name.clone(),
            FieldTerms::from_parts(
                (num_terms, sum_total_term_freq, sum_doc_freq, doc_count),
                min_term,
                max_term,
                field_info,
                postings_format,
                Arc::clone(&tim),
                trie,
                (index_start, root_fp, index_end),
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
    fn outputs_decode_to_trie_outputs_in_both_encodings() {
        // fp 5, has terms, floor, then two floor bytes.
        let code = (5 << 2) | 0x3;
        let v0 = decode_output(&[code as u8, 0x01, 0x61], VERSION_START).unwrap();
        assert_eq!(v0.fp, 5);
        assert!(v0.has_terms);
        assert_eq!(v0.floor_data.as_deref(), Some(&[0x01, 0x61][..]));
        let v1 = decode_output(&[code as u8], VERSION_MSB_VLONG_OUTPUT).unwrap();
        assert_eq!(v1.floor_data.as_deref(), Some(&[][..]));
        let plain = decode_output(&[5 << 2], VERSION_CURRENT).unwrap();
        assert!(!plain.has_terms);
        assert!(plain.floor_data.is_none());
        // A vlong that decodes negative is not a block pointer.
        let neg = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert!(decode_output(&neg, VERSION_START).is_err());
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
