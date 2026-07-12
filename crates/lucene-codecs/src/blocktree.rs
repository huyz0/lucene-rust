//! Port of `org.apache.lucene.codecs.lucene103.blocktree.Lucene103BlockTreeTermsReader`
//! (`.tim` term dictionary + `.tip` term index + `.tmd` per-field metadata) —
//! read-only, scoped to **`seekExact` + `docFreq`/`totalTermFreq`** only.
//!
//! Note on naming: the pinned Lucene version (10.5.0) uses
//! `Lucene104PostingsFormat`, whose term dictionary is
//! `Lucene103BlockTreeTermsReader`/`Writer` (package
//! `o.a.l.codecs.lucene103.blocktree`) — *not* the `lucene90.blocktree`
//! classes, which live in `backward-codecs` and are out of scope for this
//! port (see PLAN.md's "pin one Lucene version" rule). The `.tip` term index
//! in this version is **not an FST** — Lucene 10.x replaced it with a
//! purpose-built binary trie (`TrieReader`/`TrieBuilder`), a flatter,
//! pointer-chasing encoding of the same "prefix trie whose leaves are term
//! blocks" idea `fst.rs`'s module doc describes for the *older* format.
//! `fst.rs` remains useful groundwork (arc-lookup style reasoning, shared
//! `codec_util` header handling) but is not used by this module.
//!
//! ## Wire format
//!
//! - `.tmd` (`TERMS_META_EXTENSION`): `IndexHeader(codec="BlockTreeTermsMeta")`,
//!   then the postings reader's own `init` header
//!   (`IndexHeader(codec="Lucene104PostingsWriterTerms")` + `indexBlockSize: vint`,
//!   which must equal `Lucene104PostingsFormat.BLOCK_SIZE` = 256 for this pinned
//!   version), then `numFields: vint`, then per field: `fieldNumber: vint`,
//!   `numTerms: vlong`, a `sumTotalTermFreq`/`sumDocFreq` pair (see
//!   [`read_freq_pair`] for the DOCS-only aliasing trick), `docCount: vint`,
//!   `minTerm`/`maxTerm` (vint-length-prefixed byte arrays), and finally
//!   `indexStart`/`rootFP`/`indexEnd` (three vlongs locating this field's root
//!   node in `.tip`). After the field loop: `indexLength: i64`, `termsLength: i64`,
//!   `Footer`.
//! - `.tip` (`TERMS_INDEX_EXTENSION`): `IndexHeader(codec="BlockTreeTermsIndex")`,
//!   then every field's trie nodes packed back to back (each field's node
//!   region spans `[indexStart, indexEnd)` from its `.tmd` record), `Footer`.
//!   A trie node's header byte packs a 2-bit `sign` selecting one of three
//!   encodings (`SIGN_NO_CHILDREN`/`SIGN_SINGLE_CHILD_*`/`SIGN_MULTI_CHILDREN`);
//!   see `TrieReader.java` for the full byte-packing scheme.
//! - `.tim` (`TERMS_EXTENSION`): `IndexHeader(codec="BlockTreeTermsDict")`, then
//!   every field's blocks packed back to back (see [`decode_block`]), `Footer`.
//!
//! ## Scope of this slice
//!
//! Ported: opening a `.tim`/`.tip`/`.tmd` triple, per-field metadata
//! (`numTerms`/`sumTotalTermFreq`/`sumDocFreq`/`docCount`/`minTerm`/`maxTerm`),
//! and `seekExact`-equivalent term lookup with `docFreq`/`totalTermFreq`
//! readback — **for fields small enough that the writer placed every term in
//! a single, non-floor, leaf `.tim` block** (real Lucene only splits a
//! prefix into multiple blocks, or a block into floor sub-blocks, once it
//! exceeds `minItemsInBlock`/`maxItemsInBlock`, default 25/48 — this port's
//! test fixtures stay under that). Concretely: the field's trie root node
//! must be `SIGN_NO_CHILDREN` (no child nodes — the whole field is one
//! block) with no floor data, and that block's `isLastInFloor`/`isLeafBlock`
//! bits must both be set. Suffix compression (`CompressionAlgorithm::LZ4`/
//! `LowercaseAscii`) never triggers for a block whose shared prefix length is
//! 0 (see `Lucene103BlockTreeTermsWriter`'s `prefixLength > 2` gate), which a
//! single root-only block always has, so only `NO_COMPRESSION` is
//! implemented.
//!
//! Deferred (all return [`Error::Unsupported`], same stance as `fst.rs`'s
//! rejected array-node encodings): floor blocks (a prefix's terms split
//! across multiple `.tim` blocks under one trie node), multi-block fields
//! (trie nodes with children — `SIGN_SINGLE_CHILD_*`/`SIGN_MULTI_CHILDREN`,
//! needed once a field has more terms than fit one block), `next()`
//! (full enumeration) and `seekCeil` (nearest-match seeking), automaton
//! intersection (`IntersectTermsEnum`), and actual postings decode
//! (`PostingsEnum`/`ImpactsEnum` — a separate, still-unstarted piece per
//! `docs/parity.md`). Because this slice never decodes postings, the
//! per-term metadata bytes written by the postings writer (doc/pos/pay file
//! pointer deltas) are read past (to keep the `.tim` block cursor aligned)
//! but never parsed — `docFreq`/`totalTermFreq` come entirely from the
//! block's separate stats bytes, which is why that skip is safe.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::field_infos::{FieldInfos, IndexOptions};

const TERMS_CODEC_NAME: &str = "BlockTreeTermsDict";
const TERMS_INDEX_CODEC_NAME: &str = "BlockTreeTermsIndex";
const TERMS_META_CODEC_NAME: &str = "BlockTreeTermsMeta";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

/// `Lucene104PostingsFormat.TERMS_CODEC` — the postings writer's own header,
/// embedded in the `.tmd` stream right after BlockTree's own index header.
const POSTINGS_TERMS_CODEC: &str = "Lucene104PostingsWriterTerms";
const POSTINGS_VERSION_START: i32 = 0;
const POSTINGS_VERSION_CURRENT: i32 = 0;
/// `Lucene104PostingsFormat.BLOCK_SIZE` (= `ForUtil.BLOCK_SIZE`), the postings
/// block size the `.tmd` stream's `indexBlockSize` field must match.
const POSTINGS_BLOCK_SIZE: i32 = 256;

/// `TrieBuilder.SIGN_NO_CHILDREN` — the only trie node shape this slice reads.
const SIGN_NO_CHILDREN: u32 = 0x00;
/// `TrieBuilder.LEAF_NODE_HAS_TERMS` (`1 << 5`).
const LEAF_NODE_HAS_TERMS: u32 = 1 << 5;
/// `TrieBuilder.LEAF_NODE_HAS_FLOOR` (`1 << 6`).
const LEAF_NODE_HAS_FLOOR: u32 = 1 << 6;

const BYTES_MINUS_1_MASK: [u64; 8] = [
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    FieldInfos(#[from] crate::field_infos::Error),
    #[error("invalid numFields: {0}")]
    InvalidNumFields(i32),
    #[error("invalid field number: {0}")]
    InvalidFieldNumber(i32),
    #[error("illegal numTerms for field number: {0}")]
    IllegalNumTerms(i32),
    #[error("invalid docCount: {doc_count} maxDoc: {max_doc}")]
    InvalidDocCount { doc_count: i32, max_doc: i32 },
    #[error("invalid sumDocFreq: {sum_doc_freq} docCount: {doc_count}")]
    InvalidSumDocFreq { sum_doc_freq: i64, doc_count: i32 },
    #[error("invalid sumTotalTermFreq: {sum_total_term_freq} sumDocFreq: {sum_doc_freq}")]
    InvalidSumTotalTermFreq {
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
    },
    #[error("duplicate field: {0}")]
    DuplicateField(String),
    #[error(
        "index-time postings BLOCK_SIZE ({found}) != read-time BLOCK_SIZE ({POSTINGS_BLOCK_SIZE})"
    )]
    UnexpectedBlockSize { found: i32 },
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `docFreq`/`totalTermFreq` for one found term — the entirety of what this
/// slice can read back for a term (no postings/doc-ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermStats {
    pub doc_freq: i32,
    pub total_term_freq: i64,
}

/// One field's decoded term dictionary: every (term, stats) pair in the
/// field's single `.tim` block, sorted (as the writer emits them), plus the
/// field-level aggregate stats from `.tmd`.
#[derive(Debug, Clone)]
pub struct FieldTerms {
    pub num_terms: i64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: i32,
    pub min_term: Vec<u8>,
    pub max_term: Vec<u8>,
    entries: Vec<(Vec<u8>, TermStats)>,
}

impl FieldTerms {
    /// `TermsEnum.seekExact(BytesRef)`-equivalent: exact lookup only, no
    /// enumeration/range-seeking. Terms are stored sorted, so this is a
    /// binary search over the materialized block.
    pub fn seek_exact(&self, term: &[u8]) -> Option<TermStats> {
        self.entries
            .binary_search_by(|(t, _)| t.as_slice().cmp(term))
            .ok()
            .map(|idx| self.entries[idx].1)
    }
}

/// All fields' term dictionaries for one segment, keyed by field name.
#[derive(Debug, Clone, Default)]
pub struct BlockTreeFields {
    fields: Vec<(String, FieldTerms)>,
}

impl BlockTreeFields {
    pub fn field(&self, name: &str) -> Option<&FieldTerms> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, f)| f)
    }
}

fn read_bytes_ref(input: &mut SliceInput) -> Result<Vec<u8>> {
    let len = input.read_vint()?;
    if len < 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid bytes length: {len}"
        ))));
    }
    let mut buf = vec![0u8; len as usize];
    input.read_bytes(&mut buf)?;
    Ok(buf)
}

/// Reads the `sumTotalTermFreq`/`sumDocFreq` pair, aliasing the single value
/// written when `IndexOptions::Docs` (frequencies aren't stored at all, so
/// `sumTotalTermFreq == sumDocFreq` and only one vlong is on the wire) —
/// mirrors `Lucene103BlockTreeTermsReader`'s constructor exactly.
fn read_freq_pair(input: &mut SliceInput, index_options: IndexOptions) -> Result<(i64, i64)> {
    let first = input.read_vlong()?;
    if index_options == IndexOptions::Docs {
        Ok((first, first))
    } else {
        let sum_doc_freq = input.read_vlong()?;
        Ok((first, sum_doc_freq))
    }
}

/// Reads one trie node at `fp` within `slice` (the field's `[indexStart,
/// indexEnd)` region of `.tip`), returning its `(outputFp, hasTerms)` —
/// i.e. `TrieReader.load` + `loadLeafNode`, restricted to
/// `SIGN_NO_CHILDREN` (see the module doc for why that's the only shape
/// this slice's fixtures ever produce).
fn read_root_node(slice: &[u8], fp: usize) -> Result<(u64, bool)> {
    if fp + 8 > slice.len() {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "trie node read past end of index slice".into(),
        )));
    }
    let word = u64::from_le_bytes(slice[fp..fp + 8].try_into().unwrap());
    let term_flags = word as u32;
    let sign = term_flags & 0x03;
    if sign != SIGN_NO_CHILDREN {
        return Err(Error::Unsupported(
            "multi-block field (trie node has children) not supported in this slice",
        ));
    }

    let fp_bytes_minus1 = (term_flags >> 2) & 0x07;
    let output_fp = if fp_bytes_minus1 <= 6 {
        (word >> 8) & BYTES_MINUS_1_MASK[fp_bytes_minus1 as usize]
    } else {
        if fp + 9 > slice.len() {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "trie node output fp read past end of index slice".into(),
            )));
        }
        u64::from_le_bytes(slice[fp + 1..fp + 9].try_into().unwrap())
    };

    let has_terms = (term_flags & LEAF_NODE_HAS_TERMS) != 0;
    let has_floor = (term_flags & LEAF_NODE_HAS_FLOOR) != 0;
    if has_floor {
        return Err(Error::Unsupported(
            "floor blocks not supported in this slice",
        ));
    }
    Ok((output_fp, has_terms))
}

/// Decodes the single `.tim` block at `fp`, materializing every (term,
/// stats) entry — `SegmentTermsEnumFrame.loadBlock` plus a full
/// `decodeMetaData` pass over every entry, restricted to a non-floor leaf
/// block (see the module doc). Per-term postings metadata bytes are read
/// past (to stay aligned) but never decoded, since stats alone determine
/// `docFreq`/`totalTermFreq`.
fn decode_block(
    tim: &[u8],
    fp: usize,
    index_options: IndexOptions,
) -> Result<Vec<(Vec<u8>, TermStats)>> {
    let mut r = SliceInput::new(tim);
    r.seek(fp)?;

    let code = r.read_vint()?;
    let ent_count = (code as u32) >> 1;
    if ent_count == 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "empty terms block".into(),
        )));
    }
    let is_last_in_floor = (code & 1) != 0;
    if !is_last_in_floor {
        return Err(Error::Unsupported(
            "floor blocks not supported in this slice",
        ));
    }

    let code_l = r.read_vlong()? as u64;
    let is_leaf_block = (code_l & 0x04) != 0;
    if !is_leaf_block {
        return Err(Error::Unsupported(
            "non-leaf block (nested sub-blocks) not supported in this slice",
        ));
    }
    let num_suffix_bytes = (code_l >> 3) as usize;
    let compression_alg = code_l & 0x03;
    if compression_alg != 0 {
        return Err(Error::Unsupported(
            "suffix compression (LZ4/lowercase-ASCII) not supported in this slice",
        ));
    }
    let mut suffix_bytes = vec![0u8; num_suffix_bytes];
    r.read_bytes(&mut suffix_bytes)?;

    let num_suffix_length_bytes_raw = r.read_vint()? as u32;
    let all_equal = (num_suffix_length_bytes_raw & 1) != 0;
    let num_suffix_length_bytes = (num_suffix_length_bytes_raw >> 1) as usize;
    let mut suffix_length_bytes = vec![0u8; num_suffix_length_bytes];
    if all_equal {
        let b = r.read_byte()?;
        suffix_length_bytes.fill(b);
    } else {
        r.read_bytes(&mut suffix_length_bytes)?;
    }

    let num_stat_bytes = r.read_vint()? as usize;
    let mut stat_bytes = vec![0u8; num_stat_bytes];
    r.read_bytes(&mut stat_bytes)?;

    // Per-term postings metadata: read past to stay aligned, never decoded
    // (see module doc).
    let num_meta_bytes = r.read_vint()? as usize;
    r.skip(num_meta_bytes)?;

    let mut suffix_lengths_reader = SliceInput::new(&suffix_length_bytes);
    let mut suffixes_reader = SliceInput::new(&suffix_bytes);
    let mut stats_reader = SliceInput::new(&stat_bytes);

    let mut singleton_run_length: u32 = 0;
    let mut entries = Vec::with_capacity(ent_count as usize);
    for _ in 0..ent_count {
        let suffix_len = suffix_lengths_reader.read_vint()? as usize;
        let mut term = vec![0u8; suffix_len];
        suffixes_reader.read_bytes(&mut term)?;

        let (doc_freq, total_term_freq) = if singleton_run_length > 0 {
            singleton_run_length -= 1;
            (1, 1)
        } else {
            let token = stats_reader.read_vint()?;
            if token & 1 == 1 {
                singleton_run_length = (token as u32) >> 1;
                (1, 1)
            } else {
                let doc_freq = (token as u32) >> 1;
                let total_term_freq = if index_options == IndexOptions::Docs {
                    doc_freq as i64
                } else {
                    doc_freq as i64 + stats_reader.read_vlong()?
                };
                (doc_freq as i32, total_term_freq)
            }
        };

        entries.push((
            term,
            TermStats {
                doc_freq,
                total_term_freq,
            },
        ));
    }

    Ok(entries)
}

/// Opens a `.tim`/`.tip`/`.tmd` triple already read whole into memory,
/// decoding every field's single-block term dictionary eagerly (see the
/// module doc for the size/shape scope this covers).
pub fn open(
    tim: &[u8],
    tip: &[u8],
    tmd: &[u8],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
) -> Result<BlockTreeFields> {
    let mut tim_input = SliceInput::new(tim);
    let tim_header = codec_util::check_index_header(
        &mut tim_input,
        TERMS_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut tip_input = SliceInput::new(tip);
    codec_util::check_index_header(
        &mut tip_input,
        TERMS_INDEX_CODEC_NAME,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    let mut tmd_input = SliceInput::new(tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    // PostingsReaderBase.init: the postings writer's own header, embedded in
    // the same .tmd stream right after BlockTree's index header.
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
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
    if num_fields < 0 {
        return Err(Error::InvalidNumFields(num_fields));
    }

    let mut fields = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint()?;
        let num_terms = tmd_input.read_vlong()?;
        if num_terms <= 0 {
            return Err(Error::IllegalNumTerms(field_number));
        }
        let field_info = field_infos
            .field_by_number(field_number)
            .ok_or(Error::InvalidFieldNumber(field_number))?;

        let (sum_total_term_freq, sum_doc_freq) =
            read_freq_pair(&mut tmd_input, field_info.index_options)?;
        let doc_count = tmd_input.read_vint()?;
        let min_term = read_bytes_ref(&mut tmd_input)?;
        let mut max_term = read_bytes_ref(&mut tmd_input)?;
        if num_terms == 1 {
            max_term = min_term.clone();
        }

        if !(0..=max_doc).contains(&doc_count) {
            return Err(Error::InvalidDocCount { doc_count, max_doc });
        }
        if sum_doc_freq < doc_count as i64 {
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

        let index_start = tmd_input.read_vlong()? as usize;
        let root_fp = tmd_input.read_vlong()? as usize;
        let index_end = tmd_input.read_vlong()? as usize;

        if index_end > tip.len() || index_start > index_end {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "field index region out of bounds".into(),
            )));
        }
        let index_slice = &tip[index_start..index_end];
        let (output_fp, has_terms) = read_root_node(index_slice, root_fp)?;
        if !has_terms {
            return Err(Error::Unsupported(
                "root block with no terms (all sub-blocks) not supported in this slice",
            ));
        }

        let entries = decode_block(tim, output_fp as usize, field_info.index_options)?;
        if entries.len() as i64 != num_terms {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "decoded {} terms but field metadata says numTerms={num_terms}",
                entries.len()
            ))));
        }

        if fields
            .iter()
            .any(|(n, _): &(String, FieldTerms)| n == &field_info.name)
        {
            return Err(Error::DuplicateField(field_info.name.clone()));
        }
        fields.push((
            field_info.name.clone(),
            FieldTerms {
                num_terms,
                sum_total_term_freq,
                sum_doc_freq,
                doc_count,
                min_term,
                max_term,
                entries,
            },
        ));
    }

    let index_length = tmd_input.read_i64()?;
    let terms_length = tmd_input.read_i64()?;
    codec_util::check_footer(&mut tmd_input, tmd.len())?;

    if index_length as usize > tip.len() || terms_length as usize > tim.len() {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "recorded .tip/.tim length exceeds file size".into(),
        )));
    }
    codec_util::retrieve_checksum(tip)?;
    codec_util::retrieve_checksum(tim)?;

    Ok(BlockTreeFields { fields })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::FieldInfo;
    use lucene_store::data_output::DataOutput;

    fn field_info(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options,
            doc_values_type: crate::field_infos::DocValuesType::None,
            doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: crate::field_infos::VectorEncoding::Float32,
            vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    /// Hand-builds a single-field, single-block `.tim`/`.tip`/`.tmd` triple
    /// (terms `["a", "ab", "b"]`, docFreq/totalTermFreq = 1/1, 2/3, 1/1) —
    /// this port's own encoder, test-only, to exercise error/boundary paths
    /// a real (small) fixture never reaches. Mirrors the pattern used by
    /// `codec_util.rs`/`segment_info.rs`'s own test-only encoders.
    struct Builder {
        id: [u8; ID_LENGTH],
        suffix: String,
    }

    impl Builder {
        fn new() -> Self {
            Builder {
                id: [7u8; ID_LENGTH],
                suffix: String::new(),
            }
        }

        fn build(
            &self,
            index_options: IndexOptions,
            terms: &[(&str, u32, u64)],
        ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            // .tim
            let mut tim = Vec::new();
            codec_util::write_index_header(
                &mut tim,
                TERMS_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            let block_fp = tim.len();

            let ent_count = terms.len() as u32;
            let code = (ent_count << 1) | 1; // isLastInFloor
            tim.write_vint(code as i32);

            let mut suffix_bytes = Vec::new();
            let mut suffix_lengths = Vec::new();
            let mut stats = Vec::new();
            for (term, doc_freq, total_term_freq) in terms {
                suffix_bytes.extend_from_slice(term.as_bytes());
                suffix_lengths.write_vint(term.len() as i32);
                let token = (*doc_freq as i32) << 1; // never singleton-run-encoded, for test simplicity
                stats.write_vint(token);
                if index_options != IndexOptions::Docs {
                    stats.write_vlong((*total_term_freq as i64) - (*doc_freq as i64));
                }
            }

            let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // isLeafBlock, NO_COMPRESSION
            tim.write_vlong(code_l as i64);
            tim.write_bytes(&suffix_bytes);

            tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
            tim.write_bytes(&suffix_lengths);

            tim.write_vint(stats.len() as i32);
            tim.write_bytes(&stats);

            tim.write_vint(0); // metadata bytes (none — postings not exercised)

            codec_util::write_footer(&mut tim);

            // .tip: root node (SIGN_NO_CHILDREN), hasTerms, no floor.
            let mut tip = Vec::new();
            codec_util::write_index_header(
                &mut tip,
                TERMS_INDEX_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            let index_start = tip.len();
            let root_fp = 0usize;
            let output_fp_bytes = 8usize; // keep it simple: always 8 bytes
            let header = (SIGN_NO_CHILDREN as u8)
                | ((output_fp_bytes as u8 - 1) << 2)
                | (LEAF_NODE_HAS_TERMS as u8);
            tip.push(header);
            tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
            tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad
            let index_end = tip.len();
            codec_util::write_footer(&mut tip);

            // .tmd
            let mut tmd = Vec::new();
            codec_util::write_index_header(
                &mut tmd,
                TERMS_META_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            codec_util::write_index_header(
                &mut tmd,
                POSTINGS_TERMS_CODEC,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            tmd.write_vint(POSTINGS_BLOCK_SIZE);

            tmd.write_vint(1); // numFields
            tmd.write_vint(0); // field number
            let num_terms = terms.len() as i64;
            tmd.write_vlong(num_terms);
            let sum_doc_freq: i64 = terms.iter().map(|(_, d, _)| *d as i64).sum();
            let sum_total_term_freq: i64 = if index_options == IndexOptions::Docs {
                sum_doc_freq
            } else {
                terms.iter().map(|(_, _, t)| *t as i64).sum()
            };
            if index_options != IndexOptions::Docs {
                tmd.write_vlong(sum_total_term_freq);
            }
            tmd.write_vlong(sum_doc_freq);
            tmd.write_vint(1); // docCount
            let min_term = terms[0].0.as_bytes();
            let max_term = terms[terms.len() - 1].0.as_bytes();
            tmd.write_vint(min_term.len() as i32);
            tmd.write_bytes(min_term);
            tmd.write_vint(max_term.len() as i32);
            tmd.write_bytes(max_term);
            tmd.write_vlong(index_start as i64);
            tmd.write_vlong(root_fp as i64);
            tmd.write_vlong(index_end as i64);

            tmd.write_i64(index_end as i64); // indexLength
            tmd.write_i64((tim.len()) as i64); // termsLength
            codec_util::write_footer(&mut tmd);

            (tim, tip, tmd)
        }
    }

    #[test]
    fn seek_exact_found_and_not_found() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();
        assert_eq!(field.num_terms, 3);
        assert_eq!(field.sum_doc_freq, 4);
        assert_eq!(field.sum_total_term_freq, 5);
        assert_eq!(field.min_term, b"a");
        assert_eq!(field.max_term, b"b");

        assert_eq!(
            field.seek_exact(b"ab"),
            Some(TermStats {
                doc_freq: 2,
                total_term_freq: 3
            })
        );
        assert_eq!(
            field.seek_exact(b"a"),
            Some(TermStats {
                doc_freq: 1,
                total_term_freq: 1
            })
        );
        assert_eq!(field.seek_exact(b"missing"), None);
        assert_eq!(field.seek_exact(b""), None);
        assert!(fields.field("nope").is_none());
    }

    #[test]
    fn single_term_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.min_term, field.max_term);
        assert_eq!(
            field.seek_exact(b"only"),
            Some(TermStats {
                doc_freq: 1,
                total_term_freq: 1
            })
        );
        assert_eq!(field.seek_exact(b"other"), None);
    }

    #[test]
    fn docs_only_index_options_omits_total_term_freq_field() {
        // IndexOptions::Docs never writes a distinct sumTotalTermFreq, and
        // per-term stats never write the extra totalTermFreq vlong either.
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("x", 3, 3), ("y", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.sum_total_term_freq, field.sum_doc_freq);
        assert_eq!(
            field.seek_exact(b"x"),
            Some(TermStats {
                doc_freq: 3,
                total_term_freq: 3
            })
        );
    }

    #[test]
    fn invalid_num_fields_rejected() {
        let mut tmd = Vec::new();
        let id = [1u8; ID_LENGTH];
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(-1); // invalid numFields
        codec_util::write_footer(&mut tmd);

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tim);
        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tip);

        let fis = FieldInfos { fields: vec![] };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::InvalidNumFields(-1)));
    }

    #[test]
    fn unexpected_postings_block_size_rejected() {
        let mut tmd = Vec::new();
        let id = [1u8; ID_LENGTH];
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(128); // wrong block size
        codec_util::write_footer(&mut tmd);

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tim);
        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tip);

        let fis = FieldInfos { fields: vec![] };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::UnexpectedBlockSize { found: 128 }));
    }

    #[test]
    fn multi_child_trie_node_rejected() {
        // sign bits = SIGN_MULTI_CHILDREN (0x03) at the root -> Unsupported.
        let b = Builder::new();
        let (tim, mut tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        // Overwrite the root node's header byte (right after the .tip index
        // header) with SIGN_MULTI_CHILDREN.
        let mut probe = SliceInput::new(&tip);
        codec_util::check_index_header(
            &mut probe,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            &b.id,
            &b.suffix,
        )
        .unwrap();
        let header_pos = probe.position();
        tip[header_pos] = (tip[header_pos] & !0x03) | 0x03;
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn floor_block_rejected() {
        let b = Builder::new();
        let (mut tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        // Overwrite the block's leading vint (isLastInFloor bit) to false.
        let mut probe = SliceInput::new(&tim);
        codec_util::check_index_header(
            &mut probe,
            TERMS_CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            &b.id,
            &b.suffix,
        )
        .unwrap();
        let pos = probe.position();
        tim[pos] &= !0x01; // clear isLastInFloor
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn empty_terms_block_rejected() {
        let id = [2u8; ID_LENGTH];
        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        let block_fp = tim.len();
        tim.write_vint(1); // entCount=0, isLastInFloor=true -> code = 0<<1|1 = 1
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        let index_start = tip.len();
        let header = LEAF_NODE_HAS_TERMS as u8; // SIGN_NO_CHILDREN, 1-byte fp
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes());
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        let mut tmd = Vec::new();
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1);
        tmd.write_vint(0);
        tmd.write_vlong(1); // numTerms must be >0 to pass that check; block itself will be empty
        tmd.write_vlong(0); // sumDocFreq (Docs aliasing)
        tmd.write_vint(0); // docCount
        tmd.write_vint(0);
        tmd.write_bytes(&[]);
        tmd.write_vint(0);
        tmd.write_bytes(&[]);
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0);
        tmd.write_vlong(index_end as i64);
        tmd.write_i64(index_end as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn read_bytes_ref_rejects_negative_length() {
        let mut buf = Vec::new();
        buf.write_vint(-1);
        let mut input = SliceInput::new(&buf);
        let err = read_bytes_ref(&mut input).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn read_root_node_eight_byte_output_fp() {
        // fpBytesMinus1 == 7 forces a fresh 8-byte read at fp+1.
        let mut slice = Vec::new();
        let header: u8 = LEAF_NODE_HAS_TERMS as u8 | (7 << 2); // sign=0, fpBytesMinus1=7
        slice.push(header);
        let big_fp: u64 = 0x0102_0304_0506_0708;
        slice.extend_from_slice(&big_fp.to_le_bytes()); // read fresh at fp+1
        slice.extend_from_slice(&0u64.to_le_bytes()); // over-read padding

        let (output_fp, has_terms) = read_root_node(&slice, 0).unwrap();
        assert_eq!(output_fp, big_fp);
        assert!(has_terms);
    }

    #[test]
    fn read_root_node_rejects_truncated_slice() {
        let slice = [0u8; 4];
        let err = read_root_node(&slice, 0).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn decode_block_singleton_run_length_and_all_equal_suffixes() {
        // Hand-build a block with allEqual suffix lengths and a singleton
        // run (three consecutive docFreq=1/totalTermFreq=1 terms encoded via
        // the run-length token) to exercise both branches `Builder` (which
        // always emits per-entry non-run tokens and variable suffix
        // lengths) never reaches.
        let mut tim = Vec::new();
        let terms = ["aa", "bb", "cc"];
        let ent_count = terms.len() as u32;
        tim.write_vint(((ent_count << 1) | 1) as i32); // isLastInFloor

        let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // leaf, no compression
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&suffix_bytes);

        // allEqual suffix lengths: all terms are 2 bytes. The logical array
        // size is still entCount (one vint-encoded length per entry) even
        // though only a single physical byte is written on disk.
        tim.write_vint(((ent_count as i32) << 1) | 1);
        tim.write_byte(2);

        // stats: one run-length token covering all three (docFreq=1 each).
        let mut stats = Vec::new();
        stats.write_vint((3 << 1) | 1); // token&1==1 -> singleton run of length 3
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        tim.write_vint(0); // no postings metadata

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs).unwrap();
        assert_eq!(entries.len(), 3);
        for (term, stats) in &entries {
            assert_eq!(term.len(), 2);
            assert_eq!(stats.doc_freq, 1);
            assert_eq!(stats.total_term_freq, 1);
        }
        assert_eq!(entries[0].0, b"aa");
        assert_eq!(entries[2].0, b"cc");
    }

    #[test]
    fn invalid_field_number_rejected() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        // FieldInfos has no field numbered 0.
        let fis = FieldInfos {
            fields: vec![field_info(9, "other", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
        assert!(matches!(err, Error::InvalidFieldNumber(0)));
    }

    #[test]
    fn invalid_doc_count_rejected() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        // docCount (1, baked into Builder::build) exceeds maxDoc=0.
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidDocCount { .. }));
    }

    #[test]
    fn duplicate_field_rejected() {
        let id = [3u8; ID_LENGTH];
        let mut tmd = Vec::new();
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(2); // numFields

        // Build a single shared .tim block (one term "a") and .tip root node
        // that both field records point at, so the same field *name* is
        // reachable twice (two field numbers mapping to fields named "f").
        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        let block_fp = tim.len();
        tim.write_vint((1 << 1) | 1);
        tim.write_vlong(((1u64 << 3) | 0x04) as i64);
        tim.write_bytes(b"a");
        tim.write_vint(1 << 1);
        tim.write_bytes(&[1]);
        let mut stats = Vec::new();
        stats.write_vint(1 << 1); // docFreq=1, non-singleton token
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);
        tim.write_vint(0);
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        let index_start = tip.len();
        let header = LEAF_NODE_HAS_TERMS as u8;
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes());
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        for field_number in [0i32, 1i32] {
            tmd.write_vint(field_number);
            tmd.write_vlong(1); // numTerms
            tmd.write_vlong(1); // sumDocFreq (Docs aliasing)
            tmd.write_vint(1); // docCount
            tmd.write_vint(1);
            tmd.write_bytes(b"a");
            tmd.write_vint(1);
            tmd.write_bytes(b"a");
            tmd.write_vlong(index_start as i64);
            tmd.write_vlong(0);
            tmd.write_vlong(index_end as i64);
        }
        tmd.write_i64(index_end as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![
                field_info(0, "f", IndexOptions::Docs),
                field_info(1, "f", IndexOptions::Docs),
            ],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::DuplicateField(_)));
    }

    #[test]
    fn index_region_out_of_bounds_rejected() {
        let b = Builder::new();
        let (tim, tip, _tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        let id = b.id;
        let suffix = b.suffix.clone();

        // Hand-build a .tmd whose indexEnd points past the end of .tip.
        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        codec_util::write_index_header(
            &mut tmd,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1);
        tmd.write_vint(0);
        tmd.write_vlong(1);
        tmd.write_vlong(1);
        tmd.write_vint(1);
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vlong(0);
        tmd.write_vlong(0);
        tmd.write_vlong((tip.len() + 100) as i64); // out of bounds indexEnd
        tmd.write_i64(tip.len() as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }
}
