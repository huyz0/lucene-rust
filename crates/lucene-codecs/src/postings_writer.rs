//! Write side for a **single field's** term dictionary + postings —
//! `.doc`/`.tim`/`.tip`/`.tmd` — narrowly scoped to be the exact inverse of
//! what `crate::postings`/`crate::blocktree`'s existing (unmodified) read
//! side already decodes for the shapes below. Nothing here duplicates that
//! decode logic; this module only emits bytes, and the differential tests in
//! `crates/lucene-search` prove those bytes read back correctly through the
//! real, pre-existing `blocktree::open`/`postings::DocInput` functions.
//!
//! # Scope (read this before assuming more than it proves)
//!
//! - **One field per call.** Real Lucene's `.tmd`/`.tim`/`.tip` interleave
//!   every field of a segment; this writer emits exactly one field's record
//!   (`numFields = 1` in `.tmd`) and one physical `.tim` block for that
//!   field's whole term dictionary.
//! - **One `.tim` block, one trie node.** Every term must fit in a single
//!   leaf block (`Lucene103BlockTreeTermsWriter`'s block-splitting,
//!   floor sub-blocks, and multi-level tries are not implemented) and the
//!   `.tip` index is a single root `SIGN_NO_CHILDREN` node with `hasTerms`
//!   set and no floor data — i.e. the same trivial single-block/single-node
//!   shape `blocktree.rs`'s own test-only `Builder` already exercises for
//!   read-side tests, except this module's metadata is real (computed from
//!   actual `.doc` file offsets), not placeholder.
//! - **`docFreq < BLOCK_SIZE` (256) for every term.** No full `ForUtil`/
//!   `PForUtil`-encoded blocks are ever written — every non-singleton term's
//!   postings are the group-varint "tail block" encoding alone
//!   (`Lucene104PostingsWriter`'s `flushDocBlock(true)` branch that never
//!   reaches `docBufferUpto == BLOCK_SIZE`). A term at or above `BLOCK_SIZE`
//!   docs is rejected with [`Error::Unsupported`] rather than silently
//!   producing wrong bytes.
//! - **Term frequency only — no positions/offsets/payloads.** Only
//!   `IndexOptions::Docs`/`DocsAndFreqs` are accepted; `.pos`/`.pay` are
//!   never written. This mirrors `flush_stored_only_segment`'s own
//!   historical "start with the smallest defensible slice" precedent (see
//!   `crate::term_vectors::write_best_speed`'s positions-only cut for
//!   another example of the same policy).
//! - **`docFreq == 1` is pulsed into the term dictionary**, exactly like the
//!   real writer (`Lucene104PostingsWriter.java:568-577`): no `.doc` bytes at
//!   all for a singleton term, matching what `postings::singleton_postings`
//!   already expects to read back.
//!
//! # Caller obligations (not re-validated beyond what's cheap to check)
//!
//! `terms` must already be sorted ascending by term bytes with no
//! duplicates, and each term's `docs` must be sorted ascending by doc ID with
//! no duplicates and every `freq >= 1` — the same invariant
//! `indexing_chain::InMemoryInvertedIndex`'s `BTreeMap`/per-term sort already
//! guarantees for its `Vec<PostingEntry>`. Violating this produces incorrect
//! (but not memory-unsafe) output; [`write_single_field`] only checks the
//! cheap structural invariants explicitly listed above (sortedness of terms,
//! `docFreq` bound, `index_options`).
//!
//! # Wire format written (mirrors `crate::blocktree`/`crate::postings`'s own
//! module docs, writer side)
//!
//! - `.doc`: `IndexHeader(codec="Lucene104PostingsWriterDoc")`, then, for
//!   each non-singleton term in order, its tail-block bytes (group-varint
//!   `(docDelta << 1) | (freq == 1 ? 1 : 0)` values when `index_options`
//!   carries freqs, else plain `docDelta`, followed by one plain vint per
//!   `freq != 1` doc, in doc order) — see `crate::postings::read_tail_block`
//!   for the exact inverse. `Footer`.
//! - `.tim`: `IndexHeader(codec="BlockTreeTermsDict")`, one physical block
//!   (`entCount << 1 | 1` code, `isLeafBlock` + `NO_COMPRESSION` code,
//!   suffix bytes, suffix lengths, per-term stats, per-term postings
//!   metadata — see [`write_term_metadata`]), `Footer`.
//! - `.tip`: `IndexHeader(codec="BlockTreeTermsIndex")`, one
//!   `SIGN_NO_CHILDREN`/`hasTerms`/no-floor root node pointing at the `.tim`
//!   block, `Footer`.
//! - `.tmd`: `IndexHeader(codec="BlockTreeTermsMeta")`, the postings writer's
//!   own embedded header (`IndexHeader(codec="Lucene104PostingsWriterTerms")`,
//!   `indexBlockSize = 256`), `numFields = 1`, the one field's record
//!   (`fieldNumber, numTerms, sumTotalTermFreq/sumDocFreq, docCount, minTerm/maxTerm,
//!   indexStart/rootFP/indexEnd`), `indexLength`, `termsLength`, `Footer`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_output::DataOutput;

use crate::blocktree::{
    LEAF_NODE_HAS_TERMS, POSTINGS_BLOCK_SIZE, POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT,
    SIGN_NO_CHILDREN, TERMS_CODEC_NAME, TERMS_INDEX_CODEC_NAME, TERMS_META_CODEC_NAME,
    VERSION_CURRENT as BLOCKTREE_VERSION_CURRENT,
};
use crate::field_infos::IndexOptions;
use crate::postings::{
    write_group_vints, BLOCK_SIZE, DOC_CODEC, VERSION_CURRENT as DOC_VERSION_CURRENT,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("write_single_field: terms must be non-empty")]
    EmptyTerms,
    #[error("write_single_field: terms out of order or duplicated at index {0}")]
    TermsNotSorted(usize),
    #[error("write_single_field: term at index {0} has no postings (docFreq == 0)")]
    EmptyPostings(usize),
    #[error("write_single_field: term at index {index} has non-ascending/duplicate doc IDs")]
    DocIdsNotSorted { index: usize },
    #[error("write_single_field: term at index {index} has freq < 1")]
    NonPositiveFreq { index: usize },
    #[error(
        "write_single_field: term at index {index} has docFreq {doc_freq} >= BLOCK_SIZE \
         ({BLOCK_SIZE}); multi-block terms are not supported by this writer"
    )]
    DocFreqTooLarge { index: usize, doc_freq: usize },
    #[error("write_single_field: only IndexOptions::Docs/DocsAndFreqs is supported, got {0:?}")]
    UnsupportedIndexOptions(IndexOptions),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One term's postings: `docs` is `(doc_id, freq)` pairs, ascending doc-ID
/// order, no duplicates, every `freq >= 1` (see the module doc's "Caller
/// obligations").
#[derive(Debug, Clone)]
pub struct TermPostings {
    pub term: Vec<u8>,
    pub docs: Vec<(i32, i32)>,
}

/// Input to [`write_single_field`]: one field's whole term dictionary,
/// already fully materialized and sorted.
pub struct FieldPostingsInput<'a> {
    pub field_number: i32,
    pub index_options: IndexOptions,
    /// `docCount`: number of distinct docs this field occurs in at least
    /// once across the whole segment — the caller's responsibility to
    /// compute (usually `terms.iter().flat_map(|t| &t.docs).map(|(d,_)| d)`'s
    /// distinct count, but the real writer just tracks it incrementally).
    pub doc_count: i32,
    pub terms: &'a [TermPostings],
}

/// The four files this writer produces for one field.
#[derive(Debug, Clone, Default)]
pub struct Output {
    pub doc: Vec<u8>,
    pub tim: Vec<u8>,
    pub tip: Vec<u8>,
    pub tmd: Vec<u8>,
}

/// Writes `.doc`/`.tim`/`.tip`/`.tmd` bytes for `input`'s single field — see
/// the module doc for the exact scope and wire format. `segment_id`/
/// `segment_suffix` must match what the caller will later open the files
/// with (`blocktree::open`/`postings::DocInput::open` both check them).
pub fn write_single_field(
    input: &FieldPostingsInput<'_>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    if !matches!(
        input.index_options,
        IndexOptions::Docs | IndexOptions::DocsAndFreqs
    ) {
        return Err(Error::UnsupportedIndexOptions(input.index_options));
    }
    if input.terms.is_empty() {
        return Err(Error::EmptyTerms);
    }
    for (i, w) in input.terms.windows(2).enumerate() {
        if w[0].term >= w[1].term {
            return Err(Error::TermsNotSorted(i + 1));
        }
    }
    for (i, t) in input.terms.iter().enumerate() {
        if t.docs.is_empty() {
            return Err(Error::EmptyPostings(i));
        }
        if t.docs.len() >= BLOCK_SIZE as usize {
            return Err(Error::DocFreqTooLarge {
                index: i,
                doc_freq: t.docs.len(),
            });
        }
        for (j, &(_, freq)) in t.docs.iter().enumerate() {
            if freq < 1 {
                return Err(Error::NonPositiveFreq { index: i });
            }
            if j > 0 && t.docs[j - 1].0 >= t.docs[j].0 {
                return Err(Error::DocIdsNotSorted { index: i });
            }
        }
    }

    let index_has_freq = input.index_options != IndexOptions::Docs;

    // ---- .doc ----
    let mut doc = Vec::new();
    codec_util::write_index_header(
        &mut doc,
        DOC_CODEC,
        DOC_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    // `doc_start_fp[i]` is the byte offset (into `doc`, i.e. relative to the
    // whole file including its header — the same absolute convention
    // `postings::TermMetadata::doc_start_fp` decodes into) where term `i`'s
    // tail block begins, or `0` for a singleton term (never read for
    // singletons, see `postings::singleton_postings`).
    let mut doc_start_fp = vec![0u64; input.terms.len()];
    for (i, t) in input.terms.iter().enumerate() {
        if t.docs.len() == 1 {
            continue;
        }
        doc_start_fp[i] = doc.len() as u64;
        write_tail_block(&mut doc, &t.docs, index_has_freq);
    }
    codec_util::write_footer(&mut doc);

    // ---- .tim ----
    let mut tim = Vec::new();
    codec_util::write_index_header(
        &mut tim,
        TERMS_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let block_fp = tim.len();

    let ent_count = input.terms.len() as u32;
    let code = (ent_count << 1) | 1; // isLastInFloor
    tim.write_vint(code as i32);

    let mut suffix_bytes = Vec::new();
    let mut suffix_lengths = Vec::new();
    let mut stats = Vec::new();
    for t in input.terms {
        suffix_bytes.write_bytes(&t.term);
        suffix_lengths.write_vint(t.term.len() as i32);
        let doc_freq = t.docs.len() as u32;
        let total_term_freq: i64 = t.docs.iter().map(|&(_, f)| f as i64).sum();
        stats.write_vint((doc_freq << 1) as i32); // never singleton-run-encoded
        if input.index_options != IndexOptions::Docs {
            stats.write_vlong(total_term_freq - doc_freq as i64);
        }
    }

    let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // isLeafBlock, NO_COMPRESSION
    tim.write_vlong(code_l as i64);
    tim.write_bytes(&suffix_bytes);

    tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
    tim.write_bytes(&suffix_lengths);

    tim.write_vint(stats.len() as i32);
    tim.write_bytes(&stats);

    let mut meta = Vec::new();
    write_term_metadata(&mut meta, input.terms, &doc_start_fp);
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);

    codec_util::write_footer(&mut tim);

    // ---- .tip ----
    let mut tip = Vec::new();
    codec_util::write_index_header(
        &mut tip,
        TERMS_INDEX_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let index_start = tip.len();
    let root_fp = 0usize;
    let output_fp_bytes = 8usize; // keep it simple: always 8 bytes, same as blocktree.rs's test Builder
    let header =
        (SIGN_NO_CHILDREN as u8) | ((output_fp_bytes as u8 - 1) << 2) | (LEAF_NODE_HAS_TERMS as u8);
    tip.push(header);
    tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
    tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad, `load_node`'s SIGN_NO_CHILDREN reads up to fp+1..fp+9
    let index_end = tip.len();
    codec_util::write_footer(&mut tip);

    // ---- .tmd ----
    let mut tmd = Vec::new();
    codec_util::write_index_header(
        &mut tmd,
        TERMS_META_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    codec_util::write_index_header(
        &mut tmd,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    tmd.write_vint(POSTINGS_BLOCK_SIZE);

    tmd.write_vint(1); // numFields
    tmd.write_vint(input.field_number);
    let num_terms = input.terms.len() as i64;
    tmd.write_vlong(num_terms);
    let sum_doc_freq: i64 = input.terms.iter().map(|t| t.docs.len() as i64).sum();
    let sum_total_term_freq: i64 = if input.index_options == IndexOptions::Docs {
        sum_doc_freq
    } else {
        input
            .terms
            .iter()
            .flat_map(|t| t.docs.iter())
            .map(|&(_, f)| f as i64)
            .sum()
    };
    if input.index_options != IndexOptions::Docs {
        tmd.write_vlong(sum_total_term_freq);
    }
    tmd.write_vlong(sum_doc_freq);
    tmd.write_vint(input.doc_count);
    let min_term = &input.terms[0].term;
    let max_term = &input.terms[input.terms.len() - 1].term;
    tmd.write_vint(min_term.len() as i32);
    tmd.write_bytes(min_term);
    tmd.write_vint(max_term.len() as i32);
    tmd.write_bytes(max_term);
    tmd.write_vlong(index_start as i64);
    tmd.write_vlong(root_fp as i64);
    tmd.write_vlong(index_end as i64);

    tmd.write_i64(index_end as i64); // indexLength
    tmd.write_i64(tim.len() as i64); // termsLength
    codec_util::write_footer(&mut tmd);

    Ok(Output { doc, tim, tip, tmd })
}

/// Writes one term's `.doc` tail-block bytes (`docFreq < BLOCK_SIZE`, the
/// only shape this writer produces) — the exact inverse of
/// `crate::postings::read_tail_block` with `prev_doc_id == -1` (a term's
/// postings never share a running doc-ID base with another term; only
/// full-block chaining within one term does that, which this writer never
/// emits).
fn write_tail_block(out: &mut Vec<u8>, docs: &[(i32, i32)], index_has_freq: bool) {
    let mut raw = Vec::with_capacity(docs.len());
    let mut prev = -1i32;
    for &(doc_id, freq) in docs {
        let delta = (doc_id - prev) as u32;
        prev = doc_id;
        if index_has_freq {
            raw.push((delta << 1) | if freq == 1 { 1 } else { 0 });
        } else {
            raw.push(delta);
        }
    }
    write_group_vints(out, &raw);
    if index_has_freq {
        for &(_, freq) in docs {
            if freq != 1 {
                out.write_vint(freq);
            }
        }
    }
}

/// Writes every term's per-term postings metadata bytes — the write-side
/// inverse of `crate::postings::decode_term_metadata`'s no-positions branch
/// (`IndexOptions::Docs`/`DocsAndFreqs` only, matching this writer's own
/// scope restriction). Always takes the bit-clear ("absolute-ish
/// `docStartFP` delta") branch, never the zigzag-singleton-delta branch —
/// this writer has no need for that alternate encoding's extra compactness.
///
/// `doc_start_fp` deltas are threaded exactly like
/// `SegmentTermsEnumFrame.metaDataUpto`/`absolute` on the read side: the
/// first term in the (only) block decodes against `TermMetadata::EMPTY`
/// (`doc_start_fp == 0`), every subsequent term against the *previous*
/// term's already-written `doc_start_fp` — so this writer must emit the same
/// running delta, not each term's absolute offset.
fn write_term_metadata(out: &mut Vec<u8>, terms: &[TermPostings], doc_start_fp: &[u64]) {
    let mut base_doc_start_fp = 0u64;
    for (t, &fp) in terms.iter().zip(doc_start_fp) {
        let doc_freq = t.docs.len();
        // Singleton terms never advance `doc_start_fp` (no `.doc` bytes are
        // written for them, see `write_single_field`), so their delta is 0
        // and the running base is left unchanged for the next term.
        let this_fp = if doc_freq == 1 { base_doc_start_fp } else { fp };
        let delta = this_fp.wrapping_sub(base_doc_start_fp);
        out.write_vlong(((delta << 1) as i64) & !1); // bit 0 clear: absolute-ish delta branch
        if doc_freq == 1 {
            out.write_vint(t.docs[0].0);
        }
        base_doc_start_fp = this_fp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocktree::{self, FieldTerms};
    use crate::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, VectorEncoding,
        VectorSimilarityFunction,
    };
    use crate::postings::DocInput;

    const SEG_ID: [u8; ID_LENGTH] = [9u8; ID_LENGTH];
    const SUFFIX: &str = "";

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
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    fn open_written<'a>(
        output: &'a Output,
        field_infos: &FieldInfos,
        max_doc: i32,
    ) -> (blocktree::BlockTreeFields, DocInput<'a>) {
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            field_infos,
            &SEG_ID,
            SUFFIX,
            max_doc,
        )
        .expect("write_single_field's own bytes must open cleanly");
        let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
        (fields, doc_in)
    }

    /// Mixed singleton/multi-doc terms, round-tripped through the existing
    /// unmodified `blocktree::open` + `postings::DocInput` read side (no
    /// query layer here — see
    /// `crates/lucene-search/tests/postings_writer_round_trip.rs` for the
    /// required end-to-end `search_term_query` proof, which lives in
    /// `lucene-search` rather than here since this crate must not depend
    /// upward on `lucene-search`, see the `architecture` skill).
    #[test]
    fn mixed_singleton_and_multi_doc_terms_round_trip() {
        let terms = vec![
            TermPostings {
                term: b"fox".to_vec(),
                docs: vec![(1, 2), (4, 1), (7, 3)],
            },
            TermPostings {
                term: b"quick".to_vec(),
                docs: vec![(4, 1)], // singleton
            },
            TermPostings {
                term: b"the".to_vec(),
                docs: vec![(0, 1), (1, 1), (4, 2), (7, 1)],
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 8,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

        let fis = FieldInfos {
            fields: vec![field_info(0, "body", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 8);

        let field = fields.field("body").unwrap();
        assert_eq!(field.num_terms, 3);
        assert_eq!(field.min_term, b"fox");
        assert_eq!(field.max_term, b"the");

        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1, 4, 7]);
        assert_eq!(postings.freqs, vec![2, 1, 3]);

        let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![4]);
        assert_eq!(postings.freqs, vec![1]);

        let postings = field.postings(b"the", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1, 4, 7]);
        assert_eq!(postings.freqs, vec![1, 1, 2, 1]);

        assert!(field.seek_exact(b"missing").is_none());
    }

    /// Byte-level correctness on `docFreq`/`totalTermFreq`/`seek_exact`
    /// alone (no query layer), for `IndexOptions::Docs` (no freqs at all —
    /// `totalTermFreq == docFreq` aliasing) to make sure that branch, not
    /// just `DocsAndFreqs`, round-trips.
    #[test]
    fn docs_only_index_options_round_trips() {
        let terms = vec![
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(0, 1), (2, 1)],
            },
            TermPostings {
                term: b"b".to_vec(),
                docs: vec![(1, 1)],
            },
        ];
        let input = FieldPostingsInput {
            field_number: 3,
            index_options: IndexOptions::Docs,
            doc_count: 3,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(3, "f", IndexOptions::Docs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 3);
        let field: &FieldTerms = fields.field("f").unwrap();
        assert_eq!(
            field.seek_exact(b"a"),
            Some(blocktree::TermStats {
                doc_freq: 2,
                total_term_freq: 2
            })
        );
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 2]);
        assert_eq!(postings.freqs, vec![1, 1]); // freqs default to 1 when the field has no freqs

        let postings = field.postings(b"b", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1]);
    }

    /// All terms singleton (`docFreq == 1`): no `.doc` bytes are needed at
    /// all — `postings()` must still resolve every term purely from the
    /// term-dictionary metadata (`postings::singleton_postings`).
    #[test]
    fn all_singleton_terms_need_no_doc_file() {
        let terms = vec![
            TermPostings {
                term: b"alpha".to_vec(),
                docs: vec![(2, 1)],
            },
            TermPostings {
                term: b"beta".to_vec(),
                docs: vec![(5, 4)],
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            &fis,
            &SEG_ID,
            SUFFIX,
            6,
        )
        .unwrap();
        let field = fields.field("f").unwrap();
        // No `.doc` file opened at all -- `doc_in: None` is fine since every
        // term here is a singleton.
        let postings = field.postings(b"beta", None).unwrap().unwrap();
        assert_eq!(postings.docs, vec![5]);
        assert_eq!(postings.freqs, vec![4]);
    }

    #[test]
    fn rejects_empty_terms() {
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 0,
            terms: &[],
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::EmptyTerms)
        ));
    }

    #[test]
    fn rejects_unsorted_terms() {
        let terms = vec![
            TermPostings {
                term: b"b".to_vec(),
                docs: vec![(0, 1)],
            },
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(0, 1)],
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::TermsNotSorted(1))
        ));
    }

    #[test]
    fn rejects_duplicate_terms() {
        let terms = vec![
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(0, 1)],
            },
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(1, 1)],
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::TermsNotSorted(1))
        ));
    }

    #[test]
    fn rejects_empty_postings_for_a_term() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 0,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::EmptyPostings(0))
        ));
    }

    #[test]
    fn rejects_non_ascending_doc_ids() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(2, 1), (1, 1)],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 3,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::DocIdsNotSorted { index: 0 })
        ));
    }

    #[test]
    fn rejects_duplicate_doc_ids() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(1, 1), (1, 1)],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::DocIdsNotSorted { index: 0 })
        ));
    }

    #[test]
    fn rejects_non_positive_freq() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 0), (1, 1)],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::NonPositiveFreq { index: 0 })
        ));
    }

    #[test]
    fn rejects_docfreq_at_or_above_block_size() {
        let docs: Vec<(i32, i32)> = (0..BLOCK_SIZE).map(|d| (d, 1)).collect();
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs,
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: BLOCK_SIZE,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::DocFreqTooLarge {
                index: 0,
                doc_freq
            }) if doc_freq == BLOCK_SIZE as usize
        ));
    }

    #[test]
    fn rejects_unsupported_index_options() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::UnsupportedIndexOptions(
                IndexOptions::DocsAndFreqsAndPositions
            ))
        ));
    }

    /// Many terms, each with several docs, all under `BLOCK_SIZE` -- checks
    /// the running `doc_start_fp` delta-threading across more than a
    /// handful of terms (the earlier tests only ever have 2-3 terms).
    #[test]
    fn many_terms_many_docs_each() {
        let mut terms = Vec::new();
        for i in 0..20 {
            let term = format!("term{i:02}").into_bytes();
            let docs: Vec<(i32, i32)> = (0..5).map(|d| (i * 5 + d, (d + 1))).collect();
            terms.push(TermPostings { term, docs });
        }
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 100,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 100);
        let field = fields.field("f").unwrap();
        assert_eq!(field.num_terms, 20);
        for i in 0..20 {
            let term = format!("term{i:02}").into_bytes();
            let postings = field.postings(&term, Some(&doc_in)).unwrap().unwrap();
            let expected_docs: Vec<i32> = (0..5).map(|d| i * 5 + d).collect();
            let expected_freqs: Vec<i32> = (0..5).map(|d| d + 1).collect();
            assert_eq!(postings.docs, expected_docs, "term{i:02}");
            assert_eq!(postings.freqs, expected_freqs, "term{i:02}");
        }
    }
}
