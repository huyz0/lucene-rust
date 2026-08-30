//! `read_positions_for_docs` / `read_occurrences_for_docs`: the
//! `PostingsEnum.advance(doc)` + `nextPosition()`/`startOffset()`/
//! `endOffset()`/`getPayload()` shape, over real writer-produced
//! `.pos`/`.pay` bytes.
//!
//! Every assertion here is **differential against
//! [`FieldTerms::positions`]** -- the whole-term reader that
//! `blocktree_fixtures.rs` already pins against real Lucene's own occurrence
//! list. A wanted-documents walk is correct exactly when it returns what the
//! whole-term walk returns for those documents, so the expectations are never
//! hand-transcribed: they are sliced out of the verified reader's output.
//!
//! The terms are written by this crate's own `postings_writer`, which is what
//! makes multi-block streams (block skipping, the early exit past the last
//! wanted document, the vint tail after full blocks) reachable without
//! hand-assembling PForUtil blocks.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::blocktree::{self, FieldTerms};
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::{DocInput, PayInput, PosInput, Position};
use lucene_codecs::postings_writer::{
    write_single_field, FieldPostingsInput, Output, TermPostings,
};
use lucene_store::codec_util::ID_LENGTH;

const SEG_ID: [u8; ID_LENGTH] = [7u8; ID_LENGTH];
const SUFFIX: &str = "";

fn field_info(index_options: IndexOptions, has_payloads: bool) -> FieldInfo {
    FieldInfo {
        name: "body".to_string(),
        number: 0,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: has_payloads,
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

/// One term, `docs.len()` documents, `freq` occurrences each, with offsets
/// and payloads generated so that no two occurrences look alike.
fn synthetic_term(
    docs: &[i32],
    freq: i32,
    with_offsets: bool,
    with_payloads: bool,
) -> TermPostings {
    let mut positions = Vec::new();
    let mut offsets = Vec::new();
    let mut payloads = Vec::new();
    for (d, &doc) in docs.iter().enumerate() {
        // Positions strictly increasing within a document, and different in
        // every document, so a walk that mixes documents up cannot pass.
        let base = (d as i32 % 17) + 1;
        let doc_positions: Vec<i32> = (0..freq).map(|i| base + i * (1 + (d as i32 % 3))).collect();
        if with_offsets {
            offsets.push(
                doc_positions
                    .iter()
                    .map(|&p| (p * 3, p * 3 + 2 + (doc % 5)))
                    .collect(),
            );
        }
        if with_payloads {
            payloads.push(
                doc_positions
                    .iter()
                    .map(|&p| {
                        // Length varies (including empty), which is what makes
                        // the per-block payload byte-run offsets non-trivial.
                        let len = (p % 4) as usize;
                        vec![(doc % 251) as u8; len]
                    })
                    .collect(),
            );
        }
        positions.push(doc_positions);
    }
    TermPostings {
        term: b"alpha".to_vec(),
        docs: docs.iter().map(|&d| (d, freq)).collect(),
        positions,
        offsets,
        payloads,
    }
}

struct Written {
    output: Output,
    field_infos: FieldInfos,
    max_doc: i32,
}

fn write(term: &TermPostings, index_options: IndexOptions, has_payloads: bool) -> Written {
    let max_doc = term.docs.last().unwrap().0 + 1;
    let terms = vec![term.clone()];
    let input = FieldPostingsInput {
        field_number: 0,
        index_options,
        doc_count: terms[0].docs.len() as i32,
        has_payloads,
        terms: &terms,
    };
    let output = write_single_field(&input, &SEG_ID, SUFFIX).expect("write");
    Written {
        output,
        field_infos: FieldInfos {
            fields: vec![field_info(index_options, has_payloads)],
        },
        max_doc,
    }
}

impl Written {
    fn open(
        &self,
    ) -> (
        blocktree::BlockTreeFields,
        DocInput<'_>,
        PosInput<'_>,
        Option<PayInput<'_>>,
    ) {
        let fields = blocktree::open(
            &self.output.tim,
            &self.output.tip,
            &self.output.tmd,
            &self.field_infos,
            &SEG_ID,
            SUFFIX,
            self.max_doc,
        )
        .expect("open terms");
        let doc_in = DocInput::open(&self.output.doc, &SEG_ID, SUFFIX).expect("open .doc");
        let pos_in = PosInput::open(&self.output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in = (!self.output.pay.is_empty())
            .then(|| PayInput::open(&self.output.pay, &SEG_ID, SUFFIX).expect("open .pay"));
        (fields, doc_in, pos_in, pay_in)
    }
}

/// The whole-term reader's answer, which every wanted-documents answer is
/// checked against.
fn whole_term(
    field: &FieldTerms,
    doc_in: &DocInput<'_>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
) -> Vec<Vec<Position>> {
    field
        .positions(b"alpha", Some(doc_in), pos_in, pay_in)
        .expect("positions")
        .expect("term present")
}

/// Every wanted-document subset of a term with several full blocks and a
/// vint tail must return exactly what the whole-term reader returns for
/// those documents -- positions, offsets and payloads alike.
#[test]
fn wanted_documents_agree_with_the_whole_term_reader() {
    // 200 documents x 5 occurrences = 1000 occurrences: three full 256-wide
    // blocks and a 232-occurrence vint tail. (`docFreq` itself stays under
    // `BLOCK_SIZE`, which is this writer's own limit for a positions-indexing
    // field -- `total_term_freq` has no such ceiling, and it is the one that
    // decides how many `.pos` blocks there are.)
    let docs: Vec<i32> = (0..200).map(|d| d * 2).collect();
    for (index_options, has_payloads) in [
        (IndexOptions::DocsAndFreqsAndPositions, false),
        (IndexOptions::DocsAndFreqsAndPositions, true),
        (IndexOptions::DocsAndFreqsAndPositionsAndOffsets, false),
        (IndexOptions::DocsAndFreqsAndPositionsAndOffsets, true),
    ] {
        let has_offsets = index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let term = synthetic_term(&docs, 5, has_offsets, has_payloads);
        let written = write(&term, index_options, has_payloads);
        let (fields, doc_in, pos_in, pay_in) = written.open();
        let field = fields.field("body").expect("field");
        let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());
        let stats = field.seek_exact(b"alpha").expect("term present");
        let postings = field
            .postings(b"alpha", Some(&doc_in))
            .expect("postings")
            .expect("term present");

        // The whole-term flat reader (`positions_flat`, what the span paths
        // use) must agree with the same expectations -- it shares the wire
        // decode with `positions` but re-chops it differently.
        let (flat, flat_starts) = field
            .positions_flat(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref())
            .expect("positions_flat")
            .map(|(_postings, positions, starts)| (positions, starts))
            .expect("term present");
        assert_eq!(flat_starts.len(), expected.len() + 1);
        for (d, doc_expected) in expected.iter().enumerate() {
            let got = &flat[flat_starts[d] as usize..flat_starts[d + 1] as usize];
            let want: Vec<i32> = doc_expected.iter().map(|p| p.position).collect();
            assert_eq!(got, want.as_slice(), "positions_flat doc {d}");
        }

        for wanted in [
            vec![],
            vec![0],
            vec![199],
            vec![100],
            vec![0, 199],
            vec![1, 2, 3],
            (0..200).collect::<Vec<usize>>(),
            (0..200).step_by(7).collect::<Vec<usize>>(),
        ] {
            let (occurrences, starts) = field
                .occurrences_for_docs(
                    b"alpha",
                    Some(&doc_in),
                    &pos_in,
                    pay_in.as_ref(),
                    &postings.freqs,
                    stats.total_term_freq,
                    &wanted,
                )
                .expect("occurrences");
            assert_eq!(starts.len(), wanted.len() + 1, "{index_options:?}");
            for (k, &d) in wanted.iter().enumerate() {
                let got = &occurrences[starts[k] as usize..starts[k + 1] as usize];
                assert_eq!(
                    got,
                    expected[d].as_slice(),
                    "doc index {d} of {wanted:?} ({index_options:?}, payloads={has_payloads})"
                );
            }

            // The positions-only sibling must agree with the same slices.
            let (positions, pos_starts) = field
                .positions_for_docs(
                    b"alpha",
                    Some(&doc_in),
                    &pos_in,
                    pay_in.as_ref(),
                    &postings.freqs,
                    stats.total_term_freq,
                    &wanted,
                )
                .expect("positions_for_docs");
            assert_eq!(pos_starts, starts);
            let expected_positions: Vec<i32> = wanted
                .iter()
                .flat_map(|&d| expected[d].iter().map(|p| p.position))
                .collect();
            assert_eq!(positions, expected_positions, "{index_options:?}");
        }
    }
}

/// `occurrences_for_doc` is the single-document form the highlighter calls:
/// `advance(doc)` then walk. It must agree with the whole-term reader for
/// every document, answer `None` for a document the term is not in, and
/// `None` for a term the field does not have.
#[test]
fn occurrences_for_doc_matches_the_whole_term_reader_for_every_document() {
    // 200 documents x 4 occurrences: three full `.pos` blocks and a tail.
    let docs: Vec<i32> = (0..200).map(|d| d * 2).collect();
    let term = synthetic_term(&docs, 4, true, true);
    let written = write(
        &term,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        true,
    );
    let (fields, doc_in, pos_in, pay_in) = written.open();
    let field = fields.field("body").expect("field");
    let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());

    for (i, &doc_id) in docs.iter().enumerate() {
        let got = field
            .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), doc_id)
            .expect("occurrences_for_doc")
            .unwrap_or_else(|| panic!("doc {doc_id} is in the term's postings"));
        assert_eq!(got, expected[i], "doc {doc_id}");
    }

    // An odd doc id: no document in this term has one.
    assert!(field
        .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), 1)
        .expect("occurrences_for_doc")
        .is_none());
    // A term the field does not have at all.
    assert!(field
        .occurrences_for_doc(b"zulu", Some(&doc_in), &pos_in, pay_in.as_ref(), 0)
        .expect("occurrences_for_doc")
        .is_none());
}

/// A `wanted` entry the doc list does not have, and one that repeats or goes
/// backwards, keeps its slot and yields nothing -- it must not panic (which
/// is what indexing the prefix-sum array by a caller-supplied index used to
/// do) and must not shift the other entries' answers.
#[test]
fn out_of_range_and_unsorted_wanted_entries_yield_empty_slots() {
    let docs: Vec<i32> = (0..10).collect();
    let term = synthetic_term(&docs, 2, false, false);
    let written = write(&term, IndexOptions::DocsAndFreqsAndPositions, false);
    let (fields, doc_in, pos_in, _pay) = written.open();
    let field = fields.field("body").expect("field");
    let expected = whole_term(field, &doc_in, &pos_in, None);
    let stats = field.seek_exact(b"alpha").expect("term present");
    let postings = field
        .postings(b"alpha", Some(&doc_in))
        .expect("postings")
        .expect("term present");

    // 3, then 3 again (repeat), then 1 (backwards), then 99 (out of range),
    // then 7 (fine again).
    let wanted = vec![3usize, 3, 1, 99, 7];
    let (positions, starts) = field
        .positions_for_docs(
            b"alpha",
            Some(&doc_in),
            &pos_in,
            None,
            &postings.freqs,
            stats.total_term_freq,
            &wanted,
        )
        .expect("positions_for_docs");
    assert_eq!(starts.len(), wanted.len() + 1);
    let slot = |k: usize| positions[starts[k] as usize..starts[k + 1] as usize].to_vec();
    let expect_positions =
        |d: usize| -> Vec<i32> { expected[d].iter().map(|p| p.position).collect() };
    assert_eq!(slot(0), expect_positions(3));
    assert!(slot(1).is_empty(), "a repeated index yields nothing");
    assert!(slot(2).is_empty(), "a backwards index yields nothing");
    assert!(slot(3).is_empty(), "an out-of-range index yields nothing");
    assert_eq!(slot(4), expect_positions(7));
}

/// A term whose whole `total_term_freq` fits in the vint tail needs no `.pay`
/// even when the field has offsets and payloads (the tail inlines both), and
/// one that spans a full block does.
#[test]
fn tail_only_terms_need_no_pay_file_but_full_blocks_do() {
    let short = synthetic_term(&[0, 1, 2], 2, true, true);
    let written = write(
        &short,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        true,
    );
    let (fields, doc_in, pos_in, pay_in) = written.open();
    let field = fields.field("body").expect("field");
    let stats = field.seek_exact(b"alpha").expect("term present");
    let postings = field
        .postings(b"alpha", Some(&doc_in))
        .expect("postings")
        .expect("term present");
    let with_pay = field
        .occurrences_for_docs(
            b"alpha",
            Some(&doc_in),
            &pos_in,
            pay_in.as_ref(),
            &postings.freqs,
            stats.total_term_freq,
            &[1],
        )
        .expect("occurrences");
    let without_pay = field
        .occurrences_for_docs(
            b"alpha",
            Some(&doc_in),
            &pos_in,
            None,
            &postings.freqs,
            stats.total_term_freq,
            &[1],
        )
        .expect("a tail-only term never touches .pay");
    assert_eq!(with_pay, without_pay);
    assert!(!with_pay.0.is_empty());

    // 200 documents x 2 = 400 occurrences: one full block plus a tail, so
    // `.pay` is genuinely needed.
    let long_docs: Vec<i32> = (0..200).collect();
    let long = synthetic_term(&long_docs, 2, true, true);
    let written = write(
        &long,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        true,
    );
    let (fields, doc_in, pos_in, _pay) = written.open();
    let field = fields.field("body").expect("field");
    let stats = field.seek_exact(b"alpha").expect("term present");
    let postings = field
        .postings(b"alpha", Some(&doc_in))
        .expect("postings")
        .expect("term present");
    let err = field
        .occurrences_for_docs(
            b"alpha",
            Some(&doc_in),
            &pos_in,
            None,
            &postings.freqs,
            stats.total_term_freq,
            &[1],
        )
        .expect_err("a full block's offsets/payloads live in .pay");
    assert!(format!("{err}").contains(".pay"), "unexpected error: {err}");
}

/// A frequency list that does not total the term's `totalTermFreq` is a decode
/// error, not an out-of-bounds walk: `freqs` comes from `.doc` and
/// `totalTermFreq` from the term dictionary, and nothing on the wire makes
/// them agree.
#[test]
fn frequencies_that_disagree_with_total_term_freq_are_rejected() {
    let docs: Vec<i32> = (0..10).collect();
    let term = synthetic_term(&docs, 2, false, false);
    let written = write(&term, IndexOptions::DocsAndFreqsAndPositions, false);
    let (fields, doc_in, pos_in, _pay) = written.open();
    let field = fields.field("body").expect("field");
    let stats = field.seek_exact(b"alpha").expect("term present");
    let mut freqs = field
        .postings(b"alpha", Some(&doc_in))
        .expect("postings")
        .expect("term present")
        .freqs;

    for (label, doctored) in [
        ("too many", {
            let mut f = freqs.clone();
            f[0] += 1;
            f
        }),
        ("too few", {
            let mut f = freqs.clone();
            f[0] -= 1;
            f
        }),
        ("negative", {
            let mut f = freqs.clone();
            f[0] = -1;
            f
        }),
    ] {
        let err = field
            .positions_for_docs(
                b"alpha",
                Some(&doc_in),
                &pos_in,
                None,
                &doctored,
                stats.total_term_freq,
                &[0],
            )
            .expect_err(label);
        assert!(
            format!("{err}").contains("freq"),
            "{label}: unexpected error {err}"
        );
    }

    // And the honest list still works, so the test cannot pass for the wrong
    // reason.
    freqs.truncate(freqs.len());
    assert!(field
        .positions_for_docs(
            b"alpha",
            Some(&doc_in),
            &pos_in,
            None,
            &freqs,
            stats.total_term_freq,
            &[0],
        )
        .is_ok());
}
