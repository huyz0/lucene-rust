//! `.doc`'s level-0 and level-1 `.pos`/`.pay` skip pointers, both ends.
//!
//! `Lucene104PostingsWriter.flushDocBlock`/`writeLevel1SkipData` record, in
//! every `.doc` skip record of a positions-indexing field, the `.pos`/`.pay`
//! file pointer and in-block offset that record's documents' occurrences
//! start at. `Lucene104PostingsReader.seekPosData` uses them so that
//! `advance(doc)` can jump the position streams instead of walking the whole
//! postings list.
//!
//! Until `c20-postings-skip` this port wrote neither (a term indexing
//! positions was refused past `docFreq >= BLOCK_SIZE` precisely because the
//! sub-fields were missing) and read neither (they were parsed and thrown
//! away). This file covers both directions at once: the terms are written by
//! this crate's own `postings_writer`, so the skip records under test are the
//! ones it emits, and every expectation is **differential against
//! [`FieldTerms::positions`]** -- the whole-term reader that
//! `blocktree_fixtures.rs` pins against real Lucene's own occurrence list,
//! and which addresses `.pos` by a running frequency sum rather than by any
//! skip pointer. The two agree only if the pointers are right.
//!
//! `docFreq` is pushed past `BLOCK_SIZE` (256) so level-0 skip records exist,
//! and past `LEVEL1_NUM_DOCS` (8 192) in one case so a level-1 entry does
//! too -- the level-1 pointers are the ones a jump of a whole 32-block span
//! depends on, and nothing smaller reaches them.
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

const SEG_ID: [u8; ID_LENGTH] = [11u8; ID_LENGTH];
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

/// One term over `doc_count` documents with *varying* per-document
/// frequencies, so no `.pos` block boundary lines up with a `.doc` block
/// boundary and `posBufferUpto` is non-zero for almost every skip record.
///
/// Doc IDs step by 2 or 3 so the `.doc` full blocks do not all take the
/// all-consecutive encoding, and offsets/payloads are derived from the
/// position so no two occurrences look alike.
///
/// The frequency cycle's *period* is load-bearing; see the comment on it.
fn synthetic_term(doc_count: usize, with_offsets: bool, with_payloads: bool) -> TermPostings {
    let mut docs = Vec::with_capacity(doc_count);
    let mut positions = Vec::with_capacity(doc_count);
    let mut offsets = Vec::new();
    let mut payload_bytes = Vec::new();
    let mut payload_lengths = Vec::new();
    let mut doc_id = 0i32;
    for d in 0..doc_count {
        // 1..=5 occurrences, cycling on a period **coprime with 256** so the
        // occurrence count at every 256-document boundary -- and, critically,
        // at the 8 192-document level-1 boundary -- is irregular. A period of
        // 4 makes `sum(1 + d % 4)` over 8 192 documents exactly 20 480, i.e.
        // 80 whole `.pos` blocks, so every level-1 `posBufferUpto` would be
        // `0` and a reader that ignored the field entirely would still pass
        // every assertion in this file. With 5 it is 253.
        let freq = 1 + (d % 5) as i32;
        docs.push((doc_id, freq));
        doc_id += 2 + (d % 2) as i32;
        let base = (d as i32 % 17) + 1;
        let doc_positions: Vec<i32> = (0..freq).map(|i| base + i * (1 + (d as i32 % 3))).collect();
        if with_offsets {
            offsets.push(
                doc_positions
                    .iter()
                    .map(|&p| (p * 3, p * 3 + 2 + (d as i32 % 5)))
                    .collect(),
            );
        }
        if with_payloads {
            for &p in &doc_positions {
                // Lengths vary, including zero, which is what makes a
                // block's payload byte-run offsets non-trivial.
                let len = (p % 4) as usize;
                payload_lengths.push(len as u32);
                payload_bytes.extend(std::iter::repeat_n((d % 251) as u8, len));
            }
        }
        positions.push(doc_positions);
    }
    TermPostings {
        term: b"alpha".to_vec(),
        docs,
        positions,
        offsets,
        payload_bytes,
        payload_lengths,
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

/// The core property, over every field shape: for **every** document of a
/// term with several full `.doc` blocks, the skip-driven single-document walk
/// returns exactly what the whole-term reader returns for that document.
///
/// A wrong `posEndFPDelta` or `posBufferUpto` in any level-0 record shifts one
/// document's occurrence window, so the mismatch is local and this catches it
/// wherever it is: the documents are walked one at a time, each from a fresh
/// cursor, so an error cannot be masked by sequential state.
#[test]
fn every_document_of_a_multi_block_term_agrees_with_the_whole_term_reader() {
    // 700 documents: two full 256-document `.doc` blocks plus a 188-document
    // tail, and 2 100 occurrences, i.e. eight full `.pos` blocks plus a tail.
    let doc_count = 700;
    for (index_options, has_payloads) in [
        (IndexOptions::DocsAndFreqsAndPositions, false),
        (IndexOptions::DocsAndFreqsAndPositions, true),
        (IndexOptions::DocsAndFreqsAndPositionsAndOffsets, false),
        (IndexOptions::DocsAndFreqsAndPositionsAndOffsets, true),
    ] {
        let has_offsets = index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let term = synthetic_term(doc_count, has_offsets, has_payloads);
        let doc_ids: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let written = write(&term, index_options, has_payloads);
        let (fields, doc_in, pos_in, pay_in) = written.open();
        let field = fields.field("body").expect("field");
        let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());
        assert_eq!(expected.len(), doc_count);

        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let got = field
                .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), doc_id)
                .expect("occurrences_for_doc")
                .unwrap_or_else(|| panic!("doc {doc_id} is in the term's postings"));
            assert_eq!(
                got, expected[i],
                "doc {doc_id} (index {i}) of {index_options:?}, payloads={has_payloads}"
            );
        }

        // A doc id the term does not contain, in the middle of the list --
        // the skip walk has to land past it and say so rather than returning
        // the next document's occurrences.
        let absent = doc_ids[doc_count / 2] + 1;
        assert!(!doc_ids.contains(&absent));
        assert!(
            field
                .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), absent)
                .expect("occurrences_for_doc")
                .is_none(),
            "{index_options:?}"
        );
        // And one past the end.
        assert!(field
            .occurrences_for_doc(
                b"alpha",
                Some(&doc_in),
                &pos_in,
                pay_in.as_ref(),
                doc_ids[doc_count - 1] + 1
            )
            .expect("occurrences_for_doc")
            .is_none());
    }
}

/// The level-1 pointers, which only a term of at least `LEVEL1_NUM_DOCS`
/// (8 192) documents has on the wire at all.
///
/// A `skipLevel1To` that jumps a whole 32-block span must carry the span's
/// own `.pos`/`.pay` state across with it (`level0PosEndFP = level1PosEndFP`);
/// a reader that forgot to would address `.pos` from wherever the last
/// level-0 header left it, which for a document in the *second* span is a
/// wrong window by tens of thousands of occurrences. Documents are sampled
/// (every 97th, plus the span boundaries) because 8 500 fresh cursors is the
/// slow part, not the coverage.
#[test]
fn a_level1_span_jump_carries_the_position_pointers_with_it() {
    let doc_count = 8_500;
    let term = synthetic_term(doc_count, true, true);
    let doc_ids: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
    let written = write(
        &term,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        true,
    );
    let (fields, doc_in, pos_in, pay_in) = written.open();
    let field = fields.field("body").expect("field");
    let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());

    let mut sampled: Vec<usize> = (0..doc_count).step_by(97).collect();
    // The documents either side of the single level-1 entry's span boundary,
    // and the last document, which is in the trailing sub-span remainder.
    sampled.extend([8_191, 8_192, 8_193, doc_count - 1]);
    for i in sampled {
        let doc_id = doc_ids[i];
        let got = field
            .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), doc_id)
            .expect("occurrences_for_doc")
            .unwrap_or_else(|| panic!("doc {doc_id} is in the term's postings"));
        assert_eq!(got, expected[i], "doc {doc_id} (index {i})");
    }
}

/// A singleton term (`docFreq == 1`) has no `.doc` bytes at all -- it is
/// pulsed into the term dictionary -- so there is no skip data to walk and
/// the single-document accessor has to take the other route.
#[test]
fn a_pulsed_singleton_term_still_answers_for_its_one_document() {
    let mut term = synthetic_term(1, true, true);
    term.docs = vec![(41, 3)];
    term.positions = vec![vec![0, 4, 9]];
    term.offsets = vec![vec![(0, 3), (10, 14), (20, 26)]];
    term.payload_bytes = vec![1u8, 2u8, 3u8];
    term.payload_lengths = vec![1, 0, 2];
    let written = write(
        &term,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        true,
    );
    let (fields, doc_in, pos_in, pay_in) = written.open();
    let field = fields.field("body").expect("field");
    let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());

    let got = field
        .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), 41)
        .expect("occurrences_for_doc")
        .expect("the singleton's own document");
    assert_eq!(got, expected[0]);
    assert!(field
        .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), 40)
        .expect("occurrences_for_doc")
        .is_none());
    assert!(field
        .occurrences_for_doc(b"zulu", Some(&doc_in), &pos_in, pay_in.as_ref(), 41)
        .expect("occurrences_for_doc")
        .is_none());
}

/// `lastPosBlockOffset` is now load-bearing on the **read** side: a walk that
/// jumps into the middle of `.pos` has no running occurrence count, so the
/// only thing separating a `PForUtil` block from the vint tail is
/// `posStartFP + lastPosBlockOffset` (`Lucene104PostingsReader.reset`'s
/// `lastPosBlockFP`).
///
/// `b5` found this field being written as a constant `0`, which made real
/// Lucene decode a term's *first* position block as a vint tail for every
/// term with `totalTermFreq > 256` -- and no test in the repo could see it,
/// because this port's own reader re-derived the split. This one can: it
/// corrupts the field in the written `.tim` and requires the skip-driven walk
/// to disagree with the whole-term reader, which does not read it.
#[test]
fn a_wrong_last_pos_block_offset_is_visible_to_the_skip_driven_walk() {
    let doc_count = 400;
    let term = synthetic_term(doc_count, true, false);
    let written = write(
        &term,
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        false,
    );
    let doc_ids: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();

    // The honest bytes agree, so the corruption below cannot pass for the
    // wrong reason.
    {
        let (fields, doc_in, pos_in, pay_in) = written.open();
        let field = fields.field("body").expect("field");
        let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());
        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let got = field
                .occurrences_for_doc(b"alpha", Some(&doc_in), &pos_in, pay_in.as_ref(), doc_id)
                .expect("occurrences_for_doc")
                .expect("present");
            assert_eq!(got, expected[i]);
        }
    }

    // Now flip `lastPosBlockOffset` to `0` -- b5's exact defect -- by
    // rewriting the term metadata region of the `.tim`. Rather than locating
    // the vlong by hand, re-run the writer over a doctored copy: any byte
    // change is caught by `.tim`'s footer, so the honest route is to make the
    // *reader* see a zero. `TermMetadata` is decoded from `.tim`, so the
    // check is expressed at that level instead.
    let mut disagreements = 0usize;
    {
        let (fields, doc_in, pos_in, pay_in) = written.open();
        let field = fields.field("body").expect("field");
        let stats = field.seek_exact(b"alpha").expect("term present");
        assert!(
            stats.total_term_freq > 256,
            "the offset is only on the wire past one full .pos block"
        );
        let expected = whole_term(field, &doc_in, &pos_in, pay_in.as_ref());
        let meta = field
            .term_metadata(b"alpha")
            .expect("term metadata")
            .expect("term present");
        assert!(
            meta.last_pos_block_offset > 0,
            "the writer must record where the vint tail begins, not 0 (b5 F4)"
        );
        let doctored = lucene_codecs::postings::TermMetadata {
            last_pos_block_offset: 0,
            ..meta
        };
        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let got = lucene_codecs::postings::read_occurrences_for_doc(
                &doc_in,
                &pos_in,
                pay_in.as_ref(),
                doctored,
                stats.doc_freq,
                stats.total_term_freq,
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                false,
                doc_id,
            );
            match got {
                Ok(Some(occ)) if occ == expected[i] => {}
                _ => disagreements += 1,
            }
        }
    }
    assert!(
        disagreements > 0,
        "a zero lastPosBlockOffset must change what the skip-driven walk decodes; \
         if it does not, the walk is not using it and b5's defect is invisible again"
    );
}
