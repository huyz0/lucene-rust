#![cfg(test)]
use super::*;
use crate::segment_state::SegmentReadState;

const ID: [u8; 16] = *b"sharedmergetests";
const SUFFIX: &str = "SharedTermsUniformSplit_0";

fn infos() -> FieldInfos {
    FieldInfos::new(vec![
        FieldInfo::new("docs", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("freq", 1)
            .with_index_options(IndexOptions::DocsAndFreqs)
            .with_omit_norms(true),
        FieldInfo::new("pay", 2)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .with_store_payloads(true)
            .with_omit_norms(true),
    ])
    .unwrap()
}

fn state(infos: &FieldInfos) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc: 3,
        segment_id: &ID,
        segment_suffix: SUFFIX,
        field_infos: infos,
    }
}

fn read(output: &UniformSplitOutput, infos: &FieldInfos) -> StUniformSplitTermsReader {
    StUniformSplitPostingsFormat::default()
        .open(
            output.ustb.clone(),
            &output.ustd,
            &SegmentReadState {
                max_doc: 3,
                segment_id: &ID,
                segment_suffix: SUFFIX,
                field_infos: infos,
            },
        )
        .unwrap()
}

fn fixture(infos: &FieldInfos) -> UniformSplitOutput {
    let fields: Vec<_> = infos
        .fields
        .iter()
        .map(|field| {
            [b"a".to_vec(), b"z".to_vec()]
                .into_iter()
                .map(|term| TermPostings {
                    term,
                    docs: vec![(0, 1), (1, 1), (2, 1)],
                    positions: if field.index_options.subsumes_positions() {
                        vec![vec![0]; 3]
                    } else {
                        Vec::new()
                    },
                    offsets: if field.index_options.subsumes_offsets() {
                        vec![vec![(0, 1)]; 3]
                    } else {
                        Vec::new()
                    },
                    payload_lengths: if field.store_payloads {
                        vec![1; 3]
                    } else {
                        vec![]
                    },
                    payload_bytes: if field.store_payloads {
                        vec![42; 3]
                    } else {
                        vec![]
                    },
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let inputs: Vec<_> = infos
        .fields
        .iter()
        .zip(&fields)
        .map(|(field, terms)| FieldPostingsInput {
            field_number: field.number,
            index_options: field.index_options,
            has_payloads: field.store_payloads,
            doc_count: 3,
            terms,
        })
        .collect();
    StUniformSplitPostingsFormat::default()
        .write(&inputs, &[], &state(infos))
        .unwrap()
}

fn no_fallback() -> Result<UniformSplitOutput> {
    panic!("specialized merge unexpectedly delegated")
}

#[test]
fn maps_deletes_and_multiple_sources_preserve_postings() {
    let infos = infos();
    let output = fixture(&infos);
    let reader = read(&output, &infos);
    let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
    let pos = PosInput::open(&output.postings.pos, &ID, SUFFIX).unwrap();
    let pay = PayInput::open(&output.postings.pay, &ID, SUFFIX).unwrap();
    let source = StMergeSegment {
        reader: Some(&reader),
        field_infos: &infos,
        doc_map: &[0, -1, 1],
        doc: &doc,
        pos: Some(&pos),
        pay: Some(&pay),
    };
    let second = StMergeSegment {
        doc_map: &[-1, 2, -1],
        ..source
    };
    let merged = StUniformSplitPostingsFormat::default()
        .merge(&[source, second], &[], &state(&infos), false, no_fallback)
        .unwrap();
    // Both terms in all fields have all three target documents, with
    // identical positions, offsets and payloads to the original bytes.
    assert_eq!(merged.ustb, output.ustb);
    assert_eq!(merged.ustd, output.ustd);
    assert_eq!(merged.postings.doc, output.postings.doc);
    assert_eq!(merged.postings.pos, output.postings.pos);
    assert_eq!(merged.postings.pay, output.postings.pay);
    assert_eq!(merged.postings.psm, output.postings.psm);
}

#[test]
fn mixed_payload_sources_and_downgraded_offsets() {
    let full = infos();
    let a_infos = FieldInfos::new(vec![full.field_by_name("pay").unwrap().clone()]).unwrap();
    let b_infos = FieldInfos::new(vec![FieldInfo::new("pay", 17)
        .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .with_omit_norms(true)])
    .unwrap();
    let a = fixture(&a_infos);
    let b = fixture(&b_infos);
    let a_reader = read(&a, &a_infos);
    let b_reader = read(&b, &b_infos);
    let a_doc = DocInput::open(&a.postings.doc, &ID, SUFFIX).unwrap();
    let b_doc = DocInput::open(&b.postings.doc, &ID, SUFFIX).unwrap();
    let a_pos = PosInput::open(&a.postings.pos, &ID, SUFFIX).unwrap();
    let b_pos = PosInput::open(&b.postings.pos, &ID, SUFFIX).unwrap();
    let a_pay = PayInput::open(&a.postings.pay, &ID, SUFFIX).unwrap();
    let target = FieldInfos::new(vec![FieldInfo::new("pay", 42)
        .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
        .with_store_payloads(true)
        .with_omit_norms(true)])
    .unwrap();
    let target_state = SegmentWriteState {
        max_doc: 2,
        ..state(&target)
    };
    let sources = [
        StMergeSegment {
            reader: Some(&a_reader),
            field_infos: &a_infos,
            doc_map: &[0, -1, -1],
            doc: &a_doc,
            pos: Some(&a_pos),
            pay: Some(&a_pay),
        },
        StMergeSegment {
            reader: Some(&b_reader),
            field_infos: &b_infos,
            doc_map: &[-1, 1, -1],
            doc: &b_doc,
            pos: Some(&b_pos),
            pay: None,
        },
    ];
    let format = StUniformSplitPostingsFormat::default();
    let merged = format
        .merge(&sources, &[], &target_state, false, no_fallback)
        .unwrap();
    let expected_terms: Vec<_> = [b"a".to_vec(), b"z".to_vec()]
        .into_iter()
        .map(|term| TermPostings {
            term,
            docs: vec![(0, 1), (1, 1)],
            positions: vec![vec![0], vec![0]],
            payload_lengths: vec![1, 0],
            payload_bytes: vec![42],
            ..TermPostings::default()
        })
        .collect();
    let expected = format
        .write(
            &[FieldPostingsInput {
                field_number: 42,
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                has_payloads: true,
                doc_count: 2,
                terms: &expected_terms,
            }],
            &[],
            &target_state,
        )
        .unwrap();
    // The second source contributes a zero-length payload. Source
    // offsets are dropped, while the first source's payload survives.
    assert_eq!(merged.ustb, expected.ustb);
    assert_eq!(merged.ustd, expected.ustd);
    assert_eq!(merged.postings.doc, expected.postings.doc);
    assert_eq!(merged.postings.pos, expected.postings.pos);
    assert_eq!(merged.postings.pay, expected.postings.pay);
    assert_eq!(merged.postings.psm, expected.postings.psm);
}

#[test]
fn invalid_maps_and_missing_streams_are_errors() {
    let infos = infos();
    let output = fixture(&infos);
    let reader = read(&output, &infos);
    let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
    let pos = PosInput::open(&output.postings.pos, &ID, SUFFIX).unwrap();
    let pay = PayInput::open(&output.postings.pay, &ID, SUFFIX).unwrap();
    let source = StMergeSegment {
        reader: Some(&reader),
        field_infos: &infos,
        doc_map: &[0, 1, 2],
        doc: &doc,
        pos: Some(&pos),
        pay: Some(&pay),
    };
    for (map, expected) in [
        (&[0, 1][..], "outside its doc map"),
        (&[0, -2, 1][..], "not increasing"),
        (&[0, 0, 1][..], "not increasing"),
        (&[1, 0, 2][..], "not increasing"),
        (&[0, 1, 3][..], "exceeds target maxDoc"),
    ] {
        let result = StUniformSplitPostingsFormat::default().merge(
            &[StMergeSegment {
                doc_map: map,
                ..source
            }],
            &[],
            &state(&infos),
            false,
            no_fallback,
        );
        assert!(
            matches!(result, Err(Error::Store(lucene_store::Error::Corrupted(ref msg))) if msg.contains(expected)),
            "{result:?}"
        );
    }
    let missing_pos = StUniformSplitPostingsFormat::default().merge(
        &[StMergeSegment {
            pos: None,
            ..source
        }],
        &[],
        &state(&infos),
        false,
        no_fallback,
    );
    assert!(
        matches!(missing_pos, Err(Error::Store(lucene_store::Error::Corrupted(ref msg))) if msg.contains("lacks .pos"))
    );
    let missing_pay = StUniformSplitPostingsFormat::default().merge(
        &[StMergeSegment {
            pay: None,
            ..source
        }],
        &[],
        &state(&infos),
        false,
        no_fallback,
    );
    assert!(missing_pay.is_err(), "offsets and payloads require .pay");
    let negative = SegmentWriteState {
        max_doc: -1,
        ..state(&infos)
    };
    assert!(matches!(
        StUniformSplitPostingsFormat::default().merge(
            &[source],
            &[],
            &negative,
            false,
            no_fallback,
        ),
        Err(Error::IllegalArgument(_))
    ));
}

#[test]
fn empty_sources_deleted_fields_and_filtered_fields() {
    let infos = infos();
    let output = fixture(&infos);
    let reader = read(&output, &infos);
    let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
    let source = StMergeSegment {
        reader: Some(&reader),
        field_infos: &infos,
        doc_map: &[-1; 3],
        doc: &doc,
        pos: None,
        pay: None,
    };
    let empty = FieldInfos::new(vec![]).unwrap();
    let format = StUniformSplitPostingsFormat::default();
    for sources in [
        vec![],
        vec![source],
        vec![StMergeSegment {
            field_infos: &empty,
            ..source
        }],
    ] {
        // Removed target fields do not read source positions, even if
        // their source terms exist and their doc maps are empty.
        let merged = format
            .merge(&sources, &[], &state(&empty), false, no_fallback)
            .unwrap();
        assert_eq!(read(&merged, &empty).field_names().count(), 0);
    }
    let docs_only = FieldInfos::new(vec![infos.field_by_name("docs").unwrap().clone()]).unwrap();
    let merged = format
        .merge(&[source], &[], &state(&docs_only), false, no_fallback)
        .unwrap();
    assert_eq!(read(&merged, &docs_only).field_names().count(), 0);
    // A target can downgrade positions/frequencies to DOCS. It does
    // not need .pos/.pay, and writes frequency one for each survivor.
    let downgraded = FieldInfos::new(vec![FieldInfo::new("pay", 5)
        .with_index_options(IndexOptions::Docs)
        .with_omit_norms(true)])
    .unwrap();
    let merged = format
        .merge(
            &[StMergeSegment {
                doc_map: &[0, 1, 2],
                ..source
            }],
            &[],
            &state(&downgraded),
            false,
            no_fallback,
        )
        .unwrap();
    let reader = read(&merged, &downgraded);
    let field = reader.terms("pay").unwrap();
    assert_eq!(field.doc_count(), 3);
    assert_eq!(field.sum_doc_freq(), 6);
    assert_eq!(field.sum_total_term_freq(), 6);
}

#[test]
fn fallback_is_once_before_input_validation_and_propagates_errors() {
    let infos = infos();
    let output = fixture(&infos);
    let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
    let source = StMergeSegment {
        reader: None,
        field_infos: &infos,
        doc_map: &[],
        doc: &doc,
        pos: None,
        pay: None,
    };
    let state = SegmentWriteState {
        max_doc: -1,
        ..state(&infos)
    };
    for (sources, sorted) in [(vec![], true), (vec![source], false)] {
        let mut calls = 0;
        let error =
            StUniformSplitPostingsFormat::default().merge(&sources, &[], &state, sorted, || {
                calls += 1;
                Err(Error::IllegalArgument("fallback error".into()))
            });
        assert_eq!(calls, 1);
        assert!(matches!(error, Err(Error::IllegalArgument(ref msg)) if msg == "fallback error"));
    }
}
