#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::field_infos::IndexOptions;
use crate::postings_writer::TermPostings;

#[test]
fn routed_groups_require_their_own_dictionary_even_without_terms() {
    let routed = |name: &str, suffix: &str| {
        let mut field = FieldInfo::new(name, 0);
        field.index_options = crate::field_infos::IndexOptions::Docs;
        field.attributes = vec![
            (PER_FIELD_FORMAT_KEY.into(), FORMAT_NAME.into()),
            (PER_FIELD_SUFFIX_KEY.into(), suffix.into()),
        ];
        field
    };
    let fields = [routed("populated", "0"), routed("termless", "1")];
    for prefix in ["_0_", "_"] {
        let files = vec![
            format!("{prefix}Lucene104_0.tmd"),
            format!("{prefix}Lucene104_1.tmd"),
        ];
        assert!(validate_routed_groups(&fields, &files).is_ok());
        let error = validate_routed_groups(&fields, &files[..1]).unwrap_err();
        assert!(error.contains("termless") && error.contains("Lucene104_1"));
    }
    let mut field = fields[0].clone();
    for attributes in [
        vec![field.attributes[0].clone()],
        vec![field.attributes[1].clone()],
        vec![
            (PER_FIELD_FORMAT_KEY.into(), String::new()),
            (PER_FIELD_SUFFIX_KEY.into(), "0".into()),
        ],
    ] {
        field.attributes = attributes;
        assert!(validate_routed_groups(&[field.clone()], &[])
            .unwrap_err()
            .contains("incomplete"));
    }
    field.attributes.clear();
    assert!(validate_routed_groups(&[field.clone()], &[]).is_ok());
    assert!(validate_routed_groups(&[field.clone()], &["_0.tmd".into()]).is_ok());
    field.index_options = crate::field_infos::IndexOptions::None;
    field.attributes = fields[0].attributes.clone();
    assert!(validate_routed_groups(&[field], &[]).is_ok());
}

#[test]
fn routed_simpletext_groups_require_their_own_postings_text_file() {
    let mut text = FieldInfo::new("text", 0);
    text.index_options = IndexOptions::Docs;
    text.attributes = vec![
        (PER_FIELD_FORMAT_KEY.into(), "SimpleText".into()),
        (PER_FIELD_SUFFIX_KEY.into(), "0".into()),
    ];
    let mut binary = text.clone();
    binary.name = "binary".into();
    binary.number = 1;
    binary.attributes[0].1 = FORMAT_NAME.into();
    let fields = [text, binary];
    for prefix in ["_0_", "_"] {
        let files = vec![
            format!("{prefix}SimpleText_0.pst"),
            format!("{prefix}Lucene104_0.tmd"),
        ];
        assert!(validate_routed_groups(&fields, &files).is_ok());
        assert!(validate_routed_groups(&fields, &files[1..])
            .unwrap_err()
            .contains("missing postings group SimpleText_0"));
        assert!(validate_routed_groups(&fields, &files[..1])
            .unwrap_err()
            .contains("missing postings group Lucene104_0"));
        let wrong_extension = vec![format!("{prefix}SimpleText_0.tmd"), files[1].clone()];
        assert!(validate_routed_groups(&fields, &wrong_extension).is_err());
        let wrong_suffix = vec![format!("{prefix}SimpleText_1.pst"), files[1].clone()];
        assert!(validate_routed_groups(&fields, &wrong_suffix).is_err());
    }
}

#[test]
fn suffix_assignment_and_group_iteration_follow_java_utf16_order() {
    // UTF-8 puts U+E000 first; Java compares the U+D800 surrogate first.
    let names = vec![
        "\u{e000}".to_string(),
        "\u{10000}".to_string(),
        "a".to_string(),
    ];
    let route = |name: &str| {
        if name == "\u{e000}" {
            Lucene104PostingsFormat::new(2, 3).unwrap()
        } else {
            Lucene104PostingsFormat::default()
        }
    };
    let groups = group(&names, &route);
    assert_eq!(
        groups
            .iter()
            .map(|(_, suffix, members)| (*suffix, members.clone()))
            .collect::<Vec<_>>(),
        [(0, vec![2, 1]), (1, vec![0])]
    );
    let fields = vec![
        (0, names[0].clone()),
        (1, names[1].clone()),
        (2, names[2].clone()),
    ];
    assert_eq!(
        field_suffixes(&fields, &route),
        [
            (0, "Lucene104", 1),
            (1, "Lucene104", 0),
            (2, "Lucene104", 0)
        ]
    );
}

fn terms(n: usize, doc: i32) -> Vec<TermPostings> {
    (0..n)
        .map(|i| TermPostings {
            term: format!("t{i:04}").into_bytes(),
            docs: vec![(doc, 1), (doc + 1, 1)],
            ..Default::default()
        })
        .collect()
}

/// A suffix names its completion format by everything before its last
/// `_`; Lucene 8's formats and every other format are none.
#[test]
fn completion_suffixes_name_their_format() {
    assert_eq!(completion_format_of("Completion90_0"), Some("Completion90"));
    assert_eq!(
        completion_format_of("Completion912_3"),
        Some("Completion912")
    );
    assert_eq!(
        completion_format_of("Completion104_0"),
        Some("Completion104")
    );
    for other in [
        "Completion84_0",
        "completion_0",
        "Lucene104_0",
        "",
        "Completion90",
    ] {
        assert_eq!(completion_format_of(other), None, "{other}");
    }
    assert!(is_completion_format("Completion101"));
    assert!(!is_completion_format("Lucene101"));
    let delegates: Vec<&str> = COMPLETION_FORMAT_NAMES
        .iter()
        .map(|n| {
            crate::postings::PostingsFormat::completion_delegate(n)
                .unwrap()
                .name()
        })
        .collect();
    assert_eq!(
        delegates,
        [
            "Lucene90",
            "Lucene99",
            "Lucene912",
            "Lucene101",
            "Lucene104"
        ]
    );
    assert_eq!(
        crate::postings::PostingsFormat::completion_delegate("Completion84"),
        None
    );
}

/// A field whose `.fnm` indexes positions and offsets but which got no
/// terms still frames `.pos`/`.pay` in `.psm`, as Java's
/// `state.fieldInfos.hasProx()` does: `Lucene104PostingsReader` reads
/// those lengths off the FieldInfos, and without them it found 8 bytes
/// where the footer belonged (`checksum status indeterminate`).
#[test]
fn psm_frames_pos_and_pay_from_the_segment_fields_not_the_inputs() {
    let docs_only = terms(5, 0);
    let inputs = [FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::Docs,
        doc_count: 2,
        has_payloads: false,
        terms: &docs_only,
    }];
    let segment_fields = [
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..FieldInfo::new("id", 0)
        },
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            ..FieldInfo::new("body", 1)
        },
    ];
    let run = |fields: &[FieldInfo]| {
        write(
            &inputs,
            fields,
            &[],
            &|n| Some(["id", "body"][n as usize].to_string()),
            &|_| Lucene104PostingsFormat::default(),
            &[3; ID_LENGTH],
        )
        .unwrap()
        .remove(0)
        .output
    };
    let (bare, framed) = (run(&segment_fields[..1]), run(&segment_fields));
    assert!(bare.pos.is_empty() && bare.pay.is_empty());
    assert!(!framed.pos.is_empty() && !framed.pay.is_empty());
    // Two more longs: the `.pos` and the `.pay` lengths.
    assert_eq!(framed.psm.len(), bare.psm.len() + 16);
}

/// Two fields to the small format, one to the default: two groups,
/// numbered in the name order of their first field, each field listed
/// once; the second group has `.pos` because the segment has positions.
#[test]
fn fields_are_grouped_by_format_in_name_order() {
    let mut a = terms(60, 0);
    for t in &mut a {
        t.positions = vec![vec![3], vec![5]];
    }
    let (b, c) = (terms(60, 2), terms(5, 4));
    let inputs = [
        FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 2,
            has_payloads: false,
            terms: &a,
        },
        FieldPostingsInput {
            field_number: 1,
            index_options: IndexOptions::Docs,
            doc_count: 2,
            has_payloads: false,
            terms: &b,
        },
        FieldPostingsInput {
            field_number: 2,
            index_options: IndexOptions::Docs,
            doc_count: 2,
            has_payloads: false,
            terms: &c,
        },
    ];
    let names = ["z_pos", "b_small", "a_small"];
    let small = Lucene104PostingsFormat::new(10, 20).unwrap();
    let groups = write(
        &inputs,
        &[],
        &[],
        &|n| names.get(n as usize).map(|s| s.to_string()),
        &|name| {
            if name.ends_with("small") {
                small
            } else {
                Lucene104PostingsFormat::default()
            }
        },
        &[1; ID_LENGTH],
    )
    .unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].suffix, 0);
    // Name order within a group too: `a_small` (2) before `b_small` (1).
    assert_eq!(groups[0].field_numbers, [2, 1]);
    assert_eq!(groups[1].field_numbers, [0]);
    assert!(!groups[0].output.pos.is_empty(), "segment-level .pos");
    assert!(groups[0].output.pay.is_empty());
    assert_eq!(codec_suffix(FORMAT_NAME, 1), "Lucene104_1");
    assert!(Lucene104PostingsFormat::new(1, 20).is_err());
    let fields: Vec<(i32, String)> = (0..3).map(|n| (n, names[n as usize].to_string())).collect();
    assert_eq!(
        field_suffixes(&fields, &|name| if name.ends_with("small") {
            small
        } else {
            Lucene104PostingsFormat::default()
        }),
        [
            (0, FORMAT_NAME, 1),
            (1, FORMAT_NAME, 0),
            (2, FORMAT_NAME, 0)
        ]
    );

    // Read back: every field from its own group, at its own offsets.
    let infos = FieldInfos {
        fields: (0..3)
            .map(|n| crate::field_infos::FieldInfo {
                index_options: inputs[n as usize].index_options,
                ..crate::field_infos::FieldInfo::new(names[n as usize], n)
            })
            .collect(),
    };
    let suffixes: Vec<String> = groups
        .iter()
        .map(|g| codec_suffix(FORMAT_NAME, g.suffix))
        .collect();
    let files: Vec<GroupFiles<'_>> = groups
        .iter()
        .zip(&suffixes)
        .map(|(g, suffix)| GroupFiles {
            suffix,
            dictionary: vec![&g.output.tim, &g.output.tip, &g.output.tmd],
            doc: Some(&g.output.doc),
            pos: (!g.output.pos.is_empty()).then_some(&g.output.pos[..]),
            pay: None,
        })
        .collect();
    let combined = open_groups(&files, &infos, &[1; ID_LENGTH], 6).unwrap();
    let doc_in = DocInput::validated(&combined.doc);
    for (n, name) in names.iter().enumerate() {
        let field = combined.fields.field(name).unwrap();
        let p = field.postings(b"t0003", Some(&doc_in)).unwrap().unwrap();
        let doc = [0, 2, 4][n];
        assert_eq!(p.docs, [doc, doc + 1], "{name}");
    }
    // The same groups with the first one's `.tim`/`.tip` handed over
    // shared: the dictionary keeps that buffer rather than a copy.
    let tim0 = Arc::new(groups[0].output.tim.clone());
    let tip0 = Arc::new(groups[0].output.tip.clone());
    let shared = [Some([
        tim0.clone() as SharedBytes,
        tip0.clone() as SharedBytes,
    ])];
    let combined = open_groups_shared(&files, &shared, &infos, &[1; ID_LENGTH], 6).unwrap();
    assert!(Arc::strong_count(&tim0) > 1 && Arc::strong_count(&tip0) > 1);
    let doc_in = DocInput::validated(&combined.doc);
    for (n, name) in names.iter().enumerate() {
        let field = combined.fields.field(name).unwrap();
        let p = field.postings(b"t0003", Some(&doc_in)).unwrap().unwrap();
        let doc = [0, 2, 4][n];
        assert_eq!(p.docs, [doc, doc + 1], "{name}");
    }
    let mut dup = files.clone();
    dup.push(files[0].clone());
    assert!(open_groups(&dup, &infos, &[1; ID_LENGTH], 6).is_err());
    assert_eq!(
        group_suffixes(
            &[
                "_0_Lucene104_1.tmd".into(),
                "_0_Lucene104_0.tmd".into(),
                "_0_Lucene104_0.tim".into(),
                "_0.tmd".into(),
                "_Lucene104_0.tmd".into(),
            ],
            "_0"
        ),
        ["", "Lucene104_0", "Lucene104_1"]
    );
    // A foreign marker counts only under its own format's name.
    assert_eq!(
        group_suffixes(
            &[
                "_0_SharedTermsUniformSplit_0.stustd".into(),
                "_0_SharedTermsUniformSplit_0.ustd".into(),
                "_0_UniformSplit_1.ustd".into(),
                "_0_BlockTreeOrds_0.ustd".into(),
            ],
            "_0"
        ),
        ["SharedTermsUniformSplit_0", "UniformSplit_1"]
    );
}

/// A suggestion payload: vint surface length, surface, vint weight + 1,
/// type 0 (`SuggestField.buildSuggestPayload`).
fn suggestion(surface: &str, weight: u8) -> Vec<u8> {
    let mut p = vec![surface.len() as u8];
    p.extend_from_slice(surface.as_bytes());
    p.extend_from_slice(&[weight + 1, 0]);
    p
}

fn suggest_terms(words: &[&str]) -> Vec<TermPostings> {
    words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let payloads = [suggestion(w, i as u8), suggestion(w, 9)];
            TermPostings {
                term: w.as_bytes().to_vec(),
                docs: vec![(i as i32, 1), (i as i32 + 10, 1)],
                positions: vec![vec![0], vec![0]],
                payload_lengths: payloads.iter().map(|p| p.len() as u32).collect(),
                payload_bytes: payloads.concat(),
                ..Default::default()
            }
        })
        .collect()
}

/// Completion fields get their own `Completion104_0` group, numbered
/// apart from `Lucene104_0`, with `.lkp`/`.cmp` that open as Lucene's
/// reader opens them.
#[test]
fn completion_fields_are_their_own_format() {
    let completion = Lucene104PostingsFormat::completion104();
    assert_eq!(completion.name(), COMPLETION_FORMAT_NAME);
    assert_eq!(Lucene104PostingsFormat::default().name(), FORMAT_NAME);
    assert_eq!(
        completion.write_options().min_items_in_block,
        Lucene104PostingsFormat::default().min_term_block_size
    );
    let (s, t) = (suggest_terms(&["car", "cat"]), suggest_terms(&["dog"]));
    let text = terms(3, 0);
    let input = |field_number, index_options, has_payloads, terms| FieldPostingsInput {
        field_number,
        index_options,
        doc_count: 2,
        has_payloads,
        terms,
    };
    let inputs = [
        input(0, IndexOptions::DocsAndFreqsAndPositions, true, &s[..]),
        input(1, IndexOptions::Docs, false, &text[..]),
        input(2, IndexOptions::DocsAndFreqsAndPositions, true, &t[..]),
    ];
    let names = ["suggest", "body", "title"];
    let route = |name: &str| {
        if name == "body" {
            Lucene104PostingsFormat::default()
        } else {
            completion
        }
    };
    let id = [7; ID_LENGTH];
    let groups = write(
        &inputs,
        &[],
        &[],
        &|n| names.get(n as usize).map(|s| s.to_string()),
        &route,
        &id,
    )
    .unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].codec_suffix(), "Lucene104_0");
    assert!(groups[0].completion.is_none());
    assert_eq!(groups[1].codec_suffix(), "Completion104_0");
    assert_eq!(groups[1].field_numbers, [0, 2]);
    let fields: Vec<(i32, String)> = (0..3).map(|n| (n, names[n as usize].to_string())).collect();
    assert_eq!(
        field_suffixes(&fields, &route),
        [
            (0, COMPLETION_FORMAT_NAME, 0),
            (1, FORMAT_NAME, 0),
            (2, COMPLETION_FORMAT_NAME, 0)
        ]
    );
    let files = groups[1].completion.clone().unwrap();
    let opened = crate::completion::CompletionFields::open(
        COMPLETION_FORMAT_NAME,
        &id,
        "Completion104_0",
        &files.cmp,
        files.lkp,
    )
    .unwrap();
    assert_eq!(opened.delegate_format, FORMAT_NAME);
    for number in [0, 2] {
        assert!(opened.suggester(number).unwrap().is_some(), "{number}");
    }
    assert!(opened.suggester(1).unwrap().is_none());
}

/// `assert postingsEnum.getPayload() != null`: an occurrence of a
/// completion field with no payload is refused rather than indexed as no
/// suggestion, and so are mixed suggestion types in one field.
#[test]
fn a_completion_occurrence_without_a_payload_is_refused() {
    let mut empty = suggest_terms(&["car"]);
    let second = empty[0].payload_lengths[1] as usize;
    empty[0].payload_lengths[1] = 0;
    let keep = empty[0].payload_bytes.len() - second;
    empty[0].payload_bytes.truncate(keep);
    let mut mixed = suggest_terms(&["car"]);
    // the second occurrence's type byte: 1, beside the first's 0
    *mixed[0].payload_bytes.last_mut().unwrap() = 1;
    for terms in [empty, mixed] {
        let inputs = [FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 2,
            has_payloads: true,
            terms: &terms,
        }];
        let err = write(
            &inputs,
            &[],
            &[],
            &|_| Some("suggest".to_string()),
            &|_| Lucene104PostingsFormat::completion104(),
            &[7; ID_LENGTH],
        );
        assert!(
            matches!(err, Err(postings_writer::Error::Completion(_))),
            "{err:?}"
        );
    }
}

/// A term whose payload lengths disagree with its occurrences or its
/// payload bytes is an internal inconsistency, not a missing payload.
#[test]
fn inconsistent_payload_runs_are_refused() {
    let good = suggest_terms(&["car"]).remove(0);
    assert!(term_docs(&good).is_ok());
    let mut short = good.clone();
    short.payload_lengths.pop();
    let mut extra_length = good.clone();
    extra_length.payload_lengths.push(1);
    let mut overlong = good.clone();
    *overlong.payload_lengths.last_mut().unwrap() += 1;
    let mut trailing = good.clone();
    trailing.payload_bytes.push(0);
    for term in [short, extra_length, overlong, trailing] {
        let err = term_docs(&term).unwrap_err().to_string();
        assert!(err.contains("do not match"), "{err}");
    }
}
