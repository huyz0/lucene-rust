#![cfg(test)]
use super::*;
use crate::exec::{BoxScorer, Scorer, NO_MORE_DOCS};
use crate::query::{Clause, PhraseQuery};
use lucene_codecs::simpletext::fields_reader::SimpleTextFieldsReader;
use lucene_codecs::{blocktree::BlockTreeFields, segment_state::SegmentReadState};
use std::{fs, path::PathBuf};

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn score_bits(value: &str) -> u32 {
    u32::from_str_radix(value, 16).unwrap()
}

fn collect(scorer: &mut dyn Scorer, mode: Mode) -> Vec<(i32, u32)> {
    let mut result = Vec::new();
    loop {
        let doc = scorer.next_doc().unwrap();
        if doc == NO_MORE_DOCS {
            break;
        }
        if !scorer.two_phase() || scorer.matches().unwrap() {
            result.push((
                doc,
                if mode.needs_scores() {
                    scorer.score().unwrap().to_bits()
                } else {
                    0
                },
            ));
        }
    }
    result
}

#[test]
fn native_phrases_match_java_documents_scores_bounds_and_competitive_iteration() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data");
    let empty = BlockTreeFields::empty();
    let classic = crate::similarities::ClassicSimilarity::default();
    let mut comparisons = 0usize;
    let mut bound_comparisons = 0usize;
    let mut case_count = 0usize;
    let mut tied_equal_position_orders = [false; 2];
    let mut suppressed_term_thresholds = 0usize;
    for family in ["codec", "postings", "deep"] {
        for variant in ["flush", "merged"] {
            let manifests = root
                .join("simpletext_phrase_scorer")
                .join(family)
                .join(variant);
            let index = if family == "deep" {
                manifests.join("index")
            } else {
                root.join("codecs_simpletext")
                    .join(family)
                    .join(variant)
                    .join("index")
            };
            let reader = crate::directory_reader::DirectoryReader::open(
                &lucene_store::FsDirectory::open(&index),
            )
            .unwrap();
            let names: Vec<_> = reader
                .segment_readers()
                .iter()
                .flat_map(|segment| segment.field_infos().fields.iter().map(|f| f.name.clone()))
                .collect();
            let norms = reader.field_norms_by_field(&names).unwrap();
            for (ordinal, segment) in reader.segment_readers().iter().enumerate() {
                let external = if family == "postings" {
                    Some(
                        SimpleTextFieldsReader::open(
                            fs::read(
                                index.join(format!("{}_SimpleText_0.pst", segment.segment_name)),
                            )
                            .unwrap(),
                            &SegmentReadState {
                                max_doc: segment.max_doc,
                                segment_id: &segment.segment_id(),
                                segment_suffix: "SimpleText_0",
                                field_infos: segment.field_infos(),
                            },
                        )
                        .unwrap(),
                    )
                } else {
                    None
                };
                let fields = external.as_ref().or_else(|| {
                    segment
                        .simpletext
                        .as_ref()
                        .and_then(|core| core.postings.as_ref())
                });
                if family == "deep" && variant == "merged" {
                    let tied = fields.unwrap().terms("tied").unwrap().unwrap();
                    let mut frequencies = Vec::new();
                    let mut boundaries = Vec::new();
                    for term in [b"left".as_slice(), b"right".as_slice()] {
                        let mut cursor = tied.iterator();
                        assert!(cursor.seek_exact(term).unwrap());
                        frequencies.push(cursor.doc_freq());
                        let mut impacts = cursor
                            .impacts(lucene_codecs::simpletext::POSITIONS)
                            .unwrap();
                        impacts.advance_shallow(154).unwrap();
                        boundaries.push(impacts.impacts().unwrap().doc_id_up_to(0).unwrap());
                    }
                    assert_eq!(frequencies, [384, 384]);
                    assert_ne!(boundaries[0], boundaries[1]);
                }
                // The postings-only fixture uses SimpleText under a
                // binary codec. Its terms are opened explicitly above,
                // so use their counters with the real binary norms too;
                // DirectoryReader does not route this mixed format yet.
                let mut external_norms = std::collections::HashMap::new();
                if let Some(fields) = &external {
                    assert_eq!(reader.segment_readers().len(), 1);
                    for info in &segment.field_infos().fields {
                        if let (Some(terms), Some(entry), Some(data)) = (
                            fields.terms(&info.name).unwrap(),
                            segment.norms_entry(info.number),
                            segment.norms_data(),
                        ) {
                            external_norms.insert(
                                info.name.clone(),
                                crate::field_norms::FieldNorms::with_avg_field_length(
                                    data,
                                    *entry,
                                    crate::field_norms::avg_field_length(
                                        terms.sum_total_term_freq(),
                                        i64::from(terms.doc_count()),
                                    ),
                                ),
                            );
                        }
                    }
                }
                let manifest = fs::read_to_string(
                    manifests.join(format!("phrases.{}.tsv", segment.segment_name)),
                )
                .unwrap();
                let rows: Vec<Vec<_>> = manifest
                    .lines()
                    .map(|line| line.split('\t').collect())
                    .collect();
                let mut at = 0usize;
                while at < rows.len() {
                    assert_eq!(rows[at][0], "query");
                    let field = String::from_utf8(unhex(rows[at][1])).unwrap();
                    let slop = rows[at][2].parse().unwrap();
                    let term_count: usize = rows[at][3].parse().unwrap();
                    at = at.saturating_add(1);
                    let collection = if rows[at][0] == "collection" {
                        let row = rows[at].clone();
                        at = at.saturating_add(1);
                        Some(row)
                    } else {
                        None
                    };
                    let mut global = crate::GlobalStats::new();
                    let mut terms = Vec::new();
                    for _ in 0..term_count {
                        let row = &rows[at];
                        assert_eq!(row[0], "term");
                        let bytes = unhex(row[1]);
                        terms.push((bytes.clone(), row[2].parse().unwrap()));
                        if let Some(c) = &collection {
                            global.insert_term(
                                &field,
                                bytes,
                                crate::CollectionStats {
                                    max_doc: c[1].parse().unwrap(),
                                    doc_count: c[2].parse().unwrap(),
                                    sum_total_term_freq: c[3].parse().unwrap(),
                                    sum_doc_freq: c[4].parse().unwrap(),
                                    doc_freq: row[3].parse().unwrap(),
                                    total_term_freq: row[4].parse().unwrap(),
                                },
                            );
                        }
                        at = at.saturating_add(1);
                    }
                    if family == "deep"
                        && variant == "merged"
                        && field == "tied"
                        && terms.len() == 2
                        && terms[0].1 == terms[1].1
                        && terms[0].0 != terms[1].0
                        && terms.iter().all(|term| {
                            term.0.as_slice() == b"left" || term.0.as_slice() == b"right"
                        })
                    {
                        let reversed = terms[0].0 > terms[1].0;
                        tied_equal_position_orders[usize::from(reversed)] = true;
                    }
                    let query = PhraseQuery::with_positions(&field, terms)
                        .unwrap()
                        .with_slop(slop);
                    while rows[at][0] == "case" {
                        let case = &rows[at];
                        let boost = f32::from_bits(score_bits(case[2]));
                        let mode = match case[3] {
                            "COMPLETE" => Mode::Complete,
                            "TOP_SCORES" => Mode::TopScores,
                            "COMPLETE_NO_SCORES" => Mode::NoScores,
                            other => panic!("{other}"),
                        };
                        let ctx = LeafContext {
                            fields: &empty,
                            doc_in: None,
                            pos_in: None,
                            pay_in: None,
                            live_docs: None,
                            points: None,
                            norms: Some(if external.is_some() {
                                &external_norms
                            } else {
                                &norms[ordinal]
                            }),
                            global: (variant != "merged").then_some(&global),
                            max_doc: Some(segment.max_doc),
                            cache: None,
                            reader: Some(segment),
                            similarity: (case[1] == "classic")
                                .then_some(&classic as &dyn crate::similarities::Similarity),
                        };
                        let build = || -> Result<Option<BoxScorer<'_>>> {
                            if family == "postings" {
                                phrase(&ctx, fields, &query, boost, mode, true)
                            } else {
                                super::super::build(
                                    &ctx,
                                    &Clause::Phrase(query.clone()),
                                    boost,
                                    mode,
                                    true,
                                )
                            }
                        };
                        at = at.saturating_add(1);
                        let start = at;
                        while rows[at][0] != "endcase" {
                            at = at.saturating_add(1);
                        }
                        let case_rows = &rows[start..at];
                        let label = format!(
                            "{family}/{variant} {}, {query:?}, {case:?}",
                            segment.segment_name
                        );
                        if case_rows.first().is_some_and(|row| row[0] == "error") {
                            let expected = String::from_utf8(unhex(case_rows[0][1])).unwrap();
                            match build() {
                                Err(crate::Error::IllegalState(actual)) => {
                                    assert_eq!(actual, expected, "{label}")
                                }
                                _ => panic!("expected Java error {expected}: {label}"),
                            }
                        } else if case_rows.first().is_some_and(|row| row[0] == "absent") {
                            assert!(build().unwrap().is_none(), "{label}");
                        } else {
                            let expected: Vec<(i32, u32)> = case_rows
                                .iter()
                                .filter(|row| row[0] == "doc")
                                .map(|row| (row[1].parse().unwrap(), score_bits(row[2])))
                                .collect();
                            let mut scorer = build().unwrap().expect("Java has a scorer");
                            assert_eq!(collect(scorer.as_mut(), mode), expected, "{label}");
                            if query.terms.len() == 1 && mode == Mode::TopScores {
                                // Java rewrites a one-term phrase to TermQuery.
                                // Without the supplier's top-level notification,
                                // a threshold must leave its iterator exhaustive.
                                let mut exhaustive = if family == "postings" {
                                    phrase(&ctx, fields, &query, boost, mode, false)
                                } else {
                                    super::super::build(
                                        &ctx,
                                        &Clause::Phrase(query.clone()),
                                        boost,
                                        mode,
                                        false,
                                    )
                                }
                                .unwrap()
                                .unwrap();
                                exhaustive.set_min_competitive_score(f32::INFINITY).unwrap();
                                assert_eq!(
                                    collect(exhaustive.as_mut(), mode),
                                    expected,
                                    "suppressed term threshold: {label}"
                                );
                                suppressed_term_thresholds =
                                    suppressed_term_thresholds.saturating_add(1);
                            }
                            comparisons = comparisons.saturating_add(expected.len());
                            let mut bounds = build().unwrap().unwrap();
                            let mut previous_target = -1;
                            let mut boundary = -1;
                            for row in case_rows.iter().filter(|row| row[0] == "bound") {
                                let target = row[1].parse().unwrap();
                                if target != previous_target {
                                    boundary = bounds.advance_shallow(target).unwrap();
                                    assert_eq!(bounds.doc_id(), -1, "shallow moved docs: {label}");
                                    previous_target = target;
                                }
                                assert_eq!(
                                    boundary,
                                    row[2].parse::<i32>().unwrap(),
                                    "{label}, target {target}"
                                );
                                assert_eq!(
                                    bounds.max_score(row[3].parse().unwrap()).unwrap().to_bits(),
                                    score_bits(row[4]),
                                    "{label}, bound {row:?}"
                                );
                                bound_comparisons = bound_comparisons.saturating_add(1);
                            }
                            if let Some(threshold) =
                                case_rows.iter().find(|row| row[0] == "threshold")
                            {
                                let mut pruned = build().unwrap().unwrap();
                                pruned
                                    .set_min_competitive_score(f32::from_bits(score_bits(
                                        threshold[1],
                                    )))
                                    .unwrap();
                                let expected: Vec<_> = case_rows
                                    .iter()
                                    .filter(|row| row[0] == "competitive")
                                    .map(|row| (row[1].parse().unwrap(), score_bits(row[2])))
                                    .collect();
                                assert_eq!(
                                    collect(pruned.as_mut(), mode),
                                    expected,
                                    "competitive: {label}"
                                );
                            }
                        }
                        case_count = case_count.saturating_add(1);
                        at = at.saturating_add(1);
                    }
                    assert_eq!(rows[at][0], "endquery");
                    at = at.saturating_add(1);
                }
            }
        }
    }
    assert!(case_count > 1000);
    assert!(comparisons > 10000);
    assert!(bound_comparisons > 1000);
    assert_eq!(tied_equal_position_orders, [true, true]);
    assert!(suppressed_term_thresholds > 0);
}
