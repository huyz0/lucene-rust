//! Actual MockRandom selections, including successive writers on one instance.
use std::{collections::BTreeMap, fs, path::PathBuf};

use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::mockrandom::{
    MockRandomFieldsProducer, MockRandomPostingsFormat,
};
use lucene_util::java_random::JavaRandom;

fn assert_postings(
    want: &TermPostings,
    got: postings::Postings,
    positions: Vec<Vec<postings::Position>>,
) {
    assert_eq!(got.docs, want.docs.iter().map(|d| d.0).collect::<Vec<_>>());
    assert_eq!(got.freqs, want.docs.iter().map(|d| d.1).collect::<Vec<_>>());
    assert_eq!(
        positions
            .iter()
            .map(|p| p.iter().map(|p| p.position).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        want.positions
    );
}

#[test]
fn every_java_branch_matches_files_stats_postings_and_seeks() {
    for case in [
        "fst",
        "blocktree",
        "ords",
        "fixed",
        "every-n",
        "doc-freq",
        "random-gap",
    ] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/mockrandom")
            .join(case);
        let selection = fs::read_to_string(root.join("0/selection.tsv")).unwrap();
        let selection: Vec<_> = selection.trim().split('\t').collect();
        let mut caller = JavaRandom::new(selection[0].parse().unwrap());
        let mut writer = MockRandomPostingsFormat::new(&mut caller);
        assert_eq!(caller.next_long(), selection[1].parse::<i64>().unwrap());
        for slot in 0..2 {
            let root = root.join(slot.to_string());
            let mut files = BTreeMap::new();
            for entry in fs::read_dir(&root).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().unwrap() != "tsv" {
                    files.insert(
                        path.extension().unwrap().to_str().unwrap().to_string(),
                        fs::read(path).unwrap(),
                    );
                }
            }
            let id = [7; 16];
            let suffix = format!("MockRandom_{slot}");
            let infos = field_infos::parse(&files["fnm"], &id, "").unwrap();
            let state = SegmentReadState {
                max_doc: 500,
                segment_id: &id,
                segment_suffix: &suffix,
                field_infos: &infos,
            };
            let mut expected: BTreeMap<String, Vec<TermPostings>> = BTreeMap::new();
            for line in fs::read_to_string(root.join("postings.tsv"))
                .unwrap()
                .lines()
            {
                let row: Vec<_> = line.split('\t').collect();
                let term = row[1]
                    .as_bytes()
                    .chunks_exact(2)
                    .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap())
                    .collect();
                let mut term = TermPostings {
                    term,
                    ..TermPostings::default()
                };
                for doc in &row[4..] {
                    let values: Vec<i32> = doc.split(':').map(|v| v.parse().unwrap()).collect();
                    term.docs.push((values[0], values[1]));
                    term.positions.push(values[2..].to_vec());
                }
                assert_eq!(term.docs.len(), row[2].parse::<usize>().unwrap());
                assert_eq!(
                    term.docs.iter().map(|d| i64::from(d.1)).sum::<i64>(),
                    row[3].parse::<i64>().unwrap()
                );
                expected.entry(row[0].into()).or_default().push(term);
            }
            let inputs: Vec<_> = expected
                .iter()
                .map(|(name, terms)| {
                    let field = infos.field_by_name(name).unwrap();
                    FieldPostingsInput {
                        field_number: field.number,
                        index_options: field.index_options,
                        doc_count: 500,
                        has_payloads: false,
                        terms,
                    }
                })
                .collect();
            let write_state = SegmentWriteState {
                max_doc: 500,
                segment_id: &id,
                segment_suffix: &suffix,
                field_infos: &infos,
            };
            let output = writer.fields_consumer(&inputs, &[], &write_state).unwrap();
            assert_eq!(output.len(), files.len() - 1);
            for (ext, bytes) in output {
                assert_eq!(bytes, files[&ext], "{case}/{slot}.{ext}");
            }
            let reader_format = MockRandomPostingsFormat::default();
            let reader = reader_format.fields_producer(&files, &state).unwrap();
            let names: Vec<_> = match &reader {
                MockRandomFieldsProducer::Fst(r) => r.field_names().collect(),
                MockRandomFieldsProducer::Ords(r) => r.field_names().collect(),
                MockRandomFieldsProducer::BlockTerms(r) => r.field_names().collect(),
                MockRandomFieldsProducer::BlockTree(r) => {
                    r.iter_fields().map(|(name, _)| name).collect()
                }
            };
            assert_eq!(
                names,
                expected.keys().map(String::as_str).collect::<Vec<_>>()
            );
            let doc = DocInput::open(&files["doc"], &id, &suffix).unwrap();
            let pos = PosInput::open(&files["pos"], &id, &suffix).unwrap();
            let stats = fs::read_to_string(root.join("stats.tsv")).unwrap();
            for line in stats.lines() {
                let row: Vec<_> = line.split('\t').collect();
                let name = row[0];
                let want_stats: Vec<i64> = row[1..].iter().map(|v| v.parse().unwrap()).collect();
                let terms = &expected[name];
                // The dictionaries have distinct enum types; keep assertions
                // identical while invoking each reader's native API.
                macro_rules! check_dictionary {
                    ($field:expr, $iterator:ident, $($unwrap:tt)*) => {{
                        let field = $field;
                        assert_eq!(
                            [
                                field.size(),
                                field.sum_doc_freq(),
                                field.sum_total_term_freq(),
                                i64::from(field.doc_count())
                            ],
                            want_stats.as_slice()
                        );
                        let mut walk = field.$iterator();
                        let mut seek = field.$iterator();
                        for want in terms {
                            assert_eq!(walk.next().unwrap(), Some(want.term.as_slice()));
                            assert_eq!(walk.doc_freq()$($unwrap)* as usize, want.docs.len());
                            assert_eq!(walk.total_term_freq()$($unwrap)*, want.docs.iter().map(|d| i64::from(d.1)).sum::<i64>());
                            let got = walk.postings(&doc, PostingsFlags::Freqs).unwrap();
                            let metadata = walk.term_metadata().unwrap();
                            let positions = postings::read_positions(
                                &pos,
                                None,
                                metadata,
                                &got.freqs,
                                want.docs.iter().map(|d| i64::from(d.1)).sum(),
                                IndexOptions::DocsAndFreqsAndPositions,
                                false,
                            )
                            .unwrap();
                            assert_postings(want, got, positions);
                            assert_eq!(
                                format!("{:?}", seek.seek_ceil(&want.term).unwrap()),
                                "Found"
                            );
                            assert_eq!(
                                seek.postings(&doc, PostingsFlags::Freqs).unwrap().docs,
                                want.docs.iter().map(|d| d.0).collect::<Vec<_>>()
                            );
                        }
                        assert!(walk.next().unwrap().is_none());
                        assert_eq!(format!("{:?}", seek.seek_ceil(b"zzzz").unwrap()), "End");
                    }};
                }
                match &reader {
                    MockRandomFieldsProducer::Fst(r) => {
                        check_dictionary!(r.terms(name).unwrap(), iter,);
                    }
                    MockRandomFieldsProducer::Ords(r) => {
                        check_dictionary!(r.terms(name).unwrap(), iter, .unwrap());
                    }
                    MockRandomFieldsProducer::BlockTerms(r) => {
                        let field = r.terms(name).unwrap();
                        assert_eq!(
                            [
                                field.size(),
                                field.sum_doc_freq(),
                                field.sum_total_term_freq(),
                                i64::from(field.doc_count())
                            ],
                            want_stats.as_slice()
                        );
                        let mut walk = field.iterator();
                        let mut seek = field.iterator();
                        for want in terms {
                            assert_eq!(walk.next().unwrap(), Some(want.term.as_slice()));
                            let got = walk.postings(&doc, PostingsFlags::Freqs).unwrap();
                            let positions = postings::read_positions(
                                &pos,
                                None,
                                walk.term_state().unwrap().meta,
                                &got.freqs,
                                walk.total_term_freq().unwrap(),
                                IndexOptions::DocsAndFreqsAndPositions,
                                false,
                            )
                            .unwrap();
                            assert_postings(want, got, positions);
                            assert_eq!(
                                format!("{:?}", seek.seek_ceil(&want.term).unwrap()),
                                "Found"
                            );
                            assert_eq!(seek.doc_freq().unwrap() as usize, want.docs.len());
                        }
                        assert!(walk.next().unwrap().is_none());
                        assert_eq!(format!("{:?}", seek.seek_ceil(b"zzzz").unwrap()), "End");
                    }
                    MockRandomFieldsProducer::BlockTree(r) => {
                        let field = r.field(name).unwrap();
                        assert_eq!(
                            [
                                field.num_terms,
                                field.sum_doc_freq,
                                field.sum_total_term_freq,
                                i64::from(field.doc_count)
                            ],
                            want_stats.as_slice()
                        );
                        let mut walk = field.iter();
                        let mut seek = field.iter();
                        for want in terms {
                            assert_eq!(walk.try_next_term().unwrap(), Some(want.term.as_slice()));
                            let (got, positions) = walk
                                .try_current_postings_and_positions(Some(&doc), &pos, None)
                                .unwrap()
                                .unwrap();
                            assert_postings(want, got, positions);
                            assert_eq!(
                                format!("{:?}", seek.try_seek_ceil(&want.term).unwrap()),
                                "Found"
                            );
                            assert_eq!(
                                seek.try_stats().unwrap().unwrap().doc_freq as usize,
                                want.docs.len()
                            );
                        }
                        assert!(walk.try_next_term().unwrap().is_none());
                        assert_eq!(format!("{:?}", seek.try_seek_ceil(b"zzzz").unwrap()), "End");
                    }
                }
            }
            for ext in files
                .keys()
                .filter(|ext| !matches!(ext.as_str(), "fnm" | "doc" | "pos" | "psm" | "pay"))
            {
                let mut bad = files.clone();
                bad.remove(ext);
                assert!(
                    reader_format.fields_producer(&bad, &state).is_err(),
                    "missing {case}/{slot}.{ext}"
                );
                bad.insert(ext.clone(), vec![0; 4]);
                assert!(
                    reader_format.fields_producer(&bad, &state).is_err(),
                    "truncated {case}/{slot}.{ext}"
                );
            }
            let mut bad = files.clone();
            let sd = bad.get_mut("sd").unwrap();
            let at = sd.len() - 1;
            sd[at] ^= 1;
            assert!(reader_format.fields_producer(&bad, &state).is_err());
            for len in 0..files["sd"].len() {
                let mut bad = files.clone();
                bad.insert("sd".into(), files["sd"][..len].to_vec());
                assert!(
                    reader_format.fields_producer(&bad, &state).is_err(),
                    "seed truncated to {len}"
                );
            }
            let mut bad = files.clone();
            let sd = bad.get_mut("sd").unwrap();
            sd.truncate(sd.len() - lucene_store::codec_util::FOOTER_LENGTH);
            sd.push(0); // Even a valid checksum cannot hide an extra seed-file byte.
            lucene_store::codec_util::write_footer(sd);
            assert!(reader_format.fields_producer(&bad, &state).is_err());
            let wrong = SegmentReadState {
                segment_id: &[8; 16],
                ..state
            };
            assert!(reader_format.fields_producer(&files, &wrong).is_err());
            let wrong = SegmentReadState {
                segment_suffix: "wrong",
                ..state
            };
            assert!(reader_format.fields_producer(&files, &wrong).is_err());
        }
    }
}
