//! The actual Java test formats, including encoding instrumentation.
use std::{collections::BTreeMap, fs, path::PathBuf};

use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::uniformsplit::{BlockDecoder, BlockEncoder, SeekStatus};
use lucene_test_framework::codecs::rot13::{
    EncodingState, Rot13Cypher, StUniformSplitRot13PostingsFormat, UniformSplitRot13PostingsFormat,
    ENCODING_FLAGS,
};

fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap())
        .collect()
}

fn check_flags(row: &str) {
    let row: Vec<bool> = row
        .split('\t')
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(
        ENCODING_FLAGS.snapshot(),
        EncodingState {
            encoder_called: row[0],
            decoder_called: row[1],
            blocks_encoded: row[2],
            fields_metadata_encoded: row[3],
            dictionary_encoded: row[4],
        }
    );
}

#[test]
fn java_rot13_bytes_flags_and_all_postings_match() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data/rot13");
    for line in fs::read_to_string(root.join("cypher.tsv")).unwrap().lines() {
        let (plain, encoded) = line.split_once('\t').unwrap();
        let (plain, encoded) = (unhex(plain), unhex(encoded));
        assert_eq!(Rot13Cypher::encode(&plain).unwrap(), encoded);
        assert_eq!(BlockEncoder::encode(&Rot13Cypher, &plain).unwrap(), encoded);
        assert_eq!(Rot13Cypher::decode(&encoded).unwrap(), plain);
        assert_eq!(BlockDecoder::decode(&Rot13Cypher, &encoded).unwrap(), plain);
    }
    for shared in [false, true] {
        let format = UniformSplitRot13PostingsFormat::default();
        let shared_format = StUniformSplitRot13PostingsFormat;
        let name = if shared {
            shared_format.name()
        } else {
            format.name()
        };
        let (blocks, dictionary) = if shared {
            ("stustb", "stustd")
        } else {
            ("ustb", "ustd")
        };
        for count in [0, 1, 500] {
            let root = root.join(name).join(count.to_string());
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
            let suffix = format!("{name}_0");
            let infos = field_infos::parse(&files["fnm"], &id, "").unwrap();
            let state = SegmentReadState {
                max_doc: count,
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
                let mut term = TermPostings {
                    term: unhex(row[1]),
                    ..TermPostings::default()
                };
                for posting in &row[4..] {
                    let values: Vec<i32> = posting.split(':').map(|v| v.parse().unwrap()).collect();
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
                        doc_count: count,
                        has_payloads: false,
                        terms,
                    }
                })
                .collect();
            let write_state = SegmentWriteState {
                max_doc: count,
                segment_id: &id,
                segment_suffix: &suffix,
                field_infos: &infos,
            };
            ENCODING_FLAGS.reset();
            let output = if shared {
                shared_format.fields_consumer(&inputs, &[], &write_state)
            } else {
                format.fields_consumer(&inputs, &[], &write_state)
            }
            .unwrap();
            let flags = fs::read_to_string(root.join("flags.tsv")).unwrap();
            let flags: Vec<_> = flags.lines().collect();
            check_flags(flags[0]);
            assert_eq!(output.len(), files.len() - 1);
            for (ext, bytes) in output {
                assert_eq!(bytes, files[&ext], "{name}/{count}.{ext}");
            }
            let doc = DocInput::open(&files["doc"], &id, &suffix).unwrap();
            let pos = PosInput::open(&files["pos"], &id, &suffix).unwrap();
            let stats = fs::read_to_string(root.join("stats.tsv")).unwrap();
            macro_rules! check_reader {
                ($reader:expr) => {{
                    let reader = $reader.unwrap();
                    check_flags(flags[1]);
                    reader.check_integrity().unwrap();
                    assert_eq!(
                        reader.field_names().collect::<Vec<_>>(),
                        expected.keys().map(String::as_str).collect::<Vec<_>>()
                    );
                    for line in stats.lines() {
                        let row: Vec<_> = line.split('\t').collect();
                        let field = reader.terms(row[0]).unwrap();
                        let want_stats: Vec<i64> =
                            row[1..].iter().map(|s| s.parse().unwrap()).collect();
                        assert_eq!(
                            [
                                field.size(),
                                field.sum_doc_freq(),
                                field.sum_total_term_freq(),
                                i64::from(field.doc_count())
                            ],
                            want_stats.as_slice()
                        );
                        let terms = &expected[row[0]];
                        let mut walk = field.iter();
                        let mut seek = field.iter();
                        for (ord, want) in terms.iter().enumerate() {
                            assert_eq!(walk.next().unwrap(), Some(want.term.as_slice()));
                            assert_eq!(walk.doc_freq().unwrap() as usize, want.docs.len());
                            assert_eq!(
                                walk.total_term_freq().unwrap(),
                                want.docs.iter().map(|d| i64::from(d.1)).sum::<i64>()
                            );
                            let got = walk.postings(&doc, PostingsFlags::Freqs).unwrap();
                            assert_eq!(got.docs, want.docs.iter().map(|d| d.0).collect::<Vec<_>>());
                            assert_eq!(
                                got.freqs,
                                want.docs.iter().map(|d| d.1).collect::<Vec<_>>()
                            );
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
                            assert_eq!(
                                positions
                                    .iter()
                                    .map(|p| p.iter().map(|p| p.position).collect::<Vec<_>>())
                                    .collect::<Vec<_>>(),
                                want.positions
                            );
                            assert_eq!(seek.seek_ceil(&want.term).unwrap(), SeekStatus::Found);
                            assert_eq!(seek.term(), Some(want.term.as_slice()));
                            assert_eq!(seek.doc_freq().unwrap(), walk.doc_freq().unwrap());
                            assert_eq!(
                                seek.next().unwrap(),
                                terms.get(ord + 1).map(|t| t.term.as_slice())
                            );
                        }
                        assert!(walk.next().unwrap().is_none());
                        assert_eq!(seek.seek_ceil(b"common\0").unwrap(), SeekStatus::NotFound);
                        assert_eq!(
                            seek.term(),
                            terms
                                .iter()
                                .find(|t| t.term.as_slice() > b"common\0")
                                .map(|t| t.term.as_slice())
                        );
                        assert_eq!(seek.seek_ceil(b"zzzz").unwrap(), SeekStatus::End);
                    }
                    check_flags(flags[2]);
                }};
            }
            if shared {
                check_reader!(shared_format.fields_producer(&files, &state));
            } else {
                check_reader!(format.fields_producer(&files, &state));
            }
            // Each dictionary/blocks file must exist, have a valid header and
            // pass checksum verification. Blocks' full checksum is deferred
            // to check_integrity, just as in the underlying reader.
            for ext in [blocks, dictionary] {
                let mut bad = files.clone();
                bad.remove(ext);
                if shared {
                    assert!(shared_format.fields_producer(&bad, &state).is_err());
                } else {
                    assert!(format.fields_producer(&bad, &state).is_err());
                }
                bad.insert(ext.to_string(), vec![0; 4]);
                if shared {
                    assert!(shared_format.fields_producer(&bad, &state).is_err());
                } else {
                    assert!(format.fields_producer(&bad, &state).is_err());
                }
                let mut bad = files.clone();
                let bytes = bad.get_mut(ext).unwrap();
                let at = bytes.len() / 2;
                bytes[at] ^= 1;
                if shared {
                    if let Ok(reader) = shared_format.fields_producer(&bad, &state) {
                        assert!(reader.check_integrity().is_err());
                    }
                } else if let Ok(reader) = format.fields_producer(&bad, &state) {
                    assert!(reader.check_integrity().is_err());
                }
            }
            for wrong in [
                SegmentReadState {
                    segment_id: &[8; 16],
                    ..state
                },
                SegmentReadState {
                    segment_suffix: "wrong",
                    ..state
                },
            ] {
                if shared {
                    assert!(shared_format.fields_producer(&files, &wrong).is_err());
                } else {
                    assert!(format.fields_producer(&files, &wrong).is_err());
                }
            }
        }
    }
}
