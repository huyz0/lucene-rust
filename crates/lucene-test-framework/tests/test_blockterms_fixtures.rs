//! Actual Lucene test-framework formats, not local lookalike formats.
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::blockterms::SeekStatus;
use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PosInput, PostingsFlags};
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::blockterms::BlockTermsFormat;

#[test]
fn actual_java_test_formats_match_reads_seeks_and_every_written_byte() {
    for (case, format) in [
        ("LuceneFixedGap", BlockTermsFormat::fixed_gap()),
        (
            "LuceneVarGapFixedInterval",
            BlockTermsFormat::var_gap_fixed_interval(),
        ),
        (
            "LuceneVarGapDocFreqInterval",
            BlockTermsFormat::VarGapDocFreqInterval {
                doc_freq_threshold: 6,
                term_index_interval: 24,
            },
        ),
        (
            "CustomFixedGap",
            BlockTermsFormat::FixedGap {
                term_index_interval: 7,
            },
        ),
        (
            "CustomVarGapFixedInterval",
            BlockTermsFormat::VarGapFixedInterval {
                term_index_interval: 11,
            },
        ),
        (
            "DefaultVarGapDocFreqInterval",
            BlockTermsFormat::var_gap_doc_freq_interval(),
        ),
    ] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/blockterms")
            .join(case);
        let suffix = format!("{}_0", format.name());
        let mut files = BTreeMap::new();
        for entry in fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            files.insert(
                path.extension().unwrap().to_str().unwrap().to_string(),
                fs::read(path).unwrap(),
            );
        }
        let id = [7; 16];
        let infos = field_infos::parse(&files["fnm"], &id, "").unwrap();
        let state = SegmentReadState {
            max_doc: 500,
            segment_id: &id,
            segment_suffix: &suffix,
            field_infos: &infos,
        };
        let reader = format.fields_producer(&files, &state).unwrap();
        reader.check_integrity().unwrap();
        let terms = reader.terms("body").unwrap();
        assert_eq!(terms.doc_count(), 500);
        assert_eq!(terms.sum_total_term_freq(), 2000);
        let doc = DocInput::open(&files["doc"], &id, &suffix).unwrap();
        let pos = PosInput::open(&files["pos"], &id, &suffix).unwrap();
        let mut walk = terms.iterator();
        let mut seek = terms.iterator();
        let mut write_terms = Vec::new();
        let expected = fs::read_to_string(root.join("postings.tsv")).unwrap();
        for (ord, line) in expected.lines().enumerate() {
            let row: Vec<_> = line.split('\t').collect();
            let term: Vec<_> = row[0]
                .as_bytes()
                .chunks_exact(2)
                .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap())
                .collect();
            let mut write_term = TermPostings {
                term: term.clone(),
                ..TermPostings::default()
            };
            assert_eq!(walk.next().unwrap(), Some(term.as_slice()));
            assert_eq!(walk.doc_freq().unwrap(), row[1].parse::<i32>().unwrap());
            let postings = walk.postings(&doc, PostingsFlags::Freqs).unwrap();
            let positions = postings::read_positions(
                &pos,
                None,
                walk.term_state().unwrap().meta,
                &postings.freqs,
                walk.total_term_freq().unwrap(),
                IndexOptions::DocsAndFreqsAndPositions,
                false,
            )
            .unwrap();
            assert_eq!(postings.docs.len(), row.len() - 2);
            for (i, want) in row[2..].iter().enumerate() {
                let values: Vec<i32> = want.split(':').map(|v| v.parse().unwrap()).collect();
                write_term.docs.push((values[0], values[1]));
                write_term.positions.push(values[2..].to_vec());
                assert_eq!(postings.docs[i], values[0]);
                assert_eq!(postings.freqs[i], values[1]);
                assert_eq!(
                    positions[i].iter().map(|p| p.position).collect::<Vec<_>>(),
                    values[2..]
                );
            }
            assert_eq!(seek.seek_ceil(&term).unwrap(), SeekStatus::Found);
            assert_eq!(seek.doc_freq().unwrap(), walk.doc_freq().unwrap());
            if matches!(format, BlockTermsFormat::FixedGap { .. }) {
                seek.seek_exact_ord(ord as i64).unwrap();
                assert_eq!(seek.term(), term);
            }
            write_terms.push(write_term);
        }
        assert_eq!(walk.next().unwrap(), None);
        assert_eq!(seek.seek_ceil(b"zzzz").unwrap(), SeekStatus::End);
        assert!(!seek.seek_exact(b"absent").unwrap());
        // Rebuild from Java's observed postings, independently of Rust's
        // reader, and require every output file to match Java byte for byte.
        let write_state = SegmentWriteState {
            max_doc: state.max_doc,
            segment_id: &id,
            segment_suffix: &suffix,
            field_infos: &infos,
        };
        let field = infos.field_by_name("body").unwrap();
        let output = format
            .fields_consumer(
                &[FieldPostingsInput {
                    field_number: field.number,
                    index_options: field.index_options,
                    doc_count: 500,
                    has_payloads: false,
                    terms: &write_terms,
                }],
                &[],
                &write_state,
            )
            .unwrap();
        assert_eq!(output.len(), files.len() - 2); // .fnm and manifest are inputs only.
        for (ext, bytes) in output {
            assert_eq!(bytes, files[&ext], "{case}.{ext}");
        }
        // Missing inputs, truncated indexes, wrong ids and dictionary damage
        // must propagate as errors instead of opening an unverified reader.
        let extensions = if matches!(format, BlockTermsFormat::FixedGap { .. }) {
            vec!["tii", "tib"]
        } else {
            vec!["tmv", "tiv", "tib"]
        };
        for ext in extensions {
            let mut bad = files.clone();
            bad.remove(ext);
            assert!(format.fields_producer(&bad, &state).is_err());
            bad.insert(ext.to_string(), vec![0; 4]);
            assert!(format.fields_producer(&bad, &state).is_err());
        }
        let wrong_id = [8; 16];
        let wrong_state = SegmentReadState {
            segment_id: &wrong_id,
            ..state
        };
        assert!(format.fields_producer(&files, &wrong_state).is_err());
        let mut bad = files.clone();
        let tib = bad.get_mut("tib").unwrap();
        let at = tib.len() / 2;
        tib[at] ^= 1;
        let damaged = format.fields_producer(&bad, &state).unwrap();
        assert!(damaged.check_integrity().is_err());
    }
}
