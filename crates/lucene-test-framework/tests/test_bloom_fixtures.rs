//! The actual Java low-memory bloom adapter, including a saturated filter.
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lucene_codecs::field_infos::{self, IndexOptions};
use lucene_codecs::postings::{self, DocInput, PosInput};
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::bloom::TestBloomFilteredLucenePostings;

#[test]
fn java_adapter_reads_and_writes_every_byte_including_saturated_filters() {
    for (case, count) in [("sparse", 500), ("saturated", 30_000)] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/bloom")
            .join(case);
        let mut files = BTreeMap::new();
        for entry in fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            files.insert(
                path.extension().unwrap().to_str().unwrap().to_string(),
                fs::read(path).unwrap(),
            );
        }
        let id = [7; 16];
        let suffix = "TestBloomFilteredLucenePostings_0";
        let infos = field_infos::parse(&files["fnm"], &id, "").unwrap();
        let state = SegmentReadState {
            max_doc: count,
            segment_id: &id,
            segment_suffix: suffix,
            field_infos: &infos,
        };
        let format = TestBloomFilteredLucenePostings::default();
        let open = |blm: &[u8]| {
            format.fields_producer(blm, &files["tim"], &files["tip"], &files["tmd"], &state)
        };
        let reader = open(&files["blm"]).unwrap();
        assert_eq!(reader.delegate_name(), "Lucene104");
        let bloom = reader.bloom("body").unwrap();
        if case == "saturated" {
            assert!(bloom.saturation() > 0.9);
        } else {
            assert!(bloom.saturation() < 0.5);
        }
        let terms = reader.terms("body").unwrap();
        assert_eq!(terms.doc_count(), count);
        assert_eq!(terms.sum_total_term_freq(), i64::from(count) * 4);
        let doc = DocInput::open(&files["doc"], &id, suffix).unwrap();
        let pos = PosInput::open(&files["pos"], &id, suffix).unwrap();
        let mut walk = terms.iter();
        let mut write_terms = Vec::new();
        for line in fs::read_to_string(root.join("postings.tsv"))
            .unwrap()
            .lines()
        {
            let row: Vec<_> = line.split('\t').collect();
            let term: Vec<_> = row[0]
                .as_bytes()
                .chunks_exact(2)
                .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap())
                .collect();
            let (next, stats) = walk.try_next().unwrap().unwrap();
            assert_eq!(next, term);
            assert_eq!(stats.doc_freq, row[1].parse::<i32>().unwrap());
            assert_eq!(terms.try_seek_exact(&term).unwrap(), Some(stats));
            let meta = terms.delegate().term_metadata(&term).unwrap().unwrap();
            let postings = terms
                .delegate()
                .postings(&term, Some(&doc))
                .unwrap()
                .unwrap();
            let positions = postings::read_positions(
                &pos,
                None,
                meta,
                &postings.freqs,
                stats.total_term_freq,
                IndexOptions::DocsAndFreqsAndPositions,
                false,
            )
            .unwrap();
            let mut input = TermPostings {
                term,
                ..TermPostings::default()
            };
            assert_eq!(postings.docs.len(), row.len() - 2);
            for (i, want) in row[2..].iter().enumerate() {
                let values: Vec<i32> = want.split(':').map(|v| v.parse().unwrap()).collect();
                assert_eq!(postings.docs[i], values[0]);
                assert_eq!(postings.freqs[i], values[1]);
                assert_eq!(
                    positions[i].iter().map(|p| p.position).collect::<Vec<_>>(),
                    values[2..]
                );
                input.docs.push((values[0], values[1]));
                input.positions.push(values[2..].to_vec());
            }
            write_terms.push(input);
        }
        assert!(walk.try_next().unwrap().is_none());
        for absent in [b"absent".as_slice(), b"t999999", b"zzzz"] {
            assert!(terms.try_seek_exact(absent).unwrap().is_none());
        }
        let field = infos.field_by_name("body").unwrap();
        let output = format
            .fields_consumer(
                &[FieldPostingsInput {
                    field_number: field.number,
                    index_options: field.index_options,
                    doc_count: count,
                    has_payloads: false,
                    terms: &write_terms,
                }],
                &[],
                &SegmentWriteState {
                    max_doc: count,
                    segment_id: &id,
                    segment_suffix: suffix,
                    field_infos: &infos,
                },
            )
            .unwrap();
        let delegate = output.delegate;
        for (ext, bytes) in [
            ("blm", output.blm),
            ("doc", delegate.doc),
            ("pos", delegate.pos),
            ("pay", delegate.pay),
            ("psm", delegate.psm),
            ("tim", delegate.tim),
            ("tip", delegate.tip),
            ("tmd", delegate.tmd),
        ] {
            if bytes.is_empty() {
                assert!(!files.contains_key(ext));
            } else {
                assert_eq!(bytes, files[ext], "{case}.{ext}");
            }
        }
        let mut bad = files["blm"].clone();
        let at = bad.len() / 2;
        bad[at] ^= 1;
        assert!(open(&bad).is_err());
        assert!(open(&files["blm"][..8]).is_err());
        let wrong = SegmentReadState {
            segment_id: &[8; 16],
            ..state
        };
        assert!(format
            .fields_producer(
                &files["blm"],
                &files["tim"],
                &files["tip"],
                &files["tmd"],
                &wrong
            )
            .is_err());
    }
}
