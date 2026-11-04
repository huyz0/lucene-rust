//! Differential test against OpenSearch 3.9.0's neural-search plugin.
//!
//! `scripts/gen-seismic-fixture.py` indexes seeded random sparse vectors into
//! a real node (two flushed compound segments in `two_segments`, the same
//! documents force-merged into one in `merged`, `approximate_threshold` 100 so
//! every segment is clustered), runs `neural_sparse` queries with several
//! `method_parameters` and records each hit's `_id` and `_score`. This test
//! reads the same shard files and must return the same hits with bit-equal
//! scores.

// Test-side arithmetic on fixture values, not on index bytes.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::stored_fields::FieldValue;
use lucene_index::compound_reader::CompoundReader;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_sparse::search::SegmentView;
use lucene_sparse::{postings, ClusteredPostings, SeismicField, SeismicQuery};
use lucene_store::directory::MmapDirectory;
use lucene_store::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;
use std::sync::Arc;

fn fixture_dir() -> String {
    format!("{}/../../fixtures/data/seismic", env!("CARGO_MANIFEST_DIR"))
}

struct Case {
    top_n: usize,
    k: usize,
    heap_factor: f32,
    filter: Option<String>,
    tokens: Vec<(u32, f32)>,
    hits: Vec<(String, Vec<(String, f32)>)>,
}

fn load_cases() -> Vec<Case> {
    let text = std::fs::read_to_string(format!("{}/expected.tsv", fixture_dir())).unwrap();
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        match cols[0] {
            "query" => cases.push(Case {
                top_n: cols[1].parse().unwrap(),
                k: cols[2].parse().unwrap(),
                heap_factor: cols[3].parse().unwrap(),
                filter: (cols[4] != "-").then(|| cols[4].to_string()),
                tokens: Vec::new(),
                hits: Vec::new(),
            }),
            "tokens" => {
                cases.last_mut().unwrap().tokens = cols[1]
                    .split(',')
                    .map(|p| {
                        let (t, w) = p.split_once(':').unwrap();
                        (t.parse().unwrap(), w.parse().unwrap())
                    })
                    .collect();
            }
            "hits" => {
                let hits = if cols.len() < 3 || cols[2].is_empty() {
                    Vec::new()
                } else {
                    cols[2]
                        .split(',')
                        .map(|p| {
                            let (id, s) = p.split_once(':').unwrap();
                            (id.to_string(), s.parse().unwrap())
                        })
                        .collect()
                };
                cases
                    .last_mut()
                    .unwrap()
                    .hits
                    .push((cols[1].to_string(), hits));
            }
            other => panic!("unexpected line kind {other}"),
        }
    }
    cases
}

/// `Uid.decodeId` for the ids this fixture uses: UTF-8 ones (marker 0xFF)
/// and URL-safe base64 ones.
fn decode_id(bytes: &[u8]) -> String {
    if bytes.first() == Some(&0xFF) {
        return String::from_utf8(bytes[1..].to_vec()).unwrap();
    }
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | (u32::from(b) << (16 - 8 * i)));
        let chars = chunk.len() + 1;
        for i in 0..chars {
            out.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

struct Segment {
    ids: Vec<String>,
    postings: ClusteredPostings,
}

fn open_segment(dir: &dyn Directory, seg: &SegmentReader) -> Segment {
    let name = &seg.segment_name;
    let id = seg.segment_id();
    let compound = dir
        .list_all()
        .unwrap()
        .contains(&format!("{name}.cfs"))
        .then(|| CompoundReader::open(dir, name, &id).unwrap());
    let files: &dyn Directory = match &compound {
        Some(c) => c,
        None => dir,
    };
    let sit = files
        .open(&format!("{name}.{}", postings::TERMS_EXTENSION))
        .unwrap();
    let sip = Arc::new(
        files
            .open(&format!("{name}.{}", postings::POSTINGS_EXTENSION))
            .unwrap(),
    );
    let postings = ClusteredPostings::open(&sit, sip, &id).unwrap();
    let id_field = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "_id")
        .unwrap()
        .number;
    let ids = (0..seg.max_doc)
        .map(|d| {
            let doc = seg.stored_document(d).unwrap().unwrap();
            let f = doc
                .fields
                .iter()
                .find(|f| f.field_number == id_field)
                .unwrap();
            match &f.value {
                FieldValue::Binary(b) => decode_id(b),
                v => panic!("_id is {v:?}"),
            }
        })
        .collect();
    Segment { ids, postings }
}

#[test]
fn seismic_hits_match_neural_search() {
    lucene_sparse::register_codec();
    let cases = load_cases();
    assert_eq!(cases.len(), 60);
    let mut checked = 0;
    for index in ["merged", "two_segments"] {
        let path = format!("{}/{index}", fixture_dir());
        let dir = MmapDirectory::open(&path);
        let reader = DirectoryReader::open(&dir).unwrap();
        let segments: Vec<(&SegmentReader, Segment)> = reader
            .segment_readers()
            .iter()
            .map(|s| (s, open_segment(&dir, s)))
            .collect();
        let info = segments[0]
            .0
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "emb")
            .unwrap();
        let field = SeismicField::from_field_info(info).unwrap().unwrap();
        assert!(!field.native_engine);
        assert_eq!(field.approximate_threshold, 100);
        for (ci, case) in cases.iter().enumerate() {
            let query = SeismicQuery::new(
                &case.tokens,
                case.top_n,
                case.k,
                case.heap_factor,
                1.0,
                &field,
            )
            .unwrap();
            let mut hits: Vec<(String, f32)> = Vec::new();
            for (seg, s) in &segments {
                assert!(field.is_seismic_segment(seg.max_doc));
                let (meta, data) = seg.doc_values_for_field(field.number).unwrap();
                let entry = meta.binary_entry(field.number).unwrap();
                let view = SegmentView {
                    max_doc: seg.max_doc,
                    live_docs: seg.live_docs(),
                    doc_values: Some((data, entry)),
                    postings: &s.postings,
                };
                // The fixture's `tag` is "even" for d0, d2, ...: the filter
                // bitset is built from the ids rather than a term query.
                let filter = case.filter.as_ref().map(|tag| {
                    let mut bits = FixedBitSet::new(seg.max_doc as usize);
                    for (d, id) in s.ids.iter().enumerate() {
                        let n: u32 = id[1..].parse().unwrap();
                        if n.is_multiple_of(2) == (tag == "even") {
                            bits.set(d);
                        }
                    }
                    bits
                });
                for (doc, score) in
                    lucene_sparse::search(&view, &field, &query, filter.as_ref()).unwrap()
                {
                    hits.push((s.ids[doc as usize].clone(), score));
                }
            }
            // TopScoreDocCollector: score descending, then index order (by segment, doc).
            let order: Vec<usize> = {
                let mut o: Vec<usize> = (0..hits.len()).collect();
                o.sort_by(|&a, &b| hits[b].1.total_cmp(&hits[a].1).then(a.cmp(&b)));
                o
            };
            let got: Vec<(String, f32)> = order.into_iter().map(|i| hits[i].clone()).collect();
            let want = &case.hits.iter().find(|(i, _)| i == index).unwrap().1;
            assert_eq!(&got, want, "{index} case {ci}");
            checked += 1;
        }
    }
    assert_eq!(checked, 120);
}
