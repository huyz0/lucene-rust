//! Writing with `SparseCodec` and reading back: flushed segments over the
//! threshold get clustered postings, segments under it an empty `.sit`, and a
//! merge re-clusters. Searching the written postings with clusters never
//! skipped finds the exact top documents.
//!
//! `scripts/check-seismic-writer.sh` runs neural-search itself over the
//! indexes this test writes and compares its hits with this crate's.

// Test-side arithmetic on generated values, not on index bytes.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::compound_reader::CompoundReader;
use lucene_index::index_writer::{
    ExplicitDocument, ExplicitFields, IndexWriter, InvertedField, InvertedTerm,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::{DirectoryReader, SegmentReader};
use lucene_sparse::search::SegmentView;
use lucene_sparse::{postings, ClusteredPostings, SeismicField, SeismicQuery, SparseVector};
use lucene_store::directory::FsDirectory;
use lucene_store::Directory;
use lucene_util::test_support::TempDir;
use std::sync::Arc;

const THRESHOLD: i32 = 50;
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 2,
};

fn attrs() -> Vec<(String, String)> {
    [
        ("sparse_vector_field", "true"),
        ("n_postings", "1000"),
        ("summary_prune_ratio", "1.0"),
        ("cluster_ratio", "0.1"),
        ("approximate_threshold", "50"),
        ("quantization_ceiling_ingest", "3.0"),
        ("quantization_ceiling_search", "16.0"),
        ("engine", "lucene"),
        ("forward_index", "shared"),
        ("clustering_batch_size", "1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// A small LCG for test vectors.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn vectors(n: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<(u32, f32)> = Vec::new();
            while v.len() < 12 {
                // A few tokens past 32768, which fold onto small ones.
                let t = if r.next().is_multiple_of(20) {
                    32768 + r.next() % 300
                } else {
                    r.next() % 300
                };
                if v.iter().all(|&(u, _)| u != t) {
                    v.push((t, 0.05 + (r.next() % 1000) as f32 / 333.0));
                }
            }
            v
        })
        .collect()
}

fn document(id: usize, vector: &[(u32, f32)], id_field: i32, emb: i32) -> ExplicitDocument {
    let mut forward = Vec::new();
    for &(t, w) in vector {
        forward.extend_from_slice(&(t as i32).to_be_bytes());
        forward.extend_from_slice(&w.to_be_bytes());
    }
    let term = |bytes: Vec<u8>, freq: i32| InvertedTerm {
        term: bytes,
        freq,
        positions: Vec::new(),
        offsets: Vec::new(),
        payloads: Vec::new(),
    };
    ExplicitDocument {
        stored: vec![StoredField {
            field_number: id_field,
            value: FieldValue::String(format!("d{id}")),
        }],
        fields: ExplicitFields {
            inverted: vec![
                InvertedField {
                    field_number: id_field,
                    terms: vec![term(format!("d{id}").into_bytes(), 1)].into(),
                    norm: None,
                },
                InvertedField {
                    field_number: emb,
                    // `FeatureField`: the token as text, the value's top bits as the frequency.
                    terms: vector
                        .iter()
                        .map(|&(t, w)| term(t.to_string().into_bytes(), (w.to_bits() >> 15) as i32))
                        .collect::<Vec<_>>()
                        .into(),
                    norm: None,
                },
            ],
            doc_values: vec![StoredField {
                field_number: emb,
                value: FieldValue::Binary(forward),
            }],
            ..ExplicitFields::default()
        },
    }
}

fn open_postings(dir: &dyn Directory, seg: &SegmentReader) -> ClusteredPostings {
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
    ClusteredPostings::open(&sit, sip, &id).unwrap()
}

/// Checks every segment of `dir`; returns how many were clustered.
fn check(dir: &FsDirectory, all: &[Vec<(u32, f32)>]) -> usize {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut clustered = 0;
    for seg in reader.segment_readers() {
        let info = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "emb")
            .unwrap();
        assert!(info
            .attributes
            .iter()
            .any(|(k, v)| k == "engine" && v == "lucene"));
        let field = SeismicField::from_field_info(info).unwrap().unwrap();
        assert_eq!(field.approximate_threshold, THRESHOLD);
        let p = open_postings(dir, seg);
        if !field.is_seismic_segment(seg.max_doc) {
            assert!(!p.has_field(field.number), "{}", seg.segment_name);
            continue;
        }
        clustered += 1;
        assert!(p.has_field(field.number));
        let id_field = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "_id")
            .unwrap()
            .number;
        let ids: Vec<usize> = (0..seg.max_doc)
            .map(|d| {
                let doc = seg.stored_document(d).unwrap().unwrap();
                match &doc
                    .fields
                    .iter()
                    .find(|f| f.field_number == id_field)
                    .unwrap()
                    .value
                {
                    FieldValue::String(s) => s[1..].parse().unwrap(),
                    v => panic!("{v:?}"),
                }
            })
            .collect();
        // Every doc is in the clusters of each of its tokens, once.
        for (d, &id) in ids.iter().enumerate() {
            for &(t, _) in &all[id] {
                let clusters = p
                    .clusters(field.number, t.to_string().as_bytes())
                    .unwrap()
                    .unwrap();
                let hits = clusters
                    .iter()
                    .flat_map(|c| &c.docs)
                    .filter(|&&x| x == d as i32)
                    .count();
                assert_eq!(hits, 1, "doc {d} token {t}");
            }
        }
        let (meta, data) = seg.doc_values_for_field(field.number).unwrap();
        let entry = meta.binary_entry(field.number).unwrap();
        let view = SegmentView {
            max_doc: seg.max_doc,
            live_docs: seg.live_docs(),
            doc_values: Some((data, entry)),
            postings: &p,
        };
        // With `SEISMIC_PRINT` set, appends each default-parameter query's hits
        // to that file, in the format `scripts/seismic/Check.java` prints.
        if let Ok(out) = std::env::var("SEISMIC_PRINT") {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out)
                .unwrap();
            for q in 0..20u32 {
                let tokens: Vec<(u32, f32)> = (0..5)
                    .map(|i| ((q * 37 + i * 61) % 300, 0.5 + i as f32))
                    .collect();
                let query = SeismicQuery::new(&tokens, 5, 10, 1.0, 1.0, &field).unwrap();
                let mut hits = lucene_sparse::search(&view, &field, &query, None).unwrap();
                hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                for (d, s) in hits {
                    writeln!(f, "{} {q} {d} {s:?}", seg.segment_name).unwrap();
                }
            }
        }
        for q in 0..20u32 {
            let tokens: Vec<(u32, f32)> = (0..5)
                .map(|i| ((q * 37 + i * 61) % 300, 0.5 + i as f32))
                .collect();
            // Every token walked, and whole summaries with a huge heap factor
            // skip no cluster holding a scoring doc: an exact search.
            let query = SeismicQuery::new(&tokens, 5, 10, 1e30, 1.0, &field).unwrap();
            let mut got: Vec<f32> = lucene_sparse::search(&view, &field, &query, None)
                .unwrap()
                .into_iter()
                .map(|(_, s)| s)
                .collect();
            got.sort_by(|a, b| b.total_cmp(a));
            let mut want: Vec<f32> = ids
                .iter()
                .filter(|&&id| {
                    all[id]
                        .iter()
                        .any(|&(t, _)| tokens.iter().any(|&(u, _)| u == t))
                })
                .map(|&id| {
                    let v = SparseVector::quantized(&all[id], field.ceiling_ingest);
                    v.dot(&query.dense) as f32 * query.score_scale
                })
                .collect();
            want.sort_by(|a, b| b.total_cmp(a));
            want.truncate(10);
            assert_eq!(got, want, "{} query {q}", seg.segment_name);
        }
    }
    clustered
}

#[test]
fn written_segments_are_clustered_and_searchable() {
    lucene_sparse::register_codec();
    let tmp = TempDir::new("seismic-writer");
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), lucene_sparse::CODEC_NAME, VERSION).unwrap();
    w.set_codec(Some(Arc::new(lucene_sparse::SparseCodec)))
        .unwrap();
    w.enable_explicit_documents().unwrap();
    let id_field = w
        .register_field(
            FieldInfo::new("_id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    let emb = w
        .register_field(
            FieldInfo::new("emb", 0)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_omit_norms(true)
                .with_doc_values(DocValuesType::Binary, DocValuesSkipIndexType::None, -1)
                .with_attributes(attrs()),
        )
        .unwrap();
    let all = vectors(260, 7);
    // Three segments over the threshold (one compound), one under it.
    for (n, (start, end)) in [(0, 80), (80, 160), (160, 240), (240, 260)]
        .into_iter()
        .enumerate()
    {
        w.set_use_compound_file(n == 1);
        let docs = (start..end)
            .map(|i| document(i, &all[i], id_field, emb))
            .collect();
        w.add_explicit_documents(docs).unwrap();
        w.commit().unwrap();
    }
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 4);
    assert!(infos
        .segments
        .iter()
        .all(|s| s.codec_name == lucene_sparse::CODEC_NAME));
    assert_eq!(check(&dir, &all), 3);
    copy_out(tmp.path(), "flushed");

    w.force_merge(1).unwrap();
    w.commit().unwrap();
    assert_eq!(check(&dir, &all), 1);
    copy_out(tmp.path(), "merged");
}

/// With `SEISMIC_WRITER_OUT` set, copies the index to `$SEISMIC_WRITER_OUT/{name}`
/// for `scripts/check-seismic-writer.sh`, which reads it with neural-search.
fn copy_out(from: &std::path::Path, name: &str) {
    let Ok(out) = std::env::var("SEISMIC_WRITER_OUT") else {
        return;
    };
    let to = std::path::Path::new(&out).join(name);
    std::fs::create_dir_all(&to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.file_name() != "write.lock" {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}
