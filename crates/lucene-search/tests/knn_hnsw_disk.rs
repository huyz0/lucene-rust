//! Float vectors in `DiskHnswVectorsFormat`: flushed and merged segments
//! carry the disk layout (`.dhm`/`.dhc`/`.dhn`) beside the plain HNSW files,
//! the graph is the one `Lucene99HnswVectorsFormat` builds, and KNN searches
//! walk the node records and return exact scores, with filters, deletions
//! and a forced merge, checked against an exhaustive scan.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::{KnnVectorsFormat, IVF_AUTO_QUANTIZER, IVF_QUANTIZERS};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_store::codec_util;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DIM: usize = 24;
const PER_SEGMENT: usize = 1500;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;
const K: usize = 10;
const CANDIDATES: usize = 60;
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn noise(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    move || {
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// The field's similarity, and whether some documents have no vector.
#[derive(Clone, Copy)]
struct Case {
    similarity: VectorSimilarityFunction,
    /// Every fifth document has no vector, so ordinals and doc ids differ.
    sparse: bool,
}

const L2: Case = Case {
    similarity: VectorSimilarityFunction::Euclidean,
    sparse: false,
};

impl Case {
    fn has_vector(self, id: usize) -> bool {
        !(self.sparse && id % 5 == 3)
    }

    /// [`vector`], unit length for dot product (which requires it).
    fn vector(self, id: usize) -> Vec<f32> {
        let mut v = vector(id);
        if self.similarity == VectorSimilarityFunction::DotProduct {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    fn score(self, a: &[f32], b: &[f32]) -> f32 {
        self.similarity.score(a, b)
    }
}

/// Points around 20 centres.
fn vector(id: usize) -> Vec<f32> {
    let mut centre = noise((id % 20) as u64 + 1_000_000);
    let mut jitter = noise(id as u64);
    (0..DIM).map(|_| centre() * 4.0 + jitter()).collect()
}

/// `id` stored, the `parity` term (`even`/`odd`) and the `rare` term
/// (`yes` for one document in 64).
fn doc(case: Case, id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::String(id.to_string()),
    });
    for (field_number, term) in [
        (
            1,
            if id.is_multiple_of(2) {
                &b"even"[..]
            } else {
                b"odd"
            },
        ),
        (
            2,
            if id.is_multiple_of(64) {
                &b"yes"[..]
            } else {
                b"no"
            },
        ),
    ] {
        let mut terms = InvertedTerms::default();
        terms.push::<Vec<u8>>(term, 1, &[], &[], &[]);
        d.fields.inverted.push(InvertedField {
            field_number,
            terms,
            norm: None,
        });
    }
    let vectors = if case.has_vector(id) {
        vec![DocumentVector::float32("v", case.vector(id))]
    } else {
        Vec::new()
    };
    (d, vectors)
}

fn writer(dir: &FsDirectory, format: KnnVectorsFormat, case: Case) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    for name in ["parity", "rare"] {
        w.register_field(
            FieldInfo::new(name, 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    }
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        DIM as i32,
        VectorEncoding::Float32,
        case.similarity,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", format);
    w.set_ram_buffer_size_mb(256.0).unwrap();
    w.set_merge_policy(None);
    w
}

fn build(dir: &FsDirectory, format: KnnVectorsFormat) {
    build_case(dir, format, L2);
}

fn build_case(dir: &FsDirectory, format: KnnVectorsFormat, case: Case) {
    let w = ConcurrentIndexWriter::new(writer(dir, format, case), 1).unwrap();
    for start in (0..DOCS).step_by(PER_SEGMENT) {
        let (docs, vectors): (Vec<_>, Vec<_>) =
            (start..start + PER_SEGMENT).map(|id| doc(case, id)).unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

fn exact_top(case: Case, target: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut all: Vec<(usize, f32)> = (0..DOCS)
        .filter(|&id| keep(id) && case.has_vector(id))
        .map(|id| (id, case.score(target, &case.vector(id))))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@[`K`] of searches for [`CANDIDATES`] candidates against the
/// exhaustive top [`K`] of the documents `keep` passes; every hit's score is
/// checked to be its exact one. `id_of` maps a global doc id to its id.
fn recall(
    dir: &FsDirectory,
    filter: Option<&Clause>,
    keep: impl Fn(usize) -> bool,
    id_of: impl Fn(i32) -> usize,
) -> f64 {
    recall_case(dir, L2, filter, keep, id_of)
}

fn recall_case(
    dir: &FsDirectory,
    case: Case,
    filter: Option<&Clause>,
    keep: impl Fn(usize) -> bool,
    id_of: impl Fn(i32) -> usize,
) -> f64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let queries = 30;
    let mut found = 0;
    for q in 0..queries {
        let target = case.vector(1_000_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            K,
            CANDIDATES,
            filter,
        )
        .unwrap();
        assert_eq!(hits.len(), K);
        for h in &hits {
            let id = id_of(h.doc_id);
            assert!(keep(id) && case.has_vector(id), "doc {id} should not match");
            assert_eq!(
                h.score,
                case.score(&target, &case.vector(id)),
                "doc {id} score"
            );
        }
        let want = exact_top(case, &target, K, &keep);
        found += hits
            .iter()
            .filter(|h| want.contains(&id_of(h.doc_id)))
            .count();
    }
    found as f64 / (queries * K) as f64
}

fn disk(name: &str) -> KnnVectorsFormat {
    let number = IVF_QUANTIZERS.iter().position(|&n| n == name).unwrap();
    KnnVectorsFormat::hnsw_disk(number as u8, 16, 100).unwrap()
}

/// Files of the directory with `ext`, sorted by name.
fn files_with(dir: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == ext))
        .collect();
    out.sort();
    out
}

fn field_number(seg: &lucene_search::directory_reader::SegmentReader) -> i32 {
    seg.field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number
}

#[test]
fn quantizers_write_the_disk_layout_and_return_exact_scores() {
    for name in ["bbq", "osq4", "erq2", "rabitq"] {
        let tmp = TempDir::new(&format!("knn-hnswd-{name}"));
        let dir = FsDirectory::open(&tmp);
        build(&dir, disk(name));
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), SEGMENTS);
        for seg in reader.segment_readers() {
            let number = field_number(seg);
            let field = seg
                .field_infos()
                .fields
                .iter()
                .find(|f| f.number == number)
                .unwrap();
            assert!(field.attributes.iter().any(|(k, v)| {
                k == "PerFieldKnnVectorsFormat.format" && v == "DiskHnswVectorsFormat"
            }));
            let layout = seg.disk_hnsw_for(number).unwrap().expect("disk layout");
            assert_eq!(layout.field(number).unwrap().count(), PER_SEGMENT);
            layout.check_integrity().unwrap();
            seg.check_integrity().unwrap();
        }
        assert_eq!(files_with(&tmp, "dhn").len(), SEGMENTS);
        let r = recall(&dir, None, |_| true, |d| d as usize);
        assert!(r >= 0.9, "{name} recall {r}");
    }
}

/// The graph is the plain HNSW format's, byte for byte past the header (the
/// codec suffix differs): the disk layout changes how a search walks it, not
/// the graph.
#[test]
fn the_graph_is_the_plain_hnsw_graph() {
    let plain_tmp = TempDir::new("knn-hnswd-plain");
    let plain = FsDirectory::open(&plain_tmp);
    build(&plain, KnnVectorsFormat::hnsw(16, 100).unwrap());
    let disk_tmp = TempDir::new("knn-hnswd-bbq");
    let on_disk = FsDirectory::open(&disk_tmp);
    build(&on_disk, disk("bbq"));
    let body = |path: &std::path::Path, suffix: &str| -> Vec<u8> {
        let bytes = std::fs::read(path).unwrap();
        let header = codec_util::index_header_length("Lucene99HnswVectorsFormatIndex", suffix);
        bytes[header..bytes.len() - codec_util::FOOTER_LENGTH].to_vec()
    };
    let plain_vex = files_with(&plain_tmp, "vex");
    let disk_vex = files_with(&disk_tmp, "vex");
    assert_eq!(plain_vex.len(), SEGMENTS);
    assert_eq!(disk_vex.len(), SEGMENTS);
    for (p, q) in plain_vex.iter().zip(&disk_vex) {
        assert_eq!(
            body(p, "Lucene99HnswVectorsFormat_0"),
            body(q, "DiskHnswVectorsFormat_0")
        );
    }
    assert!(files_with(&plain_tmp, "dhn").is_empty());
}

#[test]
fn filters_and_deletions_hold() {
    let tmp = TempDir::new("knn-hnswd-filter");
    let dir = FsDirectory::open(&tmp);
    build(&dir, disk("bbq"));
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let r = recall(&dir, Some(&even), |id| id.is_multiple_of(2), |d| d as usize);
    assert!(r >= 0.9, "filtered recall {r}");
    // One in 64: below the leaf's top k, scanned exactly on the raw vectors.
    let rare = Clause::Term(TermQuery::new("rare", b"yes".to_vec()));
    assert_eq!(
        recall(
            &dir,
            Some(&rare),
            |id| id.is_multiple_of(64),
            |d| d as usize
        ),
        1.0
    );
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.delete_documents_by_term(&[Term {
        field: "parity".into(),
        bytes: b"odd".to_vec(),
    }])
    .unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let r = recall(&dir, None, |id| id.is_multiple_of(2), |d| d as usize);
    assert!(r >= 0.9, "recall after deletes {r}");
}

#[test]
fn a_merge_writes_the_layout_over_the_merged_graph() {
    let tmp = TempDir::new("knn-hnswd-merge");
    let dir = FsDirectory::open(&tmp);
    build(&dir, disk("erq4"));
    let mut w = writer(&dir, disk("erq4"), L2);
    w.delete_documents_by_term(&[Term {
        field: "rare".into(),
        bytes: b"yes".to_vec(),
    }])
    .unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    let number = field_number(seg);
    let layout = seg.disk_hnsw_for(number).unwrap().expect("merged layout");
    let field = layout.field(number).unwrap();
    assert_eq!(field.count(), DOCS - DOCS.div_ceil(64));
    assert_eq!(field.quantizer().name(), "erq");
    let ids: Vec<usize> = (0..seg.max_doc)
        .map(
            |doc| match &seg.stored_document(doc).unwrap().unwrap().fields[0].value {
                FieldValue::String(s) => s.parse().unwrap(),
                other => panic!("id {other:?}"),
            },
        )
        .collect();
    let r = recall(&dir, None, |id| !id.is_multiple_of(64), |d| ids[d as usize]);
    assert!(r >= 0.9, "merged recall {r}");
}

#[test]
fn bad_parameters_are_refused() {
    assert!(KnnVectorsFormat::hnsw_disk(IVF_QUANTIZERS.len() as u8, 16, 100).is_err());
    assert!(KnnVectorsFormat::hnsw_disk(IVF_AUTO_QUANTIZER, 16, 100).is_err());
    assert!(KnnVectorsFormat::hnsw_disk(0, 0, 100).is_err());
    assert!(KnnVectorsFormat::hnsw_disk(0, 16, 0).is_err());
}

#[test]
fn every_similarity_and_sparse_fields_return_exact_scores() {
    for (similarity, sparse) in [
        (VectorSimilarityFunction::Cosine, false),
        (VectorSimilarityFunction::DotProduct, false),
        (VectorSimilarityFunction::MaximumInnerProduct, false),
        (VectorSimilarityFunction::Euclidean, true),
        (VectorSimilarityFunction::Cosine, true),
    ] {
        let case = Case { similarity, sparse };
        let tmp = TempDir::new(&format!("knn-hnswd-{similarity:?}-{sparse}"));
        let dir = FsDirectory::open(&tmp);
        build_case(&dir, disk("bbq"), case);
        let r = recall_case(&dir, case, None, |_| true, |d| d as usize);
        assert!(r >= 0.9, "{similarity:?} sparse {sparse}: recall {r}");
        let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
        let r = recall_case(
            &dir,
            case,
            Some(&even),
            |id| id.is_multiple_of(2),
            |d| d as usize,
        );
        assert!(
            r >= 0.9,
            "{similarity:?} sparse {sparse}: filtered recall {r}"
        );
    }
}

/// `check_integrity` checksums the layout: a flipped byte inside `.dhn`
/// fails it.
#[test]
fn damaged_nodes_are_reported() {
    let tmp = TempDir::new("knn-hnswd-damaged");
    let dir = FsDirectory::open(&tmp);
    build(&dir, disk("bbq"));
    let dhn = files_with(&tmp, "dhn").remove(0);
    let mut bytes = std::fs::read(&dhn).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x40;
    std::fs::write(&dhn, &bytes).unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert!(reader
        .segment_readers()
        .iter()
        .any(|seg| seg.check_integrity().is_err()));
}
