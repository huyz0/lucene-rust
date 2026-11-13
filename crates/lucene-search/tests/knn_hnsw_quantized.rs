//! Float vectors in `HnswQuantizedVectorsFormat`: flushed and
//! merged segments carry quantized codes (`.vqm`/`.vqd`) beside the plain
//! HNSW files, the graph is the one `Lucene99HnswVectorsFormat` builds, and
//! KNN searches walk it on the codes and return exact scores, with filters,
//! deletions and a forced merge, checked against an exhaustive scan.

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

fn quantized(name: &str) -> KnnVectorsFormat {
    let number = IVF_QUANTIZERS.iter().position(|&n| n == name).unwrap();
    KnnVectorsFormat::hnsw_quantized(number as u8, 16, 100).unwrap()
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

#[test]
fn every_quantizer_writes_codes_and_returns_exact_scores() {
    for name in IVF_QUANTIZERS.iter().filter(|&&n| n != "auto") {
        let tmp = TempDir::new(&format!("knn-hnswq-{name}"));
        let dir = FsDirectory::open(&tmp);
        build(&dir, quantized(name));
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!(reader.segment_readers().len(), SEGMENTS);
        for seg in reader.segment_readers() {
            let field = seg
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == "v")
                .unwrap();
            assert!(field.attributes.iter().any(|(k, v)| {
                k == "PerFieldKnnVectorsFormat.format" && v == "HnswQuantizedVectorsFormat"
            }));
            let codes = seg
                .hnsw_codes_for(field.number)
                .unwrap()
                .expect("quantized codes");
            let f = codes.field(field.number).unwrap();
            assert_eq!(f.count(), PER_SEGMENT);
            assert_eq!(codes.codes(f).len(), PER_SEGMENT * f.code_len());
            codes.check_integrity().unwrap();
        }
        let r = recall(&dir, None, |_| true, |d| d as usize);
        assert!(r >= 0.9, "{name} recall {r}");
    }
}

/// The graph is the plain HNSW format's, byte for byte past the header (the
/// codec suffix differs): codes change how a search walks it, not the graph.
#[test]
fn the_graph_is_the_plain_hnsw_graph() {
    let plain_tmp = TempDir::new("knn-hnswq-plain");
    let plain = FsDirectory::open(&plain_tmp);
    build(&plain, KnnVectorsFormat::hnsw(16, 100).unwrap());
    let quant_tmp = TempDir::new("knn-hnswq-erq2");
    let quant = FsDirectory::open(&quant_tmp);
    build(&quant, quantized("erq2"));
    let body = |path: &std::path::Path, suffix: &str| -> Vec<u8> {
        let bytes = std::fs::read(path).unwrap();
        let header = codec_util::index_header_length("Lucene99HnswVectorsFormatIndex", suffix);
        bytes[header..bytes.len() - codec_util::FOOTER_LENGTH].to_vec()
    };
    let plain_vex = files_with(&plain_tmp, "vex");
    let quant_vex = files_with(&quant_tmp, "vex");
    assert_eq!(plain_vex.len(), SEGMENTS);
    assert_eq!(quant_vex.len(), SEGMENTS);
    for (p, q) in plain_vex.iter().zip(&quant_vex) {
        assert_eq!(
            body(p, "Lucene99HnswVectorsFormat_0"),
            body(q, "HnswQuantizedVectorsFormat_0")
        );
    }
    assert_eq!(files_with(&quant_tmp, "vqd").len(), SEGMENTS);
    assert!(files_with(&plain_tmp, "vqd").is_empty());
}

#[test]
fn filters_and_deletions_hold() {
    let tmp = TempDir::new("knn-hnswq-filter");
    let dir = FsDirectory::open(&tmp);
    build(&dir, quantized("rabitq"));
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
fn a_merge_encodes_the_merged_vectors() {
    let tmp = TempDir::new("knn-hnswq-merge");
    let dir = FsDirectory::open(&tmp);
    build(&dir, quantized("erq4"));
    let mut w = writer(&dir, quantized("erq4"), L2);
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
    let number = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number;
    let codes = seg.hnsw_codes_for(number).unwrap().expect("merged codes");
    let field = codes.field(number).unwrap();
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
    assert!(KnnVectorsFormat::hnsw_quantized(IVF_QUANTIZERS.len() as u8, 16, 100).is_err());
    assert!(KnnVectorsFormat::hnsw_quantized(IVF_AUTO_QUANTIZER, 16, 100).is_err());
    assert!(KnnVectorsFormat::hnsw_quantized(4, 0, 100).is_err());
    assert!(KnnVectorsFormat::hnsw_quantized(4, 16, 0).is_err());
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
        for name in ["rabitq", "erq2"] {
            let tmp = TempDir::new(&format!("knn-hnswq-{similarity:?}-{sparse}-{name}"));
            let dir = FsDirectory::open(&tmp);
            build_case(&dir, quantized(name), case);
            let r = recall_case(&dir, case, None, |_| true, |d| d as usize);
            assert!(
                r >= 0.9,
                "{similarity:?} sparse {sparse} {name}: recall {r}"
            );
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
                "{similarity:?} sparse {sparse} {name}: filtered recall {r}"
            );
        }
    }
}

/// `check_integrity` checksums the codes: a flipped byte inside `.vqd` fails
/// it, and codes whose count disagrees with the raw vectors fail a search.
#[test]
fn damaged_codes_are_reported() {
    let tmp = TempDir::new("knn-hnswq-damaged");
    let dir = FsDirectory::open(&tmp);
    build(&dir, quantized("erq2"));
    let vqd = files_with(&tmp, "vqd").remove(0);
    let mut bytes = std::fs::read(&vqd).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x40;
    std::fs::write(&vqd, &bytes).unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert!(reader
        .segment_readers()
        .iter()
        .any(|seg| seg.check_integrity().is_err()));
}

#[test]
fn codes_that_disagree_with_the_vectors_are_refused() {
    let tmp = TempDir::new("knn-hnswq-mismatch");
    let dir = FsDirectory::open(&tmp);
    build(&dir, quantized("erq2"));
    let reader = DirectoryReader::open(&dir).unwrap();
    let seg = &reader.segment_readers()[0];
    let number = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number;
    let codes = seg.hnsw_codes_for(number).unwrap().unwrap();
    let field = codes.field(number).unwrap();
    let euclidean = VectorSimilarityFunction::Euclidean;
    field.check_matches(DIM, euclidean, PER_SEGMENT).unwrap();
    assert!(field
        .check_matches(DIM + 1, euclidean, PER_SEGMENT)
        .is_err());
    assert!(field
        .check_matches(DIM, VectorSimilarityFunction::Cosine, PER_SEGMENT)
        .is_err());
    assert!(field
        .check_matches(DIM, euclidean, PER_SEGMENT - 1)
        .is_err());
}
