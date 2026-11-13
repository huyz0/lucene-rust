//! Float vectors in `IvfVectorsFormat`: flushed and merged
//! segments carry an IVF index beside their raw vectors, and KNN searches
//! probe it, with deletions, filters and a forced merge that reuses the
//! largest segment's clusters, checked against an exhaustive scan.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::{IndexSortField, LuceneVersion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DIM: usize = 16;
const PER_SEGMENT: usize = 1500;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;
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

/// Points around 20 centres, so the clusters are real ones.
fn vector(id: usize) -> Vec<f32> {
    let mut centre = noise((id % 20) as u64 + 1_000_000);
    let mut jitter = noise(id as u64);
    (0..DIM).map(|_| centre() * 4.0 + jitter() * 0.5).collect()
}

/// `id` stored, the `parity` term (`even`/`odd`), the `rare` term
/// (`yes` for one document in 64) and the `centre` term (the centre the
/// vector is drawn around, `0`-`19`).
fn doc(id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
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
        (3, (id % 20).to_string().as_bytes()),
    ] {
        let mut terms = InvertedTerms::default();
        terms.push::<Vec<u8>>(term, 1, &[], &[], &[]);
        d.fields.inverted.push(InvertedField {
            field_number,
            terms,
            norm: None,
        });
    }
    (d, vec![DocumentVector::float32("v", vector(id))])
}

fn writer(dir: &FsDirectory, format: KnnVectorsFormat) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    for name in ["parity", "rare", "centre"] {
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
        VectorSimilarityFunction::Euclidean,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", format);
    w.set_ram_buffer_size_mb(256.0).unwrap();
    w.set_merge_policy(None);
    w
}

fn build(dir: &FsDirectory, format: KnnVectorsFormat) {
    let w = ConcurrentIndexWriter::new(writer(dir, format), 1).unwrap();
    for start in (0..DOCS).step_by(PER_SEGMENT) {
        let (docs, vectors): (Vec<_>, Vec<_>) = (start..start + PER_SEGMENT).map(doc).unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

fn score(a: &[f32], b: &[f32]) -> f32 {
    VectorSimilarityFunction::Euclidean.score(a, b)
}

fn exact_top(target: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut all: Vec<(usize, f32)> = (0..DOCS)
        .filter(|&id| keep(id))
        .map(|id| (id, score(target, &vector(id))))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@`k` of the reader's searches over `queries` targets against the
/// exhaustive top `k` of the documents `keep` passes; every hit's score is
/// checked against its exact score (`f32` rerank vectors).
fn recall(
    dir: &FsDirectory,
    k: usize,
    filter: Option<&Clause>,
    keep: impl Fn(usize) -> bool,
) -> f64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let queries = 30;
    let mut found = 0;
    for q in 0..queries {
        let target = vector(1_000_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            k,
            k,
            filter,
        )
        .unwrap();
        assert_eq!(hits.len(), k);
        // Doc ids are ids: one writer thread, segments in order, no merge
        // reordering.
        for h in &hits {
            let id = h.doc_id as usize;
            assert!(keep(id), "doc {id} should not match");
            assert_eq!(h.score, score(&target, &vector(id)), "doc {id} score");
        }
        let want = exact_top(&target, k, &keep);
        found += hits
            .iter()
            .filter(|h| want.contains(&(h.doc_id as usize)))
            .count();
    }
    found as f64 / (queries * k) as f64
}

fn ivf() -> KnnVectorsFormat {
    KnnVectorsFormat::ivf(128, 0).unwrap()
}

#[test]
fn flushed_segments_are_searched_through_their_ivf_index() {
    let tmp = TempDir::new("knn-ivf-flush");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    for seg in reader.segment_readers() {
        let field = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "v")
            .unwrap();
        assert!(field
            .attributes
            .iter()
            .any(|(k, v)| { k == "PerFieldKnnVectorsFormat.format" && v == "IvfVectorsFormat" }));
        let index = seg.ivf_for(field.number).unwrap().expect("an IVF index");
        assert!(index.field(field.number).unwrap().clusters() > 1);
    }
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.9, "recall {r}");
}

#[test]
fn half_float_rerank_vectors_still_return_exact_scores() {
    let tmp = TempDir::new("knn-ivf-f16");
    let dir = FsDirectory::open(&tmp);
    build(&dir, KnnVectorsFormat::ivf(128, 1).unwrap());
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.9, "recall {r}");
}

/// The quantizer name and bits each `IVF_QUANTIZERS` wire number writes
/// into the `.ivm` state.
#[test]
fn every_quantizer_writes_its_codes_and_keeps_exact_scores() {
    use lucene_codecs::per_field_knn_vectors::IVF_QUANTIZERS;
    for (number, name) in IVF_QUANTIZERS.iter().enumerate() {
        let tmp = TempDir::new(&format!("knn-ivf-{name}"));
        let dir = FsDirectory::open(&tmp);
        build(
            &dir,
            KnnVectorsFormat::ivf_quantized(128, 1, number as u8).unwrap(),
        );
        let reader = DirectoryReader::open(&dir).unwrap();
        for seg in reader.segment_readers() {
            let field = seg
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == "v")
                .unwrap();
            let index = seg.ivf_for(field.number).unwrap().expect("an IVF index");
            // A field this small gets auto's first candidate, erq2.
            let want = if name.starts_with("erq") || *name == "auto" {
                "erq"
            } else if *name == "rabitq" {
                "rot-osq"
            } else {
                "osq"
            };
            assert_eq!(index.field(field.number).unwrap().quantizer().name(), want);
        }
        let r = recall(&dir, 10, None, |_| true);
        assert!(r >= 0.9, "{name} recall {r}");
    }
}

/// A merge re-quantizes the merged vectors with the writer's quantizer,
/// whatever the sources were written with.
#[test]
fn a_merge_requantizes_with_the_writers_quantizer() {
    let tmp = TempDir::new("knn-ivf-requant");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    let erq4 = KnnVectorsFormat::ivf_quantized(128, 1, 6).unwrap();
    let mut w = writer(&dir, erq4);
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
    let index = seg.ivf_for(number).unwrap().expect("a merged IVF index");
    assert_eq!(index.field(number).unwrap().quantizer().name(), "erq");
}

#[test]
fn bad_ivf_parameters_are_refused() {
    assert!(KnnVectorsFormat::ivf(8, 0).is_err());
    assert!(KnnVectorsFormat::ivf(128, 4).is_err());
    assert!(KnnVectorsFormat::ivf_quantized(128, 0, 13).is_err());
    let tmp = TempDir::new("knn-ivf-bad");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(
        writer(
            &dir,
            KnnVectorsFormat::Ivf {
                target_cluster_size: 0,
                rerank_encoding: 0,
                quantizer: 0,
            },
        ),
        1,
    )
    .unwrap();
    let (docs, vectors): (Vec<_>, Vec<_>) = (0..10).map(doc).unzip();
    w.add_explicit_documents_with_vectors_in(None, docs, vectors)
        .unwrap();
    assert!(w.flush().is_err());
}

#[test]
fn filters_and_deletions_hold_through_the_ivf_index() {
    let tmp = TempDir::new("knn-ivf-filter");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    // Half the documents: probed through the index.
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let r = recall(&dir, 10, Some(&even), |id| id.is_multiple_of(2));
    assert!(r >= 0.9, "filtered recall {r}");
    // One in 64: too few to probe for, scanned exactly.
    let rare = Clause::Term(TermQuery::new("rare", b"yes".to_vec()));
    assert_eq!(
        recall(&dir, 10, Some(&rare), |id| id.is_multiple_of(64)),
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
    let r = recall(&dir, 10, None, |id| id.is_multiple_of(2));
    assert!(r >= 0.9, "recall after deletes {r}");
}

#[test]
fn a_small_share_filter_on_an_hnsw_field_is_scored_exactly() {
    let tmp = TempDir::new("knn-hnsw-filter-share");
    let dir = FsDirectory::open(&tmp);
    build(&dir, KnnVectorsFormat::hnsw(16, 100).unwrap());
    // One centre: a twentieth of each segment, above `perLeafTopK`, scanned
    // exactly instead of walked.
    let centre = Clause::Term(TermQuery::new("centre", b"7".to_vec()));
    assert_eq!(recall(&dir, 10, Some(&centre), |id| id % 20 == 7), 1.0);
    // Half the documents: walked.
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let r = recall(&dir, 10, Some(&even), |id| id.is_multiple_of(2));
    assert!(r >= 0.9, "filtered recall {r}");
}

/// Documents of one large segment around 40 centres, with a `centre` term:
/// a tenant per centre, each one region of the vector space.
const REGION_DOCS: usize = 16_000;
const REGION_CENTRES: usize = 40;

fn region_vector(id: usize) -> Vec<f32> {
    let mut centre = noise((id % REGION_CENTRES) as u64 + 3_000_000);
    let mut jitter = noise(id as u64 + 7_000_000);
    (0..DIM).map(|_| centre() * 4.0 + jitter() * 1.5).collect()
}

fn build_regions(dir: &FsDirectory) {
    let w = ConcurrentIndexWriter::new(writer(dir, ivf()), 1).unwrap();
    let (docs, vectors): (Vec<_>, Vec<_>) = (0..REGION_DOCS)
        .map(|id| {
            let mut d = ExplicitDocument::default();
            d.stored.push(StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            });
            let mut terms = InvertedTerms::default();
            terms.push::<Vec<u8>>(
                (id % REGION_CENTRES).to_string().as_bytes(),
                1,
                &[],
                &[],
                &[],
            );
            d.fields.inverted.push(InvertedField {
                field_number: 3,
                terms,
                norm: None,
            });
            (d, vec![DocumentVector::float32("v", region_vector(id))])
        })
        .unzip();
    w.add_explicit_documents_with_vectors_in(None, docs, vectors)
        .unwrap();
    w.flush().unwrap();
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

#[test]
fn a_filter_on_one_region_of_the_vectors_keeps_its_recall() {
    // Tenants whose documents are one region of the vector space (one
    // centre: 400 documents; eight: 3200), queried from the other regions:
    // the clusters nearest the query hold none of their documents.
    let tmp = TempDir::new("knn-ivf-region");
    let dir = FsDirectory::open(&tmp);
    build_regions(&dir);
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let k = 10;
    for centres in [vec![3usize], (0..8).map(|c| c * 5).collect()] {
        let mut b = BooleanQuery::default();
        for c in &centres {
            b.should.push(Clause::Term(TermQuery::new(
                "centre",
                c.to_string().into_bytes(),
            )));
        }
        let clause = Clause::Boolean(Box::new(b));
        let (mut found, mut total) = (0, 0);
        for q in 0..40 {
            // A query from a centre of no tenant here.
            let target = region_vector(REGION_DOCS + q * REGION_CENTRES + 1 + q % 4);
            let hits = knn_top_hits(
                &segments,
                "v",
                &KnnTarget::Float(target.clone()),
                k,
                k,
                Some(&clause),
            )
            .unwrap();
            let mut all: Vec<(usize, f32)> = (0..REGION_DOCS)
                .filter(|id| centres.contains(&(id % REGION_CENTRES)))
                .map(|id| (id, score(&target, &region_vector(id))))
                .collect();
            all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let want: Vec<usize> = all.iter().take(k).map(|p| p.0).collect();
            total += k;
            found += hits
                .iter()
                .filter(|h| want.contains(&(h.doc_id as usize)))
                .count();
        }
        let r = found as f64 / total as f64;
        // Probing the calibrated depth alone reaches 0.94 and 0.77 here.
        let floor = if centres.len() == 1 { 0.99 } else { 0.9 };
        assert!(r >= floor, "recall {r} for centres {centres:?}");
    }
}

#[test]
fn a_merge_writes_the_ivf_index_of_the_merged_vectors() {
    let tmp = TempDir::new("knn-ivf-merge");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    let mut w = writer(&dir, ivf());
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
    let index = seg.ivf_for(number).unwrap().expect("a merged IVF index");
    let field = index.field(number).unwrap();
    assert_eq!(field.count(), DOCS - DOCS.div_ceil(64));
    // Ids no longer equal doc ids after the deletes: map them back.
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let stored = |doc: i32| -> usize {
        let stored = seg.stored_document(doc).unwrap().unwrap();
        match &stored.fields[0].value {
            FieldValue::String(s) => s.parse().unwrap(),
            other => panic!("id {other:?}"),
        }
    };
    let k = 10;
    let mut found = 0;
    for q in 0..30 {
        let target = vector(2_000_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            k,
            k,
            None,
        )
        .unwrap();
        let want = exact_top(&target, k, |id| !id.is_multiple_of(64));
        for h in &hits {
            let id = stored(h.doc_id);
            assert!(!id.is_multiple_of(64), "deleted doc {id} returned");
            found += usize::from(want.contains(&id));
        }
    }
    let r = found as f64 / (30 * k) as f64;
    assert!(r >= 0.9, "recall after merge {r}");
}

/// Partitions of very different sizes, interleaved as they arrive: half the
/// documents in partition 0, a quarter in 1, the rest in ten small ones.
fn partition(id: usize) -> i64 {
    match (id / 20) % 20 {
        0..=9 => 0,
        10..=14 => 1,
        _ => 2 + ((id / 400) % 10) as i64,
    }
}

/// [`doc`] with its partition as the numeric doc value `p` and the term
/// `p<partition>` of field `rare`'s number.
fn partitioned_doc(id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::String(id.to_string()),
    });
    let mut terms = InvertedTerms::default();
    terms.push::<Vec<u8>>(format!("p{}", partition(id)).as_bytes(), 1, &[], &[], &[]);
    d.fields.inverted.push(InvertedField {
        field_number: 2,
        terms,
        norm: None,
    });
    d.fields.doc_values.push(StoredField {
        field_number: 5,
        value: FieldValue::Long(partition(id)),
    });
    (d, vec![DocumentVector::float32("v", vector(id))])
}

fn partitioned_writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let mut w = writer(dir, ivf());
    w.set_index_sort(Some(&[IndexSortField::long("p", false, None)]))
        .unwrap();
    w.register_field(FieldInfo::new("p", 0).with_doc_values(
        DocValuesType::Numeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.set_ivf_partition_clusters(Some(300));
    w
}

/// Every segment's clusters hold documents of partition 0 (sorted first:
/// documents `0..n0` of segment `s`, whose ids are `ids(s)`) or none.
fn assert_partition_clusters(dir: &FsDirectory, ids: impl Fn(usize) -> std::ops::Range<usize>) {
    let reader = DirectoryReader::open(dir).unwrap();
    for (s, seg) in reader.segment_readers().iter().enumerate() {
        let number = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "v")
            .unwrap()
            .number;
        let index = seg.ivf_for(number).unwrap().expect("an IVF index");
        let field = index.field(number).unwrap();
        let ranges = field.doc_ranges().expect("clusters per partition");
        let n0 = ids(s).filter(|&id| partition(id) == 0).count() as i32;
        assert!(ranges.iter().filter(|r| r.1 < n0).count() >= 2);
        for &(lo, hi) in ranges {
            assert!(
                hi < n0 || lo >= n0,
                "cluster of docs {lo}..={hi} spans partition 0 (..{n0})"
            );
        }
    }
}

/// Recall@10 of searches filtered to partition `p`, from queries anywhere,
/// against an exhaustive scan of its documents; hits map back by stored id.
fn partition_recall(dir: &FsDirectory, p: i64) -> f64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let clause = Clause::Term(TermQuery::new("rare", format!("p{p}").into_bytes()));
    let k = 10;
    let mut found = 0;
    for q in 0..30 {
        let target = vector(3_000_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            k,
            k,
            Some(&clause),
        )
        .unwrap();
        assert_eq!(hits.len(), k);
        let want = exact_top(&target, k, |id| partition(id) == p);
        for h in &hits {
            let (mut doc, mut seg) = (h.doc_id, 0);
            while doc >= reader.segment_readers()[seg].max_doc {
                doc -= reader.segment_readers()[seg].max_doc;
                seg += 1;
            }
            let stored = reader.segment_readers()[seg]
                .stored_document(doc)
                .unwrap()
                .unwrap();
            let id: usize = match &stored.fields[0].value {
                FieldValue::String(s) => s.parse().unwrap(),
                other => panic!("id {other:?}"),
            };
            assert_eq!(partition(id), p, "doc {id} is not in partition {p}");
            found += usize::from(want.contains(&id));
        }
    }
    found as f64 / (30 * k) as f64
}

#[test]
fn partitions_are_clustered_on_their_own_at_flush_and_merge() {
    let tmp = TempDir::new("knn-ivf-partitions");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(partitioned_writer(&dir), 1).unwrap();
    for start in (0..DOCS).step_by(PER_SEGMENT) {
        // One at a time: a list of documents is a block, which a sorted
        // index takes only with a parent field.
        for (d, v) in (start..start + PER_SEGMENT).map(partitioned_doc) {
            w.add_explicit_documents_with_vectors_in(None, vec![d], vec![v])
                .unwrap();
        }
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
    assert_eq!(
        DirectoryReader::open(&dir).unwrap().segment_readers().len(),
        SEGMENTS
    );
    assert_partition_clusters(&dir, |s| s * PER_SEGMENT..(s + 1) * PER_SEGMENT);
    for p in [0, 1, 5] {
        let r = partition_recall(&dir, p);
        assert!(r >= 0.95, "partition {p}: recall {r} after flush");
    }

    let mut w = partitioned_writer(&dir);
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    assert_eq!(
        DirectoryReader::open(&dir).unwrap().segment_readers().len(),
        1
    );
    assert_partition_clusters(&dir, |_| 0..DOCS);
    for p in [0, 1, 5] {
        let r = partition_recall(&dir, p);
        assert!(r >= 0.95, "partition {p}: recall {r} after merge");
    }
}
