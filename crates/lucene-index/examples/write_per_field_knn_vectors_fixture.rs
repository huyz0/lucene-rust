//! Writes two indexes whose vector fields are split across
//! `PerFieldKnnVectorsFormat` instances, for `VerifyPerFieldKnnVectors`:
//! `GenPerFieldKnnVectors`' documents and routing -- `v_small`/`v_bytes` on
//! `Lucene99HnswVectorsFormat(8, 40)`, `v_sq` on
//! `Lucene104HnswScalarQuantizedVectorsFormat(UNSIGNED_BYTE, 16, 100)`,
//! `v_flat` on `Lucene104ScalarQuantizedVectorsFormat(SEVEN_BIT)`, `v_default`
//! on the default -- through [`IndexWriter::set_knn_vectors_format_for_field`].
//!
//! `<out>/flushed` holds two flushed segments; `<out>/merged` the same with
//! `d5` and `d302` deleted, force-merged into one.
//!
//! Usage: `write_per_field_knn_vectors_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, KnnByteVectorField, KnnFloatVectorField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::quantization::ScalarEncoding;

const PER_SEGMENT: usize = 300;
const DIM: usize = 16;

/// `GenPerFieldKnnVectors.value`.
fn value(i: usize, k: i64) -> i64 {
    let x = (i as i64 + 1) * 2_654_435_761 + k * 40_503;
    (x ^ ((x as u64) >> 13) as i64) % 100_000
}

fn floats(i: usize, salt: i64) -> Vec<f32> {
    (0..DIM as i64)
        .map(|j| (value(i, salt * 100 + j) % 2001 - 1000) as f32 / 100.0)
        .collect()
}

fn bytes(i: usize) -> Vec<u8> {
    (0..DIM as i64)
        .map(|j| (value(i, 500 + j) % 255 - 127) as i8 as u8)
        .collect()
}

/// `GenPerFieldKnnVectors.doc`.
fn doc(i: usize) -> Document {
    let float = |name: &str, v: Vec<f32>, sim| KnnFloatVectorField::new(name, v, sim).unwrap();
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::No));
    if !i.is_multiple_of(7) {
        d.add(float(
            "v_sq",
            floats(i, 1),
            VectorSimilarityFunction::Euclidean,
        ));
    }
    if i != 0 {
        d.add(float(
            "v_default",
            floats(i, 2),
            VectorSimilarityFunction::Cosine,
        ));
    }
    if i != PER_SEGMENT {
        d.add(float(
            "v_small",
            floats(i, 3),
            VectorSimilarityFunction::Euclidean,
        ));
    }
    d.add(
        KnnByteVectorField::new("v_bytes", bytes(i), VectorSimilarityFunction::Euclidean).unwrap(),
    );
    d.add(float(
        "v_flat",
        floats(i, 4),
        VectorSimilarityFunction::MaximumInnerProduct,
    ));
    d
}

fn write(dir: &FsDirectory, merge: bool) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    let small = KnnVectorsFormat::hnsw(8, 40).unwrap();
    w.set_knn_vectors_format_for_field("v_small", small);
    w.set_knn_vectors_format_for_field("v_bytes", small);
    w.set_knn_vectors_format_for_field(
        "v_sq",
        KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::UnsignedByte, 16, 100).unwrap(),
    );
    w.set_knn_vectors_format_for_field(
        "v_flat",
        KnnVectorsFormat::scalar_quantized(ScalarEncoding::SevenBit),
    );
    w.set_max_full_flush_merge_wait_millis(0);
    if merge {
        let mut tmp = TieredMergePolicy::default();
        tmp.compound_file_settings_mut()
            .set_no_cfs_ratio(0.0)
            .unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(tmp)));
    }
    for seg in 0..2 {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    if merge {
        w.delete_documents_by_term(&[Term::new("id", "d5"), Term::new("id", "d302")])
            .unwrap();
        w.commit().unwrap();
        w.force_merge(1).unwrap();
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_per_field_knn_vectors_fixture <output-dir>");
    for (sub, merge) in [("flushed", false), ("merged", true)] {
        let path = std::path::Path::new(&out).join(sub);
        std::fs::create_dir_all(&path).unwrap();
        write(&FsDirectory::open(&path), merge);
    }
}
