//! `PerFieldKnnVectorsFormat` write routing against Java
//! (`GenPerFieldKnnVectors`): `v_small`/`v_bytes` routed to
//! `Lucene99HnswVectorsFormat(8, 40)`, `v_sq` to
//! `Lucene104HnswScalarQuantizedVectorsFormat(UNSIGNED_BYTE, 16, 100)`,
//! `v_flat` to the graph-less `Lucene104ScalarQuantizedVectorsFormat(SEVEN_BIT)`
//! and `v_default` left on the default, with
//! [`IndexWriter::set_knn_vectors_format_for_field`]. The same documents
//! written through the document API must come out as Java's: the same
//! instances with the same fields -- numbered per format name in the order a
//! flush's documents first carry a field of each (the two flushed segments
//! number the HNSW instances oppositely) and in field-number order at a
//! merge -- and the same bytes in every vector file, the segment id aside.

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_index::buffered_updates::Term;
use lucene_index::check_index;
use lucene_index::document::{
    Document, KnnByteVectorField, KnnFloatVectorField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_store::{Directory, FsDirectory};
use lucene_util::quantization::ScalarEncoding;
use lucene_util::test_support::TempDir;

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

/// `GenPerFieldKnnVectors`' codec, as an `IndexWriterConfig.setCodec` codec.
#[derive(Debug)]
struct Codec;

impl Lucene104Codec for Codec {
    fn knn_vectors_format_for_field(&self, field: &str) -> KnnVectorsFormat {
        match field {
            "v_small" | "v_bytes" => KnnVectorsFormat::hnsw(8, 40).unwrap(),
            "v_sq" => {
                KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::UnsignedByte, 16, 100)
                    .unwrap()
            }
            "v_flat" => KnnVectorsFormat::scalar_quantized(ScalarEncoding::SevenBit),
            _ => KnnVectorsFormat::default(),
        }
    }
}

fn write_with(dir: &FsDirectory, merge: bool, codec: bool) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    if codec {
        w.set_codec(Some(Arc::new(Codec))).unwrap();
    } else {
        route(&mut w);
    }
    w.set_max_full_flush_merge_wait_millis(0);
    if merge {
        let mut tmp = TieredMergePolicy::default();
        tmp.compound_file_settings_mut()
            .set_no_cfs_ratio(0.0)
            .unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(tmp)));
    }
    add_documents(&mut w, merge);
}

/// The per-field setters' routing.
fn route(w: &mut IndexWriter<'_>) {
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
}

fn add_documents(w: &mut IndexWriter<'_>, merge: bool) {
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

fn fixture(which: &str) -> String {
    format!(
        "{}/../../fixtures/data/per_field_knn_vectors/{which}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// The manifest's segment and field lines, from a directory.
fn describe(dir: &dyn Directory) -> Vec<String> {
    let sis = segment_infos::read_latest(dir).unwrap();
    let mut lines = Vec::new();
    for sci in &sis.segments {
        let si = segment_info::parse(
            &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
            &sci.segment_id,
        )
        .unwrap();
        let mut files: Vec<&String> = si.files.iter().filter(|f| f.contains("Vectors")).collect();
        files.sort();
        let mut line = format!("segment {} files", sci.segment_name);
        for f in files {
            line.push(' ');
            line.push_str(f);
        }
        lines.push(line);
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let infos = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
        for f in infos.fields.iter().filter(|f| f.vector_dimension > 0) {
            let attr = |k: &str| {
                f.attributes
                    .iter()
                    .find(|(key, _)| key == k)
                    .map_or("null", |(_, v)| v.as_str())
            };
            lines.push(format!(
                "field {} {} {} {} {}",
                sci.segment_name,
                f.name,
                f.number,
                attr("PerFieldKnnVectorsFormat.format"),
                attr("PerFieldKnnVectorsFormat.suffix")
            ));
        }
    }
    lines
}

/// `ours` with its segment id replaced by `id` and its footer re-signed.
fn with_segment_id(ours: &[u8], our_id: &[u8; 16], id: &[u8; 16]) -> Vec<u8> {
    let mut bytes = ours.to_vec();
    let at = bytes[..96]
        .windows(16)
        .position(|w| w == our_id)
        .expect("the index header carries the segment id");
    bytes[at..at + 16].copy_from_slice(id);
    let n = bytes.len();
    let crc = u64::from(crc32fast::hash(&bytes[..n - 8]));
    bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
    bytes
}

fn check(which: &str, merge: bool) {
    check_with(which, merge, false);
}

fn check_with(which: &str, merge: bool, codec: bool) {
    let tmp = TempDir::new("per-field-knn-vectors");
    let dir = FsDirectory::open(&tmp);
    write_with(&dir, merge, codec);
    let java = FsDirectory::open(fixture(which));
    let expected: Vec<String> = std::fs::read_to_string(format!("{}/manifest.txt", fixture(which)))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("segment ") || l.starts_with("field "))
        .map(str::to_string)
        .collect();
    assert_eq!(
        describe(&java),
        expected,
        "the reader agrees with Java's manifest"
    );
    assert_eq!(describe(&dir), expected, "{which}");

    let ours = segment_infos::read_latest(&dir).unwrap();
    let theirs = segment_infos::read_latest(&java).unwrap();
    let mut differing = Vec::new();
    for (a, b) in ours.segments.iter().zip(&theirs.segments) {
        let si = segment_info::parse(
            &java.open(&format!("{}.si", b.segment_name)).unwrap(),
            &b.segment_id,
        )
        .unwrap();
        for name in si.files.iter().filter(|f| f.contains("Vectors")) {
            let mine = dir.open(name).unwrap();
            let java_bytes = java.open(name).unwrap();
            if with_segment_id(&mine, &a.segment_id, &b.segment_id) != java_bytes[..] {
                differing.push(name.clone());
            }
        }
    }
    assert!(
        differing.is_empty(),
        "{which}: differ from Java's: {differing:?}"
    );
    let results = check_index::check_directory(&dir).unwrap();
    assert!(
        results.iter().all(|r| r.failures().is_empty()),
        "{results:?}"
    );
}

/// The same routing through `IndexWriterConfig.setCodec` alone.
#[test]
fn a_codec_routes_vectors_as_java_does() {
    check_with("flushed", false, true);
    check_with("merged", true, true);
}

#[test]
fn flushed_segments_route_vectors_as_java_does() {
    check("flushed", false);
}

#[test]
fn a_merge_regroups_vectors_in_field_number_order_as_java_does() {
    check("merged", true);
}
