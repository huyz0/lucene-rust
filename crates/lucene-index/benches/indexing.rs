//! The write path end to end: buffering and flushing documents into a
//! segment, and force-merging segments into one.
//!
//! Same document shape as `benchmarks/rust-runner/src/index_bench.rs` (a
//! stored `id`, an indexed and stored 40-word `body` over a 20 000-term
//! vocabulary), into a `ByteBuffersDirectory` so the numbers are the
//! writer's own work rather than the file system's.
//!
//! Run with `cargo bench -p lucene-index --bench indexing`.

// Benchmark setup arithmetic over small constant sizes; see
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::ByteBuffersDirectory;

const DOCS: usize = 20_000;

fn field(name: &str, number: i32, index_options: IndexOptions) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type: DocValuesType::None,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: vec![],
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Byte,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn corpus() -> Vec<Document> {
    let vocab: Vec<String> = (0..20_000).map(|i| format!("t{i}")).collect();
    let mut state: u32 = 0x9E37_79B9;
    (0..DOCS)
        .map(|i| {
            let mut body = String::with_capacity(40 * 7);
            for w in 0..40 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if w > 0 {
                    body.push(' ');
                }
                body.push_str(&vocab[state as usize % vocab.len()]);
            }
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("doc{i}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(body),
                    },
                ],
            }
        })
        .collect()
}

fn writer(dir: &ByteBuffersDirectory, body: IndexOptions) -> IndexWriter<'_> {
    let fields = vec![field("id", 0, IndexOptions::None), field("body", 1, body)];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    w.set_postings_field(Some("body")).unwrap();
    w
}

fn index(dir: &ByteBuffersDirectory, docs: Vec<Document>, body: IndexOptions, max_buffered: i32) {
    let mut w = writer(dir, body);
    if max_buffered > 0 {
        w.set_max_buffered_docs(max_buffered).unwrap();
    }
    for d in docs {
        w.add_document(d).unwrap();
    }
    w.commit().unwrap();
}

fn bench_flush(c: &mut Criterion) {
    let docs = corpus();
    let mut group = c.benchmark_group("flush");
    group.sample_size(10);
    group.throughput(Throughput::Elements(DOCS as u64));
    for (name, options) in [
        ("freqs", IndexOptions::DocsAndFreqs),
        ("positions", IndexOptions::DocsAndFreqsAndPositions),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || (ByteBuffersDirectory::new(), docs.clone()),
                |(dir, docs)| index(&dir, docs, options, 0),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn bench_force_merge(c: &mut Criterion) {
    let docs = corpus();
    let mut group = c.benchmark_group("force_merge");
    group.sample_size(10);
    group.throughput(Throughput::Elements(DOCS as u64));
    for (name, options) in [
        ("freqs", IndexOptions::DocsAndFreqs),
        ("positions", IndexOptions::DocsAndFreqsAndPositions),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let dir = ByteBuffersDirectory::new();
                    index(&dir, docs.clone(), options, (DOCS / 4) as i32);
                    dir
                },
                |dir| {
                    let mut w = writer(&dir, options);
                    w.force_merge(1).unwrap();
                    w.commit().unwrap();
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// A writer indexing `id` (for the update term) as well as `body`.
fn update_writer(dir: &ByteBuffersDirectory) -> IndexWriter<'_> {
    let fields = vec![
        field("id", 0, IndexOptions::Docs),
        field("body", 1, IndexOptions::DocsAndFreqs),
    ];
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    w.set_postings_field(Some("body")).unwrap();
    w.add_postings_field("id").unwrap();
    w.set_max_buffered_docs((DOCS / 4) as i32).unwrap();
    w
}

/// `updateDocument` by a unique `id` term over an index that already holds
/// every id: the buffered term deletes, resolved against the flushed
/// segments and the in-memory buffer at each flush and at commit.
fn bench_update(c: &mut Criterion) {
    let docs = corpus();
    let mut group = c.benchmark_group("update");
    group.sample_size(10);
    group.throughput(Throughput::Elements(DOCS as u64));
    group.bench_function("by_id", |b| {
        b.iter_batched(
            || {
                let dir = ByteBuffersDirectory::new();
                {
                    let mut w = update_writer(&dir);
                    for d in docs.clone() {
                        w.add_document(d).unwrap();
                    }
                    w.commit().unwrap();
                }
                (dir, docs.clone())
            },
            |(dir, docs)| {
                let mut w = update_writer(&dir);
                for (i, d) in docs.into_iter().enumerate() {
                    w.update_document(Term::new("id", format!("doc{i}").into_bytes()), d)
                        .unwrap();
                }
                w.commit().unwrap();
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_flush, bench_force_merge, bench_update);
criterion_main!(benches);
