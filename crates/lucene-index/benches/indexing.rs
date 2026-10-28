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

/// Text fields carrying norms (`f0`..`f3`, `f3` on every other document
/// only) and NUMERIC doc-values fields (`n0`..`n3`, `n3` missing on every
/// tenth document) next to `body`: the per-field merge stages -- norms,
/// doc values -- rather than the one big postings field.
const FIELDS_TEXT: usize = 4;
const FIELDS_NUMERIC: usize = 4;

fn fields_corpus() -> Vec<Document> {
    let mut docs = corpus();
    for (i, d) in docs.iter_mut().enumerate() {
        for f in 0..FIELDS_TEXT {
            if f == FIELDS_TEXT - 1 && i % 2 == 1 {
                continue;
            }
            d.fields.push(StoredField {
                field_number: (2 + f) as i32,
                value: FieldValue::String(format!("w{} w{} x{}", i % 97, i % 13, f)),
            });
        }
        for f in 0..FIELDS_NUMERIC {
            if f == FIELDS_NUMERIC - 1 && i % 10 == 0 {
                continue;
            }
            d.fields.push(StoredField {
                field_number: (2 + FIELDS_TEXT + f) as i32,
                value: FieldValue::Long((i as i64).wrapping_mul(2_654_435_761) >> (f * 3)),
            });
        }
    }
    docs
}

fn fields_writer(dir: &ByteBuffersDirectory) -> IndexWriter<'_> {
    let mut fields = vec![
        field("id", 0, IndexOptions::None),
        field("body", 1, IndexOptions::DocsAndFreqs),
    ];
    for f in 0..FIELDS_TEXT {
        fields.push(field(
            &format!("f{f}"),
            (2 + f) as i32,
            IndexOptions::DocsAndFreqs,
        ));
    }
    for f in 0..FIELDS_NUMERIC {
        let mut info = field(
            &format!("n{f}"),
            (2 + FIELDS_TEXT + f) as i32,
            IndexOptions::None,
        );
        info.doc_values_type = DocValuesType::Numeric;
        fields.push(info);
    }
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, fields, "Lucene104", version).unwrap();
    w.set_postings_field(Some("body")).unwrap();
    for f in 0..FIELDS_TEXT {
        w.add_postings_field(&format!("f{f}")).unwrap();
    }
    for f in 0..FIELDS_NUMERIC {
        w.add_doc_values_field(&format!("n{f}")).unwrap();
    }
    w
}

/// `force_merge(1)` over four flushed segments of [`fields_corpus`].
fn bench_force_merge_fields(c: &mut Criterion) {
    let docs = fields_corpus();
    let mut group = c.benchmark_group("force_merge");
    group.sample_size(10);
    group.throughput(Throughput::Elements(DOCS as u64));
    group.bench_function("fields", |b| {
        b.iter_batched(
            || {
                let dir = ByteBuffersDirectory::new();
                {
                    let mut w = fields_writer(&dir);
                    w.set_max_buffered_docs((DOCS / 4) as i32).unwrap();
                    for d in docs.clone() {
                        w.add_document(d).unwrap();
                    }
                    w.commit().unwrap();
                }
                dir
            },
            |dir| {
                let mut w = fields_writer(&dir);
                w.force_merge(1).unwrap();
                w.commit().unwrap();
            },
            BatchSize::LargeInput,
        )
    });
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

/// Deletes by `id` against an index of 32 committed segments, in rounds of
/// 64 deletes each followed by a flush: every round resolves its terms
/// against every segment and writes their `.liv` files.
fn bench_delete(c: &mut Criterion) {
    let docs = corpus();
    let mut group = c.benchmark_group("delete");
    group.sample_size(10);
    group.bench_function("32_segments", |b| {
        b.iter_batched(
            || {
                let dir = ByteBuffersDirectory::new();
                {
                    let mut w = update_writer(&dir);
                    w.set_max_buffered_docs((DOCS / 32) as i32).unwrap();
                    for d in docs.clone() {
                        w.add_document(d).unwrap();
                    }
                    w.commit().unwrap();
                }
                dir
            },
            |dir| {
                let mut w = update_writer(&dir);
                for round in 0..16 {
                    for k in 0..64 {
                        let i = (k * 311 + round * 17) % DOCS;
                        w.delete_documents_by_term(&[Term::new(
                            "id",
                            format!("doc{i}").into_bytes(),
                        )])
                        .unwrap();
                    }
                    w.flush().unwrap();
                }
                w.commit().unwrap();
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_flush,
    bench_force_merge,
    bench_force_merge_fields,
    bench_update,
    bench_delete
);
criterion_main!(benches);
