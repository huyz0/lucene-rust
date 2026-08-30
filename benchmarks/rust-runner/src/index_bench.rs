//! Indexing throughput: the write path, which no benchmark in this project had
//! ever compared against Lucene.
//!
//! The Java counterpart is `benchmarks/micro/java/IndexMicro.java`. Both index
//! the same synthetic documents -- a stored string field plus an indexed text
//! field -- into a fresh directory and report documents per second.
//!
//! Deliberately a narrow document shape. This port's `IndexWriter` has no
//! points write path at flush time and no term-vector write path, so a document
//! matching `GenCorpus`'s would not be expressible here at all; that gap is
//! recorded in `docs/parity.md` rather than papered over by benchmarking a
//! shape only one engine can write.

use std::time::Instant;

/// Linux `VmHWM` -- the process's peak resident set size, in kilobytes. The
/// kernel's own high-water mark, so it survives the allocator returning pages
/// between the sample points.
fn peak_rss_kb() -> u64 {
    read_status_kb("VmHWM:")
}

/// Current resident set size (`VmRSS`), in kilobytes.
fn rss_kb() -> u64 {
    read_status_kb("VmRSS:")
}

fn read_status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::{DocumentVector, IndexWriter};
use lucene_index::segment_info::{IndexSortField, LuceneVersion, SortMissingValue};
use lucene_store::FsDirectory;

fn text_field(name: &str, number: i32, indexed: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: if indexed {
            IndexOptions::DocsAndFreqs
        } else {
            IndexOptions::None
        },
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

/// Same deterministic text on both sides: a fixed vocabulary sampled by a
/// xorshift, so neither engine gets a different corpus.
fn body(state: &mut u32, vocab: &[String], words: usize) -> String {
    let mut out = String::with_capacity(words * 6);
    for i in 0..words {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&vocab[(*state as usize) % vocab.len()]);
    }
    out
}

/// A deterministic unit-ish vector, so the graph built over the corpus has
/// real neighbourhood structure rather than being all-equidistant noise.
fn vector(state: &mut u32, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            (*state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_docs: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let out_dir = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/lucene-rust-index-bench".to_string());

    // `LUCENE_RUST_VECTOR_DIM`, when set, gives every document a FLOAT32
    // `KnnFloatVectorField` of that dimension as well -- the A/B that measures
    // what indexing vectors costs per document (`docs/sweep/m2/c10-vectors-wiring.md`).
    let vector_dim: usize = std::env::var("LUCENE_RUST_VECTOR_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // `LUCENE_RUST_INDEX_SORT`, when set, configures an index sort over one
    // (`rank`) or two (`rank,tie`) NUMERIC doc-values fields -- the A/B that
    // measures what sorting a flush costs per document
    // (`docs/sweep/m2/c17-index-sort.md`). Values: "", "1", "2".
    let sort_tiers: usize = std::env::var("LUCENE_RUST_INDEX_SORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // The honest control for the sorted arms: write the same two NUMERIC
    // doc-values columns and do *not* sort, so the A/B delta is the sort
    // itself rather than the columns it is defined over.
    let dv_only = std::env::var("LUCENE_RUST_DOC_VALUES_ONLY").is_ok();
    let columns = sort_tiers > 0 || dv_only;

    // `LUCENE_RUST_INDEX_OPTIONS` raises the `body` field's `IndexOptions`
    // rung -- the A/B that measures what indexing positions, then offsets,
    // then payloads costs per document (`docs/sweep/m2/c23-positions-writer.md`).
    // Unset (or "freqs") is the baseline every other arm here is measured
    // against; "positions", "offsets" and "payloads" are cumulative.
    let index_options_arm =
        std::env::var("LUCENE_RUST_INDEX_OPTIONS").unwrap_or_else(|_| "freqs".to_string());
    let body_index_options = match index_options_arm.as_str() {
        "freqs" => IndexOptions::DocsAndFreqs,
        "positions" => IndexOptions::DocsAndFreqsAndPositions,
        "offsets" | "payloads" | "payloads-empty" => {
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        }
        other => panic!("LUCENE_RUST_INDEX_OPTIONS: unknown arm {other:?}"),
    };
    let body_payloads = index_options_arm.starts_with("payloads");
    // The "payloads-empty" arm declares payloads but attaches none, so the
    // A/B against "payloads" separates the cost of the payload *machinery*
    // (the per-occurrence length stream in `.pay`, the `payloadByteUpto`
    // prefix the skip records are sampled from, and the per-occurrence slot in
    // the inverted index) from the cost of the payload *bytes*.
    let empty_payloads = index_options_arm == "payloads-empty";

    let vocab: Vec<String> = (0..20_000).map(|i| format!("t{i}")).collect();
    let mut docs = Vec::with_capacity(n_docs);
    let mut vectors: Vec<Vec<DocumentVector>> = Vec::with_capacity(n_docs);
    let mut vstate: u32 = 0x1234_5678;
    let mut state: u32 = 0x9E37_79B9;
    for i in 0..n_docs {
        vectors.push(if vector_dim > 0 {
            vec![DocumentVector::float32(
                "vec",
                vector(&mut vstate, vector_dim),
            )]
        } else {
            Vec::new()
        });
        let mut doc_fields = vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{i}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body(&mut state, &vocab, 40)),
            },
        ];
        if columns {
            // A key with many duplicates, so the second tier does real work,
            // and in an order unrelated to insertion order so the permutation
            // is a full shuffle rather than a near-identity.
            doc_fields.push(StoredField {
                field_number: 3,
                value: FieldValue::Long(((i * 7919) % 1000) as i64),
            });
            doc_fields.push(StoredField {
                field_number: 4,
                value: FieldValue::Long(((i * 104_729) % n_docs) as i64),
            });
        }
        docs.push(Document {
            fields: doc_fields,
        });
    }

    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let dir = FsDirectory::open(&out_dir);
    let mut fields = vec![
        text_field("id", 0, false),
        FieldInfo {
            index_options: body_index_options,
            store_payloads: body_payloads,
            ..text_field("body", 1, true)
        },
    ];
    if vector_dim > 0 {
        fields.push(FieldInfo {
            vector_dimension: vector_dim as i32,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
            ..text_field("vec", 2, false)
        });
    }
    if columns {
        fields.push(FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..text_field("rank", 3, false)
        });
        fields.push(FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..text_field("tie", 4, false)
        });
    }
    let mut writer = IndexWriter::open(
        &dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");
    // Java's IndexMicro writes norms (Lucene does so for any indexed field
    // that does not omit them), so measuring against it means writing them
    // here too -- otherwise the two sides are not doing the same work.
    writer
        .set_norms_field(Some("body"))
        .expect("set norms field");
    if body_payloads {
        // A four-byte payload on every token: enough to make the `.pay`
        // payload-byte run real work rather than an all-zero length stream,
        // and small enough that the arm measures the *machinery* rather than
        // the cost of moving arbitrary bytes.
        writer
            .set_payload_source(Some(Box::new(move |ctx| {
                if empty_payloads {
                    None
                } else {
                    Some((ctx.position as u32).to_le_bytes().to_vec())
                }
            })))
            .expect("set payload source");
    }
    if vector_dim > 0 {
        writer
            .set_vector_field(Some("vec"))
            .expect("set vector field");
    }
    if columns {
        writer
            .set_doc_values_field(Some("rank"))
            .expect("set doc values field");
        writer
            .add_doc_values_field("tie")
            .expect("add doc values field");
    }
    if sort_tiers > 0 {
        let mut sort = vec![IndexSortField {
            field: "rank".to_string(),
            reverse: true,
            missing: SortMissingValue::Last,
        }];
        if sort_tiers > 1 {
            sort.push(IndexSortField {
                field: "tie".to_string(),
                reverse: false,
                missing: SortMissingValue::First,
            });
        }
        writer.set_index_sort(Some(&sort)).expect("set index sort");
    }

    // `LUCENE_RUST_RAM_BUFFER_MB` drives `IndexWriter::set_ram_buffer_size_mb`.
    // The default matches Lucene's own 16 MB; setting it very high reproduces
    // the pre-flush-trigger behaviour (buffer everything until `commit()`),
    // which is what the memory A/B in `docs/sweep/m2/c3-writer-lifecycle.md`
    // measures against.
    if let Ok(mb) = std::env::var("LUCENE_RUST_RAM_BUFFER_MB") {
        let mb: f64 = mb
            .parse()
            .expect("LUCENE_RUST_RAM_BUFFER_MB must be a number");
        writer
            .set_ram_buffer_size_mb(mb)
            .expect("set ram buffer size");
    }

    // Sampled after the corpus is built and before any indexing, so the
    // writer's own peak is `peak_rss_kb() - baseline_rss_kb` rather than
    // whatever the pre-built document vector happens to cost.
    let baseline_rss_kb = rss_kb();
    let start = Instant::now();
    for (d, v) in docs.into_iter().zip(vectors) {
        if vector_dim > 0 {
            writer.add_document_with_vectors(d, v).unwrap();
        } else {
            writer.add_document(d).unwrap();
        }
    }
    writer.commit().expect("commit");
    let elapsed = start.elapsed();

    // Nanoseconds per document, matching every other micro case's
    // lower-is-better convention -- the report script divides java by rust.
    println!(
        "index[{index_options_arm}]\t{:.3}\t{}",
        elapsed.as_nanos() as f64 / n_docs as f64,
        n_docs
    );
    // Memory, on the same line format: baseline (corpus only), peak, and the
    // writer's own contribution.
    let peak = peak_rss_kb();
    println!(
        "index-mem\tbaseline_rss_kb={baseline_rss_kb}\tpeak_rss_kb={peak}\twriter_peak_kb={}\tsegments={}",
        peak.saturating_sub(baseline_rss_kb),
        writer.segment_infos().segments.len()
    );
}
