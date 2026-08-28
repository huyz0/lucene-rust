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

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_docs: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let out_dir = a
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/lucene-rust-index-bench".to_string());

    let vocab: Vec<String> = (0..20_000).map(|i| format!("t{i}")).collect();
    let mut docs = Vec::with_capacity(n_docs);
    let mut state: u32 = 0x9E37_79B9;
    for i in 0..n_docs {
        docs.push(Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(format!("doc{i}")),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(body(&mut state, &vocab, 40)),
                },
            ],
        });
    }

    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let dir = FsDirectory::open(&out_dir);
    let fields = vec![text_field("id", 0, false), text_field("body", 1, true)];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    })
        .expect("open writer");
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");

    let start = Instant::now();
    for d in docs {
        writer.add_document(d);
    }
    writer.commit().expect("commit");
    let elapsed = start.elapsed();

    // Nanoseconds per document, matching every other micro case's
    // lower-is-better convention -- the report script divides java by rust.
    println!(
        "index\t{:.3}\t{}",
        elapsed.as_nanos() as f64 / n_docs as f64,
        n_docs
    );
}
