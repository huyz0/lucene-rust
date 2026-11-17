//! Writes a multi-segment index with sparse doc values of every type in the
//! index format of one Lucene release, for `scripts/verify-index-format.sh`
//! to open with that release's own jars.
//!
//! The index: 9 000 documents flushed every 2 500 (so several segments, with
//! sparse presence in each), a delete batch and a force merge, then two more
//! segments. Doc values, all sparse:
//!
//! - `num` NUMERIC: present unless `n % 3 == 0`, value `n * 7 - 100`;
//! - `bin` BINARY: present unless `n % 5 == 0`, bytes `b<n>`;
//! - `snum` SORTED_NUMERIC: present when `n % 4 != 1`, values `n`, `n + 1`;
//! - `kw` SORTED: present when `n % 7 != 2`, value `k<n % 50>`.
//!
//! Usage: `write_index_format <output-dir> <10.5|10.6> [--merge]`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::index_format::IndexFormat;
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

const DOCS: i64 = 9_000;
const FLUSH_EVERY: i32 = 2_500;

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: true,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options: IndexOptions::None,
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

fn doc(n: i64) -> Document {
    let mut fields = vec![StoredField {
        field_number: 0,
        value: FieldValue::String(format!("doc{n}")),
    }];
    if n % 3 != 0 {
        fields.push(StoredField {
            field_number: 1,
            value: FieldValue::Long(n * 7 - 100),
        });
    }
    if n % 5 != 0 {
        fields.push(StoredField {
            field_number: 2,
            value: FieldValue::Binary(format!("b{n}").into_bytes()),
        });
    }
    if n % 4 != 1 {
        for v in [n, n + 1] {
            fields.push(StoredField {
                field_number: 3,
                value: FieldValue::Long(v),
            });
        }
    }
    if n % 7 != 2 {
        fields.push(StoredField {
            field_number: 4,
            value: FieldValue::String(format!("k{}", n % 50)),
        });
    }
    Document { fields }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("output dir");
    let format = IndexFormat::parse(args.get(2).expect("10.5 or 10.6")).expect("10.5 or 10.6");
    let merge = args.iter().any(|a| a == "--merge");
    std::fs::create_dir_all(out).unwrap();
    let dir = FsDirectory::open(out.to_string());
    let fields = vec![
        FieldInfo {
            index_options: IndexOptions::Docs,
            ..field("id", 0)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("num", 1)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Binary,
            ..field("bin", 2)
        },
        FieldInfo {
            doc_values_type: DocValuesType::SortedNumeric,
            ..field("snum", 3)
        },
        FieldInfo {
            doc_values_type: DocValuesType::Sorted,
            ..field("kw", 4)
        },
    ];
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
    writer.set_index_format(format);
    writer.set_max_buffered_docs(FLUSH_EVERY).unwrap();
    writer.set_ram_buffer_size_mb(4096.0).unwrap();
    writer.set_postings_field(Some("id")).unwrap();
    writer.set_doc_values_field(Some("num")).unwrap();
    for f in ["bin", "snum", "kw"] {
        writer.add_doc_values_field(f).unwrap();
    }
    for n in 0..DOCS {
        writer.add_document(doc(n)).unwrap();
        if n + 1 == 6_000 {
            writer.commit().unwrap();
            let terms: Vec<Term> = (0..=n)
                .step_by(13)
                .map(|d| Term::new("id", format!("doc{d}").into_bytes()))
                .collect();
            writer.delete_documents_by_term(&terms).unwrap();
            writer.commit().unwrap();
            if merge {
                writer.force_merge(1).unwrap();
            }
        }
    }
    let sis = writer.commit().unwrap();
    println!(
        "wrote {DOCS} documents in {} segments as {} to {out}",
        sis.segments.len(),
        format.name()
    );
}
