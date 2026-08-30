//! Writes two indices whose doc-values have been **updated in place**
//! (`IndexWriter::update_numeric_doc_value` /
//! `update_binary_doc_value`), so real Lucene can read the generation-suffixed
//! files back.
//!
//! This is the reverse-direction check for the one write path this port used
//! to get structurally wrong: a doc-values update is *not* a delta file, it is
//! a full rewrite of the updated field's column into a new generation of
//! ordinary `Lucene90DocValuesFormat` files, plus a new `FieldInfos`
//! generation recording that field's `FieldInfo.docValuesGen`, plus two
//! entries in `segments_N` (`docValuesGen` + the per-field
//! `dvUpdatesFiles` map). Four separate things have to agree, and every way of
//! getting them wrong is silent through this port's own reader:
//!
//! - a wrong generation suffix inside a file's index header (the name says
//!   generation 1, the header says something else) -- the file opens by name
//!   and fails `checkIndexHeader`;
//! - a `FieldInfos` generation that is written but not recorded, or recorded
//!   but not written -- the new column is on disk, referenced, checksummed,
//!   and never read;
//! - `dvUpdatesFiles` that accumulates instead of replacing -- superseded
//!   generations stay referenced forever and are never reclaimed;
//! - a generation whose merged column dropped the docs the update did not
//!   touch -- Lucene reads it back as "no value" for every other document.
//!
//! Two indices because this writer carries one base doc-values column per
//! segment: `numeric/` exercises NUMERIC over **three** update rounds (so a
//! generation is itself read back as the base of the next one), `binary/`
//! exercises BINARY. The verifier reads every document's value through a real
//! `DirectoryReader` and runs `CheckIndex` on both.
//!
//! Usage: `write_doc_values_updates_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::{DocValuesUpdate, Term};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Must match `VerifyDocValuesUpdates.java`.
const NUM_DOCS: usize = 260;
/// The value every `even`-bodied document's `val` ends at, after three rounds.
const EVEN_FINAL: i64 = 7_000;
/// ...and every `odd`-bodied one.
const ODD_FINAL: i64 = 9_000;
/// The binary index's updated value for `even` documents.
const BINARY_UPDATED: &[u8] = b"updated-payload";
/// Documents `0..NUM_RESET` have their `val` **removed** in the last round
/// (`updateDocValues` with a null value == `DocValuesFieldUpdates.reset`), so
/// the merged column is genuinely sparse and real Lucene has to agree that
/// those documents have no value at all -- not a value of zero.
const NUM_RESET: usize = 40;

fn field(name: &str, number: i32, indexed: bool, dv: DocValuesType) -> FieldInfo {
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
        doc_values_type: dv,
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

fn parity(i: usize) -> &'static str {
    if i.is_multiple_of(2) {
        "even"
    } else {
        "odd"
    }
}

fn doc(i: usize, dv: FieldValue) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{i}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(parity(i).to_string()),
            },
            StoredField {
                field_number: 2,
                value: dv,
            },
        ],
    }
}

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

fn write_numeric_index(out: &str) {
    let dir = FsDirectory::open(out);
    let fields = vec![
        field("id", 0, true, DocValuesType::None),
        field("body", 1, true, DocValuesType::None),
        field("val", 2, false, DocValuesType::Numeric),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).expect("open writer");
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");
    // `id` is indexed too, so the reset round below can name one document at a
    // time.
    writer.add_postings_field("id").expect("add postings field");
    writer
        .set_doc_values_field(Some("val"))
        .expect("set doc values field");

    for i in 0..NUM_DOCS {
        writer
            .add_document(doc(i, FieldValue::Long(i as i64)))
            .expect("add document");
    }
    writer.commit().expect("commit base segment");

    // Round 1: every `even` document. The merged column must keep the `odd`
    // documents' *base* values -- the single easiest thing to lose in a
    // full-column rewrite, and invisible unless something reads them back.
    writer
        .update_numeric_doc_value(Term::new("body", b"even"), "val", 1_000)
        .expect("update even");
    writer.commit().expect("commit generation 1");

    // Round 2: every `odd` document. Generation 1 is now the base being read,
    // so a generation that cannot itself be re-read as a base fails here.
    writer
        .update_numeric_doc_value(Term::new("body", b"odd"), "val", ODD_FINAL)
        .expect("update odd");
    writer.commit().expect("commit generation 2");

    // Round 3: the `even` documents again, to a different value. The
    // superseded generations must stop being referenced.
    writer
        .update_numeric_doc_value(Term::new("body", b"even"), "val", EVEN_FINAL)
        .expect("update even again");
    writer.commit().expect("commit generation 3");

    // Round 4: **remove** the first NUM_RESET documents' value.
    // `DocValuesFieldUpdates.reset(doc)` is what `updateDocValues` reaches for
    // a field passed with a null value, and it is the only thing that makes a
    // rewritten column sparse. Real Lucene must read those documents back as
    // having no value, which is a different assertion from reading back a
    // wrong one.
    for i in 0..NUM_RESET {
        let term = Term::new("id", format!("doc{i}").as_bytes());
        writer
            .update_doc_values(
                term.clone(),
                &[DocValuesUpdate::Numeric {
                    term,
                    field: "val".to_string(),
                    value: None,
                }],
            )
            .expect("reset val");
    }
    writer.commit().expect("commit generation 4");
}

fn write_binary_index(out: &str) {
    let dir = FsDirectory::open(out);
    let fields = vec![
        field("id", 0, false, DocValuesType::None),
        field("body", 1, true, DocValuesType::None),
        field("tag", 2, false, DocValuesType::Binary),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).expect("open writer");
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");
    writer
        .set_doc_values_field(Some("tag"))
        .expect("set doc values field");

    for i in 0..NUM_DOCS {
        writer
            .add_document(doc(i, FieldValue::Binary(format!("base-{i}").into_bytes())))
            .expect("add document");
    }
    writer.commit().expect("commit base segment");

    writer
        .update_binary_doc_value(Term::new("body", b"even"), "tag", BINARY_UPDATED)
        .expect("update even");
    writer.commit().expect("commit generation 1");
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_doc_values_updates_fixture <output-dir>");
    let numeric = format!("{out_dir}/numeric");
    let binary = format!("{out_dir}/binary");
    std::fs::create_dir_all(&numeric).expect("create numeric dir");
    std::fs::create_dir_all(&binary).expect("create binary dir");

    write_numeric_index(&numeric);
    write_binary_index(&binary);

    println!("wrote doc-values-update indices ({NUM_DOCS} documents each) to {out_dir}");
}
