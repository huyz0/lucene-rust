//! Writes a whole index the way an application would -- one `IndexWriter`,
//! `add_document`, `commit` -- rather than one codec file at a time.
//!
//! Every other fixture in `scripts/verify-write-path.sh` hands real Lucene a
//! single codec file with a *hand-built* `SegmentInfo`/`FieldInfos`, which
//! deliberately scopes each check to one format. The gap that leaves is
//! everything that binds those files into a segment: per-field format routing
//! and the `.fnm` attributes that record it, the `.psm` metadata file, and the
//! cross-file lengths `.tmd` declares. All four were broken at once while all
//! thirteen single-format checks passed (see `docs/sweep/findings.md`,
//! "Then real Lucene tried to open what we wrote").
//!
//! So this fixture writes no files itself. It drives the real write path and
//! leaves an index directory for `VerifyFullSegment` to open with
//! `DirectoryReader` and run `CheckIndex` over.
//!
//! Usage: `write_full_segment_fixture <output-dir>`.
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
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Enough documents to cross a stored-fields chunk boundary (1024 docs) and
/// several postings blocks (128 docs), so the segment exercises the block and
/// chunk paths rather than only their single-element degenerate cases.
const NUM_DOCS: usize = 2_500;

fn field(name: &str, number: i32, indexed: bool) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        // Left false on the indexed field on purpose: Lucene writes norms for
        // any indexed field that does not omit them, and refuses to open a
        // segment whose `.fnm` claims norms the files do not carry.
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

/// `title`'s text for document `i`, or `None` when the document does not
/// carry the field at all. `VerifyFullSegment` recomputes this same rule.
fn title_for(i: usize) -> Option<String> {
    match i % 7 {
        0 => None,
        1 => Some(String::new()),
        n => Some(
            (0..n)
                .map(|w| format!("t{w}"))
                .collect::<Vec<_>>()
                .join(" "),
        ),
    }
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_full_segment_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let dir = FsDirectory::open(&out_dir);
    let fields = vec![
        field("id", 0, false),
        field("body", 1, true),
        FieldInfo {
            doc_values_type: DocValuesType::Numeric,
            ..field("score", 2, false)
        },
        // A second indexed field, whose only purpose is that **nothing here
        // opts it into norms**. Lucene writes norms for every indexed field
        // that does not omit them; before c35 this port required a
        // `set_norms_field` call per field and rewrote every other indexed
        // field's `.fnm` entry as `omit_norms: true`, so BM25 scored it
        // against a constant length. `VerifyFullSegment` reads this field's
        // norm for every document back through real Lucene and compares it
        // to `SmallFloat.intToByte4(tokenCount)`.
        field("title", 3, true),
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
    writer
        .set_postings_field(Some("body"))
        .expect("set postings field");
    writer
        .add_postings_field("title")
        .expect("add postings field");
    // Doc values are the other per-field format, and so the other half of the
    // file-naming and `.fnm`-attribute contract postings exercise above.
    writer
        .set_doc_values_field(Some("score"))
        .expect("set doc values field");

    // A small vocabulary reused across documents, so terms carry real doc
    // frequencies (and so `shared` lands in every document, giving the
    // verifier a term whose postings span every block).
    // Deliberately spanning many distinct leading bytes: terms differing in
    // their first byte were what broke the term dictionary for real Lucene.
    let vocab: Vec<String> = (0..500)
        .map(|i| format!("{}{i:03}", (b'a' + (i % 26) as u8) as char))
        .collect();
    for i in 0..NUM_DOCS {
        let body = format!(
            "shared {} {}",
            vocab[i % vocab.len()],
            vocab[(i / 7) % vocab.len()]
        );
        let mut fields = vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("doc{i}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String(body),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Long(i as i64 * 3 - 1000),
            },
        ];
        // `title` covers all three cases `NormValuesWriter` distinguishes and
        // that a dense-only writer collapses into one: a document that does
        // not carry the field at all (no norm), one that carries it but
        // tokenizes to nothing (an explicit `0`), and one with a real,
        // varying length.
        if let Some(title) = title_for(i) {
            fields.push(StoredField {
                field_number: 3,
                value: FieldValue::String(title),
            });
        }
        writer.add_document(Document { fields }).unwrap();
    }
    writer.commit().expect("commit");

    println!("wrote a {NUM_DOCS}-document index to {out_dir}");
}
