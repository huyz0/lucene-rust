//! **The norms the impact writer indexes must be the norms of the documents
//! whose postings it is writing.**
//!
//! `impacts_soundness.rs` checks that a block's bound never falls below the
//! score of a document it covers -- but it hands the *same* norm array to the
//! writer and to the scorer, so it is structurally blind to a wiring error.
//! Reversing `build_norms_output`'s dense column, a misalignment in which
//! every value stays legal, leaves every Rust test in `lucene-index` and
//! `lucene-search` passing; only a `debug_assert` catches it, only in a debug
//! build, and only by the luck of landing a norm-0 document on one that has a
//! posting. In a release build the segment is written, and
//! `CheckIndex.checkImpacts` does not help either: it validates non-emptiness,
//! the first freq/norm, strict ordering and level-N domination, but never that
//! an impact *bounds* the documents it covers. So `verify-write-path.sh` would
//! pass a misaligned segment too.
//!
//! This file checks the same invariant one layer up, where the wiring lives:
//! the segment is built through `IndexWriter`, and the norms are read back out
//! of the segment's own `.nvd` rather than remembered from the input. A
//! divergence between `merge_norms`' and `merge_postings`' doc-id compaction,
//! or a field-number mismatch between `NormsFieldConfig` and
//! `PostingsFieldConfig`, produces a segment that passes every other gate and
//! silently under-bounds its blocks -- and fails here.
//!
//! Raised as a gating finding by the Tier-2 review of batch `c42-readpath-perf`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::postings::{DocInput, Postings};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_codecs::{blocktree, norms};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::similarity::{decode_norm, max_score_for_impacts, score};
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

/// Enough documents for several level-0 blocks, so a bound covers many
/// documents rather than one.
const NUM_DOCS: usize = 1_200;
/// Present in every document, so its postings span every block.
const TERM: &[u8] = b"shared";

fn field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        index_options: IndexOptions::DocsAndFreqs,
        store_term_vectors: false,
        omit_norms: false,
        store_payloads: false,
        doc_values_type: DocValuesType::None,
        doc_values_gen: -1,
        doc_values_skip_index_type: DocValuesSkipIndexType::None,
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension: 0,
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
        soft_deletes_field: false,
        parent_field: false,
        attributes: vec![],
    }
}

/// Field lengths that grow with the doc id, so the *blocks* differ from one
/// another and not merely the documents within them.
///
/// This is the part that makes the test able to fail, and the first version of
/// it could not: lengths of `(doc * 37) % 61` are uniform, so every 256-document
/// block held the same minimum norm, and reversing the column left each block's
/// bound exactly where it was. A misalignment is only observable when the norms
/// a block *should* hold differ from the ones some other block holds -- so the
/// short, high-scoring documents have to be concentrated at one end. The `% 7`
/// term keeps lengths varying inside a block as well, so the frontier is not a
/// single point.
fn body_text(doc: usize) -> String {
    let extra = doc / 24 + (doc % 7);
    let mut s = String::from("shared");
    for i in 0..extra {
        s.push(' ');
        s.push_str(if i % 3 == 0 { "alpha" } else { "beta" });
    }
    s
}

fn write_segment(dir: &FsDirectory) -> [u8; lucene_store::codec_util::ID_LENGTH] {
    let fields = vec![
        field("id", 0),
        FieldInfo {
            omit_norms: false,
            ..field("body", 1)
        },
    ];
    let mut writer = IndexWriter::open(
        dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .expect("open writer");
    // One segment: the block thresholds are per-term-per-segment, and a
    // mid-corpus flush would leave every term below all of them.
    writer.set_max_buffered_docs(NUM_DOCS as i32 + 1).unwrap();
    writer.set_ram_buffer_size_mb(4096.0).unwrap();
    writer.set_postings_field(Some("body")).unwrap();

    for doc in 0..NUM_DOCS {
        writer
            .add_document(Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String(format!("doc{doc}")),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::String(body_text(doc)),
                    },
                ],
            })
            .unwrap();
    }
    let sis = writer.commit().expect("commit");
    assert_eq!(sis.segments.len(), 1, "the corpus must land in one segment");
    sis.segments[0].segment_id
}

/// The one file-name assumption in here, kept in one place.
fn read(dir: &FsDirectory, name: &str) -> Vec<u8> {
    dir.open(name)
        .unwrap_or_else(|e| panic!("open {name}: {e}"))
        .to_vec()
}

#[test]
fn impact_bounds_hold_against_the_segments_own_norms() {
    let tmp = TempDir::new("impacts-alignment");
    let dir = FsDirectory::open(&tmp);
    let seg_id = write_segment(&dir);

    let field_infos_bytes = read(&dir, "_0.fnm");
    let field_infos = lucene_codecs::field_infos::parse(&field_infos_bytes, &seg_id, "")
        .expect("parse field infos");
    let body = field_infos
        .fields
        .iter()
        .find(|f| f.name == "body")
        .expect("body field");

    // Norms straight out of the segment -- the whole point of this file.
    let nvm = read(&dir, "_0.nvm");
    let nvd = read(&dir, "_0.nvd");
    let (_, parsed) = norms::parse_meta(&nvm, &seg_id, "").expect("parse .nvm");
    let entry = parsed.entry(body.number).expect("body must carry norms");
    let segment_norm = |doc: i32| -> i64 {
        norms::norm_value(&nvd, entry, doc)
            .expect("read norm")
            .expect("every document carries this field")
    };

    let suffix = "Lucene104_0";
    let tim = read(&dir, &format!("_0_{suffix}.tim"));
    let tip = read(&dir, &format!("_0_{suffix}.tip"));
    let tmd = read(&dir, &format!("_0_{suffix}.tmd"));
    let doc_bytes = read(&dir, &format!("_0_{suffix}.doc"));
    let fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &seg_id,
        suffix,
        NUM_DOCS as i32,
    )
    .expect("open terms");
    let doc_in = DocInput::open(&doc_bytes, &seg_id, suffix).expect("open .doc");
    let postings: Postings = fields
        .field("body")
        .expect("body postings")
        .postings(TERM, Some(&doc_in))
        .expect("read postings")
        .expect("the shared term must be present");

    let doc_freq = postings.docs.len() as i64;
    let doc_count = NUM_DOCS as i64;
    // The same average this port's scorer derives for the field.
    let total_len: i64 = (0..NUM_DOCS)
        .map(|d| decode_norm(segment_norm(d as i32)) as i64)
        .sum();
    let avg_field_length = total_len as f32 / doc_count as f32;

    assert!(
        !postings.level0_impacts.is_empty(),
        "the corpus must produce level-0 impacts, or this proves nothing"
    );

    for (label, column) in [
        ("level 0", &postings.level0_impacts),
        ("level 1", &postings.level1_impacts),
    ] {
        let mut covered_from = 0i32;
        for (last_doc, impacts) in column {
            let bound = max_score_for_impacts(impacts, doc_freq, doc_count, avg_field_length);
            for (i, &d) in postings.docs.iter().enumerate() {
                if d < covered_from || d > *last_doc {
                    continue;
                }
                let actual = score(
                    doc_freq,
                    doc_count,
                    postings.freqs[i] as f32,
                    decode_norm(segment_norm(d)),
                    avg_field_length,
                );
                assert!(
                    actual <= bound,
                    "{label}: doc {d} scores {actual} against its own segment's norm, \
                     above its block's bound {bound} -- the norm column the impact \
                     writer saw is not the one the segment stores, and MAXSCORE \
                     would skip this document"
                );
            }
            covered_from = *last_doc + 1;
        }
    }
}
