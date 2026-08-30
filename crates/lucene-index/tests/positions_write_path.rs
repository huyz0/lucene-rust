//! The positional write path, end to end and checked by this port's own
//! `CheckIndex` port.
//!
//! `scripts/verify-write-path.sh`'s `write_positions_segment_fixture` /
//! `VerifyPositionsSegment` case is the cross-engine half of batch c23: real
//! Lucene 10.5.0 opens a segment this port wrote and walks its `.pos`/`.pay`
//! occurrence by occurrence. This is the in-tree half, and it exists because
//! that one needs a JVM and the checked-in Lucene jars, so `cargo test` cannot
//! run it.
//!
//! `check_index`'s positional and offset ordering checks
//! (`postings.positions_valid:*`, `postings.offsets_valid:*`, the
//! term-vectors-versus-postings cross-check) were ported by batch c9 against
//! **Java-written** fixtures. Until this batch nothing this port *wrote*
//! carried positions at a scale that reached them, because
//! `IndexWriter`-produced segments indexed `DocsAndFreqs`. This test is their
//! first writer-produced input. It deliberately does **not** try to make those
//! checks fail -- driving their failure arms is batch `c25`'s scope -- it
//! proves they run, and pass, over bytes this port emitted.
//!
//! The corpus is sized to the format, not to convenience: 8 500 documents with
//! a term in every one of them so a whole `LEVEL1_NUM_DOCS` (8 192) span
//! closes and 33 level-0 blocks plus a group-varint tail all carry pos/pay
//! skip records, and a frequency cycle of period 5 (coprime with the
//! 256-occurrence `.pos` block) so the level-1 skip record's in-block offset
//! is non-zero rather than accidentally landing on a block boundary.
#![allow(clippy::arithmetic_side_effects)]
// Test code opts out of the arithmetic gate at its own boundary: the gate is
// about values read off disk, and this file's arithmetic is its own corpus
// layout. See `docs/arithmetic-gate.md`.

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::check_index;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Past `LEVEL1_NUM_DOCS` (8 192), so one level-1 skip record is written and
/// read.
const NUM_DOCS: usize = 8_500;

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

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir(tag: &str) -> TempDir {
    TempDir::new(&format!("c23-{tag}"))
}

/// Period 5, coprime with the 256-occurrence `.pos` block, so `.pos` block
/// boundaries drift against `.doc` block boundaries instead of lining up with
/// them -- c20's Tier-2 review found a period-4 cycle making a level-1 skip
/// record's in-block offset indistinguishable from a hardcoded zero.
fn dense_freq(doc: usize) -> usize {
    1 + doc % 5
}

fn body_text(doc: usize) -> String {
    let mut tokens: Vec<String> = Vec::new();
    for k in 0..dense_freq(doc) {
        tokens.push("dense".to_string());
        tokens.push(format!("w{:03}", (doc * 3 + k) % 400));
    }
    tokens.join(" ")
}

/// Writes the corpus into a fresh directory and returns it, plus the
/// directory so the caller can keep it alive.
fn write_index(tag: &str) -> (TempDir, usize) {
    let path = tempdir(tag);
    let dir = FsDirectory::open(&path);
    let fields = vec![
        field("id", 0),
        FieldInfo {
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            store_payloads: true,
            store_term_vectors: true,
            omit_norms: false,
            ..field("body", 1)
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
    // One segment: the block thresholds this corpus is sized against are
    // per-term-per-segment, so a RAM-triggered flush partway through would
    // leave every term below all of them.
    writer.set_max_buffered_docs(NUM_DOCS as i32 + 1).unwrap();
    writer.set_ram_buffer_size_mb(4096.0).unwrap();
    writer.set_postings_field(Some("body")).unwrap();
    writer.set_term_vector_field(Some("body")).unwrap();
    writer
        .set_payload_source(Some(Box::new(|ctx| {
            // Length cycles 0..=4, so a block's payload byte run is not
            // uniform and zero-length payloads (Lucene's `null`-payload
            // equivalent) are common.
            let len = (ctx.doc_id as usize * 7 + ctx.position as usize) % 5;
            if len == 0 {
                None
            } else {
                Some(
                    (0..len)
                        .map(|i| ((ctx.doc_id as usize + ctx.position as usize + i) & 0xFF) as u8)
                        .collect(),
                )
            }
        })))
        .unwrap();

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
    (path, NUM_DOCS)
}

/// The headline: every check this port's `CheckIndex` port runs must pass over
/// a segment whose postings carry positions, offsets and payloads and whose
/// term vectors carry the same three axes -- bytes no `CheckIndex` run, ours
/// or Java's, had ever seen this port write before batch c23.
#[test]
fn our_check_index_passes_over_a_writer_produced_positions_segment() {
    let (path, num_docs) = write_index("check-index");
    let dir = FsDirectory::open(&path);
    let results = check_index::check_directory(&dir).expect("check the written index");
    // `check_directory` returns one result per segment plus a commit-level
    // one; the segment is the one carrying a `max_doc`.
    let segment = results
        .iter()
        .find(|r| r.max_doc == Some(num_docs as i32))
        .unwrap_or_else(|| panic!("no segment result with max_doc={num_docs}: {results:?}"));
    for result in &results {
        assert!(
            result.all_passed(),
            "check_index rejected a segment this port wrote: {:?}",
            result.failures()
        );
    }
    let _ = segment;
    std::fs::remove_dir_all(&path).ok();
}

/// The checks above must actually **run**, not be skipped. A segment whose
/// postings never reached the positional path would satisfy `all_passed()`
/// vacuously, which is precisely the shape this batch exists to rule out. So
/// the positional, offset and term-vector cross-checks are each required by
/// name.
#[test]
fn the_positional_offset_and_vector_checks_all_fire_on_it() {
    let (path, _) = write_index("checks-fire");
    let dir = FsDirectory::open(&path);
    let results = check_index::check_directory(&dir).expect("check the written index");
    let names: Vec<&str> = results
        .iter()
        .flat_map(|r| r.checks.iter())
        .map(|c| c.name.as_str())
        .collect();
    for wanted in [
        "postings.positions_valid:body",
        "postings.offsets_valid:body",
    ] {
        assert!(
            names.contains(&wanted),
            "{wanted} did not run; checks were {names:?}"
        );
    }
    // The term-vectors-versus-postings cross-check is where a vector that
    // silently dropped offsets or payloads would show up: it compares them
    // occurrence by occurrence for every document.
    assert!(
        names.iter().any(|n| n.starts_with("term_vectors.")),
        "no term-vector check ran; checks were {names:?}"
    );
    std::fs::remove_dir_all(&path).ok();
}
