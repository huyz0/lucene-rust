#![cfg(test)]
// The arithmetic gate is about values read off disk; a fixture builder's
// `i * 2` is not one (see `docs/arithmetic-gate.md`).
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_codecs::blocktree;
use lucene_codecs::field_infos::{
    self as fi, DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
use lucene_store::directory::FsDirectory;

use crate::segment_info::LuceneVersion as LV;
use crate::segment_infos::SegmentCommitInfo;

// --- shared fixture plumbing (same real-Lucene fixture term_delete.rs
// uses; see that module's test doc comment for the known contents:
// field `body`, term `cat` -> docs [0, 2], term `dog` -> docs [0, 1],
// term `bird` -> docs [1, 4]; max_doc = 8959). ---

struct Fixture {
    fields: BlockTreeFields,
    doc_bytes: Vec<u8>,
    segment_id: [u8; ID_LENGTH],
    suffix: String,
    max_doc: usize,
}

impl Fixture {
    fn doc_in(&self) -> DocInput<'_> {
        DocInput::open(&self.doc_bytes, &self.segment_id, &self.suffix).expect("open .doc")
    }
}

fn open_fixture() -> Fixture {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTree)");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let id_hex = get("id_hex");
    let mut segment_id = [0u8; ID_LENGTH];
    for (i, slot) in segment_id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = get("segment_suffix");
    let max_doc: i32 = get("max_doc").parse().unwrap();

    let read_raw = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
    };
    let fnm = read_raw(&get("fnm_file_name"));
    let field_infos = fi::parse(&fnm, &segment_id, "").expect("parse .fnm");
    let tim = read_raw(&get("tim_file_name"));
    let tip = read_raw(&get("tip_file_name"));
    let tmd = read_raw(&get("tmd_file_name"));
    let fields = blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos,
        &segment_id,
        &suffix,
        max_doc,
    )
    .expect("open blocktree");
    let doc_bytes = read_raw(&get("doc_file_name"));

    Fixture {
        fields,
        doc_bytes,
        segment_id,
        suffix,
        max_doc: max_doc as usize,
    }
}

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("update-document")
}

fn version() -> LV {
    LV {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn infos_version() -> crate::segment_infos::LuceneVersion {
    crate::segment_infos::LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
    }
}

fn old_sci(segment_id: [u8; ID_LENGTH]) -> SegmentCommitInfo {
    SegmentCommitInfo {
        segment_name: "_0".to_string(),
        segment_id,
        codec_name: "Lucene104".to_string(),
        del_gen: -1,
        del_count: 0,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

fn base_segment_infos(segment_id: [u8; ID_LENGTH]) -> SegmentInfos {
    SegmentInfos {
        id: [1u8; ID_LENGTH],
        generation: 1,
        format_version: crate::segment_infos::VERSION_86,
        lucene_version: infos_version(),
        index_created_version_major: 10,
        version: 1,
        counter: 1,
        min_segment_lucene_version: Some(infos_version()),
        segments: vec![old_sci(segment_id)],
        user_data: vec![],
    }
}

fn stored_only_field(name: &str, number: i32) -> FieldInfo {
    FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms: false,
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
        vector_encoding: VectorEncoding::Float32,
        vector_similarity_function: VectorSimilarityFunction::Euclidean,
    }
}

fn new_doc(value: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(value.to_string()),
        }],
    }
}

/// Update replacing a single existing doc: the term matches exactly one
/// live doc in the old segment ("id" -> "id0" matches only doc 0, a
/// singleton postings entry, per the fixture's known contents).
#[test]
fn update_replacing_a_single_existing_doc() {
    let fx = open_fixture();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let infos = base_segment_infos(fx.segment_id);

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: None, // singleton term, no `.doc` needed
        live_docs: None,
        max_doc: fx.max_doc,
    }];

    let new_fields = vec![stored_only_field("id", 0)];
    let new_docs = vec![new_doc("replacement")];

    let updated = update_document(
        &dir,
        &infos,
        &sources,
        "id",
        b"id0",
        "_1",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
        &new_fields,
        &new_docs,
    )
    .unwrap();

    assert_eq!(updated.generation, 2);
    assert_eq!(updated.version, 2);
    assert_eq!(updated.segments.len(), 2);

    let old = &updated.segments[0];
    assert_eq!(old.segment_name, "_0");
    assert_eq!(old.del_gen, 1);
    assert_eq!(old.del_count, 1);

    let new_seg = &updated.segments[1];
    assert_eq!(new_seg.segment_name, "_1");
    assert_eq!(new_seg.del_count, 0);

    // The delete really landed on disk: doc 0 is no longer live.
    let liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
    let parsed = lucene_codecs::live_docs::parse(&liv, &fx.segment_id, 1, fx.max_doc, 1).unwrap();
    assert!(!parsed.get(0));

    // The new document really landed on disk in the new segment.
    let fdt = std::fs::read(tmp.join("_1.fdt")).unwrap();
    let fdx = std::fs::read(tmp.join("_1.fdx")).unwrap();
    let fdm = std::fs::read(tmp.join("_1.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
    let doc = reader.document(0).unwrap();
    assert_eq!(doc.fields.len(), 1);
    assert_eq!(
        doc.fields[0].value,
        FieldValue::String("replacement".to_string())
    );

    // And the commit itself is readable back as a normal segments_N.
    let segments_bytes = std::fs::read(tmp.join("segments_2")).unwrap();
    let reparsed = segment_infos::parse(&segments_bytes, 2).unwrap();
    assert_eq!(reparsed.segments.len(), 2);
}

/// Term matches zero existing docs: real Lucene's `updateDocument` on a
/// fresh term acts exactly like `addDocument` -- no delete applied
/// anywhere, del_count/del_gen on the old segment stay untouched, only
/// the new segment is added.
#[test]
fn update_with_no_matching_term_acts_like_a_pure_insert() {
    let fx = open_fixture();
    let doc_in = fx.doc_in();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let infos = base_segment_infos(fx.segment_id);

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: fx.max_doc,
    }];

    let new_fields = vec![stored_only_field("id", 0)];
    let new_docs = vec![new_doc("brand-new")];

    let updated = update_document(
        &dir,
        &infos,
        &sources,
        "body",
        b"zzz-missing-term",
        "_1",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
        &new_fields,
        &new_docs,
    )
    .unwrap();

    let old = &updated.segments[0];
    // A no-op delete round still bumps del_gen (matches
    // `resolve_and_apply_term_delete`'s existing, documented behavior)
    // but never touches del_count.
    assert_eq!(old.del_gen, 1);
    assert_eq!(old.del_count, 0);

    assert_eq!(updated.segments.len(), 2);
    assert_eq!(updated.segments[1].segment_name, "_1");
}

/// Term matches multiple existing docs: all of them are deleted, and
/// exactly one new doc is added ("body" -> "cat" matches docs [0, 2] per
/// the fixture).
#[test]
fn update_where_term_matches_multiple_docs_deletes_all_of_them() {
    let fx = open_fixture();
    let doc_in = fx.doc_in();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let infos = base_segment_infos(fx.segment_id);

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: fx.max_doc,
    }];

    let new_fields = vec![stored_only_field("id", 0)];
    let new_docs = vec![new_doc("one-replacement")];

    let updated = update_document(
        &dir,
        &infos,
        &sources,
        "body",
        b"cat",
        "_1",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
        &new_fields,
        &new_docs,
    )
    .unwrap();

    let old = &updated.segments[0];
    assert_eq!(old.del_gen, 1);
    assert_eq!(old.del_count, 2); // docs 0 and 2 both deleted

    let liv = std::fs::read(tmp.join("_0_1.liv")).unwrap();
    let parsed = lucene_codecs::live_docs::parse(&liv, &fx.segment_id, 1, fx.max_doc, 2).unwrap();
    assert!(!parsed.get(0));
    assert!(!parsed.get(2));
    assert!(parsed.get(1)); // untouched

    // Exactly one new doc was added, regardless of how many old docs
    // were deleted.
    assert_eq!(updated.segments.len(), 2);
    let fdt = std::fs::read(tmp.join("_1.fdt")).unwrap();
    let fdx = std::fs::read(tmp.join("_1.fdx")).unwrap();
    let fdm = std::fs::read(tmp.join("_1.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(reader.document(0).unwrap().fields.len(), 1);
}

/// A segment the caller has *no* opened delete source for is left
/// completely untouched -- it still shows up in the new commit
/// (real Lucene's `updateDocument` never drops a segment it didn't
/// touch), just with the same `del_gen`/`del_count` it already had.
#[test]
fn segment_with_no_delete_source_is_left_untouched() {
    let fx = open_fixture();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let infos = base_segment_infos(fx.segment_id);

    let new_fields = vec![stored_only_field("id", 0)];
    let new_docs = vec![new_doc("insert-only")];

    let updated = update_document(
        &dir,
        &infos,
        &[], // no delete sources supplied at all
        "body",
        b"cat",
        "_1",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
        &new_fields,
        &new_docs,
    )
    .unwrap();

    let old = &updated.segments[0];
    assert_eq!(old.del_gen, -1);
    assert_eq!(old.del_count, 0);
    assert_eq!(updated.segments.len(), 2);
}

/// A failing delete (out-of-range doc id surfaced through a bogus
/// `max_doc`) must abort before any commit is written -- the previous
/// `segments_1` stays the only file on disk, and no new segment's files
/// leak into a visible commit.
#[test]
fn a_failing_delete_step_commits_nothing() {
    let fx = open_fixture();
    let doc_in = fx.doc_in();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let infos = base_segment_infos(fx.segment_id);

    let sources = [SegmentDeleteSource {
        segment_name: "_0",
        fields: &fx.fields,
        doc_in: Some(&doc_in),
        live_docs: None,
        max_doc: 1, // too small: doc id 2 (from "cat") is out of range
    }];

    let new_fields = vec![stored_only_field("id", 0)];
    let new_docs = vec![new_doc("should-not-land")];

    let result = update_document(
        &dir,
        &infos,
        &sources,
        "body",
        b"cat",
        "_1",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
        &new_fields,
        &new_docs,
    );

    assert!(result.is_err());
    assert!(!tmp.join("segments_2").exists());
    assert!(!tmp.join("_1.fdt").exists());
}
