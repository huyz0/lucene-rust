//! Port of the stored-fields-only slice of `org.apache.lucene.index.SegmentMerger`
//! (plus the field-numbering half of `FieldInfos.FieldNumbers`) -- merges N
//! already-flushed, stored-fields-only segments (see `segment_writer`'s module
//! doc for exactly what "stored-fields-only" means in this port) into one new
//! stored-fields-only segment, dropping deleted docs and renumbering doc ids to
//! be contiguous (`0..mergedDocCount`).
//!
//! # What this is
//!
//! [`merge_stored_only_segments`] takes, for each source segment, its already
//! read-back [`FieldInfos`](field_infos::FieldInfos) and [`Document`]s (via
//! this port's stored-fields reader, [`stored_fields::open`] +
//! [`stored_fields::StoredFieldsReader::document`]) plus an optional
//! per-source live-docs bitset (via [`live_docs::parse`], or `None` if the
//! source has no deletions), and:
//! 1. reconciles field numbering across sources by field name (see
//!    [`reconcile_field_numbers`]) -- real Lucene's `FieldInfos.FieldNumbers`
//!    does the same job (a global, writer-wide field-number authority so the
//!    same field name gets the same number everywhere), scoped down here to
//!    exactly the merge-time case: two segments naming the same field
//!    differently, or a field only some segments have;
//! 2. filters out non-live docs per source, remaps each surviving doc's
//!    field numbers to the merged numbering, and renumbers docs contiguously
//!    by simply concatenating surviving docs in source order (matches real
//!    `SegmentMerger`'s `MergeState.docMaps`, minus any doc-ID-remapping
//!    policy fancier than "keep source order, drop gaps" -- this port has no
//!    index sort or other doc-reordering merge policy yet);
//! 3. hands the merged fields + merged docs to
//!    [`crate::segment_writer::flush_stored_only_segment`], which already
//!    does exactly the write-side work a merge's output segment needs (write
//!    `.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si`, return a [`SegmentCommitInfo`]) --
//!    nothing merge-specific needed duplicating there.
//!
//! # What this deliberately is not
//!
//! - **Not a merge policy.** No `TieredMergePolicy`-style "which segments
//!   should merge, and when" decision -- the caller picks the sources.
//! - **Not concurrent/background.** One synchronous call, like
//!   `flush_stored_only_segment`.
//! - **No merge-time codec upgrade.** The merged segment's codec/version are
//!   caller-supplied, same stance as `flush_stored_only_segment`.
//! - **No doc values / points / norms / term vectors / postings merging.**
//!   None of those have a write-side caller that produces a full segment yet
//!   in this port (see `segment_writer`'s doc comment) -- there is nothing to
//!   merge beyond stored fields today.
//! - **No `FieldInfos.FieldNumbers`-style full schema-consistency check.**
//!   Real Lucene's field-number authority also verifies that two segments
//!   agreeing on a field name agree on its indexing options, doc-values
//!   type, etc. (`verifySameSchema`). Every field in every segment this port
//!   can currently produce is stored-only (`IndexOptions::None`, no doc
//!   values/points/vectors), so there is no schema attribute left to
//!   disagree on beyond the name -- this reconciliation only needs to unify
//!   *numbers*, not resolve real schema conflicts. Revisit once a second
//!   write-side field kind exists.
//!
//! See `docs/parity.md` and `PLAN.md`'s Phase 5 section for the exact,
//! currently-true scope line.

use std::collections::HashMap;

use crate::segment_info::LuceneVersion;
use crate::segment_infos::SegmentCommitInfo;
use crate::segment_writer::{self, Error as SegmentWriterError};
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::stored_fields::Document;
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::Directory;
use lucene_util::fixed_bit_set::FixedBitSet;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    SegmentWriter(#[from] SegmentWriterError),
    #[error(transparent)]
    StoredFields(#[from] lucene_codecs::stored_fields::Error),
    /// A `MergeSource`'s stored fields referenced a field number absent from
    /// that same source's own `field_infos` -- an inconsistent/malformed
    /// `MergeSource` (its `reader` and `field_infos` don't actually describe
    /// the same segment), not something a well-formed caller can trigger.
    #[error(
        "source segment's stored field number {field_number} has no entry in that source's own field_infos"
    )]
    UnknownSourceFieldNumber { field_number: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;

/// One source segment's already-decoded input to a merge: its field infos
/// (from `.fnm`, via [`lucene_codecs::field_infos::parse`]), a stored-fields
/// reader over its `.fdt`/`.fdx`/`.fdm` (via [`stored_fields::open`]), and an
/// optional live-docs bitset (`None` means "no deletions -- every doc up to
/// `reader.max_doc()` is live", matching a segment whose `SegmentCommitInfo`
/// has `del_gen == -1`).
pub struct MergeSource<'a> {
    pub field_infos: &'a [FieldInfo],
    pub reader: &'a lucene_codecs::stored_fields::StoredFieldsReader<'a>,
    pub live_docs: Option<&'a FixedBitSet>,
}

/// Reconciles field numbering across `sources_fields` (one source's
/// [`FieldInfos`](field_infos::FieldInfos)-equivalent field list per entry):
/// assigns every distinct field *name* a single, contiguous merged field
/// number, in first-seen order across sources (source 0's fields first, then
/// any new names introduced by source 1, etc.) -- mirrors real Lucene's
/// `FieldInfos.FieldNumbers.addOrGet`, which hands out a process-wide number
/// per name and reuses it for every segment that has that field, regardless
/// of what number that segment originally used.
///
/// Returns the merged field list (one [`FieldInfo`] per distinct name, using
/// the *first* source's metadata for that name -- see this module's "what
/// this deliberately is not" note on schema consistency) and, per source, a
/// map from that source's original field number to the merged number.
pub fn reconcile_field_numbers(
    sources_fields: &[&[FieldInfo]],
) -> (Vec<FieldInfo>, Vec<HashMap<i32, i32>>) {
    let mut merged_fields: Vec<FieldInfo> = Vec::new();
    let mut name_to_merged_number: HashMap<String, i32> = HashMap::new();
    let mut per_source_maps: Vec<HashMap<i32, i32>> = Vec::with_capacity(sources_fields.len());

    for fields in sources_fields {
        let mut map = HashMap::with_capacity(fields.len());
        for f in *fields {
            let merged_number = *name_to_merged_number
                .entry(f.name.clone())
                .or_insert_with(|| {
                    let number = merged_fields.len() as i32;
                    let mut renumbered = f.clone();
                    renumbered.number = number;
                    merged_fields.push(renumbered);
                    number
                });
            map.insert(f.number, merged_number);
        }
        per_source_maps.push(map);
    }

    (merged_fields, per_source_maps)
}

/// Merges `sources` (already-opened, in source order) into one brand-new
/// stored-fields-only segment named `merged_segment_name` inside `dir`,
/// exactly as [`crate::segment_writer::flush_stored_only_segment`] writes a
/// freshly-flushed one -- deleted docs (per each source's `live_docs`) are
/// dropped, surviving docs are renumbered contiguously by concatenating
/// sources in order, and field numbers are reconciled by name (see
/// [`reconcile_field_numbers`]).
///
/// A source with `live_docs` fully cleared (every doc deleted) naturally
/// contributes zero docs to the merge -- this port merges it anyway rather
/// than requiring the caller to have already dropped it (real Lucene's
/// `IndexWriter` drops a 100%-deleted segment before a merge is even
/// scheduled, purely as a merge-policy optimization; skipping that
/// optimization here costs nothing but a no-op source pass).
pub fn merge_stored_only_segments(
    dir: &dyn Directory,
    sources: &[MergeSource],
    merged_segment_name: &str,
    merged_segment_id: [u8; ID_LENGTH],
    codec_name: &str,
    lucene_version: LuceneVersion,
) -> Result<SegmentCommitInfo> {
    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (merged_fields, per_source_maps) = reconcile_field_numbers(&sources_fields);

    let mut merged_docs: Vec<Document> = Vec::new();
    for (source, field_number_map) in sources.iter().zip(per_source_maps.iter()) {
        let max_doc = source.reader.max_doc();
        for doc_id in 0..max_doc {
            let is_live = source
                .live_docs
                .map(|bits| bits.get(doc_id as usize))
                .unwrap_or(true);
            if !is_live {
                continue;
            }
            let mut doc = source.reader.document(doc_id)?;
            for field in &mut doc.fields {
                field.field_number = *field_number_map.get(&field.field_number).ok_or(
                    Error::UnknownSourceFieldNumber {
                        field_number: field.field_number,
                    },
                )?;
            }
            merged_docs.push(doc);
        }
    }

    Ok(segment_writer::flush_stored_only_segment(
        dir,
        merged_segment_name,
        merged_segment_id,
        codec_name,
        lucene_version,
        &merged_fields,
        &merged_docs,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{
        DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
        VectorSimilarityFunction,
    };
    use lucene_codecs::stored_fields::{self, FieldValue, StoredField};
    use lucene_store::directory::FsDirectory;

    fn version() -> LuceneVersion {
        LuceneVersion {
            major: 10,
            minor: 0,
            bugfix: 0,
        }
    }

    fn field(name: &str, number: i32) -> FieldInfo {
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

    fn doc_with(field_number: i32, value: &str) -> Document {
        Document {
            fields: vec![StoredField {
                field_number,
                value: FieldValue::String(value.to_string()),
            }],
        }
    }

    // --- reconcile_field_numbers ---

    #[test]
    fn single_source_keeps_its_own_numbering_shape() {
        let fields = vec![field("id", 0), field("body", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields];
        let (merged, maps) = reconcile_field_numbers(&sources);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[0].number, 0);
        assert_eq!(merged[1].name, "body");
        assert_eq!(merged[1].number, 1);
        assert_eq!(maps[0].get(&0), Some(&0));
        assert_eq!(maps[0].get(&1), Some(&1));
    }

    #[test]
    fn same_name_different_numbers_across_sources_unify() {
        // Source 0 has "id"=0, "body"=1; source 1 has "body"=0, "id"=1 --
        // opposite numbering for the exact same two field names.
        let fields0 = vec![field("id", 0), field("body", 1)];
        let fields1 = vec![field("body", 0), field("id", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, maps) = reconcile_field_numbers(&sources);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[1].name, "body");

        // Source 0: id(0)->0, body(1)->1
        assert_eq!(maps[0][&0], 0);
        assert_eq!(maps[0][&1], 1);
        // Source 1: body(0)->1, id(1)->0
        assert_eq!(maps[1][&0], 1);
        assert_eq!(maps[1][&1], 0);
    }

    #[test]
    fn field_present_in_only_some_sources_gets_its_own_merged_number() {
        let fields0 = vec![field("id", 0)];
        let fields1 = vec![field("id", 0), field("extra", 1)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, maps) = reconcile_field_numbers(&sources);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "id");
        assert_eq!(merged[1].name, "extra");
        assert_eq!(maps[0].len(), 1);
        assert_eq!(maps[1][&1], 1);
    }

    #[test]
    fn empty_sources_produce_empty_merged_fields() {
        let sources: Vec<&[FieldInfo]> = vec![];
        let (merged, maps) = reconcile_field_numbers(&sources);
        assert!(merged.is_empty());
        assert!(maps.is_empty());
    }

    #[test]
    fn merged_field_keeps_first_sources_metadata() {
        let mut fields0 = vec![field("id", 0)];
        fields0[0].doc_values_gen = 99;
        let fields1 = vec![field("id", 5)];
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, _maps) = reconcile_field_numbers(&sources);
        assert_eq!(merged[0].doc_values_gen, 99);
    }

    // --- merge_stored_only_segments (full round-trip via real Directory I/O) ---

    fn tempdir() -> String {
        let dir = std::env::temp_dir().join(format!(
            "lucene-rust-merge-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.to_str().unwrap().to_string()
    }

    /// Flushes one stored-fields-only segment (via `flush_stored_only_segment`)
    /// and returns its bytes plus field list, ready to be re-opened as a merge
    /// source -- mirrors how a real caller would read a segment's files off
    /// disk before merging it.
    struct FlushedSegment {
        fdt: Vec<u8>,
        fdx: Vec<u8>,
        fdm: Vec<u8>,
        fields: Vec<FieldInfo>,
        segment_id: [u8; ID_LENGTH],
    }

    fn flush(
        dir: &FsDirectory,
        tmp: &str,
        name: &str,
        segment_id: [u8; ID_LENGTH],
        fields: &[FieldInfo],
        docs: &[Document],
    ) -> FlushedSegment {
        segment_writer::flush_stored_only_segment(
            dir,
            name,
            segment_id,
            "Lucene104",
            version(),
            fields,
            docs,
        )
        .unwrap();
        FlushedSegment {
            fdt: std::fs::read(std::path::Path::new(tmp).join(format!("{name}.fdt"))).unwrap(),
            fdx: std::fs::read(std::path::Path::new(tmp).join(format!("{name}.fdx"))).unwrap(),
            fdm: std::fs::read(std::path::Path::new(tmp).join(format!("{name}.fdm"))).unwrap(),
            fields: fields.to_vec(),
            segment_id,
        }
    }

    fn open_reader(seg: &FlushedSegment) -> stored_fields::StoredFieldsReader<'_> {
        stored_fields::open(&seg.fdt, &seg.fdx, &seg.fdm, &seg.segment_id, "").unwrap()
    }

    #[test]
    fn two_segments_no_deletions_merge_with_contiguous_doc_ids() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "c")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource {
                field_infos: &seg0.fields,
                reader: &reader0,
                live_docs: None,
            },
            MergeSource {
                field_infos: &seg1.fields,
                reader: &reader1,
                live_docs: None,
            },
        ];

        let sci = merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_merged");

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 3);
        let vals: Vec<String> = (0..3)
            .map(
                |i| match &merged_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "b", "c"]);
    }

    #[test]
    fn some_docs_deleted_in_each_source_are_dropped() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "c"), doc_with(0, "d")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);

        let mut live0 = FixedBitSet::new(2);
        live0.set(0); // keep "a", drop "b"
        let mut live1 = FixedBitSet::new(2);
        live1.set(1); // drop "c", keep "d"

        let sources = vec![
            MergeSource {
                field_infos: &seg0.fields,
                reader: &reader0,
                live_docs: Some(&live0),
            },
            MergeSource {
                field_infos: &seg1.fields,
                reader: &reader1,
                live_docs: Some(&live1),
            },
        ];

        let dir2 = FsDirectory::open(&tmp);
        merge_stored_only_segments(
            &dir2,
            &sources,
            "_merged2",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged2.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged2.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged2.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 2);
        let vals: Vec<String> = (0..2)
            .map(
                |i| match &merged_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "d"]);
    }

    #[test]
    fn fully_deleted_source_contributes_zero_docs() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "b"), doc_with(0, "c")],
        );

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let live1 = FixedBitSet::new(2); // all deleted, nothing set

        let sources = vec![
            MergeSource {
                field_infos: &seg0.fields,
                reader: &reader0,
                live_docs: None,
            },
            MergeSource {
                field_infos: &seg1.fields,
                reader: &reader1,
                live_docs: Some(&live1),
            },
        ];

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged3",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged3.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged3.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged3.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 1);
        match &merged_reader.document(0).unwrap().fields[0].value {
            FieldValue::String(s) => assert_eq!(s, "a"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn field_number_mismatch_across_sources_is_reconciled_during_merge() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        // Source 0: "id"=0, "body"=1. Source 1: "body"=0, "id"=1 (opposite).
        let fields0 = vec![field("id", 0), field("body", 1)];
        let fields1 = vec![field("body", 0), field("id", 1)];

        let doc0 = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("id0".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("body0".to_string()),
                },
            ],
        };
        let doc1 = Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("body1".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("id1".to_string()),
                },
            ],
        };

        let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields0, &[doc0]);
        let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields1, &[doc1]);

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource {
                field_infos: &seg0.fields,
                reader: &reader0,
                live_docs: None,
            },
            MergeSource {
                field_infos: &seg1.fields,
                reader: &reader1,
                live_docs: None,
            },
        ];

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged4",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fdm")).unwrap();
        let merged_fnm = std::fs::read(std::path::Path::new(&tmp).join("_merged4.fnm")).unwrap();
        let merged_fields =
            lucene_codecs::field_infos::parse(&merged_fnm, &[9u8; ID_LENGTH], "").unwrap();
        let id_number = merged_fields
            .fields
            .iter()
            .find(|f| f.name == "id")
            .unwrap()
            .number;
        let body_number = merged_fields
            .fields
            .iter()
            .find(|f| f.name == "body")
            .unwrap()
            .number;

        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 2);

        let doc0 = merged_reader.document(0).unwrap();
        let id0 = doc0
            .fields
            .iter()
            .find(|f| f.field_number == id_number)
            .unwrap();
        assert_eq!(id0.value, FieldValue::String("id0".to_string()));
        let body0 = doc0
            .fields
            .iter()
            .find(|f| f.field_number == body_number)
            .unwrap();
        assert_eq!(body0.value, FieldValue::String("body0".to_string()));

        let doc1 = merged_reader.document(1).unwrap();
        let id1 = doc1
            .fields
            .iter()
            .find(|f| f.field_number == id_number)
            .unwrap();
        assert_eq!(id1.value, FieldValue::String("id1".to_string()));
        let body1 = doc1
            .fields
            .iter()
            .find(|f| f.field_number == body_number)
            .unwrap();
        assert_eq!(body1.value, FieldValue::String("body1".to_string()));
    }

    #[test]
    fn no_sources_produces_an_empty_segment() {
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let sources: Vec<MergeSource> = vec![];
        let sci = merge_stored_only_segments(
            &dir,
            &sources,
            "_empty",
            [3u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();
        assert_eq!(sci.segment_name, "_empty");

        // Re-open the actually-written files rather than trusting the
        // returned `SegmentCommitInfo` alone -- confirms a zero-doc merge
        // produces a genuinely well-formed, re-openable segment (max_doc ==
        // 0, no documents iterable), not just a struct that claims success.
        let fdt = std::fs::read(std::path::Path::new(&tmp).join("_empty.fdt")).unwrap();
        let fdx = std::fs::read(std::path::Path::new(&tmp).join("_empty.fdx")).unwrap();
        let fdm = std::fs::read(std::path::Path::new(&tmp).join("_empty.fdm")).unwrap();
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        assert_eq!(reader.max_doc(), 0);
    }

    #[test]
    fn stored_field_number_absent_from_its_own_source_field_infos_is_an_error() {
        // A malformed `MergeSource`: its stored fields reference field number
        // 7, but its own `field_infos` never declares that number. Real
        // callers can't construct this from `flush_stored_only_segment` +
        // `field_infos::parse`, but merge_stored_only_segments should still
        // surface it as an `Err`, not panic, per this port's stance of never
        // trusting a caller-supplied invariant with an `unwrap`/`expect` when
        // an `Err` is easy to return instead.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];
        let docs = vec![doc_with(7, "orphaned")];
        let seg = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
        let reader = open_reader(&seg);

        let sources = vec![MergeSource {
            field_infos: &seg.fields,
            reader: &reader,
            live_docs: None,
        }];
        let result = merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(matches!(
            result,
            Err(Error::UnknownSourceFieldNumber { field_number: 7 })
        ));
    }

    #[test]
    fn full_round_trip_through_a_real_written_and_reparsed_liv_file() {
        // End-to-end: flush 2 segments, write a real `.liv` for one of them
        // via `lucene_codecs::live_docs::write`, read it back via `parse`
        // (not just constructed in memory), merge, then confirm the merged
        // segment's stored fields match exactly the surviving docs.
        let tmp = tempdir();
        let dir = FsDirectory::open(&tmp);
        let fields = vec![field("id", 0)];

        let seg0 = flush(
            &dir,
            &tmp,
            "_0",
            [1u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "a"), doc_with(0, "b"), doc_with(0, "c")],
        );
        let seg1 = flush(
            &dir,
            &tmp,
            "_1",
            [2u8; ID_LENGTH],
            &fields,
            &[doc_with(0, "d")],
        );

        // Delete doc 1 ("b") from _0 via a real .liv file, round-tripped
        // through the actual write/parse pair.
        let mut live0 = FixedBitSet::new(3);
        live0.set(0);
        live0.set(2);
        let liv_bytes = lucene_codecs::live_docs::write(&live0, &seg0.segment_id, 1, 1).unwrap();
        let parsed_live0 =
            lucene_codecs::live_docs::parse(&liv_bytes, &seg0.segment_id, 1, 3, 1).unwrap();

        let reader0 = open_reader(&seg0);
        let reader1 = open_reader(&seg1);
        let sources = vec![
            MergeSource {
                field_infos: &seg0.fields,
                reader: &reader0,
                live_docs: Some(&parsed_live0),
            },
            MergeSource {
                field_infos: &seg1.fields,
                reader: &reader1,
                live_docs: None,
            },
        ];

        merge_stored_only_segments(
            &dir,
            &sources,
            "_merged5",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .unwrap();

        let merged_fdt = std::fs::read(std::path::Path::new(&tmp).join("_merged5.fdt")).unwrap();
        let merged_fdx = std::fs::read(std::path::Path::new(&tmp).join("_merged5.fdx")).unwrap();
        let merged_fdm = std::fs::read(std::path::Path::new(&tmp).join("_merged5.fdm")).unwrap();
        let merged_reader =
            stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "")
                .unwrap();
        assert_eq!(merged_reader.max_doc(), 3);
        let vals: Vec<String> = (0..3)
            .map(
                |i| match &merged_reader.document(i).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    _ => unreachable!(),
                },
            )
            .collect();
        assert_eq!(vals, vec!["a", "c", "d"]);
    }

    #[test]
    fn stored_fields_error_wraps_into_this_modules_error_type() {
        // Confirms `Error::StoredFields`'s `#[from]` wrapping actually
        // propagates a real `stored_fields::Error` (the kind
        // `reader.document()` can return mid-merge, e.g. a corrupted chunk)
        // as an `Err` through this module's own error type, rather than
        // requiring a full corrupt-fixture integration setup to exercise the
        // conversion.
        let source_err = stored_fields::Error::DocOutOfRange(5, 3);
        let wrapped: Error = source_err.into();
        assert!(matches!(wrapped, Error::StoredFields(_)));
    }
}
