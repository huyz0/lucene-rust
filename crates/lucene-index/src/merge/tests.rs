#![cfg(test)]
// The arithmetic gate is about values read off disk; a fixture builder's
// own index arithmetic is not one (see `docs/arithmetic-gate.md`).
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::index_writer::{per_field_codec_suffix, per_field_segment, PER_FIELD_SUFFIX};
use crate::index_writer::{DOC_VALUES_FORMAT_NAME, POSTINGS_FORMAT_NAME};
use crate::segment_writer;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::postings_writer::TermPostings;
use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};
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
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
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
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();

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
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].name, "id");
    assert_eq!(merged[1].name, "extra");
    assert_eq!(maps[0].len(), 1);
    assert_eq!(maps[1][&1], 1);
}

#[test]
fn a_source_holding_only_some_fields_keeps_its_numbers() {
    // Lucene writes only the fields a segment has, under the writer's
    // global numbers: the larger segment first here has only field 3.
    let fields0 = vec![field("x3", 3)];
    let fields1 = vec![field("x0", 0), field("x1", 1), field("x3", 3)];
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
    let numbered: Vec<(&str, i32)> = merged.iter().map(|f| (f.name.as_str(), f.number)).collect();
    assert_eq!(numbered, [("x0", 0), ("x1", 1), ("x3", 3)]);
    assert!(maps.iter().all(|m| m.iter().all(|(from, to)| from == to)));
}

#[test]
fn a_number_two_sources_give_different_fields_is_reassigned() {
    let fields0 = vec![field("a", 0), field("b", 2)];
    let fields1 = vec![field("c", 0), field("b", 2), field("d", 3)];
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
    let numbered: Vec<(&str, i32)> = merged.iter().map(|f| (f.name.as_str(), f.number)).collect();
    assert_eq!(numbered, [("a", 0), ("b", 2), ("d", 3), ("c", 4)]);
    assert_eq!(maps[1][&0], 4);
    assert_eq!(maps[1][&3], 3);
}

#[test]
fn empty_sources_produce_empty_merged_fields() {
    let sources: Vec<&[FieldInfo]> = vec![];
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
    assert!(merged.is_empty());
    assert!(maps.is_empty());
}

#[test]
fn doc_id_maps_are_dense_arrays_with_minus_one_for_deleted_docs() {
    // Source 0 keeps docs 0 and 2 (1 deleted), source 1 keeps doc 1
    // only. Merged ids are assigned by concatenation: 0,1 for source 0
    // then 2 for source 1 -- Lucene's MergeState.DocMap contract.
    let live = vec![vec![0, 2], vec![1]];
    let maps = build_doc_id_maps(&[3, 2], &concat_doc_order(&live));
    assert_eq!(maps[0], vec![0, -1, 1]);
    assert_eq!(maps[1], vec![-1, 2]);

    assert_eq!(mapped_doc_id(&maps[0], 0), Some(0));
    assert_eq!(mapped_doc_id(&maps[0], 1), None, "deleted doc");
    assert_eq!(mapped_doc_id(&maps[0], 2), Some(1));
    // Past the source's own maxDoc.
    assert_eq!(mapped_doc_id(&maps[0], 3), None);
    assert_eq!(mapped_doc_id(&maps[0], -1), None);

    // A fully-deleted source gets an empty map, and every lookup misses.
    let maps = build_doc_id_maps(&[0], &[]);
    assert!(maps[0].is_empty());
    assert_eq!(mapped_doc_id(&maps[0], 0), None);
}

#[test]
fn only_order_preserving_doc_maps_allow_the_native_points_merge() {
    assert!(doc_map_is_monotonic(&[0, -1, 1, 4, -1]));
    assert!(doc_map_is_monotonic(&[]));
    // A reorder or an index sort can move a later doc ahead of an
    // earlier one; equal 1D values would then leave the k-way merge
    // with doc ids out of order.
    assert!(!doc_map_is_monotonic(&[3, -1, 1]));
    assert!(!doc_map_is_monotonic(&[2, 2]));
}

#[test]
fn concat_doc_order_walks_sources_in_order() {
    let live = vec![vec![0, 2], vec![], vec![1]];
    assert_eq!(
        concat_doc_order(&live),
        vec![(0usize, 0i32), (0, 2), (2, 1)]
    );
}

#[test]
fn a_source_naming_the_same_field_twice_is_rejected() {
    // Java's FieldInfos(FieldInfo[]) constructor: "duplicate field names".
    let fields = vec![field("body", 0), field("body", 1)];
    let sources: Vec<&[FieldInfo]> = vec![&fields];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::DuplicateFieldNameInSource {
            source_index: 0,
            ..
        })
    ));

    // The same name in *different* sources is the normal case.
    let fields0 = vec![field("body", 0)];
    let fields1 = vec![field("body", 7)];
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    let (merged, maps) = reconcile_field_numbers(&sources).unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(maps[1][&7], 0);
}

#[test]
fn store_payloads_is_ored_across_sources_not_taken_from_the_first() {
    // FieldInfos.Builder.add: `if (fi.hasPayloads()) curFi.setStorePayloads()`.
    let mut fields0 = vec![field("body", 0)];
    fields0[0].index_options = IndexOptions::DocsAndFreqsAndPositions;
    let mut fields1 = vec![field("body", 0)];
    fields1[0].index_options = IndexOptions::DocsAndFreqsAndPositions;
    fields1[0].store_payloads = true;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    let (merged, _) = reconcile_field_numbers(&sources).unwrap();
    assert!(merged[0].store_payloads);

    // ...and the OR is order-independent.
    let sources: Vec<&[FieldInfo]> = vec![&fields1, &fields0];
    let (merged, _) = reconcile_field_numbers(&sources).unwrap();
    assert!(merged[0].store_payloads);
}

#[test]
fn store_payloads_or_respects_set_store_payloads_positions_guard() {
    // `FieldInfo.setStorePayloads` only sets the flag when
    // `indexOptions.subsumes(DOCS_AND_FREQS_AND_POSITIONS)`; otherwise
    // `checkConsistency` would throw "indexed field cannot have payloads
    // without positions". A source claiming payloads on a positionless
    // field must therefore NOT push the merged field into that state.
    for options in [
        IndexOptions::Docs,
        IndexOptions::DocsAndFreqs,
        // Java special-cases DocsAndCustomFreqs to subsume as if it were
        // DocsAndFreqs, so it does not subsume positions either.
        IndexOptions::DocsAndCustomFreqs,
    ] {
        let mut fields0 = vec![field("body", 0)];
        fields0[0].index_options = options;
        let mut fields1 = fields0.clone();
        fields1[0].store_payloads = true;
        let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
        let (merged, _) = reconcile_field_numbers(&sources).unwrap();
        assert!(
            !merged[0].store_payloads,
            "{options:?} does not subsume positions, so the OR must not apply"
        );
    }
}

#[test]
fn attributes_are_put_all_ed_across_sources_not_taken_from_the_first() {
    // FieldInfos.Builder.add: `curFi.putAttributes(fi.attributes())` is a
    // Map.putAll -- a later source's value wins for a shared key, and its
    // own keys are added.
    let mut fields0 = vec![field("body", 0)];
    fields0[0].attributes = vec![
        ("shared".to_string(), "from-0".to_string()),
        ("only-0".to_string(), "a".to_string()),
    ];
    let mut fields1 = vec![field("body", 0)];
    fields1[0].attributes = vec![
        ("shared".to_string(), "from-1".to_string()),
        ("only-1".to_string(), "b".to_string()),
    ];
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    let (merged, _) = reconcile_field_numbers(&sources).unwrap();
    let mut attrs = merged[0].attributes.clone();
    attrs.sort();
    assert_eq!(
        attrs,
        vec![
            ("only-0".to_string(), "a".to_string()),
            ("only-1".to_string(), "b".to_string()),
            ("shared".to_string(), "from-1".to_string()),
        ]
    );
}

#[test]
fn disagreeing_index_options_across_sources_is_rejected_without_any_postings_data() {
    // FieldInfo.verifySameIndexOptions -- checked for every field of
    // every merged segment, not just fields whose postings are supplied.
    let mut fields0 = vec![field("body", 0)];
    fields0[0].index_options = IndexOptions::DocsAndFreqs;
    let mut fields1 = vec![field("body", 0)];
    fields1[0].index_options = IndexOptions::DocsAndFreqsAndPositions;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::PostingsIndexOptionsDisagreement { .. })
    ));
}

#[test]
fn disagreeing_store_term_vectors_across_sources_is_rejected() {
    // FieldInfo.verifySameStoreTermVectors. Without this the merged
    // .fnm would claim no term vectors while write_merged_term_vectors happily
    // wrote source 1's.
    let mut fields0 = vec![field("body", 0)];
    fields0[0].index_options = IndexOptions::DocsAndFreqs;
    let mut fields1 = fields0.clone();
    fields1[0].store_term_vectors = true;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::FieldSchemaDisagreement {
            attribute: "store_term_vectors",
            ..
        })
    ));
}

#[test]
fn disagreeing_omit_norms_across_sources_is_rejected_only_when_indexed() {
    // FieldInfo.verifySameOmitNorms, guarded by Java's
    // `if (this.indexOptions != IndexOptions.NONE)`.
    let mut fields0 = vec![field("body", 0)];
    fields0[0].index_options = IndexOptions::DocsAndFreqs;
    let mut fields1 = fields0.clone();
    fields1[0].omit_norms = true;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::FieldSchemaDisagreement {
            attribute: "omit_norms",
            ..
        })
    ));

    // A non-indexed field's omitNorms/storeTermVector are not compared
    // (they are meaningless there), exactly as in Java.
    let fields0 = vec![field("body", 0)];
    let mut fields1 = fields0.clone();
    fields1[0].omit_norms = true;
    fields1[0].store_term_vectors = true;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(reconcile_field_numbers(&sources).is_ok());
}

#[test]
fn disagreeing_doc_values_type_across_sources_is_rejected() {
    // FieldInfo.verifySameDocValuesType.
    let fields0 = vec![field("dv", 0)];
    let mut fields1 = fields0.clone();
    fields1[0].doc_values_type = DocValuesType::Numeric;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::FieldSchemaDisagreement {
            attribute: "doc_values_type",
            ..
        })
    ));
}

#[test]
fn disagreeing_points_shape_in_field_infos_is_rejected_without_any_points_data() {
    // FieldInfo.verifySamePointsOptions -- the FieldInfos-level half of
    // the check merge_points also performs against each source's actual
    // on-disk BKD tree.
    let mut fields0 = vec![field("pt", 0)];
    fields0[0].point_dimension_count = 1;
    fields0[0].point_index_dimension_count = 1;
    fields0[0].point_num_bytes = 4;
    let mut fields1 = fields0.clone();
    fields1[0].point_dimension_count = 2;
    fields1[0].point_index_dimension_count = 2;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::PointsShapeDisagreement { .. })
    ));

    // num_dims/bytes_per_dim agree but the index/data dim split differs.
    let mut fields1 = fields0.clone();
    fields1[0].point_dimension_count = 1;
    fields1[0].point_index_dimension_count = 0;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::PointsIndexDimsDisagreement { .. })
    ));
}

#[test]
fn disagreeing_vector_options_across_sources_are_rejected() {
    // FieldInfo.verifySameVectorOptions.
    let mut fields0 = vec![field("vec", 0)];
    fields0[0].vector_dimension = 8;
    let mut fields1 = fields0.clone();
    fields1[0].vector_dimension = 16;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::FieldSchemaDisagreement {
            attribute: "vector_dimension",
            ..
        })
    ));

    let mut fields1 = fields0.clone();
    fields1[0].vector_similarity_function = VectorSimilarityFunction::DotProduct;
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    assert!(matches!(
        reconcile_field_numbers(&sources),
        Err(Error::FieldSchemaDisagreement {
            attribute: "vector_similarity_function",
            ..
        })
    ));
}

#[test]
fn merged_field_keeps_first_sources_metadata() {
    let mut fields0 = vec![field("id", 0)];
    fields0[0].doc_values_gen = 99;
    let fields1 = vec![field("id", 5)];
    let sources: Vec<&[FieldInfo]> = vec![&fields0, &fields1];
    let (merged, _maps) = reconcile_field_numbers(&sources).unwrap();
    assert_eq!(merged[0].doc_values_gen, 99);
}

// --- merge_stored_only_segments (full round-trip via real Directory I/O) ---

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless the
/// test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("merge")
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
    tmp: &std::path::Path,
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
        false,
    )
    .unwrap();
    FlushedSegment {
        fdt: std::fs::read(tmp.join(format!("{name}.fdt"))).unwrap(),
        fdx: std::fs::read(tmp.join(format!("{name}.fdx"))).unwrap(),
        fdm: std::fs::read(tmp.join(format!("{name}.fdm"))).unwrap(),
        fields: fields.to_vec(),
        segment_id,
    }
}

fn open_reader(seg: &FlushedSegment) -> stored_fields::StoredFieldsReader<'_> {
    stored_fields::open(&seg.fdt, &seg.fdx, &seg.fdm, &seg.segment_id, "").unwrap()
}

#[test]
fn failed_merge_removes_earlier_outputs_and_preserves_each_colliding_file() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = [field("id", 0)];
    let stored = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "source")],
    );
    let reader = open_reader(&stored);
    for extension in ["fdt", "fdx", "fdm", "fnm", "si"] {
        let collision = format!("_merged.{extension}");
        let mut output = dir.create_output(&collision).unwrap();
        output.write_bytes(b"other owner");
        output.close().unwrap();
        let snapshot = || {
            dir.list_all()
                .unwrap()
                .into_iter()
                .map(|name| {
                    let bytes = dir.open(&name).unwrap().to_vec();
                    (name, bytes)
                })
                .collect::<Vec<_>>()
        };
        let before = snapshot();
        let source = MergeSource::stored_only(&stored.fields, &reader, None, None);
        assert!(merge_stored_only_segments(
            &dir,
            &[source],
            "_merged",
            [2u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
        .is_err());
        assert_eq!(snapshot(), before, "collision at {extension}");
        dir.delete_file(&collision).unwrap();
    }
    let source = MergeSource::stored_only(&stored.fields, &reader, None, None);
    let merged = merge_stored_only_segments(
        &dir,
        &[source],
        "_merged",
        [2u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(merged.segment_name, "_merged");
    let si = dir.open("_merged.si").unwrap();
    let info = segment_info::parse(&si, &[2u8; ID_LENGTH]).unwrap();
    assert_eq!(info.doc_count, 1);
    assert!(info.files.iter().all(|name| dir.open(name).is_ok()));
}

// --- the merge-completeness gate (`check_format_coverage`) ---

/// The table `check_format_coverage` reads must partition extensions:
/// two formats claiming one extension would attribute a file to the
/// wrong `is_opened` question, and a format claiming a non-format
/// extension (`si`/`fnm`/`liv`) would make the gate demand readers for
/// the segment's own bookkeeping.
#[test]
fn no_two_segment_formats_claim_the_same_extension() {
    let mut seen: Vec<(&str, SegmentFormat)> = Vec::new();
    for format in SegmentFormat::ALL {
        assert!(
            !format.extensions().is_empty(),
            "{format:?} claims no extension, so nothing can ever require it to be opened"
        );
        for ext in format.extensions() {
            if let Some((_, other)) = seen.iter().find(|(e, _)| e == ext) {
                panic!("extension `{ext}` is claimed by both {other:?} and {format:?}");
            }
            assert!(
                !NON_FORMAT_EXTENSIONS.iter().any(|(e, _)| e == ext),
                "extension `{ext}` is claimed by {format:?} and also listed as a non-format"
            );
            seen.push((ext, format));
        }
    }
    // Every non-format extension carries a stated reason, so the escape
    // hatch cannot grow silently.
    for (ext, reason) in NON_FORMAT_EXTENSIONS {
        assert!(
            !reason.is_empty(),
            "non-format extension `{ext}` has no reason"
        );
    }
}

/// `SegmentFormat::ALL` must actually be all of them. The fixed-length
/// array makes a *short* list a compile error, but not a list with the
/// same variant twice.
#[test]
fn segment_format_all_lists_each_variant_once() {
    let mut seen: Vec<SegmentFormat> = Vec::new();
    for format in SegmentFormat::ALL {
        assert!(!seen.contains(&format), "{format:?} appears twice in ALL");
        seen.push(format);
    }
    // Round-trips through `for_extension`, which is how the gate finds
    // the format for a file it sees.
    for format in SegmentFormat::ALL {
        for ext in format.extensions() {
            assert_eq!(SegmentFormat::for_extension(ext), Some(format), "{ext}");
        }
    }
}

/// **The gate's negative control.** A source segment
/// whose own `.si` lists `.nvm`/`.nvd` merged through a `MergeSource`
/// with no norms: `execute_merge` passing `norms: &[]` would make every
/// merged BM25 score silently wrong. The
/// merge itself would succeed and produce a `CheckIndex`-clean segment,
/// so nothing but this check reports it.
#[test]
fn a_format_the_si_lists_and_the_merge_never_opened_is_refused() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    let seg = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "a")],
    );
    let reader = open_reader(&seg);
    let source = MergeSource::stored_only(&seg.fields, &reader, None, Some(version()));

    // Exactly what a norms-bearing flush puts in its `.si`.
    let files = vec![
        "_0.fdt".to_string(),
        "_0.fdx".to_string(),
        "_0.fdm".to_string(),
        "_0.fnm".to_string(),
        "_0.si".to_string(),
        "_0.nvm".to_string(),
        "_0.nvd".to_string(),
    ];
    let err = check_format_coverage(&["_0"], &[&files], std::slice::from_ref(&source), &[false])
        .expect_err("a `.nvd` the merge never opened must be refused");
    match &err {
        Error::MergeFormatNotOpened {
            segment,
            format,
            files,
        } => {
            assert_eq!(segment, "_0");
            assert_eq!(*format, "norms");
            // Both files are named, not just the first one found.
            assert!(
                files.contains("_0.nvm") && files.contains("_0.nvd"),
                "{files}"
            );
        }
        other => panic!("expected MergeFormatNotOpened, got {other:?}"),
    }

    // The same `.si` minus the norms files is fine: the gate is about a
    // format the source *has*, not about every format existing.
    let without_norms: Vec<String> = files
        .iter()
        .filter(|f| !f.ends_with(".nvm") && !f.ends_with(".nvd"))
        .cloned()
        .collect();
    check_format_coverage(
        &["_0"],
        &[&without_norms],
        std::slice::from_ref(&source),
        &[false],
    )
    .unwrap();
}

/// One case per format, so no `is_opened` arm can be wrong in the
/// permissive direction without a test noticing. Every format except
/// stored fields (whose reader is not optional on `MergeSource`) must be
/// refused when the `.si` has it and the `MergeSource` does not.
#[test]
fn every_optional_format_is_refused_when_the_si_has_it_and_the_source_does_not() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    let seg = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "a")],
    );
    let reader = open_reader(&seg);
    let source = MergeSource::stored_only(&seg.fields, &reader, None, Some(version()));
    let base = ["_0.fdt", "_0.fdx", "_0.fdm", "_0.fnm", "_0.si"];

    for format in SegmentFormat::ALL {
        if format == SegmentFormat::StoredFields {
            assert!(
                format.is_opened(&source),
                "stored fields are not optional on a MergeSource"
            );
            continue;
        }
        let mut files: Vec<String> = base.iter().map(|f| (*f).to_string()).collect();
        for ext in format.extensions() {
            files.push(format!("_0.{ext}"));
        }
        let err =
            check_format_coverage(&["_0"], &[&files], std::slice::from_ref(&source), &[false])
                .unwrap_err();
        match err {
            Error::MergeFormatNotOpened { format: named, .. } => {
                assert_eq!(named, format.name(), "{format:?}");
            }
            other => panic!("expected MergeFormatNotOpened for {format:?}, got {other:?}"),
        }
    }
}

/// **The gate's anti-rot arm.** A flush path that learns to write a new
/// format lands here on the first merge of a segment it wrote, rather
/// than having that format silently dropped -- and the only way to
/// satisfy the error is to add a `SegmentFormat` variant, whose two
/// exhaustive `match`es then force the caller to open it.
#[test]
fn an_extension_no_format_claims_is_refused_rather_than_ignored() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    let seg = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "a")],
    );
    let reader = open_reader(&seg);
    let source = MergeSource::stored_only(&seg.fields, &reader, None, Some(version()));

    for bad in ["_0.brandnew", "_0.cfs", "_0_no_extension_at_all"] {
        let files = vec![
            "_0.fdt".to_string(),
            "_0.fdx".to_string(),
            "_0.fdm".to_string(),
            "_0.si".to_string(),
            bad.to_string(),
        ];
        let err =
            check_format_coverage(&["_0"], &[&files], std::slice::from_ref(&source), &[false])
                .unwrap_err();
        match err {
            Error::UnknownSegmentFormat { file, .. } => assert_eq!(file, bad),
            other => panic!("expected UnknownSegmentFormat for {bad}, got {other:?}"),
        }
    }
}

/// Per source, not "any source": a two-source merge where only the
/// second has norms and only the second's readers were opened must
/// still be refused for the first, and vice versa. The bug shape this
/// pins is a caller that opens a format for source 0 and forgets the
/// loop for the rest.
#[test]
fn the_gate_asks_the_question_per_source() {
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
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];
    let plain: Vec<String> = ["fdt", "fdx", "fdm", "si"]
        .iter()
        .map(|e| format!("_0.{e}"))
        .collect();
    let with_vectors: Vec<String> = ["fdt", "fdx", "fdm", "si", "vec", "vemf"]
        .iter()
        .map(|e| format!("_1.{e}"))
        .collect();
    let err = check_format_coverage(
        &["_0", "_1"],
        &[&plain, &with_vectors],
        &sources,
        &[false, false],
    )
    .unwrap_err();
    match err {
        Error::MergeFormatNotOpened {
            segment, format, ..
        } => {
            assert_eq!(
                segment, "_1",
                "the second source is the one with the `.vec`"
            );
            assert_eq!(format, "KNN vectors");
        }
        other => panic!("expected MergeFormatNotOpened, got {other:?}"),
    }
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
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
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

    let merged_fdt = std::fs::read(tmp.join("_merged.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged.fdm")).unwrap();
    let merged_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "").unwrap();
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
        MergeSource::stored_only(&seg0.fields, &reader0, Some(&live0), Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, Some(&live1), Some(version())),
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

    let merged_fdt = std::fs::read(tmp.join("_merged2.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged2.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged2.fdm")).unwrap();
    let merged_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "").unwrap();
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
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, Some(&live1), Some(version())),
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

    let merged_fdt = std::fs::read(tmp.join("_merged3.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged3.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged3.fdm")).unwrap();
    let merged_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "").unwrap();
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
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
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

    let merged_fdt = std::fs::read(tmp.join("_merged4.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged4.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged4.fdm")).unwrap();
    let merged_fnm = std::fs::read(tmp.join("_merged4.fnm")).unwrap();
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
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "").unwrap();
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
    let fdt = std::fs::read(tmp.join("_empty.fdt")).unwrap();
    let fdx = std::fs::read(tmp.join("_empty.fdx")).unwrap();
    let fdm = std::fs::read(tmp.join("_empty.fdm")).unwrap();
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
    //
    // An earlier source already holds field number 1 under another name,
    // so the merge renumbers "id" and the malformed source is *not* a
    // matching reader -- which is what
    // puts it on the VISITOR path, the only one of
    // `Lucene90CompressingStoredFieldsWriter.merge`'s three that parses
    // field numbers at all. A matching reader's chunk bytes are copied
    // verbatim, in this port exactly as in Java, so a segment whose
    // `.fdt` disagrees with its own `.fnm` is a corruption `CheckIndex`
    // reports rather than something the merge re-derives -- see
    // `a_matching_deletion_free_source_is_bulk_copied_verbatim`.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let first_fields = vec![field("first", 1)];
    let first_docs = vec![doc_with(1, "fine")];
    let first = flush(
        &dir,
        &tmp,
        "_0",
        [2u8; ID_LENGTH],
        &first_fields,
        &first_docs,
    );
    let first_reader = open_reader(&first);
    let fields = vec![field("id", 1)];
    let docs = vec![doc_with(7, "orphaned")];
    let seg = flush(&dir, &tmp, "_1", [1u8; ID_LENGTH], &fields, &docs);
    let reader = open_reader(&seg);

    let sources = vec![
        MergeSource::stored_only(&first.fields, &first_reader, None, Some(version())),
        MergeSource::stored_only(&seg.fields, &reader, None, Some(version())),
    ];
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
        MergeSource::stored_only(&seg0.fields, &reader0, Some(&parsed_live0), Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
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

    let merged_fdt = std::fs::read(tmp.join("_merged5.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged5.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged5.fdm")).unwrap();
    let merged_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &[9u8; ID_LENGTH], "").unwrap();
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

// --- doc values / norms / term vectors merging ---

fn numeric_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.doc_values_type = DocValuesType::Numeric;
    f
}

/// A field that really can carry norms: `FieldInfo.hasNorms()` is
/// `indexOptions != NONE && !omitNorms`, and it is what
/// `NormsConsumer.merge` iterates -- so a field that is not indexed has
/// no norms to merge no matter what norms data a caller hands the merge.
fn norms_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.omit_norms = false;
    f.index_options = IndexOptions::DocsAndFreqs;
    f
}

fn tv_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.store_term_vectors = true;
    f
}

/// A test-owned numeric doc-values field: writes it via the real
/// write-side encoder, then re-parses the meta via the real read-side
/// decoder to get a genuine [`NumericEntry`] -- exercises the same
/// encode/decode round trip a real caller would, rather than hand
/// building a `NumericEntry`.
struct FlushedNumericDv {
    data: Vec<u8>,
    entry: NumericEntry,
}

fn flush_numeric_dv(
    field_number: i32,
    values: &[i64],
    segment_id: [u8; ID_LENGTH],
) -> FlushedNumericDv {
    let max_doc = values.len() as i32;
    let (meta, data, _skip) = doc_values::write_single_dense_numeric_field(
        field_number,
        values,
        max_doc,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    )
    .unwrap();
    let field_infos = field_infos::FieldInfos {
        fields: vec![numeric_field("x", field_number)],
    };
    let (_version, parsed) = doc_values::parse_meta(
        &meta,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = parsed.numeric_entry(field_number).unwrap().clone();
    FlushedNumericDv { data, entry }
}

impl FlushedNumericDv {
    fn source(&self) -> SourceNumericDocValues<'_> {
        SourceNumericDocValues {
            data: &self.data,
            entry: self.entry.clone(),
        }
    }
}

fn binary_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.doc_values_type = DocValuesType::Binary;
    f
}

/// Same idea as [`FlushedNumericDv`], for BINARY doc values.
struct FlushedBinaryDv {
    data: Vec<u8>,
    entry: BinaryEntry,
}

fn flush_binary_dv(
    field_number: i32,
    values: &[Vec<u8>],
    segment_id: [u8; ID_LENGTH],
) -> FlushedBinaryDv {
    let max_doc = values.len() as i32;
    let (meta, data, _skip) = doc_values::write_single_dense_binary_field(
        field_number,
        values,
        max_doc,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    )
    .unwrap();
    let field_infos = field_infos::FieldInfos {
        fields: vec![binary_field("x", field_number)],
    };
    let (_version, parsed) = doc_values::parse_meta(
        &meta,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = parsed.binary_entry(field_number).unwrap().clone();
    FlushedBinaryDv { data, entry }
}

impl FlushedBinaryDv {
    fn source(&self) -> SourceBinaryDocValues<'_> {
        SourceBinaryDocValues {
            data: &self.data,
            entry: self.entry.clone(),
        }
    }
}

struct FlushedNorms {
    data: Vec<u8>,
    entry: NormsEntry,
}

fn flush_norms(field_number: i32, values: &[i64], segment_id: [u8; ID_LENGTH]) -> FlushedNorms {
    let max_doc = values.len() as i32;
    let (meta, data) =
        norms::write_single_dense_field(field_number, values, max_doc, &segment_id, "").unwrap();
    let (_version, parsed) = norms::parse_meta(&meta, &segment_id, "").unwrap();
    let entry = *parsed.entry(field_number).unwrap();
    FlushedNorms { data, entry }
}

impl FlushedNorms {
    fn source(&self) -> SourceNorms<'_> {
        SourceNorms {
            data: &self.data,
            entry: self.entry,
        }
    }
}

struct FlushedTermVectors {
    tvd: Vec<u8>,
    tvx: Vec<u8>,
    tvm: Vec<u8>,
    segment_id: [u8; ID_LENGTH],
}

fn flush_term_vectors(
    docs: &[TermVectorsDocument],
    segment_id: [u8; ID_LENGTH],
) -> FlushedTermVectors {
    let (tvd, tvx, tvm) = term_vectors::write_best_speed(docs, &segment_id, "");
    FlushedTermVectors {
        tvd,
        tvx,
        tvm,
        segment_id,
    }
}

impl FlushedTermVectors {
    fn reader(&self) -> TermVectorsReader<'_> {
        term_vectors::open(&self.tvd, &self.tvx, &self.tvm, &self.segment_id, "").unwrap()
    }
}

fn tv_doc(field_number: i32, terms: &[(&str, i32)]) -> TermVectorsDocument {
    TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number,
            has_positions: true,
            has_offsets: false,
            has_payloads: false,
            terms: terms
                .iter()
                .map(|(t, pos)| TermVectorTerm {
                    term: t.as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![*pos]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                })
                .collect(),
        }],
    }
}

#[test]
fn numeric_doc_values_merge_across_two_sources_with_deletions() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    // Source 0: 2 docs, doc 1 deleted -> only doc "10" survives.
    let dv0 = flush_numeric_dv(0, &[10, 20], seg0_id);
    // Source 1: 1 doc, no deletions -> "30" survives.
    let dv1 = flush_numeric_dv(0, &[30], seg1_id);

    let fields = vec![numeric_field("num", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let mut live0 = FixedBitSet::new(2);
    live0.set(0); // keep doc 0 ("a"/10), drop doc 1 ("b"/20)

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &dv0_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &dv1_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_dv",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_dv");

    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_dv", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_dv", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![numeric_field("num", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.numeric_entry(0).unwrap();
    let values: Vec<i64> = (0..2)
        .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap().unwrap())
        .collect();
    assert_eq!(values, vec![10, 30]);
}

/// Task #54's numeric doc-values update overlay
/// (`lucene_codecs::doc_values_updates`) composed with this module's
/// numeric doc-values merge -- proving, end to end through both
/// already-shipped features' real public APIs, that a doc's
/// overlay-updated value does **not** survive a merge today.
///
/// This is not a bug in either feature: [`doc_values_updates`] is
/// documented (see that module's top doc comment) as a standalone
/// single-generation overlay primitive with no `SegmentCommitInfo`/`.si`
/// `docValuesGen` wiring, and nothing in `IndexWriter` ever applies an
/// overlay to a committed segment (there is no `update_numeric_doc_value`-
/// style method on `IndexWriter`; the overlay's only production consumer
/// is `lucene_search::soft_deletes`'s query-time live-docs check, which
/// never touches `merge.rs`). [`MergeSource`]/[`SourceNumericDocValues`]
/// accordingly carry no overlay field at all -- there is no parameter a
/// caller could even use to plug an overlay into a merge call. This test
/// documents the current, honest shape of that gap: build a base numeric
/// field, write a real overlay marking doc 0 with a new value via
/// [`lucene_codecs::doc_values_updates::write_numeric_updates`], confirm
/// the overlay-aware read
/// ([`lucene_codecs::doc_values_updates::numeric_value_with_updates`])
/// sees the new value, then run an actual `merge_stored_only_segments`
/// call using only the base `data`/`entry` (since that's all
/// `SourceNumericDocValues` can express) and confirm the merged segment
/// still has the **stale** pre-update value -- i.e. if a caller ever did
/// wire an overlay-writing update path into `IndexWriter` without also
/// teaching the merge call site to resolve "effective" (overlay-folded)
/// values before constructing `MergeSource`, updates would be silently
/// lost across a merge. See `docs/parity.md` for the full
/// scope writeup.
///
/// `IndexWriter` writes doc-values updates as real Lucene generations, not
/// as these overlays (`crate::field_updates`), and `execute_merge` resolves
/// each source's newest generation before constructing the `MergeSource`.
/// `MergeSource` itself reads a field's base column and nothing else.
#[test]
fn numeric_doc_values_update_overlay_does_not_survive_a_merge_today() {
    use lucene_codecs::doc_values_updates;

    let seg0_id = [1u8; ID_LENGTH];
    // Base segment: 2 docs, values [10, 20].
    let dv0 = flush_numeric_dv(0, &[10, 20], seg0_id);

    // Apply a real overlay update: doc 0's value becomes 999.
    let overlay_bytes = doc_values_updates::write_numeric_updates(&[(0, Some(999))], &seg0_id, "");
    let overlay = doc_values_updates::read_numeric_updates(&overlay_bytes, &seg0_id, "")
        .expect("overlay round-trips");

    // Sanity: the overlay-aware read genuinely sees the new value for
    // doc 0 and correctly falls back to the base value for doc 1 --
    // proving the overlay mechanism itself works in isolation.
    assert_eq!(
        doc_values_updates::numeric_value_with_updates(&dv0.entry, &dv0.data, &overlay, 0).unwrap(),
        Some(999)
    );
    assert_eq!(
        doc_values_updates::numeric_value_with_updates(&dv0.entry, &dv0.data, &overlay, 1).unwrap(),
        Some(20)
    );

    // Now actually merge this single source through the real merge
    // entry point. `SourceNumericDocValues` has no overlay field, so the
    // only thing that can be handed to the merge is the base data/entry
    // -- exactly what a hypothetical "apply overlay before merge" caller
    // would need to have already flattened, and exactly what today's
    // codebase has no code path that does.
    let fields = vec![numeric_field("num", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let dv0_source = [dv0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &dv0_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_dv_overlay",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_dv_overlay");

    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_dv_overlay", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_dv_overlay", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![numeric_field("num", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.numeric_entry(0).unwrap();
    let merged_values: Vec<i64> = (0..2)
        .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap().unwrap())
        .collect();

    // The merged segment has the STALE base value (10) for doc 0, not
    // the overlay-updated value (999) -- proving the update was silently
    // lost across this merge, since nothing resolved it before the
    // merge call.
    assert_eq!(merged_values, vec![10, 20]);
    assert_ne!(merged_values[0], 999);
}

/// `DocValuesConsumer.getMergedNumericDocValues` skips a reader with no
/// NUMERIC column for the field, so every one of that reader's documents
/// lands in the merged column as **missing** -- it does not fail the
/// merge.
///
/// The reason it is safe to be permissive here is one level up:
/// [`check_format_coverage`] refuses a merge whose caller never opened a
/// format a source's `.si` lists, so a *dropped* column cannot hide
/// behind this branch any more.
#[test]
fn numeric_doc_values_absent_from_one_source_come_back_missing_not_as_an_error() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv0 = flush_numeric_dv(0, &[10], seg0_id);
    let fields = vec![numeric_field("num", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &dv0_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    // Source 1 has live docs but no numeric doc-values entry at all for
    // field "num": Java's `values == null` sub, i.e. all-missing.
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_dv_sparse",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let seg = per_field_segment("_merged_dv_sparse", DOC_VALUES_FORMAT_NAME);
    let dvm = std::fs::read(tmp.join(format!("{seg}.dvm"))).unwrap();
    let dvd = std::fs::read(tmp.join(format!("{seg}.dvd"))).unwrap();
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos::FieldInfos {
            fields: vec![numeric_field("num", 0)],
        },
    )
    .unwrap();
    let entry = meta
        .numeric_entry(0)
        .expect("the column source 0 has must still be written");
    assert_eq!(
        (0..2)
            .map(|d| doc_values::numeric_value(&dvd, entry, d).unwrap())
            .collect::<Vec<_>>(),
        vec![Some(10), None],
        "source 1's document has no value, rather than the merge failing"
    );
}

#[test]
fn two_numeric_doc_values_fields_land_in_one_merged_dvm() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv_a = flush_numeric_dv(0, &[1], seg0_id);
    let dv_b = flush_numeric_dv(1, &[2], seg0_id);
    let fields = vec![numeric_field("a", 0), numeric_field("b", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let sources_a = dv_a.source();
    let sources_b = dv_b.source();
    let numeric = vec![
        SourceNumericDocValues {
            data: sources_a.data,
            entry: sources_a.entry.clone(),
        },
        SourceNumericDocValues {
            data: sources_b.data,
            entry: sources_b.entry.clone(),
        },
    ];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &numeric,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_dv_two",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    // Both columns are in the one `.dvm`/`.dvd`, each still attached to
    // its own field number -- what a real multi-field
    // `Lucene90DocValuesFormat` segment looks like, and what a
    // multi-tier index sort needs (a second tier is a second column).
    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_dv_two",
        [9u8; ID_LENGTH],
        vec![numeric_field("a", 0), numeric_field("b", 1)],
    );
    assert_eq!(
        doc_values::numeric_value(&dvd, meta.numeric_entry(0).unwrap(), 0).unwrap(),
        Some(1)
    );
    assert_eq!(
        doc_values::numeric_value(&dvd, meta.numeric_entry(1).unwrap(), 0).unwrap(),
        Some(2)
    );
}

/// Reads a merged segment's `.dvm`/`.dvd` back, parsed against a
/// caller-supplied merged field list -- the multi-field analogue of the
/// single-field `read_back_*` helpers, used by the tests that check
/// several doc-values fields landed in the *same* file triple.
fn open_merged_doc_values(
    tmp: &std::path::Path,
    segment_name: &str,
    merged_id: [u8; ID_LENGTH],
    fields: Vec<FieldInfo>,
) -> (Vec<u8>, doc_values::DocValuesMeta) {
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment(segment_name, DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment(segment_name, DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &merged_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos::FieldInfos { fields },
    )
    .unwrap();
    (dvd, meta)
}

// --- binary doc-values merging (mirrors the numeric tests above) ---

#[test]
fn binary_doc_values_merge_across_two_sources_with_deletions() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    // Source 0: 2 docs, doc 1 deleted -> only "aa" survives.
    let dv0 = flush_binary_dv(0, &[b"aa".to_vec(), b"bb".to_vec()], seg0_id);
    // Source 1: 1 doc, no deletions -> "cc" survives.
    let dv1 = flush_binary_dv(0, &[b"cc".to_vec()], seg1_id);

    let fields = vec![binary_field("bin", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let mut live0 = FixedBitSet::new(2);
    live0.set(0); // keep doc 0 ("a"/"aa"), drop doc 1 ("b"/"bb")

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &dv0_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &dv1_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_bdv",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_bdv");

    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_bdv", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_bdv", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![binary_field("bin", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.binary_entry(0).unwrap();
    let values: Vec<Vec<u8>> = (0..2)
        .map(|d| {
            doc_values::binary_value(&dvd, entry, d)
                .unwrap()
                .unwrap()
                .to_vec()
        })
        .collect();
    assert_eq!(values, vec![b"aa".to_vec(), b"cc".to_vec()]);
}

#[test]
fn binary_doc_values_missing_in_a_live_contributing_source_merges_sparse() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv0 = flush_binary_dv(0, &[b"aa".to_vec()], seg0_id);
    let fields = vec![binary_field("bin", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &dv0_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    // Source 1 has live docs but no binary doc-values entry at all for
    // field "bin" -- `DocValuesConsumer`'s "not a sub" case: its
    // documents come out of the merge with no value.
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_bdv_err",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let suffix_seg = per_field_segment("_merged_bdv_err", DOC_VALUES_FORMAT_NAME);
    let dvd = std::fs::read(tmp.join(format!("{suffix_seg}.dvd"))).unwrap();
    let dvm = std::fs::read(tmp.join(format!("{suffix_seg}.dvm"))).unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![binary_field("f", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.binary_entry(0).unwrap();
    let got: Vec<Option<Vec<u8>>> = (0..2)
        .map(|d| {
            doc_values::binary_value(&dvd, entry, d)
                .unwrap()
                .map(<[u8]>::to_vec)
        })
        .collect();
    assert_eq!(got, vec![Some(b"aa".to_vec()), None]);
}

#[test]
fn two_binary_doc_values_fields_land_in_one_merged_dvm() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv_a = flush_binary_dv(0, &[b"x".to_vec()], seg0_id);
    let dv_b = flush_binary_dv(1, &[b"y".to_vec()], seg0_id);
    let fields = vec![binary_field("a", 0), binary_field("b", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let sources_a = dv_a.source();
    let sources_b = dv_b.source();
    let binary = vec![
        SourceBinaryDocValues {
            data: sources_a.data,
            entry: sources_a.entry.clone(),
        },
        SourceBinaryDocValues {
            data: sources_b.data,
            entry: sources_b.entry.clone(),
        },
    ];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &binary,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_bdv_two",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_bdv_two",
        [9u8; ID_LENGTH],
        vec![binary_field("a", 0), binary_field("b", 1)],
    );
    assert_eq!(
        doc_values::binary_value(&dvd, meta.binary_entry(0).unwrap(), 0).unwrap(),
        Some(&b"x"[..])
    );
    assert_eq!(
        doc_values::binary_value(&dvd, meta.binary_entry(1).unwrap(), 0).unwrap(),
        Some(&b"y"[..])
    );
}

// --- SORTED doc-values merging (ordinal remapping via term-bytes resolution) ---

fn sorted_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.doc_values_type = DocValuesType::Sorted;
    f
}

/// Same idea as [`FlushedBinaryDv`], for SORTED doc values -- `values`
/// is one raw term per doc (dense, `values.len() == max_doc`), exactly
/// what [`doc_values::write_single_dense_sorted_field`] takes.
struct FlushedSortedDv {
    data: Vec<u8>,
    entry: SortedEntry,
}

fn flush_sorted_dv(
    field_number: i32,
    values: &[Vec<u8>],
    segment_id: [u8; ID_LENGTH],
) -> FlushedSortedDv {
    let max_doc = values.len() as i32;
    let (meta, data, _skip) = doc_values::write_single_dense_sorted_field(
        field_number,
        values,
        max_doc,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    )
    .unwrap();
    let field_infos = field_infos::FieldInfos {
        fields: vec![sorted_field("x", field_number)],
    };
    let (_version, parsed) = doc_values::parse_meta(
        &meta,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = parsed.sorted_entry(field_number).unwrap().clone();
    FlushedSortedDv { data, entry }
}

impl FlushedSortedDv {
    fn source(&self) -> SourceSortedDocValues<'_> {
        SourceSortedDocValues {
            data: &self.data,
            entry: self.entry.clone(),
        }
    }
}

/// Resolves every doc's merged SORTED term, doc by doc, through the
/// *unmodified* reader stack (`parse_meta` + `sorted_ord` +
/// `terms_dict::decode_all_terms`) -- the critical correctness check:
/// not just "some valid ordinal", but the actual right term bytes per
/// doc, read back exactly as a real caller would.
fn read_back_sorted_terms(
    dvm: &[u8],
    dvd: &[u8],
    field_number: i32,
    doc_count: i32,
) -> Vec<Vec<u8>> {
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_field("x", field_number)],
    };
    let (_v, meta) = doc_values::parse_meta(
        dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_entry(field_number).unwrap();
    let dict = terms_dict::decode_all_terms(dvd, &entry.terms).unwrap();
    (0..doc_count)
        .map(|d| {
            let ord = doc_values::sorted_ord(dvd, entry, d).unwrap().unwrap();
            dict[ord as usize].clone()
        })
        .collect()
}

#[test]
fn sorted_doc_values_merge_with_overlapping_terms_dedupes_into_one_shared_dictionary_entry() {
    // Source 0: docs "red", "blue"; source 1: docs "red", "green" -- both
    // sources independently assign "red" ordinal 0 (it's the
    // alphabetically-first of each source's own two-term dictionary).
    // Real bug case: if this merge naively concatenated ordinals without
    // resolving to bytes, source 1's "red" (ordinal 0 in its own dict)
    // could get merged as a *different* dictionary entry than source 0's
    // "red" (also ordinal 0) purely because they came from different
    // sources -- this test would catch that by checking actual resolved
    // term bytes, not just ordinal counts.
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv0 = flush_sorted_dv(0, &[b"red".to_vec(), b"blue".to_vec()], seg0_id);
    let dv1 = flush_sorted_dv(0, &[b"red".to_vec(), b"green".to_vec()], seg1_id);

    let fields = vec![sorted_field("color", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &fields,
        &[doc_with(0, "c"), doc_with(0, "d")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv0_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv1_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let tmp_dir = FsDirectory::open(&tmp);
    merge_stored_only_segments(
        &tmp_dir,
        &[source0, source1],
        "_merged_sorted_overlap",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sorted_overlap", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sorted_overlap", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_field("color", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_entry(0).unwrap();
    // "red" is shared across both sources -- the merged dictionary must
    // dedupe it into exactly one entry, so the distinct dictionary size
    // is 3 ("red", "blue", "green"), not 4.
    assert_eq!(entry.terms.terms_dict_size, 3);

    // And every doc must resolve to the RIGHT term, not just any valid
    // ordinal -- this is the actual correctness check.
    let terms = read_back_sorted_terms(&dvm, &dvd, 0, 4);
    assert_eq!(
        terms,
        vec![
            b"red".to_vec(),
            b"blue".to_vec(),
            b"red".to_vec(),
            b"green".to_vec(),
        ]
    );
}

#[test]
fn sorted_doc_values_merge_with_disjoint_terms_contains_all_terms_from_both_sources() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv0 = flush_sorted_dv(0, &[b"apple".to_vec()], seg0_id);
    let dv1 = flush_sorted_dv(0, &[b"zebra".to_vec()], seg1_id);

    let fields = vec![sorted_field("word", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv0_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv1_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_disjoint",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sorted_disjoint", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sorted_disjoint", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let terms = read_back_sorted_terms(&dvm, &dvd, 0, 2);
    assert_eq!(terms, vec![b"apple".to_vec(), b"zebra".to_vec()]);
}

#[test]
fn sorted_doc_values_missing_in_a_live_contributing_source_merges_sparse() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv0 = flush_sorted_dv(0, &[b"x".to_vec()], seg0_id);
    let fields = vec![sorted_field("word", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv0_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    // Source 1 has live docs but no SORTED doc-values entry at all for
    // field "word".
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_err",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let suffix_seg = per_field_segment("_merged_sorted_err", DOC_VALUES_FORMAT_NAME);
    let dvd = std::fs::read(tmp.join(format!("{suffix_seg}.dvd"))).unwrap();
    let dvm = std::fs::read(tmp.join(format!("{suffix_seg}.dvm"))).unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_field("f", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_entry(0).unwrap();
    let got: Vec<Option<i64>> = (0..2)
        .map(|d| doc_values::sorted_ord(&dvd, entry, d).unwrap())
        .collect();
    assert_eq!(got, vec![Some(0), None]);
    assert_eq!(
        terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap(),
        vec![b"x".to_vec()]
    );
}

#[test]
fn two_sorted_doc_values_fields_land_in_one_merged_dvm() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv_a = flush_sorted_dv(0, &[b"x".to_vec()], seg0_id);
    let dv_b = flush_sorted_dv(1, &[b"y".to_vec()], seg0_id);
    let fields = vec![sorted_field("a", 0), sorted_field("b", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let sources_a = dv_a.source();
    let sources_b = dv_b.source();
    let sorted = vec![
        SourceSortedDocValues {
            data: sources_a.data,
            entry: sources_a.entry.clone(),
        },
        SourceSortedDocValues {
            data: sources_b.data,
            entry: sources_b.entry.clone(),
        },
    ];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &sorted,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_sdv_two",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    // Each field rebuilds its *own* merged dictionary, so ordinal 0 of
    // field 0 and ordinal 0 of field 1 are different terms.
    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_sdv_two",
        [9u8; ID_LENGTH],
        vec![sorted_field("a", 0), sorted_field("b", 1)],
    );
    for (field_number, expected) in [(0, &b"x"[..]), (1, &b"y"[..])] {
        let entry = meta.sorted_entry(field_number).unwrap();
        let ord = doc_values::sorted_ord(&dvd, entry, 0).unwrap().unwrap();
        let dict = terms_dict::decode_all_terms(&dvd, &entry.terms).unwrap();
        assert_eq!(dict[ord as usize], expected);
    }
}

#[test]
fn numeric_and_binary_doc_values_share_one_merged_dvm() {
    // Doc-values *types* mix freely in one segment: every merged field
    // goes into the same `.dvm`/`.dvd`/`.dvs` through
    // `doc_values::write_dense_fields`, which is what a real
    // `Lucene90DocValuesFormat` segment is.
    let seg0_id = [1u8; ID_LENGTH];
    let numeric_dv = flush_numeric_dv(0, &[1], seg0_id);
    let binary_dv = flush_binary_dv(1, &[b"v".to_vec()], seg0_id);
    let fields = vec![numeric_field("num", 0), binary_field("bin", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let numeric_source = [numeric_dv.source()];
    let binary_source = [binary_dv.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &numeric_source,
        binary_doc_values: &binary_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_mixed_dv",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_mixed_dv",
        [9u8; ID_LENGTH],
        vec![numeric_field("num", 0), binary_field("bin", 1)],
    );
    assert_eq!(
        doc_values::numeric_value(&dvd, meta.numeric_entry(0).unwrap(), 0).unwrap(),
        Some(1)
    );
    assert_eq!(
        doc_values::binary_value(&dvd, meta.binary_entry(1).unwrap(), 0).unwrap(),
        Some(&b"v"[..])
    );
}

#[test]
fn sorted_numeric_and_sorted_set_doc_values_share_one_merged_dvm() {
    // Same rule as `numeric_and_binary_doc_values_share_one_merged_dvm`,
    // exercised for the pair whose readers are the most intertwined:
    // a SORTED_SET field that collapsed to one value per doc is stored
    // through the very same layout a SORTED_NUMERIC field uses, so
    // getting the two into one `.dvm` without crossing their entries is
    // worth its own case.
    let seg0_id = [1u8; ID_LENGTH];
    let sorted_numeric_dv = flush_sorted_numeric_dv(0, &[vec![1]], seg0_id);
    let sorted_set_dv = flush_sorted_set_dv(1, &[vec![b"v".to_vec()]], seg0_id);
    let fields = vec![sorted_numeric_field("num", 0), sorted_set_field("set", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let sorted_numeric_source = [sorted_numeric_dv.source()];
    let sorted_set_source = [sorted_set_dv.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &sorted_numeric_source,
        sorted_set_doc_values: &sorted_set_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_mixed_dv2",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_mixed_dv2",
        [9u8; ID_LENGTH],
        vec![sorted_numeric_field("num", 0), sorted_set_field("set", 1)],
    );
    assert_eq!(
        doc_values::sorted_numeric_values(&dvd, meta.sorted_numeric_entry(0).unwrap(), 0).unwrap(),
        vec![1]
    );
    let set_entry = meta.sorted_set_entry(1).unwrap();
    let ords = sorted_set_doc_ordinals(&dvd, set_entry, 0).unwrap();
    let dict = sorted_set_source_dict(&dvd, set_entry).unwrap();
    assert_eq!(
        ords.iter()
            .map(|&o| dict[o as usize].clone())
            .collect::<Vec<_>>(),
        vec![b"v".to_vec()]
    );
}

#[test]
fn a_fully_deleted_sources_unrelated_numeric_field_does_not_trigger_too_many_fields() {
    // Source 0 (live) has numeric-dv field "a"; source 1 is 100% deleted
    // but happens to carry an unrelated numeric-dv field "junk" -- since
    // source 1 contributes zero docs to the merge, its doc-values field
    // must not count toward the "more than one field" limit (regression
    // for a bug where the too-many-fields check ran before the
    // zero-live-docs exemption already applied elsewhere in this
    // module).
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv_a = flush_numeric_dv(0, &[10, 20], seg0_id);
    let dv_junk = flush_numeric_dv(0, &[99], seg1_id);
    let fields0 = vec![numeric_field("a", 0)];
    let fields1 = vec![numeric_field("junk", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields0,
        &[
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                }],
            },
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("y".to_string()),
                }],
            },
        ],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &fields1,
        &[Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("z".to_string()),
            }],
        }],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let numeric0 = vec![dv_a.source()];
    let numeric1 = vec![dv_junk.source()];
    let all_deleted = FixedBitSet::new(1); // source 1: nothing live
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &numeric0,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: Some(&all_deleted),
        numeric_doc_values: &numeric1,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_dv_deleted_unrelated",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let merged_reader = std::fs::read(tmp.join(format!("{}.fdt", sci.segment_name)));
    assert!(
        merged_reader.is_ok(),
        "merge should succeed, not reject on source 1's unrelated deleted-only field"
    );
}

#[test]
fn a_fully_deleted_sources_unrelated_binary_field_does_not_trigger_too_many_fields() {
    // Same regression shape as the NUMERIC version above, for BINARY:
    // a 100%-deleted source's own binary-dv field must not count toward
    // the "more than one field" limit, since it contributes zero live
    // docs to the merge.
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv_a = flush_binary_dv(0, &[b"aa".to_vec(), b"bb".to_vec()], seg0_id);
    let dv_junk = flush_binary_dv(0, &[b"zz".to_vec()], seg1_id);
    let fields0 = vec![binary_field("a", 0)];
    let fields1 = vec![binary_field("junk", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields0,
        &[
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                }],
            },
            Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("y".to_string()),
                }],
            },
        ],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &fields1,
        &[Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("z".to_string()),
            }],
        }],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let binary0 = vec![dv_a.source()];
    let binary1 = vec![dv_junk.source()];
    let all_deleted = FixedBitSet::new(1); // source 1: nothing live
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &binary0,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: Some(&all_deleted),
        numeric_doc_values: &[],
        binary_doc_values: &binary1,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_binary_dv_deleted_unrelated",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let merged_reader = std::fs::read(tmp.join(format!("{}.fdt", sci.segment_name)));
    assert!(
        merged_reader.is_ok(),
        "merge should succeed, not reject on source 1's unrelated deleted-only binary field"
    );
}

#[test]
fn merge_checks_contributing_norms_crc_before_creating_outputs() {
    let id0 = [71; ID_LENGTH];
    let id1 = [72; ID_LENGTH];
    let fields = vec![norms_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        id0,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", id1, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let norms0 = flush_norms(0, &[11, 13], id0);
    let norms1 = flush_norms(0, &[17], id1);
    let mut corrupt = norms0.data.clone();
    // Only the stored CRC changes. Ordinary value reads remain successful,
    // so this case would pass without whole-source checksum validation.
    *corrupt.last_mut().unwrap() ^= 1;
    assert_eq!(
        norms::norm_value(&corrupt, &norms0.entry, 0).unwrap(),
        Some(11)
    );
    assert_eq!(
        norms::norm_value(&corrupt, &norms0.entry, 1).unwrap(),
        Some(13)
    );
    let bad_columns = [SourceNorms {
        data: &corrupt,
        entry: norms0.entry,
    }];
    let mut corrupt1 = norms1.data.clone();
    *corrupt1.last_mut().unwrap() ^= 1;
    let bad_columns1 = [SourceNorms {
        data: &corrupt1,
        entry: norms1.entry,
    }];
    let good_columns0 = [norms0.source()];
    let good_columns1 = [norms1.source()];
    let mut partial_live = FixedBitSet::new(2);
    partial_live.set(1);
    let snapshot = || {
        dir.list_all()
            .unwrap()
            .into_iter()
            .map(|name| {
                let bytes = dir.open(&name).unwrap().to_vec();
                (name, bytes)
            })
            .collect::<Vec<_>>()
    };
    let before = snapshot();
    let corrupt_before = corrupt.clone();
    for (bad_source, live) in [(0, None), (0, Some(&partial_live)), (1, None)] {
        let mut source0 = MergeSource::stored_only(&stored0.fields, &reader0, live, None);
        source0.norms = if bad_source == 0 {
            &bad_columns
        } else {
            &good_columns0
        };
        let mut source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, None);
        source1.norms = if bad_source == 1 {
            &bad_columns1
        } else {
            &good_columns1
        };
        let Err(err) = merge_stored_only_segments(
            &dir,
            &[source0, source1],
            "_bad_norms_crc",
            [73; ID_LENGTH],
            "Lucene104",
            version(),
        ) else {
            panic!("a contributing source's .nvd CRC must be verified");
        };
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(ref message))
                if message.contains("checksum failed")),
            "{err:?}"
        );
        assert_eq!(
            snapshot(),
            before,
            "failed merge created output or changed a source"
        );
        assert_eq!(corrupt, corrupt_before);
    }
    // An entirely deleted source remains intentionally ignored by this
    // public merge API. Another source still contributes an ordinary norm.
    let all_deleted = FixedBitSet::new(2);
    let mut source0 = MergeSource::stored_only(&stored0.fields, &reader0, Some(&all_deleted), None);
    source0.norms = &bad_columns;
    let mut source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, None);
    source1.norms = &good_columns1;
    let ignored = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_ignored_norms_crc",
        [74; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let si = dir.open(&format!("{}.si", ignored.segment_name)).unwrap();
    assert_eq!(
        segment_info::parse(&si, &[74; ID_LENGTH])
            .unwrap()
            .doc_count,
        1
    );
    let nvm = dir.open("_ignored_norms_crc.nvm").unwrap().to_vec();
    let nvd = dir.open("_ignored_norms_crc.nvd").unwrap().to_vec();
    let (_, meta) = norms::parse_meta(&nvm, &[74; ID_LENGTH], "").unwrap();
    assert_eq!(
        norms::norm_value(&nvd, meta.entry(0).unwrap(), 0).unwrap(),
        Some(17)
    );

    // Supplying restored original bytes succeeds on the same inputs after
    // the failed attempts and preserves every surviving source value.
    let mut source0 = MergeSource::stored_only(&stored0.fields, &reader0, None, None);
    source0.norms = &good_columns0;
    let mut source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, None);
    source1.norms = &good_columns1;
    let restored = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_restored_norms_crc",
        [75; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let si = dir.open(&format!("{}.si", restored.segment_name)).unwrap();
    assert_eq!(
        segment_info::parse(&si, &[75; ID_LENGTH])
            .unwrap()
            .doc_count,
        3
    );
    let nvm = dir.open("_restored_norms_crc.nvm").unwrap().to_vec();
    let nvd = dir.open("_restored_norms_crc.nvd").unwrap().to_vec();
    let (_, meta) = norms::parse_meta(&nvm, &[75; ID_LENGTH], "").unwrap();
    let values: Vec<_> = (0..3)
        .map(|doc| norms::norm_value(&nvd, meta.entry(0).unwrap(), doc).unwrap())
        .collect();
    assert_eq!(values, [Some(11), Some(13), Some(17)]);
}

#[test]
fn merge_norms_reuses_sparse_readers_across_forward_interleaved_and_backward_visits() {
    let id0 = [1u8; ID_LENGTH];
    let id1 = [2u8; ID_LENGTH];
    let id2 = [3u8; ID_LENGTH];
    let fields0 = vec![norms_field("body", 0)];
    let fields1 = vec![norms_field("body", 7)];
    let fields2 = vec![field("other", 2)];
    let (meta0, data0) =
        norms::write_single_sparse_field(0, &[(0, 0), (2, 13)], 4, &id0, "").unwrap();
    let (meta1, data1) =
        norms::write_single_sparse_field(7, &[(1, 21), (3, 23)], 4, &id1, "").unwrap();
    let (_, parsed0) = norms::parse_meta(&meta0, &id0, "").unwrap();
    let (_, parsed1) = norms::parse_meta(&meta1, &id1, "").unwrap();
    let norms0 = [SourceNorms {
        data: &data0,
        entry: *parsed0.entry(0).unwrap(),
    }];
    let norms1 = [SourceNorms {
        data: &data1,
        entry: *parsed1.entry(7).unwrap(),
    }];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let docs0: Vec<_> = (0..4).map(|_| doc_with(0, "a")).collect();
    let docs1: Vec<_> = (0..4).map(|_| doc_with(7, "b")).collect();
    let docs2: Vec<_> = (0..4).map(|_| doc_with(2, "c")).collect();
    let stored0 = flush(&dir, &tmp, "_0", id0, &fields0, &docs0);
    let stored1 = flush(&dir, &tmp, "_1", id1, &fields1, &docs1);
    let stored2 = flush(&dir, &tmp, "_2", id2, &fields2, &docs2);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let reader2 = open_reader(&stored2);
    let mut source0 = MergeSource::stored_only(&stored0.fields, &reader0, None, None);
    source0.norms = &norms0;
    let mut source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, None);
    source1.norms = &norms1;
    let source2 = MergeSource::stored_only(&stored2.fields, &reader2, None, None);
    let sources = [source0, source1, source2];
    let maps = [
        HashMap::from([(0, 9)]),
        HashMap::from([(7, 9)]),
        HashMap::from([(2, 10)]),
    ];
    let live_ids = vec![vec![0, 1, 2, 3]; 3];
    let merged_fields = [norms_field("body", 9)];
    let expected_by_source = [
        [Some(0), None, Some(13), None],
        [None, Some(21), None, Some(23)],
        [None, None, None, None],
    ];
    let sequential = concat_doc_order(&live_ids);
    // A sorted merge interleaves sources while each source walks forward.
    let interleaved: Vec<_> = (0..4)
        .flat_map(|doc| (0..3).map(move |source| (source, doc)))
        .collect();
    // A merge reordering hook can visit a source behind its previous doc.
    let backward = vec![
        (0, 2),
        (1, 3),
        (0, 0),
        (1, 1),
        (2, 3),
        (0, 3),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 2),
        (2, 2),
        (2, 1),
    ];
    for order in [sequential, interleaved, backward] {
        let reverse = invert_field_number_maps(&maps);
        let merged = merge_norms(&sources, &merged_fields, &reverse, &live_ids, &order).unwrap();
        let expected: Vec<_> = order
            .iter()
            .map(|&(source, doc)| expected_by_source[source][doc as usize])
            .collect();
        let merged: Vec<_> = merged
            .iter()
            .map(|m| (m.field_number, m.options()))
            .collect();
        assert_eq!(merged, [(9, expected)], "order={order:?}");
    }
}

/// A sparse norm column across sources: source 0 has a gap at one
/// document, and source 1 never declares the field at all.
///
/// `NormsConsumer.merge` iterates the **merged** `FieldInfos` and
/// `mergeNormsField` adds a sub only for a source whose own `FieldInfo`
/// says `hasNorms()`, so both are gaps in the merged column, not errors.
#[test]
fn merged_norms_are_sparse_across_a_gap_and_a_source_that_never_declared_the_field() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    // Source 0: docs 0 and 2 have a norm, doc 1 does not.
    let (meta0, data0) =
        norms::write_single_sparse_field(0, &[(0, 11), (2, 13)], 3, &seg0_id, "").unwrap();
    let (_v, parsed0) = norms::parse_meta(&meta0, &seg0_id, "").unwrap();
    let norms0 = FlushedNorms {
        data: data0,
        entry: *parsed0.entry(0).unwrap(),
    };

    let with_norms = vec![norms_field("body", 0)];
    // Source 1 never declares `body` at all -- `mergeNormsField`'s
    // `readerFieldInfo == null` branch, and the ordinary shape of a
    // schema that grew. (Two sources that *disagree* about `omitNorms`
    // for the same field are a different thing and are refused by
    // `verify_same_schema`, exactly as `FieldInfos.verifySameOmitNorms`
    // refuses them.)
    let without_norms = vec![field("other", 0)];

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &with_norms,
        &[doc_with(0, "a"), doc_with(0, "b"), doc_with(0, "c")],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &without_norms,
        &[doc_with(0, "d")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let norms0_source = [norms0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms0_source,
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sparse_norms",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let nvd = std::fs::read(tmp.join("_merged_sparse_norms.nvd")).unwrap();
    let nvm = std::fs::read(tmp.join("_merged_sparse_norms.nvm")).unwrap();
    let (_v, parsed) = norms::parse_meta(&nvm, &[9u8; ID_LENGTH], "").unwrap();
    let entry = parsed.entry(0).unwrap();
    assert!(!entry.is_dense(), "the merged column must be sparse");
    let values: Vec<Option<i64>> = (0..4)
        .map(|d| norms::norm_value(&nvd, entry, d).unwrap())
        .collect();
    assert_eq!(
        values,
        vec![Some(11), None, Some(13), None],
        "doc 1 is source 0's gap, doc 3 is the source that never declared the field"
    );

    // And the merged `.fnm` still claims the norms the column backs.
    let fnm = std::fs::read(tmp.join("_merged_sparse_norms.fnm")).unwrap();
    let merged = field_infos::parse(&fnm, &[9u8; ID_LENGTH], "").unwrap();
    let body = merged.fields.iter().find(|f| f.name == "body").unwrap();
    assert!(!body.omit_norms);
}

#[test]
fn norms_merge_across_two_sources_with_deletions() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let norms0 = flush_norms(0, &[1, 2], seg0_id);
    let norms1 = flush_norms(0, &[3], seg1_id);

    let fields = vec![norms_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let mut live0 = FixedBitSet::new(2);
    live0.set(1); // drop doc 0 ("a"/1), keep doc 1 ("b"/2)

    let norms0_source = [norms0.source()];
    let norms1_source = [norms1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms0_source,
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms1_source,
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_norms",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let nvd = std::fs::read(tmp.join("_merged_norms.nvd")).unwrap();
    let nvm = std::fs::read(tmp.join("_merged_norms.nvm")).unwrap();
    let (_v, parsed) = norms::parse_meta(&nvm, &[9u8; ID_LENGTH], "").unwrap();
    let entry = parsed.entry(0).unwrap();
    let values: Vec<i64> = (0..2)
        .map(|d| norms::norm_value(&nvd, entry, d).unwrap().unwrap())
        .collect();
    assert_eq!(values, vec![2, 3]);
}

#[test]
fn term_vectors_merge_across_two_sources_with_deletions_and_a_source_with_none() {
    // Source 1 contributes a live doc but has no term-vectors reader at
    // all, while source 0 does. Real Lucene's `TermVectorsWriter.merge`
    // handles exactly this (`mergeState.termVectorsReaders[i]` may be
    // null -> `vectors = null` -> `addAllDocVectors(null, ...)` writes a
    // vector-less document), which is what happens when term vectors
    // are turned on for an index that already has segments. So source
    // 1's doc must come through as an empty term-vectors document, and
    // source 0's surviving doc must keep its vectors.
    let seg0_id = [1u8; ID_LENGTH];
    // Source 0: 2 docs, both with a term-vectors field 0 ("id"->0).
    let tv0 = flush_term_vectors(&[tv_doc(0, &[("a", 0)]), tv_doc(0, &[("b", 0)])], seg0_id);
    let fields0 = vec![tv_field("id", 0)];
    let fields1 = vec![tv_field("id", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields0,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields1,
        &[doc_with(0, "c")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let tv0_reader = tv0.reader();

    let mut live0 = FixedBitSet::new(2);
    live0.set(1); // drop doc 0, keep doc 1 ("b")

    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: Some(&tv0_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_tv",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let tvd = std::fs::read(tmp.join("_merged_tv.tvd")).unwrap();
    let tvx = std::fs::read(tmp.join("_merged_tv.tvx")).unwrap();
    let tvm = std::fs::read(tmp.join("_merged_tv.tvm")).unwrap();
    let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(merged.max_doc(), 2);

    // Merged doc 0 == source 0's surviving doc ("b"), vectors intact.
    let doc0 = merged.document(0).unwrap().unwrap();
    assert_eq!(doc0.fields.len(), 1);
    assert_eq!(doc0.fields[0].field_number, 0);
    assert_eq!(doc0.fields[0].terms[0].term, b"b".to_vec());

    // Merged doc 1 == source 1's doc, which had no term vectors: an
    // empty term-vectors document, not an error.
    let doc1 = merged.document(1).unwrap();
    assert!(
        doc1.is_none() || doc1.as_ref().unwrap().fields.is_empty(),
        "a source without term vectors contributes a vector-less doc: {doc1:?}"
    );
}

/// Builds a single-field, single-term term-vector document with
/// POSITIONS+OFFSETS+PAYLOADS all populated -- unlike [`tv_doc`] (which
/// only exercises positions), this is the shape
/// [`term_vectors_merge_carries_offsets_and_payloads_through`] needs to
/// prove offsets/payloads survive [`write_merged_term_vectors`] unchanged.
fn tv_doc_with_offsets_and_payloads(
    field_number: i32,
    term: &str,
    position: i32,
    start_offset: i32,
    end_offset: i32,
    payload: &[u8],
) -> TermVectorsDocument {
    TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number,
            has_positions: true,
            has_offsets: true,
            has_payloads: true,
            terms: vec![TermVectorTerm {
                term: term.as_bytes().to_vec(),
                freq: 1,
                positions: Some(vec![position]),
                start_offsets: Some(vec![start_offset]),
                end_offsets: Some(vec![end_offset]),
                payloads: Some(vec![payload.to_vec()]),
            }],
        }],
    }
}

// ---------------------------------------------------------------
// `Lucene90CompressingTermVectorsWriter.merge`'s bulk path (`c8`).
// ---------------------------------------------------------------

/// Enough documents to span several 128-document term-vector chunks, so
/// the bulk path has whole chunks to copy and a ragged dirty tail to
/// re-encode.
const TV_BULK_DOCS: usize = 300;

fn tv_bulk_docs(tag: &str) -> Vec<TermVectorsDocument> {
    (0..TV_BULK_DOCS)
        .map(|n| tv_doc(0, &[(&format!("{tag}{n:04}"), (n % 5) as i32)]))
        .collect()
}

fn tv_bulk_stored(tag: &str) -> Vec<Document> {
    (0..TV_BULK_DOCS)
        .map(|n| doc_with(0, &format!("{tag}{n:04}")))
        .collect()
}

/// Merges two term-vector-bearing sources and returns the merged
/// `(tvd, tvx, tvm)`; `live` and `renumber` control which merge strategy
/// each source ends up on.
#[allow(clippy::type_complexity)]
fn merge_two_tv_sources(
    tmp: &std::path::Path,
    live1: Option<&FixedBitSet>,
    renumber_source_1: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let dir = FsDirectory::open(tmp);
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let tv0 = flush_term_vectors(&tv_bulk_docs("a"), seg0_id);
    let tv1 = flush_term_vectors(&tv_bulk_docs("b"), seg1_id);
    let fields0 = vec![tv_field("body", 0)];
    // A source whose field number the merge has to remap is not a
    // `MatchingReaders` match, so it cannot be bulk-copied.
    let fields1 = if renumber_source_1 {
        vec![tv_field("other", 0), tv_field("body", 1)]
    } else {
        vec![tv_field("body", 0)]
    };
    let stored0 = flush(&dir, tmp, "_0", seg0_id, &fields0, &tv_bulk_stored("a"));
    let stored1 = flush(&dir, tmp, "_1", seg1_id, &fields1, &tv_bulk_stored("b"));
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let tv0_reader = tv0.reader();
    let tv1_reader = tv1.reader();
    let source0 = MergeSource {
        term_vectors: Some(&tv0_reader),
        ..MergeSource::stored_only(&stored0.fields, &reader0, None, Some(version()))
    };
    let source1 = MergeSource {
        term_vectors: Some(&tv1_reader),
        ..MergeSource::stored_only(&stored1.fields, &reader1, live1, Some(version()))
    };
    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_m",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let read = |ext: &str| std::fs::read(tmp.join(format!("_m.{ext}"))).unwrap();
    (read("tvd"), read("tvx"), read("tvm"))
}

#[test]
fn matching_deletion_free_term_vector_sources_are_bulk_copied_verbatim() {
    let tmp = tempdir();
    let (tvd, tvx, tvm) = merge_two_tv_sources(&tmp, None, false);
    let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(merged.max_doc(), 2 * TV_BULK_DOCS as i32);

    // Each source contributed two whole 128-doc chunks copied verbatim
    // plus its own 44-doc dirty tail, and the tails came across *as*
    // dirty chunks -- which is the observable signature of a byte copy
    // (a re-encoding merge would have coalesced all 600 documents into
    // 4 clean chunks and one dirty tail).
    assert_eq!(merged.num_chunks(), 6);
    assert_eq!(merged.num_dirty_chunks(), 2);
    assert_eq!(merged.num_dirty_docs(), 2 * 44);

    // ... and every document still reads back, in order, with its own
    // vectors.
    let expected: Vec<TermVectorsDocument> = tv_bulk_docs("a")
        .into_iter()
        .chain(tv_bulk_docs("b"))
        .collect();
    for (i, want) in expected.iter().enumerate() {
        let got = merged.document(i as i32).unwrap().unwrap();
        assert_eq!(&got.fields[0].terms, &want.fields[0].terms, "doc {i}");
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn a_term_vector_source_with_deletions_is_re_encoded_not_copied() {
    let tmp = tempdir();
    let mut live1 = FixedBitSet::new(TV_BULK_DOCS);
    for i in 0..TV_BULK_DOCS {
        if !i.is_multiple_of(3) {
            live1.set(i);
        }
    }
    let surviving1 = TV_BULK_DOCS - TV_BULK_DOCS.div_ceil(3);
    let (tvd, tvx, tvm) = merge_two_tv_sources(&tmp, Some(&live1), false);
    let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(merged.max_doc(), (TV_BULK_DOCS + surviving1) as i32);

    // Source 0 is still bulk-copied (2 clean chunks + its dirty tail);
    // source 1's surviving documents are re-encoded, so they pack into
    // fresh full chunks with exactly one dirty tail at the very end.
    assert_eq!(merged.num_dirty_chunks(), 2);

    let src0 = tv_bulk_docs("a");
    let src1 = tv_bulk_docs("b");
    for (i, want) in src0.iter().enumerate() {
        let got = merged.document(i as i32).unwrap().unwrap();
        assert_eq!(&got.fields[0].terms, &want.fields[0].terms, "doc {i}");
    }
    let mut merged_id = TV_BULK_DOCS as i32;
    for (i, want) in src1.iter().enumerate() {
        if !live1.get(i) {
            continue;
        }
        let got = merged.document(merged_id).unwrap().unwrap();
        assert_eq!(
            &got.fields[0].terms, &want.fields[0].terms,
            "source 1 doc {i}"
        );
        merged_id += 1;
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn a_renumbered_term_vector_source_is_re_encoded_with_remapped_field_numbers() {
    let tmp = tempdir();
    let (tvd, tvx, tvm) = merge_two_tv_sources(&tmp, None, true);
    let merged = term_vectors::open(&tvd, &tvx, &tvm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(merged.max_doc(), 2 * TV_BULK_DOCS as i32);
    // `reconcile_field_numbers` keeps source 0's "body" at 0 and gives
    // source 1's "other" the next free number, 1 -- so source 1's map is
    // {0 -> 1, 1 -> 0}, not the identity, and `MatchingReaders` rules it
    // out of the bulk path. Its term-vector data sits on its own field 0
    // ("other"), which must come through as merged field 1. A bulk copy
    // would have carried the source's own number through verbatim and
    // put every one of those vectors under the wrong field name -- the
    // exact failure Java's "bulk merge is scary" comment is about, so
    // check it for *every* document, not just the first.
    for i in 0..TV_BULK_DOCS {
        let got = merged.document((TV_BULK_DOCS + i) as i32).unwrap().unwrap();
        assert_eq!(got.fields[0].field_number, 1, "source 1 doc {i}");
        assert_eq!(
            got.fields[0].terms[0].term,
            format!("b{i:04}").into_bytes(),
            "source 1 doc {i}"
        );
    }
    // Source 0 *is* a match, so it is still bulk-copied: its own dirty
    // tail chunk comes across as dirty, alongside the merge's own final
    // flush.
    assert_eq!(merged.num_dirty_chunks(), 2);
    std::fs::remove_dir_all(&tmp).ok();
}

#[derive(Debug, Default)]
struct InstalledTvMergeCalls {
    opens: std::sync::atomic::AtomicUsize,
    closes: std::sync::atomic::AtomicUsize,
}

#[derive(Debug, Default)]
struct CountingInstalledTvFactory(std::sync::Arc<InstalledTvMergeCalls>);

impl lucene_codecs::term_vectors_writer::TermVectorsWriterFactory for CountingInstalledTvFactory {
    fn vectors_writer<'a>(
        &self,
        state: lucene_codecs::term_vectors_writer::TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn lucene_codecs::term_vectors_writer::TermVectorsWriteSink + 'a>>
    {
        self.0
            .opens
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Box::new(CountingInstalledTvSink {
            native: lucene_codecs::term_vectors_writer::NativeTermVectorsWriter::new(state)?,
            calls: self.0.clone(),
        }))
    }
}

struct CountingInstalledTvSink<'a> {
    native: lucene_codecs::term_vectors_writer::NativeTermVectorsWriter<'a>,
    calls: std::sync::Arc<InstalledTvMergeCalls>,
}

impl lucene_codecs::term_vectors_writer::TermVectorsWriteSink for CountingInstalledTvSink<'_> {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
        self.native.start_document(fields)
    }
    fn finish_document(&mut self) -> term_vectors::Result<()> {
        self.native.finish_document()
    }
    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        positions: bool,
        offsets: bool,
        payloads: bool,
    ) -> term_vectors::Result<()> {
        self.native
            .start_field(info, terms, positions, offsets, payloads)
    }
    fn finish_field(&mut self) -> term_vectors::Result<()> {
        self.native.finish_field()
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
        self.native.start_term(term, frequency)
    }
    fn finish_term(&mut self) -> term_vectors::Result<()> {
        self.native.finish_term()
    }
    fn add_position(
        &mut self,
        position: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        self.native.add_position(position, start, end, payload)
    }
    fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
        self.native.finish(documents)
    }
    fn close(&mut self) -> term_vectors::Result<()> {
        self.calls
            .closes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.native.close()
    }
    fn ram_bytes_used(&self) -> usize {
        self.native.ram_bytes_used()
    }
}

fn installed_tv_directory_snapshot(dir: &dyn Directory) -> Vec<(String, Vec<u8>)> {
    let mut names = dir.list_all().unwrap();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let bytes = dir.open(&name).unwrap().to_vec();
            (name, bytes)
        })
        .collect()
}

#[test]
fn installed_term_vectors_check_fully_deleted_source_crc_before_factory_acquisition() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let source_id = [81; ID_LENGTH];
    let output_id = [82; ID_LENGTH];
    let infos = vec![tv_field("body", 0)];
    let stored = flush(
        &dir,
        &tmp,
        "_source",
        source_id,
        &infos,
        &[doc_with(0, "x")],
    );
    let stored_reader = open_reader(&stored);
    let original = flush_term_vectors(&[tv_doc(0, &[("x", 0)])], source_id);
    let mut bad = original.tvd.clone();
    // Change only the stored checksum. Structural open and document decode
    // succeed; integrity checking must reject it even with no live docs.
    *bad.last_mut().unwrap() ^= 1;
    let vectors = term_vectors::open(&bad, &original.tvx, &original.tvm, &source_id, "").unwrap();
    assert_eq!(vectors.document(0).unwrap(), Some(tv_doc(0, &[("x", 0)])));
    let deleted = FixedBitSet::new(1);
    let source = MergeSource {
        term_vectors: Some(&vectors),
        ..MergeSource::stored_only(
            &stored.fields,
            &stored_reader,
            Some(&deleted),
            Some(version()),
        )
    };
    // A colliding name must not hide the CRC failure or become owned by
    // an operation that must never acquire its native output writer.
    let mut collision = dir.create_output("_crc.tvm").unwrap();
    collision.write_bytes(b"preexisting metadata belongs to another attempt");
    collision.close().unwrap();
    let before = installed_tv_directory_snapshot(&dir);
    let maps = [HashMap::from([(0, 0)])];
    let factory = CountingInstalledTvFactory::default();
    let error = write_installed_term_vectors(
        &factory,
        lucene_codecs::term_vectors_writer::TermVectorsWriteState {
            directory: &dir,
            segment: "_crc",
            id: &output_id,
            suffix: "",
            max_doc: 0,
            format: None,
        },
        &[source],
        &maps,
        &[],
        &infos,
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::TermVectors(term_vectors::Error::Store(lucene_store::Error::Corrupted(ref message)))
            if message.contains("checksum failed")),
        "{error:?}"
    );
    assert_eq!(factory.0.opens.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        factory.0.closes.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(installed_tv_directory_snapshot(&dir), before);
    assert_eq!(bad.len(), original.tvd.len());

    // The same fully deleted source succeeds after restoring its checksum.
    // A zero-document output still runs finish/close on the native writer.
    dir.delete_file("_crc.tvm").unwrap();
    let good_vectors = original.reader();
    let restored = MergeSource {
        term_vectors: Some(&good_vectors),
        ..MergeSource::stored_only(
            &stored.fields,
            &stored_reader,
            Some(&deleted),
            Some(version()),
        )
    };
    let files = write_installed_term_vectors(
        &factory,
        lucene_codecs::term_vectors_writer::TermVectorsWriteState {
            directory: &dir,
            segment: "_crc",
            id: &output_id,
            suffix: "",
            max_doc: 0,
            format: None,
        },
        &[restored],
        &maps,
        &[],
        &infos,
    )
    .unwrap();
    assert_eq!(files, ["_crc.tvd", "_crc.tvm", "_crc.tvx"]);
    assert_eq!(factory.0.opens.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        factory.0.closes.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let data = dir.open("_crc.tvd").unwrap();
    let index = dir.open("_crc.tvx").unwrap();
    let meta = dir.open("_crc.tvm").unwrap();
    let reader = term_vectors::open(&data, &index, &meta, &output_id, "").unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(reader.max_doc(), 0);
}

#[test]
fn installed_term_vectors_unknown_source_field_keeps_typed_error_after_acquire_and_close() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let source_id = [83; ID_LENGTH];
    let output_id = [84; ID_LENGTH];
    let infos = vec![tv_field("body", 0)];
    let stored = flush(
        &dir,
        &tmp,
        "_source",
        source_id,
        &infos,
        &[doc_with(0, "x")],
    );
    let stored_reader = open_reader(&stored);
    // Native bytes have a valid CRC but reference field 7, absent from the
    // source FieldInfos and its reconciled map. This is a source-mapping
    // error, not write_document's merged-field Corrupted error.
    let encoded = flush_term_vectors(&[tv_doc(7, &[("x", 0)])], source_id);
    let vectors = encoded.reader();
    vectors.check_integrity().unwrap();
    assert_eq!(
        vectors.document(0).unwrap().unwrap().fields[0].field_number,
        7
    );
    let source = MergeSource {
        term_vectors: Some(&vectors),
        ..MergeSource::stored_only(&stored.fields, &stored_reader, None, Some(version()))
    };
    let mut unrelated = dir.create_output("_other.tvd").unwrap();
    unrelated.write_bytes(b"unrelated native candidate");
    unrelated.close().unwrap();
    let before = installed_tv_directory_snapshot(&dir);
    let (merged, maps) = reconcile_field_numbers(&[&stored.fields]).unwrap();
    let factory = CountingInstalledTvFactory::default();
    let error = write_installed_term_vectors(
        &factory,
        lucene_codecs::term_vectors_writer::TermVectorsWriteState {
            directory: &dir,
            segment: "_unknown",
            id: &output_id,
            suffix: "",
            max_doc: 1,
            format: None,
        },
        &[source],
        &maps,
        &[(0, 0)],
        &merged,
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::UnknownSourceFieldNumber { field_number: 7 }),
        "{error:?}"
    );
    assert_eq!(factory.0.opens.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        factory.0.closes.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert_eq!(installed_tv_directory_snapshot(&dir), before);
    for extension in ["tvm", "tvd", "tvx"] {
        assert!(dir.open(&format!("_unknown.{extension}")).is_err());
    }
}

#[test]
fn a_term_vector_source_whose_tvd_fails_its_own_checksum_is_never_bulk_copied() {
    // The hazard behind Java's "bulk merge is scary" comment: the bulk
    // path copies compressed bytes verbatim and then writes a freshly
    // computed, valid footer over them, so without
    // `reader.checkIntegrity(...)` a corrupt source would be laundered
    // into a merged segment that passes every checksum from then on.
    // The flipped byte leaves every length, pointer and footer field
    // intact -- exactly the corruption `retrieve_checksum` cannot see.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let seg0_id = [1u8; ID_LENGTH];
    let mut tv0 = flush_term_vectors(&tv_bulk_docs("a"), seg0_id);
    let victim = tv0.tvd.len() - lucene_store::codec_util::FOOTER_LENGTH - 20;
    tv0.tvd[victim] ^= 0x40;
    let fields0 = vec![tv_field("body", 0)];
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &tv_bulk_stored("a"));
    let reader0 = open_reader(&stored0);
    let tv0_reader = tv0.reader();
    let source0 = MergeSource {
        term_vectors: Some(&tv0_reader),
        ..MergeSource::stored_only(&stored0.fields, &reader0, None, Some(version()))
    };
    let err = merge_stored_only_segments(
        &dir,
        &[source0],
        "_m",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::TermVectors(_)),
        "expected the term-vectors checksum to reject the merge, got {err:?}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn term_vectors_merge_carries_offsets_and_payloads_through() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // Source 0: 2 docs, both with offsets+payloads term vectors.
    let tv0 = flush_term_vectors(
        &[
            tv_doc_with_offsets_and_payloads(0, "alpha", 0, 0, 5, &[0xAA]),
            tv_doc_with_offsets_and_payloads(0, "beta", 0, 0, 4, &[0xBB, 0xCC]),
        ],
        seg0_id,
    );
    // Source 1: 1 doc, also with offsets+payloads term vectors.
    let tv1 = flush_term_vectors(
        &[tv_doc_with_offsets_and_payloads(
            0,
            "gamma",
            0,
            0,
            5,
            &[0xDD],
        )],
        seg1_id,
    );
    let tv0_reader = tv0.reader();
    let tv1_reader = tv1.reader();

    // Sanity-check both flush side actually round-trips offsets/payloads
    // before using them to exercise the merge.
    let doc0 = tv0_reader.document(0).unwrap().unwrap();
    assert!(doc0.fields[0].has_offsets && doc0.fields[0].has_payloads);
    assert_eq!(doc0.fields[0].terms[0].start_offsets, Some(vec![0]));
    assert_eq!(doc0.fields[0].terms[0].end_offsets, Some(vec![5]));
    assert_eq!(doc0.fields[0].terms[0].payloads, Some(vec![vec![0xAA]]));

    let fields0 = vec![tv_field("id", 0)];
    let fields1 = vec![tv_field("id", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields0,
        &[doc_with(0, "alpha"), doc_with(0, "beta")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields1, &[doc_with(0, "gamma")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: Some(&tv0_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: Some(&tv1_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_tv_offsets_payloads",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let merged_tvd = std::fs::read(tmp.join("_merged_tv_offsets_payloads.tvd")).unwrap();
    let merged_tvx = std::fs::read(tmp.join("_merged_tv_offsets_payloads.tvx")).unwrap();
    let merged_tvm = std::fs::read(tmp.join("_merged_tv_offsets_payloads.tvm")).unwrap();
    let merged_reader =
        term_vectors::open(&merged_tvd, &merged_tvx, &merged_tvm, &[9u8; ID_LENGTH], "").unwrap();

    let merged0 = merged_reader.document(0).unwrap().unwrap();
    assert!(merged0.fields[0].has_offsets && merged0.fields[0].has_payloads);
    assert_eq!(merged0.fields[0].terms[0].term, b"alpha");
    assert_eq!(merged0.fields[0].terms[0].start_offsets, Some(vec![0]));
    assert_eq!(merged0.fields[0].terms[0].end_offsets, Some(vec![5]));
    assert_eq!(merged0.fields[0].terms[0].payloads, Some(vec![vec![0xAA]]));

    let merged1 = merged_reader.document(1).unwrap().unwrap();
    assert_eq!(merged1.fields[0].terms[0].term, b"beta");
    assert_eq!(merged1.fields[0].terms[0].start_offsets, Some(vec![0]));
    assert_eq!(merged1.fields[0].terms[0].end_offsets, Some(vec![4]));
    assert_eq!(
        merged1.fields[0].terms[0].payloads,
        Some(vec![vec![0xBB, 0xCC]])
    );

    // Doc 2 comes from source 1 (seg1), merged after source 0's 2 docs.
    let merged2 = merged_reader.document(2).unwrap().unwrap();
    assert_eq!(merged2.fields[0].terms[0].term, b"gamma");
    assert_eq!(merged2.fields[0].terms[0].start_offsets, Some(vec![0]));
    assert_eq!(merged2.fields[0].terms[0].end_offsets, Some(vec![5]));
    assert_eq!(merged2.fields[0].terms[0].payloads, Some(vec![vec![0xDD]]));
}

#[test]
fn full_round_trip_merges_stored_fields_doc_values_norms_and_term_vectors_together() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let dv0 = flush_numeric_dv(0, &[100, 200], seg0_id);
    let dv1 = flush_numeric_dv(0, &[300], seg1_id);
    let norms0 = flush_norms(0, &[1, 2], seg0_id);
    let norms1 = flush_norms(0, &[3], seg1_id);
    let tv0 = flush_term_vectors(&[tv_doc(0, &[("x", 0)]), tv_doc(0, &[("y", 0)])], seg0_id);
    let tv1 = flush_term_vectors(&[tv_doc(0, &[("z", 0)])], seg1_id);

    let mut field0 = numeric_field("body", 0);
    field0.store_term_vectors = true;
    field0.omit_norms = false;
    // `hasNorms()` is `indexOptions != NONE && !omitNorms`, and that is
    // what drives which fields the merge writes norms for.
    field0.index_options = IndexOptions::DocsAndFreqs;
    let fields = vec![field0];

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let tv0_reader = tv0.reader();
    let tv1_reader = tv1.reader();

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let norms0_source = [norms0.source()];
    let norms1_source = [norms1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &dv0_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms0_source,
        term_vectors: Some(&tv0_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &dv1_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms1_source,
        term_vectors: Some(&tv1_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let merged_id = [9u8; ID_LENGTH];
    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_all",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    // Stored fields.
    let merged_fdt = std::fs::read(tmp.join("_merged_all.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged_all.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged_all.fdm")).unwrap();
    let stored_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &merged_id, "").unwrap();
    assert_eq!(stored_reader.max_doc(), 3);
    let vals: Vec<String> = (0..3)
        .map(
            |i| match &stored_reader.document(i).unwrap().fields[0].value {
                FieldValue::String(s) => s.clone(),
                _ => unreachable!(),
            },
        )
        .collect();
    assert_eq!(vals, vec!["a", "b", "c"]);

    // Doc values.
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_all", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_all", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![numeric_field("body", 0)],
    };
    let (_v, dv_meta) = doc_values::parse_meta(
        &dvm,
        &merged_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let dv_entry = dv_meta.numeric_entry(0).unwrap();
    let dv_values: Vec<i64> = (0..3)
        .map(|d| {
            doc_values::numeric_value(&dvd, dv_entry, d)
                .unwrap()
                .unwrap()
        })
        .collect();
    assert_eq!(dv_values, vec![100, 200, 300]);

    // Norms.
    let nvd = std::fs::read(tmp.join("_merged_all.nvd")).unwrap();
    let nvm = std::fs::read(tmp.join("_merged_all.nvm")).unwrap();
    let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
    let norms_entry = norms_meta.entry(0).unwrap();
    let norms_values: Vec<i64> = (0..3)
        .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
        .collect();
    assert_eq!(norms_values, vec![1, 2, 3]);

    // Term vectors.
    let tvd = std::fs::read(tmp.join("_merged_all.tvd")).unwrap();
    let tvx = std::fs::read(tmp.join("_merged_all.tvx")).unwrap();
    let tvm = std::fs::read(tmp.join("_merged_all.tvm")).unwrap();
    let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
    assert_eq!(tv_reader.max_doc(), 3);
    let terms: Vec<Vec<u8>> = (0..3)
        .map(|d| {
            tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                .term
                .clone()
        })
        .collect();
    assert_eq!(terms, vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]);

    // Segment info lists every file.
    let si_bytes = std::fs::read(tmp.join("_merged_all.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
    for ext in [
        "fdt", "fdx", "fdm", "fnm", "nvm", "nvd", "tvd", "tvx", "tvm",
    ] {
        let name = format!("_merged_all.{ext}");
        assert!(si.files.contains(&name), "missing {name} in .si files list");
    }
    // Doc values are a per-field format, so their files carry the
    // `_<format>_<suffix>` segment name real Lucene resolves them by.
    for ext in ["dvm", "dvd", "dvs"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment("_merged_all", DOC_VALUES_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files list");
    }
}

#[test]
fn full_round_trip_merges_stored_fields_binary_doc_values_norms_and_term_vectors_together() {
    // Same shape as
    // `full_round_trip_merges_stored_fields_doc_values_norms_and_term_vectors_together`
    // above, but with a BINARY doc-values field instead of NUMERIC
    // (can't combine both in one call -- see
    // `numeric_and_binary_doc_values_in_the_same_call_is_rejected`).
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let dv0 = flush_binary_dv(0, &[b"pp".to_vec(), b"qq".to_vec()], seg0_id);
    let dv1 = flush_binary_dv(0, &[b"rr".to_vec()], seg1_id);
    let norms0 = flush_norms(0, &[1, 2], seg0_id);
    let norms1 = flush_norms(0, &[3], seg1_id);
    let tv0 = flush_term_vectors(&[tv_doc(0, &[("x", 0)]), tv_doc(0, &[("y", 0)])], seg0_id);
    let tv1 = flush_term_vectors(&[tv_doc(0, &[("z", 0)])], seg1_id);

    let mut field0 = binary_field("body", 0);
    field0.store_term_vectors = true;
    field0.omit_norms = false;
    // `hasNorms()` is `indexOptions != NONE && !omitNorms`, and that is
    // what drives which fields the merge writes norms for.
    field0.index_options = IndexOptions::DocsAndFreqs;
    let fields = vec![field0];

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let tv0_reader = tv0.reader();
    let tv1_reader = tv1.reader();

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let norms0_source = [norms0.source()];
    let norms1_source = [norms1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &dv0_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms0_source,
        term_vectors: Some(&tv0_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &dv1_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms1_source,
        term_vectors: Some(&tv1_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let merged_id = [9u8; ID_LENGTH];
    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_all_bin",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    // Stored fields.
    let merged_fdt = std::fs::read(tmp.join("_merged_all_bin.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged_all_bin.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged_all_bin.fdm")).unwrap();
    let stored_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &merged_id, "").unwrap();
    assert_eq!(stored_reader.max_doc(), 3);
    let vals: Vec<String> = (0..3)
        .map(
            |i| match &stored_reader.document(i).unwrap().fields[0].value {
                FieldValue::String(s) => s.clone(),
                _ => unreachable!(),
            },
        )
        .collect();
    assert_eq!(vals, vec!["a", "b", "c"]);

    // Doc values.
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_all_bin", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_all_bin", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![binary_field("body", 0)],
    };
    let (_v, dv_meta) = doc_values::parse_meta(
        &dvm,
        &merged_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let dv_entry = dv_meta.binary_entry(0).unwrap();
    let dv_values: Vec<Vec<u8>> = (0..3)
        .map(|d| {
            doc_values::binary_value(&dvd, dv_entry, d)
                .unwrap()
                .unwrap()
                .to_vec()
        })
        .collect();
    assert_eq!(
        dv_values,
        vec![b"pp".to_vec(), b"qq".to_vec(), b"rr".to_vec()]
    );

    // Norms.
    let nvd = std::fs::read(tmp.join("_merged_all_bin.nvd")).unwrap();
    let nvm = std::fs::read(tmp.join("_merged_all_bin.nvm")).unwrap();
    let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
    let norms_entry = norms_meta.entry(0).unwrap();
    let norms_values: Vec<i64> = (0..3)
        .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
        .collect();
    assert_eq!(norms_values, vec![1, 2, 3]);

    // Term vectors.
    let tvd = std::fs::read(tmp.join("_merged_all_bin.tvd")).unwrap();
    let tvx = std::fs::read(tmp.join("_merged_all_bin.tvx")).unwrap();
    let tvm = std::fs::read(tmp.join("_merged_all_bin.tvm")).unwrap();
    let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
    assert_eq!(tv_reader.max_doc(), 3);
    let terms: Vec<Vec<u8>> = (0..3)
        .map(|d| {
            tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                .term
                .clone()
        })
        .collect();
    assert_eq!(terms, vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]);

    // Segment info lists every file.
    let si_bytes = std::fs::read(tmp.join("_merged_all_bin.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
    for ext in [
        "fdt", "fdx", "fdm", "fnm", "nvm", "nvd", "tvd", "tvx", "tvm",
    ] {
        let name = format!("_merged_all_bin.{ext}");
        assert!(si.files.contains(&name), "missing {name} in .si files list");
    }
    // Doc values are a per-field format, so their files carry the
    // `_<format>_<suffix>` segment name real Lucene resolves them by.
    for ext in ["dvm", "dvd", "dvs"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment("_merged_all_bin", DOC_VALUES_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files list");
    }
}

#[test]
fn full_round_trip_merges_stored_fields_sorted_doc_values_norms_and_term_vectors_together() {
    // Same shape as
    // `full_round_trip_merges_stored_fields_binary_doc_values_norms_and_term_vectors_together`
    // above, but with a SORTED doc-values field (with an overlapping
    // term across sources, to exercise dictionary dedup end to end
    // alongside stored fields/norms/term vectors in one real merge call).
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let dv0 = flush_sorted_dv(0, &[b"red".to_vec(), b"blue".to_vec()], seg0_id);
    let dv1 = flush_sorted_dv(0, &[b"red".to_vec()], seg1_id);
    let norms0 = flush_norms(0, &[1, 2], seg0_id);
    let norms1 = flush_norms(0, &[3], seg1_id);
    let tv0 = flush_term_vectors(&[tv_doc(0, &[("x", 0)]), tv_doc(0, &[("y", 0)])], seg0_id);
    let tv1 = flush_term_vectors(&[tv_doc(0, &[("z", 0)])], seg1_id);

    let mut field0 = sorted_field("color", 0);
    field0.store_term_vectors = true;
    field0.omit_norms = false;
    // `hasNorms()` is `indexOptions != NONE && !omitNorms`, and that is
    // what drives which fields the merge writes norms for.
    field0.index_options = IndexOptions::DocsAndFreqs;
    let fields = vec![field0];

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let tv0_reader = tv0.reader();
    let tv1_reader = tv1.reader();

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let norms0_source = [norms0.source()];
    let norms1_source = [norms1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv0_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms0_source,
        term_vectors: Some(&tv0_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &dv1_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms1_source,
        term_vectors: Some(&tv1_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let merged_id = [9u8; ID_LENGTH];
    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_all_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    // Stored fields.
    let merged_fdt = std::fs::read(tmp.join("_merged_all_sorted.fdt")).unwrap();
    let merged_fdx = std::fs::read(tmp.join("_merged_all_sorted.fdx")).unwrap();
    let merged_fdm = std::fs::read(tmp.join("_merged_all_sorted.fdm")).unwrap();
    let stored_reader =
        stored_fields::open(&merged_fdt, &merged_fdx, &merged_fdm, &merged_id, "").unwrap();
    assert_eq!(stored_reader.max_doc(), 3);
    let vals: Vec<String> = (0..3)
        .map(
            |i| match &stored_reader.document(i).unwrap().fields[0].value {
                FieldValue::String(s) => s.clone(),
                _ => unreachable!(),
            },
        )
        .collect();
    assert_eq!(vals, vec!["a", "b", "c"]);

    // Doc values: resolved through the unmodified reader stack, checked
    // against the actual expected term per doc (not just ordinal shape)
    // -- "red" (docs 0 and 2, from different sources) must dedupe to the
    // same merged dictionary entry.
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_all_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_all_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let terms = read_back_sorted_terms(&dvm, &dvd, 0, 3);
    assert_eq!(
        terms,
        vec![b"red".to_vec(), b"blue".to_vec(), b"red".to_vec()]
    );

    // Norms.
    let nvd = std::fs::read(tmp.join("_merged_all_sorted.nvd")).unwrap();
    let nvm = std::fs::read(tmp.join("_merged_all_sorted.nvm")).unwrap();
    let (_v, norms_meta) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
    let norms_entry = norms_meta.entry(0).unwrap();
    let norms_values: Vec<i64> = (0..3)
        .map(|d| norms::norm_value(&nvd, norms_entry, d).unwrap().unwrap())
        .collect();
    assert_eq!(norms_values, vec![1, 2, 3]);

    // Term vectors.
    let tvd = std::fs::read(tmp.join("_merged_all_sorted.tvd")).unwrap();
    let tvx = std::fs::read(tmp.join("_merged_all_sorted.tvx")).unwrap();
    let tvm = std::fs::read(tmp.join("_merged_all_sorted.tvm")).unwrap();
    let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
    assert_eq!(tv_reader.max_doc(), 3);
    let tv_terms: Vec<Vec<u8>> = (0..3)
        .map(|d| {
            tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                .term
                .clone()
        })
        .collect();
    assert_eq!(tv_terms, vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()]);

    // Segment info lists every file.
    let si_bytes = std::fs::read(tmp.join("_merged_all_sorted.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
    for ext in [
        "fdt", "fdx", "fdm", "fnm", "nvm", "nvd", "tvd", "tvx", "tvm",
    ] {
        let name = format!("_merged_all_sorted.{ext}");
        assert!(si.files.contains(&name), "missing {name} in .si files list");
    }
    // Doc values are a per-field format, so their files carry the
    // `_<format>_<suffix>` segment name real Lucene resolves them by.
    for ext in ["dvm", "dvd", "dvs"] {
        let name = format!(
            "{}.{ext}",
            per_field_segment("_merged_all_sorted", DOC_VALUES_FORMAT_NAME)
        );
        assert!(si.files.contains(&name), "missing {name} in .si files list");
    }
}

// --- merge_sorted_stored_only_segments (k-way sort-preserving merge) ---

/// Reads back the merged segment's stored "id" field (a String) for every
/// doc, in doc order -- the assertion helper every k-way-merge test below
/// uses to confirm both order and content.
fn read_merged_ids(
    tmp: &std::path::Path,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
) -> Vec<String> {
    let fdt = std::fs::read(tmp.join(format!("{segment_name}.fdt"))).unwrap();
    let fdx = std::fs::read(tmp.join(format!("{segment_name}.fdx"))).unwrap();
    let fdm = std::fs::read(tmp.join(format!("{segment_name}.fdm"))).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &segment_id, "").unwrap();
    (0..reader.max_doc())
        .map(|d| match &reader.document(d).unwrap().fields[0].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        })
        .collect()
}

#[test]
fn two_sources_with_interleaved_keys_produce_globally_sorted_output_not_concatenation() {
    // Source 0 (already sorted by "num" ascending): 10, 30, 50.
    // Source 1 (already sorted by "num" ascending): 20, 40.
    // Naive concatenation would yield 10,30,50,20,40 -- visibly
    // out-of-order at the 50->20 boundary. The real k-way merge must
    // interleave to 10,20,30,40,50.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];

    let seg0 = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "10"), doc_with(0, "30"), doc_with(0, "50")],
    );
    let seg1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "20"), doc_with(0, "40")],
    );

    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];

    let keys0: Vec<Option<i64>> = vec![Some(10), Some(30), Some(50)];
    let keys1: Vec<Option<i64>> = vec![Some(20), Some(40)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &num_sort,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [9u8; ID_LENGTH];
    let sci = merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_sorted");

    let ids = read_merged_ids(&tmp, "_merged_sorted", merged_id);
    assert_eq!(ids, vec!["10", "20", "30", "40", "50"]);

    // Confirm this is NOT what naive concatenation would produce.
    assert_ne!(ids, vec!["10", "30", "50", "20", "40"]);

    // The merged .si must keep the same index-sort descriptor, not lose
    // or null it out.
    let si_bytes = std::fs::read(tmp.join("_merged_sorted.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &merged_id).unwrap();
    let sort = si.index_sort.unwrap();
    assert_eq!(sort.len(), 1);
    assert_eq!(sort[0], IndexSortField::long("num", false, Some(i64::MAX)));
}

/// A byte-keyed tier merges by its (global) rank key, and a tie on
/// every tier goes to the earlier source -- `MultiSorter`'s tie-break.
#[test]
fn a_binary_sort_tier_merges_by_rank_ties_to_the_earlier_source() {
    let sort = IndexSortField {
        field: "bytes".to_string(),
        reverse: false,
        kind: crate::segment_info::IndexSortKind::Binary(
            crate::segment_info::StringMissingValue::Last,
        ),
    };
    let a: Vec<Option<i64>> = vec![Some(0), Some(2), None];
    let b: Vec<Option<i64>> = vec![Some(0), Some(1)];
    let per_source: Vec<&[Option<i64>]> = vec![&a, &b];
    assert_eq!(
        sorted_doc_order(
            &[MergeSortKeySpec {
                sort: &sort,
                per_source_keys: &per_source,
            }],
            &[vec![0, 1, 2], vec![0, 1]],
        ),
        vec![(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)]
    );
}

#[test]
fn three_sources_k_way_merge_by_sort_key() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];

    let seg0 = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "1"), doc_with(0, "9")],
    );
    let seg1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "4"), doc_with(0, "6")],
    );
    let seg2 = flush(
        &dir,
        &tmp,
        "_2",
        [3u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "2"), doc_with(0, "5"), doc_with(0, "8")],
    );

    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let reader2 = open_reader(&seg2);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
        MergeSource::stored_only(&seg2.fields, &reader2, None, Some(version())),
    ];

    let keys0: Vec<Option<i64>> = vec![Some(1), Some(9)];
    let keys1: Vec<Option<i64>> = vec![Some(4), Some(6)];
    let keys2: Vec<Option<i64>> = vec![Some(2), Some(5), Some(8)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1, &keys2];
    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &num_sort,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [9u8; ID_LENGTH];
    merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_three",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    let ids = read_merged_ids(&tmp, "_merged_three", merged_id);
    assert_eq!(ids, vec!["1", "2", "4", "5", "6", "8", "9"]);
}

#[test]
fn tie_on_primary_field_across_sources_is_broken_by_secondary_field() {
    // Both sources' first doc ties on "num"=5; the secondary field "tie"
    // must break it (source 0's doc has tie=1, source 1's has tie=0, so
    // source 1's doc must come first despite arriving from the
    // "later" source).
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];

    let seg0 = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "A"), doc_with(0, "C")],
    );
    let seg1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "B")],
    );

    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];

    // Primary "num": source0 = [5, 8], source1 = [5].
    let num0: Vec<Option<i64>> = vec![Some(5), Some(8)];
    let num1: Vec<Option<i64>> = vec![Some(5)];
    let num_keys: Vec<&[Option<i64>]> = vec![&num0, &num1];
    // Secondary "tie": source0's doc "A" has tie=1, source1's doc "B" has
    // tie=0 -- ascending means "B" (tie=0) must sort before "A" (tie=1).
    let tie0: Vec<Option<i64>> = vec![Some(1), Some(0)];
    let tie1: Vec<Option<i64>> = vec![Some(0)];
    let tie_keys: Vec<&[Option<i64>]> = vec![&tie0, &tie1];

    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let tie_sort = IndexSortField::long("tie", false, Some(i64::MAX));
    let sort_fields = vec![
        MergeSortKeySpec {
            sort: &num_sort,
            per_source_keys: &num_keys,
        },
        MergeSortKeySpec {
            sort: &tie_sort,
            per_source_keys: &tie_keys,
        },
    ];

    let merged_id = [9u8; ID_LENGTH];
    merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_tiebreak",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    let ids = read_merged_ids(&tmp, "_merged_tiebreak", merged_id);
    // "B" (num=5,tie=0) before "A" (num=5,tie=1) before "C" (num=8).
    assert_eq!(ids, vec!["B", "A", "C"]);
}

#[test]
fn stored_field_content_stays_attached_to_the_right_doc_after_reordering() {
    // Multi-field docs where the field content itself encodes the sort
    // key, confirming the whole Document (not just a scalar) travels
    // with its doc through the k-way merge -- a shuffle bug that
    // permuted docs independently of their sort key would show up here
    // as mismatched (key, payload) pairs.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("key", 0), field("payload", 1)];

    fn doc(key: &str, payload: &str) -> Document {
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(key.to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(payload.to_string()),
                },
            ],
        }
    }

    let seg0 = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc("3", "three"), doc("7", "seven")],
    );
    let seg1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc("1", "one"), doc("5", "five")],
    );

    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];

    let keys0: Vec<Option<i64>> = vec![Some(3), Some(7)];
    let keys1: Vec<Option<i64>> = vec![Some(1), Some(5)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
    let sort_key_1 = IndexSortField::long("key", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &sort_key_1,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [9u8; ID_LENGTH];
    merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_payload",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    let fdt = std::fs::read(tmp.join("_merged_payload.fdt")).unwrap();
    let fdx = std::fs::read(tmp.join("_merged_payload.fdx")).unwrap();
    let fdm = std::fs::read(tmp.join("_merged_payload.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &merged_id, "").unwrap();
    assert_eq!(reader.max_doc(), 4);

    let expected = [("1", "one"), ("3", "three"), ("5", "five"), ("7", "seven")];
    for (i, (key, payload)) in expected.iter().enumerate() {
        let d = reader.document(i as i32).unwrap();
        let got_key = match &d.fields[0].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        let got_payload = match &d.fields[1].value {
            FieldValue::String(s) => s.clone(),
            _ => unreachable!(),
        };
        assert_eq!(&got_key, key, "doc {i} key mismatch");
        assert_eq!(&got_payload, payload, "doc {i} payload mismatch");
    }
}

#[test]
fn doc_values_norms_and_term_vectors_follow_the_sort_key_not_source_concatenation() {
    // Confirms `merge_sorted_stored_only_
    // segments` physically reorders BINARY doc-values, norms, and term
    // vectors by sort key -- not just stored fields -- exactly like
    // `three_sources_k_way_merge_by_sort_key` proves for stored fields
    // above. Interleaved sort keys mean naive source concatenation
    // would produce a visibly wrong order at the first cross-source
    // boundary; a real k-way merge must interleave.
    //
    // Source 0 (already sorted by "num"): docs with keys 10, 30.
    // Source 1 (already sorted by "num"): docs with keys 20, 40.
    // Correct global order: 10, 20, 30, 40 (source0/1/0/1 interleaved).
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // BINARY doc-values: value encodes the key ("v10", "v30" / "v20",
    // "v40") so the merged order can be read straight back off it.
    let bin0 = flush_binary_dv(1, &[b"v10".to_vec(), b"v30".to_vec()], seg0_id);
    let bin1 = flush_binary_dv(1, &[b"v20".to_vec(), b"v40".to_vec()], seg1_id);

    // Norms: same encoding scheme, divided by 10 (1, 3 / 2, 4).
    let norms0 = flush_norms(2, &[1, 3], seg0_id);
    let norms1 = flush_norms(2, &[2, 4], seg1_id);

    // Term vectors: one field, one term "t", whose position encodes the
    // key divided by 10 (same scheme as norms) -- `tv_doc`'s second
    // tuple element is a position, not a freq.
    let tv0 = flush_term_vectors(&[tv_doc(3, &[("t", 1)]), tv_doc(3, &[("t", 3)])], seg0_id);
    let tv1 = flush_term_vectors(&[tv_doc(3, &[("t", 2)]), tv_doc(3, &[("t", 4)])], seg1_id);

    let fields = vec![
        field("id", 0),
        binary_field("bin", 1),
        norms_field("nrm", 2),
        tv_field("tv", 3),
    ];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "10"), doc_with(0, "30")],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &fields,
        &[doc_with(0, "20"), doc_with(0, "40")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);
    let tv0_reader = tv0.reader();
    let tv1_reader = tv1.reader();

    let bin0_source = [bin0.source()];
    let bin1_source = [bin1.source()];
    let norms0_source = [norms0.source()];
    let norms1_source = [norms1.source()];

    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &bin0_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms0_source,
        term_vectors: Some(&tv0_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &bin1_source,
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &norms1_source,
        term_vectors: Some(&tv1_reader),
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let keys0: Vec<Option<i64>> = vec![Some(10), Some(30)];
    let keys1: Vec<Option<i64>> = vec![Some(20), Some(40)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &num_sort,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [9u8; ID_LENGTH];
    merge_sorted_stored_only_segments(
        &dir,
        &[source0, source1],
        &sort_fields,
        "_merged_dv_norms_tv_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    // Stored "id" field confirms the expected doc order up front.
    let ids = read_merged_ids(&tmp, "_merged_dv_norms_tv_sorted", merged_id);
    assert_eq!(ids, vec!["10", "20", "30", "40"]);

    // BINARY doc-values: must read back "v10","v20","v30","v40" -- the
    // sorted order -- not "v10","v30","v20","v40" (source concatenation).
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_dv_norms_tv_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_dv_norms_tv_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![binary_field("bin", 1)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &merged_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.binary_entry(1).unwrap();
    let bin_values: Vec<Vec<u8>> = (0..4)
        .map(|d| {
            doc_values::binary_value(&dvd, entry, d)
                .unwrap()
                .unwrap()
                .to_vec()
        })
        .collect();
    assert_eq!(
        bin_values,
        vec![
            b"v10".to_vec(),
            b"v20".to_vec(),
            b"v30".to_vec(),
            b"v40".to_vec(),
        ]
    );

    // Norms: must read back 1,2,3,4 (sorted), not 1,3,2,4 (concatenated).
    let nvd = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.nvd")).unwrap();
    let nvm = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.nvm")).unwrap();
    let (_v, parsed) = norms::parse_meta(&nvm, &merged_id, "").unwrap();
    let norm_entry = parsed.entry(2).unwrap();
    let norm_values: Vec<i64> = (0..4)
        .map(|d| norms::norm_value(&nvd, norm_entry, d).unwrap().unwrap())
        .collect();
    assert_eq!(norm_values, vec![1, 2, 3, 4]);

    // Term vectors: doc 0's term "t" must have position 1 (key 10),
    // doc 1's position 2 (key 20), doc 2's position 3 (key 30), doc 3's
    // position 4 (key 40) -- sorted order, not concatenation order
    // (which would put position 3 at doc 1 and position 2 at doc 2).
    let tvd = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.tvd")).unwrap();
    let tvx = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.tvx")).unwrap();
    let tvm = std::fs::read(tmp.join("_merged_dv_norms_tv_sorted.tvm")).unwrap();
    let tv_reader = term_vectors::open(&tvd, &tvx, &tvm, &merged_id, "").unwrap();
    let tv_positions: Vec<i32> = (0..4)
        .map(|d| {
            tv_reader.document(d).unwrap().unwrap().fields[0].terms[0]
                .positions
                .as_ref()
                .unwrap()[0]
        })
        .collect();
    assert_eq!(tv_positions, vec![1, 2, 3, 4]);
}

#[test]
fn numeric_sorted_sorted_numeric_and_sorted_set_doc_values_follow_the_sort_key() {
    // Sibling of `doc_values_norms_and_term_vectors_follow_the_sort_key_
    // not_source_concatenation` above, covering the four doc-values
    // types that test doesn't exercise (NUMERIC/SORTED/SORTED_NUMERIC/
    // SORTED_SET) -- reusing the exact same interleaved-sort-key setup
    // (source0: keys 10,30; source1: keys 20,40; correct order
    // 10,20,30,40) so a reorder bug in any one of these four merge
    // helpers specifically would fail here, not just leave them "not
    // dropped" (which the pre-existing concatenation-merge tests for
    // these types already prove, but can't catch a reorder bug since
    // they merge via `merge_stored_only_segments`, not the sort-aware
    // k-way path).
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let num0 = flush_numeric_dv(1, &[10, 30], seg0_id);
    let num1 = flush_numeric_dv(1, &[20, 40], seg1_id);
    let sorted0 = flush_sorted_dv(2, &[b"v10".to_vec(), b"v30".to_vec()], seg0_id);
    let sorted1 = flush_sorted_dv(2, &[b"v20".to_vec(), b"v40".to_vec()], seg1_id);
    let sn0 = flush_sorted_numeric_dv(3, &[vec![10], vec![30]], seg0_id);
    let sn1 = flush_sorted_numeric_dv(3, &[vec![20], vec![40]], seg1_id);
    let ss0 = flush_sorted_set_dv(4, &[vec![b"v10".to_vec()], vec![b"v30".to_vec()]], seg0_id);
    let ss1 = flush_sorted_set_dv(4, &[vec![b"v20".to_vec()], vec![b"v40".to_vec()]], seg1_id);

    let fields = vec![
        field("id", 0),
        numeric_field("num", 1),
        sorted_field("srt", 2),
        sorted_numeric_field("sn", 3),
        sorted_set_field("ss", 4),
    ];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "10"), doc_with(0, "30")],
    );
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &fields,
        &[doc_with(0, "20"), doc_with(0, "40")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let num0_source = [num0.source()];
    let num1_source = [num1.source()];
    let sorted0_source = [sorted0.source()];
    let sorted1_source = [sorted1.source()];
    let sn0_source = [sn0.source()];
    let sn1_source = [sn1.source()];
    let ss0_source = [ss0.source()];
    let ss1_source = [ss1.source()];

    let keys0: Vec<Option<i64>> = vec![Some(10), Some(30)];
    let keys1: Vec<Option<i64>> = vec![Some(20), Some(40)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
    let sort_id_2 = IndexSortField::long("id", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &sort_id_2,
        per_source_keys: &per_source_keys,
    }];

    // This port's doc-values merge is single-doc-values-type-per-merge
    // (pre-existing limitation, unrelated to sorting -- see the
    // `MultipleDocValuesTypesInOneMerge` error), so each type is merged
    // in its own separate call rather than all four in one `MergeSource`
    // pair; each still shares the same interleaved-sort-key setup.
    let merged_id = [9u8; ID_LENGTH];

    let source0_num = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &num0_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1_num = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &num1_source,
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    merge_sorted_stored_only_segments(
        &dir,
        &[source0_num, source1_num],
        &sort_fields,
        "_merged_num_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    let ids = read_merged_ids(&tmp, "_merged_num_sorted", merged_id);
    assert_eq!(ids, vec!["10", "20", "30", "40"]);
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_num_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_num_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_numeric_field_infos = field_infos::FieldInfos {
        fields: vec![numeric_field("x", 1)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &merged_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_numeric_field_infos,
    )
    .unwrap();
    let num_entry = meta.numeric_entry(1).unwrap();
    let numeric_values: Vec<i64> = (0..4)
        .map(|d| {
            doc_values::numeric_value(&dvd, num_entry, d)
                .unwrap()
                .unwrap()
        })
        .collect();
    // NUMERIC: 10,20,30,40 sorted, not 10,30,20,40 concatenated.
    assert_eq!(numeric_values, vec![10, 20, 30, 40]);

    let source0_sorted = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &sorted0_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1_sorted = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &sorted1_source,
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    merge_sorted_stored_only_segments(
        &dir,
        &[source0_sorted, source1_sorted],
        &sort_fields,
        "_merged_sorted_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sorted_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sorted_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    // SORTED: "v10","v20","v30","v40" sorted, not concatenated.
    let sorted_terms = read_back_sorted_terms(&dvm, &dvd, 2, 4);
    assert_eq!(
        sorted_terms,
        vec![
            b"v10".to_vec(),
            b"v20".to_vec(),
            b"v30".to_vec(),
            b"v40".to_vec(),
        ]
    );

    let source0_sn = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &sn0_source,
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1_sn = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &sn1_source,
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    merge_sorted_stored_only_segments(
        &dir,
        &[source0_sn, source1_sn],
        &sort_fields,
        "_merged_sn_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sn_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sn_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    // SORTED_NUMERIC: [10],[20],[30],[40] sorted, not concatenated.
    let sn_values = read_back_sorted_numeric_values(&dvm, &dvd, 3, 4);
    assert_eq!(sn_values, vec![vec![10], vec![20], vec![30], vec![40]]);

    let source0_ss = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &ss0_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1_ss = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &ss1_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    merge_sorted_stored_only_segments(
        &dir,
        &[source0_ss, source1_ss],
        &sort_fields,
        "_merged_ss_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_ss_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_ss_sorted", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    // SORTED_SET: ["v10"],["v20"],["v30"],["v40"] sorted, not concatenated.
    let ss_values = read_back_sorted_set_values(&dvm, &dvd, 4, 4);
    assert_eq!(
        ss_values,
        vec![
            vec![b"v10".to_vec()],
            vec![b"v20".to_vec()],
            vec![b"v30".to_vec()],
            vec![b"v40".to_vec()],
        ]
    );
}

#[test]
fn deleted_docs_are_dropped_before_the_k_way_merge_walks_a_source() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];

    let seg0 = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "10"), doc_with(0, "20"), doc_with(0, "30")],
    );
    let seg1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "15")],
    );

    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);

    // Drop doc 1 ("20") from source 0.
    let mut live0 = FixedBitSet::new(3);
    live0.set(0);
    live0.set(2);

    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, Some(&live0), Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];

    let keys0: Vec<Option<i64>> = vec![Some(10), Some(20), Some(30)];
    let keys1: Vec<Option<i64>> = vec![Some(15)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &num_sort,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [9u8; ID_LENGTH];
    merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_deleted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    let ids = read_merged_ids(&tmp, "_merged_deleted", merged_id);
    assert_eq!(ids, vec!["10", "15", "30"]);
}

#[test]
fn no_sources_produces_an_empty_sorted_segment() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let sources: Vec<MergeSource> = vec![];
    let per_source_keys: Vec<&[Option<i64>]> = vec![];
    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &num_sort,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [3u8; ID_LENGTH];
    let sci = merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_empty_sorted",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_empty_sorted");

    let fdt = std::fs::read(tmp.join("_merged_empty_sorted.fdt")).unwrap();
    let fdx = std::fs::read(tmp.join("_merged_empty_sorted.fdx")).unwrap();
    let fdm = std::fs::read(tmp.join("_merged_empty_sorted.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &merged_id, "").unwrap();
    assert_eq!(reader.max_doc(), 0);
}

#[test]
fn field_number_reconciliation_still_applies_during_the_k_way_merge() {
    // Same field-name-vs-number-mismatch setup as the concatenation
    // merge's own test, confirming the k-way merge path also reconciles
    // field numbers by name rather than trusting per-source numbering.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields0 = vec![field("num", 0), field("id", 1)];
    let fields1 = vec![field("id", 0), field("num", 1)];

    let doc0 = Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("10".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String("first".to_string()),
            },
        ],
    };
    let doc1 = Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("second".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String("5".to_string()),
            },
        ],
    };

    let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields0, &[doc0]);
    let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields1, &[doc1]);

    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];

    let keys0: Vec<Option<i64>> = vec![Some(10)];
    let keys1: Vec<Option<i64>> = vec![Some(5)];
    let per_source_keys: Vec<&[Option<i64>]> = vec![&keys0, &keys1];
    let num_sort = IndexSortField::long("num", false, Some(i64::MAX));
    let sort_fields = vec![MergeSortKeySpec {
        sort: &num_sort,
        per_source_keys: &per_source_keys,
    }];

    let merged_id = [9u8; ID_LENGTH];
    merge_sorted_stored_only_segments(
        &dir,
        &sources,
        &sort_fields,
        "_merged_reconcile",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();

    let merged_fnm = std::fs::read(tmp.join("_merged_reconcile.fnm")).unwrap();
    let merged_fields = lucene_codecs::field_infos::parse(&merged_fnm, &merged_id, "").unwrap();
    let id_number = merged_fields
        .fields
        .iter()
        .find(|f| f.name == "id")
        .unwrap()
        .number;

    let fdt = std::fs::read(tmp.join("_merged_reconcile.fdt")).unwrap();
    let fdx = std::fs::read(tmp.join("_merged_reconcile.fdx")).unwrap();
    let fdm = std::fs::read(tmp.join("_merged_reconcile.fdm")).unwrap();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &merged_id, "").unwrap();
    assert_eq!(reader.max_doc(), 2);

    // Sorted by num: doc1 (num=5, "second"/"5") comes first, doc0
    // (num=10, "10"/"first") comes second -- confirm the "id" field's
    // content followed its own doc through the reordering.
    let d0 = reader.document(0).unwrap();
    let id0 = d0
        .fields
        .iter()
        .find(|f| f.field_number == id_number)
        .unwrap();
    assert_eq!(id0.value, FieldValue::String("second".to_string()));

    let d1 = reader.document(1).unwrap();
    let id1 = d1
        .fields
        .iter()
        .find(|f| f.field_number == id_number)
        .unwrap();
    assert_eq!(id1.value, FieldValue::String("first".to_string()));
}

// --- SORTED_NUMERIC ---

fn sorted_numeric_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.doc_values_type = DocValuesType::SortedNumeric;
    f
}

/// Same idea as [`FlushedNumericDv`], for SORTED_NUMERIC doc values --
/// `values` is one doc's whole value list per entry (dense,
/// `values.len() == max_doc`, every doc non-empty), exactly what
/// [`doc_values::write_single_dense_sorted_numeric_field`] takes.
struct FlushedSortedNumericDv {
    data: Vec<u8>,
    entry: SortedNumericEntry,
}

fn flush_sorted_numeric_dv(
    field_number: i32,
    values: &[Vec<i64>],
    segment_id: [u8; ID_LENGTH],
) -> FlushedSortedNumericDv {
    let (meta, data, _skip) = doc_values::write_single_dense_sorted_numeric_field(
        field_number,
        values,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    )
    .unwrap();
    let field_infos = field_infos::FieldInfos {
        fields: vec![sorted_numeric_field("x", field_number)],
    };
    let (_version, parsed) = doc_values::parse_meta(
        &meta,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = parsed.sorted_numeric_entry(field_number).unwrap().clone();
    FlushedSortedNumericDv { data, entry }
}

impl FlushedSortedNumericDv {
    fn source(&self) -> SourceSortedNumericDocValues<'_> {
        SourceSortedNumericDocValues {
            data: &self.data,
            entry: self.entry.clone(),
        }
    }
}

/// Resolves every merged doc's whole value list, doc by doc, through the
/// *unmodified* reader stack (`parse_meta` + `sorted_numeric_values`) --
/// the correctness check that a doc's full multi-value list survived the
/// merge, not just its value count.
fn read_back_sorted_numeric_values(
    dvm: &[u8],
    dvd: &[u8],
    field_number: i32,
    doc_count: i32,
) -> Vec<Vec<i64>> {
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_numeric_field("x", field_number)],
    };
    let (_v, meta) = doc_values::parse_meta(
        dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_numeric_entry(field_number).unwrap();
    (0..doc_count)
        .map(|d| doc_values::sorted_numeric_values(dvd, entry, d).unwrap())
        .collect()
}

#[test]
fn sorted_numeric_doc_values_merge_across_two_sources_with_deletions() {
    // Source 0: 2 docs, doc 0 has 2 values, doc 1 (deleted) has 1 value --
    // confirms a surviving multi-value doc keeps *both* of its values
    // after merge, not just its first/last.
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv0 = flush_sorted_numeric_dv(0, &[vec![10, 11], vec![20]], seg0_id);
    // Source 1: 1 doc, single value.
    let dv1 = flush_sorted_numeric_dv(0, &[vec![30]], seg1_id);

    let fields = vec![sorted_numeric_field("nums", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let mut live0 = FixedBitSet::new(2);
    live0.set(0); // keep doc 0 (values [10, 11]), drop doc 1 ([20])

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &dv0_source,
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &dv1_source,
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_numeric",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sorted_numeric", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sorted_numeric", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let values = read_back_sorted_numeric_values(&dvm, &dvd, 0, 2);
    assert_eq!(values, vec![vec![10, 11], vec![30]]);
}

#[test]
fn sorted_numeric_doc_values_missing_in_a_live_contributing_source_merges_sparse() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv0 = flush_sorted_numeric_dv(0, &[vec![10]], seg0_id);
    let fields = vec![sorted_numeric_field("nums", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &dv0_source,
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    // Source 1 has live docs but no SORTED_NUMERIC doc-values entry at
    // all for field "nums".
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_numeric_err",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let suffix_seg = per_field_segment("_merged_sorted_numeric_err", DOC_VALUES_FORMAT_NAME);
    let dvd = std::fs::read(tmp.join(format!("{suffix_seg}.dvd"))).unwrap();
    let dvm = std::fs::read(tmp.join(format!("{suffix_seg}.dvm"))).unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_numeric_field("f", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_numeric_entry(0).unwrap();
    let got: Vec<Vec<i64>> = (0..2)
        .map(|d| doc_values::sorted_numeric_values(&dvd, entry, d).unwrap())
        .collect();
    assert_eq!(got, vec![vec![10], vec![]]);
}

#[test]
fn two_sorted_numeric_doc_values_fields_land_in_one_merged_dvm() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv_a = flush_sorted_numeric_dv(0, &[vec![1]], seg0_id);
    let dv_b = flush_sorted_numeric_dv(1, &[vec![2]], seg0_id);
    let fields = vec![sorted_numeric_field("a", 0), sorted_numeric_field("b", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let sorted_numeric = vec![dv_a.source(), dv_b.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &sorted_numeric,
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_sndv_two",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_sndv_two",
        [9u8; ID_LENGTH],
        vec![sorted_numeric_field("a", 0), sorted_numeric_field("b", 1)],
    );
    assert_eq!(
        doc_values::sorted_numeric_values(&dvd, meta.sorted_numeric_entry(0).unwrap(), 0).unwrap(),
        vec![1]
    );
    assert_eq!(
        doc_values::sorted_numeric_values(&dvd, meta.sorted_numeric_entry(1).unwrap(), 0).unwrap(),
        vec![2]
    );
}

// --- SORTED_SET ---

fn sorted_set_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.doc_values_type = DocValuesType::SortedSet;
    f
}

/// Same idea as [`FlushedSortedDv`], for SORTED_SET doc values --
/// `values` is one doc's whole (possibly multi-valued, possibly
/// duplicate-containing) raw value set per entry (dense,
/// `values.len() == max_doc`, every doc non-empty), exactly what
/// [`doc_values::write_single_dense_sorted_set_field`] takes.
struct FlushedSortedSetDv {
    data: Vec<u8>,
    entry: SortedSetEntry,
}

fn flush_sorted_set_dv(
    field_number: i32,
    values: &[Vec<Vec<u8>>],
    segment_id: [u8; ID_LENGTH],
) -> FlushedSortedSetDv {
    let max_doc = values.len() as i32;
    let (meta, data, _skip) = doc_values::write_single_dense_sorted_set_field(
        field_number,
        values,
        max_doc,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
    )
    .unwrap();
    let field_infos = field_infos::FieldInfos {
        fields: vec![sorted_set_field("x", field_number)],
    };
    let (_version, parsed) = doc_values::parse_meta(
        &meta,
        &segment_id,
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &field_infos,
    )
    .unwrap();
    let entry = parsed.sorted_set_entry(field_number).unwrap().clone();
    FlushedSortedSetDv { data, entry }
}

impl FlushedSortedSetDv {
    fn source(&self) -> SourceSortedSetDocValues<'_> {
        SourceSortedSetDocValues {
            data: &self.data,
            entry: self.entry.clone(),
        }
    }
}

/// Resolves every merged doc's full (sorted, deduped) value set, doc by
/// doc, through the *unmodified* reader stack ([`sorted_set_doc_ordinals`]
/// and [`sorted_set_source_dict`], the same helpers
/// [`merge_sorted_set_doc_values`] itself uses) -- the critical
/// correctness check: not just "some valid ordinals", but the actual
/// right resolved terms per doc, read back exactly as a real caller
/// would.
fn read_back_sorted_set_values(
    dvm: &[u8],
    dvd: &[u8],
    field_number: i32,
    doc_count: i32,
) -> Vec<Vec<Vec<u8>>> {
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_set_field("x", field_number)],
    };
    let (_v, meta) = doc_values::parse_meta(
        dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_set_entry(field_number).unwrap();
    let dict = sorted_set_source_dict(dvd, entry).unwrap();
    (0..doc_count)
        .map(|d| {
            sorted_set_doc_ordinals(dvd, entry, d)
                .unwrap()
                .into_iter()
                .map(|ord| dict[ord as usize].clone())
                .collect()
        })
        .collect()
}

#[test]
fn sorted_set_doc_values_merge_with_overlapping_terms_dedupes_into_one_shared_dictionary_entry() {
    // Source 0: one doc with ["red", "blue"]; source 1: one doc with
    // ["red", "green"] -- both sources independently assign "red"
    // ordinal 1 (alphabetically after "blue"/"green" respectively in
    // each source's own 2-term dictionary... actually "blue" < "red" and
    // "green" < "red", so "red" is ordinal 1 in both). Real bug case: if
    // this merge naively concatenated ordinals without resolving to
    // bytes, source 1's "red" could land on a different merged
    // dictionary entry than source 0's "red" purely because they came
    // from different sources -- this test catches that by checking the
    // actual resolved term *sets*, not just dictionary size.
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv0 = flush_sorted_set_dv(0, &[vec![b"red".to_vec(), b"blue".to_vec()]], seg0_id);
    let dv1 = flush_sorted_set_dv(0, &[vec![b"red".to_vec(), b"green".to_vec()]], seg1_id);

    let fields = vec![sorted_set_field("colors", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &dv0_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &dv1_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_set_overlap",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sorted_set_overlap", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sorted_set_overlap", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_set_field("colors", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_set_entry(0).unwrap();
    let dict = sorted_set_source_dict(&dvd, entry).unwrap();
    // "red" is shared across both sources -- the merged dictionary must
    // dedupe it into exactly one entry, so the distinct dictionary size
    // is 3 ("red", "blue", "green"), not 4.
    assert_eq!(dict.len(), 3);

    // And each doc must resolve to its RIGHT value set, not just any
    // valid ordinals -- this is the actual correctness check.
    let mut values = read_back_sorted_set_values(&dvm, &dvd, 0, 2);
    for doc_values in &mut values {
        doc_values.sort();
    }
    assert_eq!(
        values,
        vec![
            {
                let mut v = vec![b"red".to_vec(), b"blue".to_vec()];
                v.sort();
                v
            },
            {
                let mut v = vec![b"red".to_vec(), b"green".to_vec()];
                v.sort();
                v
            },
        ]
    );
}

#[test]
fn sorted_set_doc_values_merge_with_disjoint_terms_contains_all_terms_from_both_sources() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];
    let dv0 = flush_sorted_set_dv(0, &[vec![b"apple".to_vec()]], seg0_id);
    let dv1 = flush_sorted_set_dv(0, &[vec![b"zebra".to_vec()]], seg1_id);

    let fields = vec![sorted_set_field("word", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let dv1_source = [dv1.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &dv0_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &dv1_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_set_disjoint",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let dvm = std::fs::read(tmp.join(format!(
        "{}.dvm",
        per_field_segment("_merged_sorted_set_disjoint", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let dvd = std::fs::read(tmp.join(format!(
        "{}.dvd",
        per_field_segment("_merged_sorted_set_disjoint", DOC_VALUES_FORMAT_NAME)
    )))
    .unwrap();
    let values = read_back_sorted_set_values(&dvm, &dvd, 0, 2);
    assert_eq!(
        values,
        vec![vec![b"apple".to_vec()], vec![b"zebra".to_vec()]]
    );
}

#[test]
fn sorted_set_doc_values_missing_in_a_live_contributing_source_merges_sparse() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv0 = flush_sorted_set_dv(0, &[vec![b"x".to_vec()]], seg0_id);
    let fields = vec![sorted_set_field("word", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let dv0_source = [dv0.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &dv0_source,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    // Source 1 has live docs but no SORTED_SET doc-values entry at all
    // for field "word".
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_sorted_set_err",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let suffix_seg = per_field_segment("_merged_sorted_set_err", DOC_VALUES_FORMAT_NAME);
    let dvd = std::fs::read(tmp.join(format!("{suffix_seg}.dvd"))).unwrap();
    let dvm = std::fs::read(tmp.join(format!("{suffix_seg}.dvm"))).unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![sorted_set_field("f", 0)],
    };
    let (_v, meta) = doc_values::parse_meta(
        &dvm,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(DOC_VALUES_FORMAT_NAME),
        &merged_field_infos,
    )
    .unwrap();
    let entry = meta.sorted_set_entry(0).unwrap();
    let got: Vec<Vec<i64>> = (0..2)
        .map(|d| sorted_set_doc_ordinals(&dvd, entry, d).unwrap())
        .collect();
    assert_eq!(got, vec![vec![0], vec![]]);
}

#[test]
fn two_sorted_set_doc_values_fields_land_in_one_merged_dvm() {
    let seg0_id = [1u8; ID_LENGTH];
    let dv_a = flush_sorted_set_dv(0, &[vec![b"x".to_vec()]], seg0_id);
    let dv_b = flush_sorted_set_dv(1, &[vec![b"y".to_vec()]], seg0_id);
    let fields = vec![sorted_set_field("a", 0), sorted_set_field("b", 1)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("x".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String("y".to_string()),
                },
            ],
        }],
    );
    let reader0 = open_reader(&stored0);
    let sorted_set = vec![dv_a.source(), dv_b.source()];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &sorted_set,
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_ssdv_two",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (dvd, meta) = open_merged_doc_values(
        tmp.path(),
        "_merged_ssdv_two",
        [9u8; ID_LENGTH],
        vec![sorted_set_field("a", 0), sorted_set_field("b", 1)],
    );
    for (field_number, expected) in [(0, &b"x"[..]), (1, &b"y"[..])] {
        let entry = meta.sorted_set_entry(field_number).unwrap();
        let ords = sorted_set_doc_ordinals(&dvd, entry, 0).unwrap();
        let dict = sorted_set_source_dict(&dvd, entry).unwrap();
        assert_eq!(
            ords.iter()
                .map(|&o| dict[o as usize].clone())
                .collect::<Vec<_>>(),
            vec![expected.to_vec()]
        );
    }
}

// --- the sorted merge's document order, and the doc maps derived from it ---

/// `build_doc_id_maps` must be the exact inverse of `doc_order`: merged
/// doc `m` is `doc_order[m] == (s, d)` iff `maps[s][d] == m`.
///
/// This is the single most load-bearing invariant in a sorted merge.
/// Stored fields, doc values, norms and term vectors are written by
/// *walking* `doc_order`; postings, points and vectors are written by
/// *looking up* the map. If the two disagree by so much as one document,
/// every file is still well-formed, every checksum still valid, every doc
/// id still in range -- and a term, a point or a vector is attached to a
/// different document than its stored fields.
#[test]
fn the_doc_id_maps_invert_the_doc_order_for_both_merge_orders() {
    let live = vec![vec![0i32, 2, 3], vec![1], vec![], vec![0, 1]];
    let max_doc = [4i32, 3, 2, 2];
    let concat = concat_doc_order(&live);

    // A two-tier sort whose keys interleave the sources, with a missing
    // value and a tie the second tier has to break.
    let rank: Vec<Vec<Option<i64>>> = vec![
        vec![Some(5), Some(9), Some(1), None],
        vec![Some(9), Some(3), Some(7)],
        vec![Some(0), Some(0)],
        vec![Some(1), Some(5)],
    ];
    let tie: Vec<Vec<Option<i64>>> = vec![
        vec![Some(0), Some(1), Some(2), Some(3)],
        vec![Some(0), Some(1), Some(2)],
        vec![Some(0), Some(1)],
        vec![Some(9), Some(8)],
    ];
    let rank_slices: Vec<&[Option<i64>]> = rank.iter().map(|v| v.as_slice()).collect();
    let tie_slices: Vec<&[Option<i64>]> = tie.iter().map(|v| v.as_slice()).collect();
    let sort_rank = IndexSortField::long("rank", true, Some(i64::MAX));
    let sort_tie = IndexSortField::long("tie", false, Some(i64::MIN));
    let specs = vec![
        MergeSortKeySpec {
            sort: &sort_rank,
            per_source_keys: &rank_slices,
        },
        MergeSortKeySpec {
            sort: &sort_tie,
            per_source_keys: &tie_slices,
        },
    ];
    let sorted = sorted_doc_order(&specs, &live);

    for order in [&concat, &sorted] {
        assert_eq!(order.len(), 6, "every live document appears exactly once");
        let maps = build_doc_id_maps(&max_doc, order);
        for (merged, &(src, doc)) in order.iter().enumerate() {
            assert_eq!(
                mapped_doc_id(&maps[src], doc),
                Some(merged as i32),
                "source {src} doc {doc}"
            );
        }
        // Every doc the order does *not* name maps to nothing.
        let named: HashSet<(usize, i32)> = order.iter().copied().collect();
        for (src, &max) in max_doc.iter().enumerate() {
            for doc in 0..max {
                if !named.contains(&(src, doc)) {
                    assert_eq!(mapped_doc_id(&maps[src], doc), None, "src {src} doc {doc}");
                }
            }
        }
        // Within one source the map is increasing -- what lets
        // `merge_postings` treat a source's contribution to a term as
        // already ascending, and what `merge_one_flat_vector_field`
        // enforces.
        for (src, &max) in max_doc.iter().enumerate() {
            let mapped: Vec<i32> = (0..max)
                .filter_map(|d| mapped_doc_id(&maps[src], d))
                .collect();
            assert!(
                mapped.windows(2).all(|w| w[0] < w[1]),
                "source {src}'s doc map is not increasing: {mapped:?}"
            );
        }
    }
    // ...and the two orders really are different, or the check above
    // would be vacuous.
    assert_ne!(concat, sorted);
}

/// `MultiSorter.sort`'s comparator, tier by tier: the reversed first tier
/// takes its missing documents to the *front* (the sentinel is
/// `Long.MAX_VALUE` and reverse applies to it too), the second tier
/// breaks ties, and source index then doc id break what is left.
#[test]
fn the_sorted_doc_order_is_multi_tier_sentinel_reversing_and_stable() {
    // Each source is already in the merged order (the precondition):
    // source 0 is [missing, 5], source 1 is [7, 5], both descending once
    // the missing value takes its `Long.MAX_VALUE` sentinel.
    let live = vec![vec![0i32, 1], vec![0, 1]];
    let rank: Vec<Vec<Option<i64>>> = vec![vec![None, Some(5)], vec![Some(7), Some(5)]];
    let rank_slices: Vec<&[Option<i64>]> = rank.iter().map(|v| v.as_slice()).collect();
    let sort_rank_5 = IndexSortField::long("rank", true, Some(i64::MAX));
    let specs = vec![MergeSortKeySpec {
        sort: &sort_rank_5,
        per_source_keys: &rank_slices,
    }];
    assert_eq!(
        sorted_doc_order(&specs, &live),
        // missing (MAX, and `reverse` applies to the sentinel too, so it
        // comes *first* under a missing-**last** descending sort), then
        // 7, then the two 5s with the source index breaking the tie.
        vec![(0, 0), (1, 0), (0, 1), (1, 1)]
    );
}

/// A mis-shaped sort-key table is a named error, not a panic: the entry
/// point is `pub`, and the shapes are easy to get wrong (the outer list is
/// per *source*, each inner slice per *document of that source* -- not per
/// live document, because it is indexed by pre-merge doc id).
#[test]
fn a_mis_shaped_sort_key_table_is_reported_rather_than_panicking() {
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
    let reader0 = open_reader(&seg0);
    let sources = vec![MergeSource::stored_only(
        &seg0.fields,
        &reader0,
        None,
        Some(version()),
    )];

    let merge = |specs: &[MergeSortKeySpec<'_>]| {
        merge_segments(
            &dir,
            &sources,
            Some(specs),
            &MergeOptions::default(),
            "_merged_badsort",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        )
    };
    assert!(matches!(merge(&[]), Err(Error::EmptySortFields)));

    // The outer list has no entry for the one source.
    let empty: Vec<&[Option<i64>]> = Vec::new();
    let sort_rank = IndexSortField::long("rank", false, Some(i64::MAX));
    assert!(matches!(
        merge(&[MergeSortKeySpec {
            sort: &sort_rank,
            per_source_keys: &empty
        }]),
        Err(Error::SortKeysWrongLength {
            source_index: None,
            expected: 1,
            found: 0,
            ..
        })
    ));

    // One entry per *live* document instead of per document -- the shape
    // a caller who filtered deletions first would naturally build.
    let short: Vec<Option<i64>> = vec![Some(1)];
    let short_slices: Vec<&[Option<i64>]> = vec![short.as_slice()];
    assert!(matches!(
        merge(&[MergeSortKeySpec {
            sort: &sort_rank,
            per_source_keys: &short_slices
        }]),
        Err(Error::SortKeysWrongLength {
            source_index: Some(0),
            expected: 2,
            found: 1,
            ..
        })
    ));
}

fn occ<'a>(
    positions: &'a [i32],
    start_offsets: &'a [i32],
    end_offsets: &'a [i32],
    payload_lengths: &'a [u32],
    payload_bytes: &'a [u8],
) -> Option<lucene_codecs::postings::DocOccurrences<'a>> {
    Some(lucene_codecs::postings::DocOccurrences {
        positions,
        start_offsets,
        end_offsets,
        payload_lengths,
        payload_bytes,
    })
}

#[test]
fn merged_term_flat_slices_and_absent_extras() {
    let mut run = MergedRun::new(true, false, false);
    run.start_term(b"term");
    run.push_document(1, 2, occ(&[3, 7], &[], &[], &[], &[]));
    run.push_document(4, 1, occ(&[9], &[], &[], &[], &[]));
    let positions_ptr = run.positions.as_ptr();
    run.finish_term().unwrap();
    // Already in order: nothing is rebuilt.
    assert_eq!(run.positions.as_ptr(), positions_ptr);
    let term = run.views().next().unwrap();
    assert_eq!(term.term(), b"term");
    assert_eq!(term.docs(), &[(1, 2), (4, 1)]);
    assert_eq!(term.position_lists(), 2);
    assert_eq!(term.positions(0), &[3, 7]);
    assert_eq!(term.positions(1), &[9]);
    assert!(term.positions(2).is_empty());
    assert!(term.positions(usize::MAX).is_empty());
    assert_eq!(term.offset_lists(), 0);
    assert!(term.offsets(0).is_empty());
    assert!(term.payload_bytes().is_empty());
    assert!(term.payload_lengths().is_empty());
}

#[test]
fn merged_term_empty_and_docs_only() {
    // A term no live document reached leaves nothing behind.
    let mut run = MergedRun::new(true, true, true);
    run.start_term(b"gone");
    run.finish_term().unwrap();
    assert!(run.term_bytes.is_empty());
    assert_eq!(run.views().count(), 0);
    let mut docs = MergedRun::new(false, false, false);
    docs.start_term(b"docs");
    docs.push_document(4, 3, None);
    docs.push_document(1, 2, None);
    docs.finish_term().unwrap();
    let term = docs.views().next().unwrap();
    assert_eq!(term.docs(), &[(1, 2), (4, 3)]);
    assert_eq!(term.position_lists(), 0);
    assert_eq!(term.offset_lists(), 0);
    assert!(term.positions(0).is_empty());
    assert!(term.offsets(0).is_empty());
    assert!(docs.pos_ends.is_empty());
}

#[test]
fn merged_term_sorted_spans_keep_offsets_and_mixed_payloads_aligned() {
    let mut run = MergedRun::new(true, true, true);
    // A first term, so the sorted one starts mid-buffer.
    run.start_term(b"first");
    run.push_document(0, 1, occ(&[0], &[0], &[1], &[1], b"z"));
    run.finish_term().unwrap();
    run.start_term(b"mixed");
    run.push_document(2, 2, occ(&[1, 3], &[10, 30], &[12, 35], &[1, 2], b"abc"));
    run.push_document(3, 1, occ(&[5], &[50], &[55], &[], &[]));
    run.push_document(
        1,
        3,
        occ(&[2, 4, 8], &[20, 40, 80], &[22, 45, 88], &[0, 1, 0], b"d"),
    );
    // A three-cycle catches a permutation accidentally applied backwards.
    run.finish_term().unwrap();
    let views: Vec<_> = run.views().collect();
    let (first, term) = (views[0], views[1]);
    assert_eq!(first.positions(0), &[0]);
    assert_eq!(first.payload_bytes(), b"z");
    assert_eq!(term.term(), b"mixed");
    assert_eq!(term.docs(), &[(1, 3), (2, 2), (3, 1)]);
    assert_eq!(term.position_lists(), 3);
    assert_eq!(term.offset_lists(), 3);
    assert_eq!(term.positions(0), &[2, 4, 8]);
    assert_eq!(term.positions(1), &[1, 3]);
    assert_eq!(term.positions(2), &[5]);
    assert_eq!(term.offsets(0), &[(20, 22), (40, 45), (80, 88)]);
    assert_eq!(term.offsets(1), &[(10, 12), (30, 35)]);
    assert_eq!(term.offsets(2), &[(50, 55)]);
    assert!(term.offsets(3).is_empty());
    assert_eq!(term.payload_lengths(), &[0, 1, 0, 1, 2, 0]);
    assert_eq!(term.payload_bytes(), b"dabc");
    assert_eq!(run.pos_ends, &[1, 3, 5, 6]);
}

#[test]
fn merged_term_sorted_positions_without_offsets_or_payloads() {
    let mut run = MergedRun::new(true, false, false);
    run.start_term(b"positions");
    for (doc, positions) in [(2, &[5, 7][..]), (0, &[1][..])] {
        run.push_document(
            doc,
            positions.len() as i32,
            occ(positions, &[], &[], &[], &[]),
        );
    }
    run.finish_term().unwrap();
    let term = run.views().next().unwrap();
    assert_eq!(term.docs(), &[(0, 1), (2, 2)]);
    assert_eq!(term.positions(0), &[1]);
    assert_eq!(term.positions(1), &[5, 7]);
    assert!(term.offsets(1).is_empty());
    assert!(term.payload_lengths().is_empty());
}

#[test]
fn merged_term_sorted_zero_length_document_span() {
    let mut run = MergedRun::new(true, true, true);
    run.start_term(b"empty-span");
    run.push_document(2, 0, occ(&[], &[], &[], &[], &[]));
    run.push_document(0, 1, occ(&[4], &[8], &[10], &[1], b"p"));
    run.finish_term().unwrap();
    let term = run.views().next().unwrap();
    assert_eq!(term.docs(), &[(0, 1), (2, 0)]);
    assert_eq!(term.position_lists(), 2);
    assert_eq!(term.offset_lists(), 2);
    assert_eq!(term.positions(0), &[4]);
    assert_eq!(term.offsets(0), &[(8, 10)]);
    assert!(term.positions(1).is_empty());
    assert!(term.offsets(1).is_empty());
    assert_eq!(run.pos_ends, &[1, 1]);
    assert_eq!(term.payload_lengths(), &[1]);
    assert_eq!(term.payload_bytes(), b"p");
}

// --- postings ---

fn postings_field(name: &str, number: i32) -> FieldInfo {
    let mut f = field(name, number);
    f.index_options = IndexOptions::DocsAndFreqs;
    f
}

#[test]
fn two_sources_no_deletions_merge_postings_correctly() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // Source 0: 2 docs -- doc 0 has "apple" (freq 2), doc 1 has "banana"
    // (freq 1).
    let terms0 = vec![
        TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 2)],
            ..Default::default()
        },
        TermPostings {
            term: b"banana".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        },
    ];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    // Source 1: 1 doc -- doc 0 has "cherry" (freq 3).
    let terms1 = vec![TermPostings {
        term: b"cherry".to_vec(),
        docs: vec![(0, 3)],
        ..Default::default()
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos1 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let fields = vec![postings_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_postings",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_postings");

    let tim = std::fs::read(tmp.join(format!(
        "{}.tim",
        per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tip = std::fs::read(tmp.join(format!(
        "{}.tip",
        per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tmd = std::fs::read(tmp.join(format!(
        "{}.tmd",
        per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let doc = std::fs::read(tmp.join(format!(
        "{}.doc",
        per_field_segment("_merged_postings", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        3,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    let apple = merged_terms
        .postings(b"apple", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(apple.docs, vec![0]);
    assert_eq!(apple.freqs, vec![2]);

    let banana = merged_terms
        .postings(b"banana", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(banana.docs, vec![1]);
    assert_eq!(banana.freqs, vec![1]);

    // "cherry" only existed in source 1's doc 0, which is renumbered to
    // merged doc 2 (after source 0's 2 docs).
    let cherry = merged_terms
        .postings(b"cherry", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(cherry.docs, vec![2]);
    assert_eq!(cherry.freqs, vec![3]);
}

/// Task #212's coverage/scope review flagged that `IndexOptions::
/// DocsAndCustomFreqs` -- accepted by `merge_postings`'s supported-list
/// alongside `Docs`/`DocsAndFreqs`, wire-identical to `DocsAndFreqs` --
/// had zero merge test coverage despite being a real, reachable field
/// type since `IndexWriter::set_custom_freq_postings_field`. This proves
/// merging preserves the caller's opaque custom-freq values verbatim
/// (not re-derived as an occurrence count), the same way
/// [`two_sources_no_deletions_merge_postings_correctly`] proves it for
/// `DocsAndFreqs`.
#[test]
fn two_sources_no_deletions_merge_docs_and_custom_freqs_correctly() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // Source 0: 1 doc -- "score" with a custom freq of 50 (not an
    // occurrence count; the term appears once in the doc's stored text
    // below, so a bug that silently re-derived freq from occurrences
    // would produce 1, not 50).
    let terms0 = vec![TermPostings {
        term: b"score".to_vec(),
        docs: vec![(0, 50)],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndCustomFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let mut field0 = postings_field("body", 0);
    field0.index_options = IndexOptions::DocsAndCustomFreqs;
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![field0.clone()],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    // Source 1: 1 doc -- same term "score", custom freq of 5.
    let terms1 = vec![TermPostings {
        term: b"score".to_vec(),
        docs: vec![(0, 5)],
        ..Default::default()
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndCustomFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos0,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let fields = vec![field0];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_custom_freqs",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let tim = std::fs::read(tmp.join(format!(
        "{}.tim",
        per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tip = std::fs::read(tmp.join(format!(
        "{}.tip",
        per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tmd = std::fs::read(tmp.join(format!(
        "{}.tmd",
        per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let doc = std::fs::read(tmp.join(format!(
        "{}.doc",
        per_field_segment("_merged_custom_freqs", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &field_infos0,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    let score = merged_terms
        .postings(b"score", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(score.docs, vec![0, 1]);
    // The merged freqs are the two sources' custom values verbatim, in
    // merged-doc-id order -- not re-derived from occurrence counts
    // (which would both be 1).
    assert_eq!(score.freqs, vec![50, 5]);
}

#[test]
fn term_across_multiple_sources_merges_in_doc_id_order() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // Both sources index "the": source 0 doc 0 (freq 1), source 1 doc 0
    // (freq 2) -- the merged term must contain both docs in ascending
    // merged-doc-id order (source 0's docs first).
    let terms0 = vec![TermPostings {
        term: b"the".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let terms1 = vec![TermPostings {
        term: b"the".to_vec(),
        docs: vec![(0, 2)],
        ..Default::default()
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos1 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let fields = vec![postings_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let tmp2 = tmp.path().to_path_buf();
    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_the",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let tim = std::fs::read(std::path::Path::new(&tmp2).join(format!(
        "{}.tim",
        per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tip = std::fs::read(std::path::Path::new(&tmp2).join(format!(
        "{}.tip",
        per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tmd = std::fs::read(std::path::Path::new(&tmp2).join(format!(
        "{}.tmd",
        per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let doc = std::fs::read(std::path::Path::new(&tmp2).join(format!(
        "{}.doc",
        per_field_segment("_merged_the", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    let the = merged_terms
        .postings(b"the", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(the.docs, vec![0, 1]);
    assert_eq!(the.freqs, vec![1, 2]);
}

#[test]
fn deletions_drop_docs_from_merged_postings() {
    let seg0_id = [1u8; ID_LENGTH];

    // Source 0: 2 docs -- doc 0 has "apple", doc 1 has "banana"; doc 1 is
    // deleted, so "banana" must not survive the merge at all.
    let terms0 = vec![
        TermPostings {
            term: b"apple".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        },
        TermPostings {
            term: b"banana".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        },
    ];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 2,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let fields = vec![postings_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);

    let mut live0 = FixedBitSet::new(2);
    live0.set(0); // keep doc 0 only

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_del",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let tim = std::fs::read(tmp.join(format!(
        "{}.tim",
        per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tip = std::fs::read(tmp.join(format!(
        "{}.tip",
        per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tmd = std::fs::read(tmp.join(format!(
        "{}.tmd",
        per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let doc = std::fs::read(tmp.join(format!(
        "{}.doc",
        per_field_segment("_merged_del", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    let apple = merged_terms
        .postings(b"apple", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(apple.docs, vec![0]);

    assert!(merged_terms.seek_exact(b"banana").is_none());
}

#[test]
fn fully_deleted_source_contributes_nothing_to_postings() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // Source 0: 1 doc, fully deleted -- its "ghost" term must not survive.
    let terms0 = vec![TermPostings {
        term: b"ghost".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    // Source 1: 1 doc, alive -- "alive" survives.
    let terms1 = vec![TermPostings {
        term: b"alive".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos1 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let fields = vec![postings_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let live0 = FixedBitSet::new(1); // no bits set -- fully deleted

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_fully_deleted",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let tim = std::fs::read(tmp.join(format!(
        "{}.tim",
        per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tip = std::fs::read(tmp.join(format!(
        "{}.tip",
        per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tmd = std::fs::read(tmp.join(format!(
        "{}.tmd",
        per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let doc = std::fs::read(tmp.join(format!(
        "{}.doc",
        per_field_segment("_merged_fully_deleted", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    assert!(merged_terms.seek_exact(b"ghost").is_none());
    let alive = merged_terms
        .postings(b"alive", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(alive.docs, vec![0]);
}

#[test]
fn a_source_that_never_saw_the_postings_field_simply_contributes_no_terms() {
    // `FieldsConsumer.merge`: `Terms terms = fields.terms(field); if
    // (terms == null) continue;`. A segment written before the field
    // existed has no `FieldInfo` for it and contributes no terms --
    // that must not fail the merge (declaring the same field with
    // *different* index_options is a separate matter, rejected up front
    // by `reconcile_field_numbers`).
    let seg0_id = [1u8; ID_LENGTH];
    let terms0 = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &[postings_field("body", 0)],
        &[doc_with(0, "a")],
    );
    // Source 1 never saw "body" at all.
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &[field("id", 0)],
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        postings: &src_postings0,
        ..MergeSource::stored_only(&stored0.fields, &reader0, None, Some(version()))
    };
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_postings_absent_source",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let read = |ext: &str| {
        std::fs::read(tmp.join(format!(
            "{}.{ext}",
            per_field_segment("_merged_postings_absent_source", POSTINGS_FORMAT_NAME)
        )))
        .unwrap()
    };
    let (tim, tip, tmd, doc) = (read("tim"), read("tip"), read("tmd"), read("doc"));
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();
    let apple = merged_terms
        .postings(b"apple", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(
        apple.docs,
        vec![0],
        "only source 0's doc has the term; source 1 contributes none"
    );
}

#[test]
fn a_source_declaring_a_field_without_terms_contributes_none() {
    // `MultiFields`: a segment whose every document had an empty value
    // declares the field in its `.fnm` and has no terms for it.
    let seg0_id = [1u8; ID_LENGTH];

    let terms0 = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let fields = vec![postings_field("body", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    // Source 1 declares the same "body" field but supplies no postings
    // data for it at all -- a schema mismatch, since source 0 has live
    // docs indexing that field.
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_missing_postings",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let read = |ext: &str| {
        std::fs::read(tmp.join(format!(
            "{}.{ext}",
            per_field_segment("_merged_missing_postings", POSTINGS_FORMAT_NAME)
        )))
        .unwrap()
    };
    let (tim, tip, tmd, doc) = (read("tim"), read("tip"), read("tmd"), read("doc"));
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let apple = merged_fields
        .field("body")
        .unwrap()
        .postings(b"apple", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(
        apple.docs,
        vec![0],
        "source 1 declares body but has no terms"
    );
}

#[test]
fn two_sources_merge_positions_offsets_and_payloads_correctly() {
    // End-to-end: two segments, each with a field indexing positions,
    // offsets, and payloads, merged into one. Verifies the merged term
    // dictionary's positions/offsets/payloads are queryable post-merge
    // via `FieldTerms::positions`, the same read path phrase matching
    // uses.
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    // Source 0: 1 doc -- doc 0 has "apple" at positions 0 and 2, with
    // offsets and a payload on the second occurrence only.
    let terms0 = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 2)],
        positions: vec![vec![0, 2]],
        offsets: vec![vec![(0, 5), (10, 15)]],
        payload_bytes: b"pay0".to_vec(),
        payload_lengths: vec![0, 4],
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 1,
        has_payloads: true,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let mut field_with_positions = postings_field("body", 0);
    field_with_positions.index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    field_with_positions.store_payloads = true;
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![field_with_positions.clone()],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pos_in0 = lucene_codecs::postings::PosInput::open(
        &output0.pos,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pay_in0 = lucene_codecs::postings::PayInput::open(
        &output0.pay,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    // Source 1: 1 doc -- doc 0 has "banana" at position 1, no payload.
    let terms1 = vec![TermPostings {
        term: b"banana".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![1]],
        offsets: vec![vec![(6, 12)]],
        payload_bytes: Vec::new(),
        payload_lengths: vec![0],
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        doc_count: 1,
        has_payloads: true,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos1 = field_infos::FieldInfos {
        fields: vec![field_with_positions.clone()],
    };
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pos_in1 = lucene_codecs::postings::PosInput::open(
        &output1.pos,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pay_in1 = lucene_codecs::postings::PayInput::open(
        &output1.pay,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let fields = vec![field_with_positions.clone()];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: Some(&pos_in0),
        pay_in: Some(&pay_in0),
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: Some(&pos_in1),
        pay_in: Some(&pay_in1),
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_positions",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_positions");

    let tim = std::fs::read(tmp.join(format!(
        "{}.tim",
        per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tip = std::fs::read(tmp.join(format!(
        "{}.tip",
        per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let tmd = std::fs::read(tmp.join(format!(
        "{}.tmd",
        per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let doc = std::fs::read(tmp.join(format!(
        "{}.doc",
        per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let pos = std::fs::read(tmp.join(format!(
        "{}.pos",
        per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let pay = std::fs::read(tmp.join(format!(
        "{}.pay",
        per_field_segment("_merged_positions", POSTINGS_FORMAT_NAME)
    )))
    .unwrap();
    let merged_field_infos = field_infos::FieldInfos {
        fields: vec![field_with_positions],
    };
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_field_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_pos_in = lucene_codecs::postings::PosInput::open(
        &pos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_pay_in = lucene_codecs::postings::PayInput::open(
        &pay,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    let apple = merged_terms
        .positions(
            b"apple",
            Some(&merged_doc_in),
            &merged_pos_in,
            Some(&merged_pay_in),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        apple.len(),
        1,
        "apple occurs in exactly 1 doc (merged doc 0)"
    );
    assert_eq!(
        apple[0]
            .iter()
            .map(|p| (p.position, p.start_offset, p.end_offset, p.payload.clone()))
            .collect::<Vec<_>>(),
        vec![(0, 0, 5, Vec::new()), (2, 10, 15, b"pay0".to_vec()),]
    );

    // "banana" only existed in source 1's doc 0, renumbered to merged
    // doc 1 (after source 0's 1 doc).
    let banana = merged_terms
        .positions(
            b"banana",
            Some(&merged_doc_in),
            &merged_pos_in,
            Some(&merged_pay_in),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        banana.len(),
        1,
        "banana occurs in exactly 1 doc (merged doc 1)"
    );
    assert_eq!(
        banana[0]
            .iter()
            .map(|p| (p.position, p.start_offset, p.end_offset, p.payload.clone()))
            .collect::<Vec<_>>(),
        vec![(1, 6, 12, Vec::new())]
    );

    let apple_docs = merged_terms
        .postings(b"apple", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(apple_docs.docs, vec![0]);
    let banana_docs = merged_terms
        .postings(b"banana", Some(&merged_doc_in))
        .unwrap()
        .unwrap();
    assert_eq!(banana_docs.docs, vec![1]);
}

#[test]
fn a_source_with_positions_disagreeing_with_the_merged_docs_and_freqs_field_is_rejected() {
    // Source 0's "body" is Docs/DocsAndFreqs (first-seen, so it becomes
    // the merged field's canonical index_options via
    // `reconcile_field_numbers`). Source 1's own "body" indexes
    // positions. Without cross-source validation this would silently
    // drop source 1's positions data instead of erroring -- regression
    // test for `Error::PostingsIndexOptionsDisagreement`.
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let terms0 = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 1)],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqs,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![postings_field("body", 0)],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let terms1 = vec![TermPostings {
        term: b"banana".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        ..Default::default()
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let mut field_with_positions = postings_field("body", 0);
    field_with_positions.index_options = IndexOptions::DocsAndFreqsAndPositions;
    let field_infos1 = field_infos::FieldInfos {
        fields: vec![field_with_positions.clone()],
    };
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &[postings_field("body", 0)],
        &[doc_with(0, "a")],
    );
    let reader0 = open_reader(&stored0);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        seg1_id,
        &[field_with_positions],
        &[doc_with(0, "a")],
    );
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let result = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_disagreeing_index_options",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    );
    assert!(matches!(
        result,
        Err(Error::PostingsIndexOptionsDisagreement { .. })
    ));
}

#[test]
fn a_source_with_payloads_ors_into_the_merged_fields_store_payloads() {
    // Both sources index positions (same index_options), but source 1's
    // "body" also stores payloads while source 0's (first-seen, so it
    // seeds the merged FieldInfo) doesn't. Real Lucene's
    // `FieldInfos.Builder.add` ORs `hasPayloads` rather than rejecting
    // the mismatch (`if (fi.hasPayloads()) curFi.setStorePayloads()`),
    // so the merged field must store payloads and source 1's real
    // payload bytes must survive, while source 0's payload-free
    // occurrences come through as empty payloads (which the postings
    // writer treats as "no payload at this occurrence").
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let terms0 = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let mut field0 = postings_field("body", 0);
    field0.index_options = IndexOptions::DocsAndFreqsAndPositions;
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![field0.clone()],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pos_in0 = lucene_codecs::postings::PosInput::open(
        &output0.pos,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let terms1 = vec![TermPostings {
        term: b"banana".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        payload_bytes: b"pay".to_vec(),
        payload_lengths: vec![3],
        ..Default::default()
    }];
    let input1 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: true,
        terms: &terms1,
    };
    let output1 = postings_writer::write_single_field(
        &input1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let mut field1 = postings_field("body", 0);
    field1.index_options = IndexOptions::DocsAndFreqsAndPositions;
    field1.store_payloads = true;
    let field_infos1 = field_infos::FieldInfos {
        fields: vec![field1.clone()],
    };
    let fields1 = lucene_codecs::blocktree::open(
        &output1.tim,
        &output1.tip,
        &output1.tmd,
        &field_infos1,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in1 = DocInput::open(
        &output1.doc,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let pos_in1 = lucene_codecs::postings::PosInput::open(
        &output1.pos,
        &seg1_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms1 = fields1.field("body").unwrap();

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &[field0], &[doc_with(0, "a")]);
    let reader0 = open_reader(&stored0);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &[field1], &[doc_with(0, "a")]);
    let reader1 = open_reader(&stored1);

    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: Some(&pos_in0),
        pay_in: None,
        live: None,
    }];
    let src_postings1 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms1,
        doc_in: Some(&doc_in1),
        pos_in: Some(&pos_in1),
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings1,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_ored_payloads",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    // The merged .fnm must record store_payloads = true (the OR), not
    // the first-seen source's `false`.
    let fnm = std::fs::read(tmp.join("_merged_ored_payloads.fnm")).unwrap();
    let merged_infos = field_infos::parse(&fnm, &[9u8; ID_LENGTH], "").unwrap();
    let merged_body = merged_infos
        .fields
        .iter()
        .find(|f| f.name == "body")
        .unwrap();
    assert!(
        merged_body.store_payloads,
        "store_payloads must be ORed across sources"
    );

    let read = |ext: &str| {
        std::fs::read(tmp.join(format!(
            "{}.{ext}",
            per_field_segment("_merged_ored_payloads", POSTINGS_FORMAT_NAME)
        )))
        .unwrap()
    };
    let (tim, tip, tmd, doc, pos, pay) = (
        read("tim"),
        read("tip"),
        read("tmd"),
        read("doc"),
        read("pos"),
        read("pay"),
    );
    let merged_fields = lucene_codecs::blocktree::open(
        &tim,
        &tip,
        &tmd,
        &merged_infos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        2,
    )
    .unwrap();
    let merged_doc_in = DocInput::open(
        &doc,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_pos_in = lucene_codecs::postings::PosInput::open(
        &pos,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_pay_in = lucene_codecs::postings::PayInput::open(
        &pay,
        &[9u8; ID_LENGTH],
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let merged_terms = merged_fields.field("body").unwrap();

    // "apple" came from the payload-free source: empty payload.
    let apple = merged_terms
        .positions(
            b"apple",
            Some(&merged_doc_in),
            &merged_pos_in,
            Some(&merged_pay_in),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        apple[0]
            .iter()
            .map(|p| (p.position, p.payload.clone()))
            .collect::<Vec<_>>(),
        vec![(0, Vec::new())]
    );

    // "banana" came from the payload-bearing source: bytes preserved.
    let banana = merged_terms
        .positions(
            b"banana",
            Some(&merged_doc_in),
            &merged_pos_in,
            Some(&merged_pay_in),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        banana[0]
            .iter()
            .map(|p| (p.position, p.payload.clone()))
            .collect::<Vec<_>>(),
        vec![(0, b"pay".to_vec())]
    );
}

#[test]
fn a_positions_indexing_source_missing_an_opened_pos_reader_is_a_clean_error_not_a_panic() {
    // Reuses source0's fixture from the payloads-disagreement test above
    // (a real DocsAndFreqsAndPositions field), but the caller "forgets"
    // to open pos_in -- proves this is a typed
    // `Error::PostingsPositionsInputMissingInSource`, not an `expect()`
    // panic deep inside the per-term merge loop.
    let seg0_id = [1u8; ID_LENGTH];

    let terms0 = vec![TermPostings {
        term: b"apple".to_vec(),
        docs: vec![(0, 1)],
        positions: vec![vec![0]],
        ..Default::default()
    }];
    let input0 = FieldPostingsInput {
        field_number: 0,
        index_options: IndexOptions::DocsAndFreqsAndPositions,
        doc_count: 1,
        has_payloads: false,
        terms: &terms0,
    };
    let output0 = postings_writer::write_single_field(
        &input0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let mut field0 = postings_field("body", 0);
    field0.index_options = IndexOptions::DocsAndFreqsAndPositions;
    let field_infos0 = field_infos::FieldInfos {
        fields: vec![field0.clone()],
    };
    let fields0 = lucene_codecs::blocktree::open(
        &output0.tim,
        &output0.tip,
        &output0.tmd,
        &field_infos0,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
        1,
    )
    .unwrap();
    let doc_in0 = DocInput::open(
        &output0.doc,
        &seg0_id,
        &per_field_codec_suffix(POSTINGS_FORMAT_NAME),
    )
    .unwrap();
    let field_terms0 = fields0.field("body").unwrap();

    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &[field0], &[doc_with(0, "a")]);
    let reader0 = open_reader(&stored0);

    // pos_in deliberately left None despite the field indexing positions.
    let src_postings0 = [SourcePostings {
        field_number: 0,
        field_terms: field_terms0,
        doc_in: Some(&doc_in0),
        pos_in: None,
        pay_in: None,
        live: None,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &src_postings0,
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let result = merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_missing_pos_input",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    );
    assert!(matches!(
        result,
        Err(Error::PostingsPositionsInputMissingInSource { .. })
    ));
}

fn points_field(name: &str, number: i32, num_dims: i32, bytes_per_dim: i32) -> FieldInfo {
    points_field_with_index_dims(name, number, num_dims, num_dims, bytes_per_dim)
}

fn points_field_with_index_dims(
    name: &str,
    number: i32,
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
) -> FieldInfo {
    let mut f = field(name, number);
    f.point_dimension_count = num_dims;
    f.point_index_dimension_count = num_index_dims;
    f.point_num_bytes = bytes_per_dim;
    f
}

fn packed4(v: u32) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

fn write_one_field_points(
    field_number: i32,
    num_dims: i32,
    bytes_per_dim: i32,
    points: Vec<(i32, Vec<u8>)>,
    segment_id: &[u8; ID_LENGTH],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    write_one_field_points_with_index_dims(
        field_number,
        num_dims,
        num_dims,
        bytes_per_dim,
        points,
        segment_id,
    )
}

fn write_one_field_points_with_index_dims(
    field_number: i32,
    num_dims: i32,
    num_index_dims: i32,
    bytes_per_dim: i32,
    points: Vec<(i32, Vec<u8>)>,
    segment_id: &[u8; ID_LENGTH],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let fields = vec![points::WritePointsField {
        field_number,
        num_dims,
        num_index_dims,
        bytes_per_dim,
        points,
    }];
    points::write(
        &fields,
        points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        segment_id,
        "",
    )
    .unwrap()
}

#[test]
fn two_sources_no_deletions_merge_points_correctly() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let (kdm0, kdi0, kdd0) =
        write_one_field_points(0, 1, 4, vec![(0, packed4(10)), (1, packed4(20))], &seg0_id);
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

    let (kdm1, kdi1, kdd1) = write_one_field_points(0, 1, 4, vec![(0, packed4(30))], &seg1_id);
    let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

    let fields = vec![points_field("loc", 0, 1, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let src_points1 = [SourcePoints {
        field_number: 0,
        reader: &points_reader1,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points1,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_points",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_points");

    let kdm = std::fs::read(tmp.join("_merged_points.kdm")).unwrap();
    let kdi = std::fs::read(tmp.join("_merged_points.kdi")).unwrap();
    let kdd = std::fs::read(tmp.join("_merged_points.kdd")).unwrap();
    let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();

    let mut merged_points = merged_reader.decode_all_points(0).unwrap();
    merged_points.sort_by_key(|p| p.doc_id);
    assert_eq!(
        merged_points
            .iter()
            .map(|p| (p.doc_id, p.packed_value.clone()))
            .collect::<Vec<_>>(),
        vec![
            (0, packed4(10)),
            (1, packed4(20)),
            // Source 1's only doc is renumbered to merged doc 2, after
            // source 0's 2 docs.
            (2, packed4(30)),
        ]
    );

    // Full round-trip: a range query through the unmodified points-range
    // resolver (the same one `lucene_search`'s points range query
    // composes) must return exactly the merged docs whose values fall in
    // range, using the real reader/decoder stack end to end.
    let in_range = crate::points_delete::resolve_points_range_doc_ids(
        &merged_reader,
        None,
        0,
        &packed4(15),
        &packed4(30),
    )
    .unwrap();
    assert_eq!(in_range, vec![1, 2]);
}

#[test]
fn points_field_with_deletions_drops_non_live_docs() {
    let seg0_id = [1u8; ID_LENGTH];

    let (kdm0, kdi0, kdd0) = write_one_field_points(
        0,
        1,
        4,
        vec![(0, packed4(10)), (1, packed4(20)), (2, packed4(30))],
        &seg0_id,
    );
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

    let fields = vec![points_field("loc", 0, 1, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b"), doc_with(0, "c")],
    );
    let reader0 = open_reader(&stored0);

    // Drop doc 1 -- surviving docs 0 and 2 renumber to merged 0 and 1.
    let mut live0 = FixedBitSet::new(3);
    live0.set(0);
    live0.set(2);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: Some(&live0),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    merge_stored_only_segments(
        &dir,
        &[source0],
        "_merged_points_deletions",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let kdm = std::fs::read(tmp.join("_merged_points_deletions.kdm")).unwrap();
    let kdi = std::fs::read(tmp.join("_merged_points_deletions.kdi")).unwrap();
    let kdd = std::fs::read(tmp.join("_merged_points_deletions.kdd")).unwrap();
    let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();
    let mut merged_points = merged_reader.decode_all_points(0).unwrap();
    merged_points.sort_by_key(|p| p.doc_id);
    assert_eq!(
        merged_points
            .iter()
            .map(|p| (p.doc_id, p.packed_value.clone()))
            .collect::<Vec<_>>(),
        vec![(0, packed4(10)), (1, packed4(30))]
    );
}

#[test]
fn fully_deleted_source_contributes_no_points() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();
    let (kdm1, kdi1, kdd1) = write_one_field_points(0, 1, 4, vec![(0, packed4(99))], &seg1_id);
    let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

    let fields = vec![points_field("loc", 0, 1, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let all_deleted = FixedBitSet::new(1); // source 1: nothing live

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let src_points1 = [SourcePoints {
        field_number: 0,
        reader: &points_reader1,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: Some(&all_deleted),
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points1,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sources = [source0, source1];
    let (merged_fields, maps) =
        reconcile_field_numbers(&[&stored0.fields, &stored1.fields]).unwrap();
    let merged_points = merge_points(
        &sources,
        &maps,
        &[vec![0], vec![]],
        &[vec![0], vec![-1]],
        &merged_fields,
        None,
    )
    .unwrap();
    assert_eq!(merged_points[0].max_point_count, 2);
    assert_eq!(merged_points[0].sources.len(), 2);
    assert!(merged_points[0].sources[1].docs.is_empty());

    merge_stored_only_segments(
        &dir,
        &sources,
        "_merged_points_fully_deleted",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let kdm = std::fs::read(tmp.join("_merged_points_fully_deleted.kdm")).unwrap();
    let kdi = std::fs::read(tmp.join("_merged_points_fully_deleted.kdi")).unwrap();
    let kdd = std::fs::read(tmp.join("_merged_points_fully_deleted.kdd")).unwrap();
    let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();
    let merged_points = merged_reader.decode_all_points(0).unwrap();
    assert_eq!(
        merged_points
            .iter()
            .map(|p| (p.doc_id, p.packed_value.clone()))
            .collect::<Vec<_>>(),
        vec![(0, packed4(10))]
    );
}

#[test]
fn deleting_every_point_keeps_the_shape_empty_files_and_null_entry_on_remerge() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let source_id = [1u8; ID_LENGTH];
    let merged_id = [9u8; ID_LENGTH];
    let fields = vec![points_field("loc", 0, 1, 4)];
    let stored = flush(
        &dir,
        &tmp,
        "_source",
        source_id,
        &fields,
        &[doc_with(0, "deleted"), doc_with(0, "live_without_points")],
    );
    let stored_reader = open_reader(&stored);
    let (meta, index, data) = write_one_field_points(0, 1, 4, vec![(0, packed4(42))], &source_id);
    let points_reader = points::open(&meta, &index, &data, &source_id, "").unwrap();
    let source_points = [SourcePoints {
        field_number: 0,
        reader: &points_reader,
    }];
    let mut live = FixedBitSet::new(2);
    live.set(1);
    let source = MergeSource {
        points: &source_points,
        ..MergeSource::stored_only(&stored.fields, &stored_reader, Some(&live), Some(version()))
    };
    let merged = merge_stored_only_segments(
        &dir,
        &[source],
        "_empty_points",
        merged_id,
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(merged.segment_name, "_empty_points");
    for extension in ["kdm", "kdi", "kdd"] {
        assert!(tmp.join(format!("_empty_points.{extension}")).is_file());
    }
    let merged_fields = field_infos::parse(
        &std::fs::read(tmp.join("_empty_points.fnm")).unwrap(),
        &merged_id,
        "",
    )
    .unwrap();
    assert_eq!(merged_fields.fields[0].point_dimension_count, 1);
    assert_eq!(merged_fields.fields[0].point_index_dimension_count, 1);
    assert_eq!(merged_fields.fields[0].point_num_bytes, 4);
    let meta = std::fs::read(tmp.join("_empty_points.kdm")).unwrap();
    let index = std::fs::read(tmp.join("_empty_points.kdi")).unwrap();
    let data = std::fs::read(tmp.join("_empty_points.kdd")).unwrap();
    let empty_points = points::open(&meta, &index, &data, &merged_id, "").unwrap();
    assert!(empty_points.field(0).is_none());
    let merged_stored = FlushedSegment {
        fdt: std::fs::read(tmp.join("_empty_points.fdt")).unwrap(),
        fdx: std::fs::read(tmp.join("_empty_points.fdx")).unwrap(),
        fdm: std::fs::read(tmp.join("_empty_points.fdm")).unwrap(),
        fields: merged_fields.fields,
        segment_id: merged_id,
    };
    let merged_stored_reader = open_reader(&merged_stored);
    let empty_source_points = [SourcePoints {
        field_number: 0,
        reader: &empty_points,
    }];
    let source = MergeSource {
        points: &empty_source_points,
        ..MergeSource::stored_only(
            &merged_stored.fields,
            &merged_stored_reader,
            None,
            Some(version()),
        )
    };
    merge_stored_only_segments(
        &dir,
        &[source],
        "_empty_again",
        [8u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let meta = std::fs::read(tmp.join("_empty_again.kdm")).unwrap();
    let index = std::fs::read(tmp.join("_empty_again.kdi")).unwrap();
    let data = std::fs::read(tmp.join("_empty_again.kdd")).unwrap();
    let reopened = points::open(&meta, &index, &data, &[8u8; ID_LENGTH], "").unwrap();
    assert_eq!(reopened.field_numbers().count(), 0);
    let infos = field_infos::parse(
        &std::fs::read(tmp.join("_empty_again.fnm")).unwrap(),
        &[8u8; ID_LENGTH],
        "",
    )
    .unwrap();
    assert_eq!(infos.fields[0].point_dimension_count, 1);
    assert_eq!(infos.fields[0].point_num_bytes, 4);
}

#[test]
fn points_field_missing_in_a_live_contributing_source_is_an_error() {
    let seg0_id = [1u8; ID_LENGTH];

    let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

    let fields = vec![points_field("loc", 0, 1, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    // Source 1 declares the same "loc" field but supplies no points data
    // for it at all -- a schema mismatch, since source 0 has live docs
    // indexing that field.
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &[],
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let result = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_missing_points",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    );
    assert!(matches!(
        result,
        Err(Error::PointsFieldMissingInSource { .. })
    ));
}

#[test]
fn a_source_that_never_saw_the_points_field_simply_contributes_no_points() {
    // `PointsWriter.merge`: `if (readerFieldInfo == null) continue;`.
    // Adding a points field to an index that already has segments is
    // normal -- the older segments have no `FieldInfo` for it at all and
    // contribute nothing, rather than failing the merge.
    let seg0_id = [1u8; ID_LENGTH];
    let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

    let fields0 = vec![points_field("loc", 0, 1, 4)];
    // Source 1 has a different field entirely; "loc" is unknown to it.
    let fields1 = vec![field("id", 0)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &[doc_with(0, "a")]);
    let stored1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields1,
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let source0 = MergeSource {
        points: &src_points0,
        vectors: None,
        ..MergeSource::stored_only(&stored0.fields, &reader0, None, Some(version()))
    };
    let source1 = MergeSource::stored_only(&stored1.fields, &reader1, None, Some(version()));

    merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_points_absent_source",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let kdm = std::fs::read(tmp.join("_merged_points_absent_source.kdm")).unwrap();
    let kdi = std::fs::read(tmp.join("_merged_points_absent_source.kdi")).unwrap();
    let kdd = std::fs::read(tmp.join("_merged_points_absent_source.kdd")).unwrap();
    let merged = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();
    let all = merged.decode_all_points(0).unwrap();
    assert_eq!(
        all.iter()
            .map(|p| (p.doc_id, p.packed_value.clone()))
            .collect::<Vec<_>>(),
        vec![(0, packed4(10))],
        "only source 0's point survives, at merged doc 0"
    );
}

#[test]
fn points_shape_disagreement_across_sources_is_an_error() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let (kdm0, kdi0, kdd0) = write_one_field_points(0, 1, 4, vec![(0, packed4(10))], &seg0_id);
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();
    // Source 1's own points data uses 8 bytes per dimension, disagreeing
    // with the merged field's declared shape (4 bytes per dimension,
    // taken from source 0's FieldInfo since it's first-seen).
    let (kdm1, kdi1, kdd1) = write_one_field_points(
        0,
        1,
        8,
        vec![(0, vec![0u8, 0, 0, 0, 0, 0, 0, 42])],
        &seg1_id,
    );
    let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

    let fields = vec![points_field("loc", 0, 1, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let src_points1 = [SourcePoints {
        field_number: 0,
        reader: &points_reader1,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points1,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let result = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_points_shape_mismatch",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    );
    assert!(matches!(result, Err(Error::PointsShapeDisagreement { .. })));
}

/// Two sources, both `num_dims=3`/`num_index_dims=2` (a
/// `LatLonShape`-style bounding box shape with one trailing data-only
/// dimension), merge cleanly: every point's full 3-dimension packed
/// value -- including the non-indexed third dimension -- survives the
/// merge unchanged, and doc ids are renumbered into the merged id space
/// exactly like the `num_index_dims == num_dims` case already covered by
/// `two_sources_no_deletions_merge_points_correctly`.
#[test]
fn two_sources_consistent_num_index_dims_less_than_num_dims_merge_correctly() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let packed12 = |a: u32, b: u32, c: u32| -> Vec<u8> {
        [a.to_be_bytes(), b.to_be_bytes(), c.to_be_bytes()].concat()
    };

    let (kdm0, kdi0, kdd0) = write_one_field_points_with_index_dims(
        0,
        3,
        2,
        4,
        vec![(0, packed12(1, 2, 100)), (1, packed12(3, 4, 200))],
        &seg0_id,
    );
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

    let (kdm1, kdi1, kdd1) = write_one_field_points_with_index_dims(
        0,
        3,
        2,
        4,
        vec![(0, packed12(5, 6, 300))],
        &seg1_id,
    );
    let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

    let fields = vec![points_field_with_index_dims("loc", 0, 3, 2, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(
        &dir,
        &tmp,
        "_0",
        seg0_id,
        &fields,
        &[doc_with(0, "a"), doc_with(0, "b")],
    );
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields, &[doc_with(0, "c")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let src_points1 = [SourcePoints {
        field_number: 0,
        reader: &points_reader1,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points1,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let sci = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_points_index_dims_lt_dims",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    assert_eq!(sci.segment_name, "_merged_points_index_dims_lt_dims");

    let base = tmp.join("_merged_points_index_dims_lt_dims");
    let kdm = std::fs::read(base.with_extension("kdm")).unwrap();
    let kdi = std::fs::read(base.with_extension("kdi")).unwrap();
    let kdd = std::fs::read(base.with_extension("kdd")).unwrap();
    let merged_reader = points::open(&kdm, &kdi, &kdd, &[9u8; ID_LENGTH], "").unwrap();

    let merged_field = merged_reader.field(0).unwrap();
    assert_eq!(merged_field.num_dims, 3);
    assert_eq!(merged_field.num_index_dims, 2);

    let mut merged_points = merged_reader.decode_all_points(0).unwrap();
    merged_points.sort_by_key(|p| p.doc_id);
    assert_eq!(
        merged_points
            .iter()
            .map(|p| (p.doc_id, p.packed_value.clone()))
            .collect::<Vec<_>>(),
        vec![
            (0, packed12(1, 2, 100)),
            (1, packed12(3, 4, 200)),
            // Source 1's only doc is renumbered to merged doc 2, after
            // source 0's 2 docs.
            (2, packed12(5, 6, 300)),
        ]
    );
}

/// Two sources whose points data agree on `num_dims`/`bytes_per_dim` but
/// disagree on `num_index_dims` (source 0 is `num_index_dims=1`, source 1
/// is `num_index_dims=2`, both `num_dims=2`) -- field-number
/// reconciliation records source 0's `FieldInfo` (`num_index_dims=1`) as
/// the merged field's declared shape, so source 1's own
/// `num_index_dims=2` disagrees and must be rejected rather than having
/// its second dimension silently reinterpreted as a data-only payload
/// dimension.
#[test]
fn points_index_dims_disagreement_across_sources_is_rejected() {
    let seg0_id = [1u8; ID_LENGTH];
    let seg1_id = [2u8; ID_LENGTH];

    let packed8 = |a: u32, b: u32| -> Vec<u8> { [a.to_be_bytes(), b.to_be_bytes()].concat() };

    let (kdm0, kdi0, kdd0) =
        write_one_field_points_with_index_dims(0, 2, 1, 4, vec![(0, packed8(10, 11))], &seg0_id);
    let points_reader0 = points::open(&kdm0, &kdi0, &kdd0, &seg0_id, "").unwrap();

    let (kdm1, kdi1, kdd1) =
        write_one_field_points_with_index_dims(0, 2, 2, 4, vec![(0, packed8(20, 21))], &seg1_id);
    let points_reader1 = points::open(&kdm1, &kdi1, &kdd1, &seg1_id, "").unwrap();

    let fields0 = vec![points_field_with_index_dims("loc", 0, 2, 1, 4)];
    let fields1 = vec![points_field_with_index_dims("loc", 0, 2, 2, 4)];
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let stored0 = flush(&dir, &tmp, "_0", seg0_id, &fields0, &[doc_with(0, "a")]);
    let stored1 = flush(&dir, &tmp, "_1", seg1_id, &fields1, &[doc_with(0, "b")]);
    let reader0 = open_reader(&stored0);
    let reader1 = open_reader(&stored1);

    let src_points0 = [SourcePoints {
        field_number: 0,
        reader: &points_reader0,
    }];
    let src_points1 = [SourcePoints {
        field_number: 0,
        reader: &points_reader1,
    }];
    let source0 = MergeSource {
        field_infos: &stored0.fields,
        reader: &reader0,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points0,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };
    let source1 = MergeSource {
        field_infos: &stored1.fields,
        reader: &reader1,
        live_docs: None,
        numeric_doc_values: &[],
        binary_doc_values: &[],
        sorted_doc_values: &[],
        sorted_numeric_doc_values: &[],
        sorted_set_doc_values: &[],
        norms: &[],
        term_vectors: None,
        postings: &[],
        points: &src_points1,
        vectors: None,
        min_version: None,
        has_blocks: false,
        pruned_field: None,
        numeric_floor: None,
    };

    let result = merge_stored_only_segments(
        &dir,
        &[source0, source1],
        "_merged_points_index_dims_mismatch",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    );
    assert!(matches!(
        result,
        Err(Error::PointsIndexDimsDisagreement { .. })
    ));
}

// --- the three stored-fields merge strategies ---

fn strategies_for(sources: &[MergeSource]) -> Vec<StoredFieldsMergeStrategy> {
    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (_, maps) = reconcile_field_numbers(&sources_fields).unwrap();
    let matching = matching_readers(sources, &maps);
    let writer = stored_fields::StoredFieldsWriter::new(
        stored_fields::Mode::BestSpeed,
        &[0u8; ID_LENGTH],
        "",
    );
    sources
        .iter()
        .zip(&matching)
        .map(|(s, &m)| stored_fields_merge_strategy(&writer, s, m, false))
        .collect()
}

fn read_merged(
    tmp: &std::path::Path,
    name: &str,
    id: [u8; ID_LENGTH],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let base = tmp;
    let _ = id;
    (
        std::fs::read(base.join(format!("{name}.fdt"))).unwrap(),
        std::fs::read(base.join(format!("{name}.fdx"))).unwrap(),
        std::fs::read(base.join(format!("{name}.fdm"))).unwrap(),
    )
}

#[test]
fn a_matching_deletion_free_source_is_bulk_copied_verbatim() {
    // The BULK path, over several real chunk boundaries: the merged
    // segment must hold exactly the source documents, and -- because
    // nothing was recompressed -- exactly the sources' chunk counts.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    const N: usize = 2_500; // > 2 * the 1024-doc chunk cap
    let docs_a: Vec<Document> = (0..N).map(|i| doc_with(0, &format!("a{i}"))).collect();
    let docs_b: Vec<Document> = (0..N).map(|i| doc_with(0, &format!("b{i}"))).collect();
    let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs_a);
    let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields, &docs_b);
    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];
    assert_eq!(
        strategies_for(&sources),
        vec![
            StoredFieldsMergeStrategy::Bulk,
            StoredFieldsMergeStrategy::Bulk
        ]
    );

    merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (fdt, fdx, fdm) = read_merged(&tmp, "_merged", [9u8; ID_LENGTH]);
    let merged = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(merged.max_doc(), 2 * N as i32);
    assert_eq!(
        merged.num_chunks(),
        reader0.num_chunks() + reader1.num_chunks(),
        "every chunk should have been copied, not rebuilt"
    );
    for i in 0..N {
        assert_eq!(
            merged.document(i as i32).unwrap().fields[0].value,
            FieldValue::String(format!("a{i}"))
        );
        assert_eq!(
            merged.document((N + i) as i32).unwrap().fields[0].value,
            FieldValue::String(format!("b{i}"))
        );
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn a_source_with_deletions_takes_the_doc_path_and_drops_exactly_the_deleted_docs() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    const N: usize = 1_500;
    let docs: Vec<Document> = (0..N).map(|i| doc_with(0, &format!("d{i}"))).collect();
    let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
    let reader0 = open_reader(&seg0);

    // Every third document deleted, spanning both of the source's chunks.
    let mut live = FixedBitSet::new(N);
    for i in 0..N {
        if i % 3 != 0 {
            live.set(i);
        }
    }
    let sources = vec![MergeSource::stored_only(
        &seg0.fields,
        &reader0,
        Some(&live),
        Some(version()),
    )];
    assert_eq!(
        strategies_for(&sources),
        vec![StoredFieldsMergeStrategy::Doc]
    );

    merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (fdt, fdx, fdm) = read_merged(&tmp, "_merged", [9u8; ID_LENGTH]);
    let merged = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
    let survivors: Vec<usize> = (0..N).filter(|i| i % 3 != 0).collect();
    assert_eq!(merged.max_doc(), survivors.len() as i32);
    for (merged_id, &source_id) in survivors.iter().enumerate() {
        assert_eq!(
            merged.document(merged_id as i32).unwrap().fields[0].value,
            FieldValue::String(format!("d{source_id}")),
            "merged doc {merged_id}"
        );
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn a_source_whose_field_numbers_change_takes_the_visitor_path_and_is_renumbered() {
    // Source 1 numbers "body" 0 and "id" 1; the merge numbers them the
    // other way round (source 0 was seen first). Its serialized bytes
    // therefore cannot be copied -- they encode the source's own numbers.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields0 = vec![field("id", 0), field("body", 1)];
    let fields1 = vec![field("body", 0), field("id", 1)];
    let docs0 = vec![Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("id-0".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String("body-0".to_string()),
            },
        ],
    }];
    let docs1 = vec![Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String("body-1".to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::String("id-1".to_string()),
            },
        ],
    }];
    let seg0 = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields0, &docs0);
    let seg1 = flush(&dir, &tmp, "_1", [2u8; ID_LENGTH], &fields1, &docs1);
    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];
    assert_eq!(
        strategies_for(&sources),
        vec![
            StoredFieldsMergeStrategy::Bulk,
            StoredFieldsMergeStrategy::Visitor
        ]
    );

    merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();

    let (fdt, fdx, fdm) = read_merged(&tmp, "_merged", [9u8; ID_LENGTH]);
    let merged = stored_fields::open(&fdt, &fdx, &fdm, &[9u8; ID_LENGTH], "").unwrap();
    assert_eq!(merged.max_doc(), 2);
    // Merged numbering: id = 0, body = 1, for both documents.
    for (doc_id, want) in [(0, "0"), (1, "1")] {
        let doc = merged.document(doc_id).unwrap();
        let id_field = doc.fields.iter().find(|f| f.field_number == 0).unwrap();
        let body_field = doc.fields.iter().find(|f| f.field_number == 1).unwrap();
        assert_eq!(id_field.value, FieldValue::String(format!("id-{want}")));
        assert_eq!(body_field.value, FieldValue::String(format!("body-{want}")));
    }
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn matching_readers_is_the_identity_field_number_test() {
    // Source 0's numbering survives; source 1's does not, because "body"
    // was already claimed number 1 by source 0; source 2's does, because
    // it agrees with source 0.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields0 = vec![field("id", 0), field("body", 1)];
    let fields1 = vec![field("body", 0)];
    let fields2 = vec![field("id", 0), field("body", 1)];
    let seg0 = flush(
        &dir,
        &tmp,
        "_0",
        [1u8; ID_LENGTH],
        &fields0,
        &[doc_with(0, "a")],
    );
    let seg1 = flush(
        &dir,
        &tmp,
        "_1",
        [2u8; ID_LENGTH],
        &fields1,
        &[doc_with(0, "b")],
    );
    let seg2 = flush(
        &dir,
        &tmp,
        "_2",
        [3u8; ID_LENGTH],
        &fields2,
        &[doc_with(0, "c")],
    );
    let (r0, r1, r2) = (open_reader(&seg0), open_reader(&seg1), open_reader(&seg2));
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &r0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &r1, None, Some(version())),
        MergeSource::stored_only(&seg2.fields, &r2, None, Some(version())),
    ];
    let sources_fields: Vec<&[FieldInfo]> = sources.iter().map(|s| s.field_infos).collect();
    let (_, maps) = reconcile_field_numbers(&sources_fields).unwrap();
    assert_eq!(maps[0].get(&0), Some(&0));
    assert_eq!(
        maps[1].get(&0),
        Some(&1),
        "source 1's \"body\" is renumbered"
    );
    assert_eq!(matching_readers(&sources, &maps), vec![true, false, true]);
    // ...and that is what decides BULK vs VISITOR.
    assert_eq!(
        strategies_for(&sources),
        vec![
            StoredFieldsMergeStrategy::Bulk,
            StoredFieldsMergeStrategy::Visitor,
            StoredFieldsMergeStrategy::Bulk
        ]
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn a_source_whose_fdt_fails_its_own_checksum_is_never_bulk_copied() {
    // `checkIntegrity` before the strategy choice: the bulk path copies a
    // source's compressed bytes verbatim and writes a fresh, valid footer
    // over them, so without this the merge would launder a corrupt source
    // into a segment that passes every checksum from then on.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    let docs: Vec<Document> = (0..50).map(|i| doc_with(0, &format!("v{i}"))).collect();
    let mut seg = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
    // Flip one byte of the compressed payload, leaving every length,
    // pointer and footer field intact -- exactly the shape of corruption
    // `retrieve_checksum` cannot see.
    let mid = seg.fdt.len() / 2;
    seg.fdt[mid] ^= 0xFF;

    let reader = open_reader(&seg);
    let sources = vec![MergeSource::stored_only(
        &seg.fields,
        &reader,
        None,
        Some(version()),
    )];
    let result = merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    );
    assert!(
        result.is_err(),
        "a source whose .fdt fails its checksum must not be merged"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn a_live_docs_bitset_shorter_than_max_doc_is_reported_not_indexed_past() {
    // `docs/arithmetic-gate.md`'s crate rule: *never index a
    // `FixedBitSet` with an index bounded against anything other than
    // that bitset's own `len()`.* The doc id here is bounded by the
    // source's stored-fields `maxDoc` (off the `.fdm`) and the bitset
    // comes off the `.liv` -- two independent files.
    //
    // Both failure modes of the `FixedBitSet` rule are covered:
    //  * an **empty** bitset, where every index is out of range, so
    //    `FixedBitSet::get`'s own bound (checked in release as well as
    //    debug) panics -- this case aborts the test without
    //    the fix;
    //  * a bitset a few bits **short**, where the index still lands
    //    inside `words` and reads a ghost bit: five documents silently
    //    dropped from the merged segment, with nothing reported.
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![field("id", 0)];
    let docs: Vec<Document> = (0..50).map(|i| doc_with(0, &format!("v{i}"))).collect();
    let seg = flush(&dir, &tmp, "_0", [1u8; ID_LENGTH], &fields, &docs);
    let reader = open_reader(&seg);

    for short_len in [0usize, 45] {
        let mut bits = FixedBitSet::new(short_len);
        for d in 0..short_len {
            bits.set(d);
        }
        let sources = vec![MergeSource::stored_only(
            &seg.fields,
            &reader,
            Some(&bits),
            Some(version()),
        )];
        let result = merge_stored_only_segments(
            &dir,
            &sources,
            "_merged",
            [9u8; ID_LENGTH],
            "Lucene104",
            version(),
        );
        assert!(
            matches!(
                result,
                Err(Error::LiveDocsLengthMismatch {
                    source_index: 0,
                    max_doc: 50,
                    live_docs_len,
                }) if live_docs_len == short_len
            ),
            "a .liv covering {short_len} of 50 documents must be reported, got {result:?}"
        );
    }

    // The control: a bitset that *does* cover `maxDoc` merges, and the
    // deletion it records is honoured -- so the check above rejects a
    // disagreement rather than every live-docs merge.
    let mut exact = FixedBitSet::new(50);
    for d in 0..50 {
        exact.set(d);
    }
    exact.clear(7);
    let sources = vec![MergeSource::stored_only(
        &seg.fields,
        &reader,
        Some(&exact),
        Some(version()),
    )];
    let merged = merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .expect("a .liv the right length merges");
    let si_bytes = dir.open("_merged.si").unwrap().to_vec();
    let si = segment_info::parse(&si_bytes, &merged.segment_id).unwrap();
    assert_eq!(si.doc_count, 49, "the one deleted document was dropped");
    std::fs::remove_dir_all(&tmp).ok();
}

// --- SegmentMerger's per-source metadata: minVersion and hasBlocks ---

fn v(major: i32, minor: i32, bugfix: i32) -> LuceneVersion {
    LuceneVersion {
        major,
        minor,
        bugfix,
    }
}

/// `SegmentMerger`'s constructor, in isolation: the seed is
/// `Version.LATEST`, one `null` anywhere wins, and otherwise the smallest
/// reader version wins regardless of the order the readers arrive in.
#[test]
fn merged_min_version_is_the_smallest_source_version_or_none() {
    let latest = v(10, 5, 0);

    // No sources at all: Java keeps the `Version.LATEST` seed. Driven
    // through `merged_min_version` itself, which is the only shape that
    // takes `MergeSource`s.
    assert_eq!(merged_min_version(&[], latest), Some(latest));

    // Everything else drives `fold_min_version`, the function
    // `merged_min_version` *is* -- not a restatement of it. (An earlier
    // version of this test carried its own copy of the loop, which meant
    // inverting the comparison or dropping the `None` short-circuit in the
    // shipped code left it green.)
    let fold = |mins: &[Option<LuceneVersion>]| fold_min_version(latest, mins.to_vec());

    // Every ordering of the same three versions gives the same answer.
    let a = v(10, 0, 0);
    let b = v(10, 2, 1);
    let c = v(10, 5, 0);
    for order in [[a, b, c], [c, b, a], [b, c, a], [b, a, c]] {
        assert_eq!(
            fold(&order.map(Some)),
            Some(a),
            "order {order:?} did not give the oldest version"
        );
    }
    // Bugfix and minor both participate, not just major.
    assert_eq!(
        fold(&[Some(v(10, 2, 3)), Some(v(10, 2, 1))]),
        Some(v(10, 2, 1))
    );
    assert_eq!(
        fold(&[Some(v(10, 3, 0)), Some(v(10, 2, 9))]),
        Some(v(10, 2, 9))
    );
    // A source *newer* than this writer cannot raise the answer above the
    // seed -- the fold only ever lowers it. This is the case that fails if
    // `onOrAfter`'s direction is inverted.
    assert_eq!(fold(&[Some(v(11, 0, 0))]), Some(latest));
    // One `null` anywhere makes the whole thing null, wherever it sits.
    assert_eq!(fold(&[Some(a), None, Some(b)]), None);
    assert_eq!(fold(&[None, Some(a)]), None);
    assert_eq!(fold(&[None]), None);
}

/// End to end through `merge_segments`: the merged `.si` records the
/// **oldest** source `minVersion`, not the merging writer's own version,
/// and `hasBlocks` is the disjunction of the sources'. A `min_version`
/// hardcoded to the writer's version would erase, on a merge that took in
/// a segment written by an older Lucene, the only record that older bytes
/// were still present.
#[test]
fn a_merged_si_takes_the_oldest_source_min_version_and_any_sources_blocks() {
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
        &[doc_with(0, "b")],
    );
    let reader0 = open_reader(&seg0);
    let reader1 = open_reader(&seg1);

    let sources = vec![
        MergeSource {
            min_version: Some(v(10, 1, 0)),
            has_blocks: false,
            pruned_field: None,
            numeric_floor: None,
            ..MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version()))
        },
        MergeSource {
            // Older than source 0 *and* older than the merging writer's
            // own `version()`; blocks on this source only.
            min_version: Some(v(9, 11, 2)),
            has_blocks: true,
            pruned_field: None,
            numeric_floor: None,
            ..MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version()))
        },
    ];
    let merged = merge_stored_only_segments(
        &dir,
        &sources,
        "_merged",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let si_bytes = dir.open("_merged.si").unwrap().to_vec();
    let si = segment_info::parse(&si_bytes, &merged.segment_id).unwrap();
    assert_eq!(si.min_version, Some(v(9, 11, 2)));
    assert_eq!(
        si.version,
        version(),
        "the writer's own version is unchanged"
    );
    assert!(si.has_blocks, "a block-carrying source must set the flag");

    // The control: with no blocks anywhere and both sources at the
    // writer's own version, the merged `.si` says exactly that -- so the
    // assertions above are the fold doing work, not a constant.
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, Some(version())),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];
    let merged = merge_stored_only_segments(
        &dir,
        &sources,
        "_merged2",
        [10u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let si_bytes = dir.open("_merged2.si").unwrap().to_vec();
    let si = segment_info::parse(&si_bytes, &merged.segment_id).unwrap();
    assert_eq!(si.min_version, Some(version()));
    assert!(!si.has_blocks);

    // And Java's `null` case: **one** source that records no `minVersion`
    // at all (a pre-Lucene-7 segment) makes the merged segment's `null`
    // too, however new the others are. `check_index`'s
    // `commit.segment_records_min_version` is what then reports it, the
    // way `SegmentInfos.applyMergeChanges` refuses it in Java.
    let sources = vec![
        MergeSource::stored_only(&seg0.fields, &reader0, None, None),
        MergeSource::stored_only(&seg1.fields, &reader1, None, Some(version())),
    ];
    let merged = merge_stored_only_segments(
        &dir,
        &sources,
        "_merged3",
        [11u8; ID_LENGTH],
        "Lucene104",
        version(),
    )
    .unwrap();
    let si_bytes = dir.open("_merged3.si").unwrap().to_vec();
    let si = segment_info::parse(&si_bytes, &merged.segment_id).unwrap();
    assert_eq!(si.min_version, None);
    std::fs::remove_dir_all(&tmp).ok();
}

// --- the merged `.fnm` must describe the files the merge wrote ---

#[test]
fn a_merged_field_with_postings_gets_the_per_field_format_attributes() {
    let mut fields = vec![field("body", 0)];
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    fields[0]
        .attributes
        .push(("keep".to_string(), "me".to_string()));
    describe_written_files(&mut fields, &[(0, "Lucene104", 0)], &[], false, &[], &[]);
    assert!(fields[0]
        .attributes
        .contains(&("keep".to_string(), "me".to_string())));
    assert!(fields[0].attributes.contains(&(
        "PerFieldPostingsFormat.format".to_string(),
        POSTINGS_FORMAT_NAME.to_string()
    )));
    assert!(fields[0].attributes.contains(&(
        "PerFieldPostingsFormat.suffix".to_string(),
        PER_FIELD_SUFFIX.to_string()
    )));
}

#[test]
fn a_stale_per_field_attribute_inherited_from_a_source_is_stripped() {
    // The source's own `.fnm` names a postings format; this merge wrote no
    // postings, so a reader following the attribute would look for a file
    // that does not exist.
    let mut fields = vec![field("body", 0)];
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    fields[0].attributes.push((
        "PerFieldPostingsFormat.format".to_string(),
        POSTINGS_FORMAT_NAME.to_string(),
    ));
    fields[0].attributes.push((
        "PerFieldDocValuesFormat.suffix".to_string(),
        "0".to_string(),
    ));
    describe_written_files(&mut fields, &[], &[], false, &[], &[]);
    assert!(fields[0].attributes.is_empty());
}

/// `describe_written_files` must **not** touch `omit_norms`: `merge_norms`
/// writes a column for
/// exactly the fields whose merged `.fnm` claims one (Java's
/// `mergeFieldInfos`-driven loop), so rewriting the schema here would
/// undo the claim the column was written for.
#[test]
fn a_merged_indexed_field_keeps_the_norms_claim_its_column_backs() {
    let mut fields = vec![field("body", 0), field("title", 1)];
    for f in fields.iter_mut() {
        f.index_options = IndexOptions::DocsAndFreqs;
        f.omit_norms = false;
    }
    fields[1].omit_norms = true;
    describe_written_files(
        &mut fields,
        &[(0, "Lucene104", 0), (1, "Lucene104", 0)],
        &[],
        false,
        &[],
        &[],
    );
    assert!(!fields[0].omit_norms, "an indexed field keeps its norms");
    assert!(fields[1].omit_norms, "an opted-out field keeps its opt-out");
}

#[test]
fn a_field_claiming_term_vectors_loses_the_claim_when_none_were_written() {
    let mut fields = vec![field("body", 0)];
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    fields[0].store_term_vectors = true;
    describe_written_files(&mut fields, &[], &[], false, &[], &[]);
    assert!(!fields[0].store_term_vectors);

    let mut fields = vec![field("body", 0)];
    fields[0].index_options = IndexOptions::DocsAndFreqs;
    fields[0].store_term_vectors = true;
    describe_written_files(&mut fields, &[], &[], true, &[], &[]);
    assert!(fields[0].store_term_vectors);
}

#[test]
fn a_merged_doc_values_field_gets_its_per_field_format_attributes() {
    let mut fields = vec![field("score", 0)];
    fields[0].doc_values_type = DocValuesType::Numeric;
    describe_written_files(
        &mut fields,
        &[],
        &[(0, DOC_VALUES_FORMAT_NAME, 0)],
        false,
        &[],
        &[],
    );
    assert!(fields[0].attributes.contains(&(
        "PerFieldDocValuesFormat.format".to_string(),
        DOC_VALUES_FORMAT_NAME.to_string()
    )));
    assert!(fields[0].attributes.contains(&(
        "PerFieldDocValuesFormat.suffix".to_string(),
        PER_FIELD_SUFFIX.to_string()
    )));
    // A non-indexed field never gains an `omit_norms` rewrite.
    assert!(!fields[0].omit_norms);
}

/// A field big enough to split (`PARALLEL_MERGE_MIN_TERMS`), merged as
/// `term_ranges`' ranges on a four-thread pool, gives exactly the
/// single-range merge: same terms in the same order, same postings and
/// positions, same documents seen -- across overlapping dictionaries, an
/// empty source and a deleted document.
#[test]
fn parallel_term_ranges_merge_like_one_range() {
    let suffix = per_field_codec_suffix(POSTINGS_FORMAT_NAME);
    let options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
    let mut info = postings_field("body", 0);
    info.index_options = options;
    let infos = field_infos::FieldInfos { fields: vec![info] };
    let outputs: Vec<_> = (0..3u8)
        .map(|s| {
            let terms: Vec<TermPostings> = (0..6000usize)
                .map(|i| {
                    let t = i * 3 / 2 + usize::from(s) * 1777;
                    let docs: Vec<(i32, i32)> = (0..1 + t % 4)
                        .map(|d| (((t + d * 7) % 40) as i32, 1 + (d % 2) as i32))
                        .collect::<std::collections::BTreeMap<_, _>>()
                        .into_iter()
                        .collect();
                    let positions = docs
                        .iter()
                        .map(|&(doc, f)| (0..f).map(|p| doc + p * 3).collect())
                        .collect();
                    let offsets = docs
                        .iter()
                        .map(|&(_, f)| (0..f).map(|p| (p * 4, p * 4 + 2)).collect())
                        .collect();
                    TermPostings {
                        term: format!("t{t:06}").into_bytes(),
                        docs,
                        positions,
                        offsets,
                        ..Default::default()
                    }
                })
                .collect();
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: options,
                doc_count: 40,
                has_payloads: false,
                terms: &terms,
            };
            let id = [s + 1; ID_LENGTH];
            (
                id,
                postings_writer::write_single_field(&input, &id, &suffix).unwrap(),
            )
        })
        .collect();
    let fields: Vec<_> = outputs
        .iter()
        .map(|(id, o)| blocktree::open(&o.tim, &o.tip, &o.tmd, &infos, id, &suffix, 40).unwrap())
        .collect();
    let doc_ins: Vec<_> = outputs
        .iter()
        .map(|(id, o)| DocInput::open(&o.doc, id, &suffix).unwrap())
        .collect();
    let pos_ins: Vec<_> = outputs
        .iter()
        .map(|(id, o)| lucene_codecs::postings::PosInput::open(&o.pos, id, &suffix).unwrap())
        .collect();
    let pay_ins: Vec<_> = outputs
        .iter()
        .map(|(id, o)| lucene_codecs::postings::PayInput::open(&o.pay, id, &suffix).unwrap())
        .collect();
    let sources: Vec<SourcePostings<'_>> = (0..3)
        .map(|s| SourcePostings {
            field_number: 0,
            field_terms: fields[s].field("body").unwrap(),
            doc_in: Some(&doc_ins[s]),
            pos_in: Some(&pos_ins[s]),
            pay_in: Some(&pay_ins[s]),
            live: None,
        })
        .collect();
    // Source 1 sits out (a source without the field); source 2's doc 5
    // is deleted.
    let per_source_field = vec![
        Some(&sources[0]),
        None,
        Some(&sources[1]),
        Some(&sources[2]),
    ];
    let doc_id_maps: Vec<Vec<i32>> = (0..4)
        .map(|s| {
            (0..40)
                .map(|d| if s == 3 && d == 5 { -1 } else { s * 40 + d })
                .collect()
        })
        .collect();
    let shape = FieldShape {
        has_positions: true,
        has_offsets: true,
        has_payloads: false,
        merged_doc_count: 160,
    };
    let (serial, serial_seen) =
        merge_term_range(&per_source_field, &doc_id_maps, shape, None, None, 1).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let ranges = pool.install(|| term_ranges(&per_source_field)).unwrap();
    assert!(ranges.len() > 1, "{} ranges", ranges.len());
    let mut runs = Vec::new();
    let mut parallel_seen = FixedBitSet::new(160);
    for i in 0..ranges.len() {
        let lower = (i > 0).then(|| ranges[i].as_slice());
        let upper = ranges.get(i + 1).map(Vec::as_slice);
        let (run, seen) =
            merge_term_range(&per_source_field, &doc_id_maps, shape, lower, upper, 3).unwrap();
        runs.push(run);
        parallel_seen.or(&seen);
    }
    let parallel: Vec<_> = runs.iter().flat_map(MergedRun::views).collect();
    let serial: Vec<_> = serial.views().collect();
    assert_eq!(parallel.len(), serial.len());
    let shape_of = |t: &MergedTermView<'_>| {
        let docs = t.docs().len();
        (
            t.term().to_vec(),
            t.docs().to_vec(),
            (0..docs)
                .map(|d| t.positions(d).to_vec())
                .collect::<Vec<_>>(),
            (0..docs).map(|d| t.offsets(d).to_vec()).collect::<Vec<_>>(),
        )
    };
    for (p, s) in parallel.iter().zip(&serial) {
        assert_eq!(shape_of(p), shape_of(s));
    }
    assert_eq!(parallel_seen.cardinality(), serial_seen.cardinality());
    // A lower bound past every term: nothing.
    let (none, _) =
        merge_term_range(&per_source_field, &doc_id_maps, shape, Some(b"u"), None, 2).unwrap();
    assert_eq!(none.views().count(), 0);
}
