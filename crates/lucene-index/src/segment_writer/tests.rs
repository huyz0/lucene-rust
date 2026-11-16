#![cfg(test)]
use super::*;
use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_store::directory::FsDirectory;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 0,
        bugfix: 0,
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

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

#[test]
fn flushes_a_segment_with_the_expected_files_and_doc_count() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let docs = vec![doc("1"), doc("2")];

    let sci = flush_stored_only_segment(
        &dir,
        "_0",
        [7u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &docs,
        false,
    )
    .unwrap();

    assert_eq!(sci.segment_name, "_0");
    assert_eq!(sci.segment_id, [7u8; ID_LENGTH]);
    assert_eq!(sci.codec_name, "Lucene104");
    assert_eq!(sci.del_count, 0);
    for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
        assert!(
            std::path::Path::new(&tmp)
                .join(format!("_0.{ext}"))
                .exists(),
            "missing _0.{ext}"
        );
    }

    // The .si file must claim the same doc count we flushed -- cross-check
    // against segment_info::parse the same way the real fixture does.
    let si_bytes = std::fs::read(tmp.join("_0.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert_eq!(si.doc_count, docs.len() as i32);
}

#[test]
fn two_flushes_produce_two_independent_segments() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];

    let sci0 = flush_stored_only_segment(
        &dir,
        "_0",
        [1u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &[doc("1")],
        false,
    )
    .unwrap();
    let sci1 = flush_stored_only_segment(
        &dir,
        "_1",
        [2u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &[doc("2"), doc("3")],
        false,
    )
    .unwrap();

    assert_ne!(sci0.segment_name, sci1.segment_name);
    assert_ne!(sci0.segment_id, sci1.segment_id);
    for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
        assert!(std::path::Path::new(&tmp)
            .join(format!("_0.{ext}"))
            .exists());
        assert!(std::path::Path::new(&tmp)
            .join(format!("_1.{ext}"))
            .exists());
    }
}

#[test]
fn flush_surfaces_directory_io_error_rather_than_panicking() {
    // A root under a regular file cannot be created, so the very first
    // create_output (the .fdt file) fails -- confirms Error::Store's
    // #[from] wrapping propagates a real Directory I/O failure as an Err
    // rather than panicking or silently losing the error, the one path
    // this module's own error type exists to cover. (A merely missing
    // root no longer fails: the first output creates it, as Java's
    // FSDirectory constructor does.)
    let tmp = tempdir();
    let blocker = std::path::Path::new(&tmp).join("blocker");
    std::fs::write(&blocker, b"not a directory").unwrap();
    let dir = FsDirectory::open(blocker.join("index"));
    let fields = vec![stored_only_field("id", 0)];
    let docs = vec![doc("1")];

    let result = flush_stored_only_segment(
        &dir,
        "_0",
        [9u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &docs,
        false,
    );

    assert!(matches!(result, Err(Error::Store(_))));
}

/// `use_compound_file: true` must produce a `.cfs`/`.cfe` pair instead of
/// loose `.fdt`/`.fdx`/`.fdm`/`.fnm`, and the `.si` must record
/// `is_compound_file: true` -- the two facts a reader relies on to know
/// which layout to open.
#[test]
fn compound_flush_writes_cfs_cfe_pair_and_marks_si_compound() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let docs = vec![doc("1"), doc("2"), doc("3")];

    let sci = flush_stored_only_segment(
        &dir,
        "_0",
        [3u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &docs,
        true,
    )
    .unwrap();

    for ext in ["cfs", "cfe", "si"] {
        assert!(
            std::path::Path::new(&tmp)
                .join(format!("_0.{ext}"))
                .exists(),
            "missing _0.{ext}"
        );
    }
    for ext in ["fdt", "fdx", "fdm", "fnm"] {
        assert!(
            !std::path::Path::new(&tmp)
                .join(format!("_0.{ext}"))
                .exists(),
            "loose _0.{ext} should not exist in compound mode"
        );
    }

    let si_bytes = std::fs::read(tmp.join("_0.si")).unwrap();
    let si = segment_info::parse(&si_bytes, &sci.segment_id).unwrap();
    assert!(si.is_compound_file);
    // The `.si` lists itself, as `Lucene99SegmentInfoFormat.write` does.
    assert_eq!(
        si.files,
        vec![
            "_0.cfs".to_string(),
            "_0.cfe".to_string(),
            "_0.si".to_string()
        ]
    );
}

/// The meaningful end-to-end check: flush with `use_compound_file: true`,
/// then recover the original `.fdt`/`.fdx`/`.fdm`/`.fnm` sub-files
/// byte-for-byte via the already-verified `compound_format` reader
/// (`compound_format::parse_entries` + `open_input`), and confirm
/// `stored_fields::open` can read documents back out *through* those
/// recovered slices -- not by re-deriving from the original in-memory
/// buffers, so a byte-offset bug in the new wiring would show up here.
#[test]
fn compound_flush_round_trips_through_compound_reader_and_stored_fields() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let segment_id = [5u8; ID_LENGTH];
    let fields = vec![stored_only_field("id", 0)];
    let docs = vec![doc("alpha"), doc("beta"), doc("gamma")];

    flush_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        true,
    )
    .unwrap();

    let cfs = std::fs::read(tmp.join("_0.cfs")).unwrap();
    let cfe = std::fs::read(tmp.join("_0.cfe")).unwrap();

    let entries = compound_format::parse_entries(&cfe, &segment_id).unwrap();
    compound_format::check_data_header_footer(&cfs, &segment_id, &entries).unwrap();

    let fdt = compound_format::open_input(&cfs, &entries, ".fdt")
        .unwrap()
        .as_slice();
    let fdx = compound_format::open_input(&cfs, &entries, ".fdx")
        .unwrap()
        .as_slice();
    let fdm = compound_format::open_input(&cfs, &entries, ".fdm")
        .unwrap()
        .as_slice();
    let fnm = compound_format::open_input(&cfs, &entries, ".fnm")
        .unwrap()
        .as_slice();

    // Field infos recovered through the compound reader must still parse
    // and describe the one stored-only field we flushed.
    let parsed_fields = field_infos::parse(fnm, &segment_id, "").unwrap();
    assert_eq!(parsed_fields.fields.len(), 1);
    assert_eq!(parsed_fields.fields[0].name, "id");

    // Stored fields recovered through the compound reader must still
    // open and yield the exact documents flushed, in order.
    let reader = stored_fields::open(fdt, fdx, fdm, &segment_id, "").unwrap();
    for (i, expected) in docs.iter().enumerate() {
        let got = reader.document(i as i32).unwrap();
        assert_eq!(got.fields.len(), expected.fields.len());
        let expected_value = match &expected.fields[0].value {
            FieldValue::String(s) => s.clone(),
            other => panic!("unexpected fixture field value shape: {other:?}"),
        };
        match &got.fields[0].value {
            FieldValue::String(s) => assert_eq!(*s, expected_value),
            other => panic!("unexpected recovered field value shape: {other:?}"),
        }
    }
}

use lucene_util::test_support::TempDir;

#[test]
fn a_flush_with_large_stored_values_streams_the_fdt_the_buffered_path_writes() {
    let big = |seed: u32, len: usize| {
        let mut x = seed.max(1);
        let text: String = (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                char::from(b'a' + (x % 26) as u8)
            })
            .collect();
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String(text),
            }],
        }
    };
    let id = [7u8; ID_LENGTH];
    // Small ones, then ones past the spill size, one of them shared.
    let mut docs = vec![big(1, 40), big(2, 300_000), big(3, 1_500_000), big(4, 9)];
    docs.push(Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::Shared(std::sync::Arc::from("shared text ".repeat(200_000))),
        }],
    });
    let root = TempDir::new("flush-stored-spill");
    let dir = FsDirectory::open(root.path());
    let reference = build_native_stored_fields(&docs, &id, None).expect("native writer");
    let spilled = build_native_stored_fields_to(&dir, "s.fdt", &docs, &id, None)
        .unwrap()
        .expect("native writer");
    assert!(spilled.0.is_empty(), "the .fdt went to its file");
    assert_eq!(spilled.1, reference.1, ".fdx");
    assert_eq!(spilled.2, reference.2, ".fdm");
    assert_eq!(
        std::fs::read(root.path().join("s.fdt")).unwrap(),
        reference.0,
        ".fdt on disk equals the buffered one"
    );
}

#[test]
fn a_spilling_flush_lists_and_writes_the_files_a_buffered_one_does() {
    let docs: Vec<Document> = [40usize, 1_300_000, 1_900_000]
        .iter()
        .map(|&n| Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("abcdefghij".repeat(n / 10)),
            }],
        })
        .collect();
    let id = [9u8; ID_LENGTH];
    let reference = build_native_stored_fields(&docs, &id, None).expect("native writer");
    let flush = |name: &str, compound: bool| {
        let root = TempDir::new(name);
        let dir = FsDirectory::open(root.path());
        let flushed = write_stored_only_segment_files(
            &dir,
            "_0",
            id,
            "Lucene104",
            version(),
            &[stored_only_field("f", 0)],
            &docs,
            compound,
            false,
        )
        .unwrap();
        (root, flushed.info.files)
    };
    let (root, files) = flush("flush-spill-files", false);
    for ext in ["fdt", "fdx", "fdm", "fnm", "si"] {
        assert!(
            files.iter().any(|f| f.ends_with(&format!(".{ext}"))),
            "{ext} is listed: {files:?}"
        );
    }
    let fdt = files.iter().find(|f| f.ends_with(".fdt")).unwrap();
    assert_eq!(
        std::fs::read(root.path().join(fdt)).unwrap(),
        reference.0,
        "the spilled .fdt is the buffered one"
    );
    // A compound flush packs the .fdt from memory: no standalone file.
    let (root, files) = flush("flush-spill-compound", true);
    assert!(files.iter().all(|f| !f.ends_with(".fdt")), "{files:?}");
    assert!(!root.path().join("_0.fdt").exists());
    assert!(root.path().join("_0.cfs").exists());
}

#[test]
fn a_flush_with_small_stored_values_keeps_the_fdt_in_memory() {
    let docs = vec![Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String("small".into()),
        }],
    }];
    let root = TempDir::new("flush-stored-small");
    let dir = FsDirectory::open(root.path());
    let built = build_native_stored_fields_to(&dir, "s.fdt", &docs, &[7; ID_LENGTH], None)
        .unwrap()
        .unwrap();
    assert!(
        !built.0.is_empty(),
        "nothing is spilled below the spill size"
    );
    assert!(
        !root.path().join("s.fdt").exists(),
        "no file is created for a small flush"
    );
}

/// A scratch directory that removes itself when the test ends -- unless the
/// test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("segment-writer")
}

/// A `BinarySortField` tier sorts by its rank key with `StringSorter`'s
/// sentinels: missing first unless `STRING_LAST`, and `reverse` flips
/// the sentinel too.
#[test]
fn a_binary_sort_tier_sorts_by_rank_with_string_sentinels() {
    let keys = [Some(1), None, Some(0), None];
    for (missing, reverse, want) in [
        (
            crate::segment_info::StringMissingValue::None,
            false,
            vec![1, 3, 2, 0],
        ),
        (
            crate::segment_info::StringMissingValue::Last,
            false,
            vec![2, 0, 1, 3],
        ),
        (
            crate::segment_info::StringMissingValue::Last,
            true,
            vec![1, 3, 0, 2],
        ),
    ] {
        let sort = IndexSortField {
            field: "bytes".to_string(),
            reverse,
            kind: crate::segment_info::IndexSortKind::Binary(missing),
        };
        assert_eq!(
            sort_permutation(
                4,
                &[SortKeySpec {
                    sort: &sort,
                    keys: &keys,
                }],
            ),
            want,
            "{missing:?} reverse={reverse}"
        );
    }
}

fn doc_ids(
    dir: &FsDirectory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
    n: usize,
) -> Vec<String> {
    let fdt = dir.open(&format!("{segment_name}.fdt")).unwrap().to_vec();
    let fdx = dir.open(&format!("{segment_name}.fdx")).unwrap().to_vec();
    let fdm = dir.open(&format!("{segment_name}.fdm")).unwrap().to_vec();
    let reader = stored_fields::open(&fdt, &fdx, &fdm, &segment_id, "").unwrap();
    (0..n as i32)
        .map(|i| {
            let d = reader.document(i).unwrap();
            match &d.fields[0].value {
                FieldValue::String(s) => s.clone(),
                other => panic!("unexpected field value shape: {other:?}"),
            }
        })
        .collect()
}

/// Reads back the `.si`'s index-sort descriptor for a flushed segment.
fn read_index_sort(
    dir: &FsDirectory,
    segment_name: &str,
    segment_id: [u8; ID_LENGTH],
) -> Option<Vec<crate::segment_info::IndexSortField>> {
    let si_bytes = dir.open(&format!("{segment_name}.si")).unwrap().to_vec();
    segment_info::parse(&si_bytes, &segment_id)
        .unwrap()
        .index_sort
}

#[test]
fn sorted_flush_reorders_docs_ascending_by_numeric_key() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [11u8; ID_LENGTH];
    // Insertion order: c(30), a(10), b(20) -- must come out a, b, c.
    let docs = vec![doc("c"), doc("a"), doc("b")];
    let sort_keys = vec![Some(30), Some(10), Some(20)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[SortKeySpec {
            sort: &IndexSortField::long("num", false, Some(i64::MAX)),
            keys: &sort_keys,
        }],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["a", "b", "c"]);
    let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
    assert_eq!(fields_sort.len(), 1);
    assert_eq!(
        fields_sort[0],
        IndexSortField::long("num", false, Some(i64::MAX))
    );
}

#[test]
fn sorted_flush_reorders_docs_descending_by_numeric_key() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [12u8; ID_LENGTH];
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let sort_keys = vec![Some(10), Some(20), Some(30)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[SortKeySpec {
            sort: &IndexSortField::long("num", true, Some(i64::MIN)),
            keys: &sort_keys,
        }],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["c", "b", "a"]);
    let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
    assert!(fields_sort[0].reverse);
}

#[test]
fn sorted_flush_places_missing_values_first() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [13u8; ID_LENGTH];
    // b has no value; with MissingValue::First it must land before a/c
    // regardless of its insertion position.
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let sort_keys = vec![Some(10), None, Some(20)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[SortKeySpec {
            sort: &IndexSortField::long("num", false, Some(i64::MIN)),
            keys: &sort_keys,
        }],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "a", "c"]);
}

#[test]
fn sorted_flush_places_missing_values_last() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [14u8; ID_LENGTH];
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let sort_keys = vec![Some(10), None, Some(20)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[SortKeySpec {
            sort: &IndexSortField::long("num", false, Some(i64::MAX)),
            keys: &sort_keys,
        }],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["a", "c", "b"]);
}

/// The missing-value **sentinel is reversed along with every other
/// value**, which is what real Lucene does and what the `.si` this flush
/// writes actually says.
///
/// `segment_info::write_sort_field` emits `SortField(field, LONG,
/// reverse)` with `missingValue = Long.MAX_VALUE` for
/// missing-last, and Java's reader-side comparator for that
/// is `reverseMul * Long.compare(values[d1], values[d2])` over an array
/// pre-filled with the sentinel (`IndexSorter.LongSorter`). So with
/// `reverse: true` the `Long.MAX_VALUE` doc compares **greatest** and,
/// after `reverseMul`, lands **first**.
///
/// ("Missing-last stays last even when reversed" would make the physical
/// order disagree with the `.si` describing it -- a disagreement real
/// Lucene's `CheckIndex.testSort` rejects outright.)
#[test]
fn sorted_flush_missing_value_sentinel_is_reversed_like_any_other_value() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [16u8; ID_LENGTH];
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let sort_keys = vec![Some(10), None, Some(20)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[SortKeySpec {
            // descending, missing-last
            sort: &IndexSortField::long("num", true, Some(i64::MAX)),
            keys: &sort_keys,
        }],
    )
    .unwrap();

    // Values as Lucene sees them: a=10, b=Long.MAX_VALUE (missing,
    // `Last`), c=20. Ascending that is a, c, b; reversed it is b, c, a.
    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
}

#[test]
fn sorted_flush_is_stable_for_equal_keys() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [15u8; ID_LENGTH];
    // a and b tie at key 10 -- must keep original relative order (a
    // before b), with c(5) placed first.
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let sort_keys = vec![Some(10), Some(10), Some(5)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[SortKeySpec {
            sort: &IndexSortField::long("num", false, Some(i64::MAX)),
            keys: &sort_keys,
        }],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["c", "a", "b"]);
}

#[test]
#[should_panic(expected = "sort_keys must have exactly one entry per doc")]
fn sorted_flush_panics_on_mismatched_sort_keys_length() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];

    let _ = flush_sorted_stored_only_segment(
        &dir,
        "_0",
        [16u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &[doc("a"), doc("b")],
        &[SortKeySpec {
            sort: &IndexSortField::long("num", false, Some(i64::MAX)),
            keys: &[Some(1)],
        }],
    );
}

#[test]
#[should_panic(expected = "sort_fields must contain at least one sort key")]
fn sorted_flush_panics_on_empty_sort_fields() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];

    let _ = flush_sorted_stored_only_segment(
        &dir,
        "_0",
        [17u8; ID_LENGTH],
        "Lucene104",
        version(),
        &fields,
        &[doc("a")],
        &[],
    );
}

/// The core new multi-field behavior: field 1 has ties (all docs share
/// key `1`), so field 2 must break them. Without priority-ordered
/// comparison this would either stay in insertion order (ignoring field
/// 2 entirely) or crash.
#[test]
fn sorted_flush_second_field_breaks_ties_in_first() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [20u8; ID_LENGTH];
    // Every doc ties on `primary`; `secondary` must decide the order.
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let primary = vec![Some(1), Some(1), Some(1)];
    let secondary = vec![Some(30), Some(10), Some(20)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[
            SortKeySpec {
                sort: &IndexSortField::long("primary", false, Some(i64::MAX)),
                keys: &primary,
            },
            SortKeySpec {
                sort: &IndexSortField::long("secondary", false, Some(i64::MAX)),
                keys: &secondary,
            },
        ],
    )
    .unwrap();

    // Tied on `primary`; ascending `secondary` breaks the tie: b(10),
    // c(20), a(30).
    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
    let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
    assert_eq!(fields_sort.len(), 2);
    assert_eq!(fields_sort[0].field, "primary");
    assert_eq!(fields_sort[1].field, "secondary");
}

/// The primary field actually differs here, so ties never reach the
/// second field's comparator -- confirms the primary field, not the
/// secondary, decides the order whenever it can.
#[test]
fn sorted_flush_first_field_wins_when_it_differs() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [21u8; ID_LENGTH];
    let docs = vec![doc("a"), doc("b"), doc("c")];
    // Distinct primary values; secondary is reversed from primary order
    // to prove primary wins outright.
    let primary = vec![Some(30), Some(10), Some(20)];
    let secondary = vec![Some(1), Some(2), Some(3)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[
            SortKeySpec {
                sort: &IndexSortField::long("primary", false, Some(i64::MAX)),
                keys: &primary,
            },
            SortKeySpec {
                sort: &IndexSortField::long("secondary", false, Some(i64::MAX)),
                keys: &secondary,
            },
        ],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
}

/// Each field's `reverse` flag is independent: field 1 descending while
/// field 2 ascending, with field 1 tied so field 2 must decide -- proves
/// one field's reverse setting doesn't leak into another's comparison.
#[test]
fn sorted_flush_second_field_reverse_is_independent_of_first() {
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let fields = vec![stored_only_field("id", 0)];
    let segment_id = [22u8; ID_LENGTH];
    let docs = vec![doc("a"), doc("b"), doc("c")];
    let primary = vec![Some(1), Some(1), Some(1)];
    let secondary = vec![Some(10), Some(30), Some(20)];

    flush_sorted_stored_only_segment(
        &dir,
        "_0",
        segment_id,
        "Lucene104",
        version(),
        &fields,
        &docs,
        &[
            SortKeySpec {
                // descending, but tied -- irrelevant here
                sort: &IndexSortField::long("primary", true, Some(i64::MAX)),
                keys: &primary,
            },
            SortKeySpec {
                // descending: b(30), c(20), a(10)
                sort: &IndexSortField::long("secondary", true, Some(i64::MAX)),
                keys: &secondary,
            },
        ],
    )
    .unwrap();

    assert_eq!(doc_ids(&dir, "_0", segment_id, 3), vec!["b", "c", "a"]);
    let fields_sort = read_index_sort(&dir, "_0", segment_id).unwrap();
    assert!(fields_sort[0].reverse);
    assert!(fields_sort[1].reverse);
}

/// `permute_in_place` must implement `newToOld` -- `items[i]` becomes
/// what `items[new_to_old[i]]` was. The inverse permutation is a
/// plausible, self-consistent, wrong answer (it is the same permutation
/// for every involution, so a two-element or reversed test cannot tell
/// them apart): this uses a 3-cycle, which can.
#[test]
fn permute_in_place_applies_new_to_old_not_its_inverse() {
    let mut items = vec!["a", "b", "c"];
    // doc 0 <- old 2, doc 1 <- old 0, doc 2 <- old 1.
    permute_in_place(&mut items, &[2, 0, 1]);
    assert_eq!(items, vec!["c", "a", "b"]);

    // An involution, to show it is right there too, and the identity.
    let mut items = vec![1, 2, 3, 4];
    permute_in_place(&mut items, &[3, 1, 2, 0]);
    assert_eq!(items, vec![4, 2, 3, 1]);
    let mut items = vec![1, 2, 3];
    permute_in_place(&mut items, &[0, 1, 2]);
    assert_eq!(items, vec![1, 2, 3]);
    let mut empty: Vec<u8> = Vec::new();
    permute_in_place(&mut empty, &[]);
    assert!(empty.is_empty());
}

/// The permutation is the same one `flush_sorted_stored_only_segment`
/// and `IndexWriter::flush` both apply, so it is worth pinning
/// separately: multi-tier priority, stability, and the missing-value
/// sentinel reversing with everything else.
#[test]
fn sort_permutation_is_priority_ordered_stable_and_sentinel_reversing() {
    let primary = vec![Some(1), Some(1), Some(2), Some(1)];
    let secondary = vec![Some(30), Some(10), Some(0), Some(10)];
    let order = sort_permutation(
        4,
        &[
            SortKeySpec {
                sort: &IndexSortField::long("p", false, Some(i64::MAX)),
                keys: &primary,
            },
            SortKeySpec {
                sort: &IndexSortField::long("s", false, Some(i64::MAX)),
                keys: &secondary,
            },
        ],
    );
    // p ascending: {1,3} tie at (1,10) -- kept in insertion order -- then
    // 0 at (1,30), then 2 at p=2.
    assert_eq!(order, vec![1, 3, 0, 2]);

    // One tier, reversed, with a missing value: `Last` is Long.MAX_VALUE,
    // so reversed it sorts first.
    let keys = vec![Some(5), None, Some(9)];
    let order = sort_permutation(
        3,
        &[SortKeySpec {
            sort: &IndexSortField::long("p", true, Some(i64::MAX)),
            keys: &keys,
        }],
    );
    assert_eq!(order, vec![1, 2, 0]);
}
