//! Differential test for the **byte-keyed index sorts** -- `SortField.Type.STRING`
//! over SORTED, `SortedSetSortField` (every selector) over SORTED_SET and
//! `BinarySortField` over BINARY -- against the physical document order real
//! Lucene gives them. Regenerate with `fixtures/src/GenStringSortedIndex.java`.
//!
//! For each of the fixture's eight sorts:
//!
//! 1. The sort is read out of the `.si` of Java's merged segment, and
//!    [`segment_info::describe_index_sort`] must print it exactly as Lucene's
//!    `Sort.toString()` did -- so the sort this test replays *is* the one
//!    Java used, not a transcription of it.
//! 2. This port's `CheckIndex` verifies Java's sorted segment (its `testSort`
//!    runs for every kind, none skipped).
//! 3. This port's `IndexWriter` indexes the same 120 documents with that sort
//!    in the same three committed batches, and each flushed segment must hold
//!    its documents in exactly Java's order (`IndexingChain.maybeSortSegment`);
//!    then the same documents are deleted and the three segments are
//!    force-merged, and the merged segment must match Java's order too
//!    (`MultiSorter`'s k-way merge, ties broken by segment then doc id).
//!    `CheckIndex` must accept both.
// Test-support code: see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::check_index;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::log::LogDocMergePolicy;
use lucene_index::segment_info::{self, IndexSortField, LuceneVersion};
use lucene_index::segment_infos::{self, SegmentCommitInfo};
use lucene_store::directory::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

fn fixture() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/string_sorted_index"
    ))
}

fn manifest() -> Vec<(String, String)> {
    std::fs::read_to_string(fixture().join("manifest.properties"))
        .expect("run fixtures generator first (GenStringSortedIndex)")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

/// `"~"` is the empty value, `"-"` an absent field.
fn unhex(s: &str) -> Vec<u8> {
    if s == "~" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

const ID: i32 = 0;
const NAME: i32 = 1;
const TAGS: i32 = 2;
const BLOB: i32 = 3;
const SEQ: i32 = 4;

fn fields() -> Vec<FieldInfo> {
    let dv = |name: &str, number, t| {
        FieldInfo::new(name, number)
            .with_omit_norms(true)
            .with_doc_values(t, DocValuesSkipIndexType::None, -1)
    };
    vec![
        FieldInfo::new("id", ID)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        dv("name", NAME, DocValuesType::Sorted),
        dv("tags", TAGS, DocValuesType::SortedSet),
        dv("blob", BLOB, DocValuesType::Binary),
        dv("seq", SEQ, DocValuesType::Numeric),
    ]
}

/// The manifest's `doc.<i>` rows, as documents.
fn documents(m: &[(String, String)]) -> Vec<(String, Document)> {
    let n: usize = get(m, "num_docs").parse().unwrap();
    (0..n)
        .map(|i| {
            let row = get(m, &format!("doc.{i}"));
            let cols: Vec<&str> = row.split('|').collect();
            let mut fields = vec![StoredField {
                field_number: ID,
                value: FieldValue::String(cols[0].to_string()),
            }];
            if cols[1] != "-" {
                fields.push(StoredField {
                    field_number: NAME,
                    value: FieldValue::Binary(unhex(cols[1])),
                });
            }
            if cols[2] != "-" {
                for tag in cols[2].split(',') {
                    fields.push(StoredField {
                        field_number: TAGS,
                        value: FieldValue::Binary(unhex(tag)),
                    });
                }
            }
            if cols[3] != "-" {
                fields.push(StoredField {
                    field_number: BLOB,
                    value: FieldValue::Binary(unhex(cols[3])),
                });
            }
            fields.push(StoredField {
                field_number: SEQ,
                value: FieldValue::Long(cols[4].parse().unwrap()),
            });
            (cols[0].to_string(), Document { fields })
        })
        .collect()
}

/// Every document's stored `id`, in doc-id order, deleted ones included.
fn ids(dir: &dyn Directory, sci: &SegmentCommitInfo) -> Vec<String> {
    let name = &sci.segment_name;
    let id = &sci.segment_id;
    let read = |ext: &str| dir.open(&format!("{name}.{ext}")).unwrap();
    let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
    let reader = lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, id, "").unwrap();
    let fnm = read("fnm");
    let infos = lucene_codecs::field_infos::parse(&fnm, id, "").unwrap();
    let id_field = infos.fields.iter().find(|f| f.name == "id").unwrap().number;
    (0..reader.max_doc())
        .map(|d| {
            reader
                .document(d)
                .unwrap()
                .fields
                .iter()
                .find(|f| f.field_number == id_field)
                .and_then(|f| match &f.value {
                    FieldValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .expect("every document stores its id")
        })
        .collect()
}

/// The sort Java configured, read back out of Java's own `.si`.
fn java_sort(config: &str) -> Vec<IndexSortField> {
    let dir = FsDirectory::open(fixture().join(config));
    let sis = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(sis.segments.len(), 1, "{config}: Java force-merged to one");
    let sci = &sis.segments[0];
    let si = segment_info::parse(
        &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
        &sci.segment_id,
    )
    .unwrap();
    si.index_sort.expect("Java's segment declares its sort")
}

fn assert_check_index_passes(dir: &dyn Directory, what: &str) {
    let results = check_index::check_directory(dir).expect("check index");
    for result in &results {
        assert!(
            result.all_passed(),
            "{what} ({}): {:?}",
            result.segment_name,
            result.failures()
        );
        if result.segment_name.starts_with('_') {
            assert!(
                result
                    .checks
                    .iter()
                    .any(|c| c.name == "sort.docs_in_index_sort_order" && c.passed()),
                "{what} ({}): testSort must run, not skip",
                result.segment_name
            );
        }
    }
}

fn configs(m: &[(String, String)]) -> Vec<String> {
    get(m, "configs").split(',').map(str::to_string).collect()
}

/// Steps 1 and 2: the sort round-trips to Lucene's own description, and
/// this port's `CheckIndex` verifies Java's sorted segment of every kind.
#[test]
fn java_sorted_segments_of_every_byte_kind_pass_our_check_index() {
    let m = manifest();
    let configs = configs(&m);
    assert_eq!(configs.len(), 8);
    for config in &configs {
        let sort = java_sort(config);
        assert_eq!(
            segment_info::describe_index_sort(Some(&sort)),
            get(&m, &format!("{config}.sort")),
            "{config}"
        );
        let dir = FsDirectory::open(fixture().join(config));
        assert_check_index_passes(&dir, &format!("Java's {config}"));
        let sis = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(
            ids(&dir, &sis.segments[0]).join(","),
            get(&m, &format!("{config}.merged")),
            "{config}: the stored order this test reads is Java's"
        );
    }
}

/// Step 3: this port's writer produces Java's order, flushed and merged.
#[test]
fn our_writer_sorts_and_merges_every_byte_kind_as_lucene_does() {
    let m = manifest();
    let docs = documents(&m);
    let batches: usize = get(&m, "batches").parse().unwrap();
    let per_batch: usize = get(&m, "per_batch").parse().unwrap();
    let deleted: Vec<&str> = get(&m, "deleted").split(',').collect();
    for config in configs(&m) {
        let sort = java_sort(&config);
        let tmp = TempDir::new(&format!("string-sort-{config}"));
        let dir = FsDirectory::open(&tmp);
        let version = LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut writer = IndexWriter::open(&dir, fields(), "Lucene104", version).unwrap();
        writer.set_postings_field(Some("id")).unwrap();
        writer.set_doc_values_field(Some("name")).unwrap();
        for f in ["tags", "blob", "seq"] {
            writer.add_doc_values_field(f).unwrap();
        }
        writer.set_index_sort(Some(&sort)).unwrap();
        // Java's generator merges with `LogDocMergePolicy`, whose forced
        // merge takes the segments in index order -- which decides the
        // k-way merge's tie-break between documents equal on every tier.
        let mut policy = LogDocMergePolicy::new();
        policy.set_merge_factor(1000).unwrap();
        writer.set_pluggable_merge_policy(Some(Arc::new(policy)));

        for b in 0..batches {
            for (_, doc) in &docs[b * per_batch..(b + 1) * per_batch] {
                writer.add_document(doc.clone()).unwrap();
            }
            writer.commit().unwrap();
        }
        let flushed = writer.segment_infos().segments.clone();
        assert_eq!(flushed.len(), batches, "{config}: one segment per batch");
        for (ord, sci) in flushed.iter().enumerate() {
            assert_eq!(
                ids(&dir, sci).join(","),
                get(&m, &format!("{config}.flushed.{ord}")),
                "{config}: flushed segment {ord} is not in Lucene's order"
            );
        }
        assert_check_index_passes(&dir, &format!("our flushed {config}"));

        let terms: Vec<Term> = deleted.iter().map(|id| Term::new("id", *id)).collect();
        writer.delete_documents_by_term(&terms).unwrap();
        writer.commit().unwrap();
        writer.force_merge(1).unwrap();
        let merged = writer.segment_infos().segments.clone();
        assert_eq!(merged.len(), 1, "{config}");
        assert_eq!(
            ids(&dir, &merged[0]).join(","),
            get(&m, &format!("{config}.merged")),
            "{config}: the merged segment is not in Lucene's order"
        );
        assert_check_index_passes(&dir, &format!("our merged {config}"));
    }
}
