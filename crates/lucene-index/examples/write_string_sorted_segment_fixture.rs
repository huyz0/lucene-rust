//! Writes indexes sorted by the three **byte-keyed** index sorts --
//! `STRING` over SORTED, `SortedSetSortField` (every selector) over
//! SORTED_SET, `BinarySortField` over BINARY -- for
//! `fixtures/src/VerifyStringSortedIndex.java` to check against real Lucene.
//!
//! For each sort in [`configs`], `<out>/<name>/flushed` holds three committed
//! batches of 150 documents (three sort-on-flush segments) and
//! `<out>/<name>/merged` the same batches with every seventh document
//! deleted and the rest force-merged into one segment. The verifier re-indexes
//! the documents it reads out of `flushed` with Lucene's own `IndexWriter`
//! under the sort it reads out of the `.si`, and requires every segment's
//! document order to be the one Lucene produces; then runs `CheckIndex`.
//!
//! Document `i` has `id = "d{i}"`; the values are pseudo-random over
//! vocabularies chosen for unsigned-byte and prefix ordering (see
//! `fixtures/src/GenStringSortedIndex.java`, the read-direction twin).
// Example code: see `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::log::LogDocMergePolicy;
use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, LuceneVersion, NumericSortKey, SortedSetSelector,
    StringMissingValue,
};
use lucene_store::FsDirectory;

const BATCHES: usize = 3;
const PER_BATCH: usize = 150;

const NAMES: &[&str] = &[
    "apple",
    "Apple",
    "\u{e4}pfel",
    "zebra",
    "z",
    "",
    "\u{ff}",
    "\u{65e5}\u{672c}",
    "ab",
    "abc",
    "b",
];
const TAGS: &[&str] = &[
    "t0", "t1", "t10", "t2", "u", "\u{e9}", "a", "zz", "m", "", "b\u{ff}", "k",
];
const BLOB_BYTES: &[u8] = &[0x00, 0x01, 0x7f, 0x80, 0xff, b'a'];

/// A small deterministic generator (SplitMix64), so the example needs no
/// dependency and every run writes the same documents.
struct Rng(u64);

impl Rng {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % bound as u64) as usize
    }
}

fn documents() -> Vec<Document> {
    let mut r = Rng(0x5eed);
    (0..BATCHES * PER_BATCH)
        .map(|i| {
            let mut fields = vec![StoredField {
                field_number: 0,
                value: FieldValue::String(format!("d{i}")),
            }];
            if r.next(6) != 0 {
                fields.push(StoredField {
                    field_number: 1,
                    value: FieldValue::Binary(NAMES[r.next(NAMES.len())].as_bytes().to_vec()),
                });
            }
            for _ in 0..r.next(5) {
                fields.push(StoredField {
                    field_number: 2,
                    value: FieldValue::Binary(TAGS[r.next(TAGS.len())].as_bytes().to_vec()),
                });
            }
            if r.next(5) != 0 {
                let len = r.next(4);
                let blob = (0..len)
                    .map(|_| BLOB_BYTES[r.next(BLOB_BYTES.len())])
                    .collect();
                fields.push(StoredField {
                    field_number: 3,
                    value: FieldValue::Binary(blob),
                });
            }
            fields.push(StoredField {
                field_number: 4,
                value: FieldValue::Long((i % 5) as i64),
            });
            Document { fields }
        })
        .collect()
}

fn fields() -> Vec<FieldInfo> {
    let dv = |name: &str, number, t| {
        FieldInfo::new(name, number)
            .with_omit_norms(true)
            .with_doc_values(t, DocValuesSkipIndexType::None, -1)
    };
    vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        dv("name", 1, DocValuesType::Sorted),
        dv("tags", 2, DocValuesType::SortedSet),
        dv("blob", 3, DocValuesType::Binary),
        dv("seq", 4, DocValuesType::Numeric),
    ]
}

fn sf(field: &str, reverse: bool, kind: IndexSortKind) -> IndexSortField {
    IndexSortField {
        field: field.to_string(),
        reverse,
        kind,
    }
}

fn configs() -> Vec<(&'static str, Vec<IndexSortField>)> {
    use StringMissingValue::{First, Last, None as Absent};
    let set = |selector, missing| IndexSortKind::SortedSet { selector, missing };
    let seq = |reverse| {
        sf(
            "seq",
            reverse,
            IndexSortKind::Numeric(NumericSortKey::Long(None)),
        )
    };
    vec![
        (
            "string_asc_last",
            vec![sf("name", false, IndexSortKind::String(Last)), seq(false)],
        ),
        (
            "string_desc",
            vec![sf("name", true, IndexSortKind::String(Absent)), seq(true)],
        ),
        (
            "set_min",
            vec![
                sf("tags", false, set(SortedSetSelector::Min, Absent)),
                seq(false),
            ],
        ),
        (
            "set_max_desc_last",
            vec![sf("tags", true, set(SortedSetSelector::Max, Last))],
        ),
        (
            "set_middle_min_then_blob",
            vec![
                sf("tags", false, set(SortedSetSelector::MiddleMin, First)),
                sf("blob", false, IndexSortKind::Binary(Absent)),
            ],
        ),
        (
            "set_middle_max_desc_then_name",
            vec![
                sf("tags", true, set(SortedSetSelector::MiddleMax, Absent)),
                sf("name", false, IndexSortKind::String(Absent)),
            ],
        ),
        (
            "binary_desc_first",
            vec![sf("blob", true, IndexSortKind::Binary(First)), seq(false)],
        ),
        (
            "binary_last",
            vec![sf("blob", false, IndexSortKind::Binary(Last))],
        ),
    ]
}

fn write(path: &str, sort: &[IndexSortField], merge: bool) -> usize {
    std::fs::create_dir_all(path).expect("create dir");
    let dir = FsDirectory::open(path);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut writer = IndexWriter::open(&dir, fields(), "Lucene104", version).expect("open");
    writer.set_postings_field(Some("id")).expect("id postings");
    writer.set_doc_values_field(Some("name")).expect("name");
    for f in ["tags", "blob", "seq"] {
        writer.add_doc_values_field(f).expect(f);
    }
    writer.set_index_sort(Some(sort)).expect("sort");
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).expect("merge factor");
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    let docs = documents();
    for batch in docs.chunks(PER_BATCH) {
        for doc in batch {
            writer.add_document(doc.clone()).expect("add");
        }
        writer.commit().expect("commit");
    }
    if merge {
        let deleted: Vec<Term> = (0..docs.len())
            .filter(|i| i % 7 == 3)
            .map(|i| Term::new("id", format!("d{i}")))
            .collect();
        writer.delete_documents_by_term(&deleted).expect("delete");
        writer.commit().expect("commit deletes");
        writer.force_merge(1).expect("force merge");
    }
    writer.segment_infos().segments.len()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_string_sorted_segment_fixture <output-dir>");
    for (name, sort) in configs() {
        assert_eq!(
            write(&format!("{out}/{name}/flushed"), &sort, false),
            BATCHES
        );
        assert_eq!(write(&format!("{out}/{name}/merged"), &sort, true), 1);
    }
    println!("wrote {} string-sorted configurations", configs().len());
}
