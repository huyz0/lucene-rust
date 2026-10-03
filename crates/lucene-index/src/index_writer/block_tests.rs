//! Document blocks through the document API: the parent field
//! (`IndexWriterConfig.setParentField`), blocks under an index sort at flush
//! and at merge, and the refusals around both.

// Test fixtures' own arithmetic, not values read off disk -- see
// `docs/arithmetic-gate.md`'s "Test code" section.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::buffered_updates::{DocValuesUpdate, Term};
use crate::document::{
    Document as FieldsDocument, NumericDocValuesField, SortedDocValuesField,
    SortedSetDocValuesField, Store, StringField,
};
use crate::segment_info::{IndexSortField, IndexSortKind, SortedSetSelector, StringMissingValue};
use lucene_codecs::stored_fields::StoredField;
use lucene_store::directory::FsDirectory;
use lucene_util::test_support::TempDir;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn writer<'d>(dir: &'d FsDirectory) -> IndexWriter<'d> {
    IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap()
}

/// Block `b`: `children` children `c{b}_{j}` and the parent `p{b}`, every
/// document tagged `block:{b}`; the parent alone carries `rank` = `rank`
/// and `name` = `n{rank}`.
fn block(b: usize, children: usize, rank: i64) -> Vec<FieldsDocument> {
    let mut docs = Vec::new();
    for j in 0..children {
        let mut d = FieldsDocument::new();
        d.add(StringField::new("id", format!("c{b}_{j}"), Store::Yes));
        d.add(StringField::new("block", b.to_string(), Store::No));
        d.add(StringField::new("type", "child", Store::No));
        docs.push(d);
    }
    let mut p = FieldsDocument::new();
    p.add(StringField::new("id", format!("p{b}"), Store::Yes));
    p.add(StringField::new("block", b.to_string(), Store::No));
    p.add(StringField::new("type", "parent", Store::No));
    p.add(NumericDocValuesField::new("rank", rank));
    p.add(SortedDocValuesField::new(
        "name",
        format!("n{rank:04}").into_bytes(),
    ));
    p.add(SortedSetDocValuesField::new(
        "tags",
        format!("t{:04}", 1000 - rank).into_bytes(),
    ));
    p.add(SortedSetDocValuesField::new("tags", b"zz".to_vec()));
    docs.push(p);
    docs
}

/// Every live document's stored `id`, segment by segment.
fn segment_ids(dir: &FsDirectory, infos: &SegmentInfos) -> Vec<Vec<String>> {
    infos
        .segments
        .iter()
        .map(|sci| {
            let fdt = dir.open(&format!("{}.fdt", sci.segment_name)).unwrap();
            let fdx = dir.open(&format!("{}.fdx", sci.segment_name)).unwrap();
            let fdm = dir.open(&format!("{}.fdm", sci.segment_name)).unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
            let live = (sci.del_gen >= 0).then(|| {
                let liv = dir
                    .open(&deletes::liv_file_name(&sci.segment_name, sci.del_gen))
                    .unwrap();
                lucene_codecs::live_docs::parse(
                    &liv,
                    &sci.segment_id,
                    sci.del_gen,
                    reader.max_doc() as usize,
                    sci.del_count as usize,
                )
                .unwrap()
            });
            (0..reader.max_doc())
                .filter(|&d| live.as_ref().is_none_or(|l| l.get(d as usize)))
                .map(|d| match &reader.document(d).unwrap().fields[0].value {
                    FieldValue::String(s) => s.clone(),
                    other => panic!("{other:?}"),
                })
                .collect()
        })
        .collect()
}

/// The documents of a segment carrying the parent field, from its column.
fn parent_docs(dir: &FsDirectory, sci: &SegmentCommitInfo) -> (String, Vec<i32>) {
    let si = segment_info::parse_for_codec(
        &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
        &sci.segment_id,
        &sci.codec_name,
    )
    .unwrap();
    let infos = segment_field_infos(dir, sci).unwrap();
    let parent = infos.fields.iter().find(|f| f.parent_field).unwrap();
    assert_eq!(parent.doc_values_type, DocValuesType::Numeric);
    let (meta, dvd) = crate::check_index::open_doc_values(dir, sci, &si, &infos)
        .unwrap()
        .unwrap();
    let mut reader =
        doc_values::NumericReader::new(&dvd, meta.numeric_entry(parent.number).unwrap());
    let mut docs = Vec::new();
    for doc in 0..si.doc_count {
        if let Some(v) = reader.value(doc).unwrap() {
            assert_eq!(v, -1, "the parent field's value");
            docs.push(doc);
        }
    }
    (parent.name.clone(), docs)
}

fn check(dir: &FsDirectory) {
    for r in crate::check_index::check_directory(dir).unwrap() {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
}

/// Every segment's blocks are whole and in order: each run of children is
/// closed by its own parent, and the parents are exactly the documents the
/// parent field marks.
fn assert_blocks(dir: &FsDirectory, infos: &SegmentInfos) -> Vec<Vec<String>> {
    let ids = segment_ids(dir, infos);
    for (sci, seg) in infos.segments.iter().zip(&ids) {
        let mut open: Option<String> = None;
        let mut next_child = 0;
        for id in seg {
            let b = id[1..].split('_').next().unwrap().to_string();
            if id.starts_with('c') {
                if open.as_ref() != Some(&b) {
                    assert!(open.is_none() || next_child > 0, "{seg:?}");
                    open = Some(b);
                    next_child = 0;
                }
                assert_eq!(id, &format!("c{}_{next_child}", open.as_ref().unwrap()));
                next_child += 1;
            } else {
                assert!(open.is_none() || open == Some(b.clone()), "{id} in {seg:?}");
                open = None;
                next_child = 0;
            }
        }
        assert!(open.is_none(), "a segment ends in a parent: {seg:?}");
        if sci.del_gen < 0 {
            let (_, parents) = parent_docs(dir, sci);
            let expected: Vec<i32> = seg
                .iter()
                .enumerate()
                .filter(|(_, id)| id.starts_with('p'))
                .map(|(i, _)| i as i32)
                .collect();
            assert_eq!(parents, expected);
        }
    }
    ids
}

fn parents_in_order(ids: &[String]) -> Vec<String> {
    ids.iter()
        .filter(|id| id.starts_with('p'))
        .cloned()
        .collect()
}

/// `IndexingChain.processDocument(docId, doc, lastDocInBlock)`: the last
/// document of every add -- a lone document too -- gets `-1` in the parent
/// field, whose `FieldInfo` says so; the parent field is numbered before the
/// fields of the document that first carries it, as Java handles it first.
#[test]
fn the_parent_field_marks_the_last_document_of_every_add() {
    let tmp = TempDir::new("parent-field-marks");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_parent_field(Some("_parent")).unwrap();
    assert_eq!(w.parent_field(), Some("_parent"));
    let mut single = FieldsDocument::new();
    single.add(StringField::new("id", "p0", Store::Yes));
    w.add_fields_document(&single).unwrap();
    w.add_fields_documents(&block(1, 2, 5)).unwrap();
    w.add_fields_documents(&block(2, 0, 3)).unwrap();
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    let ids = assert_blocks(&dir, &infos);
    assert_eq!(ids[0], ["p0", "c1_0", "c1_1", "p1", "p2"]);
    let (name, parents) = parent_docs(&dir, &infos.segments[0]);
    assert_eq!((name.as_str(), parents), ("_parent", vec![0, 3, 4]));
    let fields = segment_field_infos(&dir, &infos.segments[0]).unwrap();
    let numbers: Vec<(&str, i32)> = fields
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.number))
        .collect();
    assert_eq!(numbers[..2], [("_parent", 0), ("id", 1)]);
    let si = segment_info::parse_for_codec(
        &dir.open(&format!("{}.si", infos.segments[0].segment_name))
            .unwrap(),
        &infos.segments[0].segment_id,
        &infos.segments[0].codec_name,
    )
    .unwrap();
    assert!(si.has_blocks);
    check(&dir);
}

/// The parent field is the writer's: a document naming it is refused,
/// through the document API and as an explicit document.
#[test]
fn a_document_carrying_the_parent_field_is_refused() {
    let tmp = TempDir::new("parent-field-reserved");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_parent_field(Some("_parent")).unwrap();
    let mut d = FieldsDocument::new();
    d.add(NumericDocValuesField::new("_parent", 7));
    let err = w.add_fields_documents(&[d]).unwrap_err();
    assert!(
        err.to_string()
            .contains("\"_parent\" is a reserved field and should not be added to any document"),
        "{err}"
    );
    // Once registered, its number is refused on an explicit document too.
    w.add_fields_document(&block(0, 0, 1)[0]).unwrap();
    let parent = w.cfg.parent_field_number().unwrap();
    let doc = ExplicitDocument {
        stored: Vec::new(),
        fields: ExplicitFields {
            doc_values: vec![StoredField {
                field_number: parent,
                value: FieldValue::Long(-1),
            }],
            ..ExplicitFields::default()
        },
    };
    let err = w.add_explicit_documents(vec![doc]).unwrap_err();
    assert!(err.to_string().contains("reserved field"), "{err}");
    // And it cannot be registered as an ordinary field, nor another field
    // as a parent.
    let err = w
        .register_field(FieldInfo::new("_parent", 0).with_doc_values(
            DocValuesType::Numeric,
            lucene_codecs::field_infos::DocValuesSkipIndexType::None,
            -1,
        ))
        .unwrap_err();
    assert!(
        err.to_string().contains("as non parent document field"),
        "{err}"
    );
    let err = w
        .register_field(
            FieldInfo::new("other", 0)
                .with_doc_values(
                    DocValuesType::Numeric,
                    lucene_codecs::field_infos::DocValuesSkipIndexType::None,
                    -1,
                )
                .with_parent_field(true),
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("is configured with [_parent]"),
        "{err}"
    );
}

/// `FieldNumbers.verifyParentFieldName` without a configured parent field,
/// and the configuration's own preconditions.
#[test]
fn the_parent_field_is_fixed_before_documents_and_for_the_index() {
    let tmp = TempDir::new("parent-field-fixed");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = writer(&dir);
        w.enable_explicit_documents().unwrap();
        let err = w
            .register_field(
                FieldInfo::new("p", 0)
                    .with_doc_values(
                        DocValuesType::Numeric,
                        lucene_codecs::field_infos::DocValuesSkipIndexType::None,
                        -1,
                    )
                    .with_parent_field(true),
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("has no parent document field configured"));
        w.add_fields_document(&block(0, 0, 1)[0]).unwrap();
        // Too late: a document is buffered.
        let err = w.set_parent_field(Some("_parent")).unwrap_err();
        assert!(err.to_string().contains("before any document is buffered"));
        w.commit().unwrap();
    }
    // An existing index with fields but no parent field.
    let mut w = writer(&dir);
    let err = w.set_parent_field(Some("_parent")).unwrap_err();
    assert!(
        err.to_string().contains(
            "can't add a parent field to an already existing index without a parent field"
        ),
        "{err}"
    );
    // A field of the index cannot become the parent field.
    let err = w.set_parent_field(Some("id")).unwrap_err();
    assert!(
        err.to_string().contains("as non parent document field"),
        "{err}"
    );
    w.set_parent_field(None).unwrap();
    assert_eq!(w.parent_field(), None);
    drop(w);

    // An index created with a parent field keeps it.
    let tmp = TempDir::new("parent-field-kept");
    let dir = FsDirectory::open(tmp.path());
    {
        let mut w = writer(&dir);
        w.set_parent_field(Some("_parent")).unwrap();
        w.add_fields_documents(&block(0, 1, 1)).unwrap();
        w.commit().unwrap();
    }
    let mut w = writer(&dir);
    let err = w.set_parent_field(Some("_other")).unwrap_err();
    assert!(
        err.to_string().contains("as parent document field"),
        "{err}"
    );
    w.set_parent_field(Some("_parent")).unwrap();
    w.add_fields_documents(&block(1, 1, 2)).unwrap();
    let infos = w.commit().unwrap().clone();
    assert_blocks(&dir, &infos);
    check(&dir);
    // Registered, the field holds the writer to it.
    let err = w.set_parent_field(None).unwrap_err();
    assert!(err.to_string().contains("uses [_parent]"), "{err}");
}

/// `DocumentsWriterPerThread.updateDocuments`: a block in a sorted index
/// needs a parent field, refused before anything is buffered; a lone
/// document does not.
#[test]
fn a_block_in_a_sorted_index_needs_a_parent_field() {
    let tmp = TempDir::new("sorted-blocks-no-parent");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    let err = w.add_fields_documents(&block(0, 2, 1)).unwrap_err();
    assert!(
        matches!(err, Error::BlocksWithIndexSortNeedParentField),
        "{err}"
    );
    assert_eq!(w.pending_doc_count(), 0);
    w.add_fields_documents(&block(1, 0, 1)).unwrap();
    assert_eq!(w.pending_doc_count(), 1);
}

/// `IndexingChain.maybeSortSegment` with blocks: every document sorts by the
/// key of the parent closing its block, so a flush moves whole blocks; the
/// segment records the sort and `CheckIndex.testSort` walks its parents.
#[test]
fn a_sorted_flush_moves_whole_blocks() {
    for sort in [
        IndexSortField::long("rank", false, None),
        IndexSortField::long("rank", true, Some(0)),
        IndexSortField {
            field: "name".to_string(),
            reverse: false,
            kind: IndexSortKind::String(StringMissingValue::Last),
        },
        IndexSortField {
            field: "tags".to_string(),
            reverse: false,
            kind: IndexSortKind::SortedSet {
                selector: SortedSetSelector::Min,
                missing: StringMissingValue::None,
            },
        },
    ] {
        let tmp = TempDir::new("sorted-blocks-flush");
        let dir = FsDirectory::open(tmp.path());
        let mut w = writer(&dir);
        w.set_parent_field(Some("_parent")).unwrap();
        w.set_index_sort(Some(std::slice::from_ref(&sort))).unwrap();
        let ranks = [7i64, 3, 9, 1, 5];
        for (b, &rank) in ranks.iter().enumerate() {
            w.add_fields_documents(&block(b, b % 3, rank)).unwrap();
        }
        let infos = w.commit().unwrap().clone();
        let ids = assert_blocks(&dir, &infos);
        let mut by_rank: Vec<(i64, usize)> = ranks.iter().copied().zip(0..).collect();
        by_rank.sort();
        if sort.reverse || sort.field == "tags" {
            by_rank.reverse();
        }
        let want: Vec<String> = by_rank.iter().map(|(_, b)| format!("p{b}")).collect();
        assert_eq!(parents_in_order(&ids[0]), want, "{sort:?}");
        check(&dir);
    }
}

/// `MultiSorter.sort` with blocks: merged segments interleave whole blocks
/// by their parents' keys, deleted blocks drop out whole, and the merged
/// segment keeps the parent field and passes `CheckIndex`.
#[test]
fn a_sorted_merge_moves_whole_blocks() {
    let tmp = TempDir::new("sorted-blocks-merge");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_parent_field(Some("_parent")).unwrap();
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    let mut b = 0;
    for round in 0..4i64 {
        for k in 0..5i64 {
            w.add_fields_documents(&block(b, b % 4, k * 4 + round))
                .unwrap();
            b += 1;
        }
        w.commit().unwrap();
    }
    // Delete two whole blocks by the term every one of their documents has.
    w.delete_documents_by_term(&[Term::new("block", "3"), Term::new("block", "10")])
        .unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    let infos = w.commit().unwrap().clone();
    assert_eq!(infos.segments.len(), 1);
    let ids = assert_blocks(&dir, &infos);
    let parents = parents_in_order(&ids[0]);
    assert_eq!(parents.len(), 18);
    let rank = |p: &str| -> i64 {
        let b: i64 = p[1..].parse().unwrap();
        (b % 5) * 4 + b / 5
    };
    let ranks: Vec<i64> = parents.iter().map(|p| rank(p)).collect();
    let mut sorted = ranks.clone();
    sorted.sort();
    assert_eq!(ranks, sorted);
    assert!(!parents.contains(&"p3".to_string()));
    let fields = segment_field_infos(&dir, &infos.segments[0]).unwrap();
    assert_eq!(fields.parent_field(), Some("_parent"));
    check(&dir);
}

/// `IndexWriter.softUpdateDocuments` through the document API: the new block
/// is added and the old one soft-deleted in one operation.
#[test]
fn a_soft_update_replaces_a_block() {
    let tmp = TempDir::new("soft-update-block");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.set_parent_field(Some("_parent")).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(
        FieldInfo::new("__soft", 0)
            .with_doc_values(
                DocValuesType::Numeric,
                lucene_codecs::field_infos::DocValuesSkipIndexType::None,
                -1,
            )
            .with_soft_deletes_field(true),
    )
    .unwrap();
    w.add_fields_documents(&block(0, 2, 1)).unwrap();
    w.commit().unwrap();
    let soft = DocValuesUpdate::Numeric {
        term: Term::new("block", "0"),
        field: "__soft".to_string(),
        value: Some(1),
    };
    w.soft_update_fields_documents(
        Term::new("block", "0"),
        &block(1, 1, 2),
        std::slice::from_ref(&soft),
    )
    .unwrap();
    let infos = w.commit().unwrap().clone();
    let total_soft: i32 = infos.segments.iter().map(|s| s.soft_del_count).sum();
    assert_eq!(total_soft, 3, "the whole old block");
    assert_blocks(&dir, &infos);
    check(&dir);
    let err = w
        .soft_update_fields_documents(Term::new("block", "1"), &block(2, 0, 3), &[])
        .unwrap_err();
    assert!(matches!(err, Error::NoSoftDeletesSupplied));
}

/// The explicit path's index sort: a field the sort names must carry the
/// doc values it reads, whenever it is registered; one no document has sorts
/// every document as missing.
#[test]
fn explicit_documents_validate_and_sort_by_their_doc_values() {
    let tmp = TempDir::new("explicit-sort");
    let dir = FsDirectory::open(tmp.path());
    let mut w = writer(&dir);
    w.enable_explicit_documents().unwrap();
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    let err = w
        .register_field(FieldInfo::new("rank", 0).with_index_options(IndexOptions::Docs))
        .unwrap_err();
    assert!(err.to_string().contains("invalid doc value type"), "{err}");
    let mut d = FieldsDocument::new();
    d.add(StringField::new("id", "p0", Store::Yes));
    w.add_fields_document(&d).unwrap();
    let infos = w.commit().unwrap().clone();
    assert_eq!(segment_ids(&dir, &infos), [["p0"]]);
    // No document carries `rank`: the segment records the sort over a
    // column it does not have (every document missing), which this port's
    // `CheckIndex` reports as unverifiable rather than passed.
    let sci = &infos.segments[0];
    let si = segment_info::parse_for_codec(
        &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
        &sci.segment_id,
        &sci.codec_name,
    )
    .unwrap();
    assert_eq!(si.index_sort.as_deref().map(<[_]>::len), Some(1));
    // A registered field of the wrong type is refused by a later sort.
    let mut w2 = {
        drop(w);
        writer(&dir)
    };
    w2.enable_explicit_documents().unwrap();
    w2.register_field(FieldInfo::new("plain", 0).with_index_options(IndexOptions::Docs))
        .unwrap();
    let err = w2
        .set_index_sort(Some(&[IndexSortField::long("plain", false, None)]))
        .unwrap_err();
    assert!(matches!(err, Error::UnsupportedIndexSortField(..)), "{err}");
}

/// `parents.nextSetBit(doc)`: children take their parent's key; a run after
/// the last parent keeps its own.
#[test]
fn keys_follow_the_parent_closing_each_block() {
    let mut keys = vec![Some(1), None, Some(3), Some(4), None, Some(6)];
    let parents = [false, false, true, false, true, false];
    explicit::key_of_parent(&mut keys, &|d| parents[d]);
    assert_eq!(keys, [Some(3), Some(3), Some(3), None, None, Some(6)]);
}
