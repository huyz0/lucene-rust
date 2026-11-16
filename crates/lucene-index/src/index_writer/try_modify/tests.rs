#![cfg(test)]
use super::*;
use crate::buffered_updates::Term;
use crate::index_writer::DISABLE_AUTO_FLUSH_MB;
use crate::segment_info::{self, LuceneVersion};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::directory::Directory;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let fields = vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("n", 1)
            .with_omit_norms(true)
            .with_doc_values(DocValuesType::Numeric, DocValuesSkipIndexType::None, -1),
        FieldInfo::new("b", 2)
            .with_omit_norms(true)
            .with_doc_values(DocValuesType::Binary, DocValuesSkipIndexType::None, -1),
    ];
    let mut w = IndexWriter::open(
        dir,
        fields,
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.set_doc_values_field(Some("n")).unwrap();
    w.add_doc_values_field("b").unwrap();
    w.set_max_buffered_docs(1000).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w
}

fn doc(i: i64) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(format!("d{i}")),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(i),
            },
            StoredField {
                field_number: 2,
                value: FieldValue::Binary(vec![i as u8]),
            },
        ],
    }
}

/// Every document's current NUMERIC value of `field` in `sci`, through
/// the newest doc-values generation.
fn numeric_values(
    dir: &FsDirectory,
    sci: &crate::segment_infos::SegmentCommitInfo,
    field: &str,
) -> Vec<Option<i64>> {
    let si = segment_info::parse(
        &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
        &sci.segment_id,
    )
    .unwrap();
    let infos = crate::field_updates::read_current_field_infos(dir, sci, &si.files).unwrap();
    let index = infos.fields.iter().position(|f| f.name == field).unwrap();
    let per_field = crate::field_updates::per_field_component(
        &infos.fields[index],
        &crate::index_writer::per_field_codec_suffix("Lucene90"),
    );
    let (meta, data) =
        crate::field_updates::read_current_column(dir, sci, &si.files, &infos, index, &per_field)
            .unwrap()
            .unwrap();
    let entry = meta.numeric_entry(infos.fields[index].number).unwrap();
    (0..si.doc_count)
        .map(|d| lucene_codecs::doc_values::numeric_value(&data, entry, d).unwrap())
        .collect()
}

/// Each segment's deletion count, buffered deletes written first.
fn live(w: &mut IndexWriter<'_>) -> Vec<i32> {
    w.write_pending_try_changes().unwrap();
    w.live_infos()
        .segments
        .iter()
        .map(|s| s.del_count)
        .collect()
}

#[test]
fn deletes_by_doc_id_of_a_reader_snapshot() {
    let tmp = TempDir::new("try-delete");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..3 {
        w.add_document(doc(i)).unwrap();
    }
    w.commit().unwrap();
    for i in 3..5 {
        w.add_document(doc(i)).unwrap();
    }
    let snap = w.nrt_snapshot(true, false).unwrap();
    // doc 4 is the second of the flushed (uncommitted) segment.
    let seq = w.try_delete_document(&snap.segment_infos, 4).unwrap();
    assert!(seq.is_some());
    assert_eq!(live(&mut w), [0, 1]);
    // Deleting it again is a no-op that still succeeds.
    assert!(w
        .try_delete_document(&snap.segment_infos, 4)
        .unwrap()
        .is_some());
    assert_eq!(live(&mut w), [0, 1]);
    w.try_delete_document(&snap.segment_infos, 1)
        .unwrap()
        .unwrap();
    assert_eq!(live(&mut w), [1, 1]);
    // The flushed segment emptied is dropped.
    w.try_delete_document(&snap.segment_infos, 3)
        .unwrap()
        .unwrap();
    assert_eq!(live(&mut w), [1]);
    w.commit().unwrap();
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
    // Out of bounds, and a segment merged away.
    assert!(w.try_delete_document(&snap.segment_infos, 9).is_err());
    assert!(w.try_delete_document(&snap.segment_infos, -1).is_err());
    w.add_document(doc(9)).unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    assert_eq!(w.try_delete_document(&snap.segment_infos, 0).unwrap(), None);
}

#[test]
fn buffered_deletes_become_one_generation_and_reach_the_next_reader() {
    let tmp = TempDir::new("try-delete-buffered");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..10 {
        w.add_document(doc(i)).unwrap();
    }
    w.commit().unwrap();
    let snap = w.nrt_snapshot(true, false).unwrap();
    let before = w.live_infos().segments[0].del_gen;
    for d in [1, 3, 5, 3] {
        w.try_delete_document(&snap.segment_infos, d)
            .unwrap()
            .unwrap();
    }
    // Nothing written yet, but the writer knows it has changes.
    assert_eq!(w.live_infos().segments[0].del_gen, before);
    assert!(w.has_uncommitted_changes());
    assert!(!w.nrt_is_current(&snap.segment_infos));
    // The next reader sees all three, from one new `.liv` generation.
    let next = w.nrt_snapshot(true, false).unwrap();
    let sci = &next.segment_infos.segments[0];
    assert_eq!(sci.del_count, 3);
    assert_eq!(sci.del_gen, before.max(0) + 1);
    // A delete by term on top of them keeps them.
    w.delete_documents_by_term(&[Term::new("id", "d7")])
        .unwrap();
    w.try_delete_document(&next.segment_infos, 8)
        .unwrap()
        .unwrap();
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments[0].del_count, 5);
    // A rollback drops a buffered delete with everything else uncommitted.
    let snap = w.nrt_snapshot(true, false).unwrap();
    w.try_delete_document(&snap.segment_infos, 0)
        .unwrap()
        .unwrap();
    w.rollback();
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments[0].del_count, 5);
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
}

#[test]
fn updates_doc_values_by_doc_id() {
    let tmp = TempDir::new("try-update");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..4 {
        w.add_document(doc(i)).unwrap();
    }
    w.commit().unwrap();
    let snap = w.nrt_snapshot(true, false).unwrap();
    let any = Term::new("", "");
    let seq = w
        .try_update_doc_value(
            &snap.segment_infos,
            2,
            &[
                DocValuesUpdate::Numeric {
                    term: any.clone(),
                    field: "n".into(),
                    value: Some(42),
                },
                DocValuesUpdate::Binary {
                    term: any.clone(),
                    field: "b".into(),
                    value: None,
                },
            ],
        )
        .unwrap();
    assert!(seq.is_some());
    // Buffered: no generation yet, but the writer has changes.
    assert_eq!(w.live_infos().segments[0].doc_values_gen, -1);
    assert!(w.has_uncommitted_changes());
    // A later update of the same document and field wins, and a second
    // document joins the same generation.
    for (doc, value) in [(2, 41), (2, 42), (3, 3)] {
        w.try_update_doc_value(
            &snap.segment_infos,
            doc,
            &[DocValuesUpdate::Numeric {
                term: any.clone(),
                field: "n".into(),
                value: Some(value),
            }],
        )
        .unwrap()
        .unwrap();
    }
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments[0].doc_values_gen, 2);
    for r in crate::check_index::check_directory(&dir).unwrap() {
        assert!(r.all_passed(), "{:?}", r.failures());
    }
    let sci = w.segment_infos().segments[0].clone();
    assert_eq!(
        numeric_values(&dir, &sci, "n"),
        [Some(0), Some(1), Some(42), Some(3)]
    );
    // An update by term on another document lands on top of it.
    w.update_numeric_doc_value(Term::new("id", "d1"), "n", 7)
        .unwrap();
    w.commit().unwrap();
    let sci = w.segment_infos().segments[0].clone();
    assert_eq!(
        numeric_values(&dir, &sci, "n"),
        [Some(0), Some(7), Some(42), Some(3)]
    );
    assert!(matches!(
        w.try_update_doc_value(&snap.segment_infos, 0, &[]),
        Err(Error::NoDocValuesUpdatesSupplied)
    ));
    assert!(w
        .try_update_doc_value(
            &snap.segment_infos,
            0,
            &[DocValuesUpdate::Numeric {
                term: any,
                field: "id".into(),
                value: Some(1)
            }]
        )
        .is_err());
}
