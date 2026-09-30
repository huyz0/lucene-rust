use super::*;
use crate::buffered_updates::Term;
use crate::index_writer::DISABLE_AUTO_FLUSH_MB;
use crate::segment_info::{IndexSortField, LuceneVersion};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn fields(rank_type: DocValuesType) -> Vec<FieldInfo> {
    vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("rank", 1)
            .with_omit_norms(true)
            .with_doc_values(rank_type, DocValuesSkipIndexType::None, -1),
    ]
}

fn writer(dir: &FsDirectory, rank_type: DocValuesType) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(
        dir,
        fields(rank_type),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.set_postings_field(Some("id")).unwrap();
    w.set_doc_values_field(Some("rank")).unwrap();
    w.set_max_buffered_docs(1000).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w
}

fn doc(id: &str, rank: i64) -> Document {
    Document {
        fields: vec![
            StoredField {
                field_number: 0,
                value: FieldValue::String(id.to_string()),
            },
            StoredField {
                field_number: 1,
                value: FieldValue::Long(rank),
            },
        ],
    }
}

/// Every live document's id, over the latest commit.
fn committed_ids(dir: &FsDirectory) -> Vec<String> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let mut out = Vec::new();
    for sci in &infos.segments {
        let name = &sci.segment_name;
        let si = segment_info::parse(&dir.open(&format!("{name}.si")).unwrap(), &sci.segment_id)
            .unwrap();
        let read = |ext: &str| dir.open(&format!("{name}.{ext}")).unwrap();
        let (fdt, fdx, fdm) = (read("fdt"), read("fdx"), read("fdm"));
        let reader =
            lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &sci.segment_id, "").unwrap();
        let live = (sci.del_gen >= 0).then(|| {
            lucene_codecs::live_docs::parse(
                &dir.open(&crate::deletes::liv_file_name(name, sci.del_gen))
                    .unwrap(),
                &sci.segment_id,
                sci.del_gen,
                si.doc_count as usize,
                sci.del_count as usize,
            )
            .unwrap()
        });
        for d in 0..si.doc_count {
            if live.as_ref().is_some_and(|l| !l.get(d as usize)) {
                continue;
            }
            if let FieldValue::String(s) = &reader.document(d).unwrap().fields[0].value {
                out.push(s.clone());
            }
        }
    }
    out.sort();
    out
}

fn assert_clean(dir: &FsDirectory) {
    for r in crate::check_index::check_directory(dir).unwrap() {
        assert!(r.all_passed(), "{}: {:?}", r.segment_name, r.failures());
    }
}

/// A source index with two segments, one of which has a deletion.
fn source(tmp: &TempDir) -> FsDirectory {
    let dir = FsDirectory::open(tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    for (id, rank) in [("s0", 5), ("s1", 1), ("s2", 3)] {
        w.add_document(doc(id, rank)).unwrap();
    }
    w.commit().unwrap();
    for (id, rank) in [("s3", 2), ("s4", 4)] {
        w.add_document(doc(id, rank)).unwrap();
    }
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "s1")])
        .unwrap();
    w.commit().unwrap();
    drop(w);
    dir
}

/// `addIndexes(Directory...)`: the source's segments arrive under new
/// names, deletions and all; nothing is durable until the commit; the
/// result passes `CheckIndex`, and a delete by term reaches the added
/// documents.
#[test]
fn add_indexes_copies_segments_as_they_are() {
    let src_tmp = TempDir::new("add-indexes-src");
    let src = source(&src_tmp);
    let tmp = TempDir::new("add-indexes-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    w.add_document(doc("own", 0)).unwrap();
    w.commit().unwrap();
    w.add_indexes(&[&src]).unwrap();
    assert_eq!(
        committed_ids(&dir),
        ["own"],
        "not durable before the commit"
    );
    let names: Vec<&str> = w
        .segment_infos()
        .segments
        .iter()
        .map(|s| s.segment_name.as_str())
        .collect();
    assert_eq!(names, ["_0", "_1", "_2"]);
    assert_eq!(w.segment_infos().segments[1].del_count, 1);
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s2", "s3", "s4"]);
    assert_clean(&dir);
    w.delete_documents_by_term(&[Term::new("id", "s4")])
        .unwrap();
    w.commit().unwrap();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s2", "s3"]);
    // The source is untouched and can be added again.
    w.add_indexes(&[&src]).unwrap();
    w.rollback();
    assert_eq!(committed_ids(&dir), ["own", "s0", "s2", "s3"]);
}

/// `addIndexes(CodecReader...)`: one new segment of the sources' live
/// documents, in the index sort.
#[test]
fn add_indexes_merged_makes_one_segment() {
    let src_tmp = TempDir::new("add-merged-src");
    let src = source(&src_tmp);
    let src2_tmp = TempDir::new("add-merged-src2");
    let src2 = source(&src2_tmp);
    let tmp = TempDir::new("add-merged-dst");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
        .unwrap();
    w.add_indexes_merged(&[]).unwrap();
    // One directory named twice is locked by its first mention, as in Java.
    assert!(w.add_indexes_merged(&[&src, &src]).is_err());
    w.add_indexes_merged(&[&src, &src2]).unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);
    w.commit().unwrap();
    let ids = committed_ids(&dir);
    assert_eq!(ids.len(), 8);
    assert_eq!(
        w.segment_infos().segments[0].del_count,
        0,
        "deletions dropped"
    );
    assert_clean(&dir);
}

/// What `addIndexes` refuses: a source whose sort is not congruent with the
/// writer's, and a field whose schema conflicts; either way nothing is
/// added.
#[test]
fn add_indexes_refuses_incompatible_sources() {
    let src_tmp = TempDir::new("add-bad-src");
    let src = source(&src_tmp);

    let tmp = TempDir::new("add-bad-sort");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::Numeric);
    w.set_index_sort(Some(&[IndexSortField::long("rank", true, None)]))
        .unwrap();
    assert!(matches!(
        w.add_indexes(&[&src]),
        Err(Error::IncongruentIndexSort { .. })
    ));
    assert!(w.segment_infos().segments.is_empty());

    let tmp = TempDir::new("add-bad-schema");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir, DocValuesType::SortedNumeric);
    let err = w.add_indexes(&[&src]).unwrap_err();
    assert!(
        matches!(&err, Error::AddIndexes(m) if m.contains("\"rank\"") && m.contains("doc values type")),
        "{err}"
    );
    assert!(w.segment_infos().segments.is_empty());

    // A source with a writer still open on it is locked.
    let tmp = TempDir::new("add-locked");
    let busy = FsDirectory::open(&tmp);
    let _open = writer(&busy, DocValuesType::Numeric);
    let tmp2 = TempDir::new("add-locked-dst");
    let dir2 = FsDirectory::open(&tmp2);
    let mut w = writer(&dir2, DocValuesType::Numeric);
    assert!(w.add_indexes(&[&busy]).is_err());
    assert_eq!(rename("_3.fdt", "_3", "_9"), "_9.fdt");
    assert_eq!(rename("_3_1.liv", "_3", "_9"), "_9_1.liv");
    assert_eq!(rename("_30.fdt", "_3", "_9"), "_30.fdt");
}
