use crate::buffered_updates::Term;
use crate::index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB};
use crate::segment_info::LuceneVersion;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn writer(dir: &FsDirectory) -> IndexWriter<'_> {
    let fields = vec![FieldInfo {
        index_options: IndexOptions::Docs,
        omit_norms: true,
        ..FieldInfo::new("id", 0)
    }];
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
    w.set_max_buffered_docs(1000).unwrap();
    w.set_ram_buffer_size_mb(DISABLE_AUTO_FLUSH_MB).unwrap();
    w
}

fn doc(id: &str) -> Document {
    Document {
        fields: vec![StoredField {
            field_number: 0,
            value: FieldValue::String(id.to_string()),
        }],
    }
}

fn deleted(w: &IndexWriter<'_>) -> i32 {
    w.segment_infos().segments.iter().map(|s| s.del_count).sum()
}

/// A second round of deletes reuses the first round's open of the segment's
/// postings; a merge drops the retired segments from the pool, a rollback
/// everything.
#[test]
fn delete_rounds_reuse_a_segments_postings() {
    let tmp = TempDir::new("reader-pool");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    for i in 0..20 {
        w.add_document(doc(&format!("d{i}"))).unwrap();
    }
    w.commit().unwrap();
    assert!(w.reader_pool().is_empty());
    for round in 0..3 {
        w.delete_documents_by_term(&[Term::new("id", format!("d{round}"))])
            .unwrap();
        w.commit().unwrap();
    }
    assert_eq!(deleted(&w), 3, "every round still deletes");
    assert_eq!(w.reader_pool().len(), 1);
    assert_eq!(w.reader_pool().opens(), 1, "one open for three rounds");
    assert!(format!("{:?}", w.reader_pool()).contains("segments: 1"));

    // A second segment, then a merge of both: the pooled ones are retired.
    w.add_document(doc("x")).unwrap();
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "x")]).unwrap();
    w.commit().unwrap();
    // `x`'s segment was left fully deleted and dropped, and with it its
    // pooled postings.
    assert_eq!(w.reader_pool().segment_names(), ["_0"]);
    w.add_document(doc("y")).unwrap();
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "d5")])
        .unwrap();
    w.commit().unwrap();
    assert_eq!(w.reader_pool().len(), 2);
    w.force_merge(1).unwrap();
    assert!(
        w.reader_pool().is_empty(),
        "{:?}",
        w.reader_pool().segment_names()
    );

    w.delete_documents_by_term(&[Term::new("id", "d10")])
        .unwrap();
    w.commit().unwrap();
    assert_eq!(w.reader_pool().len(), 1);
    w.delete_documents_by_term(&[Term::new("id", "d11")])
        .unwrap();
    w.flush().unwrap();
    w.rollback();
    assert!(w.reader_pool().is_empty());
}

/// `setReaderPooling(false)`: every round opens the segment again and nothing
/// is kept -- the behaviour before the pool existed.
#[test]
fn with_pooling_off_every_round_opens_the_segment() {
    let tmp = TempDir::new("reader-pool-off");
    let dir = FsDirectory::open(&tmp);
    let mut w = writer(&dir);
    w.set_reader_pooling(false);
    assert!(!w.reader_pool().is_enabled());
    for i in 0..5 {
        w.add_document(doc(&format!("d{i}"))).unwrap();
    }
    w.commit().unwrap();
    for round in 0..3 {
        w.delete_documents_by_term(&[Term::new("id", format!("d{round}"))])
            .unwrap();
        w.commit().unwrap();
    }
    assert_eq!(deleted(&w), 3);
    assert_eq!(w.reader_pool().opens(), 3);
    assert!(w.reader_pool().is_empty());
    w.set_reader_pooling(true);
    w.delete_documents_by_term(&[Term::new("id", "d4")])
        .unwrap();
    w.flush().unwrap();
    assert_eq!(w.reader_pool().len(), 1);
    w.delete_all().unwrap();
    assert!(w.reader_pool().is_empty());
}
