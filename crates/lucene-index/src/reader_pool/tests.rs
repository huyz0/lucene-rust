use crate::buffered_updates::Term;
use crate::index_writer::{IndexWriter, DISABLE_AUTO_FLUSH_MB};
use crate::segment_info::LuceneVersion;
use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_store::directory::{Directory, Input, TrackedInput};
use lucene_store::FsDirectory;
use lucene_store::FsIndexOutput;
use lucene_util::test_support::TempDir;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

fn writer(dir: &dyn Directory) -> IndexWriter<'_> {
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

/// Which files are open, and which were deleted while open:
/// `MockDirectoryWrapper.getOpenDeletedFiles`, reduced to the bookkeeping.
#[derive(Default)]
struct OpenFiles {
    open: BTreeMap<String, usize>,
    deleted_while_open: BTreeSet<String>,
}

struct OpenTrackingDirectory {
    inner: FsDirectory,
    files: Arc<Mutex<OpenFiles>>,
}

/// Closes its file in [`OpenFiles`] when the input it guards drops.
struct Close {
    files: Arc<Mutex<OpenFiles>>,
    name: String,
}

impl Drop for Close {
    fn drop(&mut self) {
        let mut files = self.files.lock().unwrap();
        if let Some(n) = files.open.get_mut(&self.name) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                files.open.remove(&self.name);
            }
        }
    }
}

impl Directory for OpenTrackingDirectory {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<Input> {
        let input = self.inner.open(name)?;
        let mut files = self.files.lock().unwrap();
        let n = files.open.entry(name.to_string()).or_insert(0);
        *n = n.saturating_add(1);
        drop(files);
        let close = Close {
            files: Arc::clone(&self.files),
            name: name.to_string(),
        };
        Ok(Input::Tracked(TrackedInput::new(input, close)))
    }
    fn file_length(&self, name: &str) -> lucene_store::Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        let mut files = self.files.lock().unwrap();
        if files.open.contains_key(name) {
            files.deleted_while_open.insert(name.to_string());
        }
        drop(files);
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

/// A merge closes its sources' pooled postings before the deleter reclaims
/// their files, as Java's `commitMerge` closes the merge readers before
/// `checkpoint()`: the pool holds each source's `.doc` open, and deleting it
/// first is what `threaded_stress` reported as a non-zero open-but-deleted
/// count. A segment its own deletes leave empty is the same.
#[test]
fn retired_segments_close_their_pooled_postings_before_deletion() {
    let tmp = TempDir::new("reader-pool-open-deleted");
    let files = Arc::new(Mutex::new(OpenFiles::default()));
    let dir = OpenTrackingDirectory {
        inner: FsDirectory::open(&tmp),
        files: Arc::clone(&files),
    };
    let mut w = writer(&dir);
    for seg in 0..3 {
        for i in 0..5 {
            w.add_document(doc(&format!("s{seg}d{i}"))).unwrap();
        }
        w.commit().unwrap();
        w.delete_documents_by_term(&[Term::new("id", format!("s{seg}d0"))])
            .unwrap();
        w.commit().unwrap();
    }
    assert_eq!(w.reader_pool().len(), 3, "every source is pooled");
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    assert!(w.reader_pool().is_empty());

    // A fully deleted segment: dropped, and its files reclaimed.
    w.delete_documents_by_term(&[Term::new("id", "s0d1")])
        .unwrap();
    w.commit().unwrap();
    w.add_document(doc("gone")).unwrap();
    w.commit().unwrap();
    w.delete_documents_by_term(&[Term::new("id", "gone")])
        .unwrap();
    w.commit().unwrap();
    assert_eq!(w.segment_infos().segments.len(), 1);

    let files = files.lock().unwrap();
    assert!(
        files.deleted_while_open.is_empty(),
        "deleted while open: {:?}",
        files.deleted_while_open
    );
}
