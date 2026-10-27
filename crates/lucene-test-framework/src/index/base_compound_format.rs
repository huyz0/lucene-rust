//! `BaseCompoundFormatTestCase`: the `.cfs`/`.cfe` pair
//! (`Lucene90CompoundFormat`), written by [`write_compound`] over
//! `lucene_codecs::compound_format::write` and read back through
//! `lucene_index::compound_reader::CompoundReader`.
//!
//! Differs:
//!
//! - `cfs.listAll()` is [`CompoundReader::member_files`]: the port's
//!   `list_all` also lists the directory underneath, as its `open` falls
//!   through to it for files beside the archive.
//! - A sub-file without a codec header is refused as "not a valid codec
//!   file (bad magic)", not Java's "codec header mismatch"; the wrong
//!   checksum carries Java's "checksum failed (hardware problem?)".
//! - `testCheckIntegrity` cannot count the bytes read (a directory hands
//!   out a whole file): it checks that [`CompoundReader::check_integrity`]
//!   passes, and that it fails on a `.cfs` byte flipped where opening does
//!   not look.
//! - `testLargeCFS` (512 MiB through an `NRTCachingDirectory`) is
//!   [`large_cfs`] with the size as a parameter: a port file is held in
//!   memory whole, three times over while it is packed.
//! - `testListAll`'s fields are plain `StringField`/`TextField`, not
//!   `newStringField`/`newTextField`'s, so no term vectors are packed.
//! - `testDoubleClose`: a reader closes by being dropped, once.
//!   `testPassIOContext`: the context is the size estimate a directory
//!   sees (`None` is `DEFAULT`). `testResourceNameInsideCompoundFile`: an
//!   input has no resource description, so the member's bytes and the
//!   reader's errors are checked. `testMergeStability` is a no-op in Java's
//!   suite too.

use lucene_codecs::compound_format;
use lucene_index::compound_reader::CompoundReader;
use lucene_index::document::{Document, Field, Store, StoredValue, StringField, TextField};
use lucene_index::segment_info::{self, SegmentInfo};
use lucene_index::segment_infos;
use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::{
    DataInput as _, DataOutput as _, Directory, FsDirectory, NrtCachingDirectory, SliceInput,
};

use super::base_index_file_format::{
    new_riw, write_file, IndexFileFormatSuite, SuiteTest, VERSION,
};
use crate::store::base_directory_wrapper::BaseDirectoryWrapper as _;
use crate::util::lucene_test_case::LuceneTestCase;
use crate::util::test_util;
use lucene_util::test_support::TempDir;

/// The compound suite's `addRandomFields`: a stored `foobar`.
pub struct CompoundSuite;

impl IndexFileFormatSuite for CompoundSuite {
    fn add_random_fields(&self, t: &mut LuceneTestCase, doc: &mut Document) {
        let value = test_util::random_simple_string(t.random());
        doc.add(Field::stored("foobar", StoredValue::String(value)));
    }
}

/// `newSegmentInfo(dir, name)`: 10000 documents, `minVersion` sometimes
/// unset, a fresh id (`StringHelper.randomId()`'s own stream).
pub fn new_segment_info(t: &mut LuceneTestCase, _name: &str) -> SegmentInfo {
    let min_version = if t.random().next_boolean() {
        None
    } else {
        Some(VERSION)
    };
    SegmentInfo {
        id: lucene_util::string_helper::random_id(),
        version: VERSION,
        min_version,
        doc_count: 10_000,
        is_compound_file: false,
        has_blocks: false,
        diagnostics: Vec::new(),
        files: Vec::new(),
        attributes: Vec::new(),
        index_sort: None,
    }
}

/// `compoundFormat().write(dir, si, context)`: packs `si.files` into
/// `<name>.cfs`/`<name>.cfe`, leaving the files themselves in place.
///
/// # Errors
/// A sub-file the format refuses (no codec header, a wrong checksum, another
/// segment's id) or the directory's refusal, as text.
pub fn write_compound(dir: &dyn Directory, name: &str, si: &SegmentInfo) -> Result<(), String> {
    let mut sub_files = Vec::with_capacity(si.files.len());
    for file in &si.files {
        let bytes = dir.open(file).map_err(|e| e.to_string())?.to_vec();
        let id = file.strip_prefix(name).unwrap_or(file).to_string();
        sub_files.push((id, bytes));
    }
    let (cfs, cfe) = compound_format::write(&si.id, &sub_files).map_err(|e| e.to_string())?;
    write_file(dir, &format!("{name}.cfs"), &cfs).map_err(|e| e.to_string())?;
    write_file(dir, &format!("{name}.cfe"), &cfe).map_err(|e| e.to_string())
}

/// `compoundFormat().getCompoundReader(dir, si)`.
///
/// # Panics
/// When the pair does not open.
pub fn compound_reader<'d>(
    dir: &'d dyn Directory,
    name: &str,
    si: &SegmentInfo,
) -> CompoundReader<'d> {
    CompoundReader::open(dir, name, &si.id).expect("getCompoundReader")
}

fn write_and_open<'d>(dir: &'d dyn Directory, name: &str, si: &SegmentInfo) -> CompoundReader<'d> {
    write_compound(dir, name, si).expect("compoundFormat().write");
    compound_reader(dir, name, si)
}

/// `createSequenceFile(dir, name, start, size, segID, segSuffix)`: a codec
/// file whose body counts up from `start`.
///
/// # Panics
/// When the directory refuses the file.
pub fn create_sequence_file(
    dir: &dyn Directory,
    name: &str,
    start: u8,
    size: usize,
    id: &[u8; ID_LENGTH],
    suffix: &str,
) {
    let mut bytes = Vec::new();
    codec_util::write_index_header(&mut bytes, "Foo", 0, id, suffix);
    let mut b = start;
    for _ in 0..size {
        bytes.push(b);
        b = b.wrapping_add(1);
    }
    codec_util::write_footer(&mut bytes);
    write_file(dir, name, &bytes).expect("createSequenceFile");
}

/// `createRandomFile(dir, name, size, segId)`.
///
/// # Panics
/// When the directory refuses the file.
pub fn create_random_file(
    t: &mut LuceneTestCase,
    dir: &dyn Directory,
    name: &str,
    size: usize,
    id: &[u8; ID_LENGTH],
) {
    let mut bytes = Vec::new();
    codec_util::write_index_header(&mut bytes, "Foo", 0, id, "suffix");
    for _ in 0..size {
        bytes.push(t.random().next_int_bounded(256) as u8);
    }
    codec_util::write_footer(&mut bytes);
    write_file(dir, name, &bytes).expect("createRandomFile");
}

/// `assertSameStreams(msg, expected, test)`: the same length, position and
/// remaining bytes, read 512 at a time.
pub fn assert_same_streams(msg: &str, expected: &mut SliceInput, test: &mut SliceInput) {
    assert_eq!(expected.len(), test.len(), "{msg} length");
    assert_eq!(expected.position(), test.position(), "{msg} position");
    let mut e = [0u8; 512];
    let mut a = [0u8; 512];
    let mut remainder = expected.len() - expected.position();
    while remainder > 0 {
        let n = remainder.min(e.len());
        expected.read_bytes(&mut e[..n]).expect("read expected");
        test.read_bytes(&mut a[..n]).expect("read test");
        assert_eq!(e[..n], a[..n], "{msg}, remainder {remainder}");
        remainder -= n;
    }
}

fn assert_same_streams_at(msg: &str, expected: &mut SliceInput, test: &mut SliceInput, at: i64) {
    if let Ok(at) = usize::try_from(at) {
        if at < expected.len() {
            expected.seek(at).expect("seek expected");
            test.seek(at).expect("seek test");
            assert_same_streams(&format!("{msg}, seek(mid)"), expected, test);
        }
    }
}

/// `assertSameSeekBehavior(msg, expected, actual)`.
pub fn assert_same_seek_behavior(msg: &str, expected: &mut SliceInput, test: &mut SliceInput) {
    let len = expected.len() as i64;
    for (what, at) in [
        ("seek(0)", 0),
        ("seek(mid)", len / 2),
        ("seek(end-2)", len - 2),
        ("seek(end-1)", len - 1),
        ("seek(end)", len),
        ("seek(end+1)", len + 1),
    ] {
        assert_same_streams_at(&format!("{msg}, {what}"), expected, test, at);
    }
}

fn assert_same_file(dir: &dyn Directory, cfs: &CompoundReader<'_>, file: &str) {
    let expected = dir.open(file).expect("openInput");
    let actual = cfs.open(file).expect("cfs.openInput");
    let (mut e, mut a) = (SliceInput::new(&expected), SliceInput::new(&actual));
    assert_same_streams(file, &mut e, &mut a);
    assert_same_seek_behavior(file, &mut e, &mut a);
}

/// `createLargeCFS(dir)`: `_123.f0` .. `_123.f19`, 2000 counting bytes each.
fn create_large_cfs<'d>(t: &mut LuceneTestCase, dir: &'d dyn Directory) -> CompoundReader<'d> {
    let mut si = new_segment_info(t, "_123");
    for i in 0..20 {
        let file = format!("_123.f{i}");
        create_sequence_file(dir, &file, 0, 2000, &si.id, "suffix");
        si.files.push(file);
    }
    write_and_open(dir, "_123", &si)
}

/// `testEmpty`.
pub fn test_empty(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let si = new_segment_info(t, "_123");
    let cfs = write_and_open(&*dir, "_123", &si);
    assert!(cfs.member_files().is_empty());
    drop(cfs);
    dir.close().expect("close");
}

/// `testSingleFile`: sub-files of 0, 1, 10 and 100 bytes.
pub fn test_single_file(t: &mut LuceneTestCase) {
    for (i, size) in [0, 1, 10, 100].into_iter().enumerate() {
        let file = format!("_{i}.test");
        let dir = t.new_directory();
        let mut si = new_segment_info(t, &format!("_{i}"));
        create_sequence_file(&*dir, &file, 0, size, &si.id, "suffix");
        si.files = vec![file.clone()];
        let cfs = write_and_open(&*dir, &format!("_{i}"), &si);
        assert_same_file(&*dir, &cfs, &file);
        drop(cfs);
        dir.close().expect("close");
    }
}

/// `testTwoFiles`.
pub fn test_two_files(t: &mut LuceneTestCase) {
    let files = ["_123.d1", "_123.d2"];
    let dir = t.new_directory();
    let mut si = new_segment_info(t, "_123");
    create_sequence_file(&*dir, files[0], 0, 15, &si.id, "suffix");
    create_sequence_file(&*dir, files[1], 0, 114, &si.id, "suffix");
    si.files = files.iter().map(|f| f.to_string()).collect();
    let cfs = write_and_open(&*dir, "_123", &si);
    for file in files {
        assert_same_file(&*dir, &cfs, file);
    }
    drop(cfs);
    dir.close().expect("close");
}

/// `testLargeCFS` with `chunks` 512-byte chunks (Java writes 1024 * 1024)
/// through an `NRTCachingDirectory(fsDir, 2.0, 25.0)`.
pub fn large_cfs(t: &mut LuceneTestCase, chunks: usize) {
    let tmp = TempDir::new("large-cfs");
    let dir = NrtCachingDirectory::new(FsDirectory::open(tmp.path()), 2.0, 25.0);
    let mut si = new_segment_info(t, "_123");
    let mut bytes = Vec::with_capacity(chunks * 512 + 64);
    codec_util::write_index_header(&mut bytes, "Foo", 0, &si.id, "suffix");
    bytes.resize(bytes.len() + chunks * 512, 0);
    codec_util::write_footer(&mut bytes);
    write_file(&dir, "_123.test", &bytes).expect("write");
    si.files = vec!["_123.test".to_string()];
    write_compound(&dir, "_123", &si).expect("compoundFormat().write");
    let cfs = compound_reader(&dir, "_123", &si);
    assert_eq!(cfs.open("_123.test").expect("open").len(), bytes.len());
}

/// `testLargeCFS`, at 4 MiB.
pub fn test_large_cfs(t: &mut LuceneTestCase) {
    large_cfs(t, 8 * 1024);
}

/// `testListAll`: every file a compound segment lists opens. Returns the
/// number of compound segments seen, so a caller can tell the test was
/// not vacuous.
pub fn list_all(t: &mut LuceneTestCase) -> usize {
    let dir = t.new_directory();
    let mut riw = new_riw(t, &*dir);
    for i in 0..100 {
        let body = test_util::random_unicode_string(t.random());
        let mut doc = Document::new();
        doc.add(StringField::new("id", i.to_string(), Store::No));
        doc.add(TextField::new("body", body, Store::No));
        riw.add_document(&doc).expect("addDocument");
        if t.random().next_int_bounded(7) == 0 {
            riw.commit().expect("commit");
        }
    }
    riw.close().expect("close");
    let infos = segment_infos::read_latest(&*dir).expect("readLatestCommit");
    let mut compound = 0;
    for sci in &infos.segments {
        let si_bytes = dir
            .open(&format!("{}.si", sci.segment_name))
            .expect("open .si");
        let si = segment_info::parse(&si_bytes, &sci.segment_id).expect("read .si");
        if si.is_compound_file {
            compound += 1;
            let cfs = CompoundReader::open(&*dir, &sci.segment_name, &sci.segment_id)
                .expect("getCompoundReader");
            for file in cfs.member_files() {
                cfs.open(&file).expect("openInput");
            }
        }
    }
    dir.close().expect("close");
    compound
}

/// `testListAll`.
pub fn test_list_all(t: &mut LuceneTestCase) {
    list_all(t);
}

/// The read-only tests: `op` on an empty compound reader over a directory
/// holding `_123.test` is `UnsupportedOperationException`.
fn assert_disabled(
    t: &mut LuceneTestCase,
    op: fn(&CompoundReader<'_>) -> lucene_store::Result<()>,
) {
    let dir = t.new_directory();
    write_file(&*dir, "_123.test", &3i32.to_be_bytes()).expect("write");
    let si = new_segment_info(t, "_123");
    let cfs = write_and_open(&*dir, "_123", &si);
    match op(&cfs) {
        Err(lucene_store::Error::Io(e)) => {
            assert_eq!(e.kind(), std::io::ErrorKind::Unsupported, "{e}");
        }
        other => panic!("expected UnsupportedOperationException, got {other:?}"),
    }
    drop(cfs);
    dir.close().expect("close");
}

/// `testCreateOutputDisabled`.
pub fn test_create_output_disabled(t: &mut LuceneTestCase) {
    assert_disabled(t, |cfs| cfs.create_output("bogus").map(|_| ()));
}

/// `testDeleteFileDisabled`.
pub fn test_delete_file_disabled(t: &mut LuceneTestCase) {
    assert_disabled(t, |cfs| cfs.delete_file("_123.test"));
}

/// `testRenameFileDisabled`.
pub fn test_rename_file_disabled(t: &mut LuceneTestCase) {
    assert_disabled(t, |cfs| cfs.rename("_123.test", "bogus"));
}

/// `testSyncDisabled`.
pub fn test_sync_disabled(t: &mut LuceneTestCase) {
    assert_disabled(t, |cfs| cfs.sync(&["_123.test".to_string()]));
}

/// `testMakeLockDisabled`.
pub fn test_make_lock_disabled(t: &mut LuceneTestCase) {
    assert_disabled(t, |cfs| cfs.obtain_lock("foobar").map(|_| ()));
}

/// `testRandomFiles`: sizes about the 1024-byte chunk, up to 1000 chunks.
pub fn test_random_files(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let segment = "_123";
    let chunk = 1024;
    let mut si = new_segment_info(t, segment);
    let id = si.id;
    for (ext, size) in [
        ("zero", 0),
        ("one", 1),
        ("ten", 10),
        ("hundred", 100),
        ("big1", chunk),
        ("big2", chunk - 1),
        ("big3", chunk + 1),
        ("big4", 3 * chunk),
        ("big5", 3 * chunk - 1),
        ("big6", 3 * chunk + 1),
        ("big7", 1000 * chunk),
    ] {
        create_random_file(t, &*dir, &format!("{segment}.{ext}"), size, &id);
    }
    si.files = dir
        .list_all()
        .expect("listAll")
        .into_iter()
        .filter(|f| f.starts_with(segment))
        .collect();
    assert_eq!(si.files.len(), 11);
    let cfs = write_and_open(&*dir, segment, &si);
    for file in &si.files {
        assert_same_file(&*dir, &cfs, file);
    }
    drop(cfs);
    dir.close().expect("close");
}

/// `testManySubFiles`: however many members are open, the reader holds one
/// file handle.
pub fn test_many_sub_files(t: &mut LuceneTestCase) {
    let tmp = TempDir::new("CFSManySubFiles");
    let dir = t.new_mock_fs_directory(tmp.path());
    let file_count = t.at_least(500) as usize;
    let mut si = new_segment_info(t, "_123");
    for i in 0..file_count {
        let file = format!("_123.{i}");
        let mut bytes = Vec::new();
        codec_util::write_index_header(&mut bytes, "Foo", 0, &si.id, "suffix");
        bytes.push(i as u8);
        codec_util::write_footer(&mut bytes);
        write_file(&dir, &file, &bytes).expect("write");
        si.files.push(file);
    }
    assert_eq!(dir.file_handle_count(), 0);
    let cfs = write_and_open(&dir, "_123", &si);
    let ins: Vec<_> = (0..file_count)
        .map(|i| cfs.open(&format!("_123.{i}")).expect("openInput"))
        .collect();
    let mut cursors: Vec<SliceInput> = ins.iter().map(|b| SliceInput::new(b)).collect();
    for c in &mut cursors {
        codec_util::check_index_header(c, "Foo", 0, 0, &si.id, "suffix").expect("header");
    }
    assert_eq!(dir.file_handle_count(), 1);
    for (i, c) in cursors.iter_mut().enumerate() {
        assert_eq!(c.read_byte().expect("readByte"), i as u8);
    }
    assert_eq!(dir.file_handle_count(), 1);
    drop(cursors);
    drop(ins);
    drop(cfs);
    assert_eq!(dir.file_handle_count(), 0);
    dir.close().expect("close");
}

/// `testClonedStreamsClosing`.
pub fn test_cloned_streams_closing(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let cr = create_large_cfs(t, &*dir);
    let expected_bytes = dir.open("_123.f11").expect("openInput");
    let one_bytes = cr.open("_123.f11").expect("cfs.openInput");
    let mut expected = SliceInput::new(&expected_bytes);
    let mut one = SliceInput::new(&one_bytes);
    let mut two = one.clone();
    assert_same_streams("basic clone one", &mut expected, &mut one);
    expected.seek(0).expect("seek");
    assert_same_streams("basic clone two", &mut expected, &mut two);
    drop((expected_bytes, one_bytes));
    drop(cr);
    dir.close().expect("close");
}

/// `testRandomAccess`'s and `testRandomAccessClones`' walk: two pairs of
/// cursors, each pair seeked and read in step, neither pair moving the
/// other.
fn random_access(
    e1: &mut SliceInput,
    a1: &mut SliceInput,
    e2: &mut SliceInput,
    a2: &mut SliceInput,
) {
    fn step(e: &mut SliceInput, a: &mut SliceInput, seek: Option<usize>, at: usize) {
        if let Some(to) = seek {
            e.seek(to).expect("seek");
            a.seek(to).expect("seek");
        }
        assert_eq!(e.position(), at);
        assert_eq!(a.position(), at);
        assert_eq!(
            e.read_byte().expect("readByte"),
            a.read_byte().expect("readByte")
        );
    }
    step(e1, a1, Some(100), 100);
    step(e2, a2, Some(1027), 1027);
    step(e1, a1, None, 101);
    step(e1, a1, Some(1910), 1910);
    step(e2, a2, None, 1028);
    step(e2, a2, Some(17), 17);
    step(e1, a1, None, 1911);
}

/// `testRandomAccess`. (Java's `a2` opens `_123.f3` from the directory, not
/// the compound reader; so does this.)
pub fn test_random_access(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let cr = create_large_cfs(t, &*dir);
    let e1b = dir.open("_123.f11").expect("openInput");
    let e2b = dir.open("_123.f3").expect("openInput");
    let a1b = cr.open("_123.f11").expect("cfs.openInput");
    let a2b = dir.open("_123.f3").expect("openInput");
    random_access(
        &mut SliceInput::new(&e1b),
        &mut SliceInput::new(&a1b),
        &mut SliceInput::new(&e2b),
        &mut SliceInput::new(&a2b),
    );
    drop((e1b, e2b, a1b, a2b));
    drop(cr);
    dir.close().expect("close");
}

/// `testRandomAccessClones`.
pub fn test_random_access_clones(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let cr = create_large_cfs(t, &*dir);
    let e1b = cr.open("_123.f11").expect("cfs.openInput");
    let e2b = cr.open("_123.f3").expect("cfs.openInput");
    let mut e1 = SliceInput::new(&e1b);
    let mut e2 = SliceInput::new(&e2b);
    let mut a1 = e1.clone();
    let mut a2 = e2.clone();
    random_access(&mut e1, &mut a1, &mut e2, &mut a2);
    drop(cr);
    dir.close().expect("close");
}

/// `testDoubleClose`: one member; the reader closes by being dropped,
/// which happens once by construction (Java's second `close` has no
/// counterpart), and the directory under it closes after.
pub fn test_double_close(t: &mut LuceneTestCase) {
    let testfile = "_123.test";
    let dir = t.new_directory();
    let mut si = new_segment_info(t, "_123");
    let mut bytes = Vec::new();
    codec_util::write_index_header(&mut bytes, "Foo", 0, &si.id, "suffix");
    bytes.extend_from_slice(&3i32.to_be_bytes());
    codec_util::write_footer(&mut bytes);
    write_file(&*dir, testfile, &bytes).expect("createOutput");
    si.files = vec![testfile.to_string()];
    let cfs = write_and_open(&*dir, "_123", &si);
    assert_eq!(1, cfs.member_files().len());
    drop(cfs);
    dir.close().expect("close");
}

/// A `FilterDirectory` recording the context of every output it creates:
/// the size estimate, the part of an `IOContext` the port carries (`None`
/// is `IOContext.DEFAULT`).
struct ContextRecording<'d> {
    inner: &'d dyn Directory,
    contexts: std::sync::Mutex<Vec<Option<u64>>>,
}

impl Directory for ContextRecording<'_> {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.create_output_with_estimate(name, None)
    }
    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.contexts
            .lock()
            .expect("contexts")
            .push(estimated_bytes);
        self.inner
            .create_output_with_estimate(name, estimated_bytes)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> lucene_store::Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

/// `testPassIOContext` (LUCENE-5724): every output the compound write
/// creates carries the caller's context, `IOContext.DEFAULT` (no estimate).
pub fn test_pass_io_context(t: &mut LuceneTestCase) {
    let testfile = "_123.test";
    let base = t.new_directory();
    let dir = ContextRecording {
        inner: &*base,
        contexts: std::sync::Mutex::new(Vec::new()),
    };
    let mut si = new_segment_info(t, "_123");
    create_sequence_file(&dir, testfile, 0, 4, &si.id, "suffix");
    si.files = vec![testfile.to_string()];
    write_compound(&dir, "_123", &si).expect("compoundFormat().write");
    let contexts = dir.contexts.into_inner().expect("contexts");
    assert_eq!(3, contexts.len(), "the test file, .cfs and .cfe");
    assert!(contexts.iter().all(Option::is_none), "{contexts:?}");
    base.close().expect("close");
}

/// `testResourceNameInsideCompoundFile`: an `Input` carries no resource
/// description to name its slice, so what a caller sees of the member is
/// checked instead -- its bytes, and errors naming the member and the
/// compound segment.
pub fn test_resource_name_inside_compound_file(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let sub_file = "_123.xyz";
    let mut si = new_segment_info(t, "_123");
    create_sequence_file(&*dir, sub_file, 0, 10, &si.id, "suffix");
    si.files = vec![sub_file.to_string()];
    let cfs = write_and_open(&*dir, "_123", &si);
    let input = cfs.open(sub_file).expect("openInput");
    assert_eq!(&dir.open(sub_file).expect("the sub-file")[..], &input[..]);
    let err = cfs.open("_123.abc").expect_err("not a member").to_string();
    assert!(
        err.contains("_123.abc") && err.contains("compound segment _123"),
        "the error hides that it is inside a compound file: {err}"
    );
    drop(cfs);
    dir.close().expect("close");
}

/// `testFileNotFound`.
pub fn test_file_not_found(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let cr = create_large_cfs(t, &*dir);
    assert!(cr.open("bogus").is_err());
    drop(cr);
    dir.close().expect("close");
}

/// `testReadPastEOF`.
pub fn test_read_past_eof(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let cr = create_large_cfs(t, &*dir);
    let bytes = cr.open("_123.f2").expect("cfs.openInput");
    let mut is = SliceInput::new(&bytes);
    let mut b = [0u8; 100];
    is.seek(is.len() - 10).expect("seek");
    is.read_bytes(&mut b[..10]).expect("readBytes");
    assert!(is.read_byte().is_err(), "a single byte past the end");
    is.seek(is.len() - 10).expect("seek");
    assert!(is.read_bytes(&mut b[..50]).is_err(), "a block past the end");
    drop(cr);
    dir.close().expect("close");
}

/// `testMissingCodecHeadersAreCaught`.
pub fn test_missing_codec_headers_are_caught(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let sub_file = "_123.xyz";
    let body: Vec<u8> = (0..1024).map(|i| i as u8).collect();
    write_file(&*dir, sub_file, &body).expect("write");
    let mut si = new_segment_info(t, "_123");
    si.files = vec![sub_file.to_string()];
    let e = write_compound(&*dir, "_123", &si).expect_err("a sub-file with no header");
    assert!(e.contains("bad magic"), "{e}");
    dir.close().expect("close");
}

/// A sub-file `_123.xyz` with a header, 1024 bytes and a footer whose
/// checksum is off by `skew`.
fn checksummed_sub_file(dir: &dyn Directory, id: &[u8; ID_LENGTH], skew: u64) {
    let mut bytes = Vec::new();
    codec_util::write_index_header(&mut bytes, "Foo", 0, id, "suffix");
    bytes.extend((0..1024).map(|i| i as u8));
    bytes.write_be_u32(codec_util::FOOTER_MAGIC);
    bytes.write_be_u32(0);
    let checksum = u64::from(crc32fast::hash(&bytes));
    bytes.write_be_u64(checksum.wrapping_add(skew));
    write_file(dir, "_123.xyz", &bytes).expect("write");
}

/// `testCorruptFilesAreCaught`.
pub fn test_corrupt_files_are_caught(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut si = new_segment_info(t, "_123");
    checksummed_sub_file(&*dir, &si.id, 1);
    si.files = vec!["_123.xyz".to_string()];
    let e = write_compound(&*dir, "_123", &si).expect_err("a sub-file with a wrong checksum");
    assert!(e.contains("checksum failed (hardware problem?)"), "{e}");
    dir.close().expect("close");
}

/// `testCheckIntegrity`: `checkIntegrity` passes over a sound pair, and
/// fails once a `.cfs` byte inside a member is flipped (which opening,
/// checking only the header, footer and length, lets through).
pub fn test_check_integrity(t: &mut LuceneTestCase) {
    let dir = t.new_directory();
    let mut si = new_segment_info(t, "_123");
    checksummed_sub_file(&*dir, &si.id, 0);
    si.files = vec!["_123.xyz".to_string()];
    write_compound(&*dir, "_123", &si).expect("compoundFormat().write");
    compound_reader(&*dir, "_123", &si)
        .check_integrity()
        .expect("checkIntegrity");

    let mut cfs = dir.open("_123.cfs").expect("open").to_vec();
    let mid = cfs.len() / 2;
    cfs[mid] ^= 0x01;
    dir.delete_file("_123.cfs").expect("delete");
    write_file(&*dir, "_123.cfs", &cfs).expect("write");
    let e = compound_reader(&*dir, "_123", &si)
        .check_integrity()
        .expect_err("a flipped byte");
    assert!(e.to_string().contains("checksum failed"), "{e}");
    dir.close().expect("close");
}

/// Every `BaseCompoundFormatTestCase` test this port runs by default.
pub const TESTS: &[SuiteTest] = &[
    ("testEmpty", test_empty),
    ("testSingleFile", test_single_file),
    ("testTwoFiles", test_two_files),
    ("testLargeCFS", test_large_cfs),
    ("testListAll", test_list_all),
    ("testCreateOutputDisabled", test_create_output_disabled),
    ("testDeleteFileDisabled", test_delete_file_disabled),
    ("testRenameFileDisabled", test_rename_file_disabled),
    ("testSyncDisabled", test_sync_disabled),
    ("testMakeLockDisabled", test_make_lock_disabled),
    ("testRandomFiles", test_random_files),
    ("testManySubFiles", test_many_sub_files),
    ("testClonedStreamsClosing", test_cloned_streams_closing),
    ("testRandomAccess", test_random_access),
    ("testRandomAccessClones", test_random_access_clones),
    ("testDoubleClose", test_double_close),
    ("testPassIOContext", test_pass_io_context),
    (
        "testResourceNameInsideCompoundFile",
        test_resource_name_inside_compound_file,
    ),
    ("testFileNotFound", test_file_not_found),
    ("testReadPastEOF", test_read_past_eof),
    (
        "testMissingCodecHeadersAreCaught",
        test_missing_codec_headers_are_caught,
    ),
    ("testCorruptFilesAreCaught", test_corrupt_files_are_caught),
    ("testCheckIntegrity", test_check_integrity),
];
