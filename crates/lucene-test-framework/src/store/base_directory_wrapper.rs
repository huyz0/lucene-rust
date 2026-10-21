//! `BaseDirectoryWrapper` and `RawDirectoryWrapper`: what
//! `LuceneTestCase.newDirectory` hands a test, and what closing it checks.
//!
//! Java's wrapper runs `CheckIndex` from `close()`. Here a directory has no
//! `close` (dropping it is closing it, and a check cannot fail a `Drop`), so
//! [`BaseDirectoryWrapper::close`] is an explicit call that returns the
//! check's verdict; a wrapper dropped without it checks nothing.
//!
//! Differs: `CheckIndex` has no levels in this port, so
//! `setCrossCheckTermVectorsOnClose`/`getLevelForCheckOnClose` are not
//! ported -- every check runs at the one level there is.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lucene_store::index_output::FsIndexOutput;
use lucene_store::lock::Lock;
use lucene_store::{Directory, Input, Result};
use lucene_util::test_support::TempDir;

use super::mock_directory_wrapper::MockDirectoryWrapper;

/// `BaseDirectoryWrapper`: a test's directory, which checks the index when
/// it is closed.
pub trait BaseDirectoryWrapper: Directory {
    /// `close()`: when the index exists and the check is on, runs
    /// `CheckIndex` over it; an `Err` is the failure Java throws. A second
    /// call checks nothing.
    fn close(&self) -> std::result::Result<(), String>;

    /// `isOpen()`.
    fn is_open(&self) -> bool;

    /// `setCheckIndexOnClose(value)`.
    fn set_check_index_on_close(&self, value: bool);

    /// `getCheckIndexOnClose()`.
    fn check_index_on_close(&self) -> bool;

    /// The [`MockDirectoryWrapper`] this is, when it is one (Java's
    /// `instanceof MockDirectoryWrapper` and cast).
    fn as_mock(&self) -> Option<&MockDirectoryWrapper> {
        None
    }
}

/// `DirectoryReader.indexExists(dir)`: some `segments_N` is listed.
pub fn index_exists(dir: &dyn Directory) -> Result<bool> {
    Ok(lucene_store::directory::last_commit_generation(&dir.list_all()?)? != -1)
}

/// `RawDirectoryWrapper`: a [`BaseDirectoryWrapper`] that only forwards --
/// what `newDirectory` returns when it draws a "bare" directory.
pub struct RawDirectoryWrapper {
    inner: Arc<dyn Directory>,
    open: AtomicBool,
    check_index_on_close: AtomicBool,
    /// The scratch directories under `inner`, removed when this is dropped.
    _temps: Vec<TempDir>,
}

impl RawDirectoryWrapper {
    /// `new RawDirectoryWrapper(delegate)`.
    pub fn new(delegate: impl Directory + 'static) -> Self {
        Self::over(Arc::new(delegate), Vec::new())
    }

    /// Over an already shared delegate, keeping `temps` alive with it.
    pub(crate) fn over(inner: Arc<dyn Directory>, temps: Vec<TempDir>) -> Self {
        Self {
            inner,
            open: AtomicBool::new(true),
            check_index_on_close: AtomicBool::new(true),
            _temps: temps,
        }
    }

    /// `FilterDirectory.getDelegate()`.
    pub fn delegate(&self) -> &dyn Directory {
        self.inner.as_ref()
    }

    /// Keeps `temp` alive as long as this wrapper.
    pub(crate) fn keep_alive(&mut self, temp: TempDir) {
        self._temps.push(temp);
    }
}

impl BaseDirectoryWrapper for RawDirectoryWrapper {
    fn close(&self) -> std::result::Result<(), String> {
        if self.open.swap(false, Ordering::SeqCst)
            && self.check_index_on_close()
            && index_exists(self).map_err(|e| e.to_string())?
        {
            crate::util::test_util::check_index(self)?;
        }
        Ok(())
    }

    fn is_open(&self) -> bool {
        self.open.load(Ordering::SeqCst)
    }

    fn set_check_index_on_close(&self, value: bool) {
        self.check_index_on_close.store(value, Ordering::SeqCst);
    }

    fn check_index_on_close(&self) -> bool {
        self.check_index_on_close.load(Ordering::SeqCst)
    }
}

impl Directory for RawDirectoryWrapper {
    fn list_all(&self) -> Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> Result<Input> {
        self.inner.open(name)
    }
    fn file_length(&self, name: &str) -> Result<u64> {
        self.inner.file_length(name)
    }
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.inner.create_output(name)
    }
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        self.inner.create_temp_output(prefix, suffix)
    }
    fn sync(&self, names: &[String]) -> Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.inner.rename(source, dest)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.inner.obtain_lock(name)
    }
    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.inner.pending_deletions()
    }
    fn fs_directory_path(&self) -> Option<&Path> {
        self.inner.fs_directory_path()
    }
    /// `RawDirectoryWrapper.copyFrom`: the delegate's own, not the default.
    fn copy_from(&self, from: &dyn Directory, src: &str, dest: &str) -> Result<()> {
        self.inner.copy_from(from, src, dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::{ByteBuffersDirectory, DataOutput, IndexOutput};

    #[test]
    fn raw_wrapper_forwards_and_checks_once() {
        let dir = RawDirectoryWrapper::new(ByteBuffersDirectory::new());
        assert!(dir.as_mock().is_none());
        assert!(dir.is_open());
        assert!(dir.check_index_on_close());
        let mut out = dir.create_output("a").unwrap();
        out.write_bytes(b"abc");
        out.close().unwrap();
        let out = dir.create_temp_output("p", "s").unwrap();
        let temp = out.name().to_string();
        out.close().unwrap();
        assert_eq!(dir.file_length("a").unwrap(), 3);
        assert_eq!(&*dir.open("a").unwrap(), b"abc");
        dir.sync(&["a".to_string()]).unwrap();
        dir.rename("a", "b").unwrap();
        dir.sync_meta_data().unwrap();
        dir.copy_from(dir.delegate(), "b", "c").unwrap();
        dir.delete_file(&temp).unwrap();
        assert_eq!(
            dir.list_all().unwrap(),
            vec!["b".to_string(), "c".to_string()]
        );
        assert!(dir.pending_deletions().unwrap().is_empty());
        assert!(dir.fs_directory_path().is_none());
        drop(dir.obtain_lock("write.lock").unwrap());
        assert!(!index_exists(&dir).unwrap());
        dir.close().unwrap();
        assert!(!dir.is_open());
        dir.close().unwrap();
    }

    /// The close check runs `CheckIndex` and reports what it found: here a
    /// `segments_1` that is not a commit.
    #[test]
    fn raw_wrapper_close_reports_a_broken_index() {
        let dir = RawDirectoryWrapper::new(ByteBuffersDirectory::new());
        let mut out = dir.create_output("segments_1").unwrap();
        out.write_bytes(b"not a commit");
        out.close().unwrap();
        assert!(index_exists(&dir).unwrap());
        dir.set_check_index_on_close(false);
        dir.close().unwrap();

        let dir = RawDirectoryWrapper::new(ByteBuffersDirectory::new());
        let mut out = dir.create_output("segments_1").unwrap();
        out.write_bytes(b"not a commit");
        out.close().unwrap();
        assert!(dir.close().is_err());
    }
}
