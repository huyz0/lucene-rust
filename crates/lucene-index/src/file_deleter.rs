//! `org.apache.lucene.util.FileDeleter`: reference counts for the files of a
//! directory, deleting a file when its last reference is dropped.
//!
//! [`crate::index_file_deleter::IndexFileDeleter`] keeps the same counts
//! inside the writer's commit tracking; this is the stand-alone form that
//! other users of a directory (the replicator's replica nodes) build on.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lucene_store::Directory;

use crate::index_file_deleter::Error;

type Result<T> = std::result::Result<T, Error>;

/// `FileDeleter.MsgType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgType {
    /// A reference count changed.
    Ref,
    /// A file is about to be deleted.
    File,
}

/// Where a [`FileDeleter`] reports what it does: `BiConsumer<MsgType, String>`.
pub type Messenger = Box<dyn Fn(MsgType, &str) + Send + Sync>;

/// `FileDeleter.RefCount`: `count` and whether the first increment was seen.
#[derive(Debug, Default)]
struct RefCount {
    init_done: bool,
    count: i32,
}

impl RefCount {
    fn inc_ref(&mut self) -> i32 {
        self.init_done = true;
        self.count += 1;
        self.count
    }

    fn dec_ref(&mut self) -> i32 {
        self.count -= 1;
        self.count
    }
}

/// `FileDeleter`.
pub struct FileDeleter {
    ref_counts: HashMap<String, RefCount>,
    directory: Arc<dyn Directory>,
    messenger: Option<Messenger>,
}

impl FileDeleter {
    /// `new FileDeleter(directory, messenger)`.
    pub fn new(directory: Arc<dyn Directory>, messenger: Option<Messenger>) -> Self {
        Self {
            ref_counts: HashMap::new(),
            directory,
            messenger,
        }
    }

    fn message(&self, kind: MsgType, text: &str) {
        if let Some(m) = &self.messenger {
            m(kind, text);
        }
    }

    /// `incRef(Collection)`.
    pub fn inc_ref_all<'a>(&mut self, file_names: impl IntoIterator<Item = &'a str>) {
        for file in file_names {
            self.inc_ref(file);
        }
    }

    /// `incRef(fileName)`.
    pub fn inc_ref(&mut self, file_name: &str) {
        let rc = self.ref_counts.entry(file_name.to_string()).or_default();
        let pre = rc.count;
        rc.inc_ref();
        self.message(
            MsgType::Ref,
            &format!("IncRef \"{file_name}\": pre-incr count is {pre}"),
        );
    }

    /// `decRef(Collection)`: every file whose count reaches zero is deleted,
    /// `segments*` files first. The first failure is returned after the rest
    /// were tried.
    ///
    /// # Errors
    /// The first deletion that failed.
    pub fn dec_ref_all<'a>(&mut self, file_names: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let mut to_delete: Vec<String> = Vec::new();
        for file in file_names {
            if self.dec_ref(file) && !to_delete.iter().any(|f| f == file) {
                to_delete.push(file.to_string());
            }
        }
        self.delete_all(&to_delete)
    }

    fn dec_ref(&mut self, file_name: &str) -> bool {
        let rc = self.ref_counts.entry(file_name.to_string()).or_default();
        let pre = rc.count;
        let now = rc.dec_ref();
        self.message(
            MsgType::Ref,
            &format!("DecRef \"{file_name}\": pre-decr count is {pre}"),
        );
        if now == 0 {
            self.ref_counts.remove(file_name);
            true
        } else {
            false
        }
    }

    /// `initRefCount(fileName)`: the file exists and nothing references it.
    pub fn init_ref_count(&mut self, file_name: &str) {
        self.ref_counts.entry(file_name.to_string()).or_default();
    }

    /// `getRefCount(fileName)`: 0 for a file never seen.
    pub fn ref_count(&self, file_name: &str) -> i32 {
        self.ref_counts.get(file_name).map_or(0, |rc| rc.count)
    }

    /// `getAllFiles()`.
    pub fn all_files(&self) -> HashSet<String> {
        self.ref_counts.keys().cloned().collect()
    }

    /// `exists(fileName)`: referenced at least once.
    pub fn exists(&self, file_name: &str) -> bool {
        self.ref_counts
            .get(file_name)
            .is_some_and(|rc| rc.count > 0)
    }

    /// `getUnrefedFiles()`: the files known but with a count of zero.
    pub fn unrefed_files(&self) -> HashSet<String> {
        let mut unrefed = HashSet::new();
        for (file_name, rc) in &self.ref_counts {
            if rc.count == 0 {
                self.message(
                    MsgType::File,
                    &format!("removing unreferenced file \"{file_name}\""),
                );
                unrefed.insert(file_name.clone());
            }
        }
        unrefed
    }

    /// `deleteFilesIfNoRef(files)`: deletes the ones nothing references.
    ///
    /// # Errors
    /// The first deletion that failed.
    pub fn delete_files_if_no_ref<'a>(
        &mut self,
        files: impl IntoIterator<Item = &'a str>,
    ) -> Result<()> {
        let mut to_delete: Vec<String> = Vec::new();
        for file in files {
            if !self.exists(file) {
                self.message(MsgType::File, &format!("will delete new file \"{file}\""));
                if !to_delete.iter().any(|f| f == file) {
                    to_delete.push(file.to_string());
                }
            }
        }
        self.delete_all(&to_delete)
    }

    /// `forceDelete(fileName)`: forgets the file and deletes it.
    ///
    /// # Errors
    /// The deletion's failure.
    pub fn force_delete(&mut self, file_name: &str) -> Result<()> {
        self.ref_counts.remove(file_name);
        self.delete(file_name)
    }

    /// `deleteFileIfNoRef(fileName)`.
    ///
    /// # Errors
    /// The deletion's failure.
    pub fn delete_file_if_no_ref(&mut self, file_name: &str) -> Result<()> {
        if !self.exists(file_name) {
            self.message(
                MsgType::File,
                &format!("will delete new file \"{file_name}\""),
            );
            self.delete(file_name)?;
        }
        Ok(())
    }

    /// `delete(Collection)`: `segments*` files first, then the rest, trying
    /// all and returning the first error.
    fn delete_all(&self, to_delete: &[String]) -> Result<()> {
        self.message(
            MsgType::File,
            &format!("now delete {} files: {:?}", to_delete.len(), to_delete),
        );
        let mut first: Option<Error> = None;
        let (segments, others): (Vec<&String>, Vec<&String>) =
            to_delete.iter().partition(|f| f.starts_with("segments"));
        for file in segments.into_iter().chain(others) {
            if let Err(e) = self.delete(file) {
                first.get_or_insert(e);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// `delete(fileName)`: a file already gone is an error, except on
    /// Windows, where a deletion can be reported late.
    fn delete(&self, file_name: &str) -> Result<()> {
        match self.directory.delete_file(file_name) {
            Ok(()) => Ok(()),
            Err(lucene_store::Error::Io(e))
                if e.kind() == std::io::ErrorKind::NotFound && cfg!(windows) =>
            {
                Ok(())
            }
            Err(e) => Err(Error::from(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::ByteBuffersDirectory;
    use std::sync::Mutex;

    fn dir_with(files: &[&str]) -> Arc<ByteBuffersDirectory> {
        let dir = Arc::new(ByteBuffersDirectory::new());
        for f in files {
            dir.create_output(f).unwrap().close().unwrap();
        }
        dir
    }

    #[test]
    fn a_file_is_deleted_when_its_last_reference_goes() {
        let dir = dir_with(&["_0.cfs", "_1.cfs"]);
        let mut d = FileDeleter::new(dir.clone(), None);
        d.inc_ref_all(["_0.cfs", "_1.cfs"]);
        d.inc_ref("_0.cfs");
        assert_eq!(d.ref_count("_0.cfs"), 2);
        assert!(d.exists("_1.cfs"));
        d.dec_ref_all(["_0.cfs", "_1.cfs"]).unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["_0.cfs".to_string()]);
        d.dec_ref_all(["_0.cfs"]).unwrap();
        assert!(dir.list_all().unwrap().is_empty());
        assert_eq!(d.ref_count("_0.cfs"), 0);
        assert!(!d.exists("_0.cfs"));
    }

    #[test]
    fn unreferenced_files_are_the_ones_initialised_but_never_referenced() {
        let dir = dir_with(&["a", "b", "c"]);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let mut d = FileDeleter::new(
            dir.clone(),
            Some(Box::new(move |k, s| {
                log.lock().unwrap().push((k, s.to_string()))
            })),
        );
        d.init_ref_count("a");
        d.inc_ref("b");
        assert_eq!(d.unrefed_files(), HashSet::from(["a".to_string()]));
        assert_eq!(d.all_files().len(), 2);
        d.delete_files_if_no_ref(["a", "b", "c"]).unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["b".to_string()]);
        let seen = seen.lock().unwrap();
        assert!(seen
            .iter()
            .any(|(k, s)| *k == MsgType::Ref && s == "IncRef \"b\": pre-incr count is 0"));
        assert!(seen
            .iter()
            .any(|(k, s)| *k == MsgType::File && s == "removing unreferenced file \"a\""));
    }

    #[test]
    fn forced_and_conditional_deletes() {
        let dir = dir_with(&["a", "b"]);
        let mut d = FileDeleter::new(dir.clone(), None);
        d.inc_ref("a");
        d.delete_file_if_no_ref("a").unwrap();
        assert_eq!(dir.list_all().unwrap().len(), 2);
        d.force_delete("a").unwrap();
        assert_eq!(d.ref_count("a"), 0);
        d.delete_file_if_no_ref("b").unwrap();
        assert!(dir.list_all().unwrap().is_empty());
    }

    #[test]
    fn segments_files_are_deleted_first_and_the_first_error_wins() {
        let dir = dir_with(&["segments_1", "x"]);
        let mut d = FileDeleter::new(dir.clone(), None);
        d.inc_ref_all(["segments_1", "x", "ghost"]);
        let r = d.dec_ref_all(["segments_1", "ghost", "x"]);
        // Everything that could be deleted was, whatever the ghost did.
        assert!(dir.list_all().unwrap().is_empty());
        assert_eq!(r.is_err(), !cfg!(windows));
    }
}
