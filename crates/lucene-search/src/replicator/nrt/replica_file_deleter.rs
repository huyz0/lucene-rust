//! `ReplicaFileDeleter`: the replica's reference counts over the files of its
//! directory.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use lucene_index::file_deleter::{FileDeleter, MsgType};
use lucene_store::directory::Directory;

use super::node::NodeCore;
use crate::Result;

/// `ReplicaFileDeleter`.
pub struct ReplicaFileDeleter {
    file_deleter: Mutex<FileDeleter>,
    dir: Arc<dyn Directory>,
    node: Arc<NodeCore>,
}

impl ReplicaFileDeleter {
    /// `new ReplicaFileDeleter(node, dir)`: file deletions are logged on the
    /// node when it is verbose about files.
    pub fn new(node: Arc<NodeCore>, dir: Arc<dyn Directory>) -> Self {
        let log_node = Arc::clone(&node);
        let file_deleter = FileDeleter::new(
            Arc::clone(&dir),
            Some(Box::new(move |kind, text| {
                if kind == MsgType::File && log_node.is_verbose_files() {
                    log_node.message(text);
                }
            })),
        );
        Self {
            file_deleter: Mutex::new(file_deleter),
            dir,
            node,
        }
    }

    fn deleter(&self) -> MutexGuard<'_, FileDeleter> {
        self.file_deleter.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `incRef(fileNames)`.
    pub fn inc_ref<'a>(&self, file_names: impl IntoIterator<Item = &'a str>) {
        self.deleter().inc_ref_all(file_names);
    }

    /// `decRef(fileNames)`: files nothing references any more are deleted.
    ///
    /// # Errors
    /// The first deletion that failed.
    pub fn dec_ref<'a>(&self, file_names: impl IntoIterator<Item = &'a str>) -> Result<()> {
        Ok(self.deleter().dec_ref_all(file_names)?)
    }

    /// `getRefCount(fileName)`.
    pub fn ref_count(&self, file_name: &str) -> i32 {
        self.deleter().ref_count(file_name)
    }

    /// `deleteIfNoRef(fileName)`.
    ///
    /// # Errors
    /// The deletion's failure.
    pub fn delete_if_no_ref(&self, file_name: &str) -> Result<()> {
        Ok(self.deleter().delete_file_if_no_ref(file_name)?)
    }

    /// `forceDeleteFile(fileName)`.
    ///
    /// # Errors
    /// The deletion's failure.
    pub fn force_delete_file(&self, file_name: &str) -> Result<()> {
        Ok(self.deleter().force_delete(file_name)?)
    }

    /// `deleteUnknownFiles(segmentsFileName)`: deletes every file of the
    /// directory that nothing references, except the write lock and the
    /// commit being started from.
    ///
    /// # Errors
    /// The first deletion that failed.
    pub fn delete_unknown_files(&self, segments_file_name: Option<&str>) -> Result<()> {
        let mut deleter = self.deleter();
        let mut to_delete: HashSet<String> = deleter.unrefed_files();
        for file_name in self.dir.list_all()? {
            if !deleter.exists(&file_name)
                && file_name != "write.lock"
                && Some(file_name.as_str()) != segments_file_name
            {
                self.node
                    .message(&format!("will delete unknown file \"{file_name}\""));
                to_delete.insert(file_name);
            }
        }
        Ok(deleter.delete_files_if_no_ref(to_delete.iter().map(String::as_str))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::ByteBuffersDirectory;

    fn put(dir: &ByteBuffersDirectory, name: &str) {
        dir.create_output(name).unwrap().close().unwrap();
    }

    #[test]
    fn unknown_files_go_but_the_lock_and_the_start_commit_stay() {
        let dir = Arc::new(ByteBuffersDirectory::new());
        for f in ["_0.cfs", "_1.cfs", "write.lock", "segments_3", "stray"] {
            put(&dir, f);
        }
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = Arc::clone(&log);
        let node = Arc::new(NodeCore::new(
            1,
            false,
            dir.clone(),
            Some(Arc::new(move |s: &str| {
                sink.lock().unwrap().push(s.to_string())
            })),
        ));
        let deleter = ReplicaFileDeleter::new(node, dir.clone());
        deleter.inc_ref(["_0.cfs"]);
        deleter.delete_unknown_files(Some("segments_3")).unwrap();
        let mut left = dir.list_all().unwrap();
        left.sort();
        assert_eq!(left, ["_0.cfs", "segments_3", "write.lock"]);
        assert!(log
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.contains("will delete unknown file \"stray\"")));

        assert_eq!(deleter.ref_count("_0.cfs"), 1);
        deleter.delete_if_no_ref("_0.cfs").unwrap();
        assert_eq!(dir.list_all().unwrap().len(), 3);
        deleter.dec_ref(["_0.cfs"]).unwrap();
        assert_eq!(dir.list_all().unwrap().len(), 2);
        deleter.force_delete_file("segments_3").unwrap();
        assert_eq!(dir.list_all().unwrap(), ["write.lock"]);
    }
}
