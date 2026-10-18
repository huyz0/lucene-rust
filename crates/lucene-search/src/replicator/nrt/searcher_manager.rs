//! `SegmentInfosSearcherManager`: the replica's readers, opened over a
//! `SegmentInfos` handed in instead of a commit found in the directory.

use std::sync::{Arc, Mutex, MutexGuard};

use lucene_index::segment_infos::SegmentInfos;
use lucene_store::directory::Directory;

use super::node::NodeCore;
use crate::directory_reader::DirectoryReader;
use crate::reference_manager::{ReferenceManager, Refresher};
use crate::Result;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Opens the next reader over whatever infos the manager currently holds,
/// reusing the segments of the reader it replaces.
struct InfosRefresher {
    dir: Arc<dyn Directory>,
    node: Arc<NodeCore>,
    current_infos: Arc<Mutex<SegmentInfos>>,
}

impl Refresher<DirectoryReader> for InfosRefresher {
    fn refresh_if_needed(
        &self,
        old: &Arc<DirectoryReader>,
    ) -> Result<Option<Arc<DirectoryReader>>> {
        let infos = lock(&self.current_infos).clone();
        let version = infos.version;
        let reader = old.reopen_at(self.dir.as_ref(), infos)?;
        self.node.message(&format!(
            "refreshed to version={version} segments={}",
            reader.segment_infos.segments.len()
        ));
        Ok(Some(Arc::new(reader)))
    }
}

/// `SegmentInfosSearcherManager`.
pub struct SegmentInfosSearcherManager {
    node: Arc<NodeCore>,
    current_infos: Arc<Mutex<SegmentInfos>>,
    manager: ReferenceManager<DirectoryReader>,
}

impl SegmentInfosSearcherManager {
    /// `new SegmentInfosSearcherManager(dir, node, infos, searcherFactory)`:
    /// opens the first reader over `infos`.
    ///
    /// # Errors
    /// A segment of `infos` could not be opened.
    pub fn new(dir: Arc<dyn Directory>, node: Arc<NodeCore>, infos: SegmentInfos) -> Result<Self> {
        node.message(&format!(
            "SegmentInfosSearcherManager.init: use incoming infos={}",
            describe(&infos)
        ));
        let first = DirectoryReader::open_at(dir.as_ref(), infos.clone())?;
        let current_infos = Arc::new(Mutex::new(infos));
        let refresher = InfosRefresher {
            dir,
            node: Arc::clone(&node),
            current_infos: Arc::clone(&current_infos),
        };
        Ok(Self {
            node,
            current_infos,
            manager: ReferenceManager::new(Arc::new(first), Box::new(refresher)),
        })
    }

    /// `getCurrentInfos()`.
    pub fn current_infos(&self) -> SegmentInfos {
        lock(&self.current_infos).clone()
    }

    /// `setCurrentInfos(infos)`: carries the generation over from the infos
    /// being replaced, then refreshes to `infos`.
    ///
    /// # Errors
    /// A segment of `infos` could not be opened.
    pub fn set_current_infos(&self, mut infos: SegmentInfos) -> Result<()> {
        {
            let mut cur = lock(&self.current_infos);
            infos.generation = cur.generation;
            self.node.message(&format!(
                "mgr.setCurrentInfos: carry over infos gen={}",
                segments_file_name(&infos)
            ));
            *cur = infos;
        }
        self.manager.maybe_refresh_blocking()
    }

    /// Records the commit of the point at `version` just written:
    /// `infos.updateGeneration(...)` plus the user data `commit()` set on the
    /// same object in Java. The generation always carries over (as
    /// `setCurrentInfos` carries it to a newer point); the user data only
    /// when the current point is still the one committed -- in Java a point
    /// swapped in meanwhile is another object and never saw it.
    pub fn committed(&self, version: i64, generation: i64, user_data: Vec<(String, String)>) {
        let mut cur = lock(&self.current_infos);
        cur.generation = generation;
        if cur.version == version {
            cur.user_data = user_data;
        }
    }

    /// `acquire()`.
    ///
    /// # Errors
    /// The manager is closed.
    pub fn acquire(&self) -> Result<Arc<DirectoryReader>> {
        self.manager.acquire()
    }

    /// `release(reader)`.
    pub fn release(&self, reader: Arc<DirectoryReader>) {
        self.manager.release(reader);
    }

    /// `close()`.
    ///
    /// # Errors
    /// Closing failed.
    pub fn close(&self) -> Result<()> {
        self.manager.close()
    }
}

/// `SegmentInfos.getSegmentsFileName()`.
pub(crate) fn segments_file_name(infos: &SegmentInfos) -> String {
    lucene_store::directory::segments_file_name(infos.generation).unwrap_or_default()
}

/// `SegmentInfos.toString()`, in the form its log lines use:
/// `segments_N: _0(...):C5 ...`.
pub(crate) fn describe(infos: &SegmentInfos) -> String {
    let segs: Vec<String> = infos
        .segments
        .iter()
        .map(|s| format!("{}:{}", s.segment_name, s.del_count))
        .collect();
    format!("{}({})", segments_file_name(infos), segs.join(" "))
}
