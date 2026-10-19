//! `FileMetaData`, `CopyState` and `NodeCommunicationException`: the values
//! a primary hands a replica.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use lucene_index::nrt::NrtFileHold;
use lucene_index::segment_infos::SegmentInfos;

/// `FileMetaData`: what a replica compares to decide whether it already has
/// a file -- the index header, the codec footer, the length and the checksum.
#[derive(Clone, PartialEq, Eq)]
pub struct FileMetaData {
    /// The bytes `CodecUtil.readIndexHeader` consumed.
    pub header: Vec<u8>,
    /// The 16 bytes of the codec footer.
    pub footer: Vec<u8>,
    pub length: u64,
    pub checksum: u64,
}

impl fmt::Debug for FileMetaData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FileMetaData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FileMetaData(length={} checksum={})",
            self.length, self.checksum
        )
    }
}

/// `CopyState`: one NRT point of the primary -- the files to have, the
/// `segments_N` bytes naming them, and the pin that keeps them from being
/// deleted until the state is released.
#[derive(Clone)]
pub struct CopyState {
    pub files: Arc<BTreeMap<String, FileMetaData>>,
    pub version: i64,
    pub gen: i64,
    pub infos_bytes: Arc<Vec<u8>>,
    pub completed_merge_files: Arc<BTreeSet<String>>,
    pub primary_gen: i64,
    pub infos: Arc<SegmentInfos>,
    /// The primary's `incRefDeleter(infos)`: the files stay while any clone
    /// of the state lives.
    pub hold: Option<Arc<NrtFileHold>>,
}

impl fmt::Debug for CopyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for CopyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CopyState(version={})", self.version)
    }
}

/// `NodeCommunicationException`: the other node could not be reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCommunicationError {
    /// What was being attempted.
    pub when: String,
    /// The failure underneath.
    pub cause: String,
}

impl fmt::Display for NodeCommunicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.when)
    }
}

impl std::error::Error for NodeCommunicationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_print_as_the_logs_expect() {
        let meta = FileMetaData {
            header: vec![1],
            footer: vec![2],
            length: 30,
            checksum: 7,
        };
        assert_eq!(format!("{meta:?}"), "FileMetaData(length=30 checksum=7)");
        let state = CopyState {
            files: Arc::new(BTreeMap::new()),
            version: 12,
            gen: 3,
            infos_bytes: Arc::new(Vec::new()),
            completed_merge_files: Arc::new(BTreeSet::new()),
            primary_gen: 1,
            infos: Arc::new(crate::replicator::nrt::replica::empty_infos()),
            hold: None,
        };
        assert_eq!(format!("{state:?}"), "CopyState(version=12)");
        let err = NodeCommunicationError {
            when: "sync".to_string(),
            cause: "reset".to_string(),
        };
        assert_eq!(err.to_string(), "sync");
        assert_eq!(err.clone(), err);
    }
}
