//! `ReplicaNode`'s start-up against directories it should not trust, and its
//! behaviour once closed. No primary is involved: the transport refuses
//! every job.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_search::replicator::nrt::{
    CopyJob, FileMetaData, OnceDone, ReplicaCore, ReplicaHooks, ReplicaNode,
};
use lucene_search::{Error, Result};
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;

struct NoPrimary;

impl ReplicaHooks for NoPrimary {
    fn new_copy_job(
        &self,
        _core: &Arc<ReplicaCore>,
        _reason: &str,
        _files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        _prev_files: Option<Arc<BTreeMap<String, FileMetaData>>>,
        _high_priority: bool,
        _once_done: Option<OnceDone>,
    ) -> Result<Arc<dyn CopyJob>> {
        Err(ReplicaNode::communication_error("no primary", "none here"))
    }

    fn launch(&self, _job: Arc<dyn CopyJob>) -> Result<()> {
        Ok(())
    }

    fn send_new_replica(&self) -> Result<()> {
        Ok(())
    }
}

fn touch(dir: &FsDirectory, name: &str) {
    dir.create_output(name).unwrap().close().unwrap();
}

#[test]
fn a_pending_segments_file_moves_the_generation_up() {
    let tmp = TempDir::new("replicator-edge-pending");
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    touch(&dir, "pending_segments_9");
    let replica = ReplicaNode::new(1, dir.clone(), None, Arc::new(NoPrimary)).unwrap();
    replica.start(1).unwrap();
    replica.commit().unwrap();
    // 9 + 1, in base 36.
    assert!(dir.list_all().unwrap().contains(&"segments_a".to_string()));
    replica.close().unwrap();
}

#[test]
fn a_pending_segments_file_with_a_bad_name_is_corrupt() {
    let tmp = TempDir::new("replicator-edge-badpending");
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    touch(&dir, "pending_segments_!!");
    let replica = ReplicaNode::new(1, dir, None, Arc::new(NoPrimary)).unwrap();
    let err = replica.start(1).unwrap_err();
    assert!(err.to_string().contains("pending_segments_!!"), "{err}");
}

#[test]
fn a_replica_that_is_closed_takes_no_more_points() {
    let tmp = TempDir::new("replicator-edge-closed");
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    let replica = ReplicaNode::new(1, dir, None, Arc::new(NoPrimary)).unwrap();
    replica.start(1).unwrap();
    assert_eq!(replica.state(), "idle");
    assert!(!replica.is_closed());
    replica.close().unwrap();
    assert!(replica.is_closed());
    assert!(matches!(
        replica.new_nrt_point(1, 5),
        Err(Error::AlreadyClosed(_))
    ));
}

#[test]
fn a_replica_that_cannot_reach_the_primary_ignores_the_point() {
    let tmp = TempDir::new("replicator-edge-unreachable");
    let dir = Arc::new(FsDirectory::open(tmp.path()));
    let replica = ReplicaNode::new(1, dir, None, Arc::new(NoPrimary)).unwrap();
    replica.start(1).unwrap();
    assert!(replica.new_nrt_point(1, 5).unwrap().is_none());
    assert_eq!(replica.state(), "idle");
    replica.close().unwrap();
}
