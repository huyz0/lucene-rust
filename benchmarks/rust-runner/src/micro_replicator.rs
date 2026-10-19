//! M14 T14.3's replicator pairs (`benchmarks/micro/java/ReplicatorMicro.java`
//! is the Java twin), over the 25 files of the NRT point in
//! `fixtures/data/replicator/plain` (copied to a scratch directory first):
//!
//! * `repl_file_meta`: `NodeCore::read_local_file_metadata` of every file --
//!   what a primary does per file of each new point;
//! * `repl_copy_file`: `CopyOneFile` of every file into a replica's
//!   directory, checksum verified, then the temporary file deleted.
//!
//! Units: files.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_search::replicator::nrt::local::{CopySource, FileStream, LocalReplicaHooks};
use lucene_search::replicator::nrt::{
    CopyOneFile, CopyState, CopyStream, FileMetaData, NodeCore, ReplicaNode,
};
use lucene_store::{Directory, FsDirectory};

use super::measure;

const SRC: &str = "fixtures/data/replicator/plain";

/// The replica is never asked to copy a point, only single files.
struct NoPrimary;

impl CopySource for NoPrimary {
    fn copy_state(&self) -> lucene_search::Result<CopyState> {
        Err(lucene_search::Error::IllegalState("no primary".to_string()))
    }
    fn release_copy_state(&self, _copy_state: CopyState) {}
    fn open_files(
        &self,
        _requests: &[(String, u64)],
    ) -> lucene_search::Result<Box<dyn CopyStream>> {
        Err(lucene_search::Error::IllegalState("no primary".to_string()))
    }
}

/// Copies the fixture point to a scratch directory first: read off the
/// repository's bind mount, a file open costs a round trip to the host and
/// the benchmark measures the mount (milliseconds per file, on both sides).
fn scratch_copy_of_point() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("repl-bench-src-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(SRC).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
    }
    dir
}

pub fn bench_replicator(w: Duration, m: Duration) {
    let src_path = scratch_copy_of_point();
    let src = FsDirectory::open(&src_path);
    let names: Vec<String> = src
        .list_all()
        .unwrap()
        .into_iter()
        .filter(|n| !n.starts_with("segments_") && n != "write.lock")
        .collect();
    let node = NodeCore::new(0, true, Arc::new(FsDirectory::open(&src_path)), None);
    node.set_verbose_files(false);
    measure("repl_file_meta", w, m, || {
        for name in &names {
            black_box(
                node.read_local_file_metadata(name)
                    .unwrap()
                    .unwrap()
                    .checksum,
            );
        }
        names.len() as u64
    });

    let metas: Vec<FileMetaData> = names
        .iter()
        .map(|n| node.read_local_file_metadata(n).unwrap().unwrap())
        .collect();
    let tmp = std::env::temp_dir().join(format!("repl-bench-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let replica = ReplicaNode::new(
        1,
        Arc::new(FsDirectory::open(&tmp)),
        None,
        LocalReplicaHooks::new(Arc::new(NoPrimary), None),
    )
    .unwrap();
    replica.core().node.set_verbose_files(false);
    measure("repl_copy_file", w, m, || {
        for (name, meta) in names.iter().zip(&metas) {
            let mut stream = FileStream::open(&src, &[(name.clone(), 0)]).unwrap();
            // The length the primary sends ahead of the bytes.
            black_box(stream.read_vlong().unwrap());
            let mut copy =
                CopyOneFile::new(Arc::clone(replica.core()), name, meta.clone(), 65536).unwrap();
            while !copy.visit(&mut stream).unwrap() {}
            replica.core().dir().delete_file(&copy.tmp_name).unwrap();
        }
        names.len() as u64
    });
    drop(replica);
    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::remove_dir_all(&src_path);
}
