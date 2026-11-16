#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use lucene_util::test_support::TempDir;

fn commits(n: i64) -> Vec<IndexCommit> {
    (1..=n)
        .map(|g| {
            IndexCommit::new(
                g,
                format!("segments_{g}"),
                vec![format!("segments_{g}")],
                1,
                vec![("k".into(), g.to_string())],
            )
        })
        .collect()
}

fn deleted(c: &[IndexCommit]) -> Vec<i64> {
    c.iter()
        .filter(|c| c.is_deleted())
        .map(|c| c.generation())
        .collect()
}

#[test]
fn stateless_policies() {
    let mut c = commits(3);
    KeepOnlyLastCommitDeletionPolicy.on_init(&mut c).unwrap();
    assert_eq!(deleted(&c), vec![1, 2]);
    let mut c = commits(3);
    NoDeletionPolicy.on_init(&mut c).unwrap();
    NoDeletionPolicy.on_commit(&mut c).unwrap();
    assert!(deleted(&c).is_empty());
    assert!(KeepLastNCommitsDeletionPolicy::new(0).is_err());
    let p = KeepLastNCommitsDeletionPolicy::new(2).unwrap();
    let mut c = commits(5);
    p.on_init(&mut c).unwrap();
    assert_eq!(deleted(&c), vec![1, 2, 3]);
    let mut c = commits(1);
    p.on_commit(&mut c).unwrap();
    assert!(deleted(&c).is_empty());
    let c = &commits(1)[0];
    assert_eq!(c.segments_file_name(), "segments_1");
    assert_eq!(c.file_names(), ["segments_1"]);
    assert_eq!(c.segment_count(), 1);
    assert_eq!(c.user_data()[0].1, "1");
}

#[test]
fn snapshots_pin_commits_until_released() {
    let p = SnapshotDeletionPolicy::new(Box::new(KeepOnlyLastCommitDeletionPolicy));
    assert!(matches!(p.snapshot(), Err(Error::IllegalState(_))));
    assert!(matches!(p.release_gen(1), Err(Error::IllegalState(_))));
    let mut empty: Vec<IndexCommit> = Vec::new();
    p.on_init(&mut empty).unwrap();
    assert!(matches!(p.snapshot(), Err(Error::IllegalState(_))));
    let mut c = commits(1);
    p.on_commit(&mut c).unwrap();
    let s1 = p.snapshot().unwrap();
    let s1b = p.snapshot().unwrap();
    assert_eq!(s1, s1b);
    assert_eq!(p.snapshot_count(), 2);
    let mut c = commits(2);
    p.on_commit(&mut c).unwrap();
    // gen 1 is snapshotted: the primary's delete is ignored.
    assert!(deleted(&c).is_empty());
    p.release(&s1).unwrap();
    let mut c = commits(2);
    p.on_commit(&mut c).unwrap();
    assert!(deleted(&c).is_empty());
    assert_eq!(p.snapshots().len(), 1);
    assert_eq!(p.index_commit(1).unwrap().generation(), 1);
    p.release(&s1b).unwrap();
    assert!(p.index_commit(1).is_none());
    let mut c = commits(2);
    p.on_commit(&mut c).unwrap();
    assert_eq!(deleted(&c), vec![1]);
    assert!(matches!(p.release_gen(1), Err(Error::IllegalArgument(_))));
}

#[test]
fn on_init_rebinds_known_snapshots() {
    let p = SnapshotDeletionPolicy::new(Box::new(KeepOnlyLastCommitDeletionPolicy));
    p.lock().ref_counts.insert(1, 1);
    let mut c = commits(3);
    p.on_init(&mut c).unwrap();
    assert_eq!(deleted(&c), vec![2]);
    assert_eq!(p.index_commit(1).unwrap().generation(), 1);
}

#[test]
fn persistent_snapshots_survive_reopen() {
    let tmp = TempDir::new("persistent-snapshots");
    let dir: Arc<dyn Directory> = Arc::new(lucene_store::FsDirectory::open(tmp.path()));
    let p = PersistentSnapshotDeletionPolicy::new(
        Box::new(KeepOnlyLastCommitDeletionPolicy),
        Arc::clone(&dir),
        OpenMode::CreateOrAppend,
    )
    .unwrap();
    assert!(p.last_save_file().is_none());
    assert!(matches!(
        PersistentSnapshotDeletionPolicy::new(
            Box::new(NoDeletionPolicy),
            Arc::clone(&dir),
            OpenMode::Append
        ),
        Err(Error::IllegalState(_))
    ));
    let mut c = commits(3);
    p.on_init(&mut c).unwrap();
    let s = p.snapshot().unwrap();
    assert_eq!(s.generation(), 3);
    p.snapshot().unwrap();
    assert_eq!(p.last_save_file().as_deref(), Some("snapshots_1"));
    assert_eq!(dir.list_all().unwrap(), vec!["snapshots_1".to_string()]);

    // A second policy over the same directory sees the stored counts.
    let q = PersistentSnapshotDeletionPolicy::new(
        Box::new(KeepOnlyLastCommitDeletionPolicy),
        Arc::clone(&dir),
        OpenMode::Append,
    )
    .unwrap();
    assert_eq!(q.snapshots_policy().snapshot_count(), 2);
    let mut c = commits(4);
    q.on_init(&mut c).unwrap();
    assert_eq!(deleted(&c), vec![1, 2]);
    q.release_gen(3).unwrap();
    q.release(&s).unwrap();
    assert!(q.release_gen(3).is_err());
    assert_eq!(q.snapshots_policy().snapshot_count(), 0);
    let mut c = commits(4);
    q.on_commit(&mut c).unwrap();
    assert_eq!(deleted(&c), vec![1, 2, 3]);

    // CREATE wipes them.
    let r = PersistentSnapshotDeletionPolicy::new(
        Box::new(NoDeletionPolicy),
        Arc::clone(&dir),
        OpenMode::Create,
    )
    .unwrap();
    assert!(r.last_save_file().is_none());
    assert!(dir.list_all().unwrap().is_empty());
}

#[test]
fn corrupt_snapshots_file_is_an_error() {
    let tmp = TempDir::new("persistent-snapshots-corrupt");
    let dir: Arc<dyn Directory> = Arc::new(lucene_store::FsDirectory::open(tmp.path()));
    let mut out = dir.create_output("snapshots_0").unwrap();
    out.write_bytes(b"garbage");
    out.close().unwrap();
    // The newest file failed to read: its counts are empty, and it is
    // still the one loaded (Java records the error only when no file
    // loaded at all).
    let p = PersistentSnapshotDeletionPolicy::new(
        Box::new(NoDeletionPolicy),
        Arc::clone(&dir),
        OpenMode::CreateOrAppend,
    )
    .unwrap();
    assert_eq!(p.last_save_file().as_deref(), Some("snapshots_0"));
    let out = dir.create_output("snapshots_x").unwrap();
    out.close().unwrap();
    assert!(PersistentSnapshotDeletionPolicy::new(
        Box::new(NoDeletionPolicy),
        dir,
        OpenMode::CreateOrAppend,
    )
    .is_err());
}
