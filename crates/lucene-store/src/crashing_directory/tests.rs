#![cfg(test)]
use super::*;
use crate::data_output::DataOutput;

fn tempdir(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "lucene-rust-crashing-dir-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

fn write(dir: &CrashingDirectory, name: &str, bytes: &[u8]) {
    let mut out = dir.create_output(name).unwrap();
    out.write_bytes(bytes);
    out.close().unwrap();
}

#[test]
fn synced_and_published_files_survive_every_power_loss() {
    for seed in 0..50 {
        let root = tempdir(&format!("durable-{seed}"));
        let dir = CrashingDirectory::new(&root, seed);
        write(&dir, "pending_segments_1", b"commit");
        dir.sync(&["pending_segments_1".to_string()]).unwrap();
        dir.rename("pending_segments_1", "segments_1").unwrap();
        dir.sync_meta_data().unwrap();
        write(&dir, "_0.tmp", b"scratch");
        dir.power_loss().unwrap();
        assert_eq!(fs::read(root.join("segments_1")).unwrap(), b"commit");
        assert!(!root.join("pending_segments_1").exists());
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn an_unpublished_rename_can_be_lost_but_never_reordered() {
    // Over many seeds, both outcomes happen -- and never a state where the
    // rename survived but the synced file it renamed did not exist.
    let (mut kept, mut lost) = (0, 0);
    for seed in 0..64 {
        let root = tempdir(&format!("rename-{seed}"));
        let dir = CrashingDirectory::new(&root, seed);
        write(&dir, "pending_segments_2", b"commit");
        dir.sync(&["pending_segments_2".to_string()]).unwrap();
        dir.rename("pending_segments_2", "segments_2").unwrap();
        let report = dir.power_loss().unwrap();
        assert_eq!(report.pending_events, 2);
        let (pending, published) = (
            root.join("pending_segments_2").exists(),
            root.join("segments_2").exists(),
        );
        match report.kept_events {
            0 => assert!(!pending && !published),
            1 => {
                assert!(pending && !published);
                lost += 1;
            }
            _ => {
                assert!(!pending && published);
                assert_eq!(fs::read(root.join("segments_2")).unwrap(), b"commit");
                kept += 1;
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
    assert!(kept > 0 && lost > 0, "kept {kept}, lost {lost}");
}

#[test]
fn unsynced_contents_are_damaged_and_an_unpublished_delete_can_return() {
    let mut outcomes = BTreeSet::new();
    let mut restored = 0;
    for seed in 0..64 {
        let root = tempdir(&format!("damage-{seed}"));
        fs::write(root.join("_1.si"), b"durable before the wrapper").unwrap();
        let dir = CrashingDirectory::new(&root, seed);
        write(&dir, "_2.fdt", b"0123456789");
        dir.sync_meta_data().unwrap();
        dir.delete_file("_1.si").unwrap();
        let report = dir.power_loss().unwrap();
        for (name, what) in &report.damaged {
            assert_eq!(name, "_2.fdt");
            outcomes.insert(*what);
            let after = fs::read(root.join("_2.fdt")).ok();
            match *what {
                "kept" => assert_eq!(after.as_deref(), Some(&b"0123456789"[..])),
                "zeroed" => assert_eq!(after, Some(vec![0; 10])),
                "truncated" => assert!(b"0123456789".starts_with(&after.unwrap())),
                _ => assert!(after.is_none()),
            }
        }
        if root.join("_1.si").exists() {
            assert_eq!(report.kept_events, 0);
            assert_eq!(
                fs::read(root.join("_1.si")).unwrap(),
                b"durable before the wrapper"
            );
            restored += 1;
        }
        fs::remove_dir_all(root).unwrap();
    }
    assert_eq!(outcomes.len(), 4, "{outcomes:?}");
    assert!(restored > 0);
}

#[test]
fn the_armed_call_and_everything_after_it_fail() {
    let root = tempdir("armed");
    let dir = CrashingDirectory::new(&root, 7);
    write(&dir, "a", b"x");
    dir.sync_meta_data().unwrap();
    dir.crash_after(2);
    dir.sync(&["a".to_string()]).unwrap();
    assert!(!dir.crashed());
    assert!(dir.rename("a", "b").is_err());
    assert!(dir.crashed());
    assert!(dir.list_all().is_err());
    assert!(dir.open("a").is_err());
    assert!(dir.create_output("c").is_err());
    assert!(dir.delete_file("a").is_err());
    assert!(dir.sync_meta_data().is_err());
    assert!(dir.sync(&[]).is_err());
    // The failed rename never reached the disk, and the power loss
    // disarms the crash.
    assert_eq!(dir.ops(), 4);
    dir.power_loss().unwrap();
    assert!(!dir.crashed());
    assert!(dir.list_all().unwrap().contains(&"a".to_string()));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn create_refuses_an_existing_name() {
    let root = tempdir("create-existing");
    fs::write(root.join("_0.si"), b"durable").unwrap();
    let dir = CrashingDirectory::new(&root, 1);
    let err = dir.create_output("_0.si").expect_err("refused");
    assert!(matches!(err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists));
    assert_eq!(fs::read(root.join("_0.si")).unwrap(), b"durable");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn undoing_a_delete_and_a_recreate_restores_the_durable_original() {
    // Delete a durable file, write a new one under the same name, lose
    // both: the original comes back whole and *durable* -- never damaged
    // as if it were the unsynced replacement.
    for seed in 0..32 {
        let root = tempdir(&format!("recreate-{seed}"));
        fs::write(root.join("f"), b"original").unwrap();
        let dir = CrashingDirectory::new(&root, seed);
        dir.delete_file("f").unwrap();
        write(&dir, "f", b"replacement");
        let report = dir.power_loss().unwrap();
        if report.kept_events == 0 {
            assert_eq!(
                fs::read(root.join("f")).unwrap(),
                b"original",
                "seed {seed}"
            );
            assert!(report.damaged.is_empty(), "{report:?}");
        }
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn a_lost_delete_of_an_unsynced_file_brings_it_back_damaged() {
    let mut outcomes = BTreeSet::new();
    for seed in 0..64 {
        let root = tempdir(&format!("unsynced-delete-{seed}"));
        let dir = CrashingDirectory::new(&root, seed);
        write(&dir, "tmp", b"0123456789");
        dir.sync_meta_data().unwrap();
        dir.delete_file("tmp").unwrap();
        let report = dir.power_loss().unwrap();
        if report.kept_events == 0 {
            // The name is back, and was put through the damage pass.
            assert_eq!(report.damaged.len(), 1, "{report:?}");
            outcomes.insert(report.damaged[0].1);
        } else {
            assert!(!root.join("tmp").exists());
        }
        fs::remove_dir_all(root).unwrap();
    }
    assert!(outcomes.len() > 1, "{outcomes:?}");
}

#[test]
fn undoing_a_rename_restores_the_file_it_overwrote() {
    for seed in 0..32 {
        let root = tempdir(&format!("overwrite-{seed}"));
        fs::write(root.join("dst"), b"old").unwrap();
        let dir = CrashingDirectory::new(&root, seed);
        write(&dir, "src", b"new");
        dir.sync(&["src".to_string()]).unwrap();
        dir.rename("src", "dst").unwrap();
        let report = dir.power_loss().unwrap();
        match report.kept_events {
            2 => assert_eq!(fs::read(root.join("dst")).unwrap(), b"new"),
            1 => {
                assert_eq!(fs::read(root.join("dst")).unwrap(), b"old");
                assert_eq!(fs::read(root.join("src")).unwrap(), b"new");
            }
            _ => {
                assert_eq!(fs::read(root.join("dst")).unwrap(), b"old");
                assert!(!root.join("src").exists());
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn publishes_and_the_crash_point_are_reported() {
    let root = tempdir("published");
    let dir = CrashingDirectory::new(&root, 3);
    write(&dir, "pending_segments_1", b"c");
    dir.rename("pending_segments_1", "segments_1").unwrap();
    dir.rename("segments_1", "other").unwrap();
    assert_eq!(dir.published(), 1);
    assert_eq!(dir.crash_point(), None);
    dir.crash_after(1);
    assert!(dir.delete_file("other").is_err());
    assert_eq!(dir.crash_point().as_deref(), Some("delete other"));
    fs::remove_dir_all(root).unwrap();
}
