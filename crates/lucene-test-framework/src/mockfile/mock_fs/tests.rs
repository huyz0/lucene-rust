#![cfg(test)]
use super::*;
use lucene_util::test_support::TempDir;

#[test]
fn random_draws_in_java_order() {
    // Replays the draws by hand on a twin stream.
    for seed in 0..200 {
        let mut r = JavaRandom::new(seed);
        let config = MockFsConfig::random(&mut r);
        let mut t = JavaRandom::new(seed);
        assert_eq!(config.disable_fsync, t.next_int_bounded(100) > 0);
        if t.next_int_bounded(100) > 0 {
            assert_eq!(config.shuffle_seed, Some(t.next_long()));
        } else {
            assert_eq!(config.shuffle_seed, None);
        }
        if t.next_int_bounded(10) > 0 {
            assert!(config.leak);
            assert_eq!(config.handle_limit, Some(MAX_OPEN_HANDLES));
            assert_eq!(config.windows, t.next_int_bounded(10) == 0);
            let active = t.next_int_bounded(4) == 0;
            let dir = t.next_boolean();
            assert_eq!(config.extras, active.then_some(dir));
        } else {
            assert_eq!(
                config,
                MockFsConfig {
                    disable_fsync: config.disable_fsync,
                    shuffle_seed: config.shuffle_seed,
                    ..Default::default()
                }
            );
        }
        assert_eq!(r.next_long(), t.next_long(), "same number of draws");
    }
    let mut r = JavaRandom::new(1);
    let salt = JavaRandom::new(1).next_long();
    assert_eq!(
        MockFsConfig::default()
            .with_virus_checker(&mut r)
            .virus_checker,
        Some(salt)
    );
}

#[test]
fn extras_create_a_file_or_a_directory() {
    let tmp = TempDir::new("mockfs-extras");
    for (extras, want_dir) in [
        (Some(false), Some(false)),
        (Some(true), Some(true)),
        (None, None),
    ] {
        let fs = MockFs::new(MockFsConfig {
            extras,
            ..Default::default()
        });
        let dir = tmp.join(format!("{extras:?}"));
        fs.create_directory(&dir).unwrap();
        let extra = dir.join(EXTRA_FILE_NAME);
        assert_eq!(extra.exists().then(|| extra.is_dir()), want_dir);
    }
    assert!(MockFs::is_extra("extra0"));
    assert!(!MockFs::is_extra("_0.si"));
}

#[test]
fn handle_limit_and_leaks() {
    let fs = MockFs::new(MockFsConfig {
        handle_limit: Some(2),
        leak: true,
        ..Default::default()
    });
    fs.on_open("a").unwrap();
    fs.on_open("a").unwrap();
    let err = fs.on_open("b").unwrap_err();
    assert!(err.to_string().contains("Too many open files"), "{err}");
    assert_eq!(fs.open_handles(), BTreeMap::from([("a".to_string(), 2)]));
    assert!(fs.check_leaks().unwrap_err().contains("\"a\""));
    fs.on_close("a");
    fs.on_close("a");
    fs.on_close("never-opened");
    assert!(fs.open_handles().is_empty());
    fs.check_leaks().unwrap();
    // Without LeakFS an open handle is not a failure.
    let quiet = MockFs::new(MockFsConfig::default());
    quiet.on_open("x").unwrap();
    quiet.check_leaks().unwrap();
    assert!(!quiet.fsync_disabled());
    assert!(MockFs::new(MockFsConfig {
        disable_fsync: true,
        ..Default::default()
    })
    .fsync_disabled());
}

#[test]
fn windows_refuses_deleting_or_moving_an_open_file() {
    let tmp = TempDir::new("mockfs-windows");
    let fs = MockFs::new(MockFsConfig {
        windows: true,
        ..Default::default()
    });
    let open = tmp.join("open");
    std::fs::write(&open, b"x").unwrap();
    fs.check_delete(&open).unwrap();
    fs.on_open("open").unwrap();
    let err = fs.check_delete(&open).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
    assert!(fs.check_rename(&open, &tmp.join("other")).is_err());
    // A missing file is never refused: its NoSuchFile must not be delayed.
    fs.on_open("missing").unwrap();
    fs.check_delete(&tmp.join("missing")).unwrap();
    fs.on_close("open");
    fs.check_delete(&open).unwrap();
    fs.check_rename(&open, &tmp.join("other")).unwrap();
    assert!(fs.config().windows);
}

/// The virus checker refuses exactly where Java's hash says it does,
/// never on `write.lock` or a missing file, and not while disabled.
#[test]
fn virus_checker_follows_javas_hash() {
    let tmp = TempDir::new("mockfs-virus");
    let salt = 12345i64;
    let fs = MockFs::new(MockFsConfig {
        virus_checker: Some(salt),
        ..Default::default()
    });
    assert!(fs.has_virus_checker());
    // Printed by Java running `VirusCheckingFS.delete`'s formula for
    // this salt and name sequence. `UnixPath.hashCode` is the name's
    // `String.hashCode` for ASCII names; `WindowsPath` hashes them
    // case-insensitively, so on Windows Java refuses other deletes.
    const JAVA_REFUSED: [i32; 19] = [
        72, 80, 83, 85, 96, 122, 127, 130, 135, 138, 141, 146, 172, 177, 180, 183, 185, 188, 191,
    ];
    let mut state = salt;
    let mut refused = 0;
    for i in 0..200 {
        let name = if i % 7 == 0 {
            WRITE_LOCK_NAME.to_string()
        } else {
            format!("_{i}.cfs")
        };
        let path = tmp.join(&name);
        if i % 11 != 0 {
            std::fs::write(&path, b"x").unwrap();
        }
        state += 1;
        let hash = state.wrapping_mul(i64::from(java_string_hash(&name)));
        let want = path.exists() && name != WRITE_LOCK_NAME && hash % 5 == 1;
        assert_eq!(want, JAVA_REFUSED.contains(&i), "{name}");
        let got = fs.check_delete(&path);
        assert_eq!(got.is_err(), want, "{name}");
        if let Err(e) = got {
            refused += 1;
            assert!(e.to_string().contains("VirusCheckingFS"), "{e}");
        }
    }
    assert_eq!(refused, JAVA_REFUSED.len());
    assert!(fs.disable_virus_checker());
    assert!(!fs.disable_virus_checker(), "already disabled");
    for i in 0..50 {
        let path = tmp.join(format!("_{i}.cfs"));
        std::fs::write(&path, b"x").unwrap();
        fs.check_delete(&path).unwrap();
    }
    fs.enable_virus_checker();
    assert!(!MockFs::new(MockFsConfig::default()).disable_virus_checker());
}
