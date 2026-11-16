#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::key_provider::StaticKeyProvider;
use lucene_store::ByteBuffersDirectory;

fn keys() -> Arc<dyn KeyProvider> {
    Arc::new(StaticKeyProvider::new("k1", &[1; 32]))
}

fn write(dir: &dyn Directory, name: &str, bytes: &[u8], piece: usize) -> u64 {
    let mut out = dir.create_output(name).unwrap();
    for p in bytes.chunks(piece.max(1)) {
        out.write_bytes(p);
    }
    if bytes.is_empty() {
        out.write_bytes(&[]);
    }
    assert_eq!(out.file_pointer(), bytes.len() as u64);
    let crc = out.checksum();
    assert_eq!(out.close().unwrap(), crc);
    crc
}

#[test]
fn the_lock_file_is_stored_as_is() {
    // The lock factory creates `write.lock` empty in the inner directory;
    // sizing the store lists it with every other file.
    let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
    dir.inner()
        .create_output(WRITE_LOCK_NAME)
        .unwrap()
        .close()
        .unwrap();
    assert_eq!(dir.file_length(WRITE_LOCK_NAME).unwrap(), 0);
    assert!(dir.open(WRITE_LOCK_NAME).unwrap().is_empty());
    for name in dir.list_all().unwrap() {
        dir.file_length(&name).unwrap();
    }
}

#[test]
fn files_round_trip_at_every_size_and_write_pattern() {
    let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys())
        .with_chunk_size(16)
        .unwrap();
    assert!(format!("{dir:?}").contains("chunk_size=16"));
    for len in [0usize, 1, 15, 16, 17, 32, 33, 100, 20_000] {
        for piece in [1usize, 7, 16, 64, 100_000] {
            let name = format!("f{len}_{piece}");
            let plain: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            let crc = write(&dir, &name, &plain, piece);
            // The checksum is the plaintext's, as Lucene's footers expect.
            assert_eq!(crc, crc32fast_hash(&plain));
            assert_eq!(&*dir.open(&name).unwrap(), &plain[..], "{name}");
            assert_eq!(dir.file_length(&name).unwrap(), len as u64, "{name}");
            let raw = dir.inner().open(&name).unwrap();
            assert!(raw.len() > len, "ciphertext carries a header and tags");
            if len >= 16 {
                assert!(!contains(&raw, &plain[..16]), "{name}: plaintext on disk");
            }
        }
    }
}

/// The CRC32 a plaintext output of `bytes` reports.
fn crc32fast_hash(bytes: &[u8]) -> u64 {
    let mut out = ByteBuffersDirectory::new().create_output("x").unwrap();
    out.write_bytes(bytes);
    out.checksum()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn rename_delete_and_temp_outputs_keep_lengths_right() {
    let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
    write(&dir, "pending_segments_1", b"commit bytes", 3);
    dir.sync(&["pending_segments_1".into()]).unwrap();
    dir.rename("pending_segments_1", "segments_1").unwrap();
    dir.sync_meta_data().unwrap();
    assert_eq!(dir.list_all().unwrap(), vec!["segments_1".to_string()]);
    assert_eq!(dir.file_length("segments_1").unwrap(), 12);
    assert_eq!(&*dir.open("segments_1").unwrap(), b"commit bytes");
    // A fresh wrapper over the same files learns lengths from headers.
    let again = EncryptedDirectory::new(dir.inner, keys());
    assert_eq!(again.file_length("segments_1").unwrap(), 12);
    again.delete_file("segments_1").unwrap();
    assert!(again
        .file_length("segments_1")
        .unwrap_err()
        .is_no_such_file());
    let mut t = again.create_temp_output("sort", "tmp").unwrap();
    t.write_bytes(b"abc");
    let tname = t.name().to_string();
    t.close().unwrap();
    assert_eq!(&*again.open(&tname).unwrap(), b"abc");
    let mut t = again
        .create_temp_output_with_estimate("sort", "tmp", Some(3))
        .unwrap();
    t.write_bytes(b"de");
    let tname = t.name().to_string();
    t.close().unwrap();
    assert_eq!(again.file_length(&tname).unwrap(), 2);
    let mut o = again.create_output_with_estimate("e", Some(1)).unwrap();
    o.write_byte(4);
    o.close().unwrap();
    assert_eq!(&*again.open("e").unwrap(), &[4]);
    assert!(again.pending_deletions().unwrap().is_empty());
    assert!(again.fs_directory_path().is_none());
    let lock = again.obtain_lock("write.lock").unwrap();
    assert!(again.obtain_lock("write.lock").is_err());
    drop(lock);
}

#[test]
fn plaintext_commit_metadata_keeps_only_commit_files_readable() {
    for name in [
        "segments_3",
        "pending_segments_4",
        "_0.si",
        "corrupted_x",
        "recovery.abc.segments_3",
        "recovery.abc._1.si",
    ] {
        assert!(is_commit_metadata(name), "{name}");
    }
    for name in [
        "_0.cfs",
        "_0_Lucene104_0.tim",
        "_0.fdt",
        "write.lock",
        "x_sort_1.tmp",
    ] {
        assert!(!is_commit_metadata(name), "{name}");
    }
    let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys())
        .with_plaintext_commit_metadata();
    write(&dir, "pending_segments_1", b"commit bytes", 5);
    write(&dir, "_0.si", b"segment info", 4);
    write(&dir, "_0.tim", b"secret terms", 4);
    let mut t = dir.create_temp_output("_0", "si").unwrap();
    t.write_bytes(b"tmp");
    let tname = t.name().to_string();
    t.close().unwrap();
    // A temp name ends in `.tmp`, so it is encrypted whatever its suffix.
    assert_eq!(&dir.inner().open(&tname).unwrap()[..4], b"LRSE");
    assert_eq!(&*dir.open(&tname).unwrap(), b"tmp");
    dir.rename("pending_segments_1", "segments_1").unwrap();
    assert_eq!(&*dir.inner().open("segments_1").unwrap(), b"commit bytes");
    assert_eq!(&*dir.inner().open("_0.si").unwrap(), b"segment info");
    assert_eq!(&dir.inner().open("_0.tim").unwrap()[..4], b"LRSE");
    assert_eq!(&*dir.open("segments_1").unwrap(), b"commit bytes");
    assert_eq!(dir.file_length("_0.si").unwrap(), 12);
    assert_eq!(dir.file_length("_0.tim").unwrap(), 12);
    assert_eq!(&*dir.open("_0.tim").unwrap(), b"secret terms");
    let err = dir.rename("_0.tim", "segments_9").unwrap_err();
    assert!(err.to_string().contains("plaintext"), "{err}");
    // Off by default: the same names are encrypted.
    let all = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
    write(&all, "segments_1", b"commit bytes", 5);
    assert_eq!(&all.inner().open("segments_1").unwrap()[..4], b"LRSE");
    all.rename("segments_1", "x").unwrap();
}

/// Two directories sharing a cache share one plaintext while an input
/// is alive; a delete, a rename or a rewrite of the name is never served
/// stale bytes, and nothing is kept once every input is gone.
#[test]
fn the_plaintext_cache_shares_live_inputs_and_forgets_changed_names() {
    let inner = Arc::new(ByteBuffersDirectory::new());
    let cache = Arc::new(PlaintextCache::new());
    let a = EncryptedDirectory::new(Arc::clone(&inner), keys())
        .with_plaintext_cache(Arc::clone(&cache));
    let b = EncryptedDirectory::new(Arc::clone(&inner), keys())
        .with_plaintext_cache(Arc::clone(&cache));
    write(&a, "f", b"first contents", 64);
    let one = a.open("f").unwrap();
    let two = b.open("f").unwrap();
    assert_eq!(
        one.as_ptr(),
        two.as_ptr(),
        "the second open shares the first"
    );
    drop((one, two));
    assert!(cache.get("f", inner.file_length("f").unwrap()).is_none());

    let held = a.open("f").unwrap();
    b.delete_file("f").unwrap();
    write(&b, "f", b"second, longer contents", 64);
    assert_eq!(&*a.open("f").unwrap(), b"second, longer contents");
    assert_eq!(&*held, b"first contents", "an open input keeps its bytes");

    let g = a.open("f").unwrap();
    a.rename("f", "g").unwrap();
    assert_eq!(&*b.open("g").unwrap(), b"second, longer contents");
    assert!(b.open("f").is_err());
    drop(g);

    // Replaced behind the directories' back, with a different length.
    let _keep = a.open("g").unwrap();
    let other = EncryptedDirectory::new(Arc::clone(&inner), keys());
    other.delete_file("g").unwrap();
    write(&other, "g", b"replaced", 64);
    assert_eq!(&*a.open("g").unwrap(), b"replaced");
}

#[test]
fn copy_from_reencrypts_and_a_plaintext_source_reads_back() {
    let plain = ByteBuffersDirectory::new();
    write(&plain, "a", b"hello", 5);
    let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
    dir.copy_from(&plain, "a", "b").unwrap();
    assert_eq!(&*dir.open("b").unwrap(), b"hello");
    // A plaintext file under the wrapper is refused, not misread.
    write(dir.inner(), "plain", b"not encrypted", 4);
    let err = dir.open("plain").unwrap_err().to_string();
    assert!(err.contains("plaintext index"), "{err}");
    assert!(dir.file_length("plain").is_err());
}

#[test]
fn the_wrong_key_or_a_tampered_file_fails_and_rotation_keeps_old_files() {
    let inner = Arc::new(ByteBuffersDirectory::new());
    let k1 = Arc::new(StaticKeyProvider::new("k1", &[1; 32]));
    let dir = EncryptedDirectory::new(Arc::clone(&inner), k1);
    write(&dir, "old", b"under k1", 8);

    let rotated = Arc::new(
        StaticKeyProvider::new("k1", &[1; 32])
            .with_key("k2", &[2; 32])
            .with_current("k2"),
    );
    let dir2 = EncryptedDirectory::new(Arc::clone(&inner), rotated);
    write(&dir2, "new", b"under k2", 8);
    assert_eq!(&*dir2.open("old").unwrap(), b"under k1");
    assert_eq!(&*dir2.open("new").unwrap(), b"under k2");
    let raw = inner.open("new").unwrap();
    assert!(contains(&raw, b"k2"), "the header names the key");

    // A provider without k2 cannot read the new file.
    let err = dir.open("new").unwrap_err();
    assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied));
    // A provider with a different k1 cannot unwrap the old one.
    let imposter = EncryptedDirectory::new(
        Arc::clone(&inner),
        Arc::new(StaticKeyProvider::new("k1", &[3; 32])),
    );
    assert!(matches!(imposter.open("old"), Err(Error::Corrupted(_))));

    // Flip one ciphertext byte: authentication fails.
    let mut bytes = raw.to_vec();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let mut out = inner.create_output("tampered").unwrap();
    out.write_bytes(&bytes);
    out.close().unwrap();
    let err = dir2.open("tampered").unwrap_err().to_string();
    assert!(err.contains("failed authentication"), "{err}");
    // Truncate it: its length no longer parses either.
    let mut out = inner.create_output("short").unwrap();
    out.write_bytes(&raw[..raw.len() - 17]);
    out.close().unwrap();
    assert!(dir2.open("short").is_err());
}

#[test]
fn chunk_size_is_checked() {
    let d = || EncryptedDirectory::new(ByteBuffersDirectory::new(), keys());
    assert!(d().with_chunk_size(0).is_err());
    assert!(d().with_chunk_size(MAX_CHUNK_SIZE + 1).is_err());
    assert!(d().with_chunk_size(MAX_CHUNK_SIZE).is_ok());
}

#[test]
fn a_key_provider_failure_fails_create_output() {
    let dir = EncryptedDirectory::new(
        ByteBuffersDirectory::new(),
        Arc::new(StaticKeyProvider::new("k", &[0; 32]).with_current("missing")),
    );
    assert!(dir.create_output("x").is_err());
}
