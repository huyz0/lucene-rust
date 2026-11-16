#![cfg(test)]
use super::*;
use crate::data_input::{DataInput, SliceInput};

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
/// One write op of the mixed-write property below.
#[derive(Debug, Clone)]
enum Op {
    Byte(u8),
    Bytes(Vec<u8>),
    VInt(i32),
    VLong(i64),
}

fn op() -> impl proptest::strategy::Strategy<Value = Op> {
    use proptest::prelude::*;
    prop_oneof![
        any::<u8>().prop_map(Op::Byte),
        proptest::collection::vec(any::<u8>(), 0..64).prop_map(Op::Bytes),
        // Writes around and past `CHUNK_SIZE`, which take the direct path.
        (CHUNK_SIZE - 3..CHUNK_SIZE * 2 + 5).prop_map(|n| Op::Bytes(vec![n as u8; n])),
        any::<i32>().prop_map(Op::VInt),
        (0i64..=i64::MAX).prop_map(Op::VLong),
    ]
}

fn apply(out: &mut FsIndexOutput, ops: &[Op], expect: &mut Vec<u8>) {
    for op in ops {
        match op {
            Op::Byte(b) => {
                out.write_byte(*b);
                expect.push(*b);
            }
            Op::Bytes(b) => {
                out.write_bytes(b);
                expect.extend_from_slice(b);
            }
            Op::VInt(v) => {
                out.write_vint(*v);
                expect.write_vint(*v);
            }
            Op::VLong(v) => {
                out.write_vlong(*v);
                expect.write_vlong(*v);
            }
        }
        assert_eq!(out.file_pointer(), expect.len() as u64);
    }
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]
    /// The lazy CRC and the in-buffer fast paths against the bytes
    /// written: for both sinks, any mix of single bytes, small and
    /// over-`CHUNK_SIZE` writes and varints yields the exact byte stream,
    /// file pointer and `crc32` of it, mid-stream and at close.
    #[test]
    fn mixed_writes_keep_bytes_pointer_and_checksum_exact(
        ops in proptest::collection::vec(op(), 0..80),
    ) {
        let dir = tempdir();
        let mut file_out = FsIndexOutput::create(&dir, "_0.bin").unwrap();
        let published = Arc::new(std::sync::Mutex::new(None));
        let sink = Arc::clone(&published);
        let mut mem_out = FsIndexOutput::in_memory(
            "_0.mem",
            Box::new(move |bytes| *sink.lock().unwrap() = Some(bytes)),
        );
        let mut expect = Vec::new();
        let half = ops.len() / 2;
        apply(&mut file_out, &ops[..half], &mut expect);
        let mut expect_mem = Vec::new();
        apply(&mut mem_out, &ops[..half], &mut expect_mem);
        proptest::prop_assert_eq!(file_out.checksum(), crc32fast::hash(&expect) as u64);
        proptest::prop_assert_eq!(mem_out.checksum(), crc32fast::hash(&expect_mem) as u64);
        apply(&mut file_out, &ops[half..], &mut expect);
        apply(&mut mem_out, &ops[half..], &mut expect_mem);
        proptest::prop_assert_eq!(file_out.close().unwrap(), crc32fast::hash(&expect) as u64);
        proptest::prop_assert_eq!(mem_out.close().unwrap(), crc32fast::hash(&expect_mem) as u64);
        proptest::prop_assert_eq!(std::fs::read(dir.join("_0.bin")).unwrap(), expect);
        proptest::prop_assert_eq!(published.lock().unwrap().take().unwrap(), expect_mem);
    }
}

fn tempdir() -> TempDir {
    TempDir::new("index-output")
}

#[test]
fn write_bytes_round_trips_and_tracks_file_pointer_and_checksum() {
    let dir = tempdir();
    let mut out = FsIndexOutput::create(&dir, "_0.test").unwrap();
    assert_eq!(out.name(), "_0.test");
    assert_eq!(out.file_pointer(), 0);

    out.write_vint(300);
    out.write_string("hello");
    assert_eq!(out.file_pointer(), 2 + 1 + 5);
    let checksum_before_close = out.checksum();
    let expected_crc = {
        let mut buf = Vec::new();
        buf.write_vint(300);
        buf.write_string("hello");
        crc32fast::hash(&buf) as u64
    };
    assert_eq!(checksum_before_close, expected_crc);

    let final_checksum = out.close().unwrap();
    assert_eq!(final_checksum, expected_crc);

    let bytes = std::fs::read(dir.join("_0.test")).unwrap();
    let mut input = SliceInput::new(&bytes);
    assert_eq!(input.read_vint().unwrap(), 300);
    assert_eq!(input.read_string().unwrap(), "hello");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn debug_names_the_sink_and_create_new_refuses_an_existing_file() {
    let dir = tempdir();
    let out = FsIndexOutput::create_new(&dir, "_0.test").unwrap();
    assert!(format!("{out:?}").starts_with("FSIndexOutput(path="));
    assert_eq!(out.path(), Some(dir.join("_0.test").as_path()));
    out.close().unwrap();
    let err = FsIndexOutput::create_new(&dir, "_0.test").unwrap_err();
    assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists));
    let mem = FsIndexOutput::in_memory("m", Box::new(|_| {}));
    assert_eq!(format!("{mem:?}"), "ByteBuffersIndexOutput(file=m)");
    assert_eq!(mem.path(), None);
}

#[test]
fn large_and_small_writes_interleave_in_order() {
    let dir = tempdir();
    let mut out = FsIndexOutput::create(&dir, "_0.test").unwrap();
    let big: Vec<u8> = (0..20_000u32).map(|i| i as u8).collect();
    out.write_bytes(b"head");
    out.write_bytes(&big);
    out.write_bytes(&vec![9u8; CHUNK_SIZE - 1]);
    out.write_bytes(&[8u8; 10]);
    out.write_byte(7);
    let checksum = out.close().unwrap();
    let mut want = b"head".to_vec();
    want.extend_from_slice(&big);
    want.extend(std::iter::repeat_n(9u8, CHUNK_SIZE - 1));
    want.extend(std::iter::repeat_n(8u8, 10));
    want.push(7);
    assert_eq!(std::fs::read(dir.join("_0.test")).unwrap(), want);
    assert_eq!(checksum, crc32fast::hash(&want) as u64);
}

#[test]
fn create_truncates_existing_file() {
    let dir = tempdir();
    write_all_bytes(&dir, "_0.test", b"0123456789").unwrap();
    write_all_bytes(&dir, "_0.test", b"ab").unwrap();
    assert_eq!(std::fs::read(dir.join("_0.test")).unwrap(), b"ab");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn create_output_nonexistent_dir_is_io_error() {
    let result = FsIndexOutput::create(Path::new("/nonexistent-lucene-rust-test-dir"), "x");
    assert!(matches!(result, Err(Error::Io(_))));
}

#[test]
fn write_after_pending_error_is_noop_and_close_surfaces_it() {
    // A closed underlying file descriptor is the simplest way to force a
    // write error deterministically without relying on disk-full/ENOSPC.
    let dir = tempdir();
    let path = dir.join("_0.test");
    let file = File::create(&path).unwrap();
    drop(file);
    // Re-open read-only, then wrap it directly bypassing `create` so the
    // first `write_all` fails with a real OS error (bad file descriptor
    // for writing).
    let ro_file = OpenOptions::new().read(true).open(&path).unwrap();
    let mut out = FsIndexOutput::over_file("_0.test", ro_file, path.clone());
    // Writing at least `CHUNK_SIZE` bytes goes straight to the file (a
    // real `write(2)` syscall), which fails on a read-only fd -- a single
    // `write_byte` would just sit in the buffer and never touch the OS.
    out.write_bytes(&vec![0u8; 9000]);
    assert!(out.pending_err.is_some());
    let fp_before = out.file_pointer();
    out.write_byte(2);
    assert_eq!(out.file_pointer(), fp_before, "no-op after pending error");
    assert!(matches!(out.close(), Err(Error::Io(_))));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sync_fsyncs_named_files_without_corrupting_them() {
    let dir = tempdir();
    write_all_bytes(&dir, "_0.test", b"payload-a").unwrap();
    write_all_bytes(&dir, "_1.test", b"payload-b").unwrap();
    sync(&dir, &["_0.test".to_string(), "_1.test".to_string()]).unwrap();
    assert_eq!(std::fs::read(dir.join("_0.test")).unwrap(), b"payload-a");
    assert_eq!(std::fs::read(dir.join("_1.test")).unwrap(), b"payload-b");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sync_missing_file_is_io_error() {
    let dir = tempdir();
    let result = sync(&dir, &["does-not-exist".to_string()]);
    assert!(matches!(result, Err(Error::Io(_))));
    std::fs::remove_dir_all(&dir).ok();
}

/// A [`RateLimiter`] that records each pause instead of sleeping.
struct Recording {
    pauses: std::sync::Mutex<Vec<u64>>,
    check: u64,
}

impl RateLimiter for Recording {
    fn set_mb_per_sec(&self, _mb_per_sec: f64) {}
    fn mb_per_sec(&self) -> f64 {
        1.0
    }
    fn pause(&self, bytes: u64) -> u64 {
        self.pauses.lock().unwrap().push(bytes);
        0
    }
    fn min_pause_check_bytes(&self) -> u64 {
        self.check
    }
}

/// `set_rate_limiter` behaves as `RateLimitedIndexOutput` does -- a check
/// before each write, a pause once more than `minPauseCheckBytes` built
/// up -- and changes nothing about the bytes, for a file and in memory.
#[test]
fn a_rate_limited_output_pauses_and_writes_the_same_bytes() {
    let dir = tempdir();
    let mem = crate::ByteBuffersDirectory::new();
    for mut out in [
        FsIndexOutput::create(&dir, "limited").unwrap(),
        crate::Directory::create_output(&mem, "limited").unwrap(),
    ] {
        let limiter = Arc::new(Recording {
            pauses: std::sync::Mutex::new(Vec::new()),
            check: 10,
        });
        out.set_rate_limiter(limiter.clone());
        for _ in 0..10 {
            out.write_byte(7);
        }
        assert!(limiter.pauses.lock().unwrap().is_empty());
        out.write_byte(7); // 11 > 10
        out.write_bytes(&[1u8; 100]);
        assert_eq!(*limiter.pauses.lock().unwrap(), vec![11, 100]);
        // A write larger than `CHUNK_SIZE` is checked per chunk.
        out.write_bytes(&[2u8; CHUNK_SIZE * 2 + 5]);
        assert_eq!(
            *limiter.pauses.lock().unwrap(),
            vec![11, 100, CHUNK_SIZE as u64, CHUNK_SIZE as u64]
        );
        assert_eq!(out.file_pointer(), 111 + CHUNK_SIZE as u64 * 2 + 5);
        let mut expected = vec![7u8; 11];
        expected.extend_from_slice(&[1u8; 100]);
        expected.extend_from_slice(&[2u8; CHUNK_SIZE * 2 + 5]);
        let mut crc = crc32fast::Hasher::new();
        crc.update(&expected);
        assert_eq!(out.close().unwrap(), u64::from(crc.finalize()));
    }
    assert_eq!(
        std::fs::read(dir.join("limited")).unwrap().len(),
        111 + CHUNK_SIZE * 2 + 5
    );
    assert_eq!(
        crate::Directory::file_length(&mem, "limited").unwrap(),
        111 + CHUNK_SIZE as u64 * 2 + 5
    );
}

/// An aborted limiter stops the output: the write that sees the abort
/// and every later one write nothing, and `close` fails.
#[test]
fn an_aborted_rate_limiter_fails_the_output() {
    struct Aborting(std::sync::atomic::AtomicBool);
    impl RateLimiter for Aborting {
        fn set_mb_per_sec(&self, _mb_per_sec: f64) {}
        fn mb_per_sec(&self) -> f64 {
            1.0
        }
        fn pause(&self, _bytes: u64) -> u64 {
            0
        }
        fn min_pause_check_bytes(&self) -> u64 {
            1
        }
        fn is_aborted(&self) -> bool {
            self.0.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    let dir = tempdir();
    let mut out = FsIndexOutput::create(&dir, "aborted").unwrap();
    let limiter = Arc::new(Aborting(std::sync::atomic::AtomicBool::new(false)));
    out.set_rate_limiter(limiter.clone());
    out.write_bytes(&[1u8; 10]);
    limiter.0.store(true, std::sync::atomic::Ordering::SeqCst);
    out.write_bytes(&[2u8; 10]);
    out.write_byte(3);
    assert_eq!(out.file_pointer(), 10);
    assert!(out.close().is_err());
}

/// Fails the `fail_at`-th write after keeping `keep` of its bytes, and
/// the close when `fail_close` is 1 (after the flush) or 2 (without it).
/// Counts the bytes `after_write` saw (`written`) and its own drop
/// (`dropped`).
struct Faulty {
    writes: usize,
    fail_at: usize,
    keep: usize,
    fail_close: u8,
    written: Arc<std::sync::atomic::AtomicUsize>,
    dropped: Arc<std::sync::atomic::AtomicUsize>,
}

impl OutputFaults for Faulty {
    fn before_write(&mut self, name: &str, len: usize) -> WriteVerdict {
        self.writes = self.writes.saturating_add(1);
        if self.writes == self.fail_at {
            return WriteVerdict::Fail {
                keep: self.keep.min(len),
                error: std::io::Error::other(format!("fake fault on {name}")),
            };
        }
        WriteVerdict::Write
    }
    fn after_write(&mut self, _name: &str, len: usize) {
        self.written
            .fetch_add(len, std::sync::atomic::Ordering::SeqCst);
    }
    fn on_close(&mut self, name: &str) -> CloseVerdict {
        let error = || std::io::Error::other(format!("fake close fault on {name}"));
        match self.fail_close {
            0 => CloseVerdict::Close,
            1 => CloseVerdict::FailAfterFlush(error()),
            _ => CloseVerdict::FailWithoutFlush(error()),
        }
    }
}

impl Drop for Faulty {
    fn drop(&mut self) {
        self.dropped
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

fn faulty(fail_at: usize, keep: usize, fail_close: u8) -> (Box<Faulty>, Counter, Counter) {
    let written = Counter::default();
    let dropped = Counter::default();
    let hook = Box::new(Faulty {
        writes: 0,
        fail_at,
        keep,
        fail_close,
        written: Arc::clone(&written),
        dropped: Arc::clone(&dropped),
    });
    (hook, written, dropped)
}

type Counter = Arc<std::sync::atomic::AtomicUsize>;

fn count(c: &Counter) -> usize {
    c.load(std::sync::atomic::Ordering::SeqCst)
}

/// A failed write keeps the bytes the hook allows, latches its error
/// (later writes are no-ops) and surfaces it from `close`; the hook is
/// dropped with the output.
#[test]
fn a_write_fault_keeps_its_prefix_and_surfaces_at_close() {
    let dir = tempdir();
    let (hook, written, dropped) = faulty(2, 3, 0);
    let mut out = FsIndexOutput::create(&dir, "f").unwrap();
    out.set_faults(hook);
    out.write_bytes(b"ab");
    assert_eq!(count(&written), 2, "after_write saw 2");
    out.write_bytes(b"cdefg");
    assert_eq!(out.file_pointer(), 5);
    out.write_bytes(b"zz");
    assert_eq!(out.file_pointer(), 5, "latched: no-op");
    let err = out.close().unwrap_err();
    assert!(err.to_string().contains("fake fault on f"), "{err}");
    assert_eq!(count(&written), 2, "the failed write is not after_write's");
    assert_eq!(count(&dropped), 1);
    // What was written before the fault is in the file, as in Java.
    assert_eq!(std::fs::read(dir.join("f")).unwrap(), b"abcde");
}

/// A close fault fails the close only after the bytes were flushed, for
/// a file and in memory; an output dropped unclosed drops its hook too.
#[test]
fn a_close_fault_fails_after_the_flush() {
    let dir = tempdir();
    let (hook, written, dropped) = faulty(0, 0, 1);
    let mut out = FsIndexOutput::create(&dir, "g").unwrap();
    out.set_faults(hook);
    out.write_bytes(b"hello");
    let err = out.close().unwrap_err();
    assert!(err.to_string().contains("fake close fault on g"), "{err}");
    assert_eq!(std::fs::read(dir.join("g")).unwrap(), b"hello");
    assert_eq!(count(&written), 5);
    assert_eq!(count(&dropped), 1);

    // Force-closed: the buffered bytes never reach the file.
    let (hook, _, _) = faulty(0, 0, 2);
    let mut out = FsIndexOutput::create(&dir, "k").unwrap();
    out.set_faults(hook);
    out.write_bytes(b"lost");
    assert!(out.close().is_err());
    assert_eq!(std::fs::read(dir.join("k")).unwrap(), b"");

    let mem = crate::ByteBuffersDirectory::new();
    let (hook, _, _) = faulty(0, 0, 0);
    let mut out = crate::Directory::create_output(&mem, "m").unwrap();
    out.set_faults(hook);
    out.write_bytes(b"xy");
    out.close().unwrap();
    assert_eq!(crate::Directory::file_length(&mem, "m").unwrap(), 2);

    let (hook, _, dropped) = faulty(0, 0, 0);
    let mut out = FsIndexOutput::create(&dir, "h").unwrap();
    out.set_faults(hook);
    drop(out);
    assert_eq!(count(&dropped), 1);
}
