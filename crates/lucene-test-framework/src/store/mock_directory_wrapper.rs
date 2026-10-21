//! `MockDirectoryWrapper` (with `MockIndexOutputWrapper`,
//! `MockIndexInputWrapper` and the slow opening/closing input wrappers): a
//! directory that injects I/O faults, emulates a crash, tracks open files
//! and checks the index when it is closed.
//!
//! What it does, as Java's does:
//!
//! - **Random I/O errors**: [`MockDirectoryWrapper::set_random_io_exception_rate`]
//!   fails the first write to a new file, and each file of a `sync`, with
//!   that probability; [`MockDirectoryWrapper::set_random_io_exception_rate_on_open`]
//!   fails `open`/`create_output`/`create_temp_output` (as an `IOException`,
//!   or -- unless [`MockDirectoryWrapper::set_allow_random_file_not_found_exception`]
//!   is off -- a `FileNotFoundException`/`NoSuchFileException`).
//! - **Deterministic failures**: every [`Failure`] registered with
//!   [`MockDirectoryWrapper::fail_on`] is evaluated at every failure point
//!   (creating, writing and closing an output, opening an input, sync,
//!   rename, delete, syncMetaData) and may fail it.
//! - **A full disk**: [`MockDirectoryWrapper::set_max_size_in_bytes`] writes
//!   what still fits, then fails the write ("fake disk full").
//! - **Crash**: [`MockDirectoryWrapper::crash`] damages every file written
//!   since its last `sync` (deleted, zeroed, half or fully truncated, one
//!   bit flipped, or left alone) and fails every later call until
//!   [`MockDirectoryWrapper::clear_crash`].
//! - **Open files**: an output and an input ([`Input::Tracked`]) count as
//!   open until dropped. A file cannot be created twice, read while it is
//!   still being written, or -- with [`MockDirectoryWrapper::set_assert_no_delete_open_file`]
//!   -- deleted, renamed or overwritten while open. Closing the directory
//!   with a file still open fails.
//! - **Close**: [`BaseDirectoryWrapper::close`] turns the fault rates off,
//!   crashes the directory (so a file the index needs but never synced is
//!   damaged) and runs `CheckIndex`.
//! - **Throttling**: `Throttling::Always` (and one output in 200 under
//!   `Sometimes`) slows outputs as `ThrottledIndexOutput` does.
//!
//! Differs, because the Rust shapes differ:
//!
//! - Faults reach an output through lucene-store's `OutputFaults` hook, and
//!   an output's writes are infallible by signature, so a write fault is
//!   latched and returned by the output's `close` (the first error wins), not
//!   thrown by the write. A fault after the write (a [`Failure`], the random
//!   exception) still lets the bytes through, as Java's does.
//! - An input is a byte buffer: there are no reads, clones or slices to
//!   intercept, so `setVerboseClone`/`getInputCloneCount` and the
//!   read-side checks of `MockIndexInputWrapper` are not ported; only its
//!   open/close bookkeeping is.
//! - There is no `IOContext`: `newIOContext`'s randomization and the
//!   "segments file opened without READONCE" check are not ported.
//! - [`Failure::eval`] sees the operation and file name
//!   ([`FailureContext`]) instead of a stack trace to search.
//! - `crash` cannot force-close the handles a caller holds: an output still
//!   open fails its later writes and its close (without flushing), even
//!   after `clear_crash`, as Java's closed wrapper does. Its file never got
//!   all its bytes, so whatever fate is drawn for it, it is deleted (Java
//!   flushes it first and then damages it).
//! - Not ported (listed in the test-framework ledger's gap):
//!   `corruptUnknownFiles`, `setAssertNoUnrefencedFilesOnClose` (it needs
//!   an `IndexWriter` over the directory) and `getLevelForCheckOnClose`.
//! - The seeding is Java's (`new Random(random.nextInt())`), but the draw
//!   sites differ: no `newIOContext` draws, the crash's damage deletes and
//!   lengths through the delegate (no `maybeYield`, no `Failure`), and under
//!   a rate limiter the write hook runs once per chunk.
//! - Each `newDirectory` has its own [`MockFs`] (Java shares one
//!   filesystem across the suite), so the handle limit counts per directory,
//!   and `close` checks it for leaked handles (Java's `LeakFS` is checked at
//!   suite end by `TestRuleTemporaryFilesCleanup`).
//! - `getFileHandleCount` counts the handles this wrapper has seen open,
//!   which is every output and every input it handed out.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lucene_store::directory::TrackedInput;
use lucene_store::index_output::{CloseVerdict, FsIndexOutput, OutputFaults, WriteVerdict};
use lucene_store::lock::Lock;
use lucene_store::{DataOutput, Directory, Error, IndexOutput, Input, Result};
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

use super::base_directory_wrapper::{index_exists, BaseDirectoryWrapper};
use crate::mockfile::mock_fs::MockFs;

/// The message a [`Failure`]'s [`fake_io_exception`] starts with, so a test
/// can tell one from any other error ([`is_fake_io_exception`]).
pub const FAKE_IO_EXCEPTION: &str = "MockDirectoryWrapper.FakeIOException";

/// The message the random-rate and disk-full errors start with. Java throws
/// plain `IOException`s for those, which are not `FakeIOException`s;
/// [`is_injected_io_exception`] accepts both kinds.
pub const RANDOM_IO_EXCEPTION: &str = "MockDirectoryWrapper.IOException";

/// `MockDirectoryWrapper.FakeIOException`: an error to return from a
/// [`Failure`].
pub fn fake_io_exception(detail: &str) -> io::Error {
    io::Error::other(format!("{FAKE_IO_EXCEPTION}: {detail}"))
}

/// Whether `e` is (or wraps) a [`fake_io_exception`].
pub fn is_fake_io_exception(e: &Error) -> bool {
    e.to_string().contains(FAKE_IO_EXCEPTION)
}

/// Whether `e` came from any fault this wrapper injects: a [`Failure`]'s
/// [`fake_io_exception`], a random-rate error or a fake disk full.
pub fn is_injected_io_exception(e: &dyn std::fmt::Display) -> bool {
    let text = e.to_string();
    text.contains(FAKE_IO_EXCEPTION) || text.contains(RANDOM_IO_EXCEPTION)
}

/// `MockDirectoryWrapper.Throttling`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Throttling {
    /// Always emulate a slow disk.
    Always,
    /// Emulate a slow disk for one output in 200.
    Sometimes,
    /// Never throttle.
    Never,
}

/// A failure point: which operation is about to happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureOp {
    /// `createOutput`.
    CreateOutput,
    /// `createTempOutput`.
    CreateTempOutput,
    /// A write to an output (after the bytes went through).
    Write,
    /// An output's `close`.
    CloseOutput,
    /// `openInput`.
    OpenInput,
    /// `sync`.
    Sync,
    /// `rename` (the name is the source).
    Rename,
    /// `deleteFile`.
    DeleteFile,
    /// `syncMetaData` (no file name).
    SyncMetaData,
}

/// Where a [`Failure`] is being evaluated.
#[derive(Debug, Clone, Copy)]
pub struct FailureContext<'a> {
    /// The operation.
    pub op: FailureOp,
    /// The file it acts on (empty for [`FailureOp::SyncMetaData`]).
    pub name: &'a str,
}

/// `MockDirectoryWrapper.Failure`: evaluated at every failure point; an
/// error fails the operation.
pub trait Failure: Send + Sync {
    /// `eval(dir)`.
    fn eval(&self, ctx: &FailureContext<'_>) -> io::Result<()>;
}

/// A [`Failure`] that fails every operation `predicate` accepts while
/// armed: `setDoFail`/`clearDoFail`, plus a count of how often it fired.
/// Armed from the start; [`FailOn::once`] disarms itself after firing.
pub struct FailOn {
    predicate: Box<dyn Fn(&FailureContext<'_>) -> bool + Send + Sync>,
    do_fail: AtomicBool,
    once: bool,
    fired: AtomicUsize,
}

impl FailOn {
    /// Fails every accepted operation while armed.
    pub fn new(
        predicate: impl Fn(&FailureContext<'_>) -> bool + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            predicate: Box::new(predicate),
            do_fail: AtomicBool::new(true),
            once: false,
            fired: AtomicUsize::new(0),
        })
    }

    /// Fails the first accepted operation, then disarms.
    pub fn once(
        predicate: impl Fn(&FailureContext<'_>) -> bool + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            predicate: Box::new(predicate),
            do_fail: AtomicBool::new(true),
            once: true,
            fired: AtomicUsize::new(0),
        })
    }

    /// `setDoFail()`.
    pub fn set_do_fail(&self) {
        self.do_fail.store(true, Ordering::SeqCst);
    }

    /// `clearDoFail()`.
    pub fn clear_do_fail(&self) {
        self.do_fail.store(false, Ordering::SeqCst);
    }

    /// How many operations this failed.
    pub fn fired(&self) -> usize {
        self.fired.load(Ordering::SeqCst)
    }
}

impl Failure for FailOn {
    fn eval(&self, ctx: &FailureContext<'_>) -> io::Result<()> {
        if self.do_fail.load(Ordering::SeqCst) && (self.predicate)(ctx) {
            if self.once {
                self.clear_do_fail();
            }
            self.fired.fetch_add(1, Ordering::SeqCst);
            return Err(fake_io_exception(&format!("{:?} {}", ctx.op, ctx.name)));
        }
        Ok(())
    }
}

/// What a handle is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Handle {
    Input,
    Output,
}

/// `ThrottledIndexOutput.mBitsToBytes`: Java's `int` multiply, overflow
/// included (40 Mbit already wraps).
fn m_bits_to_bytes(mbits: i32) -> i32 {
    mbits.wrapping_mul(125_000_000)
}

/// `ThrottledIndexOutput`'s state on one output.
#[derive(Debug, Clone)]
struct Throttle {
    bytes_per_second: i64,
    delay_millis: i64,
    min_bytes_written: i64,
    pending_bytes: i64,
    time_elapsed_nanos: i64,
    write_started: Option<Instant>,
}

impl Throttle {
    /// `ThrottledIndexOutput.DEFAULT_MIN_WRITTEN_BYTES`.
    const DEFAULT_MIN_WRITTEN_BYTES: i64 = 1024;

    fn new(bytes_per_second: i32, delay_millis: i64) -> Self {
        Self {
            bytes_per_second: i64::from(bytes_per_second),
            delay_millis,
            min_bytes_written: Self::DEFAULT_MIN_WRITTEN_BYTES,
            pending_bytes: 0,
            time_elapsed_nanos: 0,
            write_started: None,
        }
    }

    /// `ThrottledIndexOutput.getDelay(closing)`.
    fn delay(&mut self, closing: bool) -> i64 {
        if self.pending_bytes > 0 && (closing || self.pending_bytes > self.min_bytes_written) {
            let actual_bps =
                (self.time_elapsed_nanos / self.pending_bytes).saturating_mul(1_000_000_000);
            if actual_bps > self.bytes_per_second {
                let expected =
                    self.pending_bytes.saturating_mul(1000) / self.bytes_per_second.max(1);
                let delay = expected - self.time_elapsed_nanos / 1_000_000;
                self.pending_bytes = 0;
                self.time_elapsed_nanos = 0;
                return delay;
            }
        }
        0
    }

    fn sleep(ms: i64) {
        if ms > 0 {
            std::thread::sleep(Duration::from_millis(ms as u64));
        }
    }
}

struct State {
    random: JavaRandom,
    max_size: u64,
    max_used_size: u64,
    random_io_exception_rate: f64,
    random_io_exception_rate_on_open: f64,
    assert_no_delete_open_file: bool,
    track_disk_usage: bool,
    use_slow_open_closers: bool,
    allow_random_file_not_found_exception: bool,
    allow_reading_files_still_open_for_write: bool,
    unsynced_files: BTreeSet<String>,
    created_files: BTreeSet<String>,
    open_files_for_write: BTreeSet<String>,
    /// `openFiles`: open handles per name.
    open_files: BTreeMap<String, usize>,
    /// `openFilesDeleted`.
    open_files_deleted: BTreeSet<String>,
    /// `openFileHandles`: every open handle, by id.
    open_handles: BTreeMap<u64, (String, Handle)>,
    next_handle: u64,
    crashed: bool,
    throttling: Throttling,
    throttle: (i32, i64),
    always_corrupt: bool,
    fail_on_create_output: bool,
    fail_on_open_input: bool,
    failures: Vec<Arc<dyn Failure>>,
    check_index_on_close: bool,
    is_open: bool,
    /// How many inputs opened slowly, closed slowly, and outputs were
    /// throttled: what shows the knobs fired.
    slow_opens: usize,
    slow_closes: usize,
    throttled_outputs: usize,
}

struct Shared {
    inner: Arc<dyn Directory>,
    state: Mutex<State>,
    fs: Option<Arc<MockFs>>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `maybeThrowDeterministicException()`: every registered failure, in
    /// order, evaluated without the lock (a failure may be shared and
    /// slow).
    fn maybe_throw_deterministic_exception(&self, op: FailureOp, name: &str) -> io::Result<()> {
        let failures = self.state().failures.clone();
        let ctx = FailureContext { op, name };
        for failure in failures {
            failure.eval(&ctx)?;
        }
        Ok(())
    }

    /// `sizeInBytes()`, extras excluded.
    fn size_in_bytes(&self) -> Result<u64> {
        let mut size = 0u64;
        for file in self.inner.list_all()? {
            if !file.starts_with("extra") {
                size = size.saturating_add(self.inner.file_length(&file).unwrap_or(0));
            }
        }
        Ok(size)
    }

    /// `removeOpenFile(c, name)`.
    fn remove_open_file(&self, id: u64) {
        let mut st = self.state();
        let Some((name, kind)) = st.open_handles.remove(&id) else {
            return; // crash() forgot it
        };
        if kind == Handle::Output {
            st.open_files_for_write.remove(&name);
        }
        if let Some(n) = st.open_files.get_mut(&name) {
            *n -= 1;
            if *n == 0 {
                st.open_files.remove(&name);
            }
        }
        drop(st);
        if let Some(fs) = &self.fs {
            fs.on_close(&name);
        }
    }
}

/// `MockDirectoryWrapper`: see the module docs.
#[derive(Clone)]
pub struct MockDirectoryWrapper {
    shared: Arc<Shared>,
    /// The scratch directories under the delegate, removed with the last
    /// clone.
    _temps: Arc<Vec<TempDir>>,
}

impl std::fmt::Debug for MockDirectoryWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.shared.state();
        if st.max_size != 0 {
            write!(
                f,
                "MockDirectoryWrapper(current={},max={})",
                st.max_used_size, st.max_size
            )
        } else {
            write!(f, "MockDirectoryWrapper")
        }
    }
}

fn io_error(message: String) -> Error {
    Error::Io(io::Error::other(message))
}

impl MockDirectoryWrapper {
    /// `new MockDirectoryWrapper(random, delegate)`: a private stream seeded
    /// from `random.nextInt()`, then the throttle's rate and delay drawn
    /// from it.
    pub fn new(random: &mut JavaRandom, delegate: impl Directory + 'static) -> Self {
        Self::over(random, Arc::new(delegate), None, Vec::new())
    }

    /// [`Self::new`] over a delegate with `fs` under it: the mock reports
    /// its handles to `fs` (`HandleLimitFS`, `WindowsFS`, `LeakFS`) and
    /// follows its fsync setting. `fs` should also be the delegate's
    /// `FsFaults`, for its deletes and renames.
    pub fn with_mock_fs(
        random: &mut JavaRandom,
        delegate: impl Directory + 'static,
        fs: Arc<MockFs>,
    ) -> Self {
        Self::over(random, Arc::new(delegate), Some(fs), Vec::new())
    }

    /// Over a shared delegate, with the mock filesystem under it (if any)
    /// and the scratch directories to keep alive.
    pub(crate) fn over(
        random: &mut JavaRandom,
        inner: Arc<dyn Directory>,
        fs: Option<Arc<MockFs>>,
        temps: Vec<TempDir>,
    ) -> Self {
        let mut state_random = JavaRandom::new(i64::from(random.next_int()));
        let bytes_per_second = m_bits_to_bytes(40 + state_random.next_int_bounded(10));
        let delay = i64::from(1 + state_random.next_int_bounded(5));
        let state = State {
            random: state_random,
            max_size: 0,
            max_used_size: 0,
            random_io_exception_rate: 0.0,
            random_io_exception_rate_on_open: 0.0,
            assert_no_delete_open_file: false,
            track_disk_usage: false,
            use_slow_open_closers: false,
            allow_random_file_not_found_exception: true,
            allow_reading_files_still_open_for_write: false,
            unsynced_files: BTreeSet::new(),
            created_files: BTreeSet::new(),
            open_files_for_write: BTreeSet::new(),
            open_files: BTreeMap::new(),
            open_files_deleted: BTreeSet::new(),
            open_handles: BTreeMap::new(),
            next_handle: 0,
            crashed: false,
            throttling: Throttling::Never,
            throttle: (bytes_per_second, delay),
            always_corrupt: false,
            fail_on_create_output: true,
            fail_on_open_input: true,
            failures: Vec::new(),
            check_index_on_close: true,
            is_open: true,
            slow_opens: 0,
            slow_closes: 0,
            throttled_outputs: 0,
        };
        Self {
            shared: Arc::new(Shared {
                inner,
                state: Mutex::new(state),
                fs,
            }),
            _temps: Arc::new(temps),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.shared.state()
    }

    /// Keeps `temp` alive as long as this wrapper (called before it is
    /// cloned).
    pub(crate) fn keep_alive(&mut self, temp: TempDir) {
        Arc::get_mut(&mut self._temps)
            .expect("keep_alive before the wrapper is shared")
            .push(temp);
    }

    /// `FilterDirectory.getDelegate()`.
    pub fn delegate(&self) -> &dyn Directory {
        self.shared.inner.as_ref()
    }

    /// The mock filesystem under the delegate, if it has one.
    pub fn mock_fs(&self) -> Option<&Arc<MockFs>> {
        self.shared.fs.as_ref()
    }

    /// `setTrackDiskUsage(v)`.
    pub fn set_track_disk_usage(&self, v: bool) {
        self.state().track_disk_usage = v;
    }

    /// `setAllowRandomFileNotFoundException(value)`.
    pub fn set_allow_random_file_not_found_exception(&self, value: bool) {
        self.state().allow_random_file_not_found_exception = value;
    }

    /// `setAllowReadingFilesStillOpenForWrite(value)`.
    pub fn set_allow_reading_files_still_open_for_write(&self, value: bool) {
        self.state().allow_reading_files_still_open_for_write = value;
    }

    /// `setThrottling(throttling)`.
    pub fn set_throttling(&self, throttling: Throttling) {
        self.state().throttling = throttling;
    }

    /// `setUseSlowOpenClosers(v)`: one input in 500 sleeps 50 ms as it
    /// opens, one in 500 as it closes.
    pub fn set_use_slow_open_closers(&self, v: bool) {
        self.state().use_slow_open_closers = v;
    }

    /// `setMaxSizeInBytes(maxSize)`; 0 is unlimited.
    pub fn set_max_size_in_bytes(&self, max_size: u64) {
        self.state().max_size = max_size;
    }

    /// `getMaxSizeInBytes()`.
    pub fn max_size_in_bytes(&self) -> u64 {
        self.state().max_size
    }

    /// `getMaxUsedSizeInBytes()`: the peak tracked size.
    pub fn max_used_size_in_bytes(&self) -> u64 {
        self.state().max_used_size
    }

    /// `resetMaxUsedSizeInBytes()`.
    pub fn reset_max_used_size_in_bytes(&self) -> Result<()> {
        let size = self.shared.size_in_bytes()?;
        self.state().max_used_size = size;
        Ok(())
    }

    /// `sizeInBytes()`: the delegate's files, extras excluded.
    pub fn size_in_bytes(&self) -> Result<u64> {
        self.shared.size_in_bytes()
    }

    /// `setAssertNoDeleteOpenFile(value)`.
    pub fn set_assert_no_delete_open_file(&self, value: bool) {
        self.state().assert_no_delete_open_file = value;
    }

    /// `getAssertNoDeleteOpenFile()`.
    pub fn assert_no_delete_open_file(&self) -> bool {
        self.state().assert_no_delete_open_file
    }

    /// `setRandomIOExceptionRate(rate)`.
    pub fn set_random_io_exception_rate(&self, rate: f64) {
        self.state().random_io_exception_rate = rate;
    }

    /// `getRandomIOExceptionRate()`.
    pub fn random_io_exception_rate(&self) -> f64 {
        self.state().random_io_exception_rate
    }

    /// `setRandomIOExceptionRateOnOpen(rate)`.
    pub fn set_random_io_exception_rate_on_open(&self, rate: f64) {
        self.state().random_io_exception_rate_on_open = rate;
    }

    /// `getRandomIOExceptionRateOnOpen()`.
    pub fn random_io_exception_rate_on_open(&self) -> f64 {
        self.state().random_io_exception_rate_on_open
    }

    /// `setFailOnCreateOutput(v)`.
    pub fn set_fail_on_create_output(&self, v: bool) {
        self.state().fail_on_create_output = v;
    }

    /// `setFailOnOpenInput(v)`.
    pub fn set_fail_on_open_input(&self, v: bool) {
        self.state().fail_on_open_input = v;
    }

    /// The `alwaysCorrupt` test field: a crash never leaves a file intact.
    pub fn set_always_corrupt(&self, v: bool) {
        self.state().always_corrupt = v;
    }

    /// `failOn(fail)`.
    pub fn fail_on(&self, fail: Arc<dyn Failure>) {
        self.state().failures.push(fail);
    }

    /// `getFileHandleCount()`.
    pub fn file_handle_count(&self) -> usize {
        self.state().open_handles.len()
    }

    /// `getOpenDeletedFiles()`.
    pub fn open_deleted_files(&self) -> BTreeSet<String> {
        self.state().open_files_deleted.clone()
    }

    /// The files written but not yet synced (`unSyncedFiles`).
    pub fn unsynced_files(&self) -> BTreeSet<String> {
        self.state().unsynced_files.clone()
    }

    /// How many inputs were opened slowly and closed slowly
    /// ([`Self::set_use_slow_open_closers`]), and how many outputs were
    /// throttled ([`Self::set_throttling`]).
    pub fn slow_and_throttled_counts(&self) -> (usize, usize, usize) {
        let st = self.state();
        (st.slow_opens, st.slow_closes, st.throttled_outputs)
    }

    /// Whether [`Self::crash`] ran and [`Self::clear_crash`] has not.
    pub fn crashed(&self) -> bool {
        self.state().crashed
    }

    /// `maybeYield()`.
    fn maybe_yield(&self) {
        if self.state().random.next_boolean() {
            std::thread::yield_now();
        }
    }

    /// `maybeThrowIOException(message)`.
    fn maybe_throw_io_exception(st: &mut State, message: &str) -> io::Result<()> {
        if st.random.next_double() < st.random_io_exception_rate {
            return Err(io::Error::other(format!(
                "{RANDOM_IO_EXCEPTION}: a random IOException ({message})"
            )));
        }
        Ok(())
    }

    /// `maybeThrowIOExceptionOnOpen(name)`.
    fn maybe_throw_io_exception_on_open(&self, name: &str) -> Result<()> {
        let mut st = self.state();
        if st.random.next_double() < st.random_io_exception_rate_on_open {
            let message = format!("{RANDOM_IO_EXCEPTION}: a random IOException ({name})");
            if !st.allow_random_file_not_found_exception || st.random.next_boolean() {
                return Err(io_error(message));
            }
            // FileNotFoundException or NoSuchFileException: one kind here.
            let _ = st.random.next_boolean();
            return Err(Error::Io(io::Error::new(io::ErrorKind::NotFound, message)));
        }
        Ok(())
    }

    fn deterministic(&self, op: FailureOp, name: &str) -> Result<()> {
        Ok(self.shared.maybe_throw_deterministic_exception(op, name)?)
    }

    /// `addFileHandle(c, name, handle)`: the mock filesystem sees the
    /// handle first (`HandleLimitFS` may refuse it).
    fn add_file_handle(&self, st: &mut State, name: &str, kind: Handle) -> Result<u64> {
        if let Some(fs) = &self.shared.fs {
            fs.on_open(name)?;
        }
        *st.open_files.entry(name.to_string()).or_insert(0) += 1;
        let id = st.next_handle;
        st.next_handle += 1;
        st.open_handles.insert(id, (name.to_string(), kind));
        Ok(id)
    }

    /// The shared part of `createOutput` and `createTempOutput`, after the
    /// delegate made the output.
    fn wrap_output(
        &self,
        st: &mut State,
        name: &str,
        mut out: FsIndexOutput,
    ) -> Result<FsIndexOutput> {
        // Java's `HandleLimitFS` refuses after the delegate created the file,
        // and the caller already recorded the name as created and unsynced:
        // a refused handle leaves both behind, so a retry is refused as
        // "already written to".
        let id = self.add_file_handle(st, name, Handle::Output)?;
        st.open_files_for_write.insert(name.to_string());
        // `maybeThrottle`.
        let throttle = match st.throttling {
            Throttling::Always => true,
            Throttling::Sometimes => st.random.next_int_bounded(200) == 0,
            Throttling::Never => false,
        };
        if throttle {
            st.throttled_outputs += 1;
        }
        out.set_faults(Box::new(MockOutput {
            shared: Arc::clone(&self.shared),
            id,
            first: true,
            throttle: throttle.then(|| Throttle::new(st.throttle.0, st.throttle.1)),
        }));
        Ok(out)
    }

    /// `crash()`: forgets every open handle (Java force-closes them),
    /// damages every unsynced file and fails every later call.
    pub fn crash(&self) -> Result<()> {
        let (unsynced, still_writing) = {
            let mut st = self.state();
            st.open_files.clear();
            st.open_files_for_write.clear();
            st.open_files_deleted.clear();
            let forgotten = std::mem::take(&mut st.open_handles);
            st.crashed = true;
            drop(st);
            if let Some(fs) = &self.shared.fs {
                for (name, _) in forgotten.values() {
                    fs.on_close(name);
                }
            }
            let still_writing: BTreeSet<String> = forgotten
                .into_values()
                .filter(|(_, kind)| matches!(kind, Handle::Output))
                .map(|(name, _)| name)
                .collect();
            (self.state().unsynced_files.clone(), still_writing)
        };
        let result = self.corrupt_files_with(&unsynced, &still_writing);
        self.state().unsynced_files.clear();
        result
    }

    /// `clearCrash()`.
    pub fn clear_crash(&self) {
        self.state().crashed = false;
    }

    /// `corruptFiles(files)`, with the virus checker off while it runs.
    pub fn corrupt_files(&self, files: &BTreeSet<String>) -> Result<()> {
        self.corrupt_files_with(files, &BTreeSet::new())
    }

    /// [`Self::corrupt_files`], deleting each of `still_writing` whatever
    /// its drawn fate (see the module doc).
    fn corrupt_files_with(
        &self,
        files: &BTreeSet<String>,
        still_writing: &BTreeSet<String>,
    ) -> Result<()> {
        let disabled = self
            .shared
            .fs
            .as_ref()
            .is_some_and(|fs| fs.disable_virus_checker());
        let result = self.corrupt_files_inner(files, still_writing);
        if disabled {
            if let Some(fs) = &self.shared.fs {
                fs.enable_virus_checker();
            }
        }
        result
    }

    /// `_corruptFiles(files)`: one of six fates per file, in sorted order.
    fn corrupt_files_inner(
        &self,
        files: &BTreeSet<String>,
        still_writing: &BTreeSet<String>,
    ) -> Result<()> {
        let inner = &self.shared.inner;
        for name in files {
            let damage = {
                let mut st = self.state();
                let mut damage = st.random.next_int_bounded(6);
                if st.always_corrupt && damage == 3 {
                    damage = 4;
                }
                damage
            };
            if still_writing.contains(name) {
                self.delete_file_unchecked(name)?;
                continue;
            }
            match damage {
                0 => self.delete_file_unchecked(name)?,
                1 => {
                    // Zeroed.
                    let length = inner.file_length(name).map_err(|e| unexpected(name, e))?;
                    self.delete_file_unchecked(name)?;
                    let mut out = inner.create_output(name).map_err(|e| unexpected(name, e))?;
                    let zeroes = [0u8; 256];
                    let mut upto = 0u64;
                    while upto < length {
                        let limit = (length - upto).min(zeroes.len() as u64) as usize;
                        out.write_bytes(&zeroes[..limit]);
                        upto += limit as u64;
                    }
                    out.close().map_err(|e| unexpected(name, e))?;
                }
                2 => {
                    // Partially truncated: the first half, through a temp file.
                    let bytes = inner.open(name).map_err(|e| unexpected(name, e))?;
                    let temp = copy_to_temp(inner.as_ref(), &bytes[..bytes.len() / 2], name)?;
                    drop(bytes);
                    self.delete_file_unchecked(name)?;
                    copy_back(inner.as_ref(), &temp, name)?;
                    self.delete_file_unchecked(&temp)?;
                }
                3 => {} // "didn't change"
                4 => {
                    // One bit flipped.
                    let bytes = inner.open(name).map_err(|e| unexpected(name, e))?;
                    let mut copy = bytes.to_vec();
                    drop(bytes);
                    if !copy.is_empty() {
                        let mut st = self.state();
                        let byte_to_corrupt =
                            (st.random.next_double() * copy.len() as f64) as usize;
                        let bit_to_flip = st.random.next_int_bounded(8);
                        drop(st);
                        copy[byte_to_corrupt] ^= 1 << bit_to_flip;
                    }
                    let temp = copy_to_temp(inner.as_ref(), &copy, name)?;
                    self.delete_file_unchecked(name)?;
                    copy_back(inner.as_ref(), &temp, name)?;
                    self.delete_file_unchecked(&temp)?;
                }
                _ => {
                    // Fully truncated.
                    self.delete_file_unchecked(name)?;
                    let out = inner.create_output(name).map_err(|e| unexpected(name, e))?;
                    out.close().map_err(|e| unexpected(name, e))?;
                }
            }
        }
        Ok(())
    }

    /// `deleteFile` as `_corruptFiles` calls it: the bookkeeping and the
    /// delegate's delete, without the crash check (the crash is what is
    /// happening).
    fn delete_file_unchecked(&self, name: &str) -> Result<()> {
        {
            let mut st = self.state();
            st.unsynced_files.remove(name);
            st.created_files.remove(name);
        }
        self.shared.inner.delete_file(name)
    }

    /// The common checks of a mutating call: yield, failures, crash.
    fn before_mutation(&self, op: FailureOp, name: &str, what: &str) -> Result<()> {
        self.maybe_yield();
        self.deterministic(op, name)?;
        if self.state().crashed {
            return Err(io_error(format!("cannot {what} after crash")));
        }
        Ok(())
    }
}

/// `"hit unexpected IOException while trying to corrupt file " + name`.
fn unexpected(name: &str, e: Error) -> Error {
    io_error(format!(
        "hit unexpected IOException while trying to corrupt file {name}: {e}"
    ))
}

/// Writes `bytes` to a new temp file of `dir`, returning its name.
fn copy_to_temp(dir: &dyn Directory, bytes: &[u8], name: &str) -> Result<String> {
    let mut out = dir
        .create_temp_output("name", "mdw_corrupt")
        .map_err(|e| unexpected(name, e))?;
    let temp = out.name().to_string();
    out.write_bytes(bytes);
    out.close().map_err(|e| unexpected(name, e))?;
    Ok(temp)
}

/// Copies `temp` back to `name`.
fn copy_back(dir: &dyn Directory, temp: &str, name: &str) -> Result<()> {
    let bytes = dir.open(temp).map_err(|e| unexpected(name, e))?;
    let mut out = dir.create_output(name).map_err(|e| unexpected(name, e))?;
    out.write_bytes(&bytes);
    out.close().map_err(|e| unexpected(name, e))?;
    Ok(())
}

/// `MockIndexOutputWrapper` (and `ThrottledIndexOutput`) as an output hook.
struct MockOutput {
    shared: Arc<Shared>,
    id: u64,
    /// `first`: the random exception may fire only on the first write.
    first: bool,
    throttle: Option<Throttle>,
}

impl MockOutput {
    /// `checkDiskFull`: how many of `len` bytes fit, when not all do, and
    /// the failure message.
    fn check_disk_full(&self, st: &mut State, name: &str, len: usize) -> Option<(usize, String)> {
        if st.max_size == 0 {
            return None;
        }
        let real_usage = self.shared.size_in_bytes().unwrap_or(0);
        let free_space = st.max_size.saturating_sub(real_usage);
        if free_space > len as u64 {
            return None;
        }
        let used = real_usage.saturating_add(free_space);
        if used > st.max_used_size {
            st.max_used_size = used;
        }
        let mut message = format!("fake disk full at {real_usage} bytes when writing {name}");
        if free_space > 0 {
            message.push_str(&format!("; wrote {free_space} of {len} bytes"));
        }
        Some((free_space as usize, message))
    }
}

impl OutputFaults for MockOutput {
    /// `MockIndexOutputWrapper.writeBytes`.
    fn before_write(&mut self, name: &str, len: usize) -> WriteVerdict {
        let mut st = self.shared.state();
        if st.crashed || !st.open_handles.contains_key(&self.id) {
            return WriteVerdict::Fail {
                keep: 0,
                error: io::Error::other(format!(
                    "MockDirectoryWrapper has crashed; cannot write to {name}"
                )),
            };
        }
        if let Some((keep, message)) = self.check_disk_full(&mut st, name, len) {
            return WriteVerdict::Fail {
                keep,
                error: io::Error::other(format!("{RANDOM_IO_EXCEPTION}: {message}")),
            };
        }
        let split = st.random.next_int_bounded(200) == 0;
        drop(st);
        if split {
            // Java writes half, yields, writes the rest.
            std::thread::yield_now();
        }
        if let Err(error) = self
            .shared
            .maybe_throw_deterministic_exception(FailureOp::Write, name)
        {
            return WriteVerdict::Fail { keep: len, error };
        }
        if self.first {
            self.first = false;
            let mut st = self.shared.state();
            if let Err(error) = MockDirectoryWrapper::maybe_throw_io_exception(&mut st, name) {
                return WriteVerdict::Fail { keep: len, error };
            }
        }
        if let Some(throttle) = &mut self.throttle {
            throttle.write_started = Some(Instant::now());
        }
        WriteVerdict::Write
    }

    /// `ThrottledIndexOutput.writeBytes`, after the delegate's write.
    fn after_write(&mut self, _name: &str, len: usize) {
        if let Some(throttle) = &mut self.throttle {
            if let Some(started) = throttle.write_started.take() {
                let nanos = i64::try_from(started.elapsed().as_nanos()).unwrap_or(i64::MAX);
                throttle.time_elapsed_nanos = throttle.time_elapsed_nanos.saturating_add(nanos);
            }
            throttle.pending_bytes = throttle.pending_bytes.saturating_add(len as i64);
            let delay = throttle.delay(false);
            Throttle::sleep(delay);
        }
    }

    /// `MockIndexOutputWrapper.close` (and `ThrottledIndexOutput.close`).
    fn on_close(&mut self, name: &str) -> CloseVerdict {
        if let Some(throttle) = &mut self.throttle {
            let delay = throttle.delay_millis + throttle.delay(true);
            Throttle::sleep(delay);
        }
        let stale = {
            let st = self.shared.state();
            st.crashed || !st.open_handles.contains_key(&self.id)
        };
        if stale {
            return CloseVerdict::FailWithoutFlush(io::Error::other(format!(
                "MockDirectoryWrapper has crashed; cannot close {name}"
            )));
        }
        match self
            .shared
            .maybe_throw_deterministic_exception(FailureOp::CloseOutput, name)
        {
            Ok(()) => CloseVerdict::Close,
            Err(e) => CloseVerdict::FailAfterFlush(e),
        }
    }
}

impl Drop for MockOutput {
    /// `removeIndexOutput`, then `trackDiskUsage`'s peak.
    fn drop(&mut self) {
        self.shared.remove_open_file(self.id);
        if self.shared.state().track_disk_usage {
            let size = self.shared.size_in_bytes().unwrap_or(0);
            let mut st = self.shared.state();
            if size > st.max_used_size {
                st.max_used_size = size;
            }
        }
    }
}

/// `MockIndexInputWrapper`'s open/close bookkeeping, as an input's guard.
struct MockInput {
    shared: Arc<Shared>,
    id: u64,
    /// `SlowClosingMockIndexInputWrapper`.
    slow_close: bool,
}

impl Drop for MockInput {
    fn drop(&mut self) {
        if self.slow_close {
            std::thread::sleep(Duration::from_millis(50));
        }
        self.shared.remove_open_file(self.id);
    }
}

impl BaseDirectoryWrapper for MockDirectoryWrapper {
    /// `MockDirectoryWrapper.close()`: fails while a file is open;
    /// otherwise turns the fault rates off and, when the index exists and
    /// the check is on, crashes the directory and runs `CheckIndex`.
    fn close(&self) -> std::result::Result<(), String> {
        {
            let mut st = self.state();
            if !st.is_open {
                return Ok(());
            }
            st.is_open = false;
            if !st.open_files.is_empty() {
                return Err(format!(
                    "MockDirectoryWrapper: cannot close: there are still {} open files: {:?}",
                    st.open_files.len(),
                    st.open_files
                ));
            }
            st.random_io_exception_rate = 0.0;
            st.random_io_exception_rate_on_open = 0.0;
        }
        self.maybe_yield();
        if let Some(fs) = &self.shared.fs {
            fs.check_leaks()?;
        }
        if self.check_index_on_close() && index_exists(self).map_err(|e| e.to_string())? {
            self.crash().map_err(|e| e.to_string())?;
            // Reading is still allowed after a crash, as in Java.
            crate::util::test_util::check_index(self)?;
        }
        Ok(())
    }

    fn is_open(&self) -> bool {
        self.state().is_open
    }

    fn set_check_index_on_close(&self, value: bool) {
        self.state().check_index_on_close = value;
    }

    fn check_index_on_close(&self) -> bool {
        self.state().check_index_on_close
    }

    fn as_mock(&self) -> Option<&MockDirectoryWrapper> {
        Some(self)
    }
}

impl Directory for MockDirectoryWrapper {
    fn list_all(&self) -> Result<Vec<String>> {
        self.maybe_yield();
        self.shared.inner.list_all()
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.maybe_yield();
        self.shared.inner.file_length(name)
    }

    /// `openInput`.
    fn open(&self, name: &str) -> Result<Input> {
        self.deterministic(FailureOp::OpenInput, name)?;
        self.maybe_throw_io_exception_on_open(name)?;
        self.maybe_yield();
        if self.state().fail_on_open_input {
            self.deterministic(FailureOp::OpenInput, name)?;
        }
        // `LuceneTestCase.slowFileExists`.
        if !self.shared.inner.list_all()?.iter().any(|f| f == name) {
            let mut st = self.state();
            let _ = st.random.next_boolean(); // FileNotFound or NoSuchFile
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{name} in dir=MockDirectoryWrapper"),
            )));
        }
        let mut st = self.state();
        if !st.allow_reading_files_still_open_for_write && st.open_files_for_write.contains(name) {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("MockDirectoryWrapper: file \"{name}\" is still open for writing"),
            )));
        }
        let input = self.shared.inner.open(name)?;
        let random_int = st.random.next_int_bounded(500);
        let slow_open = st.use_slow_open_closers && random_int == 1;
        let slow_close = st.use_slow_open_closers && random_int == 0;
        let id = self.add_file_handle(&mut st, name, Handle::Input)?;
        st.slow_opens += usize::from(slow_open);
        st.slow_closes += usize::from(slow_close);
        drop(st);
        if slow_open {
            // `SlowOpeningMockIndexInputWrapper`.
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(Input::Tracked(TrackedInput::new(
            input,
            MockInput {
                shared: Arc::clone(&self.shared),
                id,
                slow_close,
            },
        )))
    }

    /// `createOutput`.
    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.deterministic(FailureOp::CreateOutput, name)?;
        self.maybe_throw_io_exception_on_open(name)?;
        self.maybe_yield();
        if self.state().fail_on_create_output {
            self.deterministic(FailureOp::CreateOutput, name)?;
        }
        let mut st = self.state();
        if st.crashed {
            return Err(io_error("cannot createOutput after crash".to_string()));
        }
        if st.created_files.contains(name) {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("File \"{name}\" was already written to."),
            )));
        }
        if st.assert_no_delete_open_file && st.open_files.contains_key(name) {
            return Err(io_error(format!(
                "MockDirectoryWrapper: file \"{name}\" is still open: cannot overwrite"
            )));
        }
        // Recorded before the delegate creates the file, as Java does.
        st.unsynced_files.insert(name.to_string());
        st.created_files.insert(name.to_string());
        let out = self.shared.inner.create_output(name)?;
        self.wrap_output(&mut st, name, out)
    }

    /// `createTempOutput`.
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        let what = format!("temp: prefix={prefix} suffix={suffix}");
        self.deterministic(FailureOp::CreateTempOutput, &what)?;
        self.maybe_throw_io_exception_on_open(&what)?;
        self.maybe_yield();
        if self.state().fail_on_create_output {
            self.deterministic(FailureOp::CreateTempOutput, &what)?;
        }
        let mut st = self.state();
        if st.crashed {
            return Err(io_error("cannot createTempOutput after crash".to_string()));
        }
        let out = self.shared.inner.create_temp_output(prefix, suffix)?;
        let name = out.name().to_string();
        if !name.to_lowercase().ends_with(".tmp") {
            return Err(io_error(format!(
                "wrapped directory failed to use .tmp extension: got: {name}"
            )));
        }
        st.unsynced_files.insert(name.clone());
        st.created_files.insert(name.clone());
        self.wrap_output(&mut st, &name, out)
    }

    /// `sync`: each file may fail at the random rate; under a mock
    /// filesystem with fsync disabled, only the file's existence is checked.
    fn sync(&self, names: &[String]) -> Result<()> {
        self.maybe_yield();
        for name in names.iter().take(1) {
            self.deterministic(FailureOp::Sync, name)?;
        }
        if names.is_empty() {
            self.deterministic(FailureOp::Sync, "")?;
        }
        if self.state().crashed {
            return Err(io_error("cannot sync after crash".to_string()));
        }
        let fsync_disabled = self
            .shared
            .fs
            .as_ref()
            .is_some_and(|fs| fs.fsync_disabled());
        for name in names {
            Self::maybe_throw_io_exception(&mut self.state(), name)?;
            if fsync_disabled {
                self.shared.inner.file_length(name)?;
            } else {
                self.shared.inner.sync(std::slice::from_ref(name))?;
            }
            self.state().unsynced_files.remove(name);
        }
        Ok(())
    }

    /// `rename`.
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        self.before_mutation(FailureOp::Rename, source, "rename")?;
        {
            let st = self.state();
            if st.assert_no_delete_open_file {
                for (file, which) in [(source, "source"), (dest, "dest")] {
                    if st.open_files.contains_key(file) {
                        return Err(io_error(format!(
                            "MockDirectoryWrapper: {which} file \"{file}\" is still open: cannot rename"
                        )));
                    }
                }
            }
        }
        self.shared.inner.rename(source, dest)?;
        let mut st = self.state();
        if st.unsynced_files.remove(source) {
            st.unsynced_files.insert(dest.to_string());
        }
        st.open_files_deleted.remove(source);
        st.created_files.remove(source);
        st.created_files.insert(dest.to_string());
        Ok(())
    }

    /// `deleteFile`.
    fn delete_file(&self, name: &str) -> Result<()> {
        self.before_mutation(FailureOp::DeleteFile, name, "delete")?;
        {
            let mut st = self.state();
            if st.open_files.contains_key(name) {
                st.open_files_deleted.insert(name.to_string());
                if st.assert_no_delete_open_file {
                    return Err(io_error(format!(
                        "MockDirectoryWrapper: file \"{name}\" is still open: cannot delete"
                    )));
                }
            } else {
                st.open_files_deleted.remove(name);
            }
            st.unsynced_files.remove(name);
        }
        self.shared.inner.delete_file(name)?;
        self.state().created_files.remove(name);
        Ok(())
    }

    /// `syncMetaData`.
    fn sync_meta_data(&self) -> Result<()> {
        self.before_mutation(FailureOp::SyncMetaData, "", "sync metadata")?;
        if self
            .shared
            .fs
            .as_ref()
            .is_some_and(|fs| fs.fsync_disabled())
        {
            return Ok(());
        }
        self.shared.inner.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.maybe_yield();
        self.shared.inner.obtain_lock(name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        self.shared.inner.pending_deletions()
    }

    fn fs_directory_path(&self) -> Option<&Path> {
        self.shared.inner.fs_directory_path()
    }
}

#[cfg(test)]
mod tests {
    use super::Throttle;

    /// `ThrottledIndexOutput.getDelay`, quirks included: the observed rate
    /// is `(elapsedNanos / pendingBytes) * 1e9` in integer arithmetic, so a
    /// delegate faster than one nanosecond per byte reads as 0 bytes/s and
    /// is never slowed, and the delay is whole milliseconds.
    #[test]
    fn delay_follows_java_including_its_integer_rate() {
        let mut t = Throttle::new(5_000_000, 3);
        // Under `minBytesWritten`, and not closing: no delay yet.
        t.pending_bytes = 1000;
        t.time_elapsed_nanos = 10_000;
        assert_eq!(t.delay(false), 0);
        // 100 kB written at 2 ns/byte against 5 MB/s: 20 ms expected, the
        // 0.2 ms already spent rounds away.
        t.pending_bytes = 100_000;
        t.time_elapsed_nanos = 200_000;
        assert_eq!(t.delay(false), 20);
        assert_eq!((t.pending_bytes, t.time_elapsed_nanos), (0, 0));
        // Faster than 1 ns/byte: Java's rate truncates to 0, no delay.
        t.pending_bytes = 100_000;
        t.time_elapsed_nanos = 50_000;
        assert_eq!(t.delay(true), 0);
        // Closing ignores `minBytesWritten`.
        let mut t = Throttle::new(1000, 0);
        t.pending_bytes = 10;
        t.time_elapsed_nanos = 100;
        assert_eq!(t.delay(true), 10);
        Throttle::sleep(0);
        Throttle::sleep(1);
    }
}
