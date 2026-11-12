//! Pages that could not be read or decoded, recorded so the calls that read
//! them can fail.
//!
//! A demand-filled input is a plain `&[u8]` ([`crate::paged_input`]), and a
//! memory access cannot return an error. When a page's stored bytes fail to
//! decode, or the disk keeps refusing to read them, the pager installs zeros
//! instead and records the failure here under the file's path. A caller
//! that knows which files a call read (the FFI layer knows each call's
//! shard path) takes [`epoch`] when the call starts and asks
//! [`failure_under`] after it, and fails the call with the recorded message.
//! [`count`] is one atomic load, so the check costs nothing while no file
//! has failed.
//!
//! - A page that fails to decode is bad on disk: its record lasts until the
//!   file is closed.
//! - A page that could not be read is held as zeros only briefly, then
//!   dropped so the next touch reads the disk again; its record is
//!   [`clear`]ed then. A cleared record still fails every call that started
//!   before it was cleared, since such a call may have read the zeros.
//!
//! Readers that announce their reads ([`crate::directory::Input::will_need`],
//! [`crate::prefetch`]) get the error directly.
//!
//! Zeros reach only threads running such a checked call ([`set_checked`]).
//! On any other thread (a merge, a warmer) nothing would fail the work that
//! read them, so the pager keeps retrying the read instead. A merge also
//! fails when a failure under its directory overlapped it, since it may
//! run inside a checked call and write what it read into a new segment.

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::lock::lock_ignoring_poison;

/// At most this many failures are kept; older ones go first.
const MAX_RECORDS: usize = 1024;
/// How long a cleared record is kept for calls that started before it was
/// cleared.
const CLEARED_KEPT: Duration = Duration::from_secs(600);

static COUNT: AtomicUsize = AtomicUsize::new(0);
/// Advances on every record and clear: calls compare it with when they
/// started.
static EPOCH: AtomicU64 = AtomicU64::new(1);
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

struct Record {
    /// The registration this record belongs to ([`record`]'s `owner`).
    owner: usize,
    path: String,
    message: String,
    /// When the failure stopped ([`clear`]): the epoch and the time.
    cleared: Option<(u64, Instant)>,
}

/// How many failures are recorded now, cleared ones still kept included.
pub fn count() -> usize {
    COUNT.load(Ordering::Relaxed)
}

/// The current epoch: a call takes it when it starts and passes it to
/// [`failure_under`] when it ends.
pub fn epoch() -> u64 {
    EPOCH.load(Ordering::Acquire)
}

fn advance() -> u64 {
    EPOCH.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
}

/// Drops cleared records past [`CLEARED_KEPT`] and, past [`MAX_RECORDS`],
/// the oldest cleared ones. A failure not cleared is never dropped.
fn prune(records: &mut Vec<Record>) {
    records.retain(|r| r.cleared.is_none_or(|(_, at)| at.elapsed() < CLEARED_KEPT));
    while records.len() >= MAX_RECORDS {
        let Some(at) = records.iter().position(|r| r.cleared.is_some()) else {
            break;
        };
        records.remove(at);
    }
}

/// Records that `path` (a file's full path) failed with `message`, until
/// [`clear`] or [`forget`] is called with the same `owner`. A path recorded
/// for `owner` and not cleared keeps its first message; one cleared is
/// made current again.
pub(crate) fn record(owner: usize, path: &str, message: &str) {
    let mut records = lock_ignoring_poison(&RECORDS);
    if let Some(r) = records
        .iter_mut()
        .find(|r| r.owner == owner && r.path == path)
    {
        if r.cleared.is_some() {
            r.cleared = None;
            r.message = message.to_string();
            advance();
        }
        return;
    }
    prune(&mut records);
    records.push(Record {
        owner,
        path: path.to_string(),
        message: message.to_string(),
        cleared: None,
    });
    COUNT.store(records.len(), Ordering::Relaxed);
    advance();
}

/// Ends `owner`'s failure: calls that start from now on no longer see it,
/// calls already running still do.
pub(crate) fn clear(owner: usize) {
    if count() == 0 {
        return;
    }
    let mut records = lock_ignoring_poison(&RECORDS);
    let now = advance();
    for r in records.iter_mut().filter(|r| r.owner == owner) {
        if r.cleared.is_none() {
            r.cleared = Some((now, Instant::now()));
        }
    }
    prune(&mut records);
    COUNT.store(records.len(), Ordering::Relaxed);
}

/// Drops `owner`'s records: its file was closed. A call still running
/// that read the file keeps seeing them, as after [`clear`].
pub(crate) fn forget(owner: usize) {
    clear(owner);
}

#[cfg(target_os = "linux")]
/// Each thread's flag for [`set_checked`], by thread id.
#[cfg(target_os = "linux")]
static CHECKED: Mutex<Option<HashMap<u32, Arc<AtomicBool>>>> = Mutex::new(None);

#[cfg(target_os = "linux")]
/// This thread's entry in [`CHECKED`], removed when the thread ends.
struct CheckedSlot {
    tid: u32,
    on: Arc<AtomicBool>,
}

#[cfg(target_os = "linux")]
impl CheckedSlot {
    fn register() -> CheckedSlot {
        // SAFETY: gettid has no preconditions.
        let tid = u32::try_from(unsafe { libc::syscall(libc::SYS_gettid) }).unwrap_or(0);
        let on = Arc::new(AtomicBool::new(false));
        lock_ignoring_poison(&CHECKED)
            .get_or_insert_with(HashMap::new)
            .insert(tid, Arc::clone(&on));
        CheckedSlot { tid, on }
    }
}

#[cfg(target_os = "linux")]
impl Drop for CheckedSlot {
    fn drop(&mut self) {
        if let Some(map) = lock_ignoring_poison(&CHECKED).as_mut() {
            if map
                .get(&self.tid)
                .is_some_and(|on| Arc::ptr_eq(on, &self.on))
            {
                map.remove(&self.tid);
            }
        }
    }
}

#[cfg(target_os = "linux")]
thread_local! {
    static SLOT: CheckedSlot = CheckedSlot::register();
}

/// Marks whether this thread runs a checked call; nothing to mark here,
/// where no pager hands out zeros.
#[cfg(not(target_os = "linux"))]
pub fn set_checked(_on: bool) {}

#[cfg(target_os = "linux")]
/// Marks whether this thread is running a call that asks
/// [`failure_under`] when it ends and fails on a failure: such a thread may
/// be handed zeros for a page that cannot be read; any other thread waits
/// for the read to succeed.
pub fn set_checked(on: bool) {
    let _ = SLOT.try_with(|s| s.on.store(on, Ordering::Release));
}

#[cfg(target_os = "linux")]
/// Whether the thread with id `tid` is running a checked call
/// ([`set_checked`]).
pub(crate) fn is_checked(tid: u32) -> bool {
    lock_ignoring_poison(&CHECKED)
        .as_ref()
        .and_then(|map| map.get(&tid))
        .is_some_and(|on| on.load(Ordering::Acquire))
}

/// The message of a failure of a file under the directory `root` (a path
/// prefix ending at a path separator) that a call started at epoch `since`
/// may have seen: one not cleared, or cleared after `since`.
pub fn failure_under(root: &str, since: u64) -> Option<String> {
    if count() == 0 || root.is_empty() {
        return None;
    }
    let root = root.trim_end_matches(std::path::MAIN_SEPARATOR);
    let records = lock_ignoring_poison(&RECORDS);
    records
        .iter()
        .find(|r| {
            r.cleared.is_none_or(|(at, _)| at > since)
                && r.path
                    .strip_prefix(root)
                    .is_some_and(|rest| rest.starts_with(std::path::MAIN_SEPARATOR))
        })
        .map(|r| r.message.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_found_by_directory_and_forgotten_with_their_file() {
        let sep = std::path::MAIN_SEPARATOR;
        let a = format!("{sep}idx{sep}s0{sep}_0.doc");
        let before = epoch();
        record(usize::MAX - 1, &a, "chunk 3 failed");
        record(usize::MAX - 1, &a, "a second message is ignored");
        assert!(count() >= 1);
        let s0 = format!("{sep}idx{sep}s0");
        assert_eq!(
            failure_under(&s0, before).as_deref(),
            Some("chunk 3 failed")
        );
        assert_eq!(
            failure_under(&format!("{sep}idx{sep}s0{sep}"), before).as_deref(),
            Some("chunk 3 failed")
        );
        assert_eq!(failure_under(&format!("{sep}idx{sep}s"), before), None);
        assert_eq!(failure_under("", before), None);
        forget(usize::MAX - 1);
        // A call running across the failure still sees it; a new one does not.
        assert_eq!(
            failure_under(&s0, before).as_deref(),
            Some("chunk 3 failed")
        );
        assert_eq!(failure_under(&s0, epoch()), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn only_threads_in_a_checked_call_are_checked() {
        // SAFETY: gettid has no preconditions.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as u32;
        set_checked(true);
        assert!(is_checked(tid));
        set_checked(false);
        assert!(!is_checked(tid));
        let other = std::thread::spawn(|| {
            set_checked(true);
            // SAFETY: as above.
            unsafe { libc::syscall(libc::SYS_gettid) as u32 }
        })
        .join()
        .unwrap();
        // The thread ended: its entry went with it.
        assert!(!is_checked(other));
    }

    #[test]
    fn a_cleared_failure_can_be_recorded_again() {
        let sep = std::path::MAIN_SEPARATOR;
        let a = format!("{sep}idx{sep}s1{sep}_0.vec");
        let s1 = format!("{sep}idx{sep}s1");
        record(usize::MAX - 2, &a, "read failed");
        clear(usize::MAX - 2);
        let after = epoch();
        assert_eq!(failure_under(&s1, after), None);
        record(usize::MAX - 2, &a, "read failed again");
        assert_eq!(
            failure_under(&s1, after).as_deref(),
            Some("read failed again")
        );
        forget(usize::MAX - 2);
        assert_eq!(failure_under(&s1, epoch()), None);
    }
}
