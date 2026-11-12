//! Pages that were read but failed to decode (an encrypted chunk that
//! failed authentication), recorded so the call that read them can fail.
//!
//! A demand-filled input is a plain `&[u8]` ([`crate::paged_input`]), and a
//! memory access cannot return an error. When a page's stored bytes fail to
//! decode, the pager installs zeros instead and records the failure here
//! under the file's path; the record lasts until the file is closed. A
//! caller that knows which files a call read (the FFI layer knows each
//! call's shard path) asks [`failure_under`] after the call and fails it
//! with the recorded message. [`count`] is one atomic load, so the check
//! costs nothing while no file has failed.
//!
//! Readers that announce their reads ([`crate::directory::Input::will_need`],
//! [`crate::prefetch`]) get the error directly.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::lock::lock_ignoring_poison;

/// At most this many failures are kept; older ones go first.
const MAX_RECORDS: usize = 1024;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

struct Record {
    /// The registration this record belongs to ([`record`]'s `owner`).
    owner: usize,
    path: String,
    message: String,
}

/// How many failures are recorded now.
pub fn count() -> usize {
    COUNT.load(Ordering::Relaxed)
}

/// Records that `path` (a file's full path) failed to decode with
/// `message`, until [`forget`] is called with the same `owner`. A path
/// already recorded for `owner` keeps its first message.
pub(crate) fn record(owner: usize, path: &str, message: &str) {
    let mut records = lock_ignoring_poison(&RECORDS);
    if records.iter().any(|r| r.owner == owner && r.path == path) {
        return;
    }
    if records.len() >= MAX_RECORDS {
        records.remove(0);
    }
    records.push(Record {
        owner,
        path: path.to_string(),
        message: message.to_string(),
    });
    COUNT.store(records.len(), Ordering::Relaxed);
}

/// Drops `owner`'s records: its file was closed.
pub(crate) fn forget(owner: usize) {
    if count() == 0 {
        return;
    }
    let mut records = lock_ignoring_poison(&RECORDS);
    records.retain(|r| r.owner != owner);
    COUNT.store(records.len(), Ordering::Relaxed);
}

/// The message of a recorded failure of a file under the directory `root`
/// (a path prefix ending at a path separator), if any.
pub fn failure_under(root: &str) -> Option<String> {
    if count() == 0 || root.is_empty() {
        return None;
    }
    let root = root.trim_end_matches(std::path::MAIN_SEPARATOR);
    let records = lock_ignoring_poison(&RECORDS);
    records
        .iter()
        .find(|r| {
            r.path
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
        record(usize::MAX - 1, &a, "chunk 3 failed");
        record(usize::MAX - 1, &a, "a second message is ignored");
        assert!(count() >= 1);
        assert_eq!(
            failure_under(&format!("{sep}idx{sep}s0")).as_deref(),
            Some("chunk 3 failed")
        );
        assert_eq!(
            failure_under(&format!("{sep}idx{sep}s0{sep}")).as_deref(),
            Some("chunk 3 failed")
        );
        assert_eq!(failure_under(&format!("{sep}idx{sep}s")), None);
        assert_eq!(failure_under(""), None);
        forget(usize::MAX - 1);
        assert_eq!(failure_under(&format!("{sep}idx{sep}s0")), None);
    }
}
