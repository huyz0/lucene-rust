//! `CloseableDirectory`: at the end of a test, fails when a directory the
//! test opened was never closed.
//!
//! Java registers it with the suite's closeables and checks
//! `failureMarker.wasSuccessful()`; here it is a guard whose `Drop` checks,
//! and a test that is already failing (unwinding) is not failed again.

use crate::store::base_directory_wrapper::BaseDirectoryWrapper;

/// `CloseableDirectory(dir, failureMarker)`.
pub struct CloseableDirectory<'a> {
    dir: &'a dyn BaseDirectoryWrapper,
}

impl<'a> CloseableDirectory<'a> {
    /// Guards `dir`.
    pub fn new(dir: &'a dyn BaseDirectoryWrapper) -> Self {
        Self { dir }
    }

    /// `close()`: an `Err` (Java's `Assert.fail`) when `dir` is still open.
    ///
    /// # Errors
    /// `"Directory not closed: ..."`.
    pub fn check(&self) -> Result<(), String> {
        if self.dir.is_open() {
            Err(format!("Directory not closed: {:?}", self.dir.list_all()))
        } else {
            Ok(())
        }
    }
}

impl Drop for CloseableDirectory<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            if let Err(e) = self.check() {
                panic!("{e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use lucene_store::ByteBuffersDirectory;

    use super::*;
    use crate::store::base_directory_wrapper::RawDirectoryWrapper;

    #[test]
    fn an_unclosed_directory_fails_the_test() {
        let dir = RawDirectoryWrapper::new(ByteBuffersDirectory::new());
        let guard = CloseableDirectory::new(&dir);
        assert!(guard
            .check()
            .unwrap_err()
            .starts_with("Directory not closed: "));
        let err = catch_unwind(AssertUnwindSafe(|| drop(guard))).unwrap_err();
        assert!(err
            .downcast_ref::<String>()
            .unwrap()
            .starts_with("Directory not closed"));
    }

    #[test]
    fn a_closed_directory_passes() {
        let dir = RawDirectoryWrapper::new(ByteBuffersDirectory::new());
        let guard = CloseableDirectory::new(&dir);
        dir.close().unwrap();
        guard.check().unwrap();
        drop(guard);
    }

    #[test]
    fn a_failing_test_is_not_failed_again() {
        let dir = RawDirectoryWrapper::new(ByteBuffersDirectory::new());
        let err = catch_unwind(AssertUnwindSafe(|| {
            let _guard = CloseableDirectory::new(&dir);
            panic!("the test's own failure");
        }))
        .unwrap_err();
        assert_eq!(
            *err.downcast_ref::<&str>().unwrap(),
            "the test's own failure"
        );
    }
}
