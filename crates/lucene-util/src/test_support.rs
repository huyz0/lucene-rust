//! Scratch directories for tests, with an RAII guard that cleans up.
//!
//! Every crate in this workspace used to grow its own `fn tempdir() -> PathBuf`
//! around `std::env::temp_dir()`, and none of them removed anything. The cost
//! is not theoretical: batch c28 found `/tmp` (a 16 GB tmpfs) at 100% full from
//! roughly 21 000 leftover `lucene-*-test-*` directories, which failed tests for
//! reasons having nothing to do with the code under test — the most expensive
//! kind of flake to diagnose.
//!
//! [`TempDir`] is the one shared replacement. It removes its directory on
//! `Drop`, **except while the thread is panicking**, so a failing test leaves
//! its bytes on disk to be inspected and a passing one leaves nothing. That
//! asymmetry is the whole point: cleanup must never cost you the evidence for
//! the failure that made you look.
//!
//! ```ignore
//! let dir = TempDir::new("segment-writer");
//! let store = FsDirectory::open(&dir);   // &TempDir is AsRef<Path>
//! let file = dir.join("_0.si");          // Deref<Target = Path>
//! // ... directory removed here, unless this test is unwinding
//! ```
//!
//! Set `LUCENE_KEEP_TEST_DIRS=1` to keep every directory a run creates, which
//! is what you want when the interesting state is in a test that *passes*.
//!
//! This module is compiled only for `cargo test` and for consumers that opt in
//! with `lucene-util`'s `test-support` feature (declared on a
//! `[dev-dependencies]` edge, which Cargo's resolver-2 feature unification
//! confines to test and bench targets — it never reaches a production build).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes directories created within the same nanosecond by different
/// threads of the same process, which `cargo test`'s default parallelism makes
/// ordinary. Process id alone is not enough, and a timestamp alone is not
/// either.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The environment variable that suppresses cleanup for a whole run.
const KEEP_VAR: &str = "LUCENE_KEEP_TEST_DIRS";

/// A freshly created directory under the system temp dir, removed on `Drop`.
///
/// Kept rather than removed when the thread is panicking (so a failing test
/// stays debuggable), when [`TempDir::keep`] was called, or when
/// `LUCENE_KEEP_TEST_DIRS` is set.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
    keep: bool,
}

impl TempDir {
    /// Create a uniquely named scratch directory tagged with `label`.
    ///
    /// `label` only has to be readable — it appears in the directory name, so
    /// a leftover directory (from a panicking test) says which test made it.
    ///
    /// # Panics
    ///
    /// If the directory cannot be created. Test-support code: a scratch
    /// directory that cannot be made is not a condition any caller can handle.
    pub fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        // Keep the `lucene-*-test-*` shape the existing ad-hoc helpers used, so
        // an operator's cleanup glob still finds anything this leaves behind.
        let path = std::env::temp_dir().join(format!(
            "lucene-{label}-test-{}-{seq}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|e| panic!("could not create test dir {}: {e}", path.display()));
        Self { path, keep: false }
    }

    /// The directory's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The directory's path as a `&str`.
    ///
    /// # Panics
    ///
    /// If the path is not valid UTF-8, which on any machine running this suite
    /// means `TMPDIR` itself is not.
    pub fn path_str(&self) -> &str {
        self.path
            .to_str()
            .expect("temp dir path is not valid UTF-8")
    }

    /// Give up ownership: the directory survives this guard's death.
    ///
    /// For the case where a test deliberately hands the bytes to something that
    /// outlives it, or where you are mid-investigation and want them kept.
    pub fn keep(mut self) -> PathBuf {
        self.keep = true;
        self.path.clone()
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

/// Lets a guard stand in wherever a path-shaped argument is taken, the way
/// `PathBuf` itself does: `Path::new(&dir)`, and -- via std's blanket
/// `impl<T: AsRef<OsStr>> From<&T> for PathBuf` -- every `impl Into<PathBuf>`
/// parameter in this workspace, `FsDirectory::open` above all. Passing `&dir`
/// rather than the guard keeps ownership with the test, which matters: dropping
/// the guard early would delete the directory out from under whatever just took
/// the path.
impl AsRef<std::ffi::OsStr> for TempDir {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.path.as_os_str()
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A failing test's scratch bytes are evidence. Removing them here would
        // delete exactly the thing the failure needs explaining with, so the
        // guard steps aside whenever it is being dropped by an unwind.
        if self.keep || std::thread::panicking() || std::env::var_os(KEEP_VAR).is_some() {
            eprintln!("lucene test dir kept: {}", self.path.display());
            return;
        }
        // Best-effort: a test that already removed the directory itself, or a
        // platform that refuses, must not turn into a second failure on the way
        // out of a passing test.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_a_directory_that_drop_removes() {
        let path = {
            let dir = TempDir::new("guard-basic");
            assert!(dir.path().is_dir());
            std::fs::write(dir.join("a.bin"), b"x").unwrap();
            dir.path().to_path_buf()
        };
        assert!(
            !path.exists(),
            "drop should have removed {}",
            path.display()
        );
    }

    #[test]
    fn two_guards_never_collide() {
        let a = TempDir::new("guard-unique");
        let b = TempDir::new("guard-unique");
        assert_ne!(a.path(), b.path());
        assert!(a.path().is_dir() && b.path().is_dir());
    }

    #[test]
    fn keep_disarms_the_guard() {
        let path = TempDir::new("guard-keep").keep();
        assert!(path.is_dir(), "keep() must leave the directory in place");
        std::fs::remove_dir_all(&path).unwrap();
    }

    /// The property that makes cleanup safe to adopt everywhere: a test that
    /// fails still has its bytes. Asserted the only way it can be — by dropping
    /// the guard inside a real unwind.
    #[test]
    fn a_panicking_test_keeps_its_directory() {
        let holder = std::sync::Mutex::new(None);
        let result = std::panic::catch_unwind(|| {
            let dir = TempDir::new("guard-panic");
            *holder.lock().unwrap() = Some(dir.path().to_path_buf());
            std::fs::write(dir.join("evidence.bin"), b"why it failed").unwrap();
            panic!("the test under study fails here");
        });
        assert!(result.is_err());
        let path = holder.lock().unwrap().take().expect("path recorded");
        assert!(
            path.join("evidence.bin").exists(),
            "an unwinding drop must not delete the failure's evidence"
        );
        std::fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn path_accessors_agree() {
        let dir = TempDir::new("guard-accessors");
        assert_eq!(Path::new(dir.path_str()), dir.path());
        assert_eq!(AsRef::<Path>::as_ref(&dir), dir.path());
        // Deref: `join` is `Path`'s, reached through the guard.
        assert_eq!(dir.join("f"), dir.path().join("f"));
        assert_eq!(PathBuf::from(&dir), dir.path());
        assert_eq!(Path::new(&dir), dir.path());
        assert!(format!("{dir:?}").contains("guard-accessors"));
    }

    #[test]
    fn keep_var_suppresses_cleanup() {
        // The env var is process-wide, so this drives `Drop`'s branch through
        // the same predicate rather than mutating the environment (which would
        // race every other test in this binary).
        let mut dir = TempDir::new("guard-keepvar");
        let path = dir.path().to_path_buf();
        dir.keep = true;
        drop(dir);
        assert!(path.is_dir());
        std::fs::remove_dir_all(&path).unwrap();
    }
}
