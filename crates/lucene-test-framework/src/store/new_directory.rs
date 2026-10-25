//! `LuceneTestCase.newDirectory` and its family: a randomized directory
//! implementation, wrapped in a [`MockDirectoryWrapper`] (or, rarely, a
//! [`RawDirectoryWrapper`]).
//!
//! As Java's `TEST_DIRECTORY=random` (the default):
//!
//! - `newDirectoryImpl`: rarely one of `NIOFSDirectory` ([`FsDirectory`]),
//!   `MMapDirectory` ([`MmapDirectory`]) and [`ByteBuffersDirectory`];
//!   else rarely a [`FileSwitchDirectory`] over two of them; else a
//!   `ByteBuffersDirectory`.
//! - `wrapDirectory`: rarely (unless bare or asked for a filesystem) an
//!   [`NrtCachingDirectory`] around it; then a raw wrapper when bare, else
//!   a mock wrapper with `TEST_THROTTLING` (`Never`, as Java's default).
//!
//! Every filesystem directory runs on a [`MockFs`] whose behaviours are
//! drawn once per suite, as `TestRuleTemporaryFilesCleanup` does: from the
//! master seed, so the draw does not move the test's own stream.
//!
//! Differs: Java's `-Dtests.directory` choice of implementation is not
//! read (every directory is drawn); the delegate of a `FileSwitchDirectory`
//! and of an `NrtCachingDirectory` is shared through an `Arc`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use lucene_store::{
    ByteBuffersDirectory, Directory, FileSwitchDirectory, FsDirectory, MmapDirectory,
    NrtCachingDirectory,
};
use lucene_util::java_random::JavaRandom;
use lucene_util::test_support::TempDir;

use super::base_directory_wrapper::{BaseDirectoryWrapper, RawDirectoryWrapper};
use super::mock_directory_wrapper::{MockDirectoryWrapper, Throttling};
use crate::mockfile::mock_fs::{MockFs, MockFsConfig};
use crate::util::lucene_test_case::{rarely, LuceneTestCase};

/// `LuceneTestCase.CORE_DIRECTORIES` (`FS_DIRECTORIES` is its first two).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryImpl {
    /// `NIOFSDirectory`.
    NioFs,
    /// `MMapDirectory`.
    MMap,
    /// `ByteBuffersDirectory`.
    ByteBuffers,
}

impl DirectoryImpl {
    const CORE: [DirectoryImpl; 3] = [Self::NioFs, Self::MMap, Self::ByteBuffers];

    /// `RandomPicks.randomFrom(random, CORE_DIRECTORIES)`.
    fn random_core(r: &mut JavaRandom) -> Self {
        Self::CORE[r.next_int_bounded(3) as usize]
    }

    /// `RandomPicks.randomFrom(random, FS_DIRECTORIES)`.
    fn random_fs(r: &mut JavaRandom) -> Self {
        Self::CORE[r.next_int_bounded(2) as usize]
    }
}

/// `TEST_THROTTLING`: Java's default (`tests.throttling` unset) is never.
pub const TEST_THROTTLING: Throttling = Throttling::Never;

/// `newFileSwitchDirectory`'s extensions, in Java's order.
const FILE_SWITCH_EXTENSIONS: [&str; 13] = [
    "fdt", "fdx", "tim", "tip", "si", "fnm", "pos", "dii", "dim", "nvm", "nvd", "dvm", "dvd",
];

/// `Collections.shuffle(list, random)`.
fn java_shuffle<T>(list: &mut [T], r: &mut JavaRandom) {
    for i in (1..list.len()).rev() {
        let j = r.next_int_bounded(i as i32 + 1) as usize;
        list.swap(i, j);
    }
}

/// One `newDirectory` call's context.
struct Factory<'a> {
    random: &'a mut JavaRandom,
    nightly: bool,
    multiplier: i32,
    fs_config: MockFsConfig,
    fs: Option<Arc<MockFs>>,
    temps: Vec<TempDir>,
}

impl Factory<'_> {
    fn rarely(&mut self) -> bool {
        rarely(self.random, self.nightly, self.multiplier)
    }

    /// The suite's mock filesystem, made on first use.
    fn mock_fs(&mut self) -> Arc<MockFs> {
        let config = self.fs_config;
        Arc::clone(self.fs.get_or_insert_with(|| MockFs::new(config)))
    }

    /// `newFSDirectoryImpl(clazz, path, lf)` over the mock filesystem.
    fn fs_directory(&mut self, kind: DirectoryImpl, path: &Path) -> Arc<dyn Directory> {
        let fs = self.mock_fs();
        // `FSDirectory` creates only a missing path, so `ExtrasFS` adds its
        // extra file only there, never to a directory the caller made.
        if !path.exists() {
            let _ = fs.create_directory(path);
        }
        let faults: Arc<dyn lucene_store::directory::FsFaults> = fs;
        match kind {
            DirectoryImpl::NioFs => Arc::new(FsDirectory::open(path).with_fs_faults(faults)),
            _ => Arc::new(MmapDirectory::open(path).with_fs_faults(faults)),
        }
    }

    /// `newDirectoryImpl(random, clazzName)`; `None` is `"random"`.
    fn directory_impl(&mut self, kind: Option<DirectoryImpl>) -> Arc<dyn Directory> {
        let kind = match kind {
            Some(kind) => kind,
            None if self.rarely() => DirectoryImpl::random_core(self.random),
            None if self.rarely() => {
                let first = self.file_switch_side();
                let second = self.file_switch_side();
                let primary = self.directory_impl(Some(first));
                let secondary = self.directory_impl(Some(second));
                let mut extensions = FILE_SWITCH_EXTENSIONS;
                java_shuffle(&mut extensions, self.random);
                let n = 1 + self.random.next_int_bounded(extensions.len() as i32) as usize;
                let extensions: HashSet<String> =
                    extensions[..n].iter().map(|e| e.to_string()).collect();
                return Arc::new(
                    FileSwitchDirectory::new(extensions, primary, secondary)
                        .expect("no reserved extension"),
                );
            }
            None => DirectoryImpl::ByteBuffers,
        };
        match kind {
            DirectoryImpl::ByteBuffers => Arc::new(ByteBuffersDirectory::new()),
            fs_kind => {
                let label = match fs_kind {
                    DirectoryImpl::NioFs => "index-NIOFSDirectory",
                    _ => "index-MMapDirectory",
                };
                let temp = TempDir::new(label);
                let dir = self.fs_directory(fs_kind, temp.path());
                self.temps.push(temp);
                dir
            }
        }
    }

    /// One side of `newFileSwitchDirectory`: rarely a core directory, else
    /// `ByteBuffersDirectory`.
    fn file_switch_side(&mut self) -> DirectoryImpl {
        if self.rarely() {
            DirectoryImpl::random_core(self.random)
        } else {
            DirectoryImpl::ByteBuffers
        }
    }

    /// `wrapDirectory(random, directory, bare, filesystem)`.
    fn wrap(mut self, directory: Arc<dyn Directory>, bare: bool, filesystem: bool) -> Wrapped {
        let directory: Arc<dyn Directory> = if self.rarely() && !bare && !filesystem {
            let max_merge_size_mb = self.random.next_double();
            let max_cached_mb = self.random.next_double();
            Arc::new(NrtCachingDirectory::new(
                directory,
                max_merge_size_mb,
                max_cached_mb,
            ))
        } else {
            directory
        };
        if bare {
            Wrapped::Raw(RawDirectoryWrapper::over(directory, self.temps))
        } else {
            let mock = MockDirectoryWrapper::over(self.random, directory, self.fs, self.temps);
            mock.set_throttling(TEST_THROTTLING);
            Wrapped::Mock(mock)
        }
    }
}

enum Wrapped {
    Raw(RawDirectoryWrapper),
    Mock(MockDirectoryWrapper),
}

impl Wrapped {
    fn boxed(self) -> Box<dyn BaseDirectoryWrapper> {
        match self {
            Wrapped::Raw(raw) => Box::new(raw),
            Wrapped::Mock(mock) => Box::new(mock),
        }
    }

    fn mock(self) -> MockDirectoryWrapper {
        match self {
            Wrapped::Mock(mock) => mock,
            Wrapped::Raw(_) => unreachable!("a mock directory is never bare"),
        }
    }
}

impl LuceneTestCase {
    /// The suite's mock-filesystem behaviours (`TestRuleTemporaryFilesCleanup`),
    /// drawn from the master seed.
    pub fn mock_fs_config(&self) -> MockFsConfig {
        MockFsConfig::random(&mut JavaRandom::new(self.master_seed()))
    }

    fn factory(&mut self, fs_config: MockFsConfig) -> Factory<'_> {
        let nightly = self.nightly();
        let multiplier = self.multiplier();
        Factory {
            random: self.random(),
            nightly,
            multiplier,
            fs_config,
            fs: None,
            temps: Vec::new(),
        }
    }

    /// `newDirectory()`.
    pub fn new_directory(&mut self) -> Box<dyn BaseDirectoryWrapper> {
        let config = self.mock_fs_config();
        let mut factory = self.factory(config);
        let dir = factory.directory_impl(None);
        let bare = factory.rarely();
        factory.wrap(dir, bare, false).boxed()
    }

    /// `newMockDirectory()`.
    pub fn new_mock_directory(&mut self) -> MockDirectoryWrapper {
        let config = self.mock_fs_config();
        let mut factory = self.factory(config);
        let dir = factory.directory_impl(None);
        factory.wrap(dir, false, false).mock()
    }

    /// `newDirectory(random, d)`: a new directory holding a copy of `d`'s
    /// index files (`segments*` and codec files).
    pub fn new_directory_from(
        &mut self,
        d: &dyn Directory,
    ) -> lucene_store::Result<Box<dyn BaseDirectoryWrapper>> {
        let config = self.mock_fs_config();
        let mut factory = self.factory(config);
        let dir = factory.directory_impl(None);
        for file in d.list_all()? {
            if file.starts_with("segments") || is_codec_file(&file) {
                dir.copy_from(d, &file, &file)?;
            }
        }
        let bare = factory.rarely();
        Ok(factory.wrap(dir, bare, false).boxed())
    }

    /// `newFSDirectory(path)`: an `NIOFSDirectory` or `MMapDirectory` over
    /// `path`, rarely bare.
    pub fn new_fs_directory(&mut self, path: &Path) -> Box<dyn BaseDirectoryWrapper> {
        let config = self.mock_fs_config();
        let bare = self.rarely();
        self.fs_directory_with(path, config, bare).boxed()
    }

    /// `newMockFSDirectory(path)`.
    pub fn new_mock_fs_directory(&mut self, path: &Path) -> MockDirectoryWrapper {
        let config = self.mock_fs_config();
        self.fs_directory_with(path, config, false).mock()
    }

    /// `newMaybeVirusCheckingDirectory()`: one time in five a filesystem
    /// directory over a fresh temp dir with a `VirusCheckingFS`.
    pub fn new_maybe_virus_checking_directory(&mut self) -> Box<dyn BaseDirectoryWrapper> {
        if self.random().next_int_bounded(5) == 4 {
            let temp = TempDir::new("tempDir");
            let config = self.mock_fs_config().with_virus_checker(self.random());
            let bare = self.rarely();
            let mut wrapped = self.fs_directory_with(temp.path(), config, bare);
            match &mut wrapped {
                Wrapped::Raw(raw) => raw.keep_alive(temp),
                Wrapped::Mock(mock) => mock.keep_alive(temp),
            }
            wrapped.boxed()
        } else {
            self.new_directory()
        }
    }

    /// `newFSDirectory(f, lf, bare)`.
    fn fs_directory_with(&mut self, path: &Path, config: MockFsConfig, bare: bool) -> Wrapped {
        let mut factory = self.factory(config);
        let kind = DirectoryImpl::random_fs(factory.random);
        let dir = factory.fs_directory(kind, path);
        factory.wrap(dir, bare, true)
    }
}

/// `IndexFileNames.CODEC_FILE_PATTERN`: `_[a-z0-9]+(_.*)?\..*`.
pub(crate) fn is_codec_file(name: &str) -> bool {
    // Java's default `.` does not match any of these line terminators.
    if name.contains(['\n', '\r', '\u{0085}', '\u{2028}', '\u{2029}']) {
        return false;
    }
    let Some(rest) = name.strip_prefix('_') else {
        return false;
    };
    let gen_len = rest
        .bytes()
        .take_while(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        .count();
    if gen_len == 0 {
        return false;
    }
    let rest = &rest[gen_len..];
    match rest.as_bytes().first() {
        Some(b'.') => true,
        Some(b'_') => rest.contains('.'),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_file_pattern_matches_javas() {
        for yes in [
            "_0.cfs",
            "_a1_Lucene90_0.dvd",
            "_z.si",
            "_0_1.liv",
            "_0.",
            "_0_.",
            "_0_x.y.z",
            "_0.\u{000b}\u{000c}\u{0000}",
        ] {
            assert!(is_codec_file(yes), "{yes}");
        }
        for no in ["segments_1", "_.cfs", "_0", "_A.cfs", "write.lock", "_0_x"] {
            assert!(!is_codec_file(no), "{no}");
        }
        for terminator in ['\n', '\r', '\u{0085}', '\u{2028}', '\u{2029}'] {
            for name in [
                format!("_0.{terminator}"),
                format!("_0_{terminator}.si"),
                format!("_0_x.{terminator}si"),
            ] {
                assert!(!is_codec_file(&name), "{name:?}");
            }
        }
    }

    /// `Collections.shuffle(FILE_SWITCH_EXTENSIONS, new Random(42))`, then
    /// `nextInt(13)`, as Java prints them.
    #[test]
    fn java_shuffle_is_collections_shuffle() {
        let mut r = JavaRandom::new(42);
        let mut extensions = FILE_SWITCH_EXTENSIONS;
        java_shuffle(&mut extensions, &mut r);
        assert_eq!(
            extensions,
            [
                "pos", "dvm", "fnm", "nvd", "nvm", "tim", "fdx", "dii", "dvd", "si", "dim", "tip",
                "fdt"
            ]
        );
        assert_eq!(r.next_int_bounded(13), 8);
    }
}
