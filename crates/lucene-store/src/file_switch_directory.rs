//! Port of `org.apache.lucene.store.FileSwitchDirectory`: two directories
//! presented as one, each file routed by its extension -- files whose
//! extension is in the primary set live in the primary directory, all others
//! in the secondary. The classic use is two views of one path (NIOFS for
//! most files, MMap for the term dictionary and doc values).

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use crate::directory::{temp_file_name, Directory, Input};
use crate::error::{Error, Result};
use crate::index_output::FsIndexOutput;
use crate::lock::Lock;

/// Port of `FileSwitchDirectory`. Owns both directories (Java's
/// `doClose = true`); pass references to share them instead.
pub struct FileSwitchDirectory<P, S> {
    primary_extensions: HashSet<String>,
    primary_dir: P,
    secondary_dir: S,
}

/// Which side a name routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Primary,
    Secondary,
}

impl<P: Directory, S: Directory> FileSwitchDirectory<P, S> {
    /// `new FileSwitchDirectory(primaryExtensions, primaryDir, secondaryDir,
    /// doClose)`. `tmp` is reserved: temp files route by the extension
    /// embedded in their name (see [`get_extension`]).
    pub fn new(
        primary_extensions: HashSet<String>,
        primary_dir: P,
        secondary_dir: S,
    ) -> Result<Self> {
        if primary_extensions.contains("tmp") {
            return Err(Error::IllegalArgument(
                "tmp is a reserved extension".to_string(),
            ));
        }
        Ok(Self {
            primary_extensions,
            primary_dir,
            secondary_dir,
        })
    }

    /// `getPrimaryDir()`.
    pub fn primary_dir(&self) -> &P {
        &self.primary_dir
    }

    /// `getSecondaryDir()`.
    pub fn secondary_dir(&self) -> &S {
        &self.secondary_dir
    }

    fn side(&self, name: &str) -> Side {
        if self.primary_extensions.contains(get_extension(name)) {
            Side::Primary
        } else {
            Side::Secondary
        }
    }

    /// `getDirectory(name)`.
    fn directory(&self, name: &str) -> &dyn Directory {
        match self.side(name) {
            Side::Primary => &self.primary_dir,
            Side::Secondary => &self.secondary_dir,
        }
    }
}

/// Port of `FileSwitchDirectory.getExtension(name)`: the text after the last
/// `.`, or `""` without one -- except for a `.tmp` file, which takes the
/// first `.letters` run *before* its `.tmp` (`_0.fdt_sort_1.tmp` is `fdt`),
/// so a temp file lands beside the files it becomes.
pub fn get_extension(name: &str) -> &str {
    let Some(i) = name.rfind('.') else {
        return "";
    };
    let ext = &name[i..][1..];
    if ext == "tmp" {
        // `Pattern.compile("\\.([a-zA-Z]+)").matcher(name.substring(0, i +
        // 1)).find()`: the leftmost `.` followed by at least one ASCII letter.
        let head = &name[..=i];
        for (dot, _) in head.match_indices('.') {
            let rest = &head[dot..][1..];
            let letters = rest
                .find(|c: char| !c.is_ascii_alphabetic())
                .unwrap_or(rest.len());
            if letters > 0 {
                return &rest[..letters];
            }
        }
    }
    ext
}

impl<P: Directory, S: Directory> Directory for FileSwitchDirectory<P, S> {
    /// Each side lists only the files that route to it, so two views of one
    /// path never list a file twice. A missing directory on one side is
    /// tolerated as long as the other lists something (LUCENE-3380).
    fn list_all(&self) -> Result<Vec<String>> {
        let mut files = Vec::new();
        let mut primary_missing = None;
        match self.primary_dir.list_all() {
            Ok(names) => files.extend(
                names
                    .into_iter()
                    .filter(|f| self.primary_extensions.contains(get_extension(f))),
            ),
            Err(e) if e.is_no_such_file() => primary_missing = Some(e),
            Err(e) => return Err(e),
        }
        match self.secondary_dir.list_all() {
            Ok(names) => files.extend(
                names
                    .into_iter()
                    .filter(|f| !self.primary_extensions.contains(get_extension(f))),
            ),
            Err(e) if e.is_no_such_file() => {
                // Both missing: rethrow the first.
                if let Some(first) = primary_missing {
                    return Err(first);
                }
                // The secondary is missing and the primary is empty.
                if files.is_empty() {
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
        // The primary is missing and the secondary is empty.
        if let Some(first) = primary_missing {
            if files.is_empty() {
                return Err(first);
            }
        }
        files.sort();
        Ok(files)
    }

    fn open(&self, name: &str) -> Result<Input> {
        self.directory(name).open(name)
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.directory(name).file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.directory(name).create_output(name)
    }

    /// Best effort, as Java's: the side is chosen from a representative temp
    /// name, since the real one is only known once created.
    fn create_temp_output(&self, prefix: &str, suffix: &str) -> Result<FsIndexOutput> {
        let tmp_file_name = temp_file_name(prefix, suffix, 0);
        self.directory(&tmp_file_name)
            .create_temp_output(prefix, suffix)
    }

    fn sync(&self, names: &[String]) -> Result<()> {
        let (primary, secondary): (Vec<String>, Vec<String>) = names
            .iter()
            .cloned()
            .partition(|n| self.side(n) == Side::Primary);
        self.primary_dir.sync(&primary)?;
        self.secondary_dir.sync(&secondary)
    }

    /// `AtomicMoveNotSupportedException` (here `ErrorKind::CrossesDevices`)
    /// when the two names route to different sides.
    fn rename(&self, source: &str, dest: &str) -> Result<()> {
        if self.side(source) != self.side(dest) {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::CrossesDevices,
                format!("{source} -> {dest}: source and dest are in different directories"),
            )));
        }
        self.directory(source).rename(source, dest)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.directory(name).delete_file(name)
    }

    fn sync_meta_data(&self) -> Result<()> {
        self.primary_dir.sync_meta_data()?;
        self.secondary_dir.sync_meta_data()
    }

    fn obtain_lock(&self, name: &str) -> Result<Box<dyn Lock>> {
        self.directory(name).obtain_lock(name)
    }

    fn pending_deletions(&self) -> Result<BTreeSet<String>> {
        let mut all = self.primary_dir.pending_deletions()?;
        all.extend(self.secondary_dir.pending_deletions()?);
        Ok(all)
    }

    /// Not an `FSDirectory` itself (Java's `instanceof` fails); locking goes
    /// through [`Directory::obtain_lock`], which routes to a side.
    fn fs_directory_path(&self) -> Option<&Path> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;
    use crate::index_output::IndexOutput;
    use crate::{ByteBuffersDirectory, FsDirectory};
    use lucene_util::test_support::TempDir;

    fn exts(e: &[&str]) -> HashSet<String> {
        e.iter().map(|s| s.to_string()).collect()
    }

    fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }

    #[test]
    fn get_extension_matches_java() {
        assert_eq!(get_extension("_0.fdt"), "fdt");
        assert_eq!(get_extension("segments_1"), "");
        assert_eq!(get_extension("write.lock"), "lock");
        assert_eq!(get_extension("a.b.c"), "c");
        assert_eq!(get_extension("trailing."), "");
        // Temp files route by the extension inside them.
        assert_eq!(get_extension("_0.fdt_sort_1.tmp"), "fdt");
        assert_eq!(get_extension("_0_sort_1.tmp"), "tmp");
        assert_eq!(get_extension(".1.tmp"), "tmp");
        assert_eq!(get_extension("x.1a.tim.tmp"), "tim");
    }

    #[test]
    fn tmp_is_a_reserved_extension() {
        let err = FileSwitchDirectory::new(
            exts(&["tmp"]),
            ByteBuffersDirectory::new(),
            ByteBuffersDirectory::new(),
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::IllegalArgument(_)));
    }

    /// Mirrors `TestFileSwitchDirectory.testBasic`: files land in the
    /// directory their extension picks, and the union lists both.
    #[test]
    fn files_route_by_extension() {
        let dir = FileSwitchDirectory::new(
            exts(&["fdt", "fdx"]),
            ByteBuffersDirectory::new(),
            ByteBuffersDirectory::new(),
        )
        .unwrap();
        write(&dir, "_0.fdt", b"stored");
        write(&dir, "_0.fdx", b"index");
        write(&dir, "_0.tim", b"terms");
        write(&dir, "segments_1", b"commit");
        assert_eq!(
            dir.primary_dir().list_all().unwrap(),
            vec!["_0.fdt", "_0.fdx"]
        );
        assert_eq!(
            dir.secondary_dir().list_all().unwrap(),
            vec!["_0.tim", "segments_1"]
        );
        assert_eq!(
            dir.list_all().unwrap(),
            vec!["_0.fdt", "_0.fdx", "_0.tim", "segments_1"]
        );
        assert_eq!(&*dir.open("_0.fdt").unwrap(), b"stored");
        assert_eq!(dir.file_length("_0.tim").unwrap(), 5);
        dir.sync(&["_0.fdt".to_string(), "_0.tim".to_string()])
            .unwrap();
        dir.sync_meta_data().unwrap();
        dir.delete_file("_0.fdx").unwrap();
        assert!(!dir.primary_dir().file_exists("_0.fdx"));
        assert!(dir.pending_deletions().unwrap().is_empty());
        assert!(dir.fs_directory_path().is_none());

        // A rename across the two sides cannot be atomic.
        let err = dir.rename("_0.fdt", "_0.tim2").unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::CrossesDevices),
            "{err}"
        );
        dir.rename("segments_1", "segments_2").unwrap();
        assert!(dir.secondary_dir().file_exists("segments_2"));

        // Temp files go where their embedded extension says.
        let tmp = dir.create_temp_output("_0.fdt", "sort").unwrap();
        assert!(dir.primary_dir().file_exists(tmp.name()));
        tmp.close().unwrap();
        let tmp = dir.create_temp_output("_0", "sort").unwrap();
        assert!(dir.secondary_dir().file_exists(tmp.name()));

        // The lock routes by its own extension.
        let lock = dir.obtain_lock("write.lock").unwrap();
        assert!(matches!(
            dir.secondary_dir().obtain_lock("write.lock"),
            Err(Error::LockObtainFailed(_))
        ));
        lock.close().unwrap();
    }

    /// `TestFileSwitchDirectory.testNoDir` / LUCENE-3380: listing tolerates a
    /// side whose directory does not exist yet, but not both.
    #[test]
    fn list_all_tolerates_one_missing_side() {
        let root = TempDir::new("file-switch");
        let missing = root.join("missing");
        let present = root.join("present");
        std::fs::create_dir(&present).unwrap();

        // Both missing: the primary's error.
        let both = FileSwitchDirectory::new(
            exts(&["fdt"]),
            FsDirectory::open(&missing),
            FsDirectory::open(root.join("also-missing")),
        )
        .unwrap();
        assert!(both.list_all().unwrap_err().is_no_such_file());

        // Primary missing, secondary empty: still an error.
        let one = FileSwitchDirectory::new(
            exts(&["fdt"]),
            FsDirectory::open(&missing),
            FsDirectory::open(&present),
        )
        .unwrap();
        assert!(one.list_all().unwrap_err().is_no_such_file());
        // ... but not once the secondary has a file.
        write(&one, "segments_1", b"c");
        assert_eq!(one.list_all().unwrap(), vec!["segments_1"]);

        // Secondary missing: fine while the primary lists something.
        let other = FileSwitchDirectory::new(
            exts(&["si"]),
            FsDirectory::open(&present),
            FsDirectory::open(&missing),
        )
        .unwrap();
        assert!(other.list_all().unwrap_err().is_no_such_file());
        write(&other, "_0.si", b"s");
        assert_eq!(other.list_all().unwrap(), vec!["_0.si"]);

        // Two views of one path list each file once.
        let same = FileSwitchDirectory::new(
            exts(&["si"]),
            FsDirectory::open(&present),
            FsDirectory::open(&present),
        )
        .unwrap();
        assert_eq!(same.list_all().unwrap(), vec!["_0.si", "segments_1"]);
    }

    #[test]
    fn list_all_propagates_other_errors() {
        let root = TempDir::new("file-switch-err");
        let file = root.join("a-file");
        std::fs::write(&file, b"").unwrap();
        // Listing a regular file is not "no such file".
        let dir = FileSwitchDirectory::new(
            exts(&["fdt"]),
            FsDirectory::open(&file),
            ByteBuffersDirectory::new(),
        )
        .unwrap();
        assert!(!dir.list_all().unwrap_err().is_no_such_file());
        let dir = FileSwitchDirectory::new(
            exts(&["fdt"]),
            ByteBuffersDirectory::new(),
            FsDirectory::open(&file),
        )
        .unwrap();
        assert!(!dir.list_all().unwrap_err().is_no_such_file());
    }
}
