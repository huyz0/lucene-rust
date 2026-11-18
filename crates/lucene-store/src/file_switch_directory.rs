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

    fn open_aligned(&self, name: &str, alignment: usize) -> Result<Input> {
        self.directory(name).open_aligned(name, alignment)
    }

    fn open_member(
        &self,
        container: &str,
        offset: u64,
        len: u64,
        member: &str,
    ) -> Result<Option<Input>> {
        self.directory(container)
            .open_member(container, offset, len, member)
    }

    #[cfg(any(test, feature = "test-util"))]
    fn close_input(&self, name: &str, input: Input) -> Result<()> {
        self.directory(name).close_input(name, input)
    }

    fn file_length(&self, name: &str) -> Result<u64> {
        self.directory(name).file_length(name)
    }

    fn create_output(&self, name: &str) -> Result<FsIndexOutput> {
        self.directory(name).create_output(name)
    }

    fn create_output_with_estimate(
        &self,
        name: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        self.directory(name)
            .create_output_with_estimate(name, estimated_bytes)
    }

    fn create_temp_output_with_estimate(
        &self,
        prefix: &str,
        suffix: &str,
        estimated_bytes: Option<u64>,
    ) -> Result<FsIndexOutput> {
        let tmp_file_name = temp_file_name(prefix, suffix, 0);
        self.directory(&tmp_file_name)
            .create_temp_output_with_estimate(prefix, suffix, estimated_bytes)
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
mod tests;
