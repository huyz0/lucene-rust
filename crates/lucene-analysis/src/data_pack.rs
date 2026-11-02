//! Data packs: the dictionaries of the language modules (`kuromoji`, `nori`,
//! `smartcn`, `morfologik`, `stempel`), looked up at run time instead of only
//! compiled in.
//!
//! Java finds a module's dictionary as a resource of the module's jar, so the
//! jar on the classpath is the opt-in. Here a module's code is always small
//! and always linkable; its dictionary files are what weigh. A module reads
//! each file through [`load`]: from the first configured pack directory that
//! has `<pack>/<file>`, else from the copy compiled in (each module's
//! `embedded-data` feature, on by default), else a [`FactoryError`] naming
//! the file and the directories searched.
//!
//! Pack directories are [`add_pack_dir`]'s, in the order added, then the
//! `LUCENE_RUST_PACKS` environment variable's (a path list, as `PATH`). A
//! pack is a plain directory of the module's files, byte for byte the
//! `src/resources` files of its crate (`scripts/make-data-packs.sh` writes
//! them), so a build without `embedded-data` and a pack directory reads
//! exactly what an embedded build reads.
//!
//! A module checks its whole pack in its factories' constructors
//! ([`require`]), so a missing pack refuses the analyzer cleanly; the
//! dictionaries' own `instance()` accessors read on first use and may then
//! assume the files are there.

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::factory::FactoryError;

/// The environment variable read for pack directories.
pub const PACKS_ENV: &str = "LUCENE_RUST_PACKS";

static DIRS: RwLock<Vec<PathBuf>> = RwLock::new(Vec::new());

/// Adds a directory to search for packs, after those added before. Packs
/// already read stay as read.
pub fn add_pack_dir(dir: impl Into<PathBuf>) {
    let dir = dir.into();
    let mut dirs = DIRS.write().unwrap_or_else(|e| e.into_inner());
    if !dirs.contains(&dir) {
        dirs.push(dir);
    }
}

/// The directories searched, in order.
pub fn pack_dirs() -> Vec<PathBuf> {
    let mut dirs = DIRS.read().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(env) = std::env::var_os(PACKS_ENV) {
        dirs.extend(std::env::split_paths(&env).filter(|p| !p.as_os_str().is_empty()));
    }
    dirs
}

/// One file of a pack: its pack and file name, and the copy compiled in.
#[derive(Clone, Copy, Debug)]
pub struct Resource {
    pub pack: &'static str,
    pub file: &'static str,
    pub embedded: Option<&'static [u8]>,
}

impl Resource {
    fn find(&self) -> Option<PathBuf> {
        pack_dirs()
            .into_iter()
            .map(|d| d.join(self.pack).join(self.file))
            .find(|p| p.is_file())
    }

    fn missing(&self) -> FactoryError {
        let dirs: Vec<_> = pack_dirs()
            .iter()
            .map(|d| d.display().to_string())
            .collect();
        FactoryError::io(format!(
            "data pack {}: {}/{} is not in any pack directory ({}; {PACKS_ENV}), and this build has no copy compiled in",
            self.pack,
            self.pack,
            self.file,
            if dirs.is_empty() { "none configured".to_string() } else { dirs.join(", ") },
        ))
    }

    /// Whether the file can be read, without reading it.
    pub fn available(&self) -> bool {
        self.embedded.is_some() || self.find().is_some()
    }
}

/// The bytes of `r`: a pack directory's file, else the copy compiled in.
pub fn load(r: &Resource) -> Result<std::borrow::Cow<'static, [u8]>, FactoryError> {
    if let Some(path) = r.find() {
        return read(&path).map(std::borrow::Cow::Owned);
    }
    r.embedded
        .map(std::borrow::Cow::Borrowed)
        .ok_or_else(|| r.missing())
}

fn read(path: &Path) -> Result<Vec<u8>, FactoryError> {
    std::fs::read(path).map_err(|e| FactoryError::io(format!("data pack: {}: {e}", path.display())))
}

/// Every resource of a pack readable, else the first missing one's error.
pub fn require(resources: &[Resource]) -> Result<(), FactoryError> {
    match resources.iter().find(|r| !r.available()) {
        Some(r) => Err(r.missing()),
        None => Ok(()),
    }
}

/// A [`Resource`] of the calling crate: `pack_resource!("kuromoji",
/// "../resources/x.dat.z")` names pack `kuromoji`'s file `x.dat.z`, with the
/// file at that path (relative to the calling source file, as
/// `include_bytes!`'s) compiled in when the calling crate's `embedded-data`
/// feature is on.
#[macro_export]
macro_rules! pack_resource {
    ($pack:literal, $path:literal) => {
        $crate::data_pack::Resource {
            pack: $pack,
            file: $crate::data_pack::file_name($path),
            embedded: {
                #[cfg(feature = "embedded-data")]
                const EMBEDDED: Option<&'static [u8]> = Some(include_bytes!($path) as &[u8]);
                #[cfg(not(feature = "embedded-data"))]
                const EMBEDDED: Option<&'static [u8]> = None;
                EMBEDDED
            },
        }
    };
}

/// The last component of a `/`-separated path, at compile time.
pub const fn file_name(path: &'static str) -> &'static str {
    let b = path.as_bytes();
    let mut i = b.len();
    while i > 0 && b[i - 1] != b'/' {
        i -= 1;
    }
    let (_, tail) = b.split_at(i);
    match std::str::from_utf8(tail) {
        Ok(s) => s,
        Err(_) => path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_is_the_last_component() {
        assert_eq!(file_name("../resources/x.dat.z"), "x.dat.z");
        assert_eq!(file_name("x.dat.z"), "x.dat.z");
    }

    #[test]
    fn a_pack_directory_wins_and_a_missing_file_names_itself() {
        let dir =
            std::env::temp_dir().join(format!("lucene-rust-pack-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("testpack")).unwrap();
        std::fs::write(dir.join("testpack").join("a.bin"), b"from disk").unwrap();
        let on_disk = Resource {
            pack: "testpack",
            file: "a.bin",
            embedded: Some(b"compiled in"),
        };
        let embedded_only = Resource {
            pack: "testpack",
            file: "b.bin",
            embedded: Some(b"compiled in"),
        };
        let missing = Resource {
            pack: "testpack",
            file: "c.bin",
            embedded: None,
        };
        assert_eq!(&*load(&embedded_only).unwrap(), b"compiled in");
        let err = require(&[embedded_only, missing]).unwrap_err().to_string();
        assert!(err.contains("testpack/c.bin"), "{err}");
        add_pack_dir(&dir);
        add_pack_dir(&dir);
        assert_eq!(pack_dirs().iter().filter(|d| **d == dir).count(), 1);
        assert_eq!(&*load(&on_disk).unwrap(), b"from disk");
        assert_eq!(&*load(&embedded_only).unwrap(), b"compiled in");
        assert!(load(&missing).is_err());
        std::fs::write(dir.join("testpack").join("c.bin"), b"now").unwrap();
        require(&[embedded_only, missing]).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
