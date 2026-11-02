//! Named, pluggable [`Directory`] implementations: the Rust counterpart of
//! OpenSearch's `index.store.type` (`IndexStorePlugin.DirectoryFactory`)
//! and of wrapping a store in `FilterDirectory`s.
//!
//! A *store type* opens the base directory for a path ([`StoreProvider`]);
//! *wrappers* ([`StoreWrapper`]) are layered on top in order, each seeing
//! the directory below it -- encryption, caching, metering, a blob store's
//! local cache. Both are looked up by name in a process-wide registry, so a
//! host (the OpenSearch plugin, an embedding application) can pick them from
//! configuration ([`StoreSpec`]) without knowing their Rust types, and a
//! third-party crate adds its own with [`register_store_type`] /
//! [`register_store_wrapper`] before the first index is opened.
//!
//! Built in, under OpenSearch's names: `fs` and `hybridfs` and `mmapfs`
//! ([`MmapDirectory`], which reads small files and maps large ones) and
//! `niofs` and `simplefs` ([`FsDirectory`]).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use crate::directory::{Directory, FsDirectory, MmapDirectory};
use crate::error::{Error, Result};

/// A store's configuration: `key -> value` strings, as index settings carry
/// them. Each provider and wrapper reads the keys it owns (an encrypting
/// wrapper's key id, a blob store's bucket) and ignores the rest.
pub type StoreSettings = BTreeMap<String, String>;

/// Opens a base [`Directory`] at a path. Implemented for any matching `Fn`.
pub trait StoreProvider: Send + Sync {
    /// The directory for `path` (a local shard path; a remote store may use
    /// it only as a cache location or a key prefix).
    fn open(&self, path: &Path, settings: &StoreSettings) -> Result<Box<dyn Directory>>;
}

impl<F> StoreProvider for F
where
    F: Fn(&Path, &StoreSettings) -> Result<Box<dyn Directory>> + Send + Sync,
{
    fn open(&self, path: &Path, settings: &StoreSettings) -> Result<Box<dyn Directory>> {
        self(path, settings)
    }
}

/// Wraps a [`Directory`] in another (Lucene's `FilterDirectory` pattern).
/// Implemented for any matching `Fn`.
pub trait StoreWrapper: Send + Sync {
    /// `inner` wrapped.
    fn wrap(
        &self,
        inner: Box<dyn Directory>,
        settings: &StoreSettings,
    ) -> Result<Box<dyn Directory>>;
}

impl<F> StoreWrapper for F
where
    F: Fn(Box<dyn Directory>, &StoreSettings) -> Result<Box<dyn Directory>> + Send + Sync,
{
    fn wrap(
        &self,
        inner: Box<dyn Directory>,
        settings: &StoreSettings,
    ) -> Result<Box<dyn Directory>> {
        self(inner, settings)
    }
}

/// What to open: a store type, the wrappers over it (innermost first) and
/// the settings they all read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreSpec {
    /// A registered store type; empty means `fs`.
    pub store_type: String,
    /// Registered wrapper names, applied in order: the first wraps the base
    /// directory, the last is what the caller gets.
    pub wrappers: Vec<String>,
    /// Settings for the provider and every wrapper.
    pub settings: StoreSettings,
}

impl StoreSpec {
    /// A spec for `store_type` with no wrappers or settings.
    pub fn of(store_type: &str) -> Self {
        Self {
            store_type: store_type.to_string(),
            ..Self::default()
        }
    }

    /// Adds a wrapper on top of those already listed.
    pub fn wrapped(mut self, wrapper: &str) -> Self {
        self.wrappers.push(wrapper.to_string());
        self
    }

    /// Sets one setting.
    pub fn with(mut self, key: &str, value: &str) -> Self {
        self.settings.insert(key.to_string(), value.to_string());
        self
    }

    /// Opens the store at `path`: the base directory, then each wrapper.
    pub fn open(&self, path: &Path) -> Result<Box<dyn Directory>> {
        let store_type = if self.store_type.is_empty() {
            "fs"
        } else {
            &self.store_type
        };
        let provider = registry_read(|r| r.types.get(store_type).cloned()).ok_or_else(|| {
            Error::IllegalArgument(format!(
                "unknown store type [{store_type}]; registered: {:?}",
                registered_store_types()
            ))
        })?;
        let mut dir = provider.open(path, &self.settings)?;
        for name in &self.wrappers {
            let wrapper =
                registry_read(|r| r.wrappers.get(name.as_str()).cloned()).ok_or_else(|| {
                    Error::IllegalArgument(format!(
                        "unknown store wrapper [{name}]; registered: {:?}",
                        registered_store_wrappers()
                    ))
                })?;
            dir = wrapper.wrap(dir, &self.settings)?;
        }
        Ok(dir)
    }
}

struct Registry {
    types: BTreeMap<String, Arc<dyn StoreProvider>>,
    wrappers: BTreeMap<String, Arc<dyn StoreWrapper>>,
}

fn registry() -> &'static RwLock<Registry> {
    static REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut types: BTreeMap<String, Arc<dyn StoreProvider>> = BTreeMap::new();
        let mmap: Arc<dyn StoreProvider> = Arc::new(|path: &Path, _: &StoreSettings| {
            Ok(Box::new(MmapDirectory::open(path)) as Box<dyn Directory>)
        });
        let fs: Arc<dyn StoreProvider> = Arc::new(|path: &Path, _: &StoreSettings| {
            Ok(Box::new(FsDirectory::open(path)) as Box<dyn Directory>)
        });
        for name in ["fs", "hybridfs", "mmapfs"] {
            types.insert(name.to_string(), Arc::clone(&mmap));
        }
        for name in ["niofs", "simplefs"] {
            types.insert(name.to_string(), Arc::clone(&fs));
        }
        RwLock::new(Registry {
            types,
            wrappers: BTreeMap::new(),
        })
    })
}

fn registry_read<T>(f: impl FnOnce(&Registry) -> T) -> T {
    let guard = registry().read().unwrap_or_else(|p| p.into_inner());
    f(&guard)
}

fn registry_write<T>(f: impl FnOnce(&mut Registry) -> T) -> T {
    let mut guard = registry().write().unwrap_or_else(|p| p.into_inner());
    f(&mut guard)
}

/// Registers (or replaces) the store type `name`.
pub fn register_store_type(name: &str, provider: Arc<dyn StoreProvider>) {
    registry_write(|r| r.types.insert(name.to_string(), provider));
}

/// Registers (or replaces) the store wrapper `name`.
pub fn register_store_wrapper(name: &str, wrapper: Arc<dyn StoreWrapper>) {
    registry_write(|r| r.wrappers.insert(name.to_string(), wrapper));
}

/// Every registered store type, sorted.
pub fn registered_store_types() -> Vec<String> {
    registry_read(|r| r.types.keys().cloned().collect())
}

/// Every registered store wrapper, sorted.
pub fn registered_store_wrappers() -> Vec<String> {
    registry_read(|r| r.wrappers.keys().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_output::DataOutput;
    use crate::ByteBuffersDirectory;
    use lucene_util::test_support::TempDir;

    #[test]
    fn built_in_types_open_filesystem_directories() {
        let tmp = TempDir::new("store-registry");
        for t in ["", "fs", "hybridfs", "mmapfs", "niofs", "simplefs"] {
            let dir = StoreSpec::of(t).open(tmp.path()).unwrap();
            assert_eq!(dir.fs_directory_path(), Some(tmp.path()), "{t}");
        }
        let types = registered_store_types();
        assert!(types.contains(&"mmapfs".to_string()), "{types:?}");
    }

    #[test]
    fn unknown_names_list_what_is_registered() {
        let tmp = TempDir::new("store-registry-unknown");
        let err = StoreSpec::of("s3")
            .open(tmp.path())
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("unknown store type [s3]"), "{err}");
        assert!(err.contains("niofs"), "{err}");
        let err = StoreSpec::of("fs")
            .wrapped("no-such-wrapper")
            .open(tmp.path())
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("unknown store wrapper [no-such-wrapper]"),
            "{err}"
        );
    }

    #[test]
    fn third_party_types_and_wrappers_compose_in_order() {
        register_store_type(
            "test-memory",
            Arc::new(|_: &Path, _: &StoreSettings| {
                Ok(Box::new(ByteBuffersDirectory::new()) as Box<dyn Directory>)
            }),
        );
        // A wrapper that reads its setting; it proves it ran by writing a
        // file, and the second proves the order by seeing that file.
        register_store_wrapper(
            "test-marker",
            Arc::new(|inner: Box<dyn Directory>, s: &StoreSettings| {
                let name = s.get("marker").cloned().unwrap_or_default();
                let mut out = inner.create_output(&name)?;
                out.write_byte(1);
                out.close()?;
                Ok(inner)
            }),
        );
        register_store_wrapper(
            "test-estimate",
            Arc::new(|inner: Box<dyn Directory>, s: &StoreSettings| {
                assert!(inner.list_all()?.contains(&s["marker"]));
                Ok(inner)
            }),
        );
        let spec = StoreSpec::of("test-memory")
            .wrapped("test-marker")
            .wrapped("test-estimate")
            .with("marker", "m");
        let dir = spec.clone().open(Path::new("/unused")).unwrap();
        assert_eq!(dir.list_all().unwrap(), vec!["m".to_string()]);
        assert!(registered_store_wrappers().contains(&"test-marker".to_string()));
        assert_eq!(spec.wrappers, vec!["test-marker", "test-estimate"]);
    }
}
