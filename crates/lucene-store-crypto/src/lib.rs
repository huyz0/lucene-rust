#![forbid(unsafe_code)]
#![deny(unsafe_code)]
//! Application-level encryption (ALE) for lucene-store directories, with
//! bring-your-own-key (BYOK) key providers.
//!
//! - [`EncryptedDirectory`] wraps any [`lucene_store::Directory`] and
//!   encrypts each file with its own AES-256-GCM data key
//!   ([`mod@format`] has the layout).
//! - [`KeyProvider`] wraps and unwraps those data keys: implement it over a
//!   customer's KMS for BYOK. [`StaticKeyProvider`] holds keys in memory;
//!   [`CachingKeyProvider`] bounds KMS round trips.
//! - [`register`] adds the `encrypted` store wrapper to lucene-store's
//!   registry ([`lucene_store::StoreSpec`]), reading the setting
//!   [`SETTING_KEY_PROVIDER`] (a provider registered with
//!   [`register_key_provider`]) and, optionally, [`SETTING_CHUNK_SIZE`].
//!
//! The crypto is AWS-LC through `aws-lc-rs` ([`backend`]): the regular build
//! by default, or AWS-LC's FIPS 140-3 validated module with feature `fips`.
//!
//! ```
//! use std::sync::Arc;
//! use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
//! use lucene_store_crypto::{EncryptedDirectory, StaticKeyProvider};
//!
//! let keys = Arc::new(StaticKeyProvider::new("tenant-a/v1", &[7; 32]));
//! let dir = EncryptedDirectory::new(ByteBuffersDirectory::new(), keys);
//! let mut out = dir.create_output("_0.si").unwrap();
//! out.write_bytes(b"segment info");
//! out.close().unwrap();
//! assert_eq!(&*dir.open("_0.si").unwrap(), b"segment info");
//! assert_ne!(&*dir.inner().open("_0.si").unwrap(), b"segment info");
//! ```

pub mod backend;
pub mod directory;
pub mod format;
pub mod key_provider;
mod pages;

use std::sync::Arc;

use lucene_store::{register_store_wrapper, Directory, Error, StoreSettings};

pub use backend::{backend_name, is_fips};
pub use directory::{is_commit_metadata, EncryptedDirectory, PlaintextCache};
pub use format::DataKey;
pub use key_provider::{
    key_provider, key_unavailable, register_key_provider, CachingKeyProvider, GeneratedKey,
    KeyProvider, StaticKeyProvider,
};

/// The store wrapper's name in lucene-store's registry.
pub const WRAPPER_NAME: &str = "encrypted";
/// The setting naming the registered [`KeyProvider`] to use.
pub const SETTING_KEY_PROVIDER: &str = "encryption.key_provider";
/// The optional setting for new files' chunk size, in bytes.
pub const SETTING_CHUNK_SIZE: &str = "encryption.chunk_size";

/// Registers the [`WRAPPER_NAME`] store wrapper. Idempotent.
pub fn register() {
    register_store_wrapper(WRAPPER_NAME, Arc::new(wrap));
}

fn wrap(
    inner: Box<dyn Directory>,
    settings: &StoreSettings,
) -> lucene_store::Result<Box<dyn Directory>> {
    let name = settings.get(SETTING_KEY_PROVIDER).ok_or_else(|| {
        Error::IllegalArgument(format!(
            "the [{WRAPPER_NAME}] store wrapper needs the [{SETTING_KEY_PROVIDER}] setting"
        ))
    })?;
    let mut dir = EncryptedDirectory::new(inner, key_provider(name)?);
    if let Some(size) = settings.get(SETTING_CHUNK_SIZE) {
        let size = size.parse::<u32>().map_err(|_| {
            Error::IllegalArgument(format!("[{SETTING_CHUNK_SIZE}] is not a number: {size}"))
        })?;
        dir = dir.with_chunk_size(size)?;
    }
    Ok(Box::new(dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::{DataOutput, StoreSpec};
    use lucene_util::test_support::TempDir;

    #[test]
    fn the_registered_wrapper_encrypts_a_filesystem_store() {
        register();
        register_key_provider("lib-test", Arc::new(StaticKeyProvider::new("k", &[4; 32])));
        let tmp = TempDir::new("crypto-registry");
        let spec = StoreSpec::of("niofs")
            .wrapped(WRAPPER_NAME)
            .with(SETTING_KEY_PROVIDER, "lib-test")
            .with(SETTING_CHUNK_SIZE, "1024");
        let dir = spec.open(tmp.path()).unwrap();
        let mut out = dir.create_output("_0.fdt").unwrap();
        out.write_bytes(b"stored field bytes");
        out.close().unwrap();
        assert_eq!(&*dir.open("_0.fdt").unwrap(), b"stored field bytes");
        let raw = std::fs::read(tmp.path().join("_0.fdt")).unwrap();
        assert!(raw.starts_with(b"LRSE"));
        assert_eq!(dir.fs_directory_path(), Some(tmp.path()));

        let err = |spec: StoreSpec| spec.open(tmp.path()).err().unwrap().to_string();
        let base = StoreSpec::of("niofs").wrapped(WRAPPER_NAME);
        assert!(err(base.clone()).contains(SETTING_KEY_PROVIDER));
        assert!(
            err(base.clone().with(SETTING_KEY_PROVIDER, "none")).contains("unknown key provider")
        );
        let bad = base.with(SETTING_KEY_PROVIDER, "lib-test");
        assert!(err(bad.clone().with(SETTING_CHUNK_SIZE, "x")).contains("not a number"));
        assert!(err(bad.with(SETTING_CHUNK_SIZE, "0")).contains("chunk size 0"));
    }
}
