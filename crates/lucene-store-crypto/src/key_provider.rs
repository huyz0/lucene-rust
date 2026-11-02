//! Where data keys are wrapped and unwrapped: the bring-your-own-key seam.
//!
//! [`KeyProvider`] is envelope encryption's key-encryption side. Each file's
//! random data key is wrapped once ([`KeyProvider::generate_data_key`]), when the file
//! is created, and the wrapped bytes are stored in the file's header; opening
//! the file calls [`KeyProvider::unwrap_key`]. A customer-managed key in a KMS
//! (AWS KMS `Encrypt`/`Decrypt`, Azure Key Vault `wrapKey`/`unwrapKey`, GCP
//! KMS, Vault Transit, an HSM over PKCS#11) is one implementation: the key
//! material never leaves the KMS, and revoking it there makes the index
//! unreadable. [`StaticKeyProvider`] keeps keys in process memory, for tests
//! and for applications that fetch their key from a secret store at start.
//!
//! [`CachingKeyProvider`] puts a time-bounded cache in front of a remote
//! provider so opening a segment's dozen files is not a dozen KMS calls.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::backend::{fill_random, Aes256Gcm};
use lucene_store::{Error, Result};

use crate::format::{DataKey, DATA_KEY_LEN, TAG_LEN};

/// A fresh data key for one file: its plaintext, and the same key wrapped
/// under the key-encryption key `key_id`.
pub struct GeneratedKey {
    /// The id of the key-encryption key that wrapped `data_key`.
    pub key_id: String,
    /// The data key itself.
    pub data_key: DataKey,
    /// `data_key`, wrapped: what the file's header stores.
    pub wrapped: Vec<u8>,
}

/// Wraps and unwraps per-file data keys under named key-encryption keys.
///
/// Implement either [`KeyProvider::current_key_id`] plus
/// [`KeyProvider::wrap_key`] (this crate draws the data key), or
/// [`KeyProvider::generate_data_key`] (the KMS draws it, as AWS KMS
/// `GenerateDataKey` and OpenSearch's `MasterKeyProvider.generateDataPair`
/// do), plus [`KeyProvider::unwrap_key`] in both cases.
pub trait KeyProvider: Send + Sync {
    /// The id of the key new files are wrapped under. Changing what this
    /// returns rotates the key for files written from then on.
    fn current_key_id(&self) -> Result<String> {
        Err(key_unavailable(
            "this key provider only generates data keys",
        ))
    }

    /// Encrypts `data_key` under the key `key_id`.
    fn wrap_key(&self, key_id: &str, data_key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
        let _ = (key_id, data_key);
        Err(key_unavailable(
            "this key provider only generates data keys",
        ))
    }

    /// A new random data key for a file, wrapped under the current key. The
    /// default draws it and calls [`KeyProvider::wrap_key`].
    fn generate_data_key(&self) -> Result<GeneratedKey> {
        let key_id = self.current_key_id()?;
        let mut data_key = DataKey::new([0; DATA_KEY_LEN]);
        random(&mut data_key[..])?;
        let wrapped = self.wrap_key(&key_id, &data_key)?;
        Ok(GeneratedKey {
            key_id,
            data_key,
            wrapped,
        })
    }

    /// Decrypts a data key [`KeyProvider::wrap_key`] produced under `key_id`.
    /// An unknown, disabled or revoked key fails with [`key_unavailable`].
    fn unwrap_key(&self, key_id: &str, wrapped: &[u8]) -> Result<DataKey>;
}

/// The error a [`KeyProvider`] returns when it cannot use a key (unknown,
/// disabled, revoked, access denied): an I/O error of kind
/// `PermissionDenied`, so a caller can tell it from corruption.
pub fn key_unavailable(message: impl Into<String>) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        message.into(),
    ))
}

/// Fills `buf` from the operating system's CSPRNG.
pub(crate) fn random(buf: &mut [u8]) -> Result<()> {
    fill_random(buf)
}

const WRAP_NONCE_LEN: usize = 12;
const WRAP_AAD: &[u8] = b"lucene-rust key wrap v1:";

/// Key-encryption keys held in memory, each 256 bits, wrapping with
/// AES-256-GCM (a module-drawn nonce, the key id as associated data).
pub struct StaticKeyProvider {
    current: String,
    keys: BTreeMap<String, Aes256Gcm>,
}

impl std::fmt::Debug for StaticKeyProvider {
    /// Ids only, never key material.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticKeyProvider")
            .field("current", &self.current)
            .field("keys", &self.keys.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl StaticKeyProvider {
    /// A provider whose only (and current) key is `key` under `key_id`.
    pub fn new(key_id: &str, key: &[u8; DATA_KEY_LEN]) -> Self {
        Self {
            current: key_id.to_string(),
            keys: BTreeMap::new(),
        }
        .with_key(key_id, key)
    }

    /// Adds (or replaces) a key, without making it current: how a rotated-out
    /// key stays available for reading old files.
    pub fn with_key(mut self, key_id: &str, key: &[u8; DATA_KEY_LEN]) -> Self {
        // Infallible for a 32-byte key in both backends; a backend failure
        // here is the FIPS module refusing to run, which `key()` reports.
        let Ok(aead) = Aes256Gcm::new(key) else {
            return self;
        };
        self.keys.insert(key_id.to_string(), aead);
        self
    }

    /// Makes `key_id` the key new files use.
    pub fn with_current(mut self, key_id: &str) -> Self {
        self.current = key_id.to_string();
        self
    }

    fn key(&self, key_id: &str) -> Result<&Aes256Gcm> {
        self.keys
            .get(key_id)
            .ok_or_else(|| key_unavailable(format!("no key [{key_id}] in this key provider")))
    }

    fn aad(key_id: &str) -> Vec<u8> {
        let mut aad = WRAP_AAD.to_vec();
        aad.extend_from_slice(key_id.as_bytes());
        aad
    }
}

impl KeyProvider for StaticKeyProvider {
    fn current_key_id(&self) -> Result<String> {
        self.key(&self.current)?;
        Ok(self.current.clone())
    }

    fn wrap_key(&self, key_id: &str, data_key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
        let aead = self.key(key_id)?;
        let mut sealed = data_key.to_vec();
        let (nonce, tag) = aead.seal(&Self::aad(key_id), &mut sealed)?;
        let mut out = nonce.to_vec();
        out.extend_from_slice(&sealed);
        out.extend_from_slice(&tag);
        Ok(out)
    }

    fn unwrap_key(&self, key_id: &str, wrapped: &[u8]) -> Result<DataKey> {
        let aead = self.key(key_id)?;
        let bad = || {
            Error::Corrupted(format!(
                "wrapped data key under [{key_id}] failed authentication"
            ))
        };
        if wrapped.len() != WRAP_NONCE_LEN + DATA_KEY_LEN + TAG_LEN {
            return Err(bad());
        }
        let (nonce, sealed) = wrapped.split_at(WRAP_NONCE_LEN);
        let nonce: &[u8; WRAP_NONCE_LEN] = nonce.try_into().map_err(|_| bad())?;
        let mut buf = zeroize::Zeroizing::new(sealed.to_vec());
        aead.open(nonce, &Self::aad(key_id), &mut buf)
            .map_err(|_| bad())?;
        let mut key = DataKey::new([0; DATA_KEY_LEN]);
        key.copy_from_slice(&buf[..DATA_KEY_LEN]);
        Ok(key)
    }
}

/// `(key id, wrapped key) -> (cached at, data key)`.
type KeyCache = HashMap<(String, Vec<u8>), (Instant, DataKey)>;

/// A [`KeyProvider`] that remembers unwrapped data keys for `ttl`, so a
/// remote KMS is asked once per file per `ttl` rather than on every open. A
/// revoked key stops working for files already cached only once their entry
/// expires: `ttl` is the revocation latency.
pub struct CachingKeyProvider<P: KeyProvider> {
    inner: P,
    ttl: Duration,
    max_entries: usize,
    cache: Mutex<KeyCache>,
}

impl<P: KeyProvider> CachingKeyProvider<P> {
    /// Caches at most `max_entries` keys, each for `ttl`.
    pub fn new(inner: P, ttl: Duration, max_entries: usize) -> Self {
        Self {
            inner,
            ttl,
            max_entries,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The provider underneath.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Forgets every cached key (after a revocation, say).
    pub fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, KeyCache> {
        self.cache.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl<P: KeyProvider> KeyProvider for CachingKeyProvider<P> {
    fn current_key_id(&self) -> Result<String> {
        self.inner.current_key_id()
    }

    fn wrap_key(&self, key_id: &str, data_key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
        let wrapped = self.inner.wrap_key(key_id, data_key)?;
        self.insert(key_id, &wrapped, DataKey::new(*data_key));
        Ok(wrapped)
    }

    fn generate_data_key(&self) -> Result<GeneratedKey> {
        let generated = self.inner.generate_data_key()?;
        self.insert(
            &generated.key_id,
            &generated.wrapped,
            generated.data_key.clone(),
        );
        Ok(generated)
    }

    fn unwrap_key(&self, key_id: &str, wrapped: &[u8]) -> Result<DataKey> {
        let entry = (key_id.to_string(), wrapped.to_vec());
        if let Some((at, key)) = self.lock().get(&entry) {
            if at.elapsed() < self.ttl {
                return Ok(key.clone());
            }
        }
        let key = self.inner.unwrap_key(key_id, wrapped)?;
        self.insert(key_id, wrapped, key.clone());
        Ok(key)
    }
}

impl<P: KeyProvider> CachingKeyProvider<P> {
    fn insert(&self, key_id: &str, wrapped: &[u8], key: DataKey) {
        if self.max_entries == 0 {
            return;
        }
        let mut cache = self.lock();
        let ttl = self.ttl;
        cache.retain(|_, (at, _)| at.elapsed() < ttl);
        while cache.len() >= self.max_entries {
            let oldest = cache
                .iter()
                .min_by_key(|(_, (at, _))| *at)
                .map(|(k, _)| k.clone());
            match oldest {
                Some(k) => cache.remove(&k),
                None => break,
            };
        }
        cache.insert(
            (key_id.to_string(), wrapped.to_vec()),
            (Instant::now(), key),
        );
    }
}

fn providers() -> &'static RwLock<BTreeMap<String, Arc<dyn KeyProvider>>> {
    static PROVIDERS: OnceLock<RwLock<BTreeMap<String, Arc<dyn KeyProvider>>>> = OnceLock::new();
    PROVIDERS.get_or_init(Default::default)
}

/// Registers (or replaces) a key provider under `name`, for the `encrypted`
/// store wrapper's `encryption.key_provider` setting to name.
pub fn register_key_provider(name: &str, provider: Arc<dyn KeyProvider>) {
    providers()
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .insert(name.to_string(), provider);
}

/// The key provider registered under `name`.
pub fn key_provider(name: &str) -> Result<Arc<dyn KeyProvider>> {
    let providers = providers().read().unwrap_or_else(|p| p.into_inner());
    providers.get(name).cloned().ok_or_else(|| {
        Error::IllegalArgument(format!(
            "unknown key provider [{name}]; registered: {:?}",
            providers.keys().collect::<Vec<_>>()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn static_keys_wrap_unwrap_and_rotate() {
        let p = StaticKeyProvider::new("k1", &[1; 32]).with_key("k2", &[2; 32]);
        assert!(
            !format!("{p:?}").contains("[1, 1"),
            "no key material in Debug"
        );
        assert_eq!(p.current_key_id().unwrap(), "k1");
        let dek = [9u8; 32];
        let w1 = p.wrap_key("k1", &dek).unwrap();
        assert_ne!(&w1[12..44], &dek[..]);
        assert_eq!(*p.unwrap_key("k1", &w1).unwrap(), dek);
        // The key id is bound: the same bytes under k2 fail.
        assert!(matches!(p.unwrap_key("k2", &w1), Err(Error::Corrupted(_))));
        assert!(matches!(
            p.unwrap_key("k1", &w1[1..]),
            Err(Error::Corrupted(_))
        ));
        let p = p.with_current("k2");
        assert_eq!(p.current_key_id().unwrap(), "k2");
        assert_eq!(*p.unwrap_key("k1", &w1).unwrap(), dek);
        let err = p.unwrap_key("gone", &w1).unwrap_err();
        assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied));
        assert!(p.with_current("gone").current_key_id().is_err());
    }

    struct Counting {
        inner: StaticKeyProvider,
        unwraps: AtomicUsize,
    }

    impl KeyProvider for Counting {
        fn current_key_id(&self) -> Result<String> {
            self.inner.current_key_id()
        }
        fn wrap_key(&self, key_id: &str, data_key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
            self.inner.wrap_key(key_id, data_key)
        }
        fn unwrap_key(&self, key_id: &str, wrapped: &[u8]) -> Result<DataKey> {
            self.unwraps.fetch_add(1, Ordering::Relaxed);
            self.inner.unwrap_key(key_id, wrapped)
        }
    }

    fn counting() -> Counting {
        Counting {
            inner: StaticKeyProvider::new("k", &[5; 32]),
            unwraps: AtomicUsize::new(0),
        }
    }

    #[test]
    fn the_cache_skips_the_provider_until_the_ttl_passes() {
        let c = CachingKeyProvider::new(counting(), Duration::from_secs(3600), 2);
        assert_eq!(c.current_key_id().unwrap(), "k");
        let w = c.wrap_key("k", &[7; 32]).unwrap();
        assert_eq!(*c.unwrap_key("k", &w).unwrap(), [7; 32]);
        assert_eq!(
            c.inner().unwraps.load(Ordering::Relaxed),
            0,
            "cached by wrap"
        );
        c.clear();
        c.unwrap_key("k", &w).unwrap();
        c.unwrap_key("k", &w).unwrap();
        assert_eq!(c.inner().unwraps.load(Ordering::Relaxed), 1);
        // Capacity two: a third key evicts the oldest.
        let w2 = c.wrap_key("k", &[8; 32]).unwrap();
        let w3 = c.wrap_key("k", &[9; 32]).unwrap();
        assert_eq!(c.lock().len(), 2);
        c.unwrap_key("k", &w2).unwrap();
        c.unwrap_key("k", &w3).unwrap();
        assert_eq!(c.inner().unwraps.load(Ordering::Relaxed), 1);

        let expired = CachingKeyProvider::new(counting(), Duration::ZERO, 4);
        let w = expired.wrap_key("k", &[7; 32]).unwrap();
        expired.unwrap_key("k", &w).unwrap();
        expired.unwrap_key("k", &w).unwrap();
        assert_eq!(expired.inner().unwraps.load(Ordering::Relaxed), 2);

        let off = CachingKeyProvider::new(counting(), Duration::from_secs(60), 0);
        let w = off.wrap_key("k", &[7; 32]).unwrap();
        off.unwrap_key("k", &w).unwrap();
        assert!(off.lock().is_empty());
    }

    /// A KMS that draws data keys itself (`GenerateDataKey`): the default
    /// wrap path is refused, and the cache still learns generated keys.
    struct GenerateOnly(StaticKeyProvider);

    impl KeyProvider for GenerateOnly {
        fn generate_data_key(&self) -> Result<GeneratedKey> {
            self.0.generate_data_key()
        }
        fn unwrap_key(&self, key_id: &str, wrapped: &[u8]) -> Result<DataKey> {
            self.0.unwrap_key(key_id, wrapped)
        }
    }

    #[test]
    fn a_generate_only_provider_needs_no_wrap() {
        let p = GenerateOnly(StaticKeyProvider::new("g", &[6; 32]));
        assert!(p.current_key_id().is_err());
        assert!(p.wrap_key("g", &[0; 32]).is_err());
        let c = CachingKeyProvider::new(p, Duration::from_secs(60), 4);
        let g = c.generate_data_key().unwrap();
        assert_eq!(g.key_id, "g");
        assert_eq!(c.lock().len(), 1);
        assert_eq!(*c.unwrap_key("g", &g.wrapped).unwrap(), *g.data_key);
    }

    #[test]
    fn providers_are_found_by_name() {
        register_key_provider(
            "test-static",
            Arc::new(StaticKeyProvider::new("a", &[0; 32])),
        );
        assert_eq!(
            key_provider("test-static")
                .unwrap()
                .current_key_id()
                .unwrap(),
            "a"
        );
        let err = key_provider("nope").err().unwrap().to_string();
        assert!(err.contains("unknown key provider [nope]"), "{err}");
        assert!(err.contains("test-static"), "{err}");
    }
}
