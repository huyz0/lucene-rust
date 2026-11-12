//! Which directory the engine opens at a shard path: plain, or encrypted
//! with keys the JVM supplies.
//!
//! The plugin's `rust_encrypted` store type binds its shard path here
//! ([`ffi_store_bind_encrypted`]) before the engine's writer or a reader
//! opens it, and unbinds it ([`ffi_store_unbind`]) when the store closes.
//! [`shard_directory`] is what [`crate::engine_writer`] and
//! [`crate::jvm_reader`] open: a bound path gets an
//! [`EncryptedDirectory`], which reads and writes the LRSE format
//! (`lucene-store-crypto/src/format.rs`) that the plugin's Java
//! `RustEncryptedDirectory` reads and writes too. Its files are read
//! through the demand pager ([`PagedFsDirectory`]) where `userfaultfd` is
//! available: ciphertext is read with `O_DIRECT`, each chunk is decrypted
//! and authenticated once as its page is read in, and the plaintext lives
//! only in the page cache's budget (`ffi_store_set_page_cache_budget`).
//! Elsewhere each file is decrypted whole when it opens.
//!
//! Keys stay in the JVM's key provider (OpenSearch's `MasterKeyProvider`,
//! over a KMS). Rust reaches it through one upcall,
//! [`ffi_store_set_key_upcall`], with the provider's handle and an
//! operation: [`KEY_OP_GENERATE`] a data key (wrapped and raw), or
//! [`KEY_OP_UNWRAP`] one read from a file's header. Unwrapped keys are cached
//! ([`CachingKeyProvider`]), so a reader reopen does not call the KMS per
//! file.

use std::collections::HashMap;
use std::os::raw::c_char;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use lucene_store::directory::{Directory, MmapDirectory};
use lucene_store::{PagedFsConfig, PagedFsDirectory};
use lucene_store_crypto::{
    key_unavailable, CachingKeyProvider, DataKey, EncryptedDirectory, GeneratedKey, KeyProvider,
    PlaintextCache,
};

use crate::error::{guard, set_last_error, FfiStatus};
use crate::raw::str_from_raw;

/// The key upcall's operation: a new data key. No input; the output is
/// `key_id_len u16 BE | key_id (UTF-8) | data key [32] | wrapped key`.
pub const KEY_OP_GENERATE: u32 = 1;
/// The key upcall's operation: unwrap a data key. The input is
/// `key_id_len u16 BE | key_id (UTF-8) | wrapped key`; the output is the
/// 32-byte data key.
pub const KEY_OP_UNWRAP: u32 = 2;
/// The upcall succeeded; `*out_len` bytes of `out` are the result.
pub const KEY_UPCALL_OK: i32 = 0;
/// `out` was too small; `*out_len` is the size needed.
pub const KEY_UPCALL_BUFFER_TOO_SMALL: i32 = 1;
/// The key provider failed; `*out_len` bytes of `out` are a UTF-8 message.
pub const KEY_UPCALL_FAILED: i32 = 2;

/// The JVM's key operation: `(provider, op, input, input_len, out, out_cap,
/// out_len) -> status`.
pub type KeyUpcall = unsafe extern "C" fn(
    provider: u64,
    op: u32,
    input: *const u8,
    input_len: usize,
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
) -> i32;

static KEY_UPCALL: OnceLock<KeyUpcall> = OnceLock::new();

/// How long an unwrapped data key stays cached, and how many are kept.
const KEY_CACHE_TTL: Duration = Duration::from_secs(300);
const KEY_CACHE_ENTRIES: usize = 4096;
/// The first output buffer an upcall gets; a larger wrapped key asks again.
const FIRST_OUT_CAP: usize = 1024;
/// The most an upcall may ask for, so a broken provider cannot ask for
/// gigabytes.
const MAX_OUT_CAP: usize = 1 << 20;

#[derive(Clone)]
struct Binding {
    keys: Arc<dyn KeyProvider>,
    chunk_size: u32,
    /// Shared by the writer's and every reader's directory at the path, so
    /// a file both open is decrypted once.
    cache: Arc<PlaintextCache>,
}

fn bindings() -> &'static RwLock<HashMap<String, Binding>> {
    static BINDINGS: OnceLock<RwLock<HashMap<String, Binding>>> = OnceLock::new();
    BINDINGS.get_or_init(Default::default)
}

/// Shard paths read through the demand pager ([`PagedFsDirectory`]), with
/// their page size.
fn paged_paths() -> &'static RwLock<HashMap<String, PagedFsConfig>> {
    static PAGED: OnceLock<RwLock<HashMap<String, PagedFsConfig>>> = OnceLock::new();
    PAGED.get_or_init(Default::default)
}

/// The directory the engine opens at `path`: an [`EncryptedDirectory`] when
/// the path is bound to the encrypted store, a [`PagedFsDirectory`] when it
/// is bound to the page cache, else the mapped directory. An encrypted
/// binding wins over a page-cache one, and reads through the pager (with
/// the page-cache binding's settings, if any) wherever the pager runs.
pub(crate) fn shard_directory(path: &str) -> lucene_store::Result<Box<dyn Directory>> {
    let binding = bindings()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .get(path)
        .cloned();
    let paged = paged_paths()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .get(path)
        .cloned();
    Ok(match binding {
        None => match paged {
            Some(config) => Box::new(PagedFsDirectory::with_config(path, config)),
            None => Box::new(MmapDirectory::open(path)),
        },
        Some(b) => {
            let encrypted = |inner: Box<dyn Directory>| {
                Ok::<_, lucene_store::Error>(Box::new(
                    EncryptedDirectory::new(inner, b.keys.clone())
                        .with_chunk_size(b.chunk_size)?
                        .with_plaintext_commit_metadata()
                        .with_plaintext_cache(b.cache.clone()),
                ))
            };
            let config = paged
                .map(|mut c| {
                    // Every encrypted file goes through the pager, never a
                    // mapping of its ciphertext.
                    c.paged_extensions = None;
                    c.mapped_extensions.clear();
                    c
                })
                .unwrap_or_default();
            let pager = PagedFsDirectory::with_config(path, config);
            if pager.is_paged() {
                encrypted(Box::new(pager))?
            } else {
                encrypted(Box::new(MmapDirectory::open(path)))?
            }
        }
    })
}

/// A [`KeyProvider`] whose keys live in the JVM, reached through the
/// registered [`KeyUpcall`].
struct UpcallKeyProvider {
    provider: u64,
}

impl UpcallKeyProvider {
    fn call(&self, op: u32, input: &[u8]) -> lucene_store::Result<Vec<u8>> {
        let upcall = KEY_UPCALL
            .get()
            .ok_or_else(|| key_unavailable("no key upcall registered"))?;
        let mut cap = FIRST_OUT_CAP;
        loop {
            let mut out = crate::raw::try_with_capacity::<u8>(cap)
                .map_err(|_| key_unavailable("cannot allocate the key upcall's buffer"))?;
            out.resize(cap, 0);
            let mut len = 0usize;
            // SAFETY: the registered upcall's contract: it reads `input` for
            // `input.len()` bytes, writes at most `cap` bytes to `out` and one
            // `usize` to `len`, all of which are live for the call.
            let rc = unsafe {
                upcall(
                    self.provider,
                    op,
                    input.as_ptr(),
                    input.len(),
                    out.as_mut_ptr(),
                    cap,
                    &mut len,
                )
            };
            match rc {
                KEY_UPCALL_OK if len <= cap => {
                    out.truncate(len);
                    return Ok(out);
                }
                KEY_UPCALL_BUFFER_TOO_SMALL if len > cap && len <= MAX_OUT_CAP => cap = len,
                KEY_UPCALL_FAILED => {
                    let message = String::from_utf8_lossy(&out[..len.min(cap)]).into_owned();
                    return Err(key_unavailable(format!("key provider: {message}")));
                }
                _ => {
                    return Err(key_unavailable(format!(
                        "key provider upcall returned {rc} with length {len}"
                    )))
                }
            }
        }
    }
}

fn put_key_id(out: &mut Vec<u8>, key_id: &str) -> lucene_store::Result<()> {
    let len = u16::try_from(key_id.len())
        .map_err(|_| key_unavailable("key id longer than 65535 bytes"))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(key_id.as_bytes());
    Ok(())
}

impl KeyProvider for UpcallKeyProvider {
    fn generate_data_key(&self) -> lucene_store::Result<GeneratedKey> {
        let out = zeroize::Zeroizing::new(self.call(KEY_OP_GENERATE, &[])?);
        let bad = || key_unavailable("key provider returned a malformed generated key");
        let id_len = usize::from(u16::from_be_bytes(
            out.get(..2)
                .ok_or_else(bad)?
                .try_into()
                .map_err(|_| bad())?,
        ));
        let id_end = 2usize.checked_add(id_len).ok_or_else(bad)?;
        let key_end = id_end.checked_add(32).ok_or_else(bad)?;
        let key_id = std::str::from_utf8(out.get(2..id_end).ok_or_else(bad)?)
            .map_err(|_| bad())?
            .to_string();
        let mut data_key = DataKey::new([0; 32]);
        data_key.copy_from_slice(out.get(id_end..key_end).ok_or_else(bad)?);
        let wrapped = out.get(key_end..).ok_or_else(bad)?.to_vec();
        if wrapped.is_empty() {
            return Err(bad());
        }
        Ok(GeneratedKey {
            key_id,
            data_key,
            wrapped,
        })
    }

    fn unwrap_key(&self, key_id: &str, wrapped: &[u8]) -> lucene_store::Result<DataKey> {
        let mut input = crate::raw::try_with_capacity::<u8>(
            2usize
                .saturating_add(key_id.len())
                .saturating_add(wrapped.len()),
        )
        .map_err(|_| key_unavailable("cannot allocate the unwrap request"))?;
        put_key_id(&mut input, key_id)?;
        input.extend_from_slice(wrapped);
        let out = zeroize::Zeroizing::new(self.call(KEY_OP_UNWRAP, &input)?);
        let key: [u8; 32] = out.as_slice().try_into().map_err(|_| {
            key_unavailable("key provider returned a data key that is not 32 bytes")
        })?;
        Ok(DataKey::new(key))
    }
}

/// Registers the JVM's key upcall. Only the first registration counts (the
/// upcall stub lives as long as the JVM); a second call is a no-op.
///
/// # Safety
/// `upcall` must be a function with [`KeyUpcall`]'s contract that stays
/// callable for the life of the process, from any thread.
#[no_mangle]
pub unsafe extern "C" fn ffi_store_set_key_upcall(upcall: Option<KeyUpcall>) -> i32 {
    guard(|| {
        let upcall = upcall.ok_or(FfiStatus::NullPointer)?;
        let _ = KEY_UPCALL.set(upcall);
        Ok(())
    })
}

/// Binds the shard path `path` to the encrypted store: from now on the
/// engine's writers and readers at that path encrypt and decrypt with data
/// keys from the JVM key provider `provider` (passed back to the upcall),
/// sealing `chunk_size` bytes per chunk (0: the default). Rebinding a path
/// replaces its binding for directories opened afterwards.
///
/// # Safety
/// `path` valid for `path_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_store_bind_encrypted(
    path: *const c_char,
    path_len: usize,
    provider: u64,
    chunk_size: u32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
        if path.is_empty() {
            set_last_error("ffi_store_bind_encrypted: empty path");
            return Err(FfiStatus::InvalidArgument);
        }
        if KEY_UPCALL.get().is_none() {
            set_last_error("ffi_store_bind_encrypted: no key upcall registered");
            return Err(FfiStatus::InvalidArgument);
        }
        let chunk_size = if chunk_size == 0 {
            lucene_store_crypto::format::DEFAULT_CHUNK_SIZE
        } else {
            chunk_size
        };
        if chunk_size > lucene_store_crypto::format::MAX_CHUNK_SIZE {
            set_last_error(format!(
                "ffi_store_bind_encrypted: chunk size {chunk_size} above the maximum"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        let keys: Arc<dyn KeyProvider> = Arc::new(CachingKeyProvider::new(
            UpcallKeyProvider { provider },
            KEY_CACHE_TTL,
            KEY_CACHE_ENTRIES,
        ));
        bindings()
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(
                path.to_string(),
                Binding {
                    keys,
                    chunk_size,
                    cache: Arc::new(PlaintextCache::new()),
                },
            );
        Ok(())
    })
}

/// Removes `path`'s binding, if any. Directories already open keep theirs.
///
/// # Safety
/// `path` valid for `path_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_store_unbind(path: *const c_char, path_len: usize) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
        bindings()
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .remove(path);
        Ok(())
    })
}

/// Removes `path`'s page-cache binding ([`ffi_store_bind_page_cache`]), if
/// any. Directories already open keep reading through the page cache.
///
/// # Safety
/// `path` valid for `path_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_store_unbind_page_cache(path: *const c_char, path_len: usize) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
        paged_paths()
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .remove(path);
        Ok(())
    })
}

/// Binds the shard path `path` to the page cache: from now on the engine's
/// writers and readers at that path read its files through the process's
/// demand pager in pages of `page_size` bytes (0: the default), within the
/// budget [`ffi_store_set_page_cache_budget`] sets, instead of mapping them.
/// With `vectors_only != 0`, only the vector formats' files go through the
/// page cache ([`PagedFsConfig::vectors_only`]) and the rest stay mapped.
/// Where the pager is unavailable ([`ffi_store_page_cache_available`] is 0)
/// nothing is bound and the path stays mapped.
///
/// # Safety
/// `path` valid for `path_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_store_bind_page_cache(
    path: *const c_char,
    path_len: usize,
    page_size: u32,
    vectors_only: i32,
) -> i32 {
    let mut config = if vectors_only != 0 {
        PagedFsConfig::vectors_only()
    } else {
        PagedFsConfig::default()
    };
    if page_size != 0 {
        config.page_size = page_size as usize;
    }
    guard(move || {
        // SAFETY: caller contract.
        let path = unsafe { str_from_raw(path, path_len)? };
        if path.is_empty() {
            set_last_error("ffi_store_bind_page_cache: empty path");
            return Err(FfiStatus::InvalidArgument);
        }
        if !PagedFsDirectory::open("").is_paged() {
            // No pager here: the path stays mapped.
            return Ok(());
        }
        if !PagedFsDirectory::with_config("", config.clone()).is_paged() {
            set_last_error(format!(
                "ffi_store_bind_page_cache: page size {page_size} is not a whole number of OS pages"
            ));
            return Err(FfiStatus::InvalidArgument);
        }
        paged_paths()
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(path.to_string(), config);
        Ok(())
    })
}

/// 1 when shards can be read through the page cache (`userfaultfd` is
/// available to this process), else 0.
#[no_mangle]
pub extern "C" fn ffi_store_page_cache_available() -> i32 {
    i32::from(PagedFsDirectory::open("").is_paged())
}

/// Sets the bytes the page cache keeps resident across every shard bound
/// to it ([`ffi_store_bind_page_cache`]); pages past it are released, least
/// recently brought in first; 0 keeps the current budget. Does nothing where
/// the pager is unavailable.
#[no_mangle]
pub extern "C" fn ffi_store_set_page_cache_budget(budget_bytes: u64) -> i32 {
    #[cfg(target_os = "linux")]
    if let Some(pager) = lucene_store::paged_input::Pager::global() {
        if budget_bytes > 0 {
            pager.set_budget(budget_bytes);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = budget_bytes;
    FfiStatus::Ok.code()
}

/// Sets how mapped shard files are read, by extension
/// ([`lucene_store::read_advice::set`]): `spec` is comma-separated
/// `ext:advice` entries (`normal`, `random`, `sequential`, `willneed`), `*`
/// for every other extension, empty for the defaults. `warm_random` 1 reads
/// a file advised random in at open when it is small next to available
/// memory. Applies to readers opened afterwards.
///
/// # Safety
/// `spec` valid for `spec_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ffi_store_set_read_advice(
    spec: *const c_char,
    spec_len: usize,
    warm_random: i32,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let spec = unsafe { str_from_raw(spec, spec_len)? };
        lucene_store::read_advice::set(spec).map_err(|e| {
            set_last_error(format!("ffi_store_set_read_advice: {e}"));
            FfiStatus::InvalidArgument
        })?;
        lucene_store::read_advice::set_warm_random(warm_random != 0);
        Ok(())
    })
}

/// Bytes the page cache holds resident now, across every bound shard.
#[no_mangle]
pub extern "C" fn ffi_store_page_cache_resident_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    if let Some(pager) = lucene_store::paged_input::Pager::global() {
        return pager.resident_bytes();
    }
    0
}

/// 1 when encrypted shards are decrypted page by page into the page cache
/// (`userfaultfd` is available), else 0: each file is then decrypted whole
/// when it opens.
#[no_mangle]
pub extern "C" fn ffi_store_lazy_decrypt_active() -> i32 {
    ffi_store_page_cache_available()
}

/// 1 when this library's encryption runs in AWS-LC's FIPS 140-3 module in
/// FIPS mode (a `fips` build), else 0.
#[no_mangle]
pub extern "C" fn ffi_store_is_fips() -> i32 {
    i32::from(lucene_store_crypto::is_fips())
}

#[cfg(test)]
mod tests {
    // Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::directory::Input;
    use lucene_store::DataOutput;
    use lucene_store_crypto::StaticKeyProvider;
    use lucene_util::test_support::TempDir;
    use std::sync::Mutex;

    /// The test "JVM": providers by handle, each a static KEK. Handle 99
    /// fails, handle 98 returns a malformed key.
    fn kek(provider: u64) -> StaticKeyProvider {
        StaticKeyProvider::new(&format!("kek-{provider}"), &[provider as u8; 32])
    }

    static CALLS: Mutex<u32> = Mutex::new(0);

    unsafe extern "C" fn test_upcall(
        provider: u64,
        op: u32,
        input: *const u8,
        input_len: usize,
        out: *mut u8,
        out_cap: usize,
        out_len: *mut usize,
    ) -> i32 {
        *CALLS.lock().unwrap() += 1;
        // SAFETY: the upcall contract.
        let (input, out) = unsafe {
            (
                std::slice::from_raw_parts(input, input_len),
                std::slice::from_raw_parts_mut(out, out_cap),
            )
        };
        let result: Result<Vec<u8>, String> = match (provider, op) {
            (99, _) => Err("revoked".into()),
            (98, KEY_OP_GENERATE) => Ok(vec![0, 1]),
            (98, _) => Ok(vec![1, 2, 3]),
            (97, KEY_OP_GENERATE) => {
                // A wrapped key larger than the first output buffer.
                let mut v = Vec::new();
                put_key_id(&mut v, "big").unwrap();
                v.extend_from_slice(&[4; 32]);
                v.extend_from_slice(&[5; 2000]);
                Ok(v)
            }
            (p, KEY_OP_GENERATE) => {
                let g = kek(p).generate_data_key().unwrap();
                let mut v = Vec::new();
                put_key_id(&mut v, &g.key_id).unwrap();
                v.extend_from_slice(&g.data_key[..]);
                v.extend_from_slice(&g.wrapped);
                Ok(v)
            }
            (p, KEY_OP_UNWRAP) => {
                let n = usize::from(u16::from_be_bytes([input[0], input[1]]));
                let id = std::str::from_utf8(&input[2..2 + n]).unwrap();
                kek(p)
                    .unwrap_key(id, &input[2 + n..])
                    .map(|k| k.to_vec())
                    .map_err(|e| e.to_string())
            }
            _ => Err("bad op".into()),
        };
        let (rc, bytes) = match result {
            Ok(v) => (KEY_UPCALL_OK, v),
            Err(m) => (KEY_UPCALL_FAILED, m.into_bytes()),
        };
        // SAFETY: the upcall contract.
        unsafe { *out_len = bytes.len() };
        if bytes.len() > out.len() {
            return KEY_UPCALL_BUFFER_TOO_SMALL;
        }
        out[..bytes.len()].copy_from_slice(&bytes);
        rc
    }

    fn bind(path: &str, provider: u64, chunk: u32) -> i32 {
        // SAFETY: a live string.
        unsafe { ffi_store_bind_encrypted(path.as_ptr().cast(), path.len(), provider, chunk) }
    }

    fn unbind(path: &str) -> i32 {
        // SAFETY: a live string.
        unsafe { ffi_store_unbind(path.as_ptr().cast(), path.len()) }
    }

    fn write(dir: &dyn Directory, name: &str, bytes: &[u8]) {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }

    #[test]
    fn a_bound_path_encrypts_with_jvm_keys_and_an_unbound_one_does_not() {
        // SAFETY: a function with the upcall's contract.
        assert_eq!(unsafe { ffi_store_set_key_upcall(Some(test_upcall)) }, 0);
        // SAFETY: a null upcall is refused.
        assert_eq!(
            unsafe { ffi_store_set_key_upcall(None) },
            FfiStatus::NullPointer.code()
        );
        let tmp = TempDir::new("ffi-store-bind");
        let path = tmp.path().to_str().unwrap().to_string();

        let plain = shard_directory(&path).unwrap();
        write(&*plain, "plain.bin", b"hello");
        assert_eq!(
            std::fs::read(tmp.path().join("plain.bin")).unwrap(),
            b"hello"
        );

        assert_eq!(bind(&path, 7, 16), 0);
        let enc = shard_directory(&path).unwrap();
        let body: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        write(&*enc, "enc.bin", &body);
        let raw = std::fs::read(tmp.path().join("enc.bin")).unwrap();
        assert!(raw.starts_with(b"LRSE"));
        assert_eq!(&enc.open("enc.bin").unwrap()[..], &body[..]);
        assert_eq!(enc.file_length("enc.bin").unwrap(), 1000);

        // A fresh directory over the same binding reads it (the key comes
        // from the header, unwrapped through the upcall or the cache).
        let again = shard_directory(&path).unwrap();
        assert_eq!(&again.open("enc.bin").unwrap()[..], &body[..]);

        // Another provider's key cannot unwrap it.
        assert_eq!(bind(&path, 8, 0), 0);
        let other = shard_directory(&path).unwrap();
        let err = other.open("enc.bin").unwrap_err().to_string();
        assert!(err.contains("key provider"), "{err}");

        assert_eq!(unbind(&path), 0);
        let unbound = shard_directory(&path).unwrap();
        assert!(unbound.open("enc.bin").unwrap().starts_with(b"LRSE"));
    }

    /// A bound shard is read through the page cache where the pager runs,
    /// and the writer's and a reader's directories at one path share the
    /// plaintext. A chunk that changes under an open reader fails the call
    /// that reads it with an I/O error instead of ending the process.
    /// The read advice table is set through the FFI, a bad spec is refused
    /// with a message, and an empty one restores the defaults.
    #[test]
    fn read_advice_is_set_refused_and_restored() {
        use lucene_store::read_advice::{for_extension, warm_random, ReadAdvice};
        let spec = "fdt:random,*:normal";
        // SAFETY: `spec` is valid for its length.
        let rc = unsafe { ffi_store_set_read_advice(spec.as_ptr().cast(), spec.len(), 0) };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(for_extension("fdt"), ReadAdvice::Random);
        assert_eq!(for_extension("vec"), ReadAdvice::Normal);
        assert!(!warm_random());
        let bad = "fdt:sideways";
        // SAFETY: as above.
        let rc = unsafe { ffi_store_set_read_advice(bad.as_ptr().cast(), bad.len(), 1) };
        assert_eq!(rc, FfiStatus::InvalidArgument.code());
        assert!(crate::error::last_error().contains("sideways"));
        assert_eq!(for_extension("fdt"), ReadAdvice::Random);
        // SAFETY: an empty spec.
        let rc = unsafe { ffi_store_set_read_advice(std::ptr::null(), 0, 1) };
        assert_eq!(rc, FfiStatus::Ok.code());
        assert_eq!(for_extension("vec"), ReadAdvice::Random);
        assert_eq!(for_extension("fdt"), ReadAdvice::Normal);
        assert!(warm_random());
    }

    #[test]
    fn encrypted_shards_read_through_the_page_cache_and_fail_bad_chunks_cleanly() {
        // SAFETY: a function with the upcall's contract.
        unsafe { ffi_store_set_key_upcall(Some(test_upcall)) };
        assert_eq!(
            ffi_store_lazy_decrypt_active(),
            ffi_store_page_cache_available()
        );
        let tmp = TempDir::new("ffi-store-paged");
        let path = tmp.path().to_str().unwrap().to_string();
        assert_eq!(bind(&path, 7, 0), 0);
        let writer = shard_directory(&path).unwrap();
        let body: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        write(&*writer, "_0.fdt", &body);
        let a = writer.open("_0.fdt").unwrap();
        let reader = shard_directory(&path).unwrap();
        let b = reader.open("_0.fdt").unwrap();
        assert_eq!(&a[..], &body[..]);
        assert_eq!(a.as_ptr(), b.as_ptr(), "one plaintext per path");
        drop((a, b));
        if ffi_store_page_cache_available() == 1 {
            let input = reader.open("_0.fdt").unwrap();
            // Flip a byte of the third chunk's ciphertext on disk.
            let file = tmp.path().join("_0.fdt");
            let mut raw = std::fs::read(&file).unwrap();
            let at = raw.len() / 2;
            raw[at] ^= 1;
            std::fs::write(&file, &raw).unwrap();
            let shard: Arc<str> = Arc::from(path.as_str());
            let held = std::panic::AssertUnwindSafe(&input);
            let rc = guard(move || {
                crate::error::note_shard(&shard);
                // The page holding the flipped chunk reads as zeros.
                let held = held;
                let input: &Input = held.0;
                let middle = input.len() / 2;
                assert_eq!(input[middle..middle + 64], [0u8; 64]);
                Ok(())
            });
            assert_eq!(rc, FfiStatus::Io.code());
            let message = crate::error::last_error();
            assert!(message.contains("failed authentication"), "{message}");
            assert!(input.will_need(0, input.len()).is_err());
            // A call on another shard is not failed by it.
            assert_eq!(guard(|| Ok(())), 0);
        }
        assert_eq!(unbind(&path), 0);
    }

    #[test]
    fn provider_failures_surface_as_key_errors() {
        // SAFETY: a function with the upcall's contract.
        unsafe { ffi_store_set_key_upcall(Some(test_upcall)) };
        let failing = UpcallKeyProvider { provider: 99 };
        let err = failing.generate_data_key().err().unwrap().to_string();
        assert!(err.contains("revoked"), "{err}");
        assert!(failing.unwrap_key("k", b"w").is_err());
        let malformed = UpcallKeyProvider { provider: 98 };
        assert!(malformed.generate_data_key().is_err());
        assert!(malformed.unwrap_key("k", b"w").is_err());
        let ok = UpcallKeyProvider { provider: 5 };
        let g = ok.generate_data_key().unwrap();
        assert_eq!(
            &ok.unwrap_key(&g.key_id, &g.wrapped).unwrap()[..],
            &g.data_key[..]
        );
        assert!(put_key_id(&mut Vec::new(), &"x".repeat(70_000)).is_err());
        let big = UpcallKeyProvider { provider: 97 }
            .generate_data_key()
            .unwrap();
        assert_eq!((big.key_id.as_str(), big.wrapped.len()), ("big", 2000));
    }

    #[test]
    fn bind_refuses_bad_arguments() {
        // SAFETY: a function with the upcall's contract.
        unsafe { ffi_store_set_key_upcall(Some(test_upcall)) };
        assert_eq!(bind("", 1, 0), FfiStatus::InvalidArgument.code());
        assert_eq!(bind("/x", 1, u32::MAX), FfiStatus::InvalidArgument.code());
        // SAFETY: null with a length is refused before any read.
        let rc = unsafe { ffi_store_bind_encrypted(std::ptr::null(), 3, 1, 0) };
        assert_eq!(rc, FfiStatus::NullPointer.code());
        assert_eq!(ffi_store_is_fips(), i32::from(cfg!(feature = "fips")));
    }
}
