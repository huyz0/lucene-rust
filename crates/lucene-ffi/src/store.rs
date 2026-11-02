//! Which directory the engine opens at a shard path: plain, or encrypted
//! with keys the JVM supplies.
//!
//! The plugin's `rust_encrypted` store type binds its shard path here
//! ([`ffi_store_bind_encrypted`]) before the engine's writer or a reader
//! opens it, and unbinds it ([`ffi_store_unbind`]) when the store closes.
//! [`shard_directory`] is what [`crate::engine_writer`] and
//! [`crate::jvm_reader`] open: a bound path gets an
//! [`EncryptedDirectory`] over the mapped directory, which reads and writes
//! the LRSE format (`lucene-store-crypto/src/format.rs`) that the plugin's
//! Java `RustEncryptedDirectory` reads and writes too.
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
use lucene_store_crypto::{
    key_unavailable, CachingKeyProvider, DataKey, EncryptedDirectory, GeneratedKey, KeyProvider,
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
}

fn bindings() -> &'static RwLock<HashMap<String, Binding>> {
    static BINDINGS: OnceLock<RwLock<HashMap<String, Binding>>> = OnceLock::new();
    BINDINGS.get_or_init(Default::default)
}

/// The directory the engine opens at `path`: an [`EncryptedDirectory`] over
/// the mapped directory when the path is bound, else the mapped directory.
pub(crate) fn shard_directory(path: &str) -> lucene_store::Result<Box<dyn Directory>> {
    let binding = bindings()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .get(path)
        .cloned();
    let mapped = MmapDirectory::open(path);
    Ok(match binding {
        None => Box::new(mapped),
        Some(b) => Box::new(
            EncryptedDirectory::new(mapped, b.keys)
                .with_chunk_size(b.chunk_size)?
                .with_plaintext_commit_metadata(),
        ),
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
            let mut out = vec![0u8; cap];
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
        let mut input = Vec::with_capacity(
            2usize
                .saturating_add(key_id.len())
                .saturating_add(wrapped.len()),
        );
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
            .insert(path.to_string(), Binding { keys, chunk_size });
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
