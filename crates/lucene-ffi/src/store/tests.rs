#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_store::directory::Input;
use lucene_store::DataOutput;
use lucene_store_crypto::StaticKeyProvider;
use lucene_util::test_support::TempDir;
use std::sync::Mutex;

#[test]
fn page_cache_stats_report_the_budget_and_whole_file_bytes() {
    // Starts the pager where it is available.
    let paged = PagedFsDirectory::open("").is_paged();
    let mut out = [u64::MAX; PAGE_CACHE_STATS];
    // SAFETY: room for every value.
    assert_eq!(
        unsafe { ffi_store_page_cache_stats(out.as_mut_ptr()) },
        FfiStatus::Ok.code()
    );
    assert!(out.iter().all(|&v| v != u64::MAX));
    if paged {
        assert!(out[0] > 0);
    }
    // SAFETY: a null pointer is refused, not written.
    assert_eq!(
        unsafe { ffi_store_page_cache_stats(std::ptr::null_mut()) },
        FfiStatus::NullPointer.code()
    );
}

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
    assert_eq!(for_extension("fdt"), ReadAdvice::Auto);
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
            lucene_engine::status::note_shard(&shard);
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
